"""
单模型 vs 多智能体流水线：同一批样本、同一主模型。

- single：workflow.single_model_answer（每样本 1 次多模态 generate）
- pipeline：clinical_workflow（检索 + Vision + Analysis + Risk）

评价：除 legacy 全文 Jaccard 外，主要报告与 gold 的 token F1/EM；流水线另报告 RAG 对齐与结构化/风控元数据（见 eval_metrics）。

用法：
  python -m medical_mvp.eval_single_vs_pipeline --n 3 --seed 42 --pipeline-variant Full_hybrid

输出：data/results/single_vs_pipeline_时间戳.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from medical_mvp import config
from medical_mvp.eval_e2e import _pick_samples
from medical_mvp.eval_metrics import (
    exact_match,
    jaccard_overlap,
    pred_text_for_pipeline,
    pred_text_for_single_model,
    rag_token_recall,
    risk_audit_fields,
    structured_audit_fields,
    token_f1,
)
from medical_mvp.eval_retrieval import (
    VARIANT_PRESETS,
    _apply_variant,
    _restore_config_state,
    _save_config_state,
)
from medical_mvp.workflow import clinical_workflow, single_model_answer


def _run_mode(
    samples: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    ok_n = 0
    jac_sum = 0.0
    f1_sum = 0.0
    rag_sum = 0.0
    rag_n = 0
    em_n = 0
    runs: list[dict[str, Any]] = []

    for rec in samples:
        q = str(rec.get("question", ""))
        img = str(rec.get("image_path", ""))
        gold = str(rec.get("answer", ""))
        rid = rec.get("id")
        row: dict[str, Any] = {
            "id": rid,
            "question": q[:200],
            "ok": False,
            "error": None,
            "jaccard_vs_answer": None,
            "f1_vs_answer": None,
            "em_vs_answer": None,
            "scoring_text": None,
            "response_len": None,
        }
        try:
            if mode == "single_model":
                full_text = single_model_answer(q, img)
                scoring = pred_text_for_single_model(full_text)
                row["scoring_text"] = "full_single_model_response"
                jac_text = full_text
                row["response_len"] = len(full_text)
            else:
                out = clinical_workflow(user_question=q, image_path=img)
                analysis = str(out.get("analysis_report", "") or "")
                scoring = pred_text_for_pipeline(out)
                row["scoring_text"] = "primary_impression_or_conclusion"
                jac_text = analysis
                row["response_len"] = len(analysis)
                risk = out.get("risk") or {}
                row.update(risk_audit_fields(risk if isinstance(risk, dict) else None))
                row.update(structured_audit_fields(out))
                hits = list(out.get("retrieval_hits") or [])
                rr = rag_token_recall(scoring, hits)
                row["rag_token_recall"] = rr
                if rr is not None:
                    rag_sum += rr
                    rag_n += 1

            row["ok"] = True
            row["jaccard_vs_answer"] = jaccard_overlap(jac_text, gold)
            jac_sum += row["jaccard_vs_answer"]
            f1 = token_f1(scoring, gold)
            row["f1_vs_answer"] = f1
            f1_sum += f1
            em = exact_match(scoring, gold)
            row["em_vs_answer"] = em
            if em:
                em_n += 1
            ok_n += 1
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
            if mode == "pipeline":
                row["rag_token_recall"] = None

        runs.append(row)

    n = len(samples)
    mean_j = jac_sum / ok_n if ok_n else None
    mean_f1 = f1_sum / ok_n if ok_n else None
    mean_rag = rag_sum / rag_n if rag_n else None
    success = ok_n / n if n else 0.0
    return {
        "success_rate": success,
        "mean_jaccard_vs_answer": mean_j,
        "mean_f1_vs_answer": mean_f1,
        "mean_em_rate": em_n / ok_n if ok_n else None,
        "mean_rag_token_recall": mean_rag if mode == "pipeline" else None,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单模型一次生成 vs 多智能体流水线（固定检索 variant）",
    )
    parser.add_argument("--n", type=int, default=3, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--pipeline-variant",
        type=str,
        default="Full_hybrid",
        help=f"流水线侧检索消融：{list(VARIANT_PRESETS.keys())}",
    )
    args = parser.parse_args()

    if args.pipeline_variant not in VARIANT_PRESETS:
        print(
            f"[ERR] 未知 --pipeline-variant: {args.pipeline_variant}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get(config.GOOGLE_API_KEY_ENV):
        print(
            f"[ERR] 未设置 {config.GOOGLE_API_KEY_ENV}。",
            file=sys.stderr,
        )
        sys.exit(1)

    qa_path = config.get_qa_database_path()
    if not qa_path.is_file():
        print(f"[ERR] 未找到 {qa_path}", file=sys.stderr)
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        records = json.load(f)

    samples = _pick_samples(records, args.n, args.seed)

    report: dict[str, Any] = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "comparison": "single_model_vs_pipeline",
        "model": config.GEMINI_MODEL_ID,
        "metric_note": (
            "优先解读 mean_f1_vs_answer / mean_em_rate：与参考答案的词级 F1 及完全匹配率；"
            "流水线侧 scoring 文本为 primary_impression（或综合结论回退），单模型侧为全文。"
            "mean_rag_token_recall：检索 Top-K 词元出现在该 scoring 文本中的比例（仅流水线）。"
            "mean_jaccard_vs_answer：legacy，流水线为全文分析报告 vs gold，体裁与 short gold 不一致时偏低常见。"
        ),
        "qa_path": str(qa_path),
        "n_requested": args.n,
        "n_samples": len(samples),
        "seed": args.seed,
        "pipeline_variant": args.pipeline_variant,
        "pipeline_retrieval": {
            "HYBRID_ENABLE_BM25": None,
            "HYBRID_ENABLE_GRAPH": None,
        },
        "single_model": {},
        "pipeline": {},
        "finished_at_utc": None,
    }

    report["single_model"] = _run_mode(samples, "single_model")

    saved = _save_config_state()
    try:
        _apply_variant(args.pipeline_variant)
        report["pipeline_retrieval"]["HYBRID_ENABLE_BM25"] = config.HYBRID_ENABLE_BM25
        report["pipeline_retrieval"]["HYBRID_ENABLE_GRAPH"] = config.HYBRID_ENABLE_GRAPH

        report["pipeline"] = _run_mode(samples, "pipeline")
    finally:
        _restore_config_state(saved)

    report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    out_dir = config.get_data_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"single_vs_pipeline_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[eval_single_vs_pipeline] 已写入: {out_path}")


if __name__ == "__main__":
    main()
