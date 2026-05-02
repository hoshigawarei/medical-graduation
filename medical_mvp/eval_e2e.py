"""
端到端小样本评测：同一批样本上对比不同检索消融配置下的 workflow 输出。

依赖环境变量 GOOGLE_API_KEY（google-genai）。成本随样本数线性增长，默认小样本。

除 legacy `mean_jaccard_vs_answer`（全文报告 vs gold）外，报告 `mean_f1_vs_answer`（综合结论短文本 vs gold）、
`mean_rag_token_recall`、`mean_em_rate` 等架构感知指标（见 eval_metrics）。

用法：
  python -m medical_mvp.eval_e2e --n 5 --seed 42

输出：data/results/e2e_eval_时间戳.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

from medical_mvp import config
from medical_mvp.eval_metrics import (
    attach_extended_metrics,
    exact_match,
    jaccard_overlap,
    pred_text_for_pipeline,
    rag_token_recall,
    risk_audit_fields,
    structured_audit_fields,
    token_f1,
)
from medical_mvp.eval_retrieval import VARIANT_PRESETS, _apply_variant, _restore_config_state, _save_config_state
from medical_mvp.workflow import clinical_workflow


def _pick_samples(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    with_img = [r for r in records if r.get("image_path") and Path(r["image_path"]).is_file()]
    rng = random.Random(seed)
    pool = with_img if len(with_img) >= n else records
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端检索消融 × workflow 小样本评测")
    parser.add_argument("--n", type=int, default=5, help="每 variant 样本数（同一批样本复用）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument(
        "--variants",
        type=str,
        default="FAISS_only,FAISS_BM25,FAISS_Graph,Full_hybrid",
        help="逗号分隔或 all",
    )
    args = parser.parse_args()

    if not os.environ.get(config.GOOGLE_API_KEY_ENV):
        print(
            f"[ERR] 未设置 {config.GOOGLE_API_KEY_ENV}，端到端评测需要 Gemini API。",
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
    variant_list = (
        list(VARIANT_PRESETS.keys())
        if args.variants.strip().lower() == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )

    saved = _save_config_state()
    report: dict[str, Any] = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qa_path": str(qa_path),
        "n_requested": args.n,
        "n_samples": len(samples),
        "seed": args.seed,
        "model": config.GEMINI_MODEL_ID,
        "metric_note": (
            "主指标建议：mean_bertscore_f1（语义）、mean_entity_token_recall_vs_gold（gold 词出现比例）、"
            "mean_ddx_covers_gold_rate（鉴别诊断是否覆盖 gold，启发式）。"
            "legacy：mean_f1_vs_answer / mean_jaccard_vs_answer（字面）。"
            "mean_rag_token_recall：检索词元出现在 scoring 文本中的比例。"
        ),
        "variants": {},
    }

    try:
        for vname in variant_list:
            if vname not in VARIANT_PRESETS:
                print(f"[WARN] 跳过未知 variant: {vname}", file=sys.stderr)
                continue
            _apply_variant(vname)
            runs: list[dict[str, Any]] = []
            ok_n = 0
            jac_sum = 0.0
            f1_sum = 0.0
            rag_sum = 0.0
            rag_cnt = 0
            em_hit = 0

            for rec in samples:
                q = str(rec.get("question", ""))
                img = str(rec.get("image_path", ""))
                gold = str(rec.get("answer", ""))
                rid = rec.get("id")
                row: dict[str, Any] = {
                    "id": rid,
                    "question": q[:200],
                    "scoring_text": "primary_impression_or_conclusion",
                    "ok": False,
                    "error": None,
                    "jaccard_vs_answer": None,
                    "f1_vs_answer": None,
                    "em_vs_answer": None,
                    "rag_token_recall": None,
                    "analysis_report_len": None,
                }
                try:
                    out = clinical_workflow(user_question=q, image_path=img)
                    analysis = str(out.get("analysis_report", "") or "")
                    scoring = pred_text_for_pipeline(out)
                    row["ok"] = True
                    row["analysis_report_len"] = len(analysis)
                    row["jaccard_vs_answer"] = jaccard_overlap(analysis, gold)
                    jac_sum += row["jaccard_vs_answer"]
                    row["f1_vs_answer"] = token_f1(scoring, gold)
                    f1_sum += row["f1_vs_answer"]
                    row["em_vs_answer"] = exact_match(scoring, gold)
                    if row["em_vs_answer"]:
                        em_hit += 1
                    hits = list(out.get("retrieval_hits") or [])
                    rr = rag_token_recall(scoring, hits)
                    row["rag_token_recall"] = rr
                    if rr is not None:
                        rag_sum += rr
                        rag_cnt += 1
                    risk = out.get("risk")
                    row.update(
                        risk_audit_fields(risk if isinstance(risk, dict) else None)
                    )
                    row.update(structured_audit_fields(out))
                    stru = out.get("analysis_structured")
                    attach_extended_metrics(
                        row,
                        scoring,
                        gold,
                        stru if isinstance(stru, dict) else None,
                    )
                    ok_n += 1
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"
                    row["traceback"] = traceback.format_exc()

                runs.append(row)

            n = len(samples)
            bs_vals = [r.get("bertscore_f1") for r in runs if r.get("bertscore_f1") is not None]
            ent_vals = [r.get("entity_token_recall_vs_gold") for r in runs if r.get("ok")]
            ddx_eligible = [r for r in runs if r.get("ok") and r.get("ddx_covers_gold") is not None]
            ddx_hits = sum(1 for r in ddx_eligible if r.get("ddx_covers_gold"))

            report["variants"][vname] = {
                "HYBRID_ENABLE_BM25": config.HYBRID_ENABLE_BM25,
                "HYBRID_ENABLE_GRAPH": config.HYBRID_ENABLE_GRAPH,
                "success_rate": ok_n / n if n else 0.0,
                "mean_jaccard_vs_answer": jac_sum / ok_n if ok_n else None,
                "mean_f1_vs_answer": f1_sum / ok_n if ok_n else None,
                "mean_em_rate": em_hit / ok_n if ok_n else None,
                "mean_rag_token_recall": rag_sum / rag_cnt if rag_cnt else None,
                "mean_bertscore_f1": sum(bs_vals) / len(bs_vals) if bs_vals else None,
                "mean_entity_token_recall_vs_gold": (
                    sum(float(x) for x in ent_vals if x is not None) / len(ent_vals) if ent_vals else None
                ),
                "mean_ddx_covers_gold_rate": (ddx_hits / len(ddx_eligible) if ddx_eligible else None),
                "runs": runs,
            }
    finally:
        _restore_config_state(saved)

    report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    out_dir = config.get_data_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"e2e_eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[eval_e2e] 已写入: {out_path}")


if __name__ == "__main__":
    main()
