"""
单模型 vs 多智能体流水线：同一批样本、同一主模型、同一粗粒度 Jaccard（相对标准答案）。

- single：workflow.single_model_answer（每样本 1 次多模态 generate）
- pipeline：clinical_workflow（检索 + Vision + Analysis + Risk，每样本多次调用）

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
from medical_mvp.eval_e2e import _pick_samples, jaccard_overlap
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
) -> tuple[float, float | None, list[dict[str, Any]]]:
    ok_n = 0
    jac_sum = 0.0
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
            "response_len": None,
        }
        try:
            if mode == "single_model":
                text = single_model_answer(q, img)
            else:
                out = clinical_workflow(user_question=q, image_path=img)
                text = str(out.get("analysis_report", "") or "")
                row["risk_level"] = (out.get("risk") or {}).get("risk_level")

            row["ok"] = True
            row["response_len"] = len(text)
            jac = jaccard_overlap(text, gold)
            row["jaccard_vs_answer"] = jac
            jac_sum += jac
            ok_n += 1
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()

        runs.append(row)

    n = len(samples)
    mean_j = jac_sum / ok_n if ok_n else None
    success = ok_n / n if n else 0.0
    return success, mean_j, runs


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
            "mean_jaccard_vs_answer：分析报告（流水线）或单次作答（单模型）与 gold answer 的词集 Jaccard；"
            "粗粒度参考，非临床准确性。"
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

    # 先跑单模型（不依赖检索开关）
    s_rate, s_mean, s_runs = _run_mode(samples, "single_model")
    report["single_model"] = {
        "success_rate": s_rate,
        "mean_jaccard_vs_answer": s_mean,
        "runs": s_runs,
    }

    saved = _save_config_state()
    try:
        _apply_variant(args.pipeline_variant)
        report["pipeline_retrieval"]["HYBRID_ENABLE_BM25"] = config.HYBRID_ENABLE_BM25
        report["pipeline_retrieval"]["HYBRID_ENABLE_GRAPH"] = config.HYBRID_ENABLE_GRAPH

        p_rate, p_mean, p_runs = _run_mode(samples, "pipeline")
        report["pipeline"] = {
            "success_rate": p_rate,
            "mean_jaccard_vs_answer": p_mean,
            "runs": p_runs,
        }
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
