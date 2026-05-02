"""
LLM-as-Judge（默认使用与流水线相同的 Gemini）：对 primary_impression（或综合结论）相对 gold 做 1–5 分多维打分。

依赖 GOOGLE_API_KEY。成本高（每样本 1×workflow + 1×裁判调用）；默认小样本。

用法：
  python -m medical_mvp.eval_llm_judge --n 5 --seed 42

输出：data/results/llm_judge_时间戳.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from google import genai

from medical_mvp import config
from medical_mvp.eval_e2e import _pick_samples
from medical_mvp.eval_metrics import pred_text_for_pipeline
from medical_mvp.eval_retrieval import (
    VARIANT_PRESETS,
    _apply_variant,
    _restore_config_state,
    _save_config_state,
)
from medical_mvp.gemini_throttle import before_gemini_request
from medical_mvp.workflow import clinical_workflow


def _parse_judge_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON in judge response")


def judge_scores(
    client: genai.Client,
    *,
    question: str,
    gold: str,
    prediction: str,
    retrieval_excerpt: str,
) -> dict[str, Any]:
    prompt = f"""You evaluate medical imaging QA outputs for a graduation-research prototype.
Score each dimension as an integer from 1 (poor) to 5 (excellent).

Return ONLY valid JSON with keys:
"factual_alignment" (vs reference answer),
"answer_relevance" (addresses the question),
"internal_consistency" (no contradictions),
"safe_wording" (appropriate uncertainty for imaging-only setting),
"notes" (one short English sentence).

Question:
{question}

Reference answer (gold):
{gold}

Model prediction (primary conclusion text):
{prediction}

Retrieved evidence excerpt (may be incomplete):
{retrieval_excerpt[:6000]}
"""
    before_gemini_request()
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL_ID,
        contents=[prompt],
    )
    raw = (resp.text or "").strip()
    return _parse_judge_json(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini LLM-as-Judge on pipeline outputs")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-variant",
        default="Full_hybrid",
        choices=list(VARIANT_PRESETS.keys()),
    )
    args = parser.parse_args()

    if not os.environ.get(config.GOOGLE_API_KEY_ENV):
        print(f"[ERR] 需要 {config.GOOGLE_API_KEY_ENV}", file=sys.stderr)
        sys.exit(1)

    qa_path = config.get_qa_database_path()
    if not qa_path.is_file():
        print(f"[ERR] 未找到 {qa_path}", file=sys.stderr)
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        records = json.load(f)
    samples = _pick_samples(records, args.n, args.seed)

    client = genai.Client(api_key=os.environ[config.GOOGLE_API_KEY_ENV])

    report: dict[str, Any] = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "judge_model": config.GEMINI_MODEL_ID,
        "pipeline_variant": args.pipeline_variant,
        "n_samples": len(samples),
        "runs": [],
    }

    saved = _save_config_state()
    try:
        _apply_variant(args.pipeline_variant)
        for rec in samples:
            q = str(rec.get("question", ""))
            img = str(rec.get("image_path", ""))
            gold = str(rec.get("answer", ""))
            rid = rec.get("id")
            row: dict[str, Any] = {"id": rid, "question": q[:200], "ok": False, "error": None}
            try:
                out = clinical_workflow(user_question=q, image_path=img)
                scoring = pred_text_for_pipeline(out)
                hits = list(out.get("retrieval_hits") or [])
                excerpt = "\n".join(h.text[:1500] for h in hits[:5])
                scores = judge_scores(
                    client,
                    question=q,
                    gold=gold,
                    prediction=scoring,
                    retrieval_excerpt=excerpt,
                )
                row["ok"] = True
                row["judge"] = scores
                row["scoring_len"] = len(scoring)
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc()
            report["runs"].append(row)
    finally:
        _restore_config_state(saved)

    report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    out_dir = config.get_data_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"llm_judge_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[eval_llm_judge] 已写入: {out_path}")


if __name__ == "__main__":
    main()
