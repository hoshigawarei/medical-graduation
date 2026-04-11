"""
Phase 5: 运行测试

从本地 qa_database.json 随机抽取若干条，驱动 ClinicalWorkflow，
验证骨架端到端可运行（需有效 GOOGLE_API_KEY 与已落盘图像）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from medical_mvp import config
from medical_mvp.data_preparation import stream_pmc_vqa_and_build_database
from medical_mvp.workflow import clinical_workflow


def _load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_random_samples(
    n: int = 3,
    qa_path: Path | None = None,
    seed: int | None = 42,
) -> None:
    qa_path = qa_path or config.get_qa_database_path()
    if not qa_path.is_file():
        print("未找到 qa_database.json，先执行 Phase 1 数据准备…", file=sys.stderr)
        stream_pmc_vqa_and_build_database(limit=200)

    records = _load_records(qa_path)
    with_img = [r for r in records if r.get("image_path") and Path(r["image_path"]).is_file()]
    if len(with_img) < n:
        print(
            f"可用带图样本仅 {len(with_img)} 条，将使用全部可用样本运行测试。",
            file=sys.stderr,
        )
        chosen = with_img
    else:
        rng = random.Random(seed)
        chosen = rng.sample(with_img, n)

    for i, rec in enumerate(chosen, start=1):
        print(f"\n>>>>>>>>>> 样本 {i}/{len(chosen)} | id={rec.get('id')} <<<<<<<<<<\n")
        q = str(rec.get("question", ""))
        img = str(rec.get("image_path", ""))
        out = clinical_workflow(user_question=q, image_path=img)
        print("\n--- 最终输出摘要 ---")
        print("风险评估:", out["risk"])
        print("分析结论（透传影像报告，前 500 字）:\n")
        text = out["analysis_report"]
        print(text[:500] + ("…" if len(text) > 500 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 端到端随机抽样测试")
    parser.add_argument("--n", type=int, default=3, help="随机样本条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    run_random_samples(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
