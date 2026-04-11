"""
Phase 5: 运行测试

从本地 qa_database.json 随机抽取若干条，驱动 ClinicalWorkflow，
验证骨架端到端可运行（需有效 GOOGLE_API_KEY 与已落盘图像）。
"""

from __future__ import annotations

import argparse
import json
import os
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
    # 与 Phase 1 使用同一规则：优先显式 qa_path，否则用 MEDICAL_MVP_DATA_ROOT（须与挂载格一致）
    qa_path = Path(qa_path) if qa_path is not None else config.get_qa_database_path()

    if not qa_path.is_file():
        dr = (os.environ.get("MEDICAL_MVP_DATA_ROOT") or "").strip()
        if dr:
            alt = Path(dr) / "qa_database.json"
            if alt.is_file():
                qa_path = alt

    if not qa_path.is_file():
        print(
            f"未找到 qa_database.json（当前查找: {qa_path}）。"
            "若你已在 Google Drive 生成过数据，请先运行「挂载 Drive」那一格，确保已设置环境变量 "
            "MEDICAL_MVP_DATA_ROOT 指向与 Phase 1 相同的目录；否则将在仓库下 ./data 自动执行 Phase 1…",
            file=sys.stderr,
        )
        stream_pmc_vqa_and_build_database(limit=200)
        qa_path = config.get_qa_database_path()

    if not qa_path.is_file():
        print(f"[ERR] Phase 1 执行后仍找不到: {qa_path}", file=sys.stderr)
        return

    records = _load_records(qa_path)
    nonempty = sum(1 for r in records if (r.get("image_path") or "").strip())
    with_img = [r for r in records if r.get("image_path") and Path(r["image_path"]).is_file()]
    rng = random.Random(seed)

    print(
        f"[run_mvp] qa_path={qa_path} 条数={len(records)} image_path非空={nonempty} 磁盘可读图={len(with_img)}",
        file=sys.stderr,
    )

    if len(with_img) >= n:
        chosen = rng.sample(with_img, n)
    elif len(with_img) > 0:
        print(
            f"可用带图样本仅 {len(with_img)} 条（少于 n={n}），将全部用于测试。",
            file=sys.stderr,
        )
        chosen = with_img
    else:
        # 无落盘图像时仍跑通工作流：VisionAgent 会退化为仅文本+RAG（见 agents.VisionAgent）
        print(
            "未找到有效 image_path 文件，将随机抽取文本 QA 样本（无真实影像输入）。",
            file=sys.stderr,
        )
        if not records:
            print("qa_database.json 为空，退出。", file=sys.stderr)
            return
        chosen = rng.sample(records, min(n, len(records)))

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
