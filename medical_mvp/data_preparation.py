"""
Phase 1: 数据准备与预处理

- 在 Google Colab 中可选挂载 Google Drive，并将数据写入 Drive 目录（避免 Colab 临时盘丢失）。
- 使用 datasets 以流式方式读取 xmcmic/PMC-VQA 的前 N 条样本。
- 将图像保存到本地/Drive 下的 images/，并生成 qa_database.json 供检索与工作流使用。
"""

from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from tqdm import tqdm

from medical_mvp import config


def mount_google_drive_if_colab(force: bool = False) -> bool:
    """
    若在 Google Colab 环境中，则挂载 Google Drive。

    Returns:
        是否执行了挂载（True 表示已尝试挂载；本地环境返回 False）。
    """
    in_colab = "google.colab" in sys.modules
    if not in_colab and not force:
        return False
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return True
    except Exception:
        # 非 Colab 或未安装 colab 包时静默跳过
        return False


def set_base_dir_for_colab(drive_subdir: str = "MyDrive/medical_mvp_data") -> Path:
    """
    将数据根目录设为 Colab Drive 下的子目录（需先 mount）。

    Args:
        drive_subdir: 相对于 /content/drive 的子路径。

    Returns:
        设置后的数据根 Path。
    """
    base = Path("/content/drive") / drive_subdir
    base.mkdir(parents=True, exist_ok=True)
    os.environ["MEDICAL_MVP_DATA_ROOT"] = str(base)
    return base


def _pil_save_image(img_obj: Any, dest_path: Path) -> None:
    """将 datasets 的 Image / PIL / bytes 统一保存为文件。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img_obj, "save"):
        img_obj.save(dest_path)
        return
    if isinstance(img_obj, dict):
        raw = img_obj.get("bytes")
        if raw:
            from PIL import Image

            Image.open(BytesIO(raw)).convert("RGB").save(dest_path)
            return
        path = img_obj.get("path")
        if path and os.path.isfile(path):
            from PIL import Image

            Image.open(path).convert("RGB").save(dest_path)
            return
    raise TypeError(f"无法识别的图像类型: {type(img_obj)}")


def _open_streaming_dataset(split: str):
    from datasets import load_dataset

    return load_dataset(config.PMC_VQA_DATASET, split=split, streaming=True)


def stream_pmc_vqa_and_build_database(
    limit: int = 200,
    split: str | None = None,
) -> Path:
    """
    流式拉取 PMC-VQA 前 limit 条，保存图片并写入 qa_database.json。

    Args:
        limit: 最大条数（毕业设计 MVP 固定为 200）。
        split: 数据集划分名；默认使用 config.PMC_VQA_STREAM_SPLIT。

    Returns:
        qa_database.json 的路径。
    """
    preferred = split or config.PMC_VQA_STREAM_SPLIT
    fallback_splits = [preferred, "train", "test", "validation"]
    ds = None
    last_err: Exception | None = None
    for sp in fallback_splits:
        try:
            ds = _open_streaming_dataset(sp)
            break
        except Exception as e:  # noqa: BLE001 — 兼容不同仓库划分命名
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"无法打开数据集 {config.PMC_VQA_DATASET}，已尝试划分: {fallback_splits}") from last_err

    root = config.get_data_root()
    img_dir = config.get_image_dir()
    root.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    iterator = iter(ds)
    for idx in tqdm(range(limit), desc="PMC-VQA 流式下载与落盘"):
        try:
            row = next(iterator)
        except StopIteration:
            break

        figure_path = row.get("Figure_path") or row.get("figure_path") or f"sample_{idx}.jpg"
        stem = Path(str(figure_path)).stem
        safe_name = f"{idx:05d}_{stem}.jpg"
        out_img = img_dir / safe_name
        image_path_str = ""

        # 优先使用 HF Image 列；保存失败时仍保留 QA 文本以便检索测试
        image_obj = row.get("image")
        if image_obj is not None:
            try:
                _pil_save_image(image_obj, out_img)
                image_path_str = str(out_img.resolve())
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 样本 {idx} 图像保存失败，将仅保留文本元数据: {exc}")

        record = {
            "id": idx,
            "figure_path": str(figure_path),
            "image_path": image_path_str,
            "question": row.get("Question") or row.get("question") or "",
            "answer": row.get("Answer") or row.get("answer") or "",
            "choice_a": row.get("Choice A") or row.get("choice_a"),
            "choice_b": row.get("Choice B") or row.get("choice_b"),
            "choice_c": row.get("Choice C") or row.get("choice_c"),
            "choice_d": row.get("Choice D") or row.get("choice_d"),
            "answer_label": row.get("Answer_label") or row.get("answer_label"),
        }
        records.append(record)

    out_json = config.get_qa_database_path()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return out_json


def run_phase1_colab(
    mount_drive: bool = True,
    drive_subdir: str = "MyDrive/medical_mvp_data",
    limit: int = 200,
) -> Path:
    """
    Colab 一键入口：挂载 Drive → 设定目录 → 拉取数据。

    本地运行时可将 mount_drive=False，数据将落在项目 ./data。
    """
    if mount_drive:
        mount_google_drive_if_colab()
        set_base_dir_for_colab(drive_subdir=drive_subdir)
    return stream_pmc_vqa_and_build_database(limit=limit)


if __name__ == "__main__":
    # 本地直接运行：python -m medical_mvp.data_preparation
    p = stream_pmc_vqa_and_build_database(limit=200)
    print("已生成:", p)
