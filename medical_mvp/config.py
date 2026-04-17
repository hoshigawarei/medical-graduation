"""
全局配置：路径、模型名称等。
在 Colab 中可将 BASE_DIR 指向 Drive，本地开发则使用项目下的 data 目录。
"""

from __future__ import annotations

import os
from pathlib import Path

# Gemini（google-genai 新 SDK）
GEMINI_MODEL_ID = "gemini-2.5-flash"

# 默认从环境变量读取 API Key（Colab 中可 os.environ["GOOGLE_API_KEY"] = "..."）
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# HuggingFace 数据集
# 默认使用社区重构的流式友好版，image 列可直接读取为真实图像（PIL Image）。
# 若需回退原版，可改回 "xmcmic/PMC-VQA"。
PMC_VQA_DATASET = "hamzamooraj99/PMC-VQA-1"
PMC_VQA_STREAM_SPLIT = "train"  # 与仓库中 train 划分一致；若不存在可改为 "test"

# 向量检索
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_TOP_K = 5

# AnalysisAgent（增强版）
ANALYSIS_TOPK_EVIDENCE = 5
ANALYSIS_DIFF_TOPK = 3

# RiskAgent（规则 + LLM 复核）
RISK_RULES_ENABLED = True
RISK_LLM_ENABLED = True
# 默认沿用主模型；如需独立模型可改为其他可用 Gemini 模型
RISK_LLM_MODEL_ID = GEMINI_MODEL_ID

# 目录（可被 data_preparation 中的 set_base_dir 覆盖）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data"


def get_data_root() -> Path:
    """返回当前生效的数据根目录。"""
    return Path(os.environ.get("MEDICAL_MVP_DATA_ROOT", str(DEFAULT_DATA_ROOT)))


def get_image_dir() -> Path:
    return get_data_root() / "images"


def get_qa_database_path() -> Path:
    return get_data_root() / "qa_database.json"


def get_faiss_index_path() -> Path:
    return get_data_root() / "faiss.index"


def get_faiss_meta_path() -> Path:
    return get_data_root() / "faiss_meta.json"
