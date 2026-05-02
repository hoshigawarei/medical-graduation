"""
Phase 4: 工作流调度中心 (Workflow Controller)

ClinicalWorkflow 串联多智能体，模拟「查资料 → 读片 → 分析 → 风控」的诊疗辅助流程，
并在控制台打印清晰思维链轨迹。

另提供 `single_model_answer`：同一多模态模型单次生成，用作与流水线对照的基线。
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from medical_mvp import config
from medical_mvp.agents import AnalysisAgent, KnowledgeAgent, RiskAgent, VisionAgent
from medical_mvp.gemini_throttle import before_gemini_request


def single_model_answer(user_question: str, image_path: str) -> str:
    """
    单模型基线：不经检索与多智能体阶段，使用 `GEMINI_MODEL_ID` 调用一次多模态生成。

    与 `clinical_workflow` 使用同一主模型，便于在固定数据集上用同一指标（如与参考答案的
    词重叠）对比「一次作答」与「RAG + 多阶段 Agent」的差异。

    若本地图像路径不存在，则退化为仅文本提示（与 VisionAgent 行为一致）。
    """
    key = os.environ.get(config.GOOGLE_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"未找到 API Key，请在环境变量 {config.GOOGLE_API_KEY_ENV} 中配置 GOOGLE_API_KEY。"
        )
    client = genai.Client(api_key=key)
    prompt = (
        "You are a medical imaging assistant. Answer the user's question using the image; "
        "be concise and cautious, in **English only**. State limitations if evidence is insufficient; "
        "do not replace a treating physician.\n\n"
        f"[User question]\n{user_question.strip()}"
    )
    path = Path(image_path)
    if path.is_file():
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/jpeg"
        contents: list[Any] = [
            types.Part.from_bytes(data=data, mime_type=mime),
            prompt,
        ]
    else:
        contents = [
            prompt
            + "\n\n(Note: no valid local image path; give only general guidance from the question.)"
        ]
    before_gemini_request()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL_ID,
        contents=contents,
    )
    return (response.text or "").strip()


def clinical_workflow(
    user_question: str,
    image_path: str,
    knowledge_agent: KnowledgeAgent | None = None,
    vision_agent: VisionAgent | None = None,
    analysis_agent: AnalysisAgent | None = None,
    risk_agent: RiskAgent | None = None,
) -> dict[str, Any]:
    """
    模拟临床多智能体协作主循环（MVP）。

    调度顺序：
        1. KnowledgeAgent：检索医学参考
        2. VisionAgent：结合图像 + 问题 + 参考完成影像解释
        3. AnalysisAgent：输出综合分析文本 + 结构化结果
        4. RiskAgent：规则先行 + LLM 复核

    Args:
        user_question: 用户临床相关问题。
        image_path: 本地医学影像路径。
        各 *_agent: 可注入 mock 以便单测；默认构造真实实例。

    Returns:
        包含各阶段产出的字典，便于日志记录或 API 封装。
    """
    ka = knowledge_agent or KnowledgeAgent()
    va = vision_agent or VisionAgent()
    aa = analysis_agent or AnalysisAgent(vision_agent=va)
    ra = risk_agent or RiskAgent()

    print("\n========== ClinicalWorkflow 开始 ==========\n")

    hits = ka.run(user_question)
    vision_report = va.run(user_question, image_path, hits)
    analysis_out = aa.run(user_question, vision_report, hits)
    analysis_report = analysis_out.get("analysis_report", "")
    analysis_structured = analysis_out.get("analysis_structured", {})
    risk = ra.run(analysis_report, analysis_structured=analysis_structured)

    print("\n========== ClinicalWorkflow 结束 ==========\n")

    return {
        "retrieval_hits": hits,
        "vision_report": vision_report,
        "analysis_report": analysis_report,
        "analysis_structured": analysis_structured,
        "risk": risk,
    }
