"""
Phase 4: 工作流调度中心 (Workflow Controller)

ClinicalWorkflow 串联多智能体，模拟「查资料 → 读片 → 分析 → 风控」的诊疗辅助流程，
并在控制台打印清晰思维链轨迹。
"""

from __future__ import annotations

from typing import Any

from medical_mvp.agents import AnalysisAgent, KnowledgeAgent, RiskAgent, VisionAgent


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
