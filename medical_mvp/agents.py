"""
Phase 3: 智能体角色定义 (Agent Definitions)

BaseAgent 提供统一日志与扩展钩子；四个子类对应
知识检索、影像解释、医学分析（Stub）、风险评估（Stub）。
"""

from __future__ import annotations

import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from medical_mvp import config
from medical_mvp.retrieval import MedicalRetriever, RetrievalHit


class BaseAgent(ABC):
    """所有智能体的基类：约束统一接口名称，便于工作流编排。"""

    name: str = "BaseAgent"

    def trace(self, message: str) -> None:
        """在控制台输出带智能体名称的轨迹信息。"""
        print(f"[{self.name}] {message}")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """子类实现具体行为。"""
        raise NotImplementedError


class KnowledgeAgent(BaseAgent):
    """
    知识检索智能体：根据用户问题调用混合检索器，返回可读医学参考资料。
    """

    name = "KnowledgeAgent"

    def __init__(self, retriever: MedicalRetriever | None = None) -> None:
        self.retriever = retriever or MedicalRetriever()
        self.retriever.build_faiss_from_qa_database()

    def run(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        self.trace("正在检索医学参考资料（当前为 FAISS 向量通道）…")
        hits = self.retriever.retrieve_context(query, top_k=top_k)
        self.trace(f"检索完成，命中 {len(hits)} 条参考片段。")
        return hits


class VisionAgent(BaseAgent):
    """
    影像解释智能体：联合医学图像与文本提示，调用 Gemini 多模态模型完成读片推理。
    """

    name = "VisionAgent"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(config.GOOGLE_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"未找到 API Key，请在环境变量 {config.GOOGLE_API_KEY_ENV} 中配置 GOOGLE_API_KEY。"
            )
        self._client = genai.Client(api_key=key)

    def _guess_mime(self, image_path: str) -> str:
        mime, _ = mimetypes.guess_type(image_path)
        if mime and mime.startswith("image/"):
            return mime
        return "image/jpeg"

    def run(
        self,
        user_question: str,
        image_path: str,
        knowledge_hits: list[RetrievalHit],
    ) -> str:
        self.trace("结合用户问题、检索到的医学资料与影像进行推理…")

        ref_lines = "\n".join(f"- ({h.score:.4f}) {h.text}" for h in knowledge_hits[:5])
        if not ref_lines:
            ref_lines = "（当前无额外检索参考）"

        system_style = (
            "你是一个影像解释智能体，擅长结合医学影像与临床背景进行谨慎、有条理的分析。"
            "请用中文作答：先给出对影像的客观描述，再结合问题给出推断；"
            "若信息不足请明确说明局限性，并避免做出超出影像证据的诊断结论。"
        )

        prompt = (
            f"{system_style}\n\n"
            f"【用户问题】\n{user_question}\n\n"
            f"【检索到的医学参考资料（RAG）】\n{ref_lines}\n\n"
            "请基于上述资料与图像给出影像解释与回答要点。"
        )

        path = Path(image_path)
        if not path.is_file():
            self.trace(f"警告：未找到图像文件 {image_path}，将仅基于文本与检索资料回答。")
            contents: list[Any] = [prompt]
        else:
            data = path.read_bytes()
            mime = self._guess_mime(str(path))
            contents = [
                types.Part.from_bytes(data=data, mime_type=mime),
                prompt,
            ]

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL_ID,
            contents=contents,
        )
        text = (response.text or "").strip()
        self.trace("影像推理完成。")
        return text


class AnalysisAgent(BaseAgent):
    """
    医学分析智能体（MVP Stub）：当前不对 Vision 结果做二次综合，直接透传。

    TODO: 未来接入多源证据融合、鉴别诊断列表、临床路径建议等综合分析能力。
    """

    name = "AnalysisAgent"

    def __init__(self, vision_agent: VisionAgent | None = None) -> None:
        # 预留：若需在独立入口将「仅文本问题」转交影像智能体，可注入 VisionAgent
        self._vision = vision_agent

    def run(self, user_question: str, vision_report: str) -> str:
        self.trace("医学分析（MVP）：透传 VisionAgent 输出，不做额外综合。")
        # Stub：直接返回影像智能体结果；后续可在此串联工具调用与思维链
        _ = user_question  # 占位，未来综合分析会显式使用用户原始问题
        return vision_report


class RiskAgent(BaseAgent):
    """
    风险评估 / 合规智能体（MVP Stub）：默认放行。

    TODO: 未来实现政策与医学安全审查（违规用药建议、歧视性内容、隐私泄露等）。
    """

    name = "RiskAgent"

    def run(self, final_text: str) -> dict[str, Any]:
        self.trace("风险评估（MVP）：跳过细粒度审查。")
        _ = final_text
        return {"is_safe": True, "reason": "MVP阶段跳过审查"}
