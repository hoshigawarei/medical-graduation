"""
Phase 3: 智能体角色定义 (Agent Definitions)

BaseAgent 提供统一日志与扩展钩子；四个子类对应
知识检索、影像解释、医学分析（Stub）、风险评估（Stub）。
"""

from __future__ import annotations

import json
import mimetypes
import os
import random
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from medical_mvp import config
from medical_mvp.gemini_throttle import before_gemini_request
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
        self.trace("正在检索医学参考资料（混合检索：retrieve_context）…")
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
        self.trace("Running vision agent: image + question + RAG context…")

        ref_lines = "\n".join(f"- ({h.score:.4f}) {h.text}" for h in knowledge_hits[:5])
        if not ref_lines:
            ref_lines = "(No retrieved references)"

        system_style = (
            "You are a radiology assistant. Combine the image, the user's question, and any retrieved text "
            "carefully. Respond in **English** only. Start with objective imaging findings, then address "
            "the question; state limitations clearly and avoid conclusions beyond the imaging evidence."
        )

        prompt = (
            f"{system_style}\n\n"
            f"[User question]\n{user_question}\n\n"
            f"[Retrieved references (RAG)]\n{ref_lines}\n\n"
            "Give an imaging interpretation and key answer points in English."
        )

        path = Path(image_path)
        if not path.is_file():
            self.trace(f"Warning: image not found at {image_path}; answering from text and RAG only.")
            contents: list[Any] = [prompt]
        else:
            data = path.read_bytes()
            mime = self._guess_mime(str(path))
            contents = [
                types.Part.from_bytes(data=data, mime_type=mime),
                prompt,
            ]

        # 服务高峰时 Gemini 可能返回 503，使用指数退避 + 抖动自动重试。
        max_retries = 5
        base_wait_sec = 2.0
        response = None
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                before_gemini_request()
                response = self._client.models.generate_content(
                    model=config.GEMINI_MODEL_ID,
                    contents=contents,
                )
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "leaked" in msg or "permission_denied" in msg or "403" in str(e):
                    raise RuntimeError(
                        "Gemini 返回 403：该 API 密钥可能被判定为泄露或已失效。"
                        "请到 https://aistudio.google.com/apikey 删除旧密钥并新建，"
                        "在 Colab「密钥」中更新 GOOGLE_API_KEY；切勿在聊天、截图或 Git 中粘贴密钥。"
                    ) from e

                is_503 = ("503" in str(e)) or ("unavailable" in msg)
                if (not is_503) or attempt == max_retries:
                    break

                wait = base_wait_sec * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
                self.trace(
                    f"Gemini 服务繁忙（503），第 {attempt}/{max_retries} 次重试前等待 {wait:.1f} 秒…"
                )
                time.sleep(wait)

        if response is None:
            assert last_err is not None
            raise RuntimeError(
                "Gemini 连续返回服务不可用（503）或其他错误。"
                "建议稍后重试，或降低并发后再次运行。原始错误: "
                + repr(last_err)
            ) from last_err
        text = (response.text or "").strip()
        self.trace("Vision inference complete.")
        return text


class AnalysisAgent(BaseAgent):
    """
    医学分析智能体（增强版）：融合 Vision 结果与 RAG 证据，输出结构化分析结果。
    """

    name = "AnalysisAgent"

    def __init__(self, vision_agent: VisionAgent | None = None, api_key: str | None = None) -> None:
        # 预留：若需在独立入口将「仅文本问题」转交影像智能体，可注入 VisionAgent
        self._vision = vision_agent
        key = api_key or os.environ.get(config.GOOGLE_API_KEY_ENV)
        self._client = genai.Client(api_key=key) if key else None

    def _extract_json(self, text: str) -> dict[str, Any]:
        """从模型响应中提取 JSON 对象。"""
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 对象")
        return json.loads(match.group(0))

    def run(
        self,
        user_question: str,
        vision_report: str,
        knowledge_hits: list[RetrievalHit],
    ) -> dict[str, Any]:
        self.trace("Synthesizing structured analysis from Vision + RAG…")

        top_hits = knowledge_hits[: config.ANALYSIS_TOPK_EVIDENCE]
        refs = []
        for i, h in enumerate(top_hits, start=1):
            refs.append(f"{i}. score={h.score:.4f} | {h.text}")
        ref_text = "\n".join(refs) if refs else "(No retrieval evidence)"

        # 若缺少 API Key，回退为可解释的规则化输出，避免中断主流程。
        if self._client is None:
            self.trace("No API key; AnalysisAgent falls back to rule-based summary.")
            structured = {
                "primary_impression": "Preliminary impression from Vision only (no LLM synthesis).",
                "differential_diagnoses": [],
                "supporting_evidence": refs[:2],
                "conflicts": [],
                "confidence": 0.5,
                "recommended_next_steps": [
                    "Add clinical history and labs if available",
                    "Repeat imaging or higher resolution if indicated",
                ],
            }
            return {"analysis_report": vision_report, "analysis_structured": structured}

        prompt = f"""
You are a clinical analysis agent. Using ONLY the inputs below, output **only valid JSON** (no markdown, no extra text).
All string values must be in **English** so outputs align with English benchmark answers.

[User question]
{user_question}

[Vision report]
{vision_report}

[Retrieved evidence (ranked)]
{ref_text}

JSON schema (fill all fields):
{{
  "primary_impression": "one concise English conclusion",
  "differential_diagnoses": [
    {{"name": "diagnosis A", "reason": "brief rationale", "likelihood": 0.0}},
    {{"name": "diagnosis B", "reason": "brief rationale", "likelihood": 0.0}}
  ],
  "supporting_evidence": ["short bullet in English", "..."],
  "conflicts": ["conflict or contradiction if any"],
  "confidence": 0.0,
  "recommended_next_steps": ["short English action", "..."]
}}

Rules:
1) likelihood and confidence are floats in [0, 1].
2) At most {config.ANALYSIS_DIFF_TOPK} differential_diagnoses entries.
3) List real conflicts in conflicts; use [] if none.
4) Do not invent labs or findings not implied by the inputs.
"""
        before_gemini_request()
        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL_ID,
            contents=[prompt],
        )
        raw = (response.text or "").strip()
        try:
            structured = self._extract_json(raw)
        except Exception:
            # 模型偶发未严格 JSON，回退最小可用结构。
            structured = {
                "primary_impression": "Model did not return strict JSON; see raw_text.",
                "differential_diagnoses": [],
                "supporting_evidence": refs[:2],
                "conflicts": [],
                "confidence": 0.5,
                "recommended_next_steps": ["Manual review of output format for this sample"],
                "raw_text": raw,
            }

        ddx = structured.get("differential_diagnoses", [])
        if isinstance(ddx, list):
            structured["differential_diagnoses"] = ddx[: config.ANALYSIS_DIFF_TOPK]

        # 生成人类可读报告（与结构化结果保持解耦，便于后续模板迭代）。
        primary = str(structured.get("primary_impression", "")).strip()
        confidence = structured.get("confidence", "N/A")
        conflicts = structured.get("conflicts", [])
        next_steps = structured.get("recommended_next_steps", [])
        evidence = structured.get("supporting_evidence", [])
        ddx_lines = []
        for item in structured.get("differential_diagnoses", []):
            name = item.get("name", "unnamed")
            likelihood = item.get("likelihood", "N/A")
            reason = item.get("reason", "")
            ddx_lines.append(f"- {name} (likelihood={likelihood}): {reason}")

        analysis_report = (
            "Primary impression\n"
            f"{primary}\n\n"
            f"Overall confidence\n{confidence}\n\n"
            "Differential diagnoses (top-K)\n"
            + ("\n".join(ddx_lines) if ddx_lines else "- none\n")
            + "\n\nSupporting evidence\n"
            + ("\n".join(f"- {x}" for x in evidence) if evidence else "- none\n")
            + "\n\nConflicts\n"
            + ("\n".join(f"- {x}" for x in conflicts) if conflicts else "- none noted\n")
            + "\n\nRecommended next steps\n"
            + ("\n".join(f"- {x}" for x in next_steps) if next_steps else "- clinical correlation\n")
        )
        return {"analysis_report": analysis_report, "analysis_structured": structured}


class RiskAgent(BaseAgent):
    """
    风险评估 / 合规智能体（增强版）：规则先行，再由 LLM 复核。
    """

    name = "RiskAgent"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(config.GOOGLE_API_KEY_ENV)
        self._client = genai.Client(api_key=key) if key else None

    def _rule_scan(self, text: str) -> list[str]:
        hits: list[str] = []
        patterns = [
            (r"\b\d+\s?(mg|g|ml|mL)\b", "Dosage detail present — verify medication safety (human review)."),
            (r"(100%|绝对|必须|唯一正确|always\s+certain|definitely\s+always)", "Overconfident / absolute language may mislead."),
            (r"(自行停药|无需就医|忽略症状|stop\s+taking|ignore\s+symptoms|no\s+need\s+to\s+see)", "Potentially unsafe care advice."),
        ]
        lowered = text.lower()
        for pattern, reason in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hits.append(reason)
        if "确诊" in text and "建议结合临床" not in text and "clinical correlation" not in lowered:
            hits.append('Uses "definitive diagnosis" tone without appropriate caution.')
        if "patient name" in lowered or "身份证" in text:
            hits.append("Possible privacy identifier in text.")
        return hits

    def run(self, final_text: str, analysis_structured: dict[str, Any] | None = None) -> dict[str, Any]:
        self.trace("Risk assessment: rule scan + optional LLM review.")
        rule_hits = self._rule_scan(final_text) if config.RISK_RULES_ENABLED else []

        # 规则命中高危时直接拦截（优先级最高）
        hard_block = any(
            any(k in x for k in ("危险", "剂量", "Dosage", "dosage", "unsafe", "Potentially unsafe"))
            for x in rule_hits
        )
        if hard_block:
            return {
                "is_safe": False,
                "risk_level": "high",
                "reason": "Rule scan flagged high-risk content.",
                "actions": ["Block output", "Escalate for human review", "Remove or rewrite risky spans"],
                "rule_hits": rule_hits,
                "llm_assessment": None,
            }

        llm_assessment: dict[str, Any] | None = None
        if config.RISK_LLM_ENABLED and self._client is not None:
            prompt = f"""
You are a medical safety reviewer. Output **only JSON** (no markdown).

[Text to review]
{final_text}

[Structured analysis (optional)]
{json.dumps(analysis_structured or {}, ensure_ascii=False)}

JSON schema:
{{
  "risk_level": "low|medium|high",
  "is_safe": true,
  "reason": "short English rationale",
  "actions": ["action 1", "action 2"]
}}
"""
            try:
                before_gemini_request()
                resp = self._client.models.generate_content(
                    model=config.RISK_LLM_MODEL_ID,
                    contents=[prompt],
                )
                raw = (resp.text or "").strip()
                if raw.startswith("{") and raw.endswith("}"):
                    llm_assessment = json.loads(raw)
                else:
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    llm_assessment = json.loads(m.group(0)) if m else None
            except Exception as e:
                llm_assessment = {
                    "risk_level": "medium",
                    "is_safe": False,
                    "reason": f"LLM risk review failed; manual check needed. error={e!r}",
                    "actions": ["Manual review", "Retry model call"],
                }

        if llm_assessment:
            out = {
                "is_safe": bool(llm_assessment.get("is_safe", True)) and (not rule_hits),
                "risk_level": llm_assessment.get("risk_level", "low"),
                "reason": llm_assessment.get("reason", "LLM review complete"),
                "actions": llm_assessment.get("actions", []),
                "rule_hits": rule_hits,
                "llm_assessment": llm_assessment,
            }
            return out

        # 无 LLM 时的可解释回退
        return {
            "is_safe": len(rule_hits) == 0,
            "risk_level": "low" if not rule_hits else "medium",
            "reason": "Rules only (LLM risk review disabled or no client)",
            "actions": ["Pass"] if not rule_hits else ["Suggest human review"],
            "rule_hits": rule_hits,
            "llm_assessment": None,
        }
