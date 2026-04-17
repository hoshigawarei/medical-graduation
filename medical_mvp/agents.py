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

        # 服务高峰时 Gemini 可能返回 503，使用指数退避 + 抖动自动重试。
        max_retries = 5
        base_wait_sec = 2.0
        response = None
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
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
        self.trace("影像推理完成。")
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
        self.trace("综合 Vision 与 RAG 证据，生成结构化医学分析…")

        top_hits = knowledge_hits[: config.ANALYSIS_TOPK_EVIDENCE]
        refs = []
        for i, h in enumerate(top_hits, start=1):
            refs.append(f"{i}. score={h.score:.4f} | {h.text}")
        ref_text = "\n".join(refs) if refs else "（无检索证据）"

        # 若缺少 API Key，回退为可解释的规则化输出，避免中断主流程。
        if self._client is None:
            self.trace("未配置 API Key，AnalysisAgent 回退为规则化摘要输出。")
            structured = {
                "primary_impression": "基于 VisionAgent 的初步结论（未进行 LLM 二次综合）",
                "differential_diagnoses": [],
                "supporting_evidence": refs[:2],
                "conflicts": [],
                "confidence": 0.5,
                "recommended_next_steps": ["补充临床病史与实验室检查", "必要时复查影像或使用更高分辨率图像"],
            }
            return {"analysis_report": vision_report, "analysis_structured": structured}

        prompt = f"""
你是医学分析智能体。请基于【用户问题】【Vision影像解释】【RAG证据】输出“仅 JSON”，不要输出任何额外文本。

【用户问题】
{user_question}

【Vision影像解释】
{vision_report}

【RAG证据（按相关度降序）】
{ref_text}

请输出 JSON，字段严格如下：
{{
  "primary_impression": "最终综合结论（中文）",
  "differential_diagnoses": [
    {{"name": "鉴别诊断1", "reason": "依据", "likelihood": 0.0}},
    {{"name": "鉴别诊断2", "reason": "依据", "likelihood": 0.0}}
  ],
  "supporting_evidence": ["证据1", "证据2"],
  "conflicts": ["冲突点1", "冲突点2"],
  "confidence": 0.0,
  "recommended_next_steps": ["下一步建议1", "下一步建议2"]
}}

要求：
1) likelihood 与 confidence 取 0~1 浮点数；
2) differential_diagnoses 最多 {config.ANALYSIS_DIFF_TOPK} 条；
3) 若证据冲突，请在 conflicts 明确写出；
4) 不得编造未提供的检查结果。
"""
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
                "primary_impression": "模型输出未严格 JSON，已回退为文本结论。",
                "differential_diagnoses": [],
                "supporting_evidence": refs[:2],
                "conflicts": [],
                "confidence": 0.5,
                "recommended_next_steps": ["建议人工复核该样本输出格式"],
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
            name = item.get("name", "未命名")
            likelihood = item.get("likelihood", "N/A")
            reason = item.get("reason", "")
            ddx_lines.append(f"- {name}（likelihood={likelihood}）：{reason}")

        analysis_report = (
            "【综合结论】\n"
            f"{primary}\n\n"
            f"【综合置信度】\n{confidence}\n\n"
            "【鉴别诊断 Top-K】\n"
            + ("\n".join(ddx_lines) if ddx_lines else "- 无\n")
            + "\n\n【支持证据】\n"
            + ("\n".join(f"- {x}" for x in evidence) if evidence else "- 无\n")
            + "\n\n【冲突证据】\n"
            + ("\n".join(f"- {x}" for x in conflicts) if conflicts else "- 未发现显著冲突\n")
            + "\n\n【建议下一步】\n"
            + ("\n".join(f"- {x}" for x in next_steps) if next_steps else "- 结合临床进一步评估\n")
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
        # 高风险规则：直接给药剂量、绝对化诊断、明显危险建议
        patterns = [
            (r"\b\d+\s?(mg|g|ml|mL)\b", "包含具体剂量信息，需人工核对用药安全"),
            (r"(100%|绝对|必须|唯一正确)", "存在绝对化表述，可能误导临床判断"),
            (r"(自行停药|无需就医|忽略症状)", "包含可能危险的就医/用药建议"),
        ]
        lowered = text.lower()
        for pattern, reason in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hits.append(reason)
        if "确诊" in text and "建议结合临床" not in text:
            hits.append("出现“确诊”措辞但缺乏谨慎免责声明")
        if "patient name" in lowered or "身份证" in text:
            hits.append("疑似包含隐私标识信息")
        return hits

    def run(self, final_text: str, analysis_structured: dict[str, Any] | None = None) -> dict[str, Any]:
        self.trace("执行风险评估：规则审查 + LLM 复核。")
        rule_hits = self._rule_scan(final_text) if config.RISK_RULES_ENABLED else []

        # 规则命中高危时直接拦截（优先级最高）
        hard_block = any("危险" in x or "剂量" in x for x in rule_hits)
        if hard_block:
            return {
                "is_safe": False,
                "risk_level": "high",
                "reason": "规则审查命中高风险项",
                "actions": ["阻断输出", "提示人工复核", "删除/改写高风险片段"],
                "rule_hits": rule_hits,
                "llm_assessment": None,
            }

        llm_assessment: dict[str, Any] | None = None
        if config.RISK_LLM_ENABLED and self._client is not None:
            prompt = f"""
你是医学安全与合规审核智能体。请基于内容输出“仅 JSON”：

【待审查文本】
{final_text}

【结构化分析（可选）】
{json.dumps(analysis_structured or {}, ensure_ascii=False)}

输出：
{{
  "risk_level": "low|medium|high",
  "is_safe": true,
  "reason": "简明中文理由",
  "actions": ["动作1", "动作2"]
}}
"""
            try:
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
                    "reason": f"LLM 风险复核失败，需要人工检查。error={e!r}",
                    "actions": ["人工复核该样本", "重试模型调用"],
                }

        if llm_assessment:
            out = {
                "is_safe": bool(llm_assessment.get("is_safe", True)) and (not rule_hits),
                "risk_level": llm_assessment.get("risk_level", "low"),
                "reason": llm_assessment.get("reason", "LLM 已复核"),
                "actions": llm_assessment.get("actions", []),
                "rule_hits": rule_hits,
                "llm_assessment": llm_assessment,
            }
            return out

        # 无 LLM 时的可解释回退
        return {
            "is_safe": len(rule_hits) == 0,
            "risk_level": "low" if not rule_hits else "medium",
            "reason": "仅执行规则审查（未启用或未配置 LLM 风险复核）",
            "actions": ["通过"] if not rule_hits else ["建议人工复核"],
            "rule_hits": rule_hits,
            "llm_assessment": None,
        }
