"""
端到端评测辅助指标：与 gold 的 token F1/EM、检索证据对齐（RAG recall）、流水线结构化字段摘要。

与 eval_e2e 中的词集 Jaccard 共用同一套拉丁词元规则，便于数值可比。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from medical_mvp import config
from medical_mvp.retrieval import RetrievalHit

_WORD = re.compile(r"[a-z0-9]+", re.I)


def normalize_answer(s: str) -> str:
    """轻量规范化：用于 EM 与词元提取前的字符串清理。"""
    t = (s or "").strip().lower()
    return re.sub(r"\s+", " ", t)


def tokenize_for_eval(s: str) -> list[str]:
    """与 `jaccard_overlap` 一致的拉丁数字词元（PMC-VQA 英文场景）。"""
    return _WORD.findall(normalize_answer(s))


def jaccard_overlap(pred: str, gold: str) -> float:
    """词集合 Jaccard（与 eval_e2e 逻辑一致）。"""
    a, b = set(tokenize_for_eval(pred)), set(tokenize_for_eval(gold))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_f1(pred: str, gold: str) -> float:
    """词袋 F1（多标签计数 overlap / pred len / gold len）。"""
    p, g = tokenize_for_eval(pred), tokenize_for_eval(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / len(p)
    rec = overlap / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def pred_text_for_pipeline(workflow_out: dict[str, Any]) -> str:
    """
    流水线侧用于对齐 gold 的短文本：优先 JSON `primary_impression`，否则从报告截取综合结论段。
    """
    structured = workflow_out.get("analysis_structured")
    if isinstance(structured, dict):
        pi = structured.get("primary_impression")
        if isinstance(pi, str) and pi.strip():
            return pi.strip()
    report = str(workflow_out.get("analysis_report") or "")
    if "【综合结论】" in report:
        tail = report.split("【综合结论】", 1)[1]
        chunk = tail.split("\n\n【", 1)[0].strip()
        chunk = chunk.replace("【综合置信度】", "").strip()
        if chunk:
            return chunk[:4000]
    return report.strip()[:4000]


def pred_text_for_single_model(full_text: str) -> str:
    """单模型基线：用完整作答与 gold 对齐（一次多模态输出）。"""
    return (full_text or "").strip()


def rag_token_recall(pred: str, hits: list[RetrievalHit]) -> float | None:
    """
    检索词元在最终表述中的覆盖率：|T_ret ∩ T_pred| / |T_ret|。
    T_ret 为 Top-K 命中文本的词元并集；若无检索词元则返回 None。
    """
    if not hits:
        return None
    top = hits[: config.ANALYSIS_TOPK_EVIDENCE]
    ret_tokens: set[str] = set()
    for h in top:
        ret_tokens |= set(tokenize_for_eval(h.text))
    if not ret_tokens:
        return None
    pred_tokens = set(tokenize_for_eval(pred))
    return len(ret_tokens & pred_tokens) / len(ret_tokens)


def structured_audit_fields(workflow_out: dict[str, Any]) -> dict[str, Any]:
    """从 analysis_structured 抽取可审计统计（仅流水线）。"""
    structured = workflow_out.get("analysis_structured")
    out: dict[str, Any] = {
        "n_supporting_evidence": 0,
        "has_ddx": False,
        "confidence": None,
    }
    if not isinstance(structured, dict):
        return out
    ev = structured.get("supporting_evidence")
    if isinstance(ev, list):
        out["n_supporting_evidence"] = len(ev)
    ddx = structured.get("differential_diagnoses")
    out["has_ddx"] = isinstance(ddx, list) and len(ddx) > 0
    out["confidence"] = structured.get("confidence")
    return out


def risk_audit_fields(risk: dict[str, Any] | None) -> dict[str, Any]:
    """风控输出摘要。"""
    if not isinstance(risk, dict):
        return {"risk_level": None, "rule_hits_n": 0}
    rh = risk.get("rule_hits")
    n = len(rh) if isinstance(rh, list) else 0
    return {
        "risk_level": risk.get("risk_level"),
        "rule_hits_n": n,
    }
