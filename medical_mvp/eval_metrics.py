"""
端到端评测辅助指标：字面 token F1/Jaccard、BERTScore（语义）、实体召回、RAG 对齐、结构化字段（含鉴别诊断 vs gold）。

BERTScore 需安装 `bert-score`（见 requirements.txt）；未安装时对应字段为 null。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from medical_mvp import config
from medical_mvp.retrieval import RetrievalHit

_WORD = re.compile(r"[a-z0-9]+", re.I)
_WORD_BOUNDARY = re.compile(r"(?<![a-z0-9])([a-z0-9]+)(?![a-z0-9])", re.I)


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
    """词袋 F1（legacy：字面重叠）。"""
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


def bertscore_f1(pred: str, gold: str) -> float | None:
    """
    BERTScore F1（语义相似度）。未安装 bert-score 或空文本时返回 None。
    使用英文默认模型；首次运行会下载权重。
    """
    p, g = (pred or "").strip(), (gold or "").strip()
    if not p or not g:
        return None
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None
    try:
        _p, _r, f1 = bert_score_fn([p], [g], lang="en", rescale_with_baseline=False)
        return float(f1.mean().item())
    except Exception:
        return None


def entity_token_recall_vs_gold(pred: str, gold: str) -> float:
    """
    gold 答案中的唯一拉丁词元，有多少以「词边界」出现在 pred 中（集合 recall）。
    无 gold 词元时视为 1.0。
    """
    gtoks = list(dict.fromkeys(tokenize_for_eval(gold)))
    if not gtoks:
        return 1.0
    pred_l = normalize_answer(pred)
    hit = 0
    for t in gtoks:
        if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", pred_l):
            hit += 1
    return hit / len(gtoks)


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
    if "Primary impression" in report:
        tail = report.split("Primary impression", 1)[1].lstrip("\r\n :")
        if "\n\nOverall confidence" in tail:
            chunk = tail.split("\n\nOverall confidence", 1)[0].strip()
        else:
            chunk = tail.split("\n\n", 1)[0].strip()
        if chunk:
            return chunk[:4000]
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


def ddx_gold_coverage(structured: dict[str, Any] | None, gold: str) -> dict[str, Any]:
    """
    鉴别诊断列表是否与参考答案在词级或子串上对齐（演示级启发式，非临床诊断准确率）。
    """
    out: dict[str, Any] = {
        "ddx_covers_gold": False,
        "ddx_gold_match_type": None,
    }
    if not isinstance(structured, dict):
        return out
    g_norm = normalize_answer(gold)
    gtoks = set(tokenize_for_eval(gold))
    if not g_norm and not gtoks:
        return out

    for item in structured.get("differential_diagnoses") or []:
        if not isinstance(item, dict):
            continue
        blob = f"{item.get('name', '')} {item.get('reason', '')}"
        n_norm = normalize_answer(blob)
        nt = set(tokenize_for_eval(blob))
        if gtoks and (gtoks & nt):
            out["ddx_covers_gold"] = True
            out["ddx_gold_match_type"] = "token_overlap"
            return out
        if len(g_norm) >= 3 and g_norm in n_norm:
            out["ddx_covers_gold"] = True
            out["ddx_gold_match_type"] = "substring"
            return out
        if len(g_norm) >= 3 and any(g_norm in normalize_answer(str(item.get(k, ""))) for k in ("name", "reason")):
            out["ddx_covers_gold"] = True
            out["ddx_gold_match_type"] = "substring_field"
            return out
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


def attach_extended_metrics(
    row: dict[str, Any],
    scoring: str,
    gold: str,
    structured: dict[str, Any] | None,
) -> None:
    """写入 BERTScore、实体召回、ddx vs gold（流水线传入 structured）。"""
    row["bertscore_f1"] = bertscore_f1(scoring, gold)
    row["entity_token_recall_vs_gold"] = entity_token_recall_vs_gold(scoring, gold)
    if structured is not None:
        row.update(ddx_gold_coverage(structured, gold))
    else:
        row["ddx_covers_gold"] = None
        row["ddx_gold_match_type"] = None
