"""
Phase 2: 混合检索模块 (Hybrid Retrieval)

MedicalRetriever：FAISS 稠密检索；可选 BM25（rank_bm25）与离线 JSON 知识图谱；
retrieve_context 使用 RRF 融合多路结果。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from medical_mvp import config

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class RetrievalHit:
    """单条检索命中结果（便于后续与 BM25 / 图谱结果对齐）。"""

    score: float
    text: str
    meta: dict[str, Any]


def _hit_dedup_key(hit: RetrievalHit) -> str:
    m = hit.meta
    if m.get("source") == "knowledge_graph":
        return str(m.get("graph_key", ""))
    if "id" in m:
        return f"qa:{m['id']}"
    return "text:" + hashlib.sha256(hit.text.encode("utf-8")).hexdigest()


class MedicalRetriever:
    """
    医学混合检索器：FAISS + 可选 BM25 + 可选离线图谱 JSON。
    """

    def __init__(
        self,
        qa_json_path: Path | None = None,
        embedding_model_name: str | None = None,
        index_path: Path | None = None,
        meta_path: Path | None = None,
        knowledge_graph_path: Path | None = None,
    ) -> None:
        self.qa_json_path = Path(qa_json_path or config.get_qa_database_path())
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
        self.index_path = Path(index_path or config.get_faiss_index_path())
        self.meta_path = Path(meta_path or config.get_faiss_meta_path())
        self.knowledge_graph_path = Path(
            knowledge_graph_path or config.get_knowledge_graph_path()
        )

        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._meta: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._graph_cache: dict[str, Any] | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _invalidate_bm25(self) -> None:
        self._bm25 = None

    def _record_to_text(self, rec: dict[str, Any]) -> str:
        """将一条 QA 记录拼接为用于向量化的文档字符串。"""
        q = rec.get("question", "")
        a = rec.get("answer", "")
        return f"Question: {q}\nAnswer: {a}"

    def build_faiss_from_qa_database(self, force_rebuild: bool = False) -> None:
        """
        读取 qa_database.json，编码为向量并构建 FAISS 内积索引（向量已 L2 归一化，等价于余弦相似度）。
        """
        if self.index_path.is_file() and self.meta_path.is_file() and not force_rebuild:
            self._load_faiss()
            return

        with open(self.qa_json_path, encoding="utf-8") as f:
            records: list[dict[str, Any]] = json.load(f)

        texts = [self._record_to_text(r) for r in records]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(embeddings, dtype=np.float32))

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        self._index = index
        self._meta = records
        self._invalidate_bm25()

    def _load_faiss(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)
        self._invalidate_bm25()

    def _ensure_index(self) -> None:
        if self._index is None:
            if not self.index_path.is_file():
                self.build_faiss_from_qa_database(force_rebuild=True)
            else:
                self._load_faiss()

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if not self._meta:
            return
        tokenized_corpus = [_tokenize(self._record_to_text(r)) for r in self._meta]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search_vector(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        """
        FAISS 向量检索：将 query 编码后与库内向量做相似度搜索。
        """
        self._ensure_index()
        assert self._index is not None
        k = top_k or config.FAISS_TOP_K
        qv = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores, idxs = self._index.search(np.asarray(qv, dtype=np.float32), k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._meta):
                continue
            rec = self._meta[idx]
            hits.append(
                RetrievalHit(
                    score=float(score),
                    text=self._record_to_text(rec),
                    meta=rec,
                )
            )
        return hits

    def search_bm25(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """BM25 稀疏检索（与 FAISS 共用 qa 元数据）。"""
        if not config.HYBRID_ENABLE_BM25:
            return []
        self._ensure_index()
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        raw_scores = self._bm25.get_scores(tokens)
        scores = np.asarray(raw_scores, dtype=np.float64)
        n = min(top_k, len(scores))
        if n <= 0:
            return []
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        hits: list[RetrievalHit] = []
        for idx in top_idx.tolist():
            rec = self._meta[int(idx)]
            hits.append(
                RetrievalHit(
                    score=float(scores[int(idx)]),
                    text=self._record_to_text(rec),
                    meta=rec,
                )
            )
        return hits

    def _load_graph_json(self) -> dict[str, Any]:
        if self._graph_cache is not None:
            return self._graph_cache
        if not self.knowledge_graph_path.is_file():
            self._graph_cache = {}
            return self._graph_cache
        with open(self.knowledge_graph_path, encoding="utf-8") as f:
            self._graph_cache = json.load(f)
        return self._graph_cache

    @staticmethod
    def _score_graph_node(query_l: str, q_tokens: set[str], node: dict[str, Any]) -> float:
        nid = str(node.get("id", "")).lower().replace("_", " ")
        score = 0.0
        aliases = [str(a).lower() for a in node.get("aliases", [])]
        if nid and nid in query_l:
            score += 3.0
        for a in aliases:
            if a and (a in query_l or query_l in a):
                score += 2.0
        blob = " ".join([nid] + aliases)
        blob_tokens = set(_tokenize(blob))
        score += float(len(q_tokens & blob_tokens)) * 0.5
        return score

    def search_graph(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """离线 JSON 图谱：nodes/edges，按别名与子串匹配命中实体并拼接一跳关系证据。"""
        if not config.HYBRID_ENABLE_GRAPH:
            return []
        data = self._load_graph_json()
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        if not nodes:
            return []

        query_l = query.lower()
        q_tokens = set(_tokenize(query_l))
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            s = self._score_graph_node(query_l, q_tokens, node)
            if s > 0:
                scored.append((s, node))
        scored.sort(key=lambda x: -x[0])

        hits: list[RetrievalHit] = []
        for score, node in scored[:top_k]:
            nid = str(node.get("id", ""))
            lines = [f"Knowledge graph — entity: {nid}"]
            shown = 0
            for e in edges:
                src, dst = e.get("src"), e.get("dst")
                if src != nid and dst != nid:
                    continue
                other = dst if src == nid else src
                rel = e.get("rel", "related")
                ev = e.get("evidence", "")
                lines.append(f"  ({rel}) {other}: {ev}")
                shown += 1
                if shown >= 8:
                    break
            text = "\n".join(lines)
            meta = {
                "source": "knowledge_graph",
                "graph_key": f"node:{nid}",
                "graph_node": nid,
                "graph_score": score,
            }
            hits.append(RetrievalHit(score=float(score), text=text, meta=meta))
        return hits

    def _rrf_merge(
        self,
        ranked_lists: list[list[RetrievalHit]],
        final_k: int,
        rrf_k: int,
    ) -> list[RetrievalHit]:
        scores: dict[str, float] = {}
        representatives: dict[str, RetrievalHit] = {}
        for hits in ranked_lists:
            if not hits:
                continue
            for rank, hit in enumerate(hits, start=1):
                key = _hit_dedup_key(hit)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                if key not in representatives:
                    representatives[key] = hit

        ordered_keys = sorted(scores.keys(), key=lambda k: -scores[k])
        out: list[RetrievalHit] = []
        for key in ordered_keys[:final_k]:
            base = representatives[key]
            out.append(
                RetrievalHit(
                    score=float(scores[key]),
                    text=base.text,
                    meta=dict(base.meta),
                )
            )
        return out

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        """
        主检索接口：FAISS；可选 BM25、离线图谱；多路时用 RRF 融合。
        """
        k_final = top_k or config.FAISS_TOP_K
        vec_hits = self.search_vector(query, top_k=k_final)

        lists: list[list[RetrievalHit]] = [vec_hits]
        if config.HYBRID_ENABLE_BM25:
            lists.append(self.search_bm25(query, top_k=config.BM25_TOP_K))
        if config.HYBRID_ENABLE_GRAPH:
            lists.append(self.search_graph(query, top_k=config.GRAPH_TOP_K))

        non_empty = [h for h in lists if h]
        if len(non_empty) <= 1:
            return non_empty[0] if non_empty else []

        return self._rrf_merge(lists, final_k=k_final, rrf_k=config.RRF_K)
