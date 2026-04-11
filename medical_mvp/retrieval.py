"""
Phase 2: 混合检索模块骨架 (Hybrid Retrieval)

MedicalRetriever 当前激活 FAISS 向量检索；BM25 与知识图谱为 Stub，
主入口 retrieve_context 预留多路融合扩展点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from medical_mvp import config


@dataclass
class RetrievalHit:
    """单条检索命中结果（便于后续与 BM25 / 图谱结果对齐）。"""

    score: float
    text: str
    meta: dict[str, Any]


class MedicalRetriever:
    """
    医学混合检索器：MVP 仅构建 FAISS 索引；BM25 与图谱检索接口已预留。
    """

    def __init__(
        self,
        qa_json_path: Path | None = None,
        embedding_model_name: str | None = None,
        index_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> None:
        self.qa_json_path = Path(qa_json_path or config.get_qa_database_path())
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
        self.index_path = Path(index_path or config.get_faiss_index_path())
        self.meta_path = Path(meta_path or config.get_faiss_meta_path())

        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._meta: list[dict[str, Any]] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

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

    def _load_faiss(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)

    def _ensure_index(self) -> None:
        if self._index is None:
            if not self.index_path.is_file():
                self.build_faiss_from_qa_database(force_rebuild=True)
            else:
                self._load_faiss()

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
        """
        BM25 稀疏检索（Stub）。

        TODO: 接入 rank_bm25 或 Elasticsearch，对同一 qa 语料构建倒排索引，
        并在 retrieve_context 中与向量分数做 RRF 或加权融合。
        """
        return []

    def search_graph(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """
        知识图谱检索（Stub）。

        TODO: 对接医学知识图谱（如 UMLS / 自建 Neo4j），基于实体链接与多跳查询
        返回结构化证据，再在 retrieve_context 与 FAISS/BM25 结果对齐融合。
        """
        return []

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        """
        主检索接口：当前仅返回 FAISS 结果。

        未来在本方法内融合：
        - search_vector 的稠密召回
        - search_bm25 的稀疏召回
        - search_graph 的结构化证据
        再通过重排（Cross-Encoder）或 RRF 合并为统一上下文列表，无需改动上层 Agent 调用约定。
        """
        return self.search_vector(query, top_k=top_k)
