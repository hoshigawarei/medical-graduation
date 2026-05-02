"""
检索层评测：同库自检 Recall@K / MRR，多配置消融（FAISS_only / FAISS_BM25 / FAISS_Graph / Full_hybrid）。

用法：
  python -m medical_mvp.eval_retrieval --n 100 --seed 42 --top-k 10

评测说明：对每条样本用其 question 作为查询，考察 gold 文档（qa_database 中同 id 条目）
在 retrieve_context 返回列表中的名次；用于横向对比稠密 / BM25 / 图谱 / RRF 融合策略。
这是一种「同库自检检索」，结论侧重相对优劣；严格无外泄评测需另行划分训练语料。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any

from medical_mvp import config
from medical_mvp.retrieval import MedicalRetriever, RetrievalHit


def _bm25_installed() -> bool:
    try:
        import rank_bm25  # noqa: F401

        return True
    except ImportError:
        return False


def _norm_id(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _gold_rank(hits: list[RetrievalHit], gold_id: Any) -> int | None:
    gid = _norm_id(gold_id)
    for i, h in enumerate(hits, start=1):
        mid = h.meta.get("id")
        if mid is None:
            continue
        if _norm_id(mid) == gid or str(mid) == str(gid):
            return i
    return None


VARIANT_PRESETS: dict[str, tuple[bool, bool]] = {
    "FAISS_only": (False, False),
    "FAISS_BM25": (True, False),
    "FAISS_Graph": (False, True),
    "Full_hybrid": (True, True),
}


def _apply_variant(name: str) -> tuple[bool, bool]:
    if name not in VARIANT_PRESETS:
        raise ValueError(f"未知 variant: {name}，可选: {list(VARIANT_PRESETS)}")
    bm25_on, graph_on = VARIANT_PRESETS[name]
    config.HYBRID_ENABLE_BM25 = bm25_on
    config.HYBRID_ENABLE_GRAPH = graph_on
    return bm25_on, graph_on


def _save_config_state() -> tuple[bool, bool]:
    return config.HYBRID_ENABLE_BM25, config.HYBRID_ENABLE_GRAPH


def _restore_config_state(state: tuple[bool, bool]) -> None:
    config.HYBRID_ENABLE_BM25, config.HYBRID_ENABLE_GRAPH = state


def _sample_records(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, n)


def evaluate_variant(
    retriever: MedicalRetriever,
    samples: list[dict[str, Any]],
    top_k: int,
    ks: list[int],
    *,
    include_per_sample: bool,
) -> dict[str, Any]:
    recalls = {k: 0 for k in ks}
    rr_sum = 0.0
    ranks_detail: list[dict[str, Any]] = []

    for rec in samples:
        q = str(rec.get("question", ""))
        gid = rec.get("id")
        hits = retriever.retrieve_context(q, top_k=top_k)
        rank = _gold_rank(hits, gid)
        if include_per_sample:
            ranks_detail.append({"id": gid, "rank": rank, "n_hits": len(hits)})
        if rank is None:
            continue
        rr_sum += 1.0 / float(rank)
        for k in ks:
            if rank <= k:
                recalls[k] += 1

    n = len(samples)
    out: dict[str, Any] = {
        "n_samples": n,
        "recall": {f"@{k}": recalls[k] / n if n else 0.0 for k in ks},
        "mrr": rr_sum / n if n else 0.0,
    }
    if include_per_sample:
        out["per_sample"] = ranks_detail
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层 Recall@K / MRR 消融评测")
    parser.add_argument("--n", type=int, default=100, help="抽样条数（不超过全集）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--top-k", type=int, default=10, help="retrieve_context 截断长度（建议 >= max(K)）")
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10",
        help="Recall@K 的 K 列表，逗号分隔",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="FAISS_only,FAISS_BM25,FAISS_Graph,Full_hybrid",
        help="逗号分隔 variant 名，或 all",
    )
    parser.add_argument(
        "--include-per-sample",
        action="store_true",
        help="在 JSON 中写入每条样本的 rank（体积较大）",
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if args.top_k < max(ks, default=1):
        print(
            f"[WARN] top_k={args.top_k} 小于 max(K)={max(ks)}，Recall@max 将被低估。",
            file=sys.stderr,
        )

    variant_list = (
        list(VARIANT_PRESETS.keys())
        if args.variants.strip().lower() == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )

    qa_path = config.get_qa_database_path()
    if not qa_path.is_file():
        print(f"[ERR] 未找到 {qa_path}，请先运行数据准备。", file=sys.stderr)
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    samples = _sample_records(records, args.n, args.seed)
    bm25_ok = _bm25_installed()
    graph_path = config.get_knowledge_graph_path()
    graph_exists = graph_path.is_file()

    saved = _save_config_state()
    results: dict[str, Any] = {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qa_path": str(qa_path),
        "n_pool": len(records),
        "n_samples": len(samples),
        "seed": args.seed,
        "top_k": args.top_k,
        "ks": ks,
        "bm25_installed": bm25_ok,
        "knowledge_graph_path": str(graph_path),
        "knowledge_graph_exists": graph_exists,
        "variants": {},
    }

    retriever = MedicalRetriever()
    retriever.build_faiss_from_qa_database()

    try:
        for vname in variant_list:
            _apply_variant(vname)
            if vname in ("FAISS_BM25", "Full_hybrid") and not bm25_ok:
                print(
                    f"[WARN] variant={vname} 需要 rank-bm25，当前未安装，BM25 一路为空，指标可能等同于仅向量。",
                    file=sys.stderr,
                )
            if vname in ("FAISS_Graph", "Full_hybrid") and not graph_exists:
                print(
                    f"[WARN] variant={vname} 需要图谱文件，当前不存在: {graph_path}",
                    file=sys.stderr,
                )

            metrics = evaluate_variant(
                retriever,
                samples,
                args.top_k,
                ks,
                include_per_sample=args.include_per_sample,
            )
            results["variants"][vname] = {
                "HYBRID_ENABLE_BM25": config.HYBRID_ENABLE_BM25,
                "HYBRID_ENABLE_GRAPH": config.HYBRID_ENABLE_GRAPH,
                **metrics,
            }
    finally:
        _restore_config_state(saved)

    results["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    out_dir = config.get_data_root() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_retrieval_{ts}.json"
    csv_path = out_dir / f"eval_retrieval_{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "variant",
        "n_samples",
        "mrr",
    ] + [f"recall@{k}" for k in ks]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vname, payload in results["variants"].items():
            row = {
                "variant": vname,
                "n_samples": payload["n_samples"],
                "mrr": f"{payload['mrr']:.6f}",
            }
            for k in ks:
                key = f"@{k}"
                row[f"recall@{k}"] = f"{payload['recall'][key]:.6f}"
            w.writerow(row)

    print(f"[eval_retrieval] JSON: {json_path}")
    print(f"[eval_retrieval] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
