"""
从 qa_database.json 构造演示用离线图谱（nodes + edges），激活混合检索中的图谱一路。

用法：
  python -m medical_mvp.build_knowledge_graph --write
  python -m medical_mvp.build_knowledge_graph --dry-run   # 只打印节点数量

输出路径：config.get_knowledge_graph_path()（默认数据根目录 knowledge_graph.json）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from medical_mvp import config

_WORD = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def build_graph(
    records: list[dict[str, Any]],
    *,
    min_token_len: int = 4,
    max_nodes: int = 400,
    max_edges_per_record: int = 15,
) -> dict[str, Any]:
    """统计问答文本中的高频词元为节点；同一记录答案内词元两两连边（共现）。"""
    freq: Counter[str] = Counter()
    for rec in records:
        blob = f"{rec.get('question', '')} {rec.get('answer', '')}"
        for t in _tokens(blob):
            if len(t) >= min_token_len:
                freq[t] += 1

    top_tokens = [w for w, _ in freq.most_common(max_nodes)]
    allowed = set(top_tokens)
    nodes = [{"id": t, "aliases": [t]} for t in top_tokens]

    edge_keys: set[tuple[str, str]] = set()
    for rec in records:
        ans_toks = [
            t
            for t in _tokens(str(rec.get("answer", "")))
            if t in allowed and len(t) >= min_token_len
        ]
        uniq = list(dict.fromkeys(ans_toks))
        n_tok = len(uniq)
        count = 0
        for i in range(n_tok):
            for j in range(i + 1, n_tok):
                if count >= max_edges_per_record:
                    break
                a, b = sorted([uniq[i], uniq[j]])
                key = (a, b)
                if key not in edge_keys:
                    edge_keys.add(key)
                    count += 1

    edges = [
        {
            "src": a,
            "dst": b,
            "rel": "co_occurs_in_answer",
            "evidence": "demo graph from qa_database token co-occurrence",
        }
        for a, b in sorted(edge_keys)
    ]

    return {"nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo knowledge_graph.json from qa_database")
    parser.add_argument("--write", action="store_true", help="写入 knowledge_graph.json")
    parser.add_argument("--dry-run", action="store_true", help="仅统计不写文件")
    parser.add_argument("--min-token-len", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=400)
    args = parser.parse_args()

    qa_path = config.get_qa_database_path()
    if not qa_path.is_file():
        raise SystemExit(f"未找到 {qa_path}，请先运行数据准备生成 qa_database.json")

    with open(qa_path, encoding="utf-8") as f:
        records = json.load(f)

    graph = build_graph(
        records,
        min_token_len=args.min_token_len,
        max_nodes=args.max_nodes,
    )

    out_path = config.get_knowledge_graph_path()
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])} -> {out_path}")

    if args.write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"已写入 {out_path}")
    elif not args.dry_run:
        print("未写入文件。添加 --write 写入，或 --dry-run 仅预览统计。")


if __name__ == "__main__":
    main()
