# 离线图谱 JSON（`knowledge_graph.json`）

检索模块 [`MedicalRetriever.search_graph`](../retrieval.py) 读取数据根目录下的 **`knowledge_graph.json`**（路径见 `config.get_knowledge_graph_path()`）。

## Schema

```json
{
  "nodes": [
    { "id": "brain", "aliases": ["cerebrum", "cerebral"] },
    { "id": "liver", "aliases": [] }
  ],
  "edges": [
    {
      "src": "brain",
      "dst": "liver",
      "rel": "co_occurs",
      "evidence": "from PMC-VQA QA pair"
    }
  ]
}
```

- **`nodes`**：`id` 为匹配主键（建议小写短语）；`aliases` 为可选别名列表。
- **`edges`**：`src` / `dst` 必须对应已有 `node.id`；`rel`、`evidence` 供检索拼接展示。

若文件不存在或 `nodes` 为空，图谱一路不向 RRF 贡献命中（与「图谱未生效」现象一致）。

## 生成演示图谱

从当前 `qa_database.json` 自动抽取高频英文词元并建共现边：

```bash
python -m medical_mvp.build_knowledge_graph --write
```

默认写入 `MEDICAL_MVP_DATA_ROOT/knowledge_graph.json`（未设置环境变量时为项目 `data/`）。
