Role and Persona
你是一位资深的人工智能架构师，正在指导一名计算机专业的本科生完成他的毕业设计代码原型。你的目标是编写高内聚、低耦合、模块化且带有详尽中文注释的 Python 代码，代码需在 Google Colab 环境下运行。

Project Context
本项目是“基于检索增强与智能体的大语言模型在医学领域的应用”系统。

最终架构：包含 4 个智能体（医学分析、知识检索、影像解释、风险评估）以及 3 路混合 RAG 检索（FAISS, BM25, 知识图谱）。

当前阶段目标 (MVP)：搭建完整的代码骨架。仅实现 FAISS 检索和其中 2 个核心智能体的底层逻辑，其余模块使用 Stub (占位符) 预留接口。

模型与数据：使用 google-genai SDK 调用 gemini-2.5-flash；使用 datasets 库流式拉取 xmcmic/PMC-VQA 的前 200 条数据。

Execution Plan (Step-by-Step)
请严格按阶段生成代码，每完成一个阶段请停下来等我确认。

Phase 1: 数据准备与预处理
挂载 Google Drive 并流式拉取 PMC-VQA 前 200 条数据，保存图片并生成 qa_database.json。

Phase 2: 混合检索模块骨架 (Hybrid Retrieval)
创建一个 MedicalRetriever 类：

实现 FAISS 检索 (已激活)：利用 sentence-transformers 和 faiss-cpu 将文本向量化并实现 search_vector()。

预留 BM25 检索 (Stub)：创建方法 search_bm25()，当前只需 pass 或返回空列表，并加上 TODO 注释。

预留知识图谱检索 (Stub)：创建方法 search_graph()，当前只需 pass 或返回空列表，并加上 TODO 注释。

主检索接口：创建 retrieve_context(query) 方法，目前仅调用并返回 FAISS 的结果，但要在注释中说明未来将在这里融合三路数据。

Phase 3: 智能体角色定义 (Agent Definitions)
设计一个基础的 BaseAgent 类，并派生出 4 个智能体：

KnowledgeAgent (知识检索)：接收指令，调用 Phase 2 的 retrieve_context 返回医学参考。

VisionAgent (影像解释)：调用 Gemini API 处理图片和文本。提示词需设定为“你是一个影像解释智能体...”。

AnalysisAgent (医学分析 - Stub)：当前阶段仅作为一个透传接口（Pass-through），直接将用户问题转发给 VisionAgent。预留 TODO：未来负责综合分析。

RiskAgent (风险评估 - Stub)：当前阶段始终返回 {"is_safe": True, "reason": "MVP阶段跳过审查"}。预留 TODO：未来负责校验最终输出是否违规。

Phase 4: 工作流调度中心 (Workflow Controller)
编写一个 ClinicalWorkflow 函数来模拟诊疗全流程：

接收输入：用户问题 + 医疗影像。

调度流：

先调用 KnowledgeAgent 查阅资料。

将资料和图片一起交给 VisionAgent 进行读片推理。

将结果交给 AnalysisAgent (当前直接放行)。

最后交由 RiskAgent 审查 (当前默认通过)。

在控制台打印出清晰的思维链轨迹 (Trajectory)，用 [Agent Name] 标明当前是哪个智能体在工作。

Phase 5: 运行测试
随机抽取 3 条本地数据，送入 ClinicalWorkflow，验证系统是否能顺畅跑通预留的骨架，并输出带医学参考的诊断结果。

Constraints
必须使用 google-genai SDK 的最新规范。

代码架构必须易于扩展，方便后续补充 BM25 和知识图谱的代码，而不需要重构主循环。