> Model: `models/gemini-3.7-flash`

# Minutes (English)

## Summary
The team met to review an AI-generated technical design and backlog document (55 findings) for the development of Sem2. They discussed adopting AI-assisted and agentic development workflows, establishing supervisory practices to prevent overcomplication, and reviewing/prioritizing specific architectural items (F01 through F09) before breaking for a department presentation.

## Key Decisions
- **AI-Assisted Workflow**: Adopt an AI-driven development model where clear planning documents, meeting transcripts, and prompt constraints supervise coding agents to prevent overcomplicated or destructive changes.
- **F01 (Capability Matrix & Stability Labels)**: Dropped / deprioritized. The project is in alpha stage; backward compatibility and formal stability labels are unnecessary overhead.
- **F02 & F03 (Named Dataset View Schema & Separating Model from Observational Status)**: Approved as **High Priority (P0/P1)**. Needed to separate the structural graph definition from dataset-specific observed/latent variable states.
- **F04 (Query Estimand Object Separate from Graph)**: Approved as **High Priority (P0/P1)**. Decouple query definitions (treatment $T$, outcome $Y$, contrasts, estimands) from global training configs and hardcoded variable names.
- **F05 & F06 (Observational/Interventional Regimes & Multiple Datasets)**: Set to **Low Priority**.
- **F07 & F08 (Empirical Joint Covariate Distribution & Vector-Valued Nodes)**: Set to **Medium-to-High Priority (P1)**. Multi-dimensional/vector-valued nodes prioritized over empirical sampling mechanisms.
- **F09 (Preprocessing Adapters & Categorical Encoding)**: Set to **Medium Priority**. A dedicated future meeting will address discrete/categorical variable handling.

## Action Items
- Generate meeting transcript and use AI agent to update the Sem2 backlog/planning document and merge the pull request (Speaker 1).
- Investigate deploying a specialized coding agent framework on the cloud VPS / local math server (Speaker 1).
- Attend Martina’s presentation in MR4 and reconvene to continue backlog review (All).

## Discussion Points
- **AI Coding Supervision**: Speaker 3 shared experiences with ChatGPT/Codex breaking slide layouts (e.g., fixing frame 27 broke frame 1), emphasizing the need to constrain AI to single-file or localized edits. Speaker 1 emphasized keeping context compaction low and managing specifications via well-defined markdown planning documents.
- **Agent Infrastructure**: Current Slack agent runs on Hermes framework on a 4-CPU cloud VPS, suited for general-purpose orchestration but lacking compute for extensive test execution.
- **Structural Model vs. Dataset Views**: The mathematical causal graph should strictly define structural mechanisms and variables, independent of whether specific nodes are observed or latent in a given dataset.
- **Estimand Architecture**: Current codebase hardcodes node names (e.g., $T$, $Y$) and estimands inside the training loss configuration. The target design will allow users to define arbitrary queries and extract multiple downstream estimands from a trained model instance.
- **Categorical & Discrete Handling**: Existing handling across forks (e.g., discrete instrumental variables) is fragmented and requires a dedicated technical design review.

## Open Questions
- Optimal deployment architecture for the specialized coding agent (Cloud VPS vs. internal departmental server, tool access permissions).
- Exact mathematical and software implementation for conditioning directly on empirical covariate distributions without fitting generative parametric models.

---

# 会议纪要 (中文)

## 摘要
团队讨论了由 AI 生成的 Sem2 软件技术评审与功能规划文档（共 55 项提议）。会议探讨了引入 AI 辅助编程及 Agent 开发流程的具体规范，确定了通过详尽设计文档和严格约束来监督 AI 的原则，并逐项评审了架构条目 F01 至 F09 的优先级。

## 关键决策
- **AI 开发流程**: 确立基于 AI Agent 的开发模式，通过会议纪要转录和结构化规划文档向 AI 明确需求，实施严格监督以避免引入过度复杂或破坏性的代码。
- **F01（功能矩阵与稳定性标签）**: 放弃/暂不实现。项目处于 Alpha 阶段，无需过早维护向后兼容性与稳定性分级。
- **F02 与 F03（数据集视图 Schema 与结构模型解耦）**: 确定为**高优先级（P0/P1）**。将因果图结构定义与具体数据集中变量的观测/隐变量状态完全分离。
- **F04（查询 Estimand 对象与图解耦）**: 确定为**高优先级（P0/P1）**。消除硬编码节点名称（如 $T$、$Y$）及配置文件中绑定的 Estimand，实现模型训练后灵活提取因果效应。
- **F05 与 F06（观测/干预机制与多数据集支持）**: 确定为**低优先级**。
- **F07 与 F08（协变量经验联合分布与多维向量节点）**: 确定为**中高优先级（P1）**。优先支持多维向量节点，再逐步完善经验分布采样。
- **F09（数据预处理适配器与分类变量编码）**: 确定为**中等优先级**。分类与离散变量的处理方案将在后续专门会议中深入探讨。

## 待办事项
- 获取本次会议转录文本，由 AI 更新规划文档并合并对应的 Pull Request（Speaker 1）。
- 调研在云端 VPS 或数学系本地服务器部署专用编程 Agent 的技术方案（Speaker 1）。
- 前往 MR4 参加 Martina 的讲座，讲座结束后视情况继续评审剩余条目（全体）。

## 讨论要点
- **AI 编程监督策略**: Speaker 3 分享了使用 ChatGPT/Codex 时因全局修改导致样式损坏的案例，建议对 AI 施加单次单文件修改等约束；Speaker 1 强调利用 Markdown 设计文档作为主要控制手段，避免上下文压缩丢失指令。
- **Agent 架构**: 当前 Slack Agent 基于 Hermes 框架运行在 4 核 VPS 上，适于任务编排，但不具备运行大型测试套件的算力，未来需配置专门的 Coding Agent。
- **因果图与数据解耦**: 模型图结构仅描述外生变量与结构方程，变量是否可观测属于数据集属性，不应混入模型本身的定义中。
- **Estimand 架构设计**: 当前系统将处理变量和结果变量硬编码在训练 Loss 配置中，未来应支持训练完成后通过单一指令提取任意因果参数（如 ATE）。
- **分类变量与离散 IV**: 目前不同分支（Fork）对离散变量的处理方式不统一，后续需组织专题会议制定统一标准。

## 未决问题
- 专用编程 Agent 的部署环境（云端 VPS 还是院系本地服务器）及权限配置方案。
- 在不拟合参数化生成模型的情况下，直接在图模型中对经验协变量分布进行条件化的具体软件实现方式。