> Model: `models/gemini-3.7-flash`

# Minutes (English)

## Summary
The team reviewed an AI-generated architectural and feature roadmap for the `sem2` causal inference/structural equation modeling library, triaging items into immediate priorities, deferred research topics, and cleanup tasks. Key decisions included implementing exact identifiability checks via the ID algorithm, modularizing and reducing notebook examples into documentation and tests, and transitioning project task coordination to dedicated, asynchronous Slack threads leading to GitHub PRs.

## Key Decisions
- **Intervention Testing & Multi-treatment Support (F32)**: Prioritized extending treatment variable handling from single strings to lists of strings and adding test cases (e.g., sequential randomized trials with intermediate confounding).
- **Identifiability & ID Algorithm (F33)**: Decided to implement the ID algorithm on single-margin ADMGs for structural identifiability checks; deferred empirical lower/upper bound identifiability testing to a future release.
- **Partial Identification Scope (F34, F44)**: Confirmed retaining core partial identification capabilities (e.g., discrete IV linear programming bounds) for the stable beta release while deferring complex training extensions (continuation, multi-start sweeps).
- **Notebook & Repository Cleanup (F45)**: Agreed to consolidate exploratory notebooks into a single `docs/` demo folder, preserving three key user-facing examples (linear SEM comparison with lavaan, confounder adjustment simulation, and discrete IV partial identification), while porting regression checks into standalone Python test scripts in `tests/`.
- **Packaging and Metadata Cleanup**: Decided to remove outdated `.gitmodules` and consolidate environment specifications (`pyproject.toml` vs. `requirements.txt`).
- **Documentation Standards (F46)**: Adopt standard Python docstring conventions and investigate automated documentation generation (e.g., Read the Docs / Sphinx) using structured templates.
- **Workflow Execution Model**: Work will proceed via specific Slack threads per topic with AI assistance, each aiming to produce a focused PR for review and merge.

## Action Items
- **Speaker 1**: Generate meeting transcript, summary, and the 3 action/feature lists via AI, and open a PR on the repository.
- **Speaker 1**: Create dedicated Slack discussion threads for prioritized tasks (e.g., documentation setup, ID algorithm) and tag relevant contributors.
- **Speaker 2 (Max) & Team**: Review package dependency configuration (`pyproject.toml` vs `requirements.txt`) and resolve duplicate setup files.
- **Speaker 3 (Chiara) / Speaker 4 (Pan)**: Consult previous package release experiences regarding open-source licensing and distribution models.
- **Speaker 1 / AI**: Generate a `.ics` calendar invite for the next meeting.

## Discussion Points
- **Causal Queries & Model Fitting Workflow**: Discussed decoupling initial model fitting from specific causal query definitions, making query-specific retraining optional (specifically required when doing partial identification).
- **Graph Visualization**: Considered adding simple graph plotting functionality upon model initialization to help users visually verify DAG specifications.
- **Licensing and Release**: Discussed open-source license tradeoffs (permissive vs. copyleft, commercial reuse constraints) ahead of eventual public release.

## Open Questions
- Optimal hosting and build pipeline for package documentation (Read the Docs vs. GitHub Pages / Sphinx).
- Feasibility and effectiveness of using previously fitted model weights as warm starts during partial identification retraining without getting trapped in local optima.

---

# 会议纪要 (中文)

## 摘要
团队审查了AI为 `sem2` 因果推断/结构方程模型库生成的架构与功能规划文档，对各项任务进行了优先级排序，明确了当前稳定版（Beta）范围、待推迟的研究课题及代码库清理计划。会议决定采用 ID 算法实现精确的可识别性检验，将现有 Jupyter Notebook 梳理为规范文档与独立测试用例，并确立了基于 Slack 专题讨论生成 GitHub PR 的异步协作模式。

## 关键决策
- **干预语义测试与多变量支持 (F32)**：确定将干预处理变量从单一字符串扩展为字符串列表，并增加测试用例（如包含中间混杂因素的序贯随机试验）。
- **可识别性检验与 ID 算法 (F33)**：决定通过将 DAG 边际化为 ADMG 并应用 ID 算法来实现结构可识别性检查；基于数值上下界的经验可识别性测试推迟至后续版本。
- **部分识别范围控制 (F34, F44)**：当前 Beta 版本保留基础部分识别能力（如离散工具变量的线性规划界），暂不引入复杂的优化退火与多起点扫描等工程扩展。
- **Notebook 与目录结构重构 (F45)**：清理过时的 notebook，将用户演示统一归入 `docs/` 目录，仅保留三个核心示例（与 lavaan 对比的线性 SEM、混杂调整模拟、离散 IV 部分识别），其余转换为 `tests/` 下的标准 Python 自动化测试脚本。
- **代码库与环境配置清理**：清理过期的 `.gitmodules`，统一规范环境安装配置（合并/梳理 `pyproject.toml` 与 `requirements.txt`）。
- **文档规范 (F46)**：采用标准 Python docstring 格式，调研自动化文档生成工具（如 Read the Docs / Sphinx），先确立单模块模板再批量推广。
- **协作流程**：针对具体任务在 Slack 中开设独立讨论线程（Thread），借助 AI 辅助探讨方案，最终产出并合并 GitHub PR。

## 待办事项
- **Speaker 1**：整理会议转录、纪要及三份任务清单并提交 PR。
- **Speaker 1**：在 Slack 中建立各专题讨论线程并 @ 相关成员。
- **Speaker 2 (Max) 及团队**：梳理依赖管理配置，确定 `pyproject.toml` 与 `requirements.txt` 的最佳整合方案。
- **Speaker 3 (Chiara) / Speaker 4 (Pan)**：借鉴以往类似因果推断工具包的开源与许可证选择经验提供建议。
- **Speaker 1 / AI**：生成下周会议的 `.ics` 日历邀请文件。

## 讨论要点
- **因果查询与拟合流程解耦**：讨论了将通用模型拟合与特定因果查询分离的流程设计，在需要部分识别时再执行针对查询的模型重新训练。
- **图结构可视化**：探讨在模型初始化时增加快速绘制 DAG 图的功能，帮助用户直观校验公式定义的模型结构。
- **开源许可证选择**：探讨了不同开源许可证（如商业友好型与传染型协议）对后续衍生项目和学术引用的影响。

## 未决问题
- 文档自动化构建与托管的具体选型（Read the Docs 还是 GitHub Pages / Sphinx）。
- 部分识别重训练时，使用已拟合模型作为热启动（Warm Start）是否会导致陷入局部最优解。