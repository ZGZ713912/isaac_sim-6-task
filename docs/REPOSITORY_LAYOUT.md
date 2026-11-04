# 仓库文件归属

Git 保存可维护源码、正式合同、测试与必要测试数据、机械模型及长期维护的说明。
文件是否由 AI 生成不决定是否提交；文件的用途决定归属。

| 内容 | 保存位置 | 进入 Git |
| --- | --- | --- |
| 项目入口、长期协作约定 | 根目录 `README.md`、`AGENTS.md` | 是 |
| 架构、接口、使用指南、模型机制、正式验收要求 | `docs/` | 是 |
| 可执行代码、测试、正式合同、机械源模型与认证必需的元数据 | `src/`、`scripts/`、`tests/`、`contracts/`、`assets/`、`model/` | 是 |
| 源码/资产旁的维护说明、许可证与第三方声明 | 原来源目录 | 是 |
| Codex 报告、计划、调查、运行流水与关联图表/数据 | `.codex/report/<主题>/` | 否 |
| OpenCode 报告及关联数据 | `.opencode/report/<主题>/` | 否 |
| 一次性脚本、草稿、中间产物 | 对应 AI 目录的 `scratch/` | 否 |
| 检查点、ONNX、导出包、冻结快照 | 对应 AI 目录的 `artifacts/` | 否 |
| 本地缓存、外部工具 | 对应 AI 目录的 `cache/`、`tools/` | 否 |

`.codex/`、`.opencode/` 整体忽略，不能为其中任何文件添加忽略例外或强制提交。
报告即使是 Markdown，也属于本地目录。`docs/` 不存放一次运行的报告或过程回执；
README 不链接本地报告。将长期有效的系统知识整理成独立说明后，才放入 `docs/`。
根目录 `AGENTS.md` 只保留协作规则，进度与 PID 等运行流水放进本地报告。

旧入口 `reports`、`.codex/reports`、`models` 可以作为指向本地目录的兼容链接；
链接本身同样不进入 Git。新增操作直接使用 `.codex/report/` 或 `.opencode/report/`。
机械源模型目录 `model/` 与训练导出目录 `models/` 用途不同；模型运行所需的参数、
manifest 和认证说明保留版本管理，导出权重与临时评测结果保存在本地。

提交前运行：

```bash
python scripts/check_repository_hygiene.py
```

启用仓库附带的本地提交检查：

```bash
git config core.hooksPath .githooks
```

CI 会检查整个 Git 索引，所以已被强制加入的文件也会拒绝。忽略规则不会自动清除
已经跟踪的文件；已提交的过程材料应先保存本地原件，再从跟踪及历史中移除。
历史改写必须保留仓库外备份和旧/新提交映射，并明确远端同步范围。
