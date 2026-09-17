# 验收与证据

最后更新：2026-09-17。状态只对所列代码版本和配置有效。

状态含义：`PASS` 为当前版本有可复核证据；`PARTIAL` 为入口可运行但覆盖不足；`NOT RUN` 为尚无当前版本证据；`RESEARCH` 为需要实验而非单元检查。

| 编号 | 验收内容 | 当前状态 | 入口与证据 | 仍缺什么 |
|---|---|---|---|---|
| V01 | 完整状态恢复及分支不污染 | PASS | `python -m onpolicy.scripts.eval.preflight_pso`；见 `docs/evidence/2026-09-17-formal-preflight.txt` | 扩大随机种子只增加覆盖，不改变当前工程结论 |
| V02 | 空干预差为零 | PASS | `python -m onpolicy.scripts.eval.diagnose_pso_interventions`；见 `docs/evidence/2026-09-17-preformal.txt` | 增加多个种子可提高覆盖 |
| V03 | `c2=0` 领袖变化无效 | PASS | 同 V02 | 增加逐代轨迹断言，而不只比较终点 |
| V04 | 两分支同策略、配对随机数、各自响应状态 | PASS | 合并预检逐代复现相同分支、执行替代分支并恢复主端点；训练烟雾运行实际神经策略分支 | 研究分析仍需记录传播时延，不属于工程不变量 |
| V05 | 所有函数评估成本可独立对账 | PASS | 合并预检核对初始化、主轨迹、替代分支、独立评估及累计恒等式 | 正式结果仍需逐 Run 检查 manifest 与 JSONL 完整性 |
| V06 | 保存、加载、评估和汇总链路可用 | PASS | 合并预检保存并重载 Actor、Critic、反事实 Critic，参数逐项一致，重载后冻结评估可运行 | 汇总统计规则仍需冻结 |
| V07 | 标签打乱处理生效且科学对照公平 | PARTIAL | 历史标签池和 `cf_shuffle_label_changed` 已实现 | 数值/样本对应检查及策略阶段、尺度、函数实例分布审查 |
| V08 | 候选 Q 不泄露残留实际动作 | PASS | 合并预检用探针检查三个候选调用中目标粒子的联合动作均等于候选动作 | 网络仍同时接收候选 one-hot，这是当前明确设计 |
| V09 | 策略更新且评估冻结 | PASS | 合并预检确认最终 Actor 不同于同种子初始化，且评估前后 Actor、Critic、反事实 Critic 均逐项不变 | 无 |
| V10 | 日期结果目录与运行清单 | PASS | `--results_dir runs/2026-09-17-smoke` 短运行；见 `docs/evidence/2026-09-17-results-root-manifest.txt` | 正式运行应在干净提交上执行，使 manifest 的 `dirty=false` |
| V11 | 评估间隔与最终 checkpoint 正确 | PASS | 合并预检使用 3 个训练回合、评估间隔 2：第 1 回合不评估，第 2 回合评估，第 3 回合作为最终回合评估 | 正式批次应在第 10、20、...、200 回合评估 |
| R01 | 归因符号、排序和校准 | RESEARCH | 尚无独立诊断集 | 多状态、多未来种子参考效应与置信区间 |
| R02 | 跨粒子和长期效应 | RESEARCH | 传播日志字段部分存在 | 将受影响粒子和时间延迟转为预注册指标 |
| R03 | 预算公平的优化收益 | PARTIAL | 80 Run 主批次完整性和描述性比较通过；见 `docs/experiments/2026-09-17-primary-analysis.md` | 干预模式平均多用 12.43% 训练评估，仍需同总预算补跑 |

## 最小回归入口

语法检查：

```powershell
python -m compileall -q onpolicy
```

干预不变量：

```powershell
python -m onpolicy.scripts.eval.diagnose_pso_interventions
```

正式前合并预检：

```powershell
python -m onpolicy.scripts.eval.preflight_pso
```

小型链路检查只用于工程验收，实验名必须以 `smoke_` 开头，完成后删除其结果：

```powershell
python -m onpolicy.scripts.train.train_pso `
  --credit_mode cf_intervention `
  --experiment_name smoke_preformal `
  --pso_particles 4 --pso_dim 2 --pso_generations 5 `
  --num_env_steps 10 --n_rollout_threads 1 `
  --ppo_epoch 1 --num_mini_batch 1 `
  --log_interval 1 --use_wandb --cuda
```

## 证据规则

- 证据记录至少包含命令、提交或工作树状态、配置、输出和日期。
- 代码改变后，受影响的 PASS 必须重新运行或降级为待复核。
- 旧 smoke 结果、TensorBoard 截图和单种子最佳值不能证明研究有效。
- 标签“被替换”只证明处理路径执行，不能单独证明随机标签消融公平。
