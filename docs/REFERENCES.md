# 文献与主张边界

最后核验：2026-09-17。

本文件记录文献能支持的具体主张及其不能支持的结论。检索未发现相同工作不等于不存在相同工作，新颖性判断需要持续复核。

## 已核验文献

### MAPPO

- Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” NeurIPS 2022 Datasets and Benchmarks Track.
- 论文：https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html
- 官方代码：https://github.com/marlbenchmark/on-policy
- 支持：共享策略、集中训练和 PPO 可作为协作多智能体基线。
- 不支持：MAPPO 对 PSO 领袖选择必然有效，也不支持本项目的因果或新颖性结论。

### COMA

- Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson. “Counterfactual Multi-Agent Policy Gradients.” AAAI 2018. DOI: 10.1609/aaai.v32i1.11794.
- 页面：https://ojs.aaai.org/index.php/AAAI/article/view/11794
- 支持：集中 Critic 下固定其他智能体动作、边缘化单个智能体动作的反事实优势。
- 不支持：真实 PSO 分支干预监督是 COMA 原方法的一部分。

### 历史信用分配

- James M. Whitacre, Tuan Q. Pham, Ruhul A. Sarker. “Credit Assignment in Adaptive Evolutionary Algorithms.” GECCO 2006. DOI: 10.1145/1143997.1144206；arXiv:0907.0592.
- 页面：https://arxiv.org/abs/0907.0592
- 支持：利用与未来代历史关联的表现为搜索算子分配信用并非新概念。
- 与本项目差异：归因对象、PSO 记忆传播、干预方式和 Actor-Critic 训练监督不同，仍需逐项比较。

### PSO 收缩系数

- Maurice Clerc, James Kennedy. “The Particle Swarm: Explosion, Stability, and Convergence in a Multidimensional Complex Space.” IEEE Transactions on Evolutionary Computation, 2002. DOI: 10.1109/4235.985692.
- 页面：https://ieeexplore.ieee.org/document/985692
- 支持：经典收缩系数 PSO 的参数化与稳定性分析背景。
- 不支持：加入学习领袖策略后的整体系统继承原理论保证。

### CLPSO

- Jing J. Liang, A. K. Qin, P. N. Suganthan, S. Baskar. “Comprehensive Learning Particle Swarm Optimizer for Global Optimization of Multimodal Functions.” IEEE Transactions on Evolutionary Computation, 2006. DOI: 10.1109/TEVC.2005.857610.
- 页面：https://doi.org/10.1109/TEVC.2005.857610
- 支持：复杂学习对象选择是已有 PSO 路线，因此 CLPSO 适合作为后续对照而非第一版骨架。

### 模型式反事实想象

- Jiajun Chai, Yuqian Fu, Dongbin Zhao, Yuanheng Zhu. “Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination.” AAMAS 2024, pp. 281–289.
- 论文：https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf
- 支持：使用反事实轨迹改善长期多智能体信用分配已有直接近邻研究。
- 与本项目差异：该工作依赖学习的世界模型；本项目候选方法利用真实、可复制的 PSO 优化器状态并显式计入函数评估。

### Counterfactual Shapley Credit Assignment

- Mingxuan Li, Kai-Zhan Lee, Elias Bareinboim. “Counterfactual Shapley Credit Assignment.” RLC 2026 / Reinforcement Learning Journal 2026；arXiv:2607.16999；CausalAI Technical Report R-148.
- 页面：https://arxiv.org/abs/2607.16999
- 支持：共享外生随机性下的反事实 Shapley 归因、时间信用重分配及模拟成本权衡。
- 与本项目差异：当前 CSPSO 标签是一次动作替换的配对终点差，不是 Shapley 联盟分解，也不应使用 Shapley 名称。

## 待继续核验

- PSO 中以个体记忆版本为事件的因果归因是否已有更接近的方法。
- 面向演化算法或群智能的反事实分支训练监督。
- 同时覆盖跨粒子传播、长期效应、真实干预监督和预算公平评估的直接近邻工作。
- 上述检索需记录数据库、关键词、日期和纳入排除标准后，才能形成论文中的新颖性陈述。
