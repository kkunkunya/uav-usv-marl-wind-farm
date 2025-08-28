第2部分:强化学习训练框架

# 0. 前置假设（与现状对齐）

- 环境：`src/env/wind_farm_env.py`，PettingZoo **ParallelEnv**，含动作掩码、Top-k 观测、全局 `state()` 或 `infos["global_state"]`（若未提供，按 §1.3 实装）。
- 路径/能量/队列：均已就绪（A.1–A.5、B.1–B.3）。
- 记录：事件/汇总 CSV、HTML 报告已就绪（E）。
- 目标：以 **MAPPO** 为主线、**QMIX** 为值分解对照，完成**可复现实验矩阵**（见 §5），并生成论文级图表与统计结论。

---

# 1. 体系结构与模块划分

## 1.1 训练运行器（Trainer Orchestrator）

- 位置：`src/rl/run_train.py`
- 责任：
    - 解析 `configs/rl/*.yaml`（算法级 + 场景级）。
    - 按 **exp_name/seed** 创建输出目录；写入 `manifest.json`（git hash / env hash / config hash）。
    - 初始化 **Ray**（或本机单进程）与 **并行环境池**（N 并发）。
    - 训练–评估–早停–checkpoint 统一调度（§4）。

## 1.2 环境适配与包装

- 位置：`src/rl/env_wrappers.py`
- 组件：
    - `EnvCreator(cfg)`：返回 PettingZoo ParallelEnv。
    - `ObsNormWrapper`：运行均值方差归一（可冻结于评估）。
    - `ActionMaskingAdapter`：为 RLlib/PPO/QMIX 提供统一的 `{"obs": x, "action_mask": m}` 结构。
    - `CentralStateProvider`：确保 `env.state()` 或 `infos["global_state"]` 存在（供 MAPPO/QMIX 的集中价值网络使用）。

## 1.3 全局状态定义（集中式价值函数所需）

- 推荐：**拼接式**全局状态
    
    s=[time_progress, T nodes (pos,status,τ)TopKT, C nodes (pos,queue)TopKC, agents (pos,SoC,mode)all]s=\big[\text{time\_progress},\ \text{T nodes (pos,status,}\tau\text{)}_{TopK_T},\ \text{C nodes (pos,queue)}_{TopK_C},\ \text{agents (pos,SoC,mode)}_{all}\big]
    
- 备选：直接使用 `env.state()` 返回的标准化向量。
- 说明：MAPPO 的 critic 与 QMIX 的 mixing network都使用 **相同的全局状态张量**，便于对齐。

## 1.4 策略映射（多策略/同策略）

- UAV 与 USV **共享一套 actor**（同动作集）或按类型拆分两套策略：
    - 共享策略（首选，样本更丰富）：`policy_mapping_fn → "shared_policy"`
    - 拆分策略（做消融）：`"uav_policy"`、`"usv_policy"`（critic 仍集中式）。

---

# 2. MAPPO（主线算法）实现规范

> MAPPO = 多智能体共享 Actor + 集中式 Critic 的 PPO 变体。训练期 critic 输入全局状态，actor 输入各自观测；评估与部署只需 actor。
> 

## 2.1 模型结构

- **Actor**：
    - 输入：个体观测 oao_a（含 Top-k 邻居等），可选 LayerNorm。
    - MLP：[256, 256]（ReLU/SiLU），**动作掩码**前置到 logits：
        - logits←MLP(oa)\text{logits} \leftarrow \text{MLP}(o_a)
        - logits[m==0]←−109\text{logits}[m==0] \leftarrow -10^{9}（防止非法动作采样）
- **Centralized Critic**：
    - 输入：全局状态 ss（§1.3），可与 actor 共享底层特征或独立 MLP。
    - MLP：[512, 512] → 标量 V(s)V(s)。

> 产物：src/rl/models/mappo_models.py（注册到 RLlib ModelCatalog 或自研 Trainer 使用）
> 

## 2.2 优势估计与损失

- **GAE(λ)**：
    
    A^t=∑l=0L−1(γλ)lδt+l,δt=rt+γV(st+1)−V(st)\hat A_t=\sum_{l=0}^{L-1}(\gamma\lambda)^l\delta_{t+l},\quad 
    \delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
    
- **PPO-Clip 损失**：
    
    Lπ=E[min⁡(rtA^t, clip(rt,1−ϵ,1+ϵ)A^t)]\mathcal{L}_{\pi}=\mathbb{E}\left[\min\left(r_t\hat A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]
    
- **价值损失**：LV=E[(Rt−V(st))2]\mathcal{L}_V=\mathbb{E}[(R_t - V(s_t))^2]，RtR_t 为折扣回报（或 GAE 目标）。
- **熵正则**：LH=−β⋅E[H(π(⋅∣oa))]\mathcal{L}_H=-\beta\cdot \mathbb{E}[H(\pi(\cdot|o_a))]。
- 总损失：L=Lπ+cvLV+LH\mathcal{L}=\mathcal{L}_\pi + c_v\mathcal{L}_V + \mathcal{L}_H。

## 2.3 采样与并行

- **并发环境数**：`n_envs = 32`（根据算力调整）
- **每迭代采样步**：`rollout_steps = 2048`（每 env），总步≈`n_envs×rollout_steps`。
- **GAE/回放**：on-policy，无经验回放；每迭代分 `minibatches = 8`，`epochs = 4`。

## 2.4 超参数（起点 & 网格）

- 固定：`γ=0.99, λ_GAE=0.95, clip=0.2, c_v=0.5, lr=3e-4, β(entropy)=0.01`
- 网格：
    - `lr ∈ {3e-4, 1e-4}`
    - `entropy ∈ {0.01, 0.005, 0.0}`
    - `rollout_steps ∈ {1024, 2048, 4096}`
    - `minibatch_size ∈ {8192, 16384}`（跨 env 合并）
- **动作掩码**：务必在 model forward 中**硬屏蔽**非法动作（logits = −∞）。

## 2.5 训练流程（逐步）

1. **Warm-up（10–20 iters）**：
    - 关闭/减弱负载均衡项（λσ=0 或 0.1），仅优化时间与安全；
    - 关闭扰动、使用 Small 场景、Top-k 节点较小（k=5）。
2. **Curriculum**：
    - 逐步开启 `O_uav` 禁飞、增大 `ρ_fan`、引入 Medium 场景与排队瓶颈；
    - 逐步提高 λσ 至目标值（0.2–0.4）。
3. **Stabilization**：
    - 冻结 ObsNorm；减小 entropy；加大 rollout_steps。
4. **Target phase**：
    - 在目标配置上跑到**性能平稳**或早停（§4.3）。

## 2.6 评估 Actor（无探索）

- 每 N iter 对**验证 seed 列表**执行 `episodes_eval=20`：
    - `explore=False`（贪婪/均匀决策）；
    - 记录 **Tmax、σ、成功率、Etot、等待时长**；
    - **词典序**选择最优 checkpoint（先 Tmax，再 σ）。

---

# 3. QMIX（对照算法）实现规范

> QMIX 使用个体 Q 网络 + 可分解的混合网络：
> 
> 
> Qtot(s,u)=fmix(s; {Qa(oa,ua)}a),∂Qtot∂Qa≥0Q_{tot}(s,\mathbf{u}) = f_{\text{mix}}\big(s;\ \{Q_a(o_a,u_a)\}_a\big),\quad \frac{\partial Q_{tot}}{\partial Q_a}\ge 0
> 

## 3.1 组件

- **个体 Q 网络**：MLP [256, 256]，输入 oao_a，输出离散动作 Q 值；**掩码**将非法动作 Q 置为 −∞。
- **Mixing Network**：两层超网络生成加权和的参数，输入全局状态 ss，权值非负（绝对值或 softplus 保证）。
- **经验回放**：`replay_size = 1e6`，`batch = 64`，`target_update_interval = 200`。

## 3.2 训练设置

- `γ=0.99`，ε-greedy 从 1.0 线性退火到 0.05（50k–200k 步），学习率 `lr=5e-4`。
- **多步回报**：n-step=3（可调）。
- **并行收集**：`n_envs = 32`；每步推送回放。

## 3.3 评估

- 与 MAPPO 同步：固定评估 seeds、`explore=False`，同指标与词典序准则。

---

# 4. 训练–评估–早停–Checkpoint 统一协议

## 4.1 输出目录规范

```
experiments/
  <exp_name>/
    mappo/ (或 qmix/)
      seed_0001/
        checkpoints/ckpt_<iter>.pth
        eval/iter_<n>/{events.csv,summary.csv,metrics.json}
        train_logs.jsonl
        manifest.json

```

## 4.2 记录的关键指标（每 iter）

- 训练：`policy_loss, value_loss, entropy, kl, explained_var, grad_norm, fps, env_steps`
- 评估：`Tmax_median, Tmax_IQR, sigma_median, success_rate, Etot_mean, wait_total_mean`
- 资源：`GPU_util, CPU_mem, cache_hit_rate`

## 4.3 早停与最佳模型选择

- **早停**：验证集 **Tmax_median** 连续 `patience=5` 次未提升；或达到 `max_env_steps`（如 3e6）。
- **最佳模型**：按**词典序**在所有评估点中选取；写入 `best.json`。

---

# 5. 实验矩阵与调度（RL 专用）

| 维度 | 取值 |
| --- | --- |
| 场景 | Small / Medium（Large 作为最终报告） |
| 禁区 | `O_uav` on/off；`ρ_fan ∈ {60, 100, 140}` |
| 能量 | Ecap_UAV ∈ {120,160,200}；reserve∈{0.1,0.15} |
| 充电 | C×K ∈ {(1,1),(2,2)} |
| 算法 | MAPPO / QMIX |
| λσ | {0, 0.1, 0.2, 0.4} |
| 种子 | 每配置 ≥ 20（论文最终 ≥ 30） |
- 调度脚本：`src/rl/run_grid.py`（串/并行执行、失败重试、断点续跑）。
- 输出聚合：`scripts/aggregate.py` 产出 `metrics_agg.parquet` 与作图数据。

---

# 6. 稳定性与调参建议（常见坑位）

1. **动作掩码遗漏** → 采样非法动作、训练发散
    - 检查：训练日志中 `violation` 必须≈0；手动钩子断言 logits[mask=0] = −∞。
2. **优势尺度过大** → PPO 震荡
    - 处理：优势标准化、`grad_clip=0.5`、降低 lr 或加大 batch。
3. **奖励量级失衡** → 学不到均衡
    - 处理：对 `ΔTmax`、`Δσ` 做量纲对齐（先做 Small 场景网格扫 λ）。
4. **评估抖动** → 不显著
    - 处理：评估 episodes ≥ 20，做中位数与 IQR；固定验证 seed 集。
5. **QMIX 不收敛**
    - 提高 n-step、延长 ε 退火、增加 replay 采样温度（优先采稀有队列拥堵片段）。

---

# 7. 统计检验与出图（RL 专项）

- **统计**：对 **MAPPO vs QMIX**、**λσ**、**禁区开关**做 **Wilcoxon**（α=0.05），报告效应量 `r = Z/√N`。
- **图表**（自动从 `metrics_agg.parquet` 生成）：
    1. `Tmax` 箱线（按算法分组）；
    2. `σ` 箱线；
    3. 成功率柱状；
    4. 帕累托前沿（`Tmax`–`σ`，标注算法/配置）；
    5. 训练曲线（迭代 → 验证 `Tmax_median` 与 `σ_median`）；
    6. 充电等待时间 CDF（拥堵/非拥堵配置对比）。

---

# 8. 复现与发布清单（RL 侧）

- **固定**：训练/评估 seed 列表、`config.yaml`、`rl_config.yaml`、代码版本（git hash）。
- **导出**：`best_policy.pt`（actor），`obs_norm_stats.pkl`，`evaluation_report.md/pdf`，`figures/*.png`。
- **一键复现**：
    
    ```
    python src/rl/run_train.py --cfg configs/rl/mappo_small.yaml --exp mappo_small --seeds 20
    python scripts/aggregate.py --exp mappo_small
    python scripts/make_report.py --exp mappo_small
    
    ```
    

---

# 9. 任务拆解（Issue 模板，RL 专用）

### R1｜Env Wrapper & State Provider〔M〕

- **输入**：ParallelEnv
- **输出**：具备 ObsNorm、ActionMask、GlobalState 的包装器
- **验收**：单步 sanity（mask/obs/state 形状一致）；随机策略评估能跑完回合。

### R2｜MAPPO 模型与 Trainer 插件〔L〕

- **内容**：Actor（掩码 logits）、Centralized Critic（用 state）、GAE、PPO-Clip、梯度裁剪
- **验收**：Small 场景 50k 步后高于贪心的 `Tmax_median`。

### R3｜QMIX 模型接入〔L〕

- **内容**：个体 Q、Mixing、n-step、ε-greedy、回放
- **验收**：Small 场景收敛曲线下降；评估可复现。

### R4｜训练运行器与 Checkpoint 管理〔M〕

- **内容**：迭代日志、周期评估、早停、最佳模型选择
- **验收**：生成 `best.json` 与 `ckpt_*`，可一键恢复继续训练。

### R5｜实验矩阵执行器与聚合〔M〕

- **内容**：批量跑矩阵、失败重试、聚合 parquet
- **验收**：产出 `metrics_agg.parquet`，字段齐全。

### R6｜统计与作图模块〔S-M〕

- **内容**：Wilcoxon、效应量、箱线/帕累托/CDF
- **验收**：生成论文级图表与 `report.md`。

### R7｜稳定性与回归测试〔M〕

- **内容**：动作掩码断言、优势/梯度范围监控、评估抖动门限
- **验收**：CI 跑 3×Small 种子 < 10 分钟全绿。