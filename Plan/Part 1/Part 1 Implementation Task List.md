# 实现任务清单

# 一、实现任务清单（Issue 列表）

## Epic A｜数据解析与地图层生成

**Definition of Ready (DoR)**：拿到 `content.png`、`content.xml` 与一份最小 `config.yaml`（含 gsd_m_per_px、grid_res_m、各缓冲半径）。

**Definition of Done (DoD)**：可稳定输出三类层（任务/站点、障碍/禁区、安全缓冲），并转为世界坐标、可视化叠加检查。

### A1｜XML 解析器（VOC→对象表）〔M〕

- **输入**：`content.xml`
- **输出**：`objects.parquet`（列：id, cls, bbox_px, center_px, center_world_m, poly_world, attrs）
- **要点**：bbox→中心点；像素→世界坐标（使用 gsd_m_per_px；图像中心为原点）；多边形闭合。
- **依赖**：无
- **DoD**：对象计数与类别分布打印；10 个随机样本与像素叠加可视化无偏移。
- **测试**：随机抽样 IoU≥0.6（矩形重投影校验）；异常 bbox（x_min>x_max）能报错。
- **风险**：XML 类名异写（大小写/空格）；需做别名表。

### A2｜缓冲与层构建（Shapely/GEOS）〔M〕

- **输入**：A1 输出、`config.yaml`
- **输出**：`layers.pkl`（T/C 点集；O_hard/O_uav/风机缓冲多边形）
- **要点**：对 `island`、`UAV obstacle`、`fan` 分别按 ρ 缓冲；自交修复；统一 CRS（米）。
- **DoD**：层级统计（面积、数量）；叠加到底图输出 `layers_preview.png`。
- **测试**：缓冲后多边形无自交；分层互斥关系正确（风机缓冲∩任务点=∅）。

### A3｜占据栅格与可通行域（UAV/USV 双域）〔M〕

- **输入**：A2、grid_res_m
- **输出**：`grid_uav.npz`, `grid_usv.npz`（二值栅格/邻接权重）
- **DoD**：通行率（可通行格/总格）统计；导出 GeoTIFF/PNG 供人工检查。
- **测试**：硬障碍在两域皆不可通行；UAV 额外屏蔽 uav_obstacle 与风机缓冲。

### A4｜导航图与最短路缓存（多对多）〔L〕

- **输入**：A3、节点集 V（T∪C∪{B}）
- **输出**：`D_uav.npy`, `D_usv.npy`（|V|×|V| 距离矩阵；不可达=+∞）；`paths_cache.zst`（可选）
- **DoD**：距阵对称性/三角不等式抽查；抽样路径可视化。
- **优化**：KNN 邻接 + 按需补全；Zstd 压缩。
- **测试**：1000 对样本上 A* 可达↔掩码可行一致；路径长度与直线下界对比≤1.2×。

---

## Epic B｜环境核心与事件机制

**DoR**：完成 A Epic；确定 `dt`、`horizon_s`、奖励权重。

**DoD**：可在 Small 场景“跑通一遍”，产出事件日志与汇总指标。

### B1｜状态机与模式流转〔M〕

- **模式**：transit/service/queue/charge/return/idle
- **DoD**：模式图与不变式（互斥、能量守恒、终止完备）单元测试通过。

### B2｜事件队列与半离散推进〔M-L〕

- **事件**：ArriveNode/StartService/EndService/QueueJoin/QueuePop/StartCharge/EndCharge/FailPanic
- **DoD**：一个决策周期内可处理多事件；跨周期事件时间戳正确。
- **测试**：极端高频充电下不丢事件。

### B3｜能量模型与安全余量掩码〔M〕

- **公式**：E_move=α·d，E_srv=η_srv·τ，E_idle=η_idle·Δt
- **掩码**：不可达、不可回站、不可完工即置 0。
- **DoD**：临界 SoC 场景测试（±ε）能精确开/关。

### B4｜充电与排队（固定站 + 可选移动站）〔M〕

- **排队**：FCFS，多桩 K 并发，速率 r_j。
- **DoD**：等待时间统计与理论 M/D/1 形态一致（趋势）。

### B5｜观测打包与动作掩码输出〔M〕

- **观测**：Top-k 近邻（T/C/友邻）、时间进度、自身态；归一化。
- **DoD**：动作空间与掩码长度完全对齐；越界动作自动 STAY + violation++。

### B6｜奖励与标量化策略〔S〕

- **训练**：线性加权；**评估**：词典序（先 Tmax，再 σ）。
- **DoD**：奖励量级 sanity check（各项同量级）。

### B7｜日志与回放（CSV + Polyline）〔M〕

- **事件日志字段**：见规范；加入 `map_hash/xml_hash/config_hash`。
- **DoD**：可用脚本将一次仿真 replay 成动图/GeoJSON。

---

## Epic C｜验证、基线与可视化

### C1｜属性测试套件（Invariant Tests）〔M〕

- **包含**：可达性、能量边界、队列公平、任务互斥、日志完整性。
- **DoD**：`pytest -q` 全绿；CI 中跑 Small 场景 3 种 seed。

### C2｜基线算法适配层〔L〕

- **MILP（任务分配 + 简化路径代价）**：PuLP/OR-Tools 任一
- **启发式**：最近邻、贪心、GA-SA
- **DoD**：能读写同一 `env` 的距离矩阵与任务集；输出统一计划格式供环境重放评估。

### C3｜绘图与报告模板〔M〕

- **图型**：Tmax 箱线、σ 雷达/散点、成功率柱状、充电等待 CDF、帕累托前沿。
- **DoD**：一键从实验目录生成 `figures/` 与 `report.md`。

---

## Epic D｜RL 训练接入与实验编排

### D1｜PettingZoo × RLlib 适配层（ParallelEnv）〔M〕

- **DoD**：MAPPO 可开箱即训；`env_creator(config)` 提供。

### D2｜MAPPO 配置与超参网格〔M〕

- **关键超参**：γ, λ_GAE, lr_actor/critic, clip, vf_coef, entropy_coef, batch_size, n_rollout_steps, n_envs
- **DoD**：默认配置可收敛到“优于贪心”的成绩。

### D3｜QMIX（或其它值分解）配置〔L〕

- **DoD**：能在同一观测/动作上训练；记录混合网络结构参数。

### D4｜实验矩阵执行器（Hydra/自研脚本）〔M〕

- **功能**：按矩阵调度 seeds×场景×算法×权重；产出目录结构与命名规范。
- **DoD**：失败自动重试，局部断点续跑。

---

## Epic E｜工程化与可复现

### E1｜配置与哈希管控〔S〕

- **DoD**：所有输出（cache/日志/图表）带 SHA-256；`manifest.json` 记录版本。

### E2｜CLI 与 README〔S〕

- `build_layers → build_grids → build_cache → run_env → eval_plan → make_report`
- **DoD**：README 跑通“5 分钟上手”。