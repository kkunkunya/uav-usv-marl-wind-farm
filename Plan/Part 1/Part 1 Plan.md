# 第1部分：仿真环境设计

---

# A. 数据→环境的端到端流水线

## A.1 输入工件（不可变真值）

- 影像：`content.png`（宽 W=2048W=2048、高 H=2048H=2048）。
- 标注：`content.xml`（PASCAL VOC 树，含对象类别与 `<bndbox>`）。
- 研究者配置：`config.yaml`（见附录，含米/像素、缓冲半径、速度、能耗参数、是否启用移动充电等）。

> 术语映射
> 
> - **任务点 TT**：来自类别 `fan`（风机），每个风机即一个任务。
> - **固定充电站 CC**：来自类别 `charge station`。
> - **硬障碍 OhardO^{hard}**：来自类别 `island`（对 USV 与 UAV 均不可穿越；UAV 同时视作禁飞缓冲区）。
> - **UAV 软障碍 OuavO^{uav}**：来自类别 `UAV obstacle`（仅限制 UAV；USV 可忽略，除非在 `config.yaml` 中指定也为浅滩/禁航）。

## A.2 像素坐标→世界坐标

- 采用右手、屏幕系到世界系的一致线性变换：
    
    x=(u−u0)⋅s,y=(v0−v)⋅s,\begin{aligned}
    x &= (u - u_0)\cdot s,\\
    y &= (v_0 - v)\cdot s,
    \end{aligned}
    
    其中 (u,v)(u,v) 为像素中心坐标（VOC 取 center=xmin+xmax2, ymin+ymax2\text{center}=\frac{x_{min}+x_{max}}{2},\ \frac{y_{min}+y_{max}}{2}），
    
    s=gsd_m_per_pxs=\texttt{gsd\_m\_per\_px} 为地面分辨率（米/像素），(u0,v0)=(W/2,H/2)(u_0,v_0)=(W/2,H/2) 为图像中心。
    
- **来源**：若没有外部标定，ss 作为实验变量给定；建议论文中报告 s∈{2,5,10}s\in\{2,5,10\} m/px 三档以做灵敏度分析。

## A.3 多层几何体生成

把 XML 矩形框统一转换为多边形并做安全膨胀，得到三套**层（layers）**：

1. **任务/站点层**
    - 任务节点 T={ti}\mathcal{T}=\{t_i\}：用 `fan` 框中心作为任务坐标 x(ti)x(t_i)。
    - 充电站 C={cj}\mathcal{C}=\{c_j\}：用 `charge station` 框中心作为站点坐标 x(cj)x(c_j)。
2. **障碍/禁区层**
    - OhardO^{hard}：`island` 框 → 多边形 → 按 ρisland\rho^{island}（米）做形态学膨胀（缓冲），得到禁航/禁飞区。
    - OuavO^{uav}：`UAV obstacle` 框 → 多边形 → 按 ρuav\rho^{uav}（米）膨胀，得到禁飞区。
3. **安全缓冲层（风机）**
    - 每个 `fan` 再生成 UAV 安全圈 B(ti,ρfan)B(t_i,\rho^{fan})（典型取 50–150 m，可做消融），避免低空穿桨。

> 实现要点
> 
> - 均以**世界坐标**（米）做缓冲；缓冲结果再栅格化/用于可达性测试。
> - 所有多边形保持**闭合且无自交**（Shapely/GEOS 层面校验）。

## A.4 双介质可通行域与占据栅格

- 地图矩形工作域 Ω=[xmin,xmax]×[ymin,ymax]\Omega=[x_{min},x_{max}]\times[y_{min},y_{max}]。
- 分辨率 rr（m/格，建议 5–20 m）：
    - **UAV 栅格**：可通行域 ΩUAV=Ω∖(Ohard∪Ouav∪⋃iB(ti,ρfan))\Omega^{UAV}=\Omega \setminus \big(O^{hard}\cup O^{uav}\cup \bigcup_i B(t_i,\rho^{fan})\big)。
    - **USV 栅格**：可通行域 ΩUSV=Ω∖Ohard\Omega^{USV}=\Omega \setminus O^{hard}（如需把 `UAV obstacle` 视为浅滩，则额外并入）。
- 输出二值占据栅格 GUAV,GUSV\mathcal{G}^{UAV}, \mathcal{G}^{USV} 与对应的 8 邻接/16 邻接权重图（代价=栅格中心欧氏距离）。

## A.5 导航图与最短路缓存

- **多源多汇最短路**（A*/Dijkstra）：
    - 节点集 V=T∪C∪{B}∪采样航点V = \mathcal{T}\cup \mathcal{C}\cup \{B\}\cup \text{采样航点}。
    - 对每一对 (p,q)∈V2(p,q)\in V^2 ，分别在 GUAV\mathcal{G}^{UAV}、GUSV\mathcal{G}^{USV} 上求最短可行路，得到距离 dUAV(p,q)d^{UAV}(p,q)、dUSV(p,q)d^{USV}(p,q)。
- **缓存矩阵** Dtype∈R∣V∣×∣V∣D^{type}\in\mathbb{R}^{|V|\times|V|}，训练/求解阶段 O(1) 读取。
- **不可达对**：距离置为 +∞+\infty，并在动作掩码中禁用。

---

# B. 实体、动力学与约束（地图感知版）

## B.1 任务与服务

- 任务位置：x(ti)x(t_i) 取自 `fan` 框中心。
- 服务时长 τi\tau_i：若无外部表，设定为对数正态分布 LogN(μτ,στ)\mathrm{LogN}(\mu_\tau,\sigma_\tau)（报告中位数与 IQR）；或统一常数以便做消融。
- **服务安全**：若 UAV 到达时预计 SoC\mathrm{SoC} 不能满足“完成任务并到达最近可行充电点”，则**禁止开工**（动作掩码实现）。

## B.2 智能体运动与能耗

- 旅行时间： ttype(p ⁣→ ⁣q)=dtype(p,q)vtype\ t^{type}(p\!\to\!q)=\frac{d^{type}(p,q)}{v^{type}}。
- 能耗（线性路程模型，先不考虑速度平方项）：
    
    Emove=αtype⋅dtype(p,q),Esrv=ηtypesrv⋅τi,Eidle=ηtypeidle⋅Δt.E^{move} = \alpha_{type}\cdot d^{type}(p,q),\quad 
    E^{srv}=\eta^{srv}_{type}\cdot \tau_i,\quad 
    E^{idle}=\eta^{idle}_{type}\cdot \Delta t.
    
- **能量安全余量**
    
    SoC ≥ Etypereserve+min⁡q∈C∪{B}αtype⋅dtype(p,q).\mathrm{SoC}\ \ge\ E^{reserve}_{type} + \min_{q\in \mathcal{C}\cup\{B\}} \alpha_{type}\cdot d^{type}(p,q).
    

## B.3 充电与排队

- 固定站 cjc_j：来自 `charge station`，每站并发桩数 KjK_j、充电速率 rjr_j 在配置中给定。
- 排队：FCFS，进入/离开事件写入日志（用于还原等待时间分布）。
- 移动充电（可选）：USV 作为移动站，半径 RchgR^{chg}、速率 rUSVr^{USV}；受 USV SoC 约束。

---

# C. 观测、动作与奖励（与数据层联动）

## C.1 观测（PettingZoo ParallelEnv）

- 自身态：x~,SoC~,L~,mode one-hot\tilde{x},\tilde{\mathrm{SoC}},\tilde{L},\text{mode one-hot}。
- 邻近 Top-k **真实节点**（基于 DtypeD^{type} 距离）：任务/充电/友邻的坐标、状态、排队长度、到达代价。
- 时间进度 t~=t/Thorizon\tilde{t}=t/T_{horizon}。
- **动作掩码**：若 dtype(p,q)=∞d^{type}(p,q)=\infty 或违反能量安全，则掩码 0。

## C.2 动作（高层离散）

- `GO_TASK(i)`、`GO_CHARGE(j)`、`GO_USV(k)`、`STAY`。
- 环境内部通过 DtypeD^{type} 赋予行程时间与能耗；真实轨迹用于日志可视化（可复用 A* 路径点列）。

## C.3 多目标奖励（Tmax + 均衡 σ）

- 训练用线性加权，评估用词典序（先 Tmax⁡T_{\max}，再 σ\sigma）：
    
    Rt=−λTΔTmax⁡−λσΔσ−λEΔE−λQΔt⋅Waiters.R_t = -\lambda_T \Delta T_{\max}-\lambda_\sigma \Delta \sigma-\lambda_E \Delta E -\lambda_Q \Delta t\cdot \text{Waiters}.
    

---

# D. 地图特化的生成与校验

## D.1 统计校验（XML→对象）

- **计数一致性**：解析后应给出
    
    ∣T∣=#fan|\mathcal{T}|=\#\text{fan}，∣C∣=#charge station|\mathcal{C}|=\#\text{charge station}，∣Ohard∣=#island|O^{hard}|=\#\text{island}，∣Ouav∣=#UAV obstacle|O^{uav}|=\#\text{UAV obstacle}。
    
- **几何覆盖**：将缓冲后的 OhardO^{hard}、OuavO^{uav} 叠加到底图上，人工抽查 5–10 处 IoU 是否合理（≥0.6 期望）。

## D.2 可达性与距离回归测试

- **UAV/USV 双域**：随机采样 1000 对 (p,q)(p,q)，若 Gtype\mathcal{G}^{type} 上 A* 返回可达，则 `GO_*` 掩码必为 1；若不可达则为 0。
- **距离/时间对比**：对 100 对样本，验证
    
    ttype(p ⁣→ ⁣q)≈dtype(p,q)vtype\ t^{type}(p\!\to\!q)\approx \frac{d^{type}(p,q)}{v^{type}}；若采用 8 邻接，允许 ≤1.1×1.1\times 误差。
    

## D.3 能量边界与安全回站

- 构造“刚好能去–能回”的临界 SoC 场景：
    
    令 SoC=Esrv+αd(p,ti)+min⁡qαd(ti,q)+ϵ\mathrm{SoC}=E^{srv}+\alpha d(p,t_i)+\min_q \alpha d(t_i,q)+\epsilon。
    
    - 当 ϵ≥0\epsilon\ge 0 时允许开工；ϵ<0\epsilon<0 时掩码应禁止。

---

# E. 统一日志与可复现（地图感知字段）

- 事件日志（CSV）：
    
    `sim_id, seed, step, time, agent_id, event_type, from_node, to_node, px,py, soc_before, soc_after, energy_delta, task_id, station_id, queue_len_after, global_Tmax, global_sigma, unfinished_tasks, map_hash, xml_hash`
    
    - `map_hash`= `content.png` 的 SHA-256（字节级），`xml_hash`= `content.xml` 的 SHA-256。
- 汇总表：
    
    `sim_id, seed, |A|, |T|, |C|, |O_hard|, |O_uav|, Tmax, sigma, Jain, Etot, charge_wait_total, violations, gsd_m_per_px, r_res_m, rho_fan, rho_island, rho_uav`
    

---

# F. 关键数学与实现细节（与地图数据强绑定）

## F.1 由像素框到世界多边形

- VOC 框 B=[xmin,ymin,xmax,ymax]B=[x_{min},y_{min},x_{max},y_{max}] 对应四点像素坐标 → 世界坐标后形成矩形多边形 PP。
- 缓冲运算：P⊕B(0,ρ)P\oplus \mathbb{B}(0,\rho)；所有缓冲半径以米为单位。

## F.2 双图距离函数

- 对每个介质 m∈{UAV,USV}m\in\{UAV,USV\}，定义
    
    dm(p,q)={A*(Gm,p,q),若连通+∞,若不可达d^m(p,q)=
    \begin{cases}
    \text{A*}(\mathcal{G}^m,p,q),& \text{若连通}\\
    +\infty,& \text{若不可达}
    \end{cases}
    
- 旅行时间、能耗与掩码、奖励均基于 dmd^m。

## F.3 负载均衡与统计量

- La=∑i∈Ta(ta,itravel+τi)L_a=\sum_{i\in \mathcal{T}_a}(t^{travel}_{a,i}+\tau_i)，
    
    σ=1∣A∣∑a(La−Lˉ)2\sigma=\sqrt{\frac{1}{|\mathcal{A}|}\sum_a (L_a-\bar L)^2}， Lˉ=1∣A∣∑aLa\bar L=\frac{1}{|\mathcal{A}|}\sum_a L_a。
    
- 论文可同时汇报 **Jain 指数** J=(∑aLa)2∣A∣∑aLa2J=\frac{(\sum_a L_a)^2}{|\mathcal{A}|\sum_a L_a^2}。

---

# G. 性能测试与基线（地图特定）

## G.1 解析与渲染基准

- **解析时间**：XML→层生成→栅格化在 Medium 场景（~几百个对象）应 ≤ 2s（单线程 Python 参考），否则调高 rr 或做分块。
- **路径缓存命中**：训练期**命中率**≥95%（∣V∣|V| 小时建议全对预计算；∣V∣|V| 大时采用 KNN 邻接+按需缓存）。

## G.2 拓扑健康度

- **联通性**：
    - UAV：至少存在 90% 的任务对最近充电站连通；
    - USV：所有充电站间应连通（若不连通需在 `config.yaml` 指定可通水道或添加航点）。

## G.3 统计显著性

- 每个配置跑 ≥30 个随机种子（对 τi\tau_i、扰动、起点分布采样），对 Tmax⁡,σT_{\max},\sigma 做 Wilcoxon 检验（p<0.05p<0.05）以比较：
    - **是否启用 `O^{uav}`**（真实海上养殖区带来的禁飞影响）；
    - **不同缓冲半径 ρfan\rho^{fan} / ρuav\rho^{uav}**；
    - **不同 ss（米/像素）** 尺度误差下的鲁棒性。

---

# H. 与训练/求解的接口契约

- PettingZoo `ParallelEnv`：`reset(seed, config_path) → (obs, infos)`；`step(action_dict) → (next_obs, rewards, term, trunc, infos)`。
- `infos` 字段必须包含：`dist_cache_hits, queue_len@cj, next_event_time, current_path_polyline, node_ids`。
- **动作掩码提供**：`infos[a]["action_mask"]`（布尔向量，与离散动作空间一一对应）。

---

# I. 默认配置建议（片段）

```yaml
# config.yaml（片段）
map:
  image: content.png
  xml: content.xml
  gsd_m_per_px: 5.0         # 地面分辨率
  grid_res_m: 10.0          # 栅格分辨率 r
  buffer:
    rho_fan: 80.0           # 风机禁飞缓冲
    rho_uav: 40.0           # UAV obstacle 膨胀
    rho_island: 10.0        # 岛屿外扩（保守）
agents:
  uav:
    n: 5; v: 15.0; Ecap: 160; alpha: 0.20; reserve_ratio: 0.15; eta_srv: 0.005; eta_idle: 0.0005
  usv:
    n: 1; v: 5.0;  Ecap: 1200; alpha: 0.05; reserve_ratio: 0.10; eta_srv: 0.0
charging:
  fixed:
    default_rate: 6.0       # 能量单位/秒
    default_K: 2
  mobile:
    enable: false
    R: 50.0
    rate: 3.0
reward:
  lambda_T: 1.0; lambda_sigma: 0.2; lambda_E: 0.01; lambda_Q: 0.05
sim:
  dt: 2.0; horizon_s: 36000
disturb:
  v_sigma: 0.05; alpha_sigma: 0.05; dynamic_nfz: false

```

---

# J. 交付与验收清单（开发需落地）

1. **解析器**：XML→对象层（含缓冲与世界坐标转换）；输出 `layers.pkl`（任务/站点/障碍多边形+hash）。
2. **栅格器**：按 `grid_res_m` 生成 GUAV,GUSV\mathcal{G}^{UAV},\mathcal{G}^{USV}，并导出 GeoTIFF/PNG 供肉眼检视。
3. **最短路缓存器**：生成 DUAV,DUSVD^{UAV}, D^{USV} 与可视化热力图（抽样对）。
4. **环境核心**：ParallelEnv（事件队列、排队、能量、掩码、日志）。
5. **验证脚本**：
    - 计数/IoU/连通性/能量边界/距离回归单元测试；
    - 关键统计图（箱线图/等待时间柱状图/可达性矩阵）。
6. **可复现说明**：固定 `config.yaml` + `seed_list.txt`，导出哈希与最终 CSV 汇总表。