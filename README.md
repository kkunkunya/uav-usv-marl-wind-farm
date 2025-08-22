# 海上风电场UAV-USV协同任务分配仿真平台

多智能体强化学习框架，用于海上风电场无人机与无人船协同任务分配和路径规划研究。

## 项目结构

```
my_paper - 2/
├── config.yaml              # 主配置文件
├── content.png              # 2048x2048 卫星影像
├── content.xml              # PASCAL VOC 标注文件
├── requirements.txt         # Python依赖
├── run_pipeline.py          # 主执行脚本
├── src/                     # 源代码
│   ├── data/               # 数据处理
│   │   ├── parser.py       # XML解析器
│   │   ├── layers.py       # 层生成器
│   │   └── grid.py         # 栅格生成器
│   ├── navigation/         # 路径规划
│   │   └── pathfinder.py   # 路径缓存系统
│   ├── env/               # 仿真环境
│   │   ├── agent_state.py  # 智能体状态机
│   │   ├── events.py       # 事件系统
│   │   ├── energy.py       # 能量模型
│   │   └── charging.py     # 充电系统
│   ├── baselines/         # 基线算法
│   └── viz/               # 可视化
├── scripts/               # 执行脚本
│   ├── build_layers.py    # 构建层数据
│   ├── build_grids.py     # 构建栅格
│   ├── build_cache.py     # 构建路径缓存
│   └── validate_env.py    # 环境验证
└── tests/                 # 测试套件
```

## 快速开始

### 1. 环境配置

```bash
# 创建conda虚拟环境
conda create -n wind_farm python=3.10
conda activate wind_farm

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下文件存在：
- `content.png`: 2048×2048 海上风电场卫星影像
- `content.xml`: PASCAL VOC格式标注文件，包含：
  - 39个风机 (`fan`)
  - 2个充电站 (`charge station`)
  - 6个岛屿 (`island`)
  - 134个UAV禁飞区 (`UAV obstacle`)

### 3. 运行完整流水线

```bash
# 运行完整数据处理流水线
python run_pipeline.py --all

# 或分步执行
python run_pipeline.py --step layers    # 构建地图层
python run_pipeline.py --step grids     # 构建导航栅格
python run_pipeline.py --step cache     # 构建路径缓存
python run_pipeline.py --step validate  # 验证环境
```

### 4. 验证结果

运行成功后会生成：
- `layers.pkl`: 地图层数据
- `layers_preview.png`: 地图层可视化
- `grids/`: UAV和USV导航栅格
- `grids_visualization.png`: 栅格可视化
- `cache/`: 路径缓存文件

## 核心功能

### 数据处理流水线

1. **XML解析** (`src/data/parser.py`)
   - 解析PASCAL VOC标注文件
   - 像素坐标→世界坐标转换
   - 对象分类和验证

2. **层生成** (`src/data/layers.py`)
   - 任务点层（风机位置）
   - 充电站层
   - 障碍物层（岛屿、禁飞区）
   - 安全缓冲区生成

3. **栅格化** (`src/data/grid.py`)
   - UAV/USV差异化可通行域
   - 占据栅格生成
   - 导航图构建

4. **路径缓存** (`src/navigation/pathfinder.py`)
   - A*最短路径计算
   - 全对距离矩阵
   - 压缩缓存存储

### 仿真环境

1. **智能体状态机** (`src/env/agent_state.py`)
   - 6种工作模式：idle/transit/service/queue/charge/return
   - 能量管理和安全约束
   - 状态转换验证

2. **事件驱动系统** (`src/env/events.py`)
   - 半离散时间仿真
   - 优先队列事件调度
   - 多类型事件处理器

3. **能量模型** (`src/env/energy.py`)
   - 线性/二次能耗模型
   - 环境因素修正
   - 能量优化策略

4. **充电系统** (`src/env/charging.py`)
   - 固定/移动充电站
   - FCFS/优先级队列
   - 充电性能统计

## 配置说明

主要配置参数在 `config.yaml` 中：

```yaml
map:
  gsd_m_per_px: 5.0          # 地面分辨率（米/像素）
  grid_res_m: 10.0           # 栅格分辨率（米/格）
  buffer:
    rho_fan: 80.0            # 风机安全缓冲（米）
    rho_uav: 40.0            # UAV障碍膨胀（米）
    rho_island: 10.0         # 岛屿缓冲（米）

agents:
  uav:
    n: 5                     # UAV数量
    v: 15.0                  # 速度（米/秒）
    Ecap: 160                # 电池容量
    alpha: 0.20              # 能耗系数（单位/米）
  usv:
    n: 1                     # USV数量
    v: 5.0                   # 速度（米/秒）
    Ecap: 1200               # 电池容量
    alpha: 0.05              # 能耗系数

reward:
  lambda_T: 1.0              # 总时间权重
  lambda_sigma: 0.2          # 负载均衡权重
  lambda_E: 0.01             # 能量权重
  lambda_Q: 0.05             # 排队权重
```

## 验证和测试

```bash
# 运行环境验证
python scripts/validate_env.py --test all

# 分类测试
python scripts/validate_env.py --test xml        # XML解析
python scripts/validate_env.py --test agent      # 智能体状态机
python scripts/validate_env.py --test charging   # 充电系统
python scripts/validate_env.py --test event      # 事件系统
python scripts/validate_env.py --test integration # 系统集成
```

## 性能优化

- **路径缓存**: 使用zstd压缩，支持O(1)距离查询
- **栅格化**: 基于rasterio的高效几何体栅格化
- **事件系统**: 优先队列确保时间复杂度为O(log n)
- **并行处理**: 支持多进程路径计算

## 扩展功能

- 支持动态障碍物和禁飞区
- 移动充电站（USV充电）
- 多种能量模型（线性、二次、环境相关）
- 可插拔的充电策略
- 详细的性能分析和可视化

## 故障排除

### 常见问题

1. **XML解析失败**
   - 检查文件格式是否为PASCAL VOC
   - 验证坐标范围是否在图像内
   - 确认类别名称拼写正确

2. **栅格生成失败**
   - 检查内存使用（大栅格需要更多内存）
   - 调整栅格分辨率参数
   - 验证几何体有效性

3. **路径缓存超时**
   - 减少节点数量或使用KNN邻接
   - 启用并行计算
   - 增加搜索超时时间

### 日志分析

所有模块使用标准Python logging，设置环境变量控制日志级别：

```bash
export PYTHONPATH=$(pwd)
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## 引用

如果使用本项目，请引用：

```bibtex
@misc{wind_farm_marl_2024,
  title={Multi-Agent Reinforcement Learning for UAV-USV Cooperative Task Allocation in Offshore Wind Farms},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 许可证

MIT License

## 联系方式

- 项目维护者：Research Team
- 邮箱：your-email@domain.com
- 问题反馈：[GitHub Issues](https://github.com/your-repo/issues)