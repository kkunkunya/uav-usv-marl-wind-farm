# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent reinforcement learning (MARL) research project focused on task allocation and path planning for heterogeneous UAV-USV systems in offshore wind farm environments. The project implements a simulation platform that supports cooperative task execution between Unmanned Aerial Vehicles (UAVs) and Unmanned Surface Vehicles (USVs) with charging stations.

### Core Research Objectives
- Minimize total task completion time (Makespan, Tmax)
- Achieve load balancing across agents (σ)
- Compare MARL approaches against baseline algorithms (MILP, GA-SA, heuristics)
- Support ablation studies on load balancing and energy configurations

## Architecture Overview

The project follows a layered architecture centered around PettingZoo-compatible multi-agent environments:

### 1. Data Layer (`content.png` + `content.xml`)
- **Input**: 2048×2048 annotated imagery with PASCAL VOC format
- **Object Classes**: 
  - `fan`: Wind turbines (task locations)
  - `charge station`: Fixed charging points
  - `island`: Hard obstacles (impassable for both UAV/USV)
  - `UAV obstacle`: Soft obstacles (UAV no-fly zones only)

### 2. Environment Processing Pipeline
```
XML Parser → Coordinate Transform → Buffer Generation → Grid Occupancy → Navigation Cache
```

### 3. Multi-Agent Simulation Core
- **Agents**: UAVs (aerial) + USVs (surface) with distinct motion/energy models
- **State Machine**: transit/service/queue/charge/return/idle modes
- **Event-Driven**: Semi-discrete time progression with event queue
- **Constraints**: Energy safety margins, collision avoidance, charging queues

### 4. Algorithm Framework
- **MARL**: Ray RLlib integration (MAPPO, QMIX)
- **Baselines**: MILP solvers (PuLP), genetic algorithms, heuristics
- **Evaluation**: Multi-objective optimization with statistical validation

## Development Commands

Since this is a research codebase that will be implemented, the following command structure is planned:

### Environment Setup
```bash
# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Build core data layers
python scripts/build_layers.py --config config.yaml
python scripts/build_grids.py --input layers.pkl
python scripts/build_cache.py --input grids/

# Validate environment
python scripts/validate_env.py --run-tests
```

### Running Experiments
```bash
# Single simulation run
python run_simulation.py --config config.yaml --algorithm MAPPO --seed 42

# Batch experiments
python run_experiments.py --matrix experiments/matrix.yaml --output results/

# Baseline comparisons
python run_baselines.py --algorithms MILP,GA-SA,greedy --config config.yaml
```

### Testing and Validation
```bash
# Run environment tests
pytest tests/ -v

# Validate data integrity
python scripts/validate_data.py --check-xml --check-grids

# Performance benchmarks
python benchmarks/run_perf_tests.py
```

### Analysis and Visualization
```bash
# Generate result plots
python analysis/make_plots.py --input results/ --output figures/

# Statistical analysis
python analysis/stats_analysis.py --compare-algorithms --significance-test

# Generate report
python analysis/generate_report.py --template paper --output report.md
```

## Key Configuration Files

### config.yaml Structure
```yaml
map:
  image: content.png
  xml: content.xml
  gsd_m_per_px: 5.0        # Ground sampling distance
  grid_res_m: 10.0         # Grid resolution
  buffer:
    rho_fan: 80.0          # Wind turbine safety buffer
    rho_uav: 40.0          # UAV obstacle expansion
    rho_island: 10.0       # Island buffer

agents:
  uav:
    n: 5                   # Number of UAVs
    v: 15.0               # Velocity (m/s)
    Ecap: 160             # Energy capacity
    alpha: 0.20           # Energy consumption rate
  usv:
    n: 1                   # Number of USVs
    v: 5.0                # Velocity (m/s)
    Ecap: 1200            # Energy capacity
    alpha: 0.05           # Energy consumption rate

reward:
  lambda_T: 1.0           # Makespan weight
  lambda_sigma: 0.2       # Load balance weight
  lambda_E: 0.01          # Energy weight
  lambda_Q: 0.05          # Queue waiting weight
```

## Programming Notes

Unless necessary, there is no need to create a new program; modify the existing one

Test program deletes itself after completion of testing

The program has Chinese comments

When making large-scale modifications to code, be cautious (read through the project code thoroughly to understand the situation before making changes)

Your programming skills are much stronger than mine

If you have problems or questions while programming, communicate with me promptly for corrections

Please proceed with the programming work based on the core plan and mathematical derivations in the Plan folder

Specify Microsoft YaHei as the font for generating charts to ensure Chinese characters are rendered properly in the charts

## Environment Configuration

Run the test using python under this path: C:\Users\sxk27\anaconda3\envs\marl_uav_usv\python.exe



