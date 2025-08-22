#!/usr/bin/env python3
"""
构建导航栅格脚本
Build grids script: Convert geometric layers to navigation grids
"""

import sys
import os
import argparse
import yaml
import logging
import pickle
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.grid import GridGenerator, GridConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建导航栅格')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--layers', default='layers.pkl', help='层数据文件路径')
    parser.add_argument('--output', default='grids', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图片')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return 1
        
        if not os.path.exists(args.layers):
            logger.error(f"层数据文件不存在: {args.layers}")
            return 1
        
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载层数据
        logger.info(f"加载层数据: {args.layers}")
        with open(args.layers, 'rb') as f:
            layers_data = pickle.load(f)
        
        # 创建栅格配置
        grid_config = GridConfig.from_config(config)
        
        # 生成栅格
        logger.info("生成导航栅格...")
        grid_generator = GridGenerator(grid_config)
        grids = grid_generator.generate_all_grids(layers_data)
        
        # 保存结果
        logger.info(f"保存栅格数据到: {args.output}")
        grid_generator.save_grids(grids, args.output)
        
        # 生成可视化
        if args.visualize:
            logger.info("生成栅格可视化...")
            grid_generator.visualize_grids(grids, args.output)
        
        # 输出统计信息
        logger.info("=== 栅格构建完成 ===")
        for agent_type, grid in grids.items():
            logger.info(f"{agent_type.upper()}栅格:")
            logger.info(f"  尺寸: {grid.width} x {grid.height}")
            logger.info(f"  分辨率: {grid.resolution}m/格")
            logger.info(f"  可通行率: {grid.passable_ratio:.2%}")
            if grid.graph:
                logger.info(f"  图节点数: {grid.graph.number_of_nodes()}")
                logger.info(f"  图边数: {grid.graph.number_of_edges()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"构建栅格失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())