#!/usr/bin/env python3
"""
构建路径缓存脚本
Build cache script: Generate all-pairs shortest path cache
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

from src.navigation.pathfinder import generate_path_cache

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建路径缓存')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--layers', default='layers.pkl', help='层数据文件路径')
    parser.add_argument('--grids', default='grids', help='栅格数据目录')
    parser.add_argument('--output', default='cache', help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return 1
        
        if not os.path.exists(args.layers):
            logger.error(f"层数据文件不存在: {args.layers}")
            return 1
        
        if not os.path.exists(args.grids):
            logger.error(f"栅格数据目录不存在: {args.grids}")
            return 1
        
        # 生成路径缓存
        logger.info("开始生成路径缓存...")
        caches = generate_path_cache(args.layers, args.grids, args.config, args.output)
        
        # 输出统计信息
        logger.info("=== 路径缓存构建完成 ===")
        for agent_type, cache in caches.items():
            logger.info(f"{agent_type.upper()}缓存:")
            logger.info(f"  节点数: {len(cache.nodes)}")
            logger.info(f"  路径数: {len(cache.path_cache)}")
            logger.info(f"  缓存命中率: {cache.cache_hit_rate:.2%}")
            
            # 可达性统计
            reachable = sum(1 for r in cache.path_cache.values() if r.is_reachable)
            total = len(cache.path_cache)
            logger.info(f"  路径可达率: {reachable}/{total} ({reachable/total:.1%})")
        
        return 0
        
    except Exception as e:
        logger.error(f"构建路径缓存失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())