#!/usr/bin/env python3
"""
构建地图层脚本
Build layers script: Convert XML data to geometric layers
"""

import sys
import os
import argparse
import yaml
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.parser import VOCParser
from src.data.layers import LayerGenerator, LayerConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建地图层数据')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--xml', default='content.xml', help='XML标注文件路径')
    parser.add_argument('--output', default='layers.pkl', help='输出文件路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图片')
    
    args = parser.parse_args()
    
    try:
        # 检查输入文件
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return 1
        
        if not os.path.exists(args.xml):
            logger.error(f"XML文件不存在: {args.xml}")
            return 1
        
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建层配置
        layer_config = LayerConfig.from_config(config)
        gsd = config['map']['gsd_m_per_px']
        
        # 解析XML
        logger.info(f"解析XML文件: {args.xml}")
        xml_parser = VOCParser(args.xml, gsd)
        objects = xml_parser.parse()
        
        # 生成层
        logger.info("生成地图层...")
        layer_generator = LayerGenerator(layer_config)
        layers_data = layer_generator.generate_all_layers(objects)
        
        # 保存结果
        logger.info(f"保存层数据到: {args.output}")
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump(layers_data, f)
        
        # 生成可视化
        if args.visualize:
            viz_path = args.output.replace('.pkl', '_preview.png')
            logger.info(f"生成可视化: {viz_path}")
            layer_generator.visualize_layers(viz_path)
        
        # 输出统计信息
        logger.info("=== 层构建完成 ===")
        logger.info(f"任务点数量: {len(layers_data['task_points'])}")
        logger.info(f"充电站数量: {len(layers_data['charging_stations'])}")
        logger.info(f"障碍物层数量: {len(layers_data['obstacle_layers'])}")
        
        for name, layer in layers_data['obstacle_layers'].items():
            logger.info(f"  {name}: {len(layer.polygons)}个多边形")
        
        return 0
        
    except Exception as e:
        logger.error(f"构建层数据失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())