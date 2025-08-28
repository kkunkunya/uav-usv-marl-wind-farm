#!/usr/bin/env python3
"""
MAPPO训练运行脚本
MAPPO Training Runner Script

使用方法：
python -m src.rl.run_train --config configs/rl/mappo_small.yaml --exp mappo_small
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import yaml
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rl.mappo_trainer import MAPPOTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件，支持继承
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理配置继承
    if '__base__' in config:
        base_config_path = config_file.parent / config['__base__']
        base_config = load_config(str(base_config_path))
        
        # 递归合并配置
        merged_config = merge_configs(base_config, config)
        del merged_config['__base__']
        return merged_config
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置完整性
    
    Args:
        config: 配置字典
    """
    required_keys = [
        'training.algorithm',
        'training.max_iterations',
        'env_config.config_path',
        'mappo.gamma',
        'model.actor_hidden_sizes'
    ]
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if key not in current:
                raise ValueError(f"配置缺少必需项: {key_path}")
            current = current[key]
    
    logger.info("配置验证通过")


def setup_experiment_dir(config: Dict[str, Any], exp_name: str) -> Path:
    """
    设置实验目录
    
    Args:
        config: 配置字典
        exp_name: 实验名称
        
    Returns:
        实验目录路径
    """
    base_dir = Path(config.get('output', {}).get('output_dir', 'experiments/rl'))
    exp_dir = base_dir / exp_name
    
    # 创建目录
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'eval').mkdir(exist_ok=True)
    
    logger.info(f"实验目录创建: {exp_dir}")
    return exp_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAPPO训练脚本')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--exp', type=str, required=True,
                       help='实验名称')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅执行评估')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    try:
        # 加载配置
        logger.info(f"加载配置: {args.config}")
        config = load_config(args.config)
        
        # 验证配置
        validate_config(config)
        
        # 设置实验目录
        exp_dir = setup_experiment_dir(config, args.exp)
        config['output']['output_dir'] = str(exp_dir)
        
        # 设置随机种子
        if args.seed is not None:
            import numpy as np
            import torch
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
            logger.info(f"随机种子设置: {args.seed}")
        
        # 创建训练器
        logger.info("初始化MAPPO训练器")
        trainer = MAPPOTrainer(config)
        
        if args.eval_only:
            # 仅评估模式
            if args.resume:
                logger.info(f"加载模型: {args.resume}")
                checkpoint = trainer.checkpoint_manager.load_checkpoint(args.resume)
                trainer.policy.load_state_dict(checkpoint['state_dict']['policy_state_dict'])
            
            logger.info("开始评估")
            eval_stats = trainer.evaluate()
            
            # 输出评估结果
            logger.info("评估结果:")
            for key, value in eval_stats.items():
                logger.info(f"  {key}: {value}")
                
        else:
            # 训练模式
            if args.resume:
                logger.info(f"恢复训练: {args.resume}")
                checkpoint = trainer.checkpoint_manager.load_checkpoint(args.resume)
                trainer.policy.load_state_dict(checkpoint['state_dict']['policy_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['state_dict']['optimizer_state_dict'])
            
            # 开始训练
            max_iterations = config['training']['max_iterations']
            logger.info(f"开始训练: 最大迭代={max_iterations}")
            
            trainer.train(max_iterations)
            
            logger.info("训练完成")
            
            # 输出最终结果
            if trainer.checkpoint_manager.best_metrics:
                logger.info("最优模型指标:")
                for key, value in trainer.checkpoint_manager.best_metrics.items():
                    logger.info(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()