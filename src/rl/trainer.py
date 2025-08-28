"""
基础训练器
Base Trainer

提供统一的训练接口和组件管理
"""

import os
import json
import time
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from .buffer import RolloutBuffer
from .env_wrappers import create_wrapped_env

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    检查点管理器
    负责模型保存、加载和最优模型选择
    """
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            max_checkpoints: 最大保存检查点数量
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # 最优模型信息
        self.best_metrics = None
        self.best_checkpoint_path = None
        
        # 保存的检查点列表
        self.checkpoints = []
        
        logger.info(f"CheckpointManager初始化: {save_dir}")
    
    def save_checkpoint(self, iteration: int, state_dict: Dict[str, Any], 
                       metrics: Dict[str, float], 
                       is_best: bool = False) -> str:
        """
        保存检查点
        
        Args:
            iteration: 迭代次数
            state_dict: 模型状态字典
            metrics: 评估指标
            is_best: 是否为最优模型
            
        Returns:
            检查点文件路径
        """
        checkpoint_data = {
            'iteration': iteration,
            'state_dict': state_dict,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        # 生成文件名
        checkpoint_name = f"checkpoint_iter_{iteration:06d}.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoints.append({
            'iteration': iteration,
            'path': checkpoint_path,
            'metrics': metrics
        })
        
        # 保存最优模型
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            self.best_checkpoint_path = str(best_path)
            self.best_metrics = metrics.copy()
            
            # 保存最优模型信息
            best_info = {
                'iteration': iteration,
                'metrics': metrics,
                'checkpoint_path': str(checkpoint_path),
                'timestamp': time.time()
            }
            with open(self.save_dir / "best_info.json", 'w') as f:
                json.dump(best_info, f, indent=2)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        logger.info(f"检查点已保存: {checkpoint_path}" + 
                   (" (最优)" if is_best else ""))
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"检查点已加载: {checkpoint_path}")
        return checkpoint_data
    
    def is_better_model(self, new_metrics: Dict[str, float]) -> bool:
        """
        判断是否为更好的模型（词典序比较）
        
        Args:
            new_metrics: 新的评估指标
            
        Returns:
            是否更优
        """
        if self.best_metrics is None:
            return True
        
        # 词典序比较：先比较Tmax，再比较sigma
        tmax_key = 'Tmax_median'
        sigma_key = 'sigma_median'
        
        if tmax_key not in new_metrics or sigma_key not in new_metrics:
            logger.warning(f"评估指标缺少必要字段: {list(new_metrics.keys())}")
            return False
        
        tolerance = 1e-3  # 容忍误差
        
        # 1. 优先最小化Tmax
        if new_metrics[tmax_key] < self.best_metrics[tmax_key] - tolerance:
            return True
        
        # 2. Tmax相近时，最小化sigma
        if abs(new_metrics[tmax_key] - self.best_metrics[tmax_key]) < tolerance:
            return new_metrics[sigma_key] < self.best_metrics[sigma_key] - tolerance
        
        return False
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # 按迭代次数排序
        self.checkpoints.sort(key=lambda x: x['iteration'])
        
        # 删除最旧的检查点
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint['path'].exists():
                old_checkpoint['path'].unlink()
                logger.debug(f"已删除旧检查点: {old_checkpoint['path']}")


class ParallelEnvRunner:
    """
    并行环境运行器
    管理多个环境实例的并行执行
    """
    
    def __init__(self, env_config: Dict[str, Any], n_envs: int = 1):
        """
        初始化并行环境运行器
        
        Args:
            env_config: 环境配置
            n_envs: 并行环境数量
        """
        self.env_config = env_config
        self.n_envs = n_envs
        
        # 创建环境实例
        self.envs = []
        for i in range(n_envs):
            env = create_wrapped_env(env_config)
            self.envs.append(env)
        
        # 获取环境信息
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.agent_ids = self.envs[0].agents  # agents属性已经是ID列表
        self.n_agents = len(self.agent_ids)
        
        # 创建智能体ID到索引的映射
        self.agent_id_to_idx = {
            agent_id: idx for idx, agent_id in enumerate(self.agent_ids)
        }
        
        logger.info(f"并行环境运行器初始化: {n_envs}个环境, {self.n_agents}个智能体")
    
    def reset(self) -> Tuple[List[Dict], List[Dict]]:
        """重置所有环境"""
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        
        return observations, infos
    
    def step(self, actions_list: List[Dict]) -> Tuple[List[Dict], List[Dict], 
                                                    List[Dict], List[Dict], List[Dict]]:
        """
        所有环境同步步进
        
        Args:
            actions_list: 每个环境的动作字典列表
            
        Returns:
            观测、奖励、完成标志、截断标志、信息的列表
        """
        observations = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        
        for i, env in enumerate(self.envs):
            if i < len(actions_list):
                obs, reward, done, trunc, info = env.step(actions_list[i])
            else:
                # 如果动作不足，使用空动作
                empty_actions = {agent_id: 0 for agent_id in self.agent_ids}
                obs, reward, done, trunc, info = env.step(empty_actions)
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncated.append(trunc)
            infos.append(info)
        
        return observations, rewards, dones, truncated, infos


class BaseTrainer(ABC):
    """
    基础训练器抽象类
    定义训练器的统一接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 训练状态
        self.iteration = 0
        self.total_steps = 0
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'experiments/default'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            str(self.output_dir / 'checkpoints'),
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
        # 创建并行环境运行器
        self.env_runner = ParallelEnvRunner(
            env_config=config.get('env_config', {}),
            n_envs=config.get('n_envs', 1)
        )
        
        # 初始化缓冲器（由子类实现具体参数）
        self.buffer = None
        
        # 评估配置
        self.eval_freq = config.get('eval_freq', 10)
        self.eval_episodes = config.get('eval_episodes', 10)
        self.eval_seeds = config.get('eval_seeds', [42, 123, 456])
        
        # 早停配置
        self.patience = config.get('patience', 10)
        self.no_improve_count = 0
        
        # 保存配置信息
        self._save_config_and_manifest()
        
        logger.info(f"BaseTrainer初始化完成: 设备={self.device}")
    
    def _save_config_and_manifest(self):
        """保存配置和清单信息"""
        # 保存配置
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存清单信息
        manifest = {
            'timestamp': time.time(),
            'config_path': str(config_path),
            'output_dir': str(self.output_dir),
            'git_hash': self._get_git_hash(),
            'python_version': self._get_python_version()
        }
        
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _get_git_hash(self) -> str:
        """获取Git哈希值"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """获取Python版本"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    @abstractmethod
    def collect_rollouts(self, n_steps: int) -> Dict[str, Any]:
        """收集回合数据（由子类实现）"""
        pass
    
    @abstractmethod
    def update_policy(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """更新策略（由子类实现）"""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """评估当前策略"""
        eval_stats = {
            'Tmax_median': [],
            'sigma_median': [],
            'success_rate': [],
            'avg_reward': []
        }
        
        # 设置评估模式
        self._set_eval_mode(True)
        
        for seed in self.eval_seeds:
            # 使用固定种子评估
            episode_stats = self._evaluate_episodes(seed, self.eval_episodes)
            
            for key in eval_stats:
                if key in episode_stats:
                    eval_stats[key].append(episode_stats[key])
        
        # 计算统计量
        final_stats = {}
        for key, values in eval_stats.items():
            if values:
                final_stats[key] = float(np.median(values))
                final_stats[f"{key}_std"] = float(np.std(values))
        
        # 恢复训练模式
        self._set_eval_mode(False)
        
        logger.info(f"评估完成: Tmax={final_stats.get('Tmax_median', 0):.1f}, "
                   f"σ={final_stats.get('sigma_median', 0):.3f}")
        
        return final_stats
    
    @abstractmethod
    def _evaluate_episodes(self, seed: int, n_episodes: int) -> Dict[str, float]:
        """评估多个回合（由子类实现）"""
        pass
    
    @abstractmethod
    def _set_eval_mode(self, eval_mode: bool):
        """设置评估模式（由子类实现）"""
        pass
    
    def train(self, max_iterations: int):
        """
        主训练循环
        
        Args:
            max_iterations: 最大迭代次数
        """
        logger.info(f"开始训练: 最大迭代次数={max_iterations}")
        
        rollout_steps = self.config.get('rollout_steps', 2048)
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # 收集经验
            start_time = time.time()
            rollout_data = self.collect_rollouts(rollout_steps)
            collect_time = time.time() - start_time
            
            # 更新策略
            start_time = time.time()
            train_stats = self.update_policy(rollout_data)
            update_time = time.time() - start_time
            
            # 更新总步数
            self.total_steps += rollout_data.get('n_steps', rollout_steps)
            
            # 记录训练统计
            train_stats.update({
                'iteration': iteration,
                'total_steps': self.total_steps,
                'collect_time': collect_time,
                'update_time': update_time,
                'fps': rollout_data.get('n_steps', rollout_steps) / collect_time
            })
            
            # 周期性评估
            if iteration % self.eval_freq == 0 or iteration == max_iterations - 1:
                eval_stats = self.evaluate()
                
                # 检查是否为最优模型
                is_best = self.checkpoint_manager.is_better_model(eval_stats)
                
                # 保存检查点
                state_dict = self._get_state_dict()
                self.checkpoint_manager.save_checkpoint(
                    iteration, state_dict, eval_stats, is_best
                )
                
                # 记录评估统计
                train_stats.update(eval_stats)
                
                # 早停检查
                if is_best:
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                
                if self.no_improve_count >= self.patience:
                    logger.info(f"早停触发: {self.patience}次迭代无改善")
                    break
            
            # 输出训练信息
            self._log_training_stats(train_stats)
        
        logger.info("训练完成")
    
    @abstractmethod
    def _get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典（由子类实现）"""
        pass
    
    def _log_training_stats(self, stats: Dict[str, Any]):
        """记录训练统计信息"""
        # 格式化输出关键指标
        log_str = f"Iter {stats.get('iteration', 0):4d} | "
        log_str += f"Steps {stats.get('total_steps', 0):7d} | "
        log_str += f"FPS {stats.get('fps', 0):5.0f} | "
        
        if 'policy_loss' in stats:
            log_str += f"PL {stats['policy_loss']:6.3f} | "
        if 'value_loss' in stats:
            log_str += f"VL {stats['value_loss']:6.3f} | "
        if 'Tmax_median' in stats:
            log_str += f"Tmax {stats['Tmax_median']:5.1f} | "
        if 'sigma_median' in stats:
            log_str += f"σ {stats['sigma_median']:5.3f}"
        
        logger.info(log_str)