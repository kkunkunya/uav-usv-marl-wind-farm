"""
数据缓冲器
Data Buffer

多智能体强化学习的经验缓冲和回合数据存储
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    回合数据缓冲器
    用于存储MAPPO等on-policy算法的回合经验
    """
    
    def __init__(self, buffer_size: int, obs_dim: int, global_state_dim: int, 
                 action_dim: int, n_agents: int, device: str = "cpu"):
        """
        初始化缓冲器
        
        Args:
            buffer_size: 缓冲器大小（时间步数）
            obs_dim: 观测维度
            global_state_dim: 全局状态维度
            action_dim: 动作维度（离散动作的数量）
            n_agents: 智能体数量
            device: 设备类型
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        
        # 当前存储位置
        self.ptr = 0
        self.size = 0
        
        # 为每个智能体分别存储数据
        self._init_buffers()
        
        logger.info(f"RolloutBuffer初始化: 大小={buffer_size}, "
                   f"观测维度={obs_dim}, 全局状态维度={global_state_dim}, "
                   f"智能体数量={n_agents}")
    
    def _init_buffers(self):
        """初始化存储缓冲区"""
        # 观测和动作
        self.observations = np.zeros(
            (self.buffer_size, self.n_agents, self.obs_dim), 
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=np.int64
        )
        self.action_masks = np.zeros(
            (self.buffer_size, self.n_agents, self.action_dim),
            dtype=bool
        )
        
        # 奖励和完成标志
        self.rewards = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=bool
        )
        
        # 价值和优势
        self.values = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=np.float32
        )
        self.advantages = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=np.float32
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_agents), 
            dtype=np.float32
        )
        
        # 全局状态（用于集中式价值函数）
        self.global_states = np.zeros(
            (self.buffer_size, self.global_state_dim),
            dtype=np.float32
        )
        
        # 动作概率（用于重要性采样）
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_agents),
            dtype=np.float32
        )
        
        # 智能体ID映射
        self.agent_ids = []  # 存储智能体ID列表
    
    def add(self, agent_id_to_idx: Dict[str, int], 
            obs: Dict[str, np.ndarray],
            actions: Dict[str, int],
            rewards: Dict[str, float], 
            dones: Dict[str, bool],
            values: Dict[str, float],
            log_probs: Dict[str, float],
            action_masks: Dict[str, np.ndarray],
            global_state: np.ndarray):
        """
        添加一步经验
        
        Args:
            agent_id_to_idx: 智能体ID到索引的映射
            obs: 观测字典
            actions: 动作字典
            rewards: 奖励字典
            dones: 完成标志字典
            values: 价值估计字典
            log_probs: 动作对数概率字典
            action_masks: 动作掩码字典
            global_state: 全局状态
        """
        if self.ptr >= self.buffer_size:
            logger.warning("缓冲器已满，无法添加更多数据")
            return
        
        # 存储每个智能体的数据
        for agent_id, idx in agent_id_to_idx.items():
            if agent_id in obs:  # 检查智能体是否仍在环境中
                # 提取观测（处理动作掩码适配后的格式）
                if isinstance(obs[agent_id], dict) and "obs" in obs[agent_id]:
                    agent_obs = obs[agent_id]["obs"]
                else:
                    agent_obs = obs[agent_id]
                
                self.observations[self.ptr, idx] = agent_obs
                self.actions[self.ptr, idx] = actions.get(agent_id, 0)
                self.rewards[self.ptr, idx] = rewards.get(agent_id, 0.0)
                self.dones[self.ptr, idx] = dones.get(agent_id, False)
                self.values[self.ptr, idx] = values.get(agent_id, 0.0)
                self.log_probs[self.ptr, idx] = log_probs.get(agent_id, 0.0)
                
                # 处理动作掩码
                if agent_id in action_masks:
                    self.action_masks[self.ptr, idx] = action_masks[agent_id]
        
        # 存储全局状态
        self.global_states[self.ptr] = global_state
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        计算GAE优势估计
        
        Args:
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = 0
        
        # 从后往前计算GAE
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_nonterminal = 1.0 - self.dones[step]
                next_values = np.zeros_like(self.values[step])
            else:
                next_nonterminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = (self.rewards[step] + 
                    gamma * next_values * next_nonterminal - 
                    self.values[step])
            
            advantages[step] = (delta + 
                               gamma * gae_lambda * next_nonterminal * last_gae)
            last_gae = advantages[step]
        
        self.advantages[:self.size] = advantages[:self.size]
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]
        
        logger.debug(f"GAE计算完成: 优势均值={np.mean(self.advantages[:self.size]):.3f}, "
                    f"标准差={np.std(self.advantages[:self.size]):.3f}")
    
    def get_minibatches(self, batch_size: int, shuffle: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        获取小批次数据
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            小批次数据列表
        """
        if self.size == 0:
            return []
        
        # 生成索引
        indices = np.arange(self.size)
        if shuffle:
            np.random.shuffle(indices)
        
        # 分割成小批次
        batches = []
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            
            # 构建批次数据
            batch_data = {
                'observations': torch.FloatTensor(
                    self.observations[batch_indices]
                ).to(self.device),
                'actions': torch.LongTensor(
                    self.actions[batch_indices]
                ).to(self.device),
                'action_masks': torch.BoolTensor(
                    self.action_masks[batch_indices]
                ).to(self.device),
                'old_log_probs': torch.FloatTensor(
                    self.log_probs[batch_indices]
                ).to(self.device),
                'advantages': torch.FloatTensor(
                    self.advantages[batch_indices]
                ).to(self.device),
                'returns': torch.FloatTensor(
                    self.returns[batch_indices]
                ).to(self.device),
                'values': torch.FloatTensor(
                    self.values[batch_indices]
                ).to(self.device),
                'global_states': torch.FloatTensor(
                    self.global_states[batch_indices]
                ).to(self.device)
            }
            
            batches.append(batch_data)
        
        return batches
    
    def clear(self):
        """清空缓冲器"""
        self.ptr = 0
        self.size = 0
        
        # 重置所有数组
        self.observations.fill(0)
        self.actions.fill(0)
        self.action_masks.fill(False)
        self.rewards.fill(0)
        self.dones.fill(False)
        self.values.fill(0)
        self.advantages.fill(0)
        self.returns.fill(0)
        self.global_states.fill(0)
        self.log_probs.fill(0)
        
        logger.debug("缓冲器已清空")
    
    def normalize_advantages(self):
        """标准化优势"""
        if self.size == 0:
            return
        
        advantages = self.advantages[:self.size]
        mean_adv = np.mean(advantages)
        std_adv = np.std(advantages)
        
        if std_adv > 1e-8:
            self.advantages[:self.size] = (advantages - mean_adv) / (std_adv + 1e-8)
        
        logger.debug(f"优势标准化完成: 原始均值={mean_adv:.3f}, 标准差={std_adv:.3f}")
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓冲器统计信息"""
        if self.size == 0:
            return {}
        
        return {
            'buffer_size': self.size,
            'avg_reward': float(np.mean(self.rewards[:self.size])),
            'avg_value': float(np.mean(self.values[:self.size])),
            'avg_advantage': float(np.mean(self.advantages[:self.size])),
            'std_advantage': float(np.std(self.advantages[:self.size])),
            'avg_return': float(np.mean(self.returns[:self.size]))
        }


class ExperienceReplay:
    """
    经验回放缓冲器
    用于QMIX等off-policy算法
    """
    
    def __init__(self, capacity: int, obs_dim: int, global_state_dim: int,
                 action_dim: int, n_agents: int, n_step: int = 1):
        """
        初始化经验回放缓冲器
        
        Args:
            capacity: 缓冲器容量
            obs_dim: 观测维度
            global_state_dim: 全局状态维度  
            action_dim: 动作维度
            n_agents: 智能体数量
            n_step: n步回报
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_step = n_step
        
        self.ptr = 0
        self.size = 0
        
        # 存储缓冲区
        self._init_replay_buffer()
        
        logger.info(f"ExperienceReplay初始化: 容量={capacity}, n步={n_step}")
    
    def _init_replay_buffer(self):
        """初始化回放缓冲区"""
        self.observations = np.zeros(
            (self.capacity, self.n_agents, self.obs_dim),
            dtype=np.float32
        )
        self.next_observations = np.zeros(
            (self.capacity, self.n_agents, self.obs_dim),
            dtype=np.float32
        )
        self.actions = np.zeros(
            (self.capacity, self.n_agents),
            dtype=np.int64
        )
        self.rewards = np.zeros(
            (self.capacity, self.n_agents),
            dtype=np.float32
        )
        self.dones = np.zeros(
            (self.capacity, self.n_agents),
            dtype=bool
        )
        self.global_states = np.zeros(
            (self.capacity, self.global_state_dim),
            dtype=np.float32
        )
        self.next_global_states = np.zeros(
            (self.capacity, self.global_state_dim),
            dtype=np.float32
        )
    
    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
            next_obs: np.ndarray, dones: np.ndarray, 
            global_state: np.ndarray, next_global_state: np.ndarray):
        """添加一个transition"""
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.global_states[self.ptr] = global_state
        self.next_global_states[self.ptr] = next_global_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样一个批次"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'dones': torch.BoolTensor(self.dones[indices]),
            'global_states': torch.FloatTensor(self.global_states[indices]),
            'next_global_states': torch.FloatTensor(self.next_global_states[indices])
        }
    
    def __len__(self):
        return self.size