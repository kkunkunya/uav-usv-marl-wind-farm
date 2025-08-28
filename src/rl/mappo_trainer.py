"""
MAPPO训练器
MAPPO Trainer

实现MAPPO算法的完整训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import time

from .trainer import BaseTrainer
from .buffer import RolloutBuffer
from .models.mappo_actor import MAPPOActorCritic
from .models.mappo_critic import MAPPOCritic

logger = logging.getLogger(__name__)


class MAPPOTrainer(BaseTrainer):
    """
    MAPPO训练器
    
    实现Multi-Agent Proximal Policy Optimization算法
    特点：共享策略参数，集中式价值函数，分散式执行
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MAPPO训练器
        
        Args:
            config: 训练配置字典
        """
        super().__init__(config)
        
        # MAPPO超参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.n_epochs = config.get('n_epochs', 4)
        self.n_minibatches = config.get('n_minibatches', 8)
        
        # 学习率
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.lr_decay = config.get('lr_decay', True)
        
        # 网络配置
        model_config = config.get('model', {})
        self.use_shared_backbone = model_config.get('shared_backbone', False)
        
        # 初始化网络和优化器
        self._init_networks()
        self._init_optimizers()
        
        # 初始化缓冲器
        self._init_buffer()
        
        logger.info("MAPPOTrainer初始化完成")
    
    def _init_networks(self):
        """初始化神经网络"""
        # 获取环境信息
        obs_dim = self.env_runner.observation_space.shape[0]
        action_dim = self.env_runner.action_space.n
        
        # 获取全局状态维度（需要创建一个环境实例来确定）
        temp_env = self.env_runner.envs[0]
        temp_obs, temp_infos = temp_env.reset()
        global_state = temp_infos[list(temp_infos.keys())[0]]["global_state"]
        global_state_dim = len(global_state)
        
        model_config = self.config.get('model', {})
        
        if self.use_shared_backbone:
            # 使用共享backbone的Actor-Critic
            self.policy = MAPPOActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                hidden_sizes=model_config.get('actor_hidden_sizes', [256, 256]),
                critic_hidden_sizes=model_config.get('critic_hidden_sizes', [512, 512]),
                shared_backbone=True,
                activation=model_config.get('activation', 'relu'),
                use_layer_norm=model_config.get('use_layer_norm', False)
            ).to(self.device)
        else:
            # 独立的Actor和Critic
            self.policy = MAPPOActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                hidden_sizes=model_config.get('actor_hidden_sizes', [256, 256]),
                critic_hidden_sizes=model_config.get('critic_hidden_sizes', [512, 512]),
                shared_backbone=False,
                activation=model_config.get('activation', 'relu'),
                use_layer_norm=model_config.get('use_layer_norm', False)
            ).to(self.device)
        
        logger.info(f"网络初始化: 观测维度={obs_dim}, 动作维度={action_dim}, "
                   f"全局状态维度={global_state_dim}")
        
        # 保存维度信息
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
    
    def _init_optimizers(self):
        """初始化优化器"""
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # 学习率调度器
        if self.lr_decay:
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.get('max_iterations', 1000)
            )
    
    def _init_buffer(self):
        """初始化经验缓冲器"""
        buffer_size = self.config.get('rollout_steps', 2048)
        n_envs = self.config.get('n_envs', 1)
        
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size * n_envs,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            action_dim=self.action_dim,
            n_agents=self.env_runner.n_agents,
            device=self.device
        )
    
    def collect_rollouts(self, n_steps: int) -> Dict[str, Any]:
        """
        收集回合经验
        
        Args:
            n_steps: 收集步数
            
        Returns:
            回合统计信息
        """
        self.policy.eval()  # 设置为评估模式
        
        # 重置缓冲器
        self.buffer.clear()
        
        # 重置环境
        observations, infos = self.env_runner.reset()
        
        # 收集统计信息
        episode_rewards = []
        episode_lengths = []
        n_episodes = 0
        
        for step in range(n_steps):
            # 准备批次数据
            batch_obs = []
            batch_global_states = []
            batch_action_masks = []
            
            for env_idx in range(self.env_runner.n_envs):
                env_obs = observations[env_idx]
                env_info = infos[env_idx]
                
                # 提取观测和掩码
                for agent_id in self.env_runner.agent_ids:
                    if isinstance(env_obs[agent_id], dict):
                        # ActionMaskingAdapter的输出格式
                        obs = env_obs[agent_id]["obs"]
                        action_mask = env_obs[agent_id]["action_mask"]
                    else:
                        # 原始观测格式
                        obs = env_obs[agent_id]
                        action_mask = env_info[agent_id]["action_mask"]
                    
                    batch_obs.append(obs)
                    batch_action_masks.append(action_mask)
                
                # 全局状态
                global_state = env_info[list(env_info.keys())[0]]["global_state"]
                batch_global_states.append(global_state)
            
            # 转换为张量
            batch_obs = torch.FloatTensor(np.array(batch_obs)).to(self.device)
            batch_obs = batch_obs.reshape(self.env_runner.n_envs, 
                                        self.env_runner.n_agents, -1)
            
            batch_action_masks = torch.BoolTensor(np.array(batch_action_masks)).to(self.device)
            batch_action_masks = batch_action_masks.reshape(self.env_runner.n_envs,
                                                          self.env_runner.n_agents, -1)
            
            batch_global_states = torch.FloatTensor(np.array(batch_global_states)).to(self.device)
            
            # 获取动作和价值
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action_and_value(
                    batch_obs, batch_global_states, batch_action_masks
                )
            
            # 转换为列表格式供环境使用
            actions_list = []
            for env_idx in range(self.env_runner.n_envs):
                env_actions = {}
                for agent_idx, agent_id in enumerate(self.env_runner.agent_ids):
                    env_actions[agent_id] = int(actions[env_idx, agent_idx])
                actions_list.append(env_actions)
            
            # 环境步进
            next_obs, rewards, dones, truncated, next_infos = self.env_runner.step(actions_list)
            
            # 存储经验到缓冲器
            for env_idx in range(self.env_runner.n_envs):
                env_obs = observations[env_idx]
                env_rewards = rewards[env_idx]
                env_dones = dones[env_idx]
                env_info = infos[env_idx]
                
                # 构造数据字典
                obs_dict = {}
                rewards_dict = {}
                dones_dict = {}
                values_dict = {}
                log_probs_dict = {}
                action_masks_dict = {}
                actions_dict = actions_list[env_idx]
                
                for agent_idx, agent_id in enumerate(self.env_runner.agent_ids):
                    if isinstance(env_obs[agent_id], dict):
                        obs_dict[agent_id] = env_obs[agent_id]["obs"]
                        action_masks_dict[agent_id] = env_obs[agent_id]["action_mask"]
                    else:
                        obs_dict[agent_id] = env_obs[agent_id]
                        action_masks_dict[agent_id] = env_info[agent_id]["action_mask"]
                    
                    rewards_dict[agent_id] = env_rewards[agent_id]
                    dones_dict[agent_id] = env_dones[agent_id]
                    # 集中式价值函数：所有智能体共享同一个全局价值
                    values_dict[agent_id] = float(values[env_idx])
                    log_probs_dict[agent_id] = float(log_probs[env_idx, agent_idx])
                
                global_state = env_info[list(env_info.keys())[0]]["global_state"]
                
                # 添加到缓冲器
                self.buffer.add(
                    agent_id_to_idx=self.env_runner.agent_id_to_idx,
                    obs=obs_dict,
                    actions=actions_dict,
                    rewards=rewards_dict,
                    dones=dones_dict,
                    values=values_dict,
                    log_probs=log_probs_dict,
                    action_masks=action_masks_dict,
                    global_state=global_state
                )
            
            # 更新观测
            observations = next_obs
            infos = next_infos
            
            # 统计完成的回合
            for env_idx in range(self.env_runner.n_envs):
                env_dones = dones[env_idx]
                if any(env_dones.values()):
                    # 计算回合奖励和长度
                    env_rewards = rewards[env_idx]
                    episode_reward = sum(env_rewards.values())
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(step + 1)
                    n_episodes += 1
        
        # 计算GAE优势
        self.buffer.compute_advantages(self.gamma, self.gae_lambda)
        
        # 统计信息
        rollout_stats = {
            'n_steps': n_steps * self.env_runner.n_envs,
            'n_episodes': n_episodes,
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0
        }
        
        # 添加缓冲器统计
        rollout_stats.update(self.buffer.get_stats())
        
        logger.debug(f"回合收集完成: {rollout_stats}")
        
        return rollout_stats
    
    def update_policy(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        使用PPO更新策略
        
        Args:
            rollout_data: 回合数据
            
        Returns:
            训练统计信息
        """
        self.policy.train()  # 设置为训练模式
        
        # 标准化优势
        self.buffer.normalize_advantages()
        
        # 计算批次大小
        total_samples = self.buffer.size * self.env_runner.n_agents
        batch_size = total_samples // self.n_minibatches
        
        # 训练统计
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        
        for epoch in range(self.n_epochs):
            # 获取小批次
            minibatches = self.buffer.get_minibatches(batch_size, shuffle=True)
            
            for batch in minibatches:
                # 提取批次数据
                obs = batch['observations']  # [batch_size, n_agents, obs_dim]
                actions = batch['actions']   # [batch_size, n_agents]
                action_masks = batch['action_masks']  # [batch_size, n_agents, action_dim]
                old_log_probs = batch['old_log_probs']  # [batch_size, n_agents]
                advantages = batch['advantages']  # [batch_size, n_agents]
                returns = batch['returns']  # [batch_size, n_agents]
                global_states = batch['global_states']  # [batch_size, global_state_dim]
                
                # 前向传播
                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs, global_states, actions, action_masks
                )
                
                # PPO损失计算
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失：集中式价值函数预测全局价值，需要对returns求平均
                # values: [batch_size], returns: [batch_size, n_agents]
                global_returns = returns.mean(dim=1)  # [batch_size]
                value_loss = F.mse_loss(values, global_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = (policy_loss + 
                            self.value_coef * value_loss + 
                            self.entropy_coef * entropy_loss)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), 
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # 统计信息
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # KL散度和裁剪比例
                with torch.no_grad():
                    kl_div = (old_log_probs - log_probs).mean().item()
                    kl_divs.append(kl_div)
                    
                    clip_fraction = ((ratio < 1.0 - self.clip_range) | 
                                   (ratio > 1.0 + self.clip_range)).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        # 学习率衰减
        if self.lr_decay:
            self.lr_scheduler.step()
        
        # 返回统计信息
        train_stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_div': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return train_stats
    
    def _evaluate_episodes(self, seed: int, n_episodes: int) -> Dict[str, float]:
        """
        评估多个回合
        
        Args:
            seed: 随机种子
            n_episodes: 评估回合数
            
        Returns:
            评估统计信息
        """
        self.policy.eval()
        
        episode_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'completion_times': [],
            'load_balances': []
        }
        
        # 使用单个环境进行评估
        eval_env = self.env_runner.envs[0]
        
        for episode in range(n_episodes):
            # 设置种子
            np.random.seed(seed + episode)
            
            obs, infos = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # 准备输入
                batch_obs = []
                batch_action_masks = []
                
                for agent_id in self.env_runner.agent_ids:
                    if isinstance(obs[agent_id], dict):
                        agent_obs = obs[agent_id]["obs"]
                        action_mask = obs[agent_id]["action_mask"]
                    else:
                        agent_obs = obs[agent_id]
                        action_mask = infos[agent_id]["action_mask"]
                    
                    batch_obs.append(agent_obs)
                    batch_action_masks.append(action_mask)
                
                # 转换为张量
                batch_obs = torch.FloatTensor(np.array(batch_obs)).unsqueeze(0).to(self.device)
                batch_action_masks = torch.BoolTensor(np.array(batch_action_masks)).unsqueeze(0).to(self.device)
                global_state = torch.FloatTensor(
                    infos[list(infos.keys())[0]]["global_state"]
                ).unsqueeze(0).to(self.device)
                
                # 获取确定性动作
                with torch.no_grad():
                    actions, _, _ = self.policy.get_action_and_value(
                        batch_obs, global_state, batch_action_masks, deterministic=True
                    )
                
                # 转换为字典格式
                actions_dict = {}
                for agent_idx, agent_id in enumerate(self.env_runner.agent_ids):
                    actions_dict[agent_id] = int(actions[0, agent_idx])
                
                # 环境步进
                obs, rewards, dones, truncated, infos = eval_env.step(actions_dict)
                
                # 统计
                episode_reward += sum(rewards.values())
                episode_length += 1
                done = any(dones.values()) or any(truncated.values())
            
            # 记录统计信息
            episode_stats['episode_rewards'].append(episode_reward)
            episode_stats['episode_lengths'].append(episode_length)
            
            # 从环境获取任务完成时间和负载均衡统计
            if hasattr(eval_env, 'get_completion_stats'):
                completion_stats = eval_env.get_completion_stats()
                episode_stats['completion_times'].append(completion_stats.get('Tmax', 0))
                episode_stats['load_balances'].append(completion_stats.get('sigma', 0))
        
        # 计算统计量
        final_stats = {
            'avg_reward': np.mean(episode_stats['episode_rewards']),
            'avg_length': np.mean(episode_stats['episode_lengths']),
            'success_rate': 1.0  # 简化处理，实际需要根据任务完成情况计算
        }
        
        if episode_stats['completion_times']:
            final_stats['Tmax_median'] = np.median(episode_stats['completion_times'])
            final_stats['sigma_median'] = np.median(episode_stats['load_balances'])
        
        return final_stats
    
    def _set_eval_mode(self, eval_mode: bool):
        """设置评估模式"""
        if eval_mode:
            self.policy.eval()
        else:
            self.policy.train()
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态字典"""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }