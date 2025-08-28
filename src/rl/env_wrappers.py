"""
环境包装器
Environment Wrappers

为WindFarmParallelEnv添加强化学习训练所需的功能：
- 观测归一化
- 动作掩码适配
- 全局状态提供
"""

import numpy as np
import pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from ..env.wind_farm_env import WindFarmParallelEnv
from ..env.agent_state import AgentMode

logger = logging.getLogger(__name__)


class ObsNormWrapper:
    """
    观测归一化包装器
    使用Welford增量算法维护运行统计量，支持训练/评估模式切换
    """
    
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        
        # 运行统计量
        self.running_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.running_var = np.ones(self.obs_dim, dtype=np.float64)
        self.count = 0
        
        # 配置参数
        self.epsilon = 1e-8
        self.clip_obs = 10.0
        self.training = True
        
        logger.info(f"ObsNormWrapper初始化: 观测维度={self.obs_dim}")
    
    def _update_stats(self, obs: np.ndarray):
        """使用Welford算法更新运行统计量"""
        if not self.training:
            return
            
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += delta * delta2
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        if self.count == 0:
            return obs
            
        # 计算标准差
        var = self.running_var / max(1, self.count - 1)
        std = np.sqrt(var + self.epsilon)
        
        # 归一化并裁剪
        norm_obs = (obs - self.running_mean) / std
        norm_obs = np.clip(norm_obs, -self.clip_obs, self.clip_obs)
        
        return norm_obs.astype(np.float32)
    
    def reset(self, **kwargs):
        """重置环境并归一化初始观测"""
        obs, infos = self.env.reset(**kwargs)
        
        # 归一化每个智能体的观测
        norm_obs = {}
        for agent_id, agent_obs in obs.items():
            if self.training:
                self._update_stats(agent_obs)
            norm_obs[agent_id] = self._normalize(agent_obs)
            
        return norm_obs, infos
    
    def step(self, actions):
        """环境步进并归一化观测"""
        obs, rewards, dones, truncated, infos = self.env.step(actions)
        
        # 归一化每个智能体的观测
        norm_obs = {}
        for agent_id, agent_obs in obs.items():
            if self.training:
                self._update_stats(agent_obs)
            norm_obs[agent_id] = self._normalize(agent_obs)
            
        return norm_obs, rewards, dones, truncated, infos
    
    def set_training(self, training: bool):
        """设置训练/评估模式"""
        self.training = training
        
    def get_stats(self) -> Dict[str, np.ndarray]:
        """获取当前统计量"""
        return {
            'mean': self.running_mean.copy(),
            'var': self.running_var.copy() / max(1, self.count - 1),
            'count': self.count
        }
    
    def set_stats(self, stats: Dict[str, Any]):
        """设置统计量（用于加载已保存的统计量）"""
        self.running_mean = stats['mean'].copy()
        self.running_var = stats['var'].copy() * max(1, stats['count'] - 1)
        self.count = stats['count']
    
    def save_stats(self, path: str):
        """保存统计量到文件"""
        stats = self.get_stats()
        with open(path, 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f"观测归一化统计量已保存到: {path}")
    
    def load_stats(self, path: str):
        """从文件加载统计量"""
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        self.set_stats(stats)
        logger.info(f"观测归一化统计量已加载: {path}")
    
    def __getattr__(self, name):
        """透传其他属性到原环境"""
        return getattr(self.env, name)


class ActionMaskingAdapter:
    """
    动作掩码适配器
    将WindFarmEnv提供的action_mask重组为RLlib兼容的观测格式
    """
    
    def __init__(self, env):
        self.env = env
        logger.info("ActionMaskingAdapter初始化完成")
    
    def reset(self, **kwargs):
        """重置环境并重组观测格式"""
        obs, infos = self.env.reset(**kwargs)
        
        # 重组为{"obs": ..., "action_mask": ...}格式
        adapted_obs = {}
        for agent_id in obs:
            adapted_obs[agent_id] = {
                "obs": obs[agent_id],
                "action_mask": infos[agent_id]["action_mask"]
            }
            
        return adapted_obs, infos
    
    def step(self, actions):
        """环境步进并重组观测格式"""
        obs, rewards, dones, truncated, infos = self.env.step(actions)
        
        # 重组观测格式
        adapted_obs = {}
        for agent_id in obs:
            adapted_obs[agent_id] = {
                "obs": obs[agent_id],
                "action_mask": infos[agent_id]["action_mask"]
            }
            
        return adapted_obs, rewards, dones, truncated, infos
    
    def __getattr__(self, name):
        """透传其他属性到原环境"""
        return getattr(self.env, name)


class GlobalStateProvider:
    """
    全局状态提供器
    为WindFarmEnv添加全局状态，用于集中式价值函数（MAPPO/QMIX）
    """
    
    def __init__(self, env):
        self.env = env
        self.global_state_dim = None  # 运行时确定
        logger.info("GlobalStateProvider初始化完成")
    
    def _build_global_state(self) -> np.ndarray:
        """构建全局状态向量"""
        state_components = []
        
        # 1. 时间进度 (1维)
        time_progress = self.env.simulation_time / self.env.config['sim']['horizon_s']
        state_components.append(time_progress)
        
        # 2. 任务状态汇总 (前10个任务，每个4维)
        task_states = []
        task_items = list(self.env.tasks.items())
        for i in range(min(10, len(task_items))):
            task_id, task = task_items[i]
            task_states.extend([
                task['position'][0] / 10000.0,  # 归一化X坐标
                task['position'][1] / 10000.0,  # 归一化Y坐标
                1.0 if task['completed'] else 0.0,  # 完成状态
                1.0 if task['assigned_to'] is not None else 0.0  # 分配状态
            ])
        
        # 填充到固定长度
        while len(task_states) < 40:  # 10任务 * 4维
            task_states.extend([0.0, 0.0, 0.0, 0.0])
        state_components.extend(task_states[:40])
        
        # 3. 充电站状态 (前2个充电站，每个3维)
        charge_states = []
        for i in range(min(2, len(self.env.charging_stations))):
            station = self.env.charging_stations[i]
            # 获取队列长度（简化处理，后续可从charging_manager获取）
            queue_len = 0
            charge_states.extend([
                station.position[0] / 10000.0,  # 归一化X坐标
                station.position[1] / 10000.0,  # 归一化Y坐标
                queue_len / 10.0  # 归一化队列长度
            ])
        
        # 填充到固定长度
        while len(charge_states) < 6:  # 2站 * 3维
            charge_states.extend([0.0, 0.0, 0.0])
        state_components.extend(charge_states[:6])
        
        # 4. 所有智能体状态 (每个5维)
        agent_states = []
        for agent_id, agent in self.env._agents.items():
            agent_states.extend([
                agent.state.position[0] / 10000.0,  # 归一化X坐标
                agent.state.position[1] / 10000.0,  # 归一化Y坐标
                agent.state.energy / agent.config.energy_capacity,  # 归一化能量
                1.0 if agent.state.mode == AgentMode.IDLE else 0.0,  # 空闲状态
                1.0 if agent.state.mode == AgentMode.SERVICE else 0.0  # 服务状态
            ])
        
        # 拼接全局状态
        global_state = np.array(state_components, dtype=np.float32)
        
        # 记录维度（首次运行时）
        if self.global_state_dim is None:
            self.global_state_dim = len(global_state)
            logger.info(f"全局状态维度确定: {self.global_state_dim}")
        
        return global_state
    
    def state(self) -> np.ndarray:
        """获取全局状态（PettingZoo标准方法）"""
        return self._build_global_state()
    
    def reset(self, **kwargs):
        """重置环境并在infos中添加全局状态"""
        obs, infos = self.env.reset(**kwargs)
        
        # 构建全局状态并添加到所有智能体的infos中
        global_state = self._build_global_state()
        for agent_id in infos:
            infos[agent_id]["global_state"] = global_state
            
        return obs, infos
    
    def step(self, actions):
        """环境步进并在infos中添加全局状态"""
        obs, rewards, dones, truncated, infos = self.env.step(actions)
        
        # 构建全局状态并添加到所有智能体的infos中
        global_state = self._build_global_state()
        for agent_id in infos:
            infos[agent_id]["global_state"] = global_state
            
        return obs, rewards, dones, truncated, infos
    
    def __getattr__(self, name):
        """透传其他属性到原环境"""
        return getattr(self.env, name)


def create_wrapped_env(config: Dict[str, Any]) -> GlobalStateProvider:
    """
    创建完整包装的环境
    
    Args:
        config: 配置字典，应包含env_config等配置项
        
    Returns:
        包装后的环境实例
    """
    # 1. 创建基础环境
    env_config = config.get("env_config", {})
    base_env = WindFarmParallelEnv(
        config_path=env_config.get("config_path", "config.yaml"),
        layers_path=env_config.get("layers_path", "layers.pkl"),
        cache_dir=env_config.get("cache_dir", "cache")
    )
    
    # 2. 添加全局状态支持
    env = GlobalStateProvider(base_env)
    
    # 3. 观测归一化（可选）
    if config.get("use_obs_norm", True):
        env = ObsNormWrapper(env)
        logger.info("已启用观测归一化")
    
    # 4. 动作掩码适配（可选）
    if config.get("use_action_masking", True):
        env = ActionMaskingAdapter(env)
        logger.info("已启用动作掩码适配")
    
    logger.info(f"环境包装完成，最终类型: {type(env).__name__}")
    return env