"""
强化学习训练框架
Reinforcement Learning Training Framework

支持MAPPO、QMIX算法的多智能体强化学习训练
"""

__version__ = "0.1.0"
__author__ = "MARL UAV-USV Project"

from .env_wrappers import (
    ObsNormWrapper,
    ActionMaskingAdapter,
    GlobalStateProvider,
    create_wrapped_env
)

from .trainer import BaseTrainer
from .buffer import RolloutBuffer, ExperienceReplay
from .mappo_trainer import MAPPOTrainer

__all__ = [
    "ObsNormWrapper",
    "ActionMaskingAdapter", 
    "GlobalStateProvider",
    "create_wrapped_env",
    "BaseTrainer",
    "RolloutBuffer",
    "ExperienceReplay",
    "MAPPOTrainer"
]