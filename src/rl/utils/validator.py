"""
兼容性验证器
Compatibility Validator

验证环境包装器与训练器的兼容性
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env.wind_farm_env import WindFarmParallelEnv
from src.rl.env_wrappers import create_wrapped_env
from src.rl.mappo_trainer import MAPPOTrainer

logger = logging.getLogger(__name__)


class CompatibilityValidator:
    """兼容性验证器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化验证器
        
        Args:
            config_path: 环境配置文件路径
        """
        self.config_path = config_path
        self.validation_results = {}
        
        logger.info(f"CompatibilityValidator初始化: {config_path}")
    
    def validate_environment_wrappers(self) -> bool:
        """验证环境包装器兼容性"""
        logger.info("开始验证环境包装器兼容性")
        
        try:
            # 1. 创建原始环境
            base_env = WindFarmParallelEnv(
                config_path=self.config_path,
                layers_path="layers.pkl", 
                cache_dir="cache"
            )
            
            # 2. 创建包装环境
            env_config = {
                "env_config": {
                    "config_path": self.config_path,
                    "layers_path": "layers.pkl",
                    "cache_dir": "cache"
                },
                "use_obs_norm": True,
                "use_action_masking": True
            }
            wrapped_env = create_wrapped_env(env_config)
            
            # 3. 验证环境属性
            assert hasattr(wrapped_env, 'agents'), "包装环境缺少agents属性"
            assert hasattr(wrapped_env, 'observation_space'), "包装环境缺少observation_space属性"
            assert hasattr(wrapped_env, 'action_space'), "包装环境缺少action_space属性"
            
            # 4. 验证智能体一致性
            base_agents = set(base_env.agents)  # agents属性已经是ID列表
            wrapped_agents = set(wrapped_env.agents)  # agents属性已经是ID列表
            assert base_agents == wrapped_agents, f"智能体不一致: {base_agents} vs {wrapped_agents}"
            
            # 5. 验证reset接口
            base_obs, base_info = base_env.reset()
            wrap_obs, wrap_info = wrapped_env.reset()
            
            # 验证智能体数量一致
            assert set(base_obs.keys()) == set(wrap_obs.keys()), "reset返回的智能体不一致"
            
            # 6. 验证观测格式
            for agent_id in base_obs:
                if isinstance(wrap_obs[agent_id], dict):
                    # ActionMaskingAdapter的输出格式
                    assert "obs" in wrap_obs[agent_id], f"智能体{agent_id}缺少obs字段"
                    assert "action_mask" in wrap_obs[agent_id], f"智能体{agent_id}缺少action_mask字段"
                    
                    # 验证观测维度
                    base_shape = base_obs[agent_id].shape
                    wrap_shape = wrap_obs[agent_id]["obs"].shape
                    assert base_shape == wrap_shape, f"观测维度不匹配: {base_shape} vs {wrap_shape}"
                    
                    # 验证动作掩码维度
                    mask_shape = wrap_obs[agent_id]["action_mask"].shape
                    expected_mask_shape = (base_env.action_space.n,)
                    assert mask_shape == expected_mask_shape, f"动作掩码维度不匹配: {mask_shape} vs {expected_mask_shape}"
            
            # 7. 验证全局状态
            first_agent = list(wrap_info.keys())[0]
            assert "global_state" in wrap_info[first_agent], "缺少全局状态"
            global_state = wrap_info[first_agent]["global_state"]
            assert isinstance(global_state, np.ndarray), "全局状态不是numpy数组"
            assert global_state.ndim == 1, "全局状态应该是一维数组"
            
            # 8. 验证step接口
            actions = {agent_id: 0 for agent_id in base_agents}  # 使用STAY动作
            
            base_step_result = base_env.step(actions)
            wrap_step_result = wrapped_env.step(actions)
            
            assert len(base_step_result) == 5, "基础环境step返回值不是5元组"
            assert len(wrap_step_result) == 5, "包装环境step返回值不是5元组"
            
            # 验证返回值类型
            wrap_obs, wrap_rewards, wrap_dones, wrap_truncated, wrap_infos = wrap_step_result
            
            for agent_id in wrap_obs:
                assert isinstance(wrap_obs[agent_id], dict), f"智能体{agent_id}观测不是字典格式"
                assert "global_state" in wrap_infos[agent_id], f"智能体{agent_id}缺少全局状态"
            
            self.validation_results['environment_wrappers'] = True
            logger.info("✓ 环境包装器兼容性验证通过")
            return True
            
        except Exception as e:
            self.validation_results['environment_wrappers'] = False
            logger.error(f"✗ 环境包装器兼容性验证失败: {e}")
            return False
    
    def validate_data_flow(self) -> bool:
        """验证数据流正确性"""
        logger.info("开始验证数据流正确性")
        
        try:
            # 创建包装环境
            env_config = {
                "env_config": {
                    "config_path": self.config_path,
                    "layers_path": "layers.pkl",
                    "cache_dir": "cache"
                },
                "use_obs_norm": True,
                "use_action_masking": True
            }
            env = create_wrapped_env(env_config)
            
            # 重置环境
            obs, infos = env.reset()
            
            # 验证数据格式一致性
            for step in range(10):
                # 构造随机动作（确保使用可行动作）
                actions = {}
                for agent_id in obs:
                    action_mask = obs[agent_id]["action_mask"]
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        actions[agent_id] = int(np.random.choice(valid_actions))
                    else:
                        actions[agent_id] = env.action_space.n - 1  # STAY动作
                
                # 环境步进
                next_obs, rewards, dones, truncated, next_infos = env.step(actions)
                
                # 验证数据类型
                for agent_id in next_obs:
                    assert isinstance(next_obs[agent_id], dict), "观测不是字典格式"
                    assert isinstance(rewards[agent_id], (int, float)), "奖励不是数值类型"
                    assert isinstance(dones[agent_id], bool), "done不是布尔类型"
                    assert isinstance(next_infos[agent_id], dict), "info不是字典格式"
                    
                    # 验证观测字段
                    assert "obs" in next_obs[agent_id], "缺少obs字段"
                    assert "action_mask" in next_obs[agent_id], "缺少action_mask字段"
                    assert "global_state" in next_infos[agent_id], "缺少global_state字段"
                    
                    # 验证数据维度
                    obs_array = next_obs[agent_id]["obs"]
                    mask_array = next_obs[agent_id]["action_mask"]
                    global_state_array = next_infos[agent_id]["global_state"]
                    
                    assert obs_array.ndim == 1, "观测应该是一维数组"
                    assert mask_array.ndim == 1, "动作掩码应该是一维数组"
                    assert global_state_array.ndim == 1, "全局状态应该是一维数组"
                    
                    # 验证动作掩码有效性
                    assert mask_array.dtype == bool, "动作掩码应该是布尔数组"
                    assert np.any(mask_array), "动作掩码至少应有一个可行动作"
                
                # 更新观测
                obs = next_obs
                infos = next_infos
                
                # 如果回合结束，重置环境
                if any(dones.values()):
                    obs, infos = env.reset()
            
            self.validation_results['data_flow'] = True
            logger.info("✓ 数据流正确性验证通过")
            return True
            
        except Exception as e:
            self.validation_results['data_flow'] = False
            logger.error(f"✗ 数据流正确性验证失败: {e}")
            return False
    
    def validate_training_integration(self) -> bool:
        """验证训练器集成"""
        logger.info("开始验证训练器集成")
        
        try:
            # 创建最小配置
            config = {
                'training': {
                    'algorithm': 'MAPPO',
                    'max_iterations': 2,
                    'rollout_steps': 10,
                    'n_envs': 1,
                    'device': 'cpu',
                    'eval_freq': 1,
                    'eval_episodes': 1,
                    'eval_seeds': [42],
                    'patience': 10
                },
                'mappo': {
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'n_epochs': 1,
                    'n_minibatches': 1,
                    'value_coef': 0.5,
                    'entropy_coef': 0.01,
                    'learning_rate': 1e-4,
                    'lr_decay': False,
                    'max_grad_norm': 0.5
                },
                'model': {
                    'shared_backbone': False,
                    'actor_hidden_sizes': [64, 64],
                    'critic_hidden_sizes': [128, 128],
                    'activation': 'relu',
                    'use_layer_norm': False
                },
                'env_config': {
                    'config_path': self.config_path,
                    'layers_path': 'layers.pkl',
                    'cache_dir': 'cache'
                },
                'output': {
                    'output_dir': 'test_output',
                    'save_models': False
                }
            }
            
            # 创建训练器
            trainer = MAPPOTrainer(config)
            
            # 验证训练器属性
            assert hasattr(trainer, 'policy'), "训练器缺少policy属性"
            assert hasattr(trainer, 'buffer'), "训练器缺少buffer属性"
            assert hasattr(trainer, 'env_runner'), "训练器缺少env_runner属性"
            
            # 验证网络初始化
            assert trainer.policy is not None, "策略网络未初始化"
            
            # 执行一步训练（smoke test）
            rollout_data = trainer.collect_rollouts(5)
            assert isinstance(rollout_data, dict), "rollout_data不是字典类型"
            assert 'n_steps' in rollout_data, "rollout_data缺少n_steps"
            
            # 执行策略更新
            train_stats = trainer.update_policy(rollout_data)
            assert isinstance(train_stats, dict), "train_stats不是字典类型"
            assert 'policy_loss' in train_stats, "train_stats缺少policy_loss"
            
            self.validation_results['training_integration'] = True
            logger.info("✓ 训练器集成验证通过")
            return True
            
        except Exception as e:
            self.validation_results['training_integration'] = False
            logger.error(f"✗ 训练器集成验证失败: {e}")
            return False
    
    def validate_model_forward(self) -> bool:
        """验证模型前向传播"""
        logger.info("开始验证模型前向传播")
        
        try:
            from src.rl.models.mappo_actor import MAPPOActor, MAPPOActorCritic
            from src.rl.models.mappo_critic import MAPPOCritic
            
            # 模拟数据维度
            batch_size = 4
            n_agents = 6
            obs_dim = 68  # 根据环境配置
            action_dim = 42
            global_state_dim = 100
            
            # 创建模拟数据
            obs = torch.randn(batch_size, n_agents, obs_dim)
            action_mask = torch.ones(batch_size, n_agents, action_dim, dtype=torch.bool)
            # 随机屏蔽一些动作
            action_mask[:, :, -5:] = False
            
            global_state = torch.randn(batch_size, global_state_dim)
            actions = torch.randint(0, action_dim, (batch_size, n_agents))
            
            # 测试Actor
            actor = MAPPOActor(obs_dim, action_dim)
            logits = actor(obs, action_mask)
            assert logits.shape == (batch_size, n_agents, action_dim), f"Actor输出维度错误: {logits.shape}"
            
            # 验证动作掩码生效
            masked_logits = logits[action_mask == False]
            assert torch.all(masked_logits < -1e8), "动作掩码未生效"
            
            # 测试Critic
            critic = MAPPOCritic(global_state_dim)
            values = critic(global_state)
            assert values.shape == (batch_size,), f"Critic输出维度错误: {values.shape}"
            
            # 测试ActorCritic
            actor_critic = MAPPOActorCritic(obs_dim, action_dim, global_state_dim)
            test_actions, log_probs, test_values = actor_critic.get_action_and_value(
                obs, global_state, action_mask
            )
            assert test_actions.shape == (batch_size, n_agents), f"动作维度错误: {test_actions.shape}"
            assert log_probs.shape == (batch_size, n_agents), f"对数概率维度错误: {log_probs.shape}"
            assert test_values.shape == (batch_size,), f"价值维度错误: {test_values.shape}"
            
            # 测试evaluate_actions
            eval_log_probs, eval_values, entropy = actor_critic.evaluate_actions(
                obs, global_state, actions, action_mask
            )
            assert eval_log_probs.shape == (batch_size, n_agents), "评估对数概率维度错误"
            assert eval_values.shape == (batch_size,), "评估价值维度错误" 
            assert entropy.shape == (batch_size, n_agents), "熵维度错误"
            
            self.validation_results['model_forward'] = True
            logger.info("✓ 模型前向传播验证通过")
            return True
            
        except Exception as e:
            self.validation_results['model_forward'] = False
            logger.error(f"✗ 模型前向传播验证失败: {e}")
            return False
    
    def run_all_validations(self) -> bool:
        """运行所有验证"""
        logger.info("开始运行完整兼容性验证")
        
        validations = [
            ('环境包装器', self.validate_environment_wrappers),
            ('数据流', self.validate_data_flow),
            ('模型前向传播', self.validate_model_forward),
            ('训练器集成', self.validate_training_integration),
        ]
        
        all_passed = True
        for name, validation_func in validations:
            logger.info(f"\n{'='*50}")
            logger.info(f"验证: {name}")
            logger.info(f"{'='*50}")
            
            passed = validation_func()
            if not passed:
                all_passed = False
        
        # 输出总结
        logger.info(f"\n{'='*50}")
        logger.info("验证总结")
        logger.info(f"{'='*50}")
        
        for validation, result in self.validation_results.items():
            status = "✓" if result else "✗"
            logger.info(f"{status} {validation}: {'通过' if result else '失败'}")
        
        if all_passed:
            logger.info("\n🎉 所有兼容性验证通过！")
        else:
            logger.error("\n❌ 部分验证失败，请检查上述错误信息")
        
        return all_passed


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='兼容性验证脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='环境配置文件路径')
    parser.add_argument('--test', type=str, 
                       choices=['wrappers', 'dataflow', 'models', 'training', 'all'],
                       default='all', help='测试类型')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建验证器
    validator = CompatibilityValidator(args.config)
    
    # 运行指定验证
    if args.test == 'wrappers':
        success = validator.validate_environment_wrappers()
    elif args.test == 'dataflow':
        success = validator.validate_data_flow()
    elif args.test == 'models':
        success = validator.validate_model_forward()
    elif args.test == 'training':
        success = validator.validate_training_integration()
    else:  # all
        success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()