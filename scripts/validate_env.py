#!/usr/bin/env python3
"""
环境验证脚本
Environment validation script: Test core functionality
"""

import sys
import os
import argparse
import yaml
import logging
import pickle
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.parser import VOCParser
from src.data.layers import LayerGenerator
from src.env.agent_state import create_uav_state_machine, create_usv_state_machine
from src.env.events import EventScheduler, SimulationContext, Event, EventType
from src.env.charging import ChargingSystemManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_xml_parsing(xml_path: str, config: dict) -> bool:
    """测试XML解析功能"""
    logger.info("=== 测试XML解析 ===")
    
    try:
        gsd = config['map']['gsd_m_per_px']
        parser = VOCParser(xml_path, gsd)
        objects = parser.parse()
        
        # 验证对象计数
        class_counts = {}
        for obj in objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        logger.info(f"解析成功，共{len(objects)}个对象:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
        
        # 基本验证
        assert len(objects) > 0, "应该至少有一个对象"
        assert 'fan' in class_counts, "应该包含风机对象"
        
        logger.info("✓ XML解析测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ XML解析测试失败: {e}")
        return False


def test_agent_state_machine(config: dict) -> bool:
    """测试智能体状态机"""
    logger.info("=== 测试智能体状态机 ===")
    
    try:
        # 创建UAV
        uav = create_uav_state_machine("UAV_001", config)
        
        # 测试初始状态
        assert uav.state.mode.name == 'IDLE', "初始状态应为IDLE"
        assert uav.state.energy > 0, "初始能量应大于0"
        
        # 测试状态转换
        success = uav.transition_to(uav.state.mode.__class__.TRANSIT, 0.0, 
                                   target_pos=(100, 200))
        assert success, "应该能从IDLE转换到TRANSIT"
        
        # 测试移动和能量消耗
        initial_energy = uav.state.energy
        uav.update_position((50, 100), distance_traveled=111.8)
        assert uav.state.energy < initial_energy, "移动应该消耗能量"
        
        # 测试能量可行性检查
        feasible = uav.check_energy_feasibility(1000.0, 0.0, 120.0)
        logger.info(f"能量可行性检查: {feasible}")
        
        logger.info("✓ 智能体状态机测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 智能体状态机测试失败: {e}")
        return False


def test_charging_system(config: dict) -> bool:
    """测试充电系统"""
    logger.info("=== 测试充电系统 ===")
    
    try:
        # 创建充电系统管理器
        charging_manager = ChargingSystemManager(config)
        
        # 添加充电站
        station = charging_manager.add_fixed_station("test_station", (0, 0))
        
        # 创建测试智能体
        uav = create_uav_state_machine("UAV_001", config)
        uav.state.energy = 30.0  # 低电量
        
        # 测试队列加入
        position, wait_time = station.add_to_queue("UAV_001", 100.0, uav.state)
        assert position >= 0, "队列位置应该有效"
        
        # 测试开始充电
        session = station.start_charging("UAV_001", uav.state, 100.0)
        assert session is not None, "应该能开始充电"
        
        # 测试充电完成
        if session:
            completed = station.complete_charging(session.slot_id, 500.0)
            assert completed is not None, "应该能完成充电"
            assert completed.energy_added > 0, "应该有能量充入"
        
        logger.info("✓ 充电系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 充电系统测试失败: {e}")
        return False


def test_event_system(config: dict) -> bool:
    """测试事件系统"""
    logger.info("=== 测试事件系统 ===")
    
    try:
        # 创建仿真上下文
        context = SimulationContext(config)
        
        # 创建智能体
        uav = create_uav_state_machine("UAV_001", config)
        context.add_agent(uav)
        
        # 创建事件调度器
        scheduler = EventScheduler(context)
        
        # 创建测试事件
        event = Event(
            time=0.0,
            event_type=EventType.START_TRANSIT,
            agent_id="UAV_001",
            data={
                'target_pos': (100, 200),
                'target_id': 'task_001',
                'travel_time': 10.0,
                'travel_distance': 223.6
            }
        )
        
        scheduler.schedule_event(event)
        
        # 运行少量事件
        events_processed = 0
        max_events = 5
        
        while scheduler.event_queue and events_processed < max_events:
            if not scheduler.process_next_event():
                break
            events_processed += 1
        
        assert events_processed > 0, "应该处理至少一个事件"
        
        # 检查智能体状态更新
        agent = context.get_agent("UAV_001")
        assert agent is not None, "应该能找到智能体"
        
        logger.info(f"处理了{events_processed}个事件")
        logger.info("✓ 事件系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 事件系统测试失败: {e}")
        return False


def test_integration(xml_path: str, config: dict) -> bool:
    """测试系统集成"""
    logger.info("=== 测试系统集成 ===")
    
    try:
        # 端到端流程测试
        # 1. 解析XML
        gsd = config['map']['gsd_m_per_px']
        parser = VOCParser(xml_path, gsd)
        objects = parser.parse()
        
        # 2. 生成层
        from src.data.layers import LayerConfig
        layer_config = LayerConfig.from_config(config)
        layer_generator = LayerGenerator(layer_config)
        layers_data = layer_generator.generate_all_layers(objects)
        
        # 3. 创建智能体
        num_uav = config['agents']['uav']['n']
        num_usv = config['agents']['usv']['n']
        
        agents = []
        for i in range(num_uav):
            uav = create_uav_state_machine(f"UAV_{i:03d}", config)
            agents.append(uav)
        
        for i in range(num_usv):
            usv = create_usv_state_machine(f"USV_{i:03d}", config)
            agents.append(usv)
        
        # 4. 创建充电系统
        charging_manager = ChargingSystemManager(config)
        for i, station in enumerate(layers_data['charging_stations']):
            charging_manager.add_fixed_station(station.id, station.position)
        
        # 验证集成结果
        assert len(agents) == num_uav + num_usv, f"应该有{num_uav + num_usv}个智能体"
        assert len(charging_manager.fixed_stations) == len(layers_data['charging_stations']), "充电站数量应匹配"
        
        logger.info(f"成功创建{len(agents)}个智能体")
        logger.info(f"成功创建{len(charging_manager.fixed_stations)}个充电站")
        logger.info(f"地图包含{len(layers_data['task_points'])}个任务点")
        logger.info("✓ 系统集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 系统集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='验证环境功能')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--xml', default='content.xml', help='XML标注文件路径')
    parser.add_argument('--test', choices=['all', 'xml', 'agent', 'charging', 'event', 'integration'],
                       default='all', help='选择测试类型')
    
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
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 运行测试
        test_results = {}
        
        if args.test in ['all', 'xml']:
            test_results['xml'] = test_xml_parsing(args.xml, config)
        
        if args.test in ['all', 'agent']:
            test_results['agent'] = test_agent_state_machine(config)
        
        if args.test in ['all', 'charging']:
            test_results['charging'] = test_charging_system(config)
        
        if args.test in ['all', 'event']:
            test_results['event'] = test_event_system(config)
        
        if args.test in ['all', 'integration']:
            test_results['integration'] = test_integration(args.xml, config)
        
        # 汇总结果
        logger.info("\n=== 测试结果汇总 ===")
        all_passed = True
        for test_name, result in test_results.items():
            status = "通过" if result else "失败"
            logger.info(f"{test_name}: {status}")
            all_passed = all_passed and result
        
        if all_passed:
            logger.info("🎉 所有测试通过！环境验证成功！")
            return 0
        else:
            logger.warning("⚠️  部分测试失败，请检查相关功能")
            return 1
        
    except Exception as e:
        logger.error(f"验证过程失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())