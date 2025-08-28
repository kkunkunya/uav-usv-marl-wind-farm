"""
层生成器：地图数据的几何层处理
Layer Generator: Geometric layer processing for map data

功能：
1. 基于解析的对象数据生成多层地理信息
2. 应用安全缓冲区和形态学膨胀
3. 生成任务点层、充电站层、障碍物层
4. 确保几何体的有效性和一致性
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
import yaml
import logging
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import affinity
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from dataclasses import dataclass, field
import hashlib

from .parser import DetectedObject, VOCParser

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerConfig:
    """层配置数据类"""
    buffer_fan: float = 80.0        # 风机安全缓冲半径（米）
    buffer_uav: float = 40.0        # UAV障碍物膨胀半径（米）
    buffer_island: float = 10.0     # 岛屿外扩半径（米）
    
    @classmethod
    def from_config(cls, config: Dict) -> 'LayerConfig':
        """从配置字典创建"""
        buffer_cfg = config.get('map', {}).get('buffer', {})
        return cls(
            buffer_fan=buffer_cfg.get('rho_fan', 80.0),
            buffer_uav=buffer_cfg.get('rho_uav', 40.0),
            buffer_island=buffer_cfg.get('rho_island', 10.0)
        )


@dataclass
class TaskPoint:
    """任务点数据类"""
    id: str
    position: Tuple[float, float]   # 世界坐标位置
    service_zone: Polygon           # 服务区域（包含安全缓冲）
    original_object: DetectedObject # 原始检测对象
    attributes: Dict = field(default_factory=dict)


@dataclass
class ChargingStation:
    """充电站数据类"""
    id: str
    position: Tuple[float, float]   # 世界坐标位置
    service_area: Polygon           # 服务区域
    original_object: DetectedObject # 原始检测对象
    capacity: int = 2               # 充电桩数量
    power: float = 6.0             # 充电功率
    attributes: Dict = field(default_factory=dict)


@dataclass
class ObstacleLayer:
    """障碍物层数据类"""
    name: str                       # 层名称
    polygons: List[Polygon]         # 障碍物多边形列表
    merged_polygon: Optional[Union[Polygon, MultiPolygon]] = None  # 合并后的多边形
    buffer_radius: float = 0.0      # 缓冲半径
    affects_uav: bool = True        # 是否影响UAV
    affects_usv: bool = True        # 是否影响USV


class GeometryProcessor:
    """几何处理器：处理缓冲、合并、验证等操作"""
    
    @staticmethod
    def apply_buffer(geometry: Union[Polygon, Point], radius: float) -> Polygon:
        """
        应用缓冲操作
        
        Args:
            geometry: 输入几何体
            radius: 缓冲半径（米）
            
        Returns:
            缓冲后的多边形
        """
        if radius <= 0:
            if isinstance(geometry, Point):
                # 点需要转换为最小多边形
                return Point(geometry.x, geometry.y).buffer(0.1)
            return geometry
        
        buffered = geometry.buffer(radius)
        
        # 确保结果有效
        if not buffered.is_valid:
            buffered = make_valid(buffered)
        
        return buffered
    
    @staticmethod
    def merge_polygons(polygons: List[Polygon]) -> Union[Polygon, MultiPolygon]:
        """
        合并多个多边形
        
        Args:
            polygons: 多边形列表
            
        Returns:
            合并后的几何体
        """
        if not polygons:
            return Polygon()
        
        if len(polygons) == 1:
            return polygons[0]
        
        try:
            merged = unary_union(polygons)
            
            # 确保结果有效
            if not merged.is_valid:
                merged = make_valid(merged)
            
            return merged
            
        except Exception as e:
            logger.warning(f"多边形合并失败: {e}")
            return MultiPolygon(polygons)
    
    @staticmethod
    def validate_geometry(geometry: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon]:
        """
        验证并修复几何体
        
        Args:
            geometry: 输入几何体
            
        Returns:
            修复后的几何体
        """
        if geometry.is_valid:
            return geometry
        
        try:
            fixed = make_valid(geometry)
            logger.debug(f"几何体已修复")
            return fixed
        except Exception as e:
            logger.warning(f"几何体修复失败: {e}")
            return geometry


class LayerGenerator:
    """层生成器主类"""
    
    def __init__(self, config: LayerConfig):
        """
        初始化层生成器
        
        Args:
            config: 层配置
        """
        self.config = config
        self.geometry_processor = GeometryProcessor()
        
        # 存储生成的层
        self.task_points: List[TaskPoint] = []
        self.charging_stations: List[ChargingStation] = []
        self.obstacle_layers: Dict[str, ObstacleLayer] = {}
        
        logger.info(f"层生成器初始化，缓冲配置: 风机={config.buffer_fan}m, "
                   f"UAV障碍={config.buffer_uav}m, 岛屿={config.buffer_island}m")
    
    def generate_task_layer(self, objects: List[DetectedObject]) -> List[TaskPoint]:
        """
        生成任务点层
        
        Args:
            objects: 风机对象列表（class_name='fan'）
            
        Returns:
            任务点列表
        """
        logger.info("生成任务点层...")
        
        task_points = []
        fan_objects = [obj for obj in objects if obj.class_name == 'fan']
        
        for i, obj in enumerate(fan_objects):
            # 创建任务点
            task_point = TaskPoint(
                id=f"task_{i:03d}",
                position=obj.center_world,
                service_zone=self.geometry_processor.apply_buffer(
                    Point(*obj.center_world), 
                    self.config.buffer_fan
                ),
                original_object=obj
            )
            
            task_points.append(task_point)
        
        self.task_points = task_points
        logger.info(f"生成了{len(task_points)}个任务点")
        
        return task_points
    
    def generate_charging_layer(self, objects: List[DetectedObject]) -> List[ChargingStation]:
        """
        生成充电站层
        
        Args:
            objects: 充电站对象列表（class_name='charge station'）
            
        Returns:
            充电站列表
        """
        logger.info("生成充电站层...")
        
        charging_stations = []
        station_objects = [obj for obj in objects if obj.class_name == 'charge station']
        
        for i, obj in enumerate(station_objects):
            # 创建充电站
            station = ChargingStation(
                id=f"station_{i:03d}",
                position=obj.center_world,
                service_area=obj.polygon_world,  # 使用原始区域作为服务区域
                original_object=obj
            )
            
            charging_stations.append(station)
        
        self.charging_stations = charging_stations
        logger.info(f"生成了{len(charging_stations)}个充电站")
        
        return charging_stations
    
    def generate_obstacle_layers(self, objects: List[DetectedObject]) -> Dict[str, ObstacleLayer]:
        """
        生成障碍物层
        
        Args:
            objects: 所有对象列表
            
        Returns:
            障碍物层字典
        """
        logger.info("生成障碍物层...")
        
        obstacle_layers = {}
        
        # 1. 硬障碍层（岛屿）- 影响UAV和USV
        island_objects = [obj for obj in objects if obj.class_name == 'island']
        if island_objects:
            island_polygons = []
            for obj in island_objects:
                buffered = self.geometry_processor.apply_buffer(
                    obj.polygon_world, 
                    self.config.buffer_island
                )
                island_polygons.append(buffered)
            
            obstacle_layers['hard_obstacles'] = ObstacleLayer(
                name='hard_obstacles',
                polygons=island_polygons,
                merged_polygon=self.geometry_processor.merge_polygons(island_polygons),
                buffer_radius=self.config.buffer_island,
                affects_uav=True,
                affects_usv=True
            )
        
        # 2. UAV软障碍层（UAV专用禁飞区）- 仅影响UAV
        uav_obstacle_objects = [obj for obj in objects if obj.class_name == 'UAV obstacle']
        if uav_obstacle_objects:
            uav_polygons = []
            for obj in uav_obstacle_objects:
                buffered = self.geometry_processor.apply_buffer(
                    obj.polygon_world, 
                    self.config.buffer_uav
                )
                uav_polygons.append(buffered)
            
            obstacle_layers['uav_obstacles'] = ObstacleLayer(
                name='uav_obstacles',
                polygons=uav_polygons,
                merged_polygon=self.geometry_processor.merge_polygons(uav_polygons),
                buffer_radius=self.config.buffer_uav,
                affects_uav=True,
                affects_usv=False
            )
        
        # 3. 风机安全缓冲层 - 仅影响UAV
        fan_objects = [obj for obj in objects if obj.class_name == 'fan']
        if fan_objects:
            fan_buffer_polygons = []
            for obj in fan_objects:
                buffered = self.geometry_processor.apply_buffer(
                    Point(*obj.center_world), 
                    self.config.buffer_fan
                )
                fan_buffer_polygons.append(buffered)
            
            obstacle_layers['fan_buffers'] = ObstacleLayer(
                name='fan_buffers',
                polygons=fan_buffer_polygons,
                merged_polygon=self.geometry_processor.merge_polygons(fan_buffer_polygons),
                buffer_radius=self.config.buffer_fan,
                affects_uav=True,
                affects_usv=False
            )
        
        self.obstacle_layers = obstacle_layers
        
        # 记录统计信息
        for name, layer in obstacle_layers.items():
            logger.info(f"障碍物层 '{name}': {len(layer.polygons)}个多边形, "
                       f"影响UAV={layer.affects_uav}, 影响USV={layer.affects_usv}")
        
        return obstacle_layers
    
    def generate_all_layers(self, objects: List[DetectedObject]) -> Dict:
        """
        生成所有层
        
        Args:
            objects: 检测对象列表
            
        Returns:
            包含所有层的字典
        """
        logger.info("开始生成所有地图层...")
        
        # 生成各层
        task_points = self.generate_task_layer(objects)
        charging_stations = self.generate_charging_layer(objects)
        obstacle_layers = self.generate_obstacle_layers(objects)
        
        # 计算工作域边界
        work_domain = self._calculate_work_domain(objects)
        
        layers_data = {
            'task_points': task_points,
            'charging_stations': charging_stations,
            'obstacle_layers': obstacle_layers,
            'work_domain': work_domain,
            'config': self.config,
            'metadata': {
                'total_objects': len(objects),
                'num_tasks': len(task_points),
                'num_stations': len(charging_stations),
                'num_obstacle_layers': len(obstacle_layers),
                'generation_time': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info("所有地图层生成完成")
        return layers_data
    
    def _calculate_work_domain(self, objects: List[DetectedObject]) -> Tuple[float, float, float, float]:
        """
        计算工作域边界
        
        Args:
            objects: 所有对象列表
            
        Returns:
            (xmin, ymin, xmax, ymax) 工作域边界
        """
        if not objects:
            return (0, 0, 0, 0)
        
        all_x = []
        all_y = []
        
        for obj in objects:
            x, y = obj.center_world
            all_x.append(x)
            all_y.append(y)
        
        # 添加一定的边界裕量
        margin = 200  # 米
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin
        
        logger.info(f"工作域: ({xmin:.1f}, {ymin:.1f}) 到 ({xmax:.1f}, {ymax:.1f})")
        
        return (xmin, ymin, xmax, ymax)
    
    def save_layers(self, output_path: str) -> None:
        """
        保存层数据到文件
        
        Args:
            output_path: 输出文件路径
        """
        layers_data = {
            'task_points': self.task_points,
            'charging_stations': self.charging_stations,
            'obstacle_layers': self.obstacle_layers,
            'config': self.config
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(layers_data, f)
        
        logger.info(f"层数据已保存到: {output_path}")
    
    def visualize_layers(self, output_path: str = None, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        可视化所有层
        
        Args:
            output_path: 图片保存路径（可选）
            figsize: 图片尺寸
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置颜色方案
        colors = {
            'task_points': 'red',
            'charging_stations': 'blue',
            'hard_obstacles': 'gray',
            'uav_obstacles': 'orange',
            'fan_buffers': 'lightcoral'
        }
        
        # 绘制障碍物层
        for name, layer in self.obstacle_layers.items():
            color = colors.get(name, 'black')
            alpha = 0.3 if 'buffer' in name else 0.6
            
            for polygon in layer.polygons:
                if polygon.geom_type == 'Polygon':
                    x, y = polygon.exterior.xy
                    ax.fill(x, y, color=color, alpha=alpha, label=name if polygon == layer.polygons[0] else "")
                elif polygon.geom_type == 'MultiPolygon':
                    for poly in polygon.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color=color, alpha=alpha, label=name if poly == polygon.geoms[0] else "")
        
        # 绘制任务点
        for task in self.task_points:
            ax.plot(task.position[0], task.position[1], 'o', 
                   color=colors['task_points'], markersize=8, label='任务点' if task == self.task_points[0] else "")
        
        # 绘制充电站
        for station in self.charging_stations:
            ax.plot(station.position[0], station.position[1], 's', 
                   color=colors['charging_stations'], markersize=10, label='充电站' if station == self.charging_stations[0] else "")
        
        ax.set_xlabel('X坐标 (米)')
        ax.set_ylabel('Y坐标 (米)')
        ax.set_title('海上风电场地图层可视化')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图片已保存到: {output_path}")
        
        plt.show()


def generate_layers_from_config(xml_path: str, config_path: str, output_path: str = None) -> Dict:
    """
    从配置文件生成层数据
    
    Args:
        xml_path: XML文件路径
        config_path: 配置文件路径
        output_path: 输出文件路径（可选）
        
    Returns:
        层数据字典
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    layer_config = LayerConfig.from_config(config)
    gsd = config['map']['gsd_m_per_px']
    
    # 解析XML
    parser = VOCParser(xml_path, gsd)
    objects = parser.parse()
    
    # 生成层
    generator = LayerGenerator(layer_config)
    layers_data = generator.generate_all_layers(objects)
    
    # 保存结果
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(layers_data, f)
        logger.info(f"层数据已保存到: {output_path}")
    
    return layers_data


if __name__ == "__main__":
    # 测试代码
    xml_path = "content.xml"
    config_path = "config.yaml"
    
    try:
        # 生成层数据
        layers_data = generate_layers_from_config(xml_path, config_path, "layers.pkl")
        
        # 打印统计信息
        print("\n=== 层生成结果统计 ===")
        print(f"任务点数量: {len(layers_data['task_points'])}")
        print(f"充电站数量: {len(layers_data['charging_stations'])}")
        print(f"障碍物层数量: {len(layers_data['obstacle_layers'])}")
        
        for name, layer in layers_data['obstacle_layers'].items():
            print(f"  {name}: {len(layer.polygons)}个多边形")
        
        # 生成可视化
        generator = LayerGenerator(layers_data['config'])
        generator.task_points = layers_data['task_points']
        generator.charging_stations = layers_data['charging_stations']
        generator.obstacle_layers = layers_data['obstacle_layers']
        
        generator.visualize_layers("layers_preview.png")
        
    except Exception as e:
        logger.error(f"层生成失败: {e}")
        raise