"""
栅格生成器：将几何层转换为导航栅格
Grid Generator: Convert geometric layers to navigation grids

功能：
1. 将连续几何体栅格化为离散占据网格
2. 为UAV和USV生成不同的可通行域
3. 构建8邻接或16邻接权重图
4. 支持多分辨率栅格生成
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
import yaml
import logging
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import transform
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
import cv2

from .layers import LayerGenerator, ObstacleLayer, TaskPoint, ChargingStation

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """栅格配置数据类"""
    resolution: float = 10.0        # 栅格分辨率（米/格）
    connectivity: int = 8           # 邻接类型（4, 8, 16）
    boundary_buffer: float = 50.0   # 边界缓冲（米）
    
    @classmethod
    def from_config(cls, config: Dict) -> 'GridConfig':
        """从配置字典创建"""
        map_cfg = config.get('map', {})
        return cls(
            resolution=map_cfg.get('grid_res_m', 10.0),
            connectivity=8,
            boundary_buffer=50.0
        )


@dataclass
class NavigationGrid:
    """导航栅格数据类"""
    occupancy: np.ndarray           # 占据栅格 (H, W) - 0:可通行, 1:障碍
    resolution: float               # 分辨率（米/格）
    origin: Tuple[float, float]     # 原点世界坐标
    bounds: Tuple[float, float, float, float]  # 边界 (xmin, ymin, xmax, ymax)
    transform_matrix: np.ndarray    # 仿射变换矩阵
    
    # 图结构
    graph: Optional[nx.Graph] = None        # NetworkX图
    node_coords: Optional[np.ndarray] = None # 节点坐标 (N, 2)
    
    def __post_init__(self):
        """后处理：计算基本属性"""
        self.height, self.width = self.occupancy.shape
        self.passable_ratio = 1.0 - np.mean(self.occupancy)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((self.origin[1] + self.bounds[3] - self.bounds[1] - y) / self.resolution)
        return (row, col)
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        x = self.origin[0] + col * self.resolution + self.resolution / 2
        y = self.origin[1] + self.bounds[3] - self.bounds[1] - row * self.resolution - self.resolution / 2
        return (x, y)
    
    def is_valid_cell(self, row: int, col: int) -> bool:
        """检查栅格单元是否有效"""
        return (0 <= row < self.height and 
                0 <= col < self.width and 
                self.occupancy[row, col] == 0)
    
    def get_neighbors(self, row: int, col: int, connectivity: int = 8) -> List[Tuple[int, int]]:
        """获取邻接单元"""
        neighbors = []
        
        # 8邻接偏移
        if connectivity == 8:
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        elif connectivity == 4:
            offsets = [(-1,0), (0,-1), (0,1), (1,0)]
        else:
            raise ValueError(f"不支持的邻接类型: {connectivity}")
        
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if self.is_valid_cell(nr, nc):
                neighbors.append((nr, nc))
        
        return neighbors


class GridGenerator:
    """栅格生成器主类"""
    
    def __init__(self, config: GridConfig):
        """
        初始化栅格生成器
        
        Args:
            config: 栅格配置
        """
        self.config = config
        logger.info(f"栅格生成器初始化，分辨率={config.resolution}m/格, "
                   f"邻接类型={config.connectivity}邻接")
    
    def _calculate_grid_bounds(self, work_domain: Tuple[float, float, float, float]) -> Tuple[float, float, float, float, int, int]:
        """
        计算栅格边界和尺寸
        
        Args:
            work_domain: 工作域边界 (xmin, ymin, xmax, ymax)
            
        Returns:
            (xmin, ymin, xmax, ymax, width, height)
        """
        xmin, ymin, xmax, ymax = work_domain
        
        # 添加边界缓冲
        buffer = self.config.boundary_buffer
        xmin -= buffer
        ymin -= buffer
        xmax += buffer
        ymax += buffer
        
        # 对齐到栅格分辨率
        res = self.config.resolution
        xmin = np.floor(xmin / res) * res
        ymin = np.floor(ymin / res) * res
        xmax = np.ceil(xmax / res) * res
        ymax = np.ceil(ymax / res) * res
        
        # 计算栅格尺寸
        width = int((xmax - xmin) / res)
        height = int((ymax - ymin) / res)
        
        logger.info(f"栅格边界: ({xmin:.1f}, {ymin:.1f}) 到 ({xmax:.1f}, {ymax:.1f})")
        logger.info(f"栅格尺寸: {width} x {height} = {width*height:,} 格")
        
        return (xmin, ymin, xmax, ymax, width, height)
    
    def _rasterize_obstacles(self, obstacle_layers: Dict[str, ObstacleLayer], 
                           bounds: Tuple[float, float, float, float], 
                           width: int, height: int,
                           agent_type: str) -> np.ndarray:
        """
        栅格化障碍物层
        
        Args:
            obstacle_layers: 障碍物层字典
            bounds: 栅格边界
            width, height: 栅格尺寸
            agent_type: 智能体类型 ('uav' 或 'usv')
            
        Returns:
            占据栅格 (0=可通行, 1=障碍)
        """
        xmin, ymin, xmax, ymax = bounds
        
        # 创建仿射变换
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
        
        # 初始化占据栅格
        occupancy = np.zeros((height, width), dtype=np.uint8)
        
        # 遍历障碍物层
        for layer_name, layer in obstacle_layers.items():
            # 检查该层是否影响指定智能体类型
            if ((agent_type == 'uav' and layer.affects_uav) or 
                (agent_type == 'usv' and layer.affects_usv)):
                
                logger.debug(f"栅格化层 '{layer_name}' 用于 {agent_type.upper()}")
                
                # 收集所有几何体
                geometries = []
                for poly in layer.polygons:
                    if poly.is_valid and not poly.is_empty:
                        geometries.append((poly, 1))  # 1表示障碍
                
                if geometries:
                    # 栅格化
                    layer_grid = rasterize(
                        geometries,
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )
                    
                    # 合并到总占据栅格
                    occupancy = np.maximum(occupancy, layer_grid)
        
        return occupancy
    
    def _build_graph(self, grid: NavigationGrid) -> Tuple[nx.Graph, np.ndarray]:
        """
        构建导航图
        
        Args:
            grid: 导航栅格
            
        Returns:
            (图对象, 节点坐标数组)
        """
        logger.info("构建导航图...")
        
        G = nx.Graph()
        node_coords = []
        node_map = {}  # (row, col) -> node_id
        node_id = 0
        
        # 添加所有可通行节点
        for row in range(grid.height):
            for col in range(grid.width):
                if grid.is_valid_cell(row, col):
                    world_x, world_y = grid.grid_to_world(row, col)
                    G.add_node(node_id, row=row, col=col, x=world_x, y=world_y)
                    node_coords.append([world_x, world_y])
                    node_map[(row, col)] = node_id
                    node_id += 1
        
        # 添加边
        for row in range(grid.height):
            for col in range(grid.width):
                if not grid.is_valid_cell(row, col):
                    continue
                
                current_node = node_map[(row, col)]
                neighbors = grid.get_neighbors(row, col, self.config.connectivity)
                
                for nr, nc in neighbors:
                    neighbor_node = node_map[(nr, nc)]
                    
                    # 计算边权重（欧氏距离）
                    dx = (nc - col) * grid.resolution
                    dy = (nr - row) * grid.resolution
                    weight = np.sqrt(dx*dx + dy*dy)
                    
                    G.add_edge(current_node, neighbor_node, weight=weight)
        
        logger.info(f"图构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        
        return G, np.array(node_coords)
    
    def generate_uav_grid(self, layers_data: Dict) -> NavigationGrid:
        """
        生成UAV导航栅格
        
        Args:
            layers_data: 层数据字典
            
        Returns:
            UAV导航栅格
        """
        logger.info("生成UAV导航栅格...")
        
        work_domain = layers_data['work_domain']
        obstacle_layers = layers_data['obstacle_layers']
        
        # 计算栅格参数
        xmin, ymin, xmax, ymax, width, height = self._calculate_grid_bounds(work_domain)
        
        # 栅格化障碍物
        occupancy = self._rasterize_obstacles(
            obstacle_layers, 
            (xmin, ymin, xmax, ymax), 
            width, height,
            'uav'
        )
        
        # 创建导航栅格
        grid = NavigationGrid(
            occupancy=occupancy,
            resolution=self.config.resolution,
            origin=(xmin, ymin),
            bounds=(xmin, ymin, xmax, ymax),
            transform_matrix=from_bounds(xmin, ymin, xmax, ymax, width, height)
        )
        
        # 构建图
        graph, node_coords = self._build_graph(grid)
        grid.graph = graph
        grid.node_coords = node_coords
        
        logger.info(f"UAV栅格生成完成: {width}x{height}, 可通行率={grid.passable_ratio:.2%}")
        
        return grid
    
    def generate_usv_grid(self, layers_data: Dict) -> NavigationGrid:
        """
        生成USV导航栅格
        
        Args:
            layers_data: 层数据字典
            
        Returns:
            USV导航栅格
        """
        logger.info("生成USV导航栅格...")
        
        work_domain = layers_data['work_domain']
        obstacle_layers = layers_data['obstacle_layers']
        
        # 计算栅格参数
        xmin, ymin, xmax, ymax, width, height = self._calculate_grid_bounds(work_domain)
        
        # 栅格化障碍物
        occupancy = self._rasterize_obstacles(
            obstacle_layers, 
            (xmin, ymin, xmax, ymax), 
            width, height,
            'usv'
        )
        
        # 创建导航栅格
        grid = NavigationGrid(
            occupancy=occupancy,
            resolution=self.config.resolution,
            origin=(xmin, ymin),
            bounds=(xmin, ymin, xmax, ymax),
            transform_matrix=from_bounds(xmin, ymin, xmax, ymax, width, height)
        )
        
        # 构建图
        graph, node_coords = self._build_graph(grid)
        grid.graph = graph
        grid.node_coords = node_coords
        
        logger.info(f"USV栅格生成完成: {width}x{height}, 可通行率={grid.passable_ratio:.2%}")
        
        return grid
    
    def generate_all_grids(self, layers_data: Dict) -> Dict[str, NavigationGrid]:
        """
        生成所有导航栅格
        
        Args:
            layers_data: 层数据字典
            
        Returns:
            包含所有栅格的字典
        """
        logger.info("开始生成所有导航栅格...")
        
        grids = {
            'uav': self.generate_uav_grid(layers_data),
            'usv': self.generate_usv_grid(layers_data)
        }
        
        logger.info("所有导航栅格生成完成")
        return grids
    
    def save_grids(self, grids: Dict[str, NavigationGrid], output_dir: str) -> None:
        """
        保存栅格数据
        
        Args:
            grids: 栅格字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for agent_type, grid in grids.items():
            # 保存二进制文件
            grid_file = output_path / f"grid_{agent_type}.npz"
            np.savez_compressed(
                grid_file,
                occupancy=grid.occupancy,
                resolution=grid.resolution,
                origin=np.array(grid.origin),
                bounds=np.array(grid.bounds),
                transform_matrix=grid.transform_matrix,
                node_coords=grid.node_coords if grid.node_coords is not None else np.array([])
            )
            
            # 保存图结构
            if grid.graph is not None:
                graph_file = output_path / f"graph_{agent_type}.pickle"
                with open(graph_file, 'wb') as f:
                    pickle.dump(grid.graph, f)
            
            logger.info(f"{agent_type.upper()}栅格已保存到: {grid_file}")
    
    def visualize_grids(self, grids: Dict[str, NavigationGrid], 
                       output_dir: str = None, 
                       figsize: Tuple[int, int] = (15, 6)) -> None:
        """
        可视化栅格
        
        Args:
            grids: 栅格字典
            output_dir: 输出目录
            figsize: 图片尺寸
        """
        fig, axes = plt.subplots(1, len(grids), figsize=figsize)
        if len(grids) == 1:
            axes = [axes]
        
        colors = {'uav': 'Reds', 'usv': 'Blues'}
        
        for i, (agent_type, grid) in enumerate(grids.items()):
            ax = axes[i]
            
            # 显示占据栅格
            im = ax.imshow(
                grid.occupancy, 
                extent=[grid.bounds[0], grid.bounds[2], grid.bounds[1], grid.bounds[3]],
                origin='lower',
                cmap=colors.get(agent_type, 'gray'),
                alpha=0.7
            )
            
            ax.set_title(f'{agent_type.upper()}导航栅格\n'
                        f'分辨率: {grid.resolution}m, 可通行率: {grid.passable_ratio:.1%}')
            ax.set_xlabel('X坐标 (米)')
            ax.set_ylabel('Y坐标 (米)')
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='占据状态 (0=可通行, 1=障碍)')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig_file = output_path / "grids_visualization.png"
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            logger.info(f"栅格可视化已保存到: {fig_file}")
        
        plt.show()


def generate_grids_from_layers(layers_path: str, config_path: str, output_dir: str = None) -> Dict[str, NavigationGrid]:
    """
    从层数据生成栅格
    
    Args:
        layers_path: 层数据文件路径
        config_path: 配置文件路径
        output_dir: 输出目录
        
    Returns:
        栅格字典
    """
    # 加载层数据
    with open(layers_path, 'rb') as f:
        layers_data = pickle.load(f)
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    grid_config = GridConfig.from_config(config)
    
    # 生成栅格
    generator = GridGenerator(grid_config)
    grids = generator.generate_all_grids(layers_data)
    
    # 保存结果
    if output_dir:
        generator.save_grids(grids, output_dir)
        generator.visualize_grids(grids, output_dir)
    
    return grids


if __name__ == "__main__":
    # 测试代码
    layers_path = "layers.pkl"
    config_path = "config.yaml"
    output_dir = "grids"
    
    try:
        # 生成栅格
        grids = generate_grids_from_layers(layers_path, config_path, output_dir)
        
        # 打印统计信息
        print("\n=== 栅格生成结果统计 ===")
        for agent_type, grid in grids.items():
            print(f"{agent_type.upper()}栅格:")
            print(f"  尺寸: {grid.width} x {grid.height}")
            print(f"  分辨率: {grid.resolution}m/格")
            print(f"  可通行率: {grid.passable_ratio:.2%}")
            if grid.graph:
                print(f"  图节点数: {grid.graph.number_of_nodes()}")
                print(f"  图边数: {grid.graph.number_of_edges()}")
        
    except Exception as e:
        logger.error(f"栅格生成失败: {e}")
        raise