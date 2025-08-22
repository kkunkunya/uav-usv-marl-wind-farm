"""
路径缓存系统：多对多最短路径计算与缓存
Path Cache System: All-pairs shortest path computation and caching

功能：
1. 基于栅格图进行A*最短路径搜索
2. 预计算所有重要节点间的距离矩阵
3. 支持按需计算和缓存机制
4. 处理不可达路径和动态障碍
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
import pickle
import yaml
import logging
from collections import defaultdict, deque
import heapq
import networkx as nx
from dataclasses import dataclass, field
import time
import zstandard as zstd
import hashlib

from ..data.grid import NavigationGrid, GridGenerator
from ..data.layers import TaskPoint, ChargingStation

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PathNode:
    """路径节点"""
    id: str                         # 节点ID
    position: Tuple[float, float]   # 世界坐标位置
    grid_coord: Tuple[int, int]     # 栅格坐标
    node_type: str                  # 节点类型: task/station/base/waypoint
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class PathResult:
    """路径搜索结果"""
    source: str                     # 起点ID
    target: str                     # 终点ID
    distance: float                 # 路径长度（米）
    path_coords: List[Tuple[float, float]] = field(default_factory=list)  # 路径坐标列表
    computation_time: float = 0.0   # 计算耗时（秒）
    is_valid: bool = True           # 路径是否有效
    
    @property
    def is_reachable(self) -> bool:
        """是否可达"""
        return self.is_valid and self.distance < float('inf')


class AStarPathfinder:
    """A*路径搜索器"""
    
    def __init__(self, grid: NavigationGrid):
        """
        初始化A*搜索器
        
        Args:
            grid: 导航栅格
        """
        self.grid = grid
        self.graph = grid.graph
        
        # 创建栅格坐标到节点ID的映射
        self.coord_to_node = {}
        if self.graph and grid.node_coords is not None:
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                self.coord_to_node[(node_data['row'], node_data['col'])] = node_id
    
    def _heuristic(self, node1: int, node2: int) -> float:
        """A*启发式函数（欧氏距离）"""
        data1 = self.graph.nodes[node1]
        data2 = self.graph.nodes[node2]
        
        dx = data1['x'] - data2['x']
        dy = data1['y'] - data2['y']
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _find_nearest_node(self, world_pos: Tuple[float, float]) -> Optional[int]:
        """
        找到最近的可通行节点
        
        Args:
            world_pos: 世界坐标位置
            
        Returns:
            最近节点ID，如果没有找到返回None
        """
        x, y = world_pos
        row, col = self.grid.world_to_grid(x, y)
        
        # 首先检查精确位置
        if (row, col) in self.coord_to_node:
            return self.coord_to_node[(row, col)]
        
        # 搜索附近的可通行节点
        search_radius = 5  # 搜索半径（格数）
        min_dist = float('inf')
        nearest_node = None
        
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nr, nc = row + dr, col + dc
                if (nr, nc) in self.coord_to_node:
                    node_id = self.coord_to_node[(nr, nc)]
                    node_data = self.graph.nodes[node_id]
                    
                    # 计算距离
                    dx = node_data['x'] - x
                    dy = node_data['y'] - y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node_id
        
        return nearest_node
    
    def find_path(self, start_pos: Tuple[float, float], 
                  end_pos: Tuple[float, float]) -> PathResult:
        """
        使用A*算法搜索路径
        
        Args:
            start_pos: 起点世界坐标
            end_pos: 终点世界坐标
            
        Returns:
            路径搜索结果
        """
        start_time = time.time()
        
        # 找到最近的节点
        start_node = self._find_nearest_node(start_pos)
        end_node = self._find_nearest_node(end_pos)
        
        if start_node is None or end_node is None:
            return PathResult(
                source=f"{start_pos}",
                target=f"{end_pos}",
                distance=float('inf'),
                computation_time=time.time() - start_time,
                is_valid=False
            )
        
        if start_node == end_node:
            # 起点和终点相同
            return PathResult(
                source=f"{start_pos}",
                target=f"{end_pos}",
                distance=0.0,
                path_coords=[start_pos, end_pos],
                computation_time=time.time() - start_time
            )
        
        # A*搜索
        try:
            path_nodes = nx.astar_path(
                self.graph, 
                start_node, 
                end_node, 
                heuristic=self._heuristic,
                weight='weight'
            )
            
            # 计算路径长度
            path_length = nx.astar_path_length(
                self.graph, 
                start_node, 
                end_node, 
                heuristic=self._heuristic,
                weight='weight'
            )
            
            # 提取路径坐标
            path_coords = [start_pos]  # 从实际起点开始
            for node_id in path_nodes:
                node_data = self.graph.nodes[node_id]
                path_coords.append((node_data['x'], node_data['y']))
            path_coords.append(end_pos)  # 到实际终点
            
            return PathResult(
                source=f"{start_pos}",
                target=f"{end_pos}",
                distance=path_length,
                path_coords=path_coords,
                computation_time=time.time() - start_time
            )
            
        except nx.NetworkXNoPath:
            # 无路径
            return PathResult(
                source=f"{start_pos}",
                target=f"{end_pos}",
                distance=float('inf'),
                computation_time=time.time() - start_time,
                is_valid=False
            )
        except Exception as e:
            logger.warning(f"路径搜索异常: {e}")
            return PathResult(
                source=f"{start_pos}",
                target=f"{end_pos}",
                distance=float('inf'),
                computation_time=time.time() - start_time,
                is_valid=False
            )


class PathCache:
    """路径缓存管理器"""
    
    def __init__(self, compression: bool = True):
        """
        初始化路径缓存
        
        Args:
            compression: 是否启用压缩
        """
        self.compression = compression
        self.distance_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.path_cache: Dict[str, PathResult] = {}
        self.nodes: Dict[str, PathNode] = {}
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_computations = 0
    
    def add_node(self, node: PathNode) -> None:
        """添加节点到缓存"""
        self.nodes[node.id] = node
    
    def add_path(self, result: PathResult) -> None:
        """添加路径到缓存"""
        key = f"{result.source}->{result.target}"
        self.path_cache[key] = result
        
        # 更新距离矩阵
        self.distance_matrix[result.source][result.target] = result.distance
        
        # 对称路径（如果适用）
        if result.is_reachable:
            reverse_key = f"{result.target}->{result.source}"
            if reverse_key not in self.path_cache:
                reverse_result = PathResult(
                    source=result.target,
                    target=result.source,
                    distance=result.distance,
                    path_coords=list(reversed(result.path_coords)),
                    computation_time=result.computation_time,
                    is_valid=result.is_valid
                )
                self.path_cache[reverse_key] = reverse_result
                self.distance_matrix[result.target][result.source] = result.distance
    
    def get_distance(self, source: str, target: str) -> float:
        """
        获取两点间距离
        
        Args:
            source: 起点ID
            target: 终点ID
            
        Returns:
            距离（米），不可达返回inf
        """
        if source == target:
            return 0.0
        
        if source in self.distance_matrix and target in self.distance_matrix[source]:
            self.cache_hits += 1
            return self.distance_matrix[source][target]
        
        self.cache_misses += 1
        return float('inf')  # 未缓存的路径认为不可达
    
    def get_path(self, source: str, target: str) -> Optional[PathResult]:
        """获取路径结果"""
        key = f"{source}->{target}"
        return self.path_cache.get(key)
    
    def save_to_file(self, filepath: str) -> None:
        """保存缓存到文件"""
        cache_data = {
            'distance_matrix': dict(self.distance_matrix),
            'path_cache': self.path_cache,
            'nodes': self.nodes,
            'metadata': {
                'total_nodes': len(self.nodes),
                'total_paths': len(self.path_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'save_time': pd.Timestamp.now().isoformat()
            }
        }
        
        if self.compression:
            # 使用zstd压缩
            data_bytes = pickle.dumps(cache_data)
            compressed_data = zstd.compress(data_bytes)
            
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
        
        logger.info(f"路径缓存已保存到: {filepath}")
        logger.info(f"缓存统计: {len(self.nodes)}个节点, {len(self.path_cache)}条路径")
    
    def load_from_file(self, filepath: str) -> None:
        """从文件加载缓存"""
        try:
            if self.compression:
                with open(filepath, 'rb') as f:
                    compressed_data = f.read()
                data_bytes = zstd.decompress(compressed_data)
                cache_data = pickle.loads(data_bytes)
            else:
                with open(filepath, 'rb') as f:
                    cache_data = pickle.load(f)
            
            self.distance_matrix = defaultdict(dict, cache_data['distance_matrix'])
            self.path_cache = cache_data['path_cache']
            self.nodes = cache_data['nodes']
            
            metadata = cache_data.get('metadata', {})
            logger.info(f"路径缓存已加载: {metadata.get('total_nodes', 0)}个节点, "
                       f"{metadata.get('total_paths', 0)}条路径")
            
        except Exception as e:
            logger.error(f"加载路径缓存失败: {e}")
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PathCacheGenerator:
    """路径缓存生成器"""
    
    def __init__(self, grids: Dict[str, NavigationGrid]):
        """
        初始化缓存生成器
        
        Args:
            grids: 导航栅格字典
        """
        self.grids = grids
        self.pathfinders = {
            agent_type: AStarPathfinder(grid) 
            for agent_type, grid in grids.items()
        }
        self.caches = {
            agent_type: PathCache() 
            for agent_type in grids.keys()
        }
    
    def _create_path_nodes(self, layers_data: Dict) -> Dict[str, List[PathNode]]:
        """
        创建路径节点
        
        Args:
            layers_data: 层数据
            
        Returns:
            按智能体类型分组的节点列表
        """
        nodes_by_type = {agent_type: [] for agent_type in self.grids.keys()}
        
        # 添加基地节点（假设在原点）
        base_node = PathNode(
            id="base",
            position=(0.0, 0.0),
            grid_coord=(0, 0),  # 将在运行时计算
            node_type="base"
        )
        
        for agent_type in self.grids.keys():
            grid = self.grids[agent_type]
            
            # 更新基地节点的栅格坐标
            base_grid_coord = grid.world_to_grid(*base_node.position)
            base_node.grid_coord = base_grid_coord
            
            nodes_by_type[agent_type].append(base_node)
        
        # 添加任务节点
        for i, task in enumerate(layers_data['task_points']):
            task_node = PathNode(
                id=task.id,
                position=task.position,
                grid_coord=(0, 0),  # 将在运行时计算
                node_type="task"
            )
            
            for agent_type in self.grids.keys():
                grid = self.grids[agent_type]
                task_grid_coord = grid.world_to_grid(*task.position)
                task_node.grid_coord = task_grid_coord
                
                nodes_by_type[agent_type].append(task_node)
        
        # 添加充电站节点
        for i, station in enumerate(layers_data['charging_stations']):
            station_node = PathNode(
                id=station.id,
                position=station.position,
                grid_coord=(0, 0),  # 将在运行时计算
                node_type="station"
            )
            
            for agent_type in self.grids.keys():
                grid = self.grids[agent_type]
                station_grid_coord = grid.world_to_grid(*station.position)
                station_node.grid_coord = station_grid_coord
                
                nodes_by_type[agent_type].append(station_node)
        
        return nodes_by_type
    
    def generate_full_cache(self, layers_data: Dict) -> None:
        """
        生成完整的路径缓存（所有节点对）
        
        Args:
            layers_data: 层数据
        """
        logger.info("开始生成完整路径缓存...")
        
        nodes_by_type = self._create_path_nodes(layers_data)
        
        for agent_type in self.grids.keys():
            logger.info(f"生成{agent_type.upper()}路径缓存...")
            
            pathfinder = self.pathfinders[agent_type]
            cache = self.caches[agent_type]
            nodes = nodes_by_type[agent_type]
            
            # 添加所有节点到缓存
            for node in nodes:
                cache.add_node(node)
            
            # 计算所有节点对的路径
            total_pairs = len(nodes) * (len(nodes) - 1)
            computed_pairs = 0
            
            start_time = time.time()
            
            for i, source_node in enumerate(nodes):
                for j, target_node in enumerate(nodes):
                    if i != j:  # 跳过自身到自身
                        # 计算路径
                        result = pathfinder.find_path(
                            source_node.position,
                            target_node.position
                        )
                        result.source = source_node.id
                        result.target = target_node.id
                        
                        # 添加到缓存
                        cache.add_path(result)
                        
                        computed_pairs += 1
                        
                        # 进度报告
                        if computed_pairs % 100 == 0 or computed_pairs == total_pairs:
                            elapsed = time.time() - start_time
                            progress = computed_pairs / total_pairs
                            eta = elapsed / progress - elapsed if progress > 0 else 0
                            
                            logger.info(f"  {agent_type.upper()}: {computed_pairs}/{total_pairs} "
                                       f"({progress:.1%}) - 剩余时间: {eta:.1f}s")
            
            elapsed = time.time() - start_time
            logger.info(f"{agent_type.upper()}缓存生成完成: {computed_pairs}对路径, 耗时{elapsed:.1f}s")
            
            # 统计可达性
            reachable_count = sum(1 for result in cache.path_cache.values() if result.is_reachable)
            logger.info(f"  可达路径: {reachable_count}/{len(cache.path_cache)} ({reachable_count/len(cache.path_cache):.1%})")
    
    def save_caches(self, output_dir: str) -> None:
        """保存所有缓存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for agent_type, cache in self.caches.items():
            cache_file = output_path / f"path_cache_{agent_type}.zst"
            cache.save_to_file(str(cache_file))


def generate_path_cache(layers_path: str, grids_dir: str, 
                       config_path: str, output_dir: str) -> Dict[str, PathCache]:
    """
    生成路径缓存
    
    Args:
        layers_path: 层数据文件路径
        grids_dir: 栅格数据目录
        config_path: 配置文件路径
        output_dir: 输出目录
        
    Returns:
        路径缓存字典
    """
    # 加载层数据
    with open(layers_path, 'rb') as f:
        layers_data = pickle.load(f)
    
    # 加载栅格数据
    grids = {}
    grids_path = Path(grids_dir)
    
    for agent_type in ['uav', 'usv']:
        grid_file = grids_path / f"grid_{agent_type}.npz"
        graph_file = grids_path / f"graph_{agent_type}.pickle"
        
        if grid_file.exists() and graph_file.exists():
            # 加载栅格数据
            grid_data = np.load(grid_file)
            
            # 加载图数据
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            
            # 重建NavigationGrid对象
            grid = NavigationGrid(
                occupancy=grid_data['occupancy'],
                resolution=float(grid_data['resolution']),
                origin=tuple(grid_data['origin']),
                bounds=tuple(grid_data['bounds']),
                transform_matrix=grid_data['transform_matrix'],
                graph=graph,
                node_coords=grid_data['node_coords'] if 'node_coords' in grid_data else None
            )
            
            grids[agent_type] = grid
    
    if not grids:
        raise ValueError("未找到有效的栅格数据")
    
    # 生成缓存
    generator = PathCacheGenerator(grids)
    generator.generate_full_cache(layers_data)
    
    # 保存缓存
    generator.save_caches(output_dir)
    
    return generator.caches


if __name__ == "__main__":
    # 测试代码
    layers_path = "layers.pkl"
    grids_dir = "grids"
    config_path = "config.yaml"
    output_dir = "cache"
    
    try:
        # 生成路径缓存
        caches = generate_path_cache(layers_path, grids_dir, config_path, output_dir)
        
        # 打印统计信息
        print("\n=== 路径缓存统计 ===")
        for agent_type, cache in caches.items():
            print(f"{agent_type.upper()}缓存:")
            print(f"  节点数: {len(cache.nodes)}")
            print(f"  路径数: {len(cache.path_cache)}")
            print(f"  缓存命中率: {cache.cache_hit_rate:.2%}")
            
            # 可达性统计
            reachable = sum(1 for r in cache.path_cache.values() if r.is_reachable)
            total = len(cache.path_cache)
            print(f"  路径可达率: {reachable}/{total} ({reachable/total:.1%})")
        
    except Exception as e:
        logger.error(f"路径缓存生成失败: {e}")
        raise