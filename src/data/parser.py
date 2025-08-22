"""
XML解析器：PASCAL VOC格式标注文件解析
XML Parser: PASCAL VOC annotation file parser

功能：
1. 解析content.xml中的对象标注信息
2. 将像素坐标转换为世界坐标系
3. 分类提取风机、充电站、岛屿、UAV障碍物
4. 输出结构化的对象数据
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
from shapely.geometry import box, Point
from dataclasses import dataclass
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """边界框数据类"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """返回边界框中心点"""
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)
    
    @property
    def area(self) -> float:
        """返回边界框面积"""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


@dataclass
class DetectedObject:
    """检测对象数据类"""
    id: str                          # 对象ID
    class_name: str                  # 类别名称
    bbox_px: BoundingBox            # 像素坐标边界框
    center_px: Tuple[float, float]   # 像素坐标中心点
    center_world: Tuple[float, float] # 世界坐标中心点
    polygon_world: object           # 世界坐标多边形（Shapely对象）
    confidence: float = 1.0         # 置信度
    attributes: Dict = None         # 额外属性


class CoordinateTransformer:
    """坐标变换器：像素坐标 ↔ 世界坐标"""
    
    def __init__(self, image_width: int, image_height: int, gsd_m_per_px: float):
        """
        初始化坐标变换器
        
        Args:
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
            gsd_m_per_px: 地面分辨率（米/像素）
        """
        self.width = image_width
        self.height = image_height
        self.gsd = gsd_m_per_px
        
        # 图像中心作为世界坐标原点
        self.u0 = image_width / 2
        self.v0 = image_height / 2
        
        logger.info(f"坐标变换器初始化: {image_width}x{image_height}, GSD={gsd_m_per_px}m/px")
    
    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        像素坐标转世界坐标
        
        Args:
            u, v: 像素坐标
            
        Returns:
            x, y: 世界坐标（米）
        """
        x = (u - self.u0) * self.gsd
        y = (self.v0 - v) * self.gsd  # 注意Y轴翻转
        return (x, y)
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """
        世界坐标转像素坐标
        
        Args:
            x, y: 世界坐标（米）
            
        Returns:
            u, v: 像素坐标
        """
        u = x / self.gsd + self.u0
        v = self.v0 - y / self.gsd  # 注意Y轴翻转
        return (u, v)
    
    def bbox_to_world_polygon(self, bbox: BoundingBox) -> object:
        """
        将像素边界框转换为世界坐标多边形
        
        Args:
            bbox: 像素坐标边界框
            
        Returns:
            Shapely多边形对象
        """
        # 获取四个角点的世界坐标
        corners_px = [
            (bbox.xmin, bbox.ymin),  # 左上
            (bbox.xmax, bbox.ymin),  # 右上
            (bbox.xmax, bbox.ymax),  # 右下
            (bbox.xmin, bbox.ymax)   # 左下
        ]
        
        corners_world = [self.pixel_to_world(u, v) for u, v in corners_px]
        
        # 创建多边形
        return box(
            min(x for x, y in corners_world),
            min(y for x, y in corners_world),
            max(x for x, y in corners_world),
            max(y for x, y in corners_world)
        )


class VOCParser:
    """PASCAL VOC格式XML解析器"""
    
    # 类别名称映射表（处理可能的命名变体）
    CLASS_ALIASES = {
        'fan': ['fan', 'wind_turbine', 'turbine'],
        'charge station': ['charge station', 'charging_station', 'charger'],
        'island': ['island', 'land', 'obstacle'],
        'UAV obstacle': ['UAV obstacle', 'uav_obstacle', 'no_fly_zone', 'nfz']
    }
    
    def __init__(self, xml_path: str, gsd_m_per_px: float):
        """
        初始化VOC解析器
        
        Args:
            xml_path: XML文件路径
            gsd_m_per_px: 地面分辨率
        """
        self.xml_path = Path(xml_path)
        self.gsd = gsd_m_per_px
        self.transformer = None
        self.objects = []
        
        # 验证文件存在
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML文件不存在: {xml_path}")
    
    def _normalize_class_name(self, class_name: str) -> str:
        """标准化类别名称"""
        class_name = class_name.strip().lower()
        
        for standard_name, aliases in self.CLASS_ALIASES.items():
            if class_name in [alias.lower() for alias in aliases]:
                return standard_name
        
        logger.warning(f"未知类别名称: {class_name}")
        return class_name
    
    def _parse_bounding_box(self, obj_elem) -> BoundingBox:
        """解析边界框"""
        bbox_elem = obj_elem.find('bndbox')
        if bbox_elem is None:
            raise ValueError("边界框信息缺失")
        
        try:
            xmin = float(bbox_elem.find('xmin').text)
            ymin = float(bbox_elem.find('ymin').text)
            xmax = float(bbox_elem.find('xmax').text)
            ymax = float(bbox_elem.find('ymax').text)
            
            # 验证边界框有效性
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(f"无效边界框: ({xmin},{ymin},{xmax},{ymax})")
            
            return BoundingBox(xmin, ymin, xmax, ymax)
            
        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError(f"边界框解析失败: {e}")
    
    def parse(self) -> List[DetectedObject]:
        """
        解析XML文件
        
        Returns:
            检测对象列表
        """
        logger.info(f"开始解析XML文件: {self.xml_path}")
        
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"XML文件格式错误: {e}")
        
        # 获取图像尺寸
        size_elem = root.find('size')
        if size_elem is None:
            raise ValueError("图像尺寸信息缺失")
        
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        
        # 初始化坐标变换器
        self.transformer = CoordinateTransformer(width, height, self.gsd)
        
        # 解析所有对象
        objects = []
        object_elems = root.findall('object')
        
        for i, obj_elem in enumerate(object_elems):
            try:
                # 获取类别名称
                name_elem = obj_elem.find('name')
                if name_elem is None:
                    logger.warning(f"对象{i}缺少类别名称，跳过")
                    continue
                
                class_name = self._normalize_class_name(name_elem.text)
                
                # 解析边界框
                bbox_px = self._parse_bounding_box(obj_elem)
                
                # 计算中心点
                center_px = bbox_px.center
                center_world = self.transformer.pixel_to_world(*center_px)
                
                # 生成世界坐标多边形
                polygon_world = self.transformer.bbox_to_world_polygon(bbox_px)
                
                # 创建对象
                obj = DetectedObject(
                    id=f"{class_name}_{i}",
                    class_name=class_name,
                    bbox_px=bbox_px,
                    center_px=center_px,
                    center_world=center_world,
                    polygon_world=polygon_world
                )
                
                objects.append(obj)
                
            except Exception as e:
                logger.error(f"解析对象{i}失败: {e}")
                continue
        
        self.objects = objects
        
        # 统计信息
        class_counts = {}
        for obj in objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        logger.info(f"解析完成，共{len(objects)}个对象:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
        
        return objects
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将解析结果转换为DataFrame
        
        Returns:
            包含所有对象信息的DataFrame
        """
        if not self.objects:
            return pd.DataFrame()
        
        data = []
        for obj in self.objects:
            data.append({
                'id': obj.id,
                'class': obj.class_name,
                'bbox_xmin': obj.bbox_px.xmin,
                'bbox_ymin': obj.bbox_px.ymin,
                'bbox_xmax': obj.bbox_px.xmax,
                'bbox_ymax': obj.bbox_px.ymax,
                'center_px_u': obj.center_px[0],
                'center_px_v': obj.center_px[1],
                'center_world_x': obj.center_world[0],
                'center_world_y': obj.center_world[1],
                'bbox_area_px': obj.bbox_px.area,
                'confidence': obj.confidence
            })
        
        return pd.DataFrame(data)
    
    def save_to_parquet(self, output_path: str) -> None:
        """
        保存解析结果为Parquet文件
        
        Args:
            output_path: 输出文件路径
        """
        df = self.to_dataframe()
        df.to_parquet(output_path, index=False)
        logger.info(f"解析结果已保存到: {output_path}")
    
    def get_objects_by_class(self, class_name: str) -> List[DetectedObject]:
        """
        按类别获取对象
        
        Args:
            class_name: 类别名称
            
        Returns:
            指定类别的对象列表
        """
        return [obj for obj in self.objects if obj.class_name == class_name]
    
    def get_file_hash(self) -> str:
        """
        计算XML文件的SHA-256哈希值
        
        Returns:
            文件哈希值
        """
        with open(self.xml_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()


def parse_xml_to_objects(xml_path: str, config_path: str) -> List[DetectedObject]:
    """
    便捷函数：解析XML文件到对象列表
    
    Args:
        xml_path: XML文件路径
        config_path: 配置文件路径
        
    Returns:
        检测对象列表
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    gsd = config['map']['gsd_m_per_px']
    
    # 解析XML
    parser = VOCParser(xml_path, gsd)
    objects = parser.parse()
    
    return objects


if __name__ == "__main__":
    # 测试代码
    xml_path = "content.xml"
    config_path = "config.yaml"
    
    try:
        # 解析XML文件
        parser = VOCParser(xml_path, gsd_m_per_px=5.0)
        objects = parser.parse()
        
        # 保存结果
        parser.save_to_parquet("objects.parquet")
        
        # 打印统计信息
        print("\n=== 解析结果统计 ===")
        df = parser.to_dataframe()
        print(f"总对象数: {len(df)}")
        print("\n各类别统计:")
        print(df['class'].value_counts())
        
        print(f"\nXML文件哈希: {parser.get_file_hash()}")
        
    except Exception as e:
        logger.error(f"解析失败: {e}")
        raise