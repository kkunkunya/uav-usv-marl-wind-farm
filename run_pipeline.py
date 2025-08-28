#!/usr/bin/env python3
"""
完整流水线运行脚本
Complete pipeline runner: Execute full data processing pipeline

使用方法：
python run_pipeline.py --all              # 运行完整流水线
python run_pipeline.py --step layers      # 只运行层构建
python run_pipeline.py --step grids       # 只运行栅格构建
python run_pipeline.py --step cache       # 只运行缓存构建
python run_pipeline.py --step validate    # 只运行验证
"""

import sys
import os
import argparse
import subprocess
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str, realtime_output: bool = False) -> bool:
    """
    运行命令并检查结果
    
    Args:
        cmd: 命令列表
        description: 命令描述
        realtime_output: 是否显示实时输出
        
    Returns:
        是否成功执行
    """
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        if realtime_output:
            # 实时输出模式
            logger.info(f"🔄 {description} 执行中...")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    logger.info(f"  📝 {line}")
                    output_lines.append(line)
            
            process.stdout.close()
            return_code = process.wait()
            
            elapsed = time.time() - start_time
            
            if return_code == 0:
                logger.info(f"✓ {description} 完成 (耗时 {elapsed:.1f}s)")
                return True
            else:
                logger.error(f"✗ {description} 失败 (耗时 {elapsed:.1f}s)")
                logger.error(f"返回码: {return_code}")
                return False
        else:
            # 原有的批量输出模式
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ {description} 完成 (耗时 {elapsed:.1f}s)")
            
            # 输出标准输出的关键信息
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:  # 只显示最后5行
                    if line.strip():
                        logger.info(f"  {line}")
            
            return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} 失败 (耗时 {elapsed:.1f}s)")
        logger.error(f"返回码: {e.returncode}")
        
        if e.stdout:
            logger.error("标准输出:")
            for line in e.stdout.strip().split('\n'):
                logger.error(f"  {line}")
        
        if e.stderr:
            logger.error("错误输出:")
            for line in e.stderr.strip().split('\n'):
                logger.error(f"  {line}")
        
        return False
    
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} 异常 (耗时 {elapsed:.1f}s): {e}")
        return False


def check_files_exist(files: list, description: str) -> bool:
    """检查文件是否存在"""
    missing_files = [f for f in files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"{description} - 缺少文件:")
        for f in missing_files:
            logger.error(f"  {f}")
        return False
    
    logger.info(f"✓ {description} - 所有必需文件存在")
    return True


def run_layers_step(config_file: str, xml_file: str) -> bool:
    """运行层构建步骤"""
    logger.info("=" * 50)
    logger.info("步骤 1: 构建地图层")
    logger.info("=" * 50)
    
    # 检查输入文件
    if not check_files_exist([config_file, xml_file], "层构建输入检查"):
        return False
    
    # 运行层构建
    cmd = [
        sys.executable, "scripts/build_layers.py",
        "--config", config_file,
        "--xml", xml_file,
        "--output", "layers.pkl",
        "--visualize"
    ]
    
    return run_command(cmd, "构建地图层", realtime_output=True)


def run_grids_step(config_file: str) -> bool:
    """运行栅格构建步骤"""
    logger.info("=" * 50)
    logger.info("步骤 2: 构建导航栅格")
    logger.info("=" * 50)
    
    # 检查输入文件
    if not check_files_exist([config_file, "layers.pkl"], "栅格构建输入检查"):
        return False
    
    # 运行栅格构建
    cmd = [
        sys.executable, "scripts/build_grids.py",
        "--config", config_file,
        "--layers", "layers.pkl",
        "--output", "grids",
        "--visualize"
    ]
    
    return run_command(cmd, "构建导航栅格", realtime_output=True)


def run_cache_step(config_file: str) -> bool:
    """运行缓存构建步骤"""
    logger.info("=" * 50)
    logger.info("步骤 3: 构建路径缓存")
    logger.info("=" * 50)
    
    # 检查输入文件
    required_files = [config_file, "layers.pkl", "grids"]
    if not check_files_exist(required_files, "缓存构建输入检查"):
        return False
    
    # 检查栅格文件
    grid_files = ["grids/grid_uav.npz", "grids/grid_usv.npz"]
    if not check_files_exist(grid_files, "栅格文件检查"):
        return False
    
    # 运行缓存构建
    cmd = [
        sys.executable, "scripts/build_cache.py",
        "--config", config_file,
        "--layers", "layers.pkl",
        "--grids", "grids",
        "--output", "cache"
    ]
    
    return run_command(cmd, "构建路径缓存", realtime_output=True)


def run_validate_step(config_file: str, xml_file: str) -> bool:
    """运行验证步骤"""
    logger.info("=" * 50)
    logger.info("步骤 4: 环境验证")
    logger.info("=" * 50)
    
    # 检查输入文件
    if not check_files_exist([config_file, xml_file], "验证输入检查"):
        return False
    
    # 运行验证
    cmd = [
        sys.executable, "scripts/validate_env.py",
        "--config", config_file,
        "--xml", xml_file,
        "--test", "all"
    ]
    
    return run_command(cmd, "环境功能验证")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行完整的数据处理流水线')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--xml', default='content.xml', help='XML标注文件路径')
    parser.add_argument('--step', choices=['layers', 'grids', 'cache', 'validate'],
                       help='只运行指定步骤')
    parser.add_argument('--all', action='store_true', help='运行完整流水线')
    parser.add_argument('--skip-validation', action='store_true', help='跳过最终验证')
    
    args = parser.parse_args()
    
    # 如果没有指定步骤且没有指定--all，则默认运行完整流水线
    if not args.step and not args.all:
        args.all = True
    
    # 检查输入文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return 1
    
    if not os.path.exists(args.xml):
        logger.error(f"XML文件不存在: {args.xml}")
        return 1
    
    # 记录开始时间
    pipeline_start = time.time()
    
    logger.info("🚀 海上风电场UAV-USV协同仿真平台构建")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"XML文件: {args.xml}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    
    # 运行流水线步骤
    success = True
    
    if args.all or args.step == 'layers':
        success = success and run_layers_step(args.config, args.xml)
    
    if success and (args.all or args.step == 'grids'):
        success = success and run_grids_step(args.config)
    
    if success and (args.all or args.step == 'cache'):
        success = success and run_cache_step(args.config)
    
    if success and (args.all or args.step == 'validate') and not args.skip_validation:
        success = success and run_validate_step(args.config, args.xml)
    
    # 总结结果
    pipeline_elapsed = time.time() - pipeline_start
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 流水线执行成功！")
        logger.info("生成的文件:")
        
        # 列出生成的主要文件
        output_files = [
            "layers.pkl",
            "layers_preview.png",
            "grids/grid_uav.npz",
            "grids/grid_usv.npz", 
            "grids/grids_visualization.png",
            "cache/path_cache_uav.zst",
            "cache/path_cache_usv.zst"
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                logger.info(f"  ✓ {file_path} ({size:,} bytes)")
            else:
                logger.info(f"  - {file_path} (未生成)")
        
        logger.info("\n📋 后续步骤:")
        logger.info("  1. 检查生成的可视化图片确认数据正确性")
        logger.info("  2. 运行强化学习训练脚本")
        logger.info("  3. 执行基线算法对比实验")
        logger.info("  4. 生成实验报告和分析图表")
        
    else:
        logger.error("❌ 流水线执行失败！")
        logger.error("请检查上述错误信息并修正问题后重新运行")
    
    logger.info(f"总耗时: {pipeline_elapsed:.1f}秒")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())