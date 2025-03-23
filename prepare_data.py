#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import glob
import shutil
import tarfile
import zipfile
import hashlib
from tqdm import tqdm
import requests
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = {
    'aishell': {
        'name': 'AISHELL-1',
        'description': '中文普通话语音数据集，约178小时',
        'url': 'https://openslr.magicdatatech.com/resources/33/data_aishell.tgz',
        'backup_url': 'https://www.openslr.org/resources/33/data_aishell.tgz',
        'size': '15GB',
        'md5': 'f4d4ea3290891c51d139bfe781ca2f19',
        'structure': {
            'transcript': 'data_aishell/transcript/aishell_transcript_v0.8.txt',
            'audio': 'data_aishell/wav'
        }
    }
}

def download_file(url, save_path, chunk_size=1024*1024):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"下载出错: {str(e)}")
        return False

def extract_archive(archive_path, extract_dir):
    """解压压缩文件"""
    logger.info(f"正在解压 {archive_path}")
    
    try:
        if archive_path.endswith('.tgz') or archive_path.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(path=extract_dir)
        else:
            logger.error(f"不支持的压缩格式: {archive_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"解压出错: {str(e)}")
        return False

def fix_transcript_encoding(transcript_path):
    """修复转录文件的编码问题"""
    try:
        # 尝试不同的编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
        content = None
        
        for encoding in encodings:
            try:
                with open(transcript_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content:
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        logger.error(f"无法解码转录文件: {transcript_path}")
        return False
    except Exception as e:
        logger.error(f"修复转录文件编码时出错: {str(e)}")
        return False

def prepare_aishell_data(data_dir, output_dir, force_download=False, keep_archives=False):
    """准备AISHELL数据集"""
    dataset = DATASETS['aishell']
    logger.info(f"准备 {dataset['name']} 数据集")
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载文件
    archive_path = os.path.join(data_dir, os.path.basename(dataset['url']))
    if not os.path.exists(archive_path) or force_download:
        logger.info(f"下载 {dataset['name']} 数据集 ({dataset['size']})")
        if not download_file(dataset['url'], archive_path) and not download_file(dataset['backup_url'], archive_path):
            logger.error("下载失败")
            return False
    
    # 验证MD5
    if dataset['md5'] and hashlib.md5(open(archive_path, 'rb').read()).hexdigest() != dataset['md5']:
        logger.warning("MD5校验失败，文件可能已损坏")
    
    # 解压文件
    extracted_path = os.path.join(data_dir, 'data_aishell')
    if not os.path.exists(extracted_path) or force_download:
        if not extract_archive(archive_path, data_dir):
            return False
    
    # 检查转录文件
    transcript_path = os.path.join(data_dir, dataset['structure']['transcript'])
    if not os.path.exists(transcript_path):
        logger.error(f"转录文件不存在: {transcript_path}")
        return False
    
    # 修复转录文件编码
    fix_transcript_encoding(transcript_path)
    
    # 处理音频文件
    wav_dir = os.path.join(data_dir, dataset['structure']['audio'])
    if not os.path.exists(wav_dir):
        logger.error(f"音频目录不存在: {wav_dir}")
        return False
    
    # 提取音频文件
    extracted_wavs_dir = os.path.join(output_dir, 'extracted_wavs')
    os.makedirs(extracted_wavs_dir, exist_ok=True)
    
    # 创建转录文件目录
    transcript_output_dir = os.path.join(output_dir, 'transcript')
    os.makedirs(transcript_output_dir, exist_ok=True)
    transcript_output_path = os.path.join(transcript_output_dir, os.path.basename(transcript_path))
    
    # 复制转录文件
    shutil.copy2(transcript_path, transcript_output_path)
    
    # 复制音频文件
    logger.info("提取音频文件...")
    file_count = 0
    for root, _, files in os.walk(wav_dir):
        for file in tqdm(files):
            if file.endswith('.wav'):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, wav_dir)
                target_dir = extracted_wavs_dir if rel_path == '.' else os.path.join(extracted_wavs_dir, rel_path)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, file)
                if not os.path.exists(target_path):
                    shutil.copy2(src_path, target_path)
                file_count += 1
    
    # 加载转录文件
    transcripts = {}
    with open(transcript_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) > 1:
                transcripts[parts[0]] = parts[1]
    
    # 删除原始压缩包（可选）
    if not keep_archives and os.path.exists(archive_path):
        os.remove(archive_path)
    
    logger.info(f"{dataset['name']} 数据集准备完成，处理了 {file_count} 个音频文件和 {len(transcripts)} 个转录文本")
    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'audio_files': file_count,
        'transcripts': len(transcripts)
    }

def main():
    parser = argparse.ArgumentParser(description="准备语音识别数据集")
    parser.add_argument("--dataset", type=str, default="aishell", choices=DATASETS.keys(), help="数据集名称")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据下载和解压目录")
    parser.add_argument("--output-dir", type=str, default="./data/data_aishell", help="处理后数据的输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载和处理数据")
    parser.add_argument("--keep-archives", action="store_true", help="保留原始压缩包")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if args.dataset == 'aishell':
        result = prepare_aishell_data(args.data_dir, args.output_dir, args.force, args.keep_archives)
        if result:
            logger.info(f"准备好的数据可用于测试ASR模型: python test_asr.py --data-dir {os.path.join(args.output_dir, 'extracted_wavs')} --transcript {os.path.join(args.output_dir, 'transcript/aishell_transcript_v0.8.txt')} --model iic/SenseVoiceSmall --device cuda:0 --samples 10")
    else:
        logger.error(f"暂不支持的数据集: {args.dataset}")

if __name__ == "__main__":
    main() 