import argparse
import os
import time
import pandas as pd
import re
from tqdm import tqdm
import glob
from funasr import AutoModel

def main():
    parser = argparse.ArgumentParser(description="测试ASR模型")
    parser.add_argument("--model", type=str, default="iic/SenseVoiceSmall", help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--data-dir", type=str, default="./data/data_aishell/extracted_wavs", help="解压后的音频目录")
    parser.add_argument("--transcript", type=str, default="./data/data_aishell/transcript/aishell_transcript_v0.8.txt", help="转录文件")
    parser.add_argument("--samples", type=int, default=0, help="测试样本数量，0表示全部")
    parser.add_argument("--output", type=str, default="asr_results.csv", help="结果输出文件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = AutoModel(
        model=args.model,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device,
    )
    
    # 加载数据
    audio_files = glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True)
    print(f"找到{len(audio_files)}个音频文件")
    
    transcripts = {}
    with open(args.transcript, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) > 1:
                transcripts[parts[0]] = parts[1]
    print(f"加载了{len(transcripts)}个转录文本")
    
    # 创建测试数据集
    test_dataset = []
    for audio_file in audio_files:
        file_id = os.path.basename(audio_file).split('.')[0]
        if file_id in transcripts:
            test_dataset.append((audio_file, transcripts[file_id]))
    
    print(f"创建了{len(test_dataset)}个测试样本")
    
    # 限制样本数量
    if args.samples > 0:
        test_dataset = test_dataset[:args.samples]
        print(f"限制测试样本数量为: {args.samples}")
    
    # 执行测试
    results = []
    stats = {
        "total_chars": 0,
        "total_errors": 0,
        "total_process_time": 0,
        "total_audio_duration": 0,
        "total_chars_in_hyp": 0
    }
    
    print(f"开始测试，共{len(test_dataset)}个样本...")
    for audio_file, reference_text in tqdm(test_dataset):
        try:
            # 获取音频时长
            try:
                import librosa
                audio_duration = librosa.get_duration(path=audio_file)
            except:
                audio_duration = 0
            
            # 识别音频
            start_time = time.time()
            res = model.generate(input=audio_file, language="zh", use_itn=True)
            process_time = time.time() - start_time
            
            hypothesis_text = res[0]["text"]
            
            # 处理文本
            ref_cleaned = preprocess_text(reference_text)
            merged_result = merge_asr_segments(hypothesis_text)
            hyp_cleaned = preprocess_text(merged_result)
            
            ref_norm = normalize_numbers(ref_cleaned)
            hyp_norm = normalize_numbers(hyp_cleaned)
            
            # 计算字错率
            ref_len, error_chars, cer = calculate_char_errors(ref_norm, hyp_norm)
            
            # 计算识别速度
            hyp_len = len(hyp_cleaned)
            chars_per_second = hyp_len / process_time if process_time > 0 else 0
            rtf = process_time / audio_duration if audio_duration > 0 else 0
            
            # 累计统计
            stats["total_chars"] += ref_len
            stats["total_errors"] += error_chars
            stats["total_process_time"] += process_time
            stats["total_audio_duration"] += audio_duration
            stats["total_chars_in_hyp"] += hyp_len
            
            # 调试信息
            if args.debug:
                print(f"\n文件: {os.path.basename(audio_file)}")
                print(f"参考原文: {reference_text}")
                print(f"清理后参考: {ref_cleaned}")
                print(f"识别原文: {hypothesis_text}")
                print(f"合并后识别: {merged_result}")
                print(f"清理后识别: {hyp_cleaned}")
                print(f"标准化参考: {ref_norm}")
                print(f"标准化识别: {hyp_norm}")
                print(f"参考字数: {ref_len}, 错误字数: {error_chars}")
                print(f"识别字数: {hyp_len}, 处理时间: {process_time:.2f}秒")
                print(f"识别速度: {chars_per_second:.2f}字/秒")
                print(f"音频时长: {audio_duration:.2f}秒, RTF: {rtf:.4f}")
                print(f"字错率: {cer:.4f}")
            
            # 结果记录
            results.append({
                "文件名": os.path.basename(audio_file),
                "参考文本": reference_text,
                "识别结果": hypothesis_text,
                "清理后参考": ref_cleaned,
                "清理后识别": hyp_cleaned,
                "合并后识别": merged_result,
                "标准化参考": ref_norm,
                "标准化识别": hyp_norm,
                "参考字数": ref_len,
                "识别字数": hyp_len,
                "错误字数": error_chars,
                "字错率": cer,
                "音频时长(秒)": audio_duration,
                "处理时间(秒)": process_time,
                "识别速度(字/秒)": chars_per_second,
                "实时率(RTF)": rtf
            })
            
            # 打印高错误率样本
            if cer > 0.3 and len(ref_norm) > 0:
                print(f"\n高错误率样本 - CER: {cer:.4f}")
                print(f"  参考文本: {reference_text}")
                print(f"  识别结果: {hypothesis_text}")
                
                # 分析错误类型
                try:
                    import Levenshtein
                    operations = Levenshtein.opcodes(ref_norm, hyp_norm)
                    for tag, i1, i2, j1, j2 in operations:
                        if tag != 'equal':
                            print(f"  错误类型: {tag}, 参考[{i1}:{i2}]=\"{ref_norm[i1:i2]}\", 识别[{j1}:{j2}]=\"{hyp_norm[j1:j2]}\"")
                except ImportError:
                    pass
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    
    # 计算整体指标
    total_chars = stats["total_chars"]
    total_errors = stats["total_errors"]
    overall_cer = total_errors / total_chars if total_chars > 0 else 1.0
    overall_chars_per_second = stats["total_chars_in_hyp"] / stats["total_process_time"] if stats["total_process_time"] > 0 else 0
    overall_rtf = stats["total_process_time"] / stats["total_audio_duration"] if stats["total_audio_duration"] > 0 else 0
    
    # 输出汇总信息
    print(f"\n测试完成! 总字符数: {total_chars}, 总错误字数: {total_errors}, 整体字错率: {overall_cer:.4f}")
    print(f"总识别字数: {stats['total_chars_in_hyp']}, 总处理时间: {stats['total_process_time']:.2f}秒, 总音频时长: {stats['total_audio_duration']:.2f}秒")
    print(f"平均识别速度: {overall_chars_per_second:.2f}字/秒, 平均实时率(RTF): {overall_rtf:.4f}")
    if overall_rtf > 0:
        print(f"处理速度: {1/overall_rtf:.2f}倍速")
    print(f"测试样本数: {len(results)}, 结果已保存到: {args.output}")
    
    # 将汇总信息添加到CSV
    with open(args.output, 'a', encoding='utf-8-sig') as f:
        f.write(f"\n总字符数,{total_chars}\n")
        f.write(f"总错误字数,{total_errors}\n")
        f.write(f"整体字错率,{overall_cer:.4f}\n")
        f.write(f"总识别字数,{stats['total_chars_in_hyp']}\n")
        f.write(f"总处理时间(秒),{stats['total_process_time']:.2f}\n")
        f.write(f"总音频时长(秒),{stats['total_audio_duration']:.2f}\n")
        f.write(f"平均识别速度(字/秒),{overall_chars_per_second:.2f}\n")
        f.write(f"平均实时率(RTF),{overall_rtf:.4f}\n")
        if overall_rtf > 0:
            f.write(f"处理速度(倍速),{1/overall_rtf:.2f}\n")
        f.write(f"测试样本数,{len(results)}\n")

def preprocess_text(text):
    """预处理文本，只保留中文字符"""
    return re.sub(r'[^\u4e00-\u9fa5]', '', text)

def merge_asr_segments(text):
    """合并ASR输出中因标记而分段的文本片段"""
    segments = re.split(r'<\|.*?\|>', text)
    cleaned_segments = [re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', segment) for segment in segments]
    return ''.join([segment for segment in cleaned_segments if segment])

def normalize_numbers(text):
    """标准化文本中的数字表示，将阿拉伯数字转为中文数字"""
    digit_map = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四', 
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    
    result = ''
    i = 0
    while i < len(text):
        if text[i].isdigit():
            num_str = ''
            while i < len(text) and text[i].isdigit():
                num_str += text[i]
                i += 1
            
            if num_str == '10':
                result += '十'
            elif num_str == '100':
                result += '百'
            elif num_str == '1000':
                result += '千'
            elif num_str == '10000':
                result += '万'
            else:
                for digit in num_str:
                    result += digit_map.get(digit, digit)
        else:
            result += text[i]
            i += 1
    
    return result

def calculate_char_errors(ref_text, hyp_text):
    """计算中文字错误数"""
    if not ref_text:
        return 0, 0, 1.0
    
    try:
        import Levenshtein
        distance = Levenshtein.distance(ref_text, hyp_text)
        ref_len = len(ref_text)
        cer = distance / ref_len if ref_len > 0 else 1.0
        return ref_len, distance, cer
        
    except ImportError:
        # 手动计算编辑距离
        ref_len = len(ref_text)
        if ref_text == hyp_text:
            return ref_len, 0, 0.0
        
        m, n = len(ref_text), len(hyp_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_text[i-1] == hyp_text[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )
        
        distance = dp[m][n]
        return ref_len, distance, distance / ref_len 

if __name__ == "__main__":
    main() 