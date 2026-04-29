import json
import os
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

# 环境变量：屏蔽代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

# ================= 配置区 =================
DEBUG_MODE = False   # True: 只运行3条；False: 全量运行
MODEL_NAME = 'llama3.2:1b'
NUM_THREADS = 4      # 并发数（CPU运行建议3-5，GPU建议8-12）
# ==========================================

class SentimentRefiner:
    def __init__(self, model_name=MODEL_NAME):
        self.client = Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        # 极性映射表
        self.pos_labels = ['happy', 'joyful', 'love', 'happiness', 'positive']
        self.neg_labels = ['sad', 'angry', 'fear', 'disgust', 'sadness', 'negative', 'fearful']

    def get_polarity(self, label):
        if not label: return "neu"
        l = label.lower()
        if any(x in l for x in self.pos_labels): return "pos"
        if any(x in l for x in self.neg_labels): return "neg"
        return "neu"

    def get_llm_text_sentiment(self, text):
        """分析文本情感，包含完整的错误捕获"""
        prompt = f"""Analyze the sentiment of the text. Return ONLY one word from this list: [happy, sad, angry, fear, disgust, neutral].
Text: "{text}"
Label:"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": 5, "temperature": 0}
            )
            result = response['response'].strip().lower().replace(".", "").replace("'", "")
            allowed = ['happy', 'sad', 'angry', 'fear', 'disgust', 'neutral']
            for word in allowed:
                if word in result: return word
            return "neutral"
        except Exception as e:
            # 按照您的要求保留并打印详细错误
            print(f"\n[Error] Ollama调用异常 | 文本: {text[:20]}... | Info: {e}")
            return "ERROR_LLM"

    def process_session(self, line):
        """处理单条 Session 数据"""
        try:
            data = json.loads(line)
            last_sticker_turn_idx = -1
            
            for i, turn in enumerate(data['dialogue']):
                # 1. 文本情感识别
                text_sent = self.get_llm_text_sentiment(turn['text'])
                turn['text_sentiment'] = text_sent
                
                # 2. 处理表情包和冲突
                if turn.get('sticker_path'):
                    # 关键修正：从 emotion_label 提取原始标签
                    s_emo_gt = turn.get('emotion_label', 'neutral') 
                    turn['sticker_emotion'] = s_emo_gt
                    
                    if "ERROR" not in text_sent:
                        t_polar = self.get_polarity(text_sent)
                        s_polar = self.get_polarity(s_emo_gt)
                        if t_polar != "neu" and s_polar != "neu" and t_polar != s_polar:
                            turn['is_conflict'] = 1
                        else:
                            turn['is_conflict'] = 0
                    else:
                        turn['is_conflict'] = -2 # LLM 判定失败
                    
                    last_sticker_turn_idx = i
                else:
                    turn['sticker_emotion'] = None
                    turn['is_conflict'] = -1
                
                # 清理旧字段
                if 'emotion_label' in turn:
                    del turn['emotion_label']

            # 3. 更新顶层 Session 标志
            if last_sticker_turn_idx != -1:
                data['is_session_conflict'] = data['dialogue'][last_sticker_turn_idx]['is_conflict']
            else:
                data['is_session_conflict'] = -1
            
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            print(f"\n[Fatal Error] Session处理失败 | Info: {e}")
            return None

    def refine_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return

        # --- 断点续传核心逻辑 ---
        finished_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f_check:
                finished_count = sum(1 for _ in f_check)
        
        print(f"📊 检查点: 已处理 {finished_count} 条，将从第 {finished_count + 1} 条开始。")

        with open(input_file, 'r', encoding='utf-8') as f_in:
            all_lines = f_in.readlines()

        if DEBUG_MODE:
            all_lines = all_lines[:3]
            print("⚠️ DEBUG 模式：仅处理前3条数据")
        
        # 过滤掉已处理的行
        remaining_lines = all_lines[finished_count:]
        if not remaining_lines:
            print(f"✅ 文件 {input_file} 已全部处理完成。")
            return

        # --- 实时保存逻辑 ---
        # 使用 'a' 模式追加写入
        with open(output_file, 'a', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                # 提交任务，并保持顺序
                futures = {executor.submit(self.process_session, line): line for line in remaining_lines}
                
                pbar = tqdm(total=len(all_lines), initial=finished_count, desc=f"Processing {input_file}")
                
                # as_completed 虽然不保证顺序，但对于 JSONL 这种行式存储没关系
                # 如果您必须保持原始顺序，可以改用 executor.map
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        f_out.write(result + '\n')
                        f_out.flush() # 强制写入磁盘，防止程序崩溃丢失缓存
                    pbar.update(1)
                pbar.close()

if __name__ == "__main__":
    refiner = SentimentRefiner()
    
    # 任务清单
    tasks = [
        # ('StickerConv_Cleaned_Vail.jsonl', 'StickerConv_Refined_Vail.jsonl'),
        # ('StickerConv_Cleaned_Test.jsonl', 'StickerConv_Refined_Test.jsonl'),
        ('StickerConv_Cleaned_Train.jsonl', 'StickerConv_Refined_Train.jsonl')
    ]

    for in_f, out_f in tasks:
        refiner.refine_file(in_f, out_f)