import json
import os
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

# 环境变量：屏蔽代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

# ================= 配置区 =================
DEBUG_MODE = False  # True: 只运行前3条；False: 全量运行
MODEL_NAME = 'llama3.2:1b'
NUM_THREADS = 3 
# ==========================================

class SentimentRefiner:
    def __init__(self, model_name=MODEL_NAME, num_threads=NUM_THREADS):
        self.client = Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        self.num_threads = num_threads
        
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
        """仅分析文本情感"""
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
        except Exception:
            return "ERROR_LLM"

    def process_session(self, line, pbar):
        try:
            data = json.loads(line)
            
            # 记录最后一条有表情包的轮次索引，用于更新顶层 is_session_conflict
            last_sticker_turn_idx = -1
            
            for i, turn in enumerate(data['dialogue']):
                # 1. 无论有没有表情包，文本情感总是要识别的
                text_sent = self.get_llm_text_sentiment(turn['text'])
                turn['text_sentiment'] = text_sent
                
                # 2. 处理表情包相关的字段
                if turn.get('sticker_path'):
                    # 命名修正：改回具体的 sticker_emotion
                    # 这里的 emotion_label 是你清理脚本里从原始数据提取的 origin_anno
                    s_emo_gt = turn.get('emotion_label', 'neutral') 
                    turn['sticker_emotion'] = s_emo_gt
                    
                    # 判定冲突
                    if "ERROR" not in text_sent:
                        t_polar = self.get_polarity(text_sent)
                        s_polar = self.get_polarity(s_emo_gt)
                        if t_polar != "neu" and s_polar != "neu" and t_polar != s_polar:
                            turn['is_conflict'] = 1
                        else:
                            turn['is_conflict'] = 0
                    else:
                        turn['is_conflict'] = -2 # 代表 LLM 出错
                    
                    last_sticker_turn_idx = i
                else:
                    # 【核心修正】没有表情包的轮次
                    turn['sticker_emotion'] = None
                    turn['is_conflict'] = -1 # 统一置为 -1，表示不适用
                
                # 清理掉之前可能存在的歧义字段（如果有的话）
                if 'emotion_label' in turn:
                    del turn['emotion_label']
                
                pbar.update(1)

            # 3. 更新顶层 Session 标志
            # 必须基于真正带有表情包的最后一轮
            if last_sticker_turn_idx != -1:
                data['is_session_conflict'] = data['dialogue'][last_sticker_turn_idx]['is_conflict']
            else:
                data['is_session_conflict'] = -1
            
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return None

    def refine_file(self, input_file, output_file):
        if not os.path.exists(input_file): return
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if DEBUG_MODE:
            print(f"⚠️ DEBUG MODE: Only processing 3 sessions.")
            lines = lines[:3]

        total_turns = sum(len(json.loads(l)['dialogue']) for l in lines)
        pbar = tqdm(total=total_turns, desc="Processing Turns")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_session, line, pbar) for line in lines]
            for future in as_completed(futures):
                res = future.result()
                if res: results.append(res)
        pbar.close()

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for r in results: f_out.write(r + '\n')
        print(f"✅ Refinement finished: {output_file}")

if __name__ == "__main__":
    refiner = SentimentRefiner()
    tasks = [
        ('StickerConv_Cleaned_Vail.jsonl', 'StickerConv_Refined_Vail.jsonl'),
        ('StickerConv_Cleaned_Test.jsonl', 'StickerConv_Refined_Test.jsonl'),
        ('StickerConv_Cleaned_Train.jsonl', 'StickerConv_Refined_Train.jsonl')
    ]
    for in_f, out_f in tasks:
        refiner.refine_file(in_f, out_f)