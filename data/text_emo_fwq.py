import json
import os
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 环境变量：防止代理干扰
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'


class SentimentRefiner:
    def __init__(self, model_name='llama3.2:3b', num_threads=8):
        # 连接到服务器本地的 Ollama
        self.client = Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        self.num_threads = num_threads

        # 极性映射
        self.pos_emotions = ['joyful', 'happiness', 'joy', 'love', 'happy', 'positive']
        self.neg_emotions = ['fear', 'sad', 'angry', 'disgust', 'sadness', 'negative', 'fearful']

    def get_llm_sentiment(self, text):
        """调用 Ollama，极简指令模式"""
        prompt = f"Categorize sentiment: '{text}'. Return ONLY 'pos', 'neg', or 'neu'."
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": 5,
                    "temperature": 0
                }
            )
            result = response['response'].strip().lower()
            if 'pos' in result: return 'pos'
            if 'neg' in result: return 'neg'
            if 'neu' in result: return 'neu'
            return "ERROR_FORMAT"
        except Exception as e:
            print(f"\n[Error] Text: {text[:20]}... | Info: {e}")
            return "ERROR_LLM"

    def get_sticker_polarity(self, sticker_emo):
        if not sticker_emo: return "neu"
        emo = sticker_emo.lower()
        if any(e in emo for e in self.pos_emotions): return "pos"
        if any(e in emo for e in self.neg_emotions): return "neg"
        return "neu"

    def process_session(self, line):
        """处理单条 Session（JSONL 的一行）"""
        try:
            data = json.loads(line)
            # data = data[0:2]
            for turn in data['dialogue']:
                # 初始状态设为 -1
                turn['is_conflict'] = -1

                # 获取 LLM 判定结果
                text_sent = self.get_llm_sentiment(turn['text'])
                turn['text_sentiment'] = text_sent

                # --- 修复后的判断逻辑 ---
                if "ERROR" not in text_sent:
                    if turn.get('sticker_path'):
                        s_polar = self.get_sticker_polarity(turn.get('emotion_label', ''))
                        t_polar = text_sent

                        # 冲突：两个极性明确且相反
                        if t_polar != "neu" and s_polar != "neu" and t_polar != s_polar:
                            turn['is_conflict'] = 1
                        else:
                            turn['is_conflict'] = 0
                    else:
                        # 纯文本不参与音文冲突判定
                        turn['is_conflict'] = 0
                else:
                    # 如果报错，is_conflict 保持 -1
                    pass

            # 更新 Session 顶层冲突标志
            final_conflict = 0
            last_turn_with_sticker = None
            for turn in reversed(data['dialogue']):
                if turn.get('sticker_path'):
                    last_turn_with_sticker = turn
                    break

            if last_turn_with_sticker:
                final_conflict = last_turn_with_sticker['is_conflict']

            data['is_session_conflict'] = final_conflict
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return None

    def refine_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found.")
            return

        print(f"🚀 Processing {input_file} (Threads: {self.num_threads})")

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交任务
            future_to_index = {executor.submit(self.process_session, line): i for i, line in enumerate(lines)}

            # 使用 tqdm 进度条
            for future in tqdm(as_completed(future_to_index), total=len(lines), desc="Refining"):
                res = future.result()
                if res:
                    results.append(res)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for r in results:
                f_out.write(r + '\n')

        print(f"✅ Saved refined data to: {output_file}")


if __name__ == "__main__":
    # 请根据显存大小调整 num_threads
    # 8GB 显存建议 4-6, 24GB 以上建议 10-15
    refiner = SentimentRefiner(model_name='llama3.2:3b', num_threads=10)

    # 任务列表
    tasks = [
        # ('StickerConv_Cleaned_Test.jsonl', 'StickerConv_Refined_Test.jsonl'),
        # ('StickerConv_Cleaned_Train.jsonl', 'StickerConv_Refined_Train.jsonl'),
        ('StickerConv_Cleaned_Vail.jsonl', 'StickerConv_Refined_Vail.jsonl')
    ]

    for in_f, out_f in tasks:
        refiner.refine_file(in_f, out_f)