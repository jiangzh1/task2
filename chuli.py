import pandas as pd
import json
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 备注：
# 需要放置在SER_Dataset同目录下进行，否则需要调整图像存储目录的表述

# 确保 VADER 资源可用
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class Spch2StiProcessor:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        # 情感映射
        self.pos_list = ['joyful', 'happiness', 'joy', 'love', 'happy', 'positive']
        self.neg_list = ['fear', 'sad', 'angry', 'disgust', 'sadness', 'negative', 'fearful']

    def get_text_sentiment(self, text):
        """计算文本字面情感极性"""
        score = self.sid.polarity_scores(text)['compound']
        if score > 0.05: return "pos"
        if score < -0.05: return "neg"
        return "neu"

    def get_sticker_sentiment(self, sticker_emo):
        """计算表情包标注情感极性"""
        if not sticker_emo: return "neu"
        emo = sticker_emo.lower()
        if any(e in emo for e in self.pos_list): return "pos"
        if any(e in emo for e in self.neg_list): return "neg"
        return "neu"

    def get_relative_path(self, raw_path):
        """将绝对或错误的路径转为统一的相对路径: ./SER_Dataset/Images/..."""
        if not raw_path: return None
        # 提取核心路径部分 (从 Images 开始)
        target_str = "Images/"
        idx = raw_path.find(target_str)
        if idx != -1:
            path_part = raw_path[idx + len(target_str):]
            return f"./SER_Dataset/Images/{path_part}".replace('\\', '/')
        return raw_path.replace('\\', '/')

    def process(self, input_parquet, output_jsonl):
        df = pd.read_parquet(input_parquet)
        processed_count = 0

        with open(output_jsonl, 'w', encoding='utf-8') as f_out:
            for idx, row in df.iterrows():
                turns = row['conversations']

                # 1. 寻找最后一个带图的索引进行截断
                last_img_idx = -1
                for i in range(len(turns) - 1, -1, -1):
                    if turns[i].get('image') is not None:
                        last_img_idx = i
                        break

                # 2. 基础清洗条件：有图且有上下文
                if last_img_idx < 1:
                    continue

                valid_turns = turns[:last_img_idx + 1]
                dialogue_data = []

                for i, turn in enumerate(valid_turns):
                    content = turn['content']
                    sticker = turn.get('image')

                    # 判定单轮文本情感
                    t_polar = self.get_text_sentiment(content)

                    is_conflict = 0
                    sticker_emo_label = "neutral"
                    if sticker:
                        sticker_emo_label = sticker.get('origin_anno', 'neutral')
                        s_polar = self.get_sticker_sentiment(sticker_emo_label)
                        # 判定冲突
                        if t_polar != "neu" and s_polar != "neu" and t_polar != s_polar:
                            is_conflict = 1

                    # 轮次数据封装
                    dialogue_data.append({
                        "turn": i,  # 修正 turn ID
                        "role": turn['role'],
                        "text": content,
                        "text_sentiment": t_polar,  # 保留文本情感字段
                        "sticker_path": self.get_relative_path(sticker.get('image')) if sticker else None,
                        "is_conflict": is_conflict,
                        "emotion_label": sticker_emo_label,
                        "description": sticker.get('description') if sticker else None
                    })

                # 3. Session 级别的冲突字段：由最后一条数据决定
                session_conflict = dialogue_data[-1]['is_conflict']

                # 4. 封装输出
                session_output = {
                    "session_id": f"SESS_{idx:05d}",
                    "is_session_conflict": session_conflict,  # 新增整条数据冲突字段
                    "user_persona": row.get('user_persona', ""),
                    "user_status": row.get('user_status', ""),
                    "dialogue": dialogue_data
                }

                f_out.write(json.dumps(session_output, ensure_ascii=False) + '\n')
                processed_count += 1

        print(f"处理完成！有效样本: {processed_count}")


# --- 主程序运行 ---
if __name__ == "__main__":
    # 配置你的实际路径
    # PARQUET_FILE = r'train-00000-of-00001.parquet'
    # OUTPUT_FILE = r'StickerConv_Cleaned_Train.jsonl'
    PARQUET_FILE = r'test-00000-of-00001.parquet'
    OUTPUT_FILE = r'StickerConv_Cleaned_Test.jsonl'
    # PARQUET_FILE = r'validation-00000-of-00001.parquet'
    # OUTPUT_FILE = r'StickerConv_Cleaned_Vail.jsonl'

    # 这里的 STICKER_BASE 指向 SER30K 包含 'Images' 的文件夹
    # 比如 E:/SER30K/Images
    # STICKER_BASE = r'E:\python_files\acl\数据集\SER30K\ours\SER_Dataset\Images'

    processor = Spch2StiProcessor()
    processor.process(PARQUET_FILE, OUTPUT_FILE)