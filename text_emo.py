import json
import os
import os
from tqdm import tqdm
from ollama import Client  # 使用 Client 类以便更精确地控制连接

# 强制禁用所有代理干扰
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'


class SentimentRefiner:
    def __init__(self, model_name='llama3.1:8b'):
        # 显式指定 127.0.0.1 避开 localhost 解析可能导致的 502
        self.client = Client(host='http://127.0.0.1:11434')
        self.model_name = model_name
        self.pos_emotions = ['joyful', 'happiness', 'joy', 'love', 'happy', 'positive']
        self.neg_emotions = ['fear', 'sad', 'angry', 'disgust', 'sadness', 'negative', 'fearful']

    def get_llm_sentiment(self, text):
        """调用 Ollama，失败返回 ERROR_LLM"""
        # prompt = f"""Analyze the sentiment of the following text.
        # Return ONLY one word: 'pos', 'neg', or 'neu'.
        # - 'pos': happy, joyful, grateful, loving, etc.
        # - 'neg': sad, angry, anxious, depressed, etc.
        # - 'neu': neutral, factual, or informative.
        #
        # Text: "{text}"
        # Sentiment:"""
        prompt = f"Analyze sentiment of text. Return ONLY 'pos', 'neg', or 'neu'.\nText: \"{text}\"\nSentiment:"
        try:
            # 增加 options 参数延长超时时间，防止 502
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": 10, "temperature": 0}
            )
            result = response['response'].strip().lower()
            if 'pos' in result: return 'pos'
            if 'neg' in result: return 'neg'
            if 'neu' in result: return 'neu'
            return "ERROR_FORMAT"  # 如果 LLM 返回了奇怪的内容
        except Exception as e:
            # 不再返回 neg/neu，直接返回错误标识
            # 如果报错，打印具体的错误信息，方便排查
            print(f"\n[Error] Text: {text[:20]}... | Info: {e}")
            return f"ERROR_LLM"

    def get_sticker_polarity(self, sticker_emo):
        if not sticker_emo: return "neu"
        emo = sticker_emo.lower()
        if any(e in emo for e in self.pos_emotions): return "pos"
        if any(e in emo for e in self.neg_emotions): return "neg"
        return "neu"

    def refine_file(self, input_file, output_file):
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            return

        print(f"Processing {input_file} using {self.model_name}...")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            lines = f_in.readlines()
            lines = lines[0:2]
            for line in tqdm(lines):
                data = json.loads(line)

                for turn in data['dialogue']:
                    # 初始化冲突为 -1
                    turn['is_conflict'] = -1

                    new_text_sent = self.get_llm_sentiment(turn['text'])
                    turn['text_sentiment'] = new_text_sent

                    # 只有在没有错误的情况下才进行冲突判断
                    if "ERROR" not in new_text_sent:
                        if turn.get('sticker_path'):
                            s_polar = self.get_sticker_polarity(turn.get('emotion_label', ''))
                            t_polar = new_text_sent

                            if t_polar != "neu" and s_polar != "neu":
                                turn['is_conflict'] = 1 if t_polar != s_polar else 0
                            else:
                                turn['is_conflict'] = 0
                        else:
                            turn['is_conflict'] = 0
                    else:
                        # 如果是 ERROR，保持 is_conflict 为 -1，方便后续过滤
                        pass

                # 顶层冲突判定
                final_conflict = 0
                # 只有在最后一条数据没有 ERROR 时才更新顶层状态
                last_turn_with_sticker = None
                for turn in reversed(data['dialogue']):
                    if turn.get('sticker_path'):
                        last_turn_with_sticker = turn
                        break

                if last_turn_with_sticker:
                    # 如果最后一条带图的检测失败了，顶层也标为 -1
                    if last_turn_with_sticker['is_conflict'] == -1:
                        data['is_session_conflict'] = -1
                    else:
                        data['is_session_conflict'] = last_turn_with_sticker['is_conflict']

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 使用你 list 里有的准确名称
    refiner = SentimentRefiner(model_name='llama3.1:8b')

    # tasks = [
    #     ('StickerConv_Cleaned_Test.jsonl', 'StickerConv_Refined_Test.jsonl'),
    #     ('StickerConv_Cleaned_Train.jsonl', 'StickerConv_Refined_Train.jsonl'),
    #     ('StickerConv_Cleaned_Vail.jsonl', 'StickerConv_Refined_Vail.jsonl')
    # ]
    tasks = [
        ('StickerConv_Cleaned_Test.jsonl', 'StickerConv_Refined_Test.jsonl')
    ]

    for in_f, out_f in tasks:
        refiner.refine_file(in_f, out_f)