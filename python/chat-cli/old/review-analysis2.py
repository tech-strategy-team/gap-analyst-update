# 必要なライブラリのインストール（初回のみ）
# pip install google-generativeai

# このコードはgeminiーAPIを使ってLLMとチャットでの対話を可能にしたコードです
#------------------------------------------
#　#2.ライブラリのインポート
#------------------------------------------
import os
import google.generativeai as genai  # Googleの生成AIライブラリ
# from google.colab import userdata  # Google Colabのユーザーデータモジュール
import pandas as pd  # データ分析ライブラリ
import json
import time
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
# import japanize_matplotlib  # 使用しないのでコメントアウト
import re

# 日本語表示のためにフォントを設定（例：Windowsの場合 "Meiryo"）
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号が文字化けするのを防ぐ

# APIキーを直接設定（セキュリティに留意してください）
# 環境変数からAPIキーを取得する
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY is None:
    raise ValueError("環境変数 'GOOGLE_API_KEY' が設定されていません。")

genai.configure(api_key=GOOGLE_API_KEY)

# 使用可能なGeminiモデルの一覧を表示
print("使用可能なGeminiのモデル一覧：")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)

# Geminiモデルの選択
model = genai.GenerativeModel("models/gemini-2.0-flash-001")
print(f"選択されたモデル: {model.model_name}")

#------------------------------------------
#　#6.レビューデータをセット
#------------------------------------------
reviews = [
    "数か月使用した時点でのレビューです。低音強めでパンチがありますが、高音がこもりがちです。外観は高級感があり、ケーブルも絡まりにくいです。",
    "全体的に満足しています。梱包の箱が壊れていたのは残念ですが、音質は期待以上で、クリアでバランスが取れたサウンドを楽しめます。価格は手頃で、各サイズのイヤーピースが付属しているのも便利です。長時間の使用に耐えられるかどうかはまだわかりませんが、予備のイヤホンとしては最適な選択肢だと思います。",
    "梱包の段ボールが破損して届いた。中身は問題なかった",
    "低音は豊かでパンチがありますが、高音は少しこもってしまいます。ケーブルがもつれやすいのも気になります。",
    "装着感は良いですが、ノイズキャンセリングは微妙です。",
    # 以下、他のレビュー...
]
df = pd.DataFrame({
    'レビュー文': reviews
})
df.head(5)

#------------------------------------------
#　#7.レビューテキストの前処理
#------------------------------------------
df['レビュー文'] = df['レビュー文'].str.replace('\n', '', regex=False)   # 改行のみを削除
df['レビュー文'] = df['レビュー文'].str.replace('、', '', regex=False)   # 読点のみを削除
df['レビュー文'] = df['レビュー文'].str.replace('[！？]', '。', regex=True) # ビックリマークと疑問符を句点に変更
df.head(5)

#---------------------------------------------------
#　#8.Geminiでテキスト生成
#---------------------------------------------------
def generate_content(model, prompt, temperature=0.0):
    config = genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=temperature
    )
    while True:
        try:
            response = model.generate_content(prompt, generation_config=config)
            return response.text
        except Exception as e:
            if "429" in str(e):
                print("1分間に使用できる上限に到達しました。1分間待機して処理を続行します。")
                time.sleep(60)
            else:
                raise e

#---------------------------------------------------
#　#9.レビューテキストから要素を抽出する関数
#---------------------------------------------------
def extract_review_elements(model, review_text):
    prompt = f"""
    以下のテキストから高評価要素と低評価要素を分析してください。
    結果は必ず次のJSON形式で返してください。他の説明や文章は不要です。

    テキスト: 音質は素晴らしかった。長時間使用しても疲れず、耳にもフィットして快適でした。
    {{
        "高評価要素": ["音質", "装着感"],
        "低評価要素": []
    }}

    テキスト: 迫力がなく音がスカスカ。音漏れが酷く電車などの公共の場で使えないレベル。
    {{
        "高評価要素": [],
        "低評価要素": ["音質", "音漏れ"]
    }}

    テキスト: 値段を考えると音質はまあまあ良い。でも耳が痛くなるのが難点。長時間は使えない。あと届いた箱がつぶれていたのが残念。
    {{
        "高評価要素": ["音質"],
        "低評価要素": ["装着感", "梱包"]
    }}

    テキスト: {review_text}
    """
    response = generate_content(model, prompt)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            if "高評価要素" in parsed_json and "低評価要素" in parsed_json:
                return json.dumps(parsed_json, ensure_ascii=False, indent=4)
        except json.JSONDecodeError:
            pass
    default_json = {"高評価要素": [], "低評価要素": []}
    return json.dumps(default_json, ensure_ascii=False, indent=4)

#---------------------------------------------------
#　#10.データフレームのレビューを処理して結果を追加
#---------------------------------------------------
def analyze_reviews(df, model):
    json_data = json.loads(df.to_json(orient='records'))
    for review in tqdm(json_data, desc='レビュー文をGeminiで分析中'):
        result = extract_review_elements(model, review['レビュー文'])
        review.update(json.loads(result))
    return json_data

#---------------------------------------------------
#　#11.カウント結果をプロット
#---------------------------------------------------
def plot_elements(elements, title, color, subplot_index):
    elements.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*elements)
    plt.subplot(1, 2, subplot_index)
    plt.title(title)
    bars = plt.barh(labels, values, color=color)
    for bar in bars:
        plt.text(bar.get_width() - bar.get_width() * 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width()}', va='center', color='white', fontweight='bold', fontfamily='DejaVu Sans')
    plt.gca().invert_yaxis()
    plt.tight_layout()

#-----------------------------------------------------------
#　#12.特定した高評価と低評価の要素を基に簡単なレポートを作成
#-----------------------------------------------------------
def summarize_review_elements(review_elements, model):
    prompt = f"""
        以下はイヤホンに関するレビューで集計された高評価の要素と、低評価の要素です。このイヤホンの評価をまとめてください。
        例：ユーザーはこのスマートウォッチの操作性とバッテリー寿命を高く評価しています。ただし、価格に対しては低評価が多くなっています。また、防水機能に関しては高評価と低評価、両方の意見が寄せられています。
        {review_elements}"""
    response = generate_content(model, prompt)
    response += "\n以下にユーザーからの高評価・低評価の要素のグラフを表示します。"
    return response

#-----------------------------------------------------------
#　#13.メイン処理
#-----------------------------------------------------------
reviews_json = analyze_reviews(df, model)
top_5_high_elements = Counter([element for review in reviews_json for element in review["高評価要素"]]).most_common(5)
top_5_low_elements = Counter([element for review in reviews_json for element in review["低評価要素"]]).most_common(5)

review_elements = f"高評価レビューの主要要素:{top_5_high_elements}\n低評価レビューの主要要素:{top_5_low_elements}"
summary_review_elements = summarize_review_elements(review_elements, model)
print("\n-----分析結果レポート-----")
print(summary_review_elements)
print("\n-----分析結果のグラフ-----")
plt.figure(figsize=(10, 5))
plot_elements(top_5_high_elements, '高評価レビューの主要要素', '#3652AD', 1)
plot_elements(top_5_low_elements, '低評価レビューの主要要素', '#FE7A36', 2)
plt.show()
