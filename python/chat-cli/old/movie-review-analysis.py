# 必要なライブラリのインストール（初回のみ）
# pip install google-generativeai

# このコードはgeminiーAPIを使ってLLMとチャットでの対話を可能にしたコードです

import google.generativeai as genai
import os

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

# GenerationConfigでパラメータを設定
config = genai.GenerationConfig(
    max_output_tokens=2048,  # 生成されるトークンの最大数
    temperature=0.8,         # 出力のランダム性を制御
)

# Geminiモデルを使ってコンテンツを生成する関数
def generate_content(model, prompt):
    response = model.generate_content(prompt, generation_config=config)
    return response.text

# ユーザーから入力を受け取り、Geminiと会話
user_input = input("質問を入力してください: ")
response = generate_content(model, user_input)
print(f"Gemini: {response}")


#------------------------------------------
#　9.GenerationConfigでパラメータを設定
#------------------------------------------
config = genai.GenerationConfig(
    max_output_tokens=30,  # 生成されるトークンの最大数
    temperature=0,  # 出力のランダム性を制御
)
def generate_content(model, prompt):
    response = model.generate_content(prompt, generation_config=config)
    return response.text

#------------------------------------------
#　10.レビュー分析に応用
#------------------------------------------
review_text = "最高だったのは予告編までだった"
prompt = (
    f"""以下の映画レビュー文を高評価か低評価か分類してください。\n
    レビュー: {review_text}\n
    評価: """
)
response = generate_content(model, prompt)
print(response)

#------------------------------------------
#　11.フューショットラーニングを利用した分類
#------------------------------------------
review_text = "最高だったのは予告編までだった"
few_shot_prompt = (
    f"""
    映画レビュー文を以下に分類してください。
    分類:
    - 高評価
    - 低評価

    テキスト:この映画はとても面白かったし、感動的だった。もう一度見たいと思った。
    評価：高評価
    テキスト:ストーリーが複雑で理解できなかったし、アクションシーンも退屈だった。
    評価：低評価
    テキスト:この映画はキャストもストーリーも素晴らしく、何度でも見たいと思える作品だった。
    評価：高評価

    テキスト:{review_text}\n
    評価：
    """
)
response = generate_content(model, few_shot_prompt)
print(response)

#------------------------------------------
#　12.複数のレビュー文を表に準備
#------------------------------------------
import pandas as pd
data = {
    'レビュー文': [
        '感動的で、登場人物も魅力的だった',
        'ストーリーがあまり面白くなかった',
        '期待していたほど面白くなかった',
        '最高。この映画をずっと待っていた',
        '各シーンの映像が綺麗で、ロケ地に旅行したくなった'
    ],
}
df = pd.DataFrame(data)
display(df)

#------------------------------------------
#　13.表（データフレーム）のレビュー文を自動で分類
#------------------------------------------
def evaluate_review(review_text):
    few_shot_prompt = (
        f"""
        映画レビュー文を以下に分類してください。
        分類:
        - 高評価
        - 低評価

        テキスト:この映画はとても面白かったし、感動的だった。もう一度見たいと思った。
        評価：高評価
        テキスト:ストーリーが複雑で理解できなかったし、アクションシーンも退屈だった。
        評価：低評価
        テキスト:この映画はキャストもストーリーも素晴らしく、何度でも見たいと思える作品だった。
        評価：高評価

        テキスト:{review_text}\n
        評価：
        """
    )
    response = generate_content(model, few_shot_prompt)
    if "高評価" in response:
        return "高評価"
    elif "低評価" in response:
        return "低評価"
    else:
        return "評価なし"
    return response
df['評価'] = df['レビュー文'].apply(evaluate_review)
display(df)