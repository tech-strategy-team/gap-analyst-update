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
