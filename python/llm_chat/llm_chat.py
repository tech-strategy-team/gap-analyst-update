import os
from openai import OpenAI


def chat_with_gpt(prompt):

    # OpenAI APIクライアントを初期化
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("環境変数 'OPENAI_API_KEY' が設定されていません。以下のコマンドで設定してください:\nexport OPENAI_API_KEY='your-api-key'")
    client = OpenAI(api_key=api_key)

    # Chat APIを呼び出して応答を取得
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    # 応答のテキストを返す
    return response.choices[0].message.content.strip()

def main():
    print("ChatGPT CLIへようこそ！終了するには 'exit' と入力してください。")
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            print("終了します。")
            break
        response = chat_with_gpt(user_input)
        print(f"ChatGPT: {response}")

if __name__ == "__main__":
    main()