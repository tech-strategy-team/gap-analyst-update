# ChatGPT API デモ


https://github.com/openai/openai-python


## 事前準備

OpenAIのAPI keyを下記のURLで`Create new secret key"から取得する（事前にいくらかチャージが必要）

https://platform.openai.com/settings/organization/api-keys

## コード実行

1. **OPENAIのAPIキーを環境変数に設定**

   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

2. **Pythonのインストール**

   ```bash
   sudo apt update
   sudo apt install python3-pip
   ```

3. **OpenAIライブラリのインストール**

   ```bash
   pip install openai
   ```

4. **アプリの実行**

   ```bash
   $ python3 llm-chat-demo.py
   ```


