# ChatGPT API デモ


https://github.com/openai/openai-python


## 事前準備

OpenAIのAPI keyを下記のURLで`Create new secret key"から取得する（事前にいくらかチャージが必要）

https://platform.openai.com/settings/organization/api-keys

## コード実行(Windows コマンドプロンプト)

1. **OPENAIのAPIキーを環境変数に設定**

   ```cmd
   set OPENAI_API_KEY=your-api-key
   ```

2. **Pythonのインストール**

   Windowsでは通常、Pythonは公式サイトからインストーラーをダウンロードしてインストールします。以下のURLからインストーラーをダウンロードしてください。

   https://www.python.org/downloads/

3. **OpenAIライブラリのインストール**

   コマンドプロンプトを開き、以下のコマンドを実行します。

   ```cmd
   pip install openai
   ```

4. **アプリの実行**

   コマンドプロンプトで以下のコマンドを実行します。

   ```cmd
   python llm-chat.py
   ```


テスト

