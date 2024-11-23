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

3. **Pipのインストール**

   Pythonのインストール後、通常pipは自動的にインストールされますが、もしインストールされていない場合は以下のURLからget-pip.pyをダウンロードし、次のコマンドを実行してください。

   ```cmd
   python get-pip.py
   ```

4. **OpenAIライブラリのインストール**

   コマンドプロンプトを開き、以下のコマンドを実行します。

   ```cmd
   pip install openai
   ```

5. **アプリの実行**

   コマンドプロンプトで以下のコマンドを実行します。

   ```cmd
   python llm-chat-demo.py
   ```


