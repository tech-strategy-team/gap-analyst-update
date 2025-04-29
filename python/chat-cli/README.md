# Chat with LLM

![badge](https://github.com/Hiralion/n-group/actions/workflows/python-test.yaml/badge.svg)

ユーザーがCLIでLLMと対話するアプリケーションです。  
このアプリケーションを使用すると、OpenAIのAPIを利用してAIと対話できます。  
詳細な設計については[設計ドキュメント](./design.md)を参照してください。

## 必要条件

- Python 3.8以上
- OpenAI APIキー

## 手順

1. リポジトリをクローン

    ```console
    git clone https://github.com/Hiralion/n-group.git
    cd n-group/python/chat-cli
    ```

1. モジュールのインストール

    ```console
    pip install -r requirements.txt
    ```

1. .envファイルの作成

    ```console
    echo "OPENAI_API_KEY=<API-KEY>" >> .env
    ```

    `.env`ファイルが正しく設定されていることを確認してください。

    ```console
    $ cat .env
    OPENAI_API_KEY=...
    ```

1. アプリケーションの実行

    ```console
    python3 chat_with_llm.py
    ```

    実行後、以下のように対話を開始できます。

    ```text
    LLMとチャットを開始します。終了するには 'exit' または 'quit' と入力してください。
    システムプロンプトを入力してください（デフォルト: あなたは親切で役立つAIアシスタントです。）:  

    あなた: あなたは誰？

    AI: 私は、あなたがお話し相手や質問相手として利用できるAIアシスタントです。親切で役立つアシスタントとして、さまざまな質問やお手伝いをすることができます。お困りごとや疑問があれば、遠慮なく聞いてくださいね。

    あなた: exit
    チャットを終了します。
    ```

## 注意事項

- OpenAI APIキーは個人の責任で管理してください。
- 実行環境にPython 3.8以上がインストールされていることを確認してください。

## コントリビュータ

- @Hiralion
- @kouichihara
