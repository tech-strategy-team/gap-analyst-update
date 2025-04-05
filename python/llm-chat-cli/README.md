# Chat with LLM

![](https://github.com/Hiralion/n-group/actions/workflows/python-test.yaml/badge.svg)


1. モジュールのインストール

    ```console
    pip install -r requirements.txt
    ```

1. .envファイルの作成

    ```console
    echo "OPENAI_API_KEY=<API-KEY>" >> .env
    ```

    envファイルが正しく設定できていること
    ```consle
    $ cat .env
    OPENAI_API_KEY=...
    ```
1. カレントディレクトリに.envファイルが存在することを確認

    ```cosole
    $ ls -la
    total 112
    drwxr-xr-x  10 kouichihara  staff   320  3 26 22:56 .
    drwxr-xr-x@  7 kouichihara  staff   224  3 25 23:19 ..
    -rw-r--r--   1 kouichihara  staff   180  3 26 23:02 .env
    -rw-r--r--   1 kouichihara  staff     0  3 26 22:26 README.md
    -rw-r--r--   1 kouichihara  staff  8386  3 26 22:55 chat_with_llm.py
    -rw-r--r--   1 kouichihara  staff    82  3 26 22:56 requirements.txt
    ```
1. Pythonの実行

    ```console
    $ python3 chat_with_llm.py
    LLMとチャットを開始します。終了するには 'exit' または 'quit' と入力してください。
    システムプロンプトを入力してください（デフォルト: あなたは親切で役立つAIアシスタントです。）:  

    あなた: あなたは誰？

    AI: 私は、あなたがお話し相手や質問相手として利用できるAIアシスタントです。親切で役立つアシスタントとして、さまざまな質問やお手伝いをすることができます。お困りごとや疑問があれば、遠慮なく聞いてくださいね。

    あなた: exit
    チャットを終了します。
    ```
