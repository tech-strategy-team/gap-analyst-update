#------------------------------------------
#　パッケージのインストール
#------------------------------------------
# !pip install -q langchain
# !pip install -q langchain_core
# !pip install -q langchain-google-genai
# !pip install -q langgraph langchain_chroma langchain_community langchainhub
# !pip install -q markdown bs4 httpx

#------------------------------------------
#　ライブラリのインポート
#------------------------------------------
# ライブラリのインポート
import getpass
import os
import pandas as pd
import markdown
from bs4 import BeautifulSoup, SoupStrainer
from IPython.display import display, HTML
from google.colab import userdata

# langchain のインポート
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import create_react_agent

#------------------------------------------
#　APIのセットアップ
#------------------------------------------
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

#------------------------------------------
#　LLMの出力を表示する関数。Markdownテキストを受け取り、HTMLに変換して表示。
#------------------------------------------
def display_markdown(md_text):
    html_output = markdown.markdown(md_text)
    display(HTML(html_output))

#------------------------------------------
#　LLM（Gemini）を初期化
#------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=GOOGLE_API_KEY,
)

#------------------------------------------
#　通常の生成: 全ての文章の生成が完了した後に、生成した文章をまとめて出力する。
#------------------------------------------
result = llm.invoke("LLMについて説明してください。")
display_markdown(result.content)

#------------------------------------------
#　ストリーミング生成: 全ての文章の生成が完了する前に、生成した文章を随時出力する。
#------------------------------------------
for chunk in llm.stream("LLMについて説明してください。"):
    display_markdown(chunk.content)

    
#------------------------------------------
#　通常の生成：順番にテキストを生成する。
#------------------------------------------
result = llm.invoke("20-2-8を計算して")
display_markdown(result.content)

result = llm.invoke("3+25-8を計算して")
display_markdown(result.content)

result = llm.invoke("29+3-2を計算して")
display_markdown(result.content)

#------------------------------------------
#　バッチ処理：複数の処理を同時に行う。これにより、処理時間を削減可能
#------------------------------------------
results = llm.batch(
    [
        "20-2-8を計算して",
        "3+25-8を計算して",
        "29+3-2を計算して",
    ]
)
for res in results:
    display_markdown(res.content)

#------------------------------------------
#　LLM（Gemini）を初期化
#------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key = GOOGLE_API_KEY,
)

#------------------------------------------
#　プロンプトにシステムメッセージを追加：LLMの役割を定義
#------------------------------------------
prompt = [
    (
        "system",
        "あなたは日本語で具体例を使いながら、わかりやすく回答するチャットボットです。",
    ),
    (
        "user",
        "LLMについて教えてください。"
    ),
]
result = llm.invoke(prompt)
display_markdown(result.content)

#------------------------------------------
#　プロンプトテンプレートを使う
#------------------------------------------
prompt_template = PromptTemplate.from_template(
    "{topic}について教えてください。"
)
prompt = prompt_template.invoke({"topic": "LLM"})
display(prompt.text)
prompt = prompt_template.invoke({"topic": "Google"})
display(prompt.text)

#------------------------------------------
#　参考：システムメッセージとプロンプトテンプレートの組み合わせ例
#------------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "あなたは日本語で具体例を使いながら、わかりやすく回答するチャットボットです。",
    ),
    (
        "user",
        "{topic}について教えてください。"
    ),
])
prompt = prompt_template.invoke({"topic": "LLM"})
result = llm.invoke(prompt)
display_markdown(result.content)

#------------------------------------------
#　メモリー用プロンプトを定義
#------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは日本語で会話するチャットボットです"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

#------------------------------------------
#　チャット履歴とLLMを連携したchain_with_historyを作成
#------------------------------------------

chain = prompt | llm | StrOutputParser()

history = InMemoryChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(chain, lambda x: history)

#------------------------------------------
# chain_with_historyを呼び出し、チャット履歴を参照したテキスト生成を行う。終了するには「終了」と入力します。
#------------------------------------------
while True:
    user_input = input("あなた: ")
    if user_input.strip().lower() == "終了":
        break

    # チャット履歴を参照してレスポンスを生成
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "42"}},
    )

    # LLMのレスポンスを表示
    print("LLM: " + response)
    print("\n")  # 改行を追加してレスポンス後に空白行を入れる