import os
import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class ChatWithLLM:
    """LLMとチャット形式でやり取りするクラス（GPT‑API版）"""
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gpt-4o", 
        provider: str = "openai",
        system_prompt: str = "あなたは親切で役立つAIアシスタントです。"
    ):
        """
        初期化メソッド
        
        Args:
            api_key: LLM APIキー（環境変数などで管理推奨）
            model_name: 使用するモデル名（例："gpt-3.5-turbo" または "gpt-4"）
            provider: LLMプロバイダー ("openai" など)
            system_prompt: システムプロンプト
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt
        self.history = InMemoryChatMessageHistory()
        self._initialize_llm() # LLMの初期化
        self._setup_prompt_template() #プロンプトテンプレートの設定
        self._setup_chain() # チェーンの設定
    def _initialize_llm(self):
        """使用するLLMを初期化し、エラーをハンドリング"""
        try:
            if self.provider.lower() == "openai":
                # OpenAIのChatOpenAIを初期化
                self.llm = ChatOpenAI(
                    model_name=self.model_name,      # 例："gpt-3.5-turbo" または "gpt-4"
                    openai_api_key=self.api_key,
                )
            else:
                raise ValueError(f"サポートされていないプロバイダー: {self.provider}")
        except Exception as e:
            # 初期化中の予期しないエラーをハンドリング
            print(f"LLMの初期化中にエラーが発生しました: {e}")
            self.llm = None
    
    def _setup_prompt_template(self):
        """プロンプトテンプレートを設定"""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

    def _setup_chain(self):
        """チェーンを設定"""
        chain = self.prompt_template | self.llm | StrOutputParser()
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    def update_system_prompt(self, new_system_prompt: str):
        """
        システムプロンプトを更新
        
        Args:
            new_system_prompt: 新しいシステムプロンプト
        """
        self.system_prompt = new_system_prompt
        self._setup_prompt_template()
        self._setup_chain()
        print(f"システムプロンプトを更新しました: {new_system_prompt}")
    
    def chat(self, user_input: str) -> str:
        """
        ユーザー入力に対するレスポンスを生成
        
        Args:
            user_input: ユーザーの入力テキスト
            
        Returns:
            LLMからのレスポンス
        """
        if not self.llm:
            return "LLMが初期化されていないため、応答できません。"
        
        try:
            response = self.chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}},
            )
            return response
        except openai.error.OpenAIError as e:
            print(f"OpenAI APIでエラーが発生しました: {e}")
            return "APIとの通信でエラーが発生しました。時間をおいて再度お試しください。"
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}")
            return "予期しないエラーが発生しました。"
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        チャット履歴を取得
        
        Returns:
            チャット履歴のリスト
        """
        history = []
        for message in self.history.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"role": "system", "content": message.content})
        return history
    
    def clear_history(self):
        """チャット履歴をクリア"""
        self.history.clear()
        print("チャット履歴をクリアしました。")


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def save_chat_log(chat_history: List[Dict[str, Any]]):
    """
    チャット履歴をログファイルに保存
    
    Args:
        chat_history: 保存するチャット履歴
    """
    # 現在の日付を取得してファイル名を生成
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"chat_log_{today}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"===== チャットログ ({today}) =====\n\n")
            for msg in chat_history:
                role_mapping = {
                    "system": "システム",
                    "user": "あなた",
                    "assistant": "AI"
                }
                role = role_mapping.get(msg["role"], "不明") # デフォルト値を設定
                f.write(f"{role}: {msg['content']}\n\n")
        print(f"チャットログを {filename} に保存しました。")
        print(f"ログファイルの保存中にエラーが発生しました: {type(e)} - {e}")

def main():
    """メイン関数"""
    print("LLMとチャットを開始します。終了するには 'exit' または 'quit' と入力してください。")
    
    # APIキーは環境変数OPENAI_API_KEYを推奨
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # 環境変数が設定されていなければ手動入力を促す
        api_key = input("OpenAI API キーを入力してください: ")
    
    system_prompt = input("システムプロンプトを入力してください（デフォルト: あなたは親切で役立つAIアシスタントです。）: ")
    if not system_prompt:
        system_prompt = "あなたは親切で役立つAIアシスタントです。"
    
    # ChatWithLLMクラスを初期化（デフォルトプロバイダーはopenai）
    chat_bot = ChatWithLLM(api_key=api_key, system_prompt=system_prompt)
    
    while True:
        user_input = input("\nあなた: ")
        
        # 終了コマンド
        if user_input.lower() in ["exit", "quit", "終了"]:
            # チャット履歴を取得してログファイルに保存
            chat_history = chat_bot.get_chat_history()
            save_chat_log(chat_history)
            print("チャットを終了します。")
            break
        
        # システムプロンプト変更コマンド
        if user_input.startswith("/system "):
            new_system_prompt = user_input[8:].strip()
            chat_bot.update_system_prompt(new_system_prompt)
            continue
        
        # 履歴クリアコマンド
        if user_input.lower() in ["/clear", "/クリア"]:
            chat_bot.clear_history()
            continue
        
        # 履歴表示コマンド
        if user_input.lower() in ["/history", "/履歴"]:
            history = chat_bot.get_chat_history()
            print("\n===== チャット履歴 =====")
            for msg in history:
                role = (
                    "システム" if msg["role"] == "system" 
                    else "あなた" if msg["role"] == "user" 
                    else "AI"
                )
                print(f"{role}: {msg['content']}")
            print("=======================\n")
            continue
        
        # ヘルプコマンド
        if user_input.lower() in ["/help", "/ヘルプ"]:
            print("\n===== コマンド一覧 =====")
            print("/system [プロンプト] - システムプロンプトを変更")
            print("/clear または /クリア - チャット履歴をクリア")
            print("/history または /履歴 - チャット履歴を表示")
            print("/help または /ヘルプ - コマンド一覧を表示")
            print("exit, quit, または 終了 - チャットを終了")
            print("=======================\n")
            continue
        
        # LLMとチャット
        response = chat_bot.chat(user_input)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
