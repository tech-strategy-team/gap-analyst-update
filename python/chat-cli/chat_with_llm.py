import os
import datetime
import re
import sys
import time
import threading
import mimetypes
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
import PyPDF2
import openai
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LoadingIndicator:
    """APIレスポンス待ち中に表示するローディングインジケータークラス"""

    def __init__(self, message="APIからの応答を待っています"):
        self.message = message
        self.is_running = False
        self.thread = None
        self.animation_chars = [
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"
        ]
        # 別のアニメーションスタイルのオプション:
        # self.animation_chars = ["-", "\\", "|", "/"]
        # self.animation_chars = [".", "..", "...", "...."]

    def animate(self):
        """アニメーションを表示するメソッド"""
        i = 0
        while self.is_running:
            idx = i % len(self.animation_chars)
            animation_char = self.animation_chars[idx]
            # カーソルを行の先頭に移動し、メッセージと
            # アニメーションを表示
            output = f"\r{self.message}  {animation_char}"
            sys.stdout.write(output)
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # アニメーション終了時に行をクリアする
        self._clear_line()

    def _clear_line(self):
        """現在の行を完全にクリアする（ANSI エスケープシーケンスを使用）。

        ANSI エスケープシーケンスを使用して行を完全にクリアします。
        \033[2K: 現在の行を完全にクリア
        \r: カーソルを行の先頭に移動
        """
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    def start(self):
        """ローディングアニメーションを開始"""
        self.is_running = True
        self.thread = threading.Thread(target=self.animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """ローディングアニメーションを停止"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        # スレッド終了後、確実に行をクリア
        self._clear_line()


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
        self._initialize_llm()  # LLMの初期化
        self._setup_prompt_template()  # プロンプトテンプレートの設定
        self._setup_chain()  # チェーンの設定

    def _initialize_llm(self):
        """使用するLLMを初期化し、エラーをハンドリング"""
        try:
            if self.provider.lower() == "openai":
                # OpenAIのChatOpenAIを初期化
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    # 例："gpt-3.5-turbo" または "gpt-4"
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

        # ローディングインジケーターを初期化
        loading = LoadingIndicator("応答を待っています")

        try:
            # ローディングアニメーションを開始
            loading.start()

            # APIリクエストを実行
            response = self.chain_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}},
            )

            # ローディングアニメーションを停止
            loading.stop()

            return response
        except openai.error.OpenAIError as e:
            # エラー発生時もローディングアニメーションを停止
            loading.stop()
            print(f"OpenAI APIでエラーが発生しました: {e}")
            return "APIとの通信でエラーが発生しました。時間をおいて再度お試しください。"
        except Exception as e:
            # エラー発生時もローディングアニメーションを停止
            loading.stop()
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
                history.append(
                    {"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"role": "system", "content": message.content})
        return history

    def clear_history(self):
        """チャット履歴をクリア"""
        self.history.clear()
        print("チャット履歴をクリアしました。")

    def add_file_to_context(self, file_path: str) -> Tuple[bool, str]:
        """
        ファイルの内容をLLMのコンテキストに追加

        Args:
            file_path: 追加するファイルのパス

        Returns:
            成功したかどうかのブール値とメッセージのタプル
        """
        try:
            # ファイルの存在確認
            if not os.path.exists(file_path):
                return False, f"ファイルが見つかりません: {file_path}"

            # ファイルサイズの確認（大きすぎるファイルは処理しない）
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB以上のファイルは拒否
                return False, f"ファイルサイズが大きすぎます（10MB以下にしてください）: {file_path}"

            # MIMEタイプの確認
            mime_type, _ = mimetypes.guess_type(file_path)

            # ファイル名と拡張子を取得
            file_name = os.path.basename(file_path)
            _, ext = os.path.splitext(file_path)
            ext_lower = ext.lower()

            # PDFファイルの処理
            if ext_lower == '.pdf' or (
                    mime_type and mime_type == 'application/pdf'):
                try:
                    # PDFファイルからテキストを抽出
                    text_content = ""
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)

                        # 各ページからテキストを抽出
                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text_content += page.extract_text() + "\n\n"

                    # 抽出したテキストが空でないか確認
                    if not text_content.strip():
                        return False, (
                            f"PDFファイル「{file_name}」からテキストを抽出できませんでした。"
                            f"スキャンされたPDFの可能性があります。"
                        )

                    # PDFの内容をユーザーメッセージとして追加
                    file_message = (
                        f"以下は「{file_name}」（PDF）から抽出したテキスト内容です：\n\n"
                        f"```\n{text_content}\n```"
                    )
                    self.history.add_user_message(file_message)

                    return True, f"PDFファイル「{file_name}」を追加しました。"
                except Exception as e:
                    return False, f"PDFファイル「{file_name}」の処理中にエラーが発生しました: {e}"

            # テキストファイルかどうかの確認
            is_text_file = mime_type and mime_type.startswith('text/')

            # 一般的なテキストファイル拡張子のリスト
            text_extensions = [
                '.txt',
                '.md',
                '.py',
                '.js',
                '.html',
                '.css',
                '.json',
                '.xml',
                '.csv',
                '.log',
                '.sh',
                '.bat',
                '.c',
                '.cpp',
                '.h',
                '.java',
                '.rb',
                '.php',
                '.ts',
                '.tsx',
                '.jsx']

            # テキストファイルでない場合はエラー
            if not is_text_file and ext_lower not in text_extensions:
                return False, f"サポートされていないファイル形式です: {file_path}"

            # テキストファイルの内容を読み取り
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except UnicodeDecodeError:
                # UTF-8でデコードできない場合は、他のエンコーディングを試す
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                except UnicodeDecodeError:
                    # chardetでエンコーディングを検出
                    import chardet
                    with open(file_path, 'rb') as f:
                        result = chardet.detect(f.read())
                        encoding = result['encoding']
                    if encoding:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                file_content = f.read()
                        except UnicodeDecodeError:
                            return False, (
                                f"ファイルのエンコーディングを認識できません: {file_path}"
                            )
                    else:
                        return False, (
                            f"ファイルのエンコーディングを認識できません: {file_path}"
                        )

            # ファイルの内容をユーザーメッセージとして追加
            file_message = (
                f"以下は「{file_name}」の内容です："
                f"\n\n```{ext[1:]}\n{file_content}\n```"
            )
            self.history.add_user_message(file_message)

            return True, f"ファイル「{file_name}」を追加しました。"
        except Exception as e:
            return False, f"ファイル追加中にエラーが発生しました: {e}"


# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def get_first_user_message(chat_history: List[Dict[str, Any]]) -> str:
    """
    チャット履歴から最初の非コマンドメッセージを取得します。

    Args:
        chat_history: チャット履歴

    Returns:
        最初の非コマンドメッセージ、または空文字列
    """
    for msg in chat_history:
        if msg["role"] == "user" and not msg["content"].startswith("/"):
            return msg["content"]
    return ""


def get_api_key(provided_key: str = None) -> str:
    """
    APIキーを取得します。

    Args:
        provided_key: 指定されたAPIキー

    Returns:
        APIキー、または None
    """
    if provided_key:
        return provided_key
    return os.environ.get("OPENAI_API_KEY")


def generate_title_with_llm(user_message: str, api_key: str) -> str:
    """
    LLMを使用してタイトルを生成します。

    Args:
        user_message: ユーザーのメッセージ
        api_key: OpenAI APIキー

    Returns:
        生成されたタイトル
    """
    # ローディングインジケーターを初期化
    loading = LoadingIndicator("タイトルを生成中です")

    try:
        # ローディングアニメーションを開始
        loading.start()

        # LLMを使用してタイトルを生成
        llm = ChatOpenAI(
            model_name="o3-mini",  # 軽量モデルを使用
            openai_api_key=api_key,
        )

        # プロンプトテンプレート
        prompt = ChatPromptTemplate.from_messages([
            ("system", "以下のチャットメッセージから、"
                       "ファイル名に適した短い要約タイトル"
                       "（15文字程度）を生成してください。"
                       "チャット内容をそのまま使わず、"
                       "内容を要約したタイトルにしてください。"
                       "日本語で返してください。"),
            ("human", f"{user_message}")
        ])

        # チェーンを実行
        chain = prompt | llm | StrOutputParser()
        title = chain.invoke({})

        # 長すぎる場合はカット
        if len(title) > 30:
            title = title[:30]

        return title
    except Exception as e:
        error_msg = f"タイトル生成中にエラーが発生しました: {e}"
        print(error_msg)
        logger.error(error_msg)
        return ""
    finally:
        # エラーが発生した場合もローディングアニメーションを停止
        if loading.is_running:
            loading.stop()


def sanitize_filename(title: str) -> str:
    """
    文字列をファイル名として安全な形式に変換します。

    Args:
        title: 変換する文字列

    Returns:
        ファイル名として安全な文字列
    """
    # 1. ファイル名に使用できない文字を除去
    # Windows, macOS, Linuxで共通して使用できない文字: / \ : * ? " < > |
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)

    # 2. 制御文字や非表示文字を除去
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', "", sanitized)

    # 3. 先頭と末尾の空白、ピリオド、ハイフンを除去（多くのファイルシステムで問題になる可能性がある）
    sanitized = sanitized.strip(" .-")

    # 4. 連続する空白をハイフンに変換
    sanitized = re.sub(r'\s+', "-", sanitized)

    # 5. 残りの非英数字（日本語などのUnicode文字は保持）をアンダースコアに変換
    sanitized = re.sub(r'[^\w\s\-\.]', "_", sanitized)

    # 6. ファイル名が空の場合はデフォルト値を使用
    if not sanitized:
        return "untitled"

    # 7. ファイル名の長さを制限（多くのファイルシステムでは255文字が上限）
    # 余裕を持って200文字に制限
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    # 8. Windowsの予約語をチェック
    reserved_names = [
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"
    ]

    # ファイル名の先頭部分（拡張子を除く）が予約語と一致する場合、接頭辞を追加
    name_lower = sanitized.lower()
    if name_lower in reserved_names or any(name_lower.startswith(rn + ".") for rn in reserved_names):
        sanitized = "file_" + sanitized

    return sanitized


def fallback_title(user_message: str) -> str:
    """
    LLMが使用できない場合のフォールバックタイトルを生成します。

    Args:
        user_message: ユーザーのメッセージ

    Returns:
        生成されたタイトル
    """
    # 最初の10単語を使用
    words = user_message.split()
    return " ".join(words[:10])


def generate_title_from_chat(chat_history: List[Dict[str, Any]], api_key: str = None) -> str:
    """
    チャット履歴からファイル名に利用するタイトルを生成します。
    ユーザーの非コマンドメッセージの内容を要約し、
    不要な記号を除去してファイル名として安全な形式に変換します。

    Args:
        chat_history: チャット履歴
        api_key: OpenAI APIキー（指定されていない場合は環境変数から取得）

    Returns:
        生成されたタイトル
    """
    # ユーザーの最初の非コマンドメッセージを取得
    user_message = get_first_user_message(chat_history)

    if not user_message:
        return "untitled"

    # APIキーを取得
    api_key = get_api_key(api_key)

    # タイトルを生成
    if api_key:
        title = generate_title_with_llm(user_message, api_key)
        if not title:  # LLMでの生成に失敗した場合
            title = fallback_title(user_message)
    else:
        # APIキーがない場合はフォールバック
        title = fallback_title(user_message)

    # ファイル名として安全な形式に変換
    return sanitize_filename(title)


def save_chat_log(chat_history: List[Dict[str, Any]], api_key: str = None):
    """
    チャット履歴をログファイルに保存

    Args:
        chat_history: 保存するチャット履歴

    Returns:
        保存したファイルのパス、または保存に失敗した場合はNone
    """
    # ログ用のディレクトリを作成（存在しない場合）
    # カレントディレクトリからの相対パスでlogsディレクトリを指定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"ディレクトリ作成中にエラーが発生しました: {e}")
        return None

    # チャット内容からタイトルを生成して、日付と組み合わせたファイル名を作成
    title = generate_title_from_chat(chat_history, api_key)
    date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"chat-log-{date}-{title}.txt"
    filepath = os.path.join(log_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            # ヘッダーを書き込み
            now = datetime.datetime.now()
            header = f"===== チャットログ ({
                now.strftime('%Y-%m-%d %H:%M:%S')}) =====\n\n"
            f.write(header)

            # チャット履歴を書き込み
            role_mapping = {"system": "システム", "user": "あなた", "assistant": "AI"}
            for msg in chat_history:
                role = role_mapping.get(msg["role"], "不明")
                f.write(f"{role}: {msg['content']}\n\n")

        print(f"チャットログを {filepath} に保存しました。")
        return filepath
    except Exception as e:
        print(f"ログファイルの保存中にエラーが発生しました: {type(e)} - {e}")
        return None


def load_chat_log(filepath: str) -> List[Dict[str, str]]:
    """
    ログファイルからチャット履歴を読み込む

    Args:
        filepath: ログファイルのパス

    Returns:
        チャット履歴のリスト
    """
    try:
        # スクリプトのディレクトリを取得
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 絶対パスでない場合は、相対パスとして処理
        if not os.path.isabs(filepath):
            # まず、指定されたパスをそのまま試す
            if not os.path.exists(filepath):
                # 次に、スクリプトディレクトリからの相対パスを試す
                filepath_from_script = os.path.join(script_dir, filepath)
                if os.path.exists(filepath_from_script):
                    filepath = filepath_from_script
                else:
                    # 最後に、logsディレクトリ内を確認
                    filepath_in_logs = os.path.join(
                        script_dir, "logs", os.path.basename(filepath))
                    if os.path.exists(filepath_in_logs):
                        filepath = filepath_in_logs
                    else:
                        print(f"ファイルが見つかりません: {filepath}")
                        return []

        # ファイルを読み込む
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # ヘッダー部分を除外
        header_pattern = r"===== チャットログ \(.*?\) =====\n\n"
        content = re.sub(header_pattern, "", content)

        # メッセージを解析
        messages = []

        # メッセージのパターン: "役割: 内容" + 空行
        message_pattern = r"(システム|あなた|AI): (.*?)(?:\n\n|$)"
        matches = re.findall(message_pattern, content, re.DOTALL)

        role_mapping = {"システム": "system", "あなた": "user", "AI": "assistant"}

        for role_ja, content in matches:
            role = role_mapping.get(role_ja, "unknown")
            messages.append({"role": role, "content": content.strip()})

        return messages
    except Exception as e:
        print(f"ログファイルの読み込み中にエラーが発生しました: {e}")
        return []  # エラーが発生した場合は空のリストを返す


def get_available_models():
    """利用可能なGPTモデルのリストを返す"""
    return {
        "1": "gpt-4o",
        "2": "o3-mini",
        "3": "o1"
    }


def main():
    """メイン関数"""
    print("LLMとチャットを開始します。終了するには 'exit' または 'quit' と入力してください。")

    # APIキーは環境変数OPENAI_API_KEYを推奨
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # 環境変数が設定されていなければ手動入力を促す
        api_key = prompt("OpenAI API キーを入力してください: ")

    # モデルを選択
    available_models = get_available_models()
    print("\n利用可能なモデルを選択してください:")
    for key, model in available_models.items():
        print(f"{key}: {model}")

    model_choice = prompt("モデル番号を入力してください (デフォルト: 1 - gpt-4o): ").strip()
    if model_choice:
        model_name = available_models.get(model_choice, "gpt-4o")
    else:
        model_name = "gpt-4o"
    print(f"選択されたモデル: {model_name}")

    # 過去のチャットログを読み込むか確認
    load_log = prompt("過去のチャットログを読み込みますか？ (y/n): ").lower()

    system_prompt = "あなたは親切で役立つAIアシスタントです。"
    loaded_messages = []

    if load_log == 'y':
        # スクリプトのディレクトリを取得
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, "logs")

        # logsディレクトリが存在するか確認
        if not os.path.exists(logs_dir):
            print("ログディレクトリが見つかりません。新規チャットを開始します。")
        else:
            # logsディレクトリ内のtxtファイルを取得
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]

            if not log_files:
                print("ログファイルが見つかりません。新規チャットを開始します。")
            else:
                # ファイルを日付順に並べ替え（新しい順）
                log_files.sort(reverse=True)

                print("読み込むログファイルの番号を入力してください")
                for i, file in enumerate(log_files, 1):
                    print(f"{i}: {file}")

                # ユーザーの選択を取得
                try:
                    choice = int(prompt("番号: "))
                    if 1 <= choice <= len(log_files):
                        log_path = os.path.join(
                            logs_dir, log_files[choice - 1])
                        loaded_messages = load_chat_log(log_path)
                        if loaded_messages:
                            print(f"チャットログを読み込みました。"
                                  f"メッセージ数: {len(loaded_messages)}")
                        else:
                            print("指定されたファイルからチャット履歴を読み込めませんでした。新規チャットを開始します。")
                    else:
                        print("無効な番号です。新規チャットを開始します。")
                        loaded_messages = []
                except ValueError:
                    print("無効な入力です。新規チャットを開始します。")
                    loaded_messages = []

    if not loaded_messages:
        # 新規チャットの場合はシステムプロンプトを入力
        user_system_prompt = prompt(
            "システムプロンプトを入力してください（デフォルト: あなたは親切で役立つAIアシスタントです。）: ")
        if user_system_prompt:
            system_prompt = user_system_prompt

    # ChatWithLLMクラスを初期化（デフォルトプロバイダーはopenai）
    chat_bot = ChatWithLLM(
        api_key=api_key,
        model_name=model_name,
        system_prompt=system_prompt)

    # 過去のチャット履歴を復元
    if loaded_messages:
        for msg in loaded_messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                chat_bot.history.add_user_message(content)
            elif role == "assistant":
                chat_bot.history.add_ai_message(content)
            elif role == "system":
                chat_bot.history.add_message(SystemMessage(content=content))

        print("チャット履歴を復元しました。続きからチャットを開始します。")

    while True:
        # 入力履歴を管理するためのオブジェクト
        input_history = InMemoryHistory()
        user_input = prompt("\nあなた: ", history=input_history)

        # 終了コマンド
        if user_input.lower() in ["exit", "quit", "終了"]:
            # チャット履歴を取得してログファイルに保存
            chat_history = chat_bot.get_chat_history()
            log_path = save_chat_log(chat_history, api_key)
            if log_path:
                print(f"チャットログを保存しました: {log_path}")
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
            print("/model - モデルを変更")
            print("/clear または /クリア - チャット履歴をクリア")
            print("/history または /履歴 - チャット履歴を表示")
            print("/save または /保存 - 現在のチャット履歴を保存")
            print("/add [ファイルパス1] [ファイルパス2] ... - ファイルをLLMに渡す")
            print("/help または /ヘルプ - コマンド一覧を表示")
            print("exit, quit, または 終了 - チャットを終了")
            print("=======================\n")
            continue

        # モデル変更コマンド
        if user_input.lower() == "/model":
            available_models = get_available_models()
            print("\n利用可能なモデルを選択してください:")
            for key, model in available_models.items():
                print(f"{key}: {model}")

            model_choice = prompt("モデル番号を入力してください: ").strip()
            if model_choice in available_models:
                new_model = available_models[model_choice]
                # 新しいモデルで再初期化
                try:
                    # 新しいモデルで再初期化
                    chat_bot = ChatWithLLM(
                        api_key=api_key,
                        model_name=new_model,
                        system_prompt=chat_bot.system_prompt
                    )
                except Exception as e:
                    print(f"モデルの初期化に失敗しました: {e}")
                    continue
                # 履歴を復元
                history = chat_bot.get_chat_history()
                for msg in history:
                    if msg["role"] == "user":
                        chat_bot.history.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        chat_bot.history.add_ai_message(msg["content"])
                    elif msg["role"] == "system":
                        chat_bot.history.add_message(
                            SystemMessage(content=msg["content"]))

                print(f"モデルを {new_model} に変更しました。")
            else:
                print("無効なモデル番号です。")
            continue

        # 保存コマンド
        if user_input.lower() in ["/save", "/保存"]:
            chat_history = chat_bot.get_chat_history()
            log_path = save_chat_log(chat_history, api_key)
            if log_path:
                print(f"チャットログを保存しました: {log_path}")
            continue

        # ファイル追加コマンド
        if user_input.lower().startswith("/add "):
            # コマンドからファイルパスを抽出
            file_paths = user_input[5:].strip().split()

            if not file_paths:
                print("ファイルパスを指定してください。例: /add file1.txt file2.py")
                continue

            # 各ファイルを処理
            success_count = 0
            for file_path in file_paths:
                success, message = chat_bot.add_file_to_context(file_path)
                print(message)
                if success:
                    success_count += 1

            # 追加結果のサマリーを表示
            if success_count > 0:
                if len(file_paths) == 1:
                    print("ファイルをLLMに渡しました。質問や指示を入力してください。")
                else:
                    print(
                        f"{success_count}/{len(file_paths)}個のファイルをLLMに渡しました。"
                        f"質問や指示を入力してください。"
                    )
            continue

        # LLMとチャット
        response = chat_bot.chat(user_input)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
