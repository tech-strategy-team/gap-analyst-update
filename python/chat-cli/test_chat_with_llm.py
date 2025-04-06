import unittest
import warnings  # 警告フィルタ用
from unittest.mock import patch, MagicMock
from chat_with_llm import ChatWithLLM

# 定数としてテスト用APIキーを定義
TEST_API_KEY = "test_api_key"

# DeprecationWarningを無視
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Failing to pass a value to the 'type_params' parameter.*"
)

class TestChatWithLLM(unittest.TestCase):
    def setUp(self):
        # テスト用のAPIキーとシステムプロンプトを設定
        self.api_key = TEST_API_KEY
        self.system_prompt = "あなたは親切で役立つAIアシスタントです。"
        # ChatWithLLMインスタンスを初期化
        self.chat_bot = ChatWithLLM(api_key=self.api_key, system_prompt=self.system_prompt)

    @patch("chat_with_llm.ChatOpenAI")
    def test_initialize_llm_with_openai_provider(self, mock_chat_openai):
        # LLMの初期化メソッドが正しくOpenAIプロバイダーを使用するかをテスト
        self.chat_bot._initialize_llm()
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4o", openai_api_key=self.api_key
        )

    def test_update_system_prompt(self):
        # システムプロンプトの更新が正しく反映されるかをテスト
        new_prompt = "新しいプロンプト"
        self.chat_bot.update_system_prompt(new_prompt)
        self.assertEqual(self.chat_bot.system_prompt, new_prompt)

    def test_get_chat_history(self):
        # チャット履歴が正しく取得されるかをテスト
        self.chat_bot.history.add_user_message("こんにちは")
        self.chat_bot.history.add_ai_message("こんにちは！")
        history = self.chat_bot.get_chat_history()
        self.assertEqual(len(history), 2)  # 履歴のメッセージ数を確認
        self.assertEqual(history[0]["role"], "user")  # 最初のメッセージがユーザーからのものか確認
        self.assertEqual(history[0]["content"], "こんにちは")  # ユーザーのメッセージ内容を確認
        self.assertEqual(history[1]["role"], "assistant")  # 次のメッセージがAIからのものか確認
        self.assertEqual(history[1]["content"], "こんにちは！")  # AIのメッセージ内容を確認

    def test_clear_history(self):
        # チャット履歴が正しくクリアされるかをテスト
        self.chat_bot.history.add_user_message("こんにちは")
        self.chat_bot.clear_history()
        self.assertEqual(len(self.chat_bot.history.messages), 0)  # 履歴が空であることを確認

    @patch("chat_with_llm.ChatOpenAI")
    @patch("chat_with_llm.ChatWithLLM")  # ChatWithLLMをモック化
    def test_chat(self, mock_chat_with_llm, mock_chat_openai):
        # ChatOpenAIのモックを設定
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "こんにちは！"
        mock_chat_openai.return_value = mock_instance

        # ChatWithLLMのモックを設定
        mock_chat_bot = MagicMock()
        mock_chat_bot.chat.return_value = "こんにちは！"
        mock_chat_with_llm.return_value = mock_chat_bot

        # ユーザー入力に対するレスポンスをテスト
        user_input = "こんにちは"
        response = mock_chat_bot.chat(user_input)
        self.assertEqual(response, "こんにちは！")
        mock_chat_bot.chat.assert_called_once_with(user_input)

    @patch("chat_with_llm.save_chat_log")
    def test_save_chat_log(self, mock_save_chat_log):
        # チャット履歴を保存する機能をテスト
        self.chat_bot.history.add_user_message("こんにちは")
        self.chat_bot.history.add_ai_message("こんにちは！")
        chat_history = self.chat_bot.get_chat_history()

        # モックされたsave_chat_logを呼び出し
        mock_save_chat_log.return_value = "/path/to/log.txt"
        log_path = mock_save_chat_log(chat_history)
        self.assertEqual(log_path, "/path/to/log.txt")
        mock_save_chat_log.assert_called_once_with(chat_history)

    @patch("chat_with_llm.load_chat_log")
    def test_load_chat_log(self, mock_load_chat_log):
        # ログファイルからチャット履歴を読み込む機能をテスト
        mock_load_chat_log.return_value = [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは！"}
        ]
        filepath = "/path/to/log.txt"
        loaded_history = mock_load_chat_log(filepath)

        self.assertEqual(len(loaded_history), 2)
        self.assertEqual(loaded_history[0]["role"], "user")
        self.assertEqual(loaded_history[0]["content"], "こんにちは")
        self.assertEqual(loaded_history[1]["role"], "assistant")
        self.assertEqual(loaded_history[1]["content"], "こんにちは！")
        mock_load_chat_log.assert_called_once_with(filepath)

if __name__ == "__main__":
    # テストを実行
    unittest.main()