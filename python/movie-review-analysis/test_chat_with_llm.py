import unittest
import warnings  # 警告フィルタ用
from unittest.mock import patch, MagicMock
from chat_with_llm import ChatWithLLM

# DeprecationWarningを無視
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Failing to pass a value to the 'type_params' parameter.*"
)

class TestChatWithLLM(unittest.TestCase):
    def setUp(self):
        # テスト用のAPIキーとシステムプロンプトを設定
        self.api_key = "test_api_key"
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

if __name__ == "__main__":
    # テストを実行
    unittest.main()