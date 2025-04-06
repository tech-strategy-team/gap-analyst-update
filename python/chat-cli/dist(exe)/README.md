# Python Scriptのexeファイル化手順

このREADMEは、Pythonスクリプト `chat_with_llm.py` をWindows環境でexe化する手順と、その際の環境設定の記録です。

Windows環境で実行することを目的としており、以下の作業を実施しています。

- Windows上に同じPython環境をセットアップする
- 必要な依存パッケージをインストールする
- PyInstallerを使用してexeファイルを作成する

---

## 1. 環境準備

### 1.1 Pythonのインストール
- **バージョン**：Python 3.12 を使用
- **インストール方法**：公式インストーラーを使用（ユーザのPATH設定済み）

### 1.2 パッケージのインストール
以下のコマンドで必要なパッケージをインストールしました。  
※ユーザインストールの場合、ScriptsディレクトリがPATHに追加されていないため注意してください。

```bash
pip install python-dotenv openai langchain langchain_core langchain_community langchain-openai prompt_toolkit pyinstaller


pyinstaller --onefile --noconsole chat_with_llm.py