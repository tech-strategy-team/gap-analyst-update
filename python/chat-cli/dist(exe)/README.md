# Python Scriptのexeファイル化手順

このREADMEは、Pythonスクリプト `chat_with_llm.py` をWindows環境でexeファイルに変換し、配布可能にするための手順をまとめたものです。この手順書は、プログラミング初心者でも簡単にexeファイルを作成し、配布できるように設計されています。

- Windows上に同じPython環境をセットアップする
- 必要な依存パッケージをインストールする
- PyInstallerを使用してexeファイルを作成する

## 1. 環境準備

### 1.1 Pythonのインストール

- **バージョン**: Python 3.12 を使用
- **インストール方法**:
   公式インストーラーを使用（インストール時に「Add Python to PATH」にチェックを入れてください）

### 1.2 必要なパッケージのインストール

#### 使用するライブラリ

- `python-dotenv`
- `PyPDF2`
- `openai`
- `prompt_toolkit`
- `langchain_openai`
- `langchain_core`
- `chardet`

#### exeファイル化に必要なライブラリ

- `PyInstaller`

#### コマンド例

1. **各ライブラリのインストール**
    Windowsのコマンドプロンプトを起動し、以下のコマンドを実行してください:

`pip install python-dotenv`
`pip install PyPDF2`
`pip install openai`
`pip install prompt_toolkit`
`pip install langchain_openai`
`pip install langchain_core`
`pip install chardet`

PyInstallerのインストール

`pip install pyinstaller`

### 1.3 環境変数設定について

- スクリプトは、`OPENAI_API_KEY` が未設定の場合、CLI上で入力を促す仕様になっています。

------

## 2. exeファイルの生成

1. **スクリプトのあるディレクトリに移動**
    対象ファイル: `chat_with_llm.py`
2. **PyInstallerによるビルド実行**

pyinstaller --onefile chat_with_llm.py

`--onefile` オプションにより、すべての依存関係が1つの実行ファイルにまとめられます。

生成されたexeファイルは `dist` フォルダ内に作成されます。

## 3. 実行時の環境変数設定方法

### 方法1：バッチファイルの利用

以下の内容で `run_chat.bat` などのバッチファイルを作成し、環境変数を設定後にexeを実行します。

`@echo off`
`set OPENAI_API_KEY=あなたのAPIキー`
`dist\chat_with_llm.exe`
`pause`

### 方法2：システム環境変数の設定

- Windowsのシステム設定から `OPENAI_API_KEY` を設定することで、exe実行時に毎回キー入力を求められるのを防げます。

## 4. 動作確認

- `dist\chat_with_llm.exe` をダブルクリック、または作成したバッチファイルを実行して、正常に動作するか確認してください。
- エラーが発生した場合は、エラーメッセージを確認し、必要に応じて `--hidden-import` オプションなどで不足している依存ライブラリを追加指定してください。