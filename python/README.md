# Pythonコーディングルール

Pythonコードを書く際には、以下のガイドラインに従うこと。

## PEP8に従う

Pythonの公式スタイルガイドであるPEP8に従うことを推奨します。  

参考：[PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)

## 自動フォーマッタ: black

`black`を使用することで、PEP8に準拠したコーディングスタイルを自動的に適用できます。  
インストール:  

```bash
pip install black
```

使用方法:  

```bash
black your_code.py
```

## コードスタイルチェック: flake8

`flake8`を使用して、PEP8に基づくコーディングスタイルのチェックを行います。  
インストール:  

```bash
pip install flake8
```

使用方法:  

```bash
flake8 your_code.py
```

## 型チェック: mypy

`mypy`を使用して、Type Hintingで記述した型情報を静的にチェックできます。  
インストール:  

```bash
pip install mypy
```

使用方法:  

```bash
mypy your_code.py
```

## その他の推奨事項

- **関数やクラスにドキュメンテーションを追加する**: `docstring`を使ってコードの意図を明確にしましょう。
- **変数名や関数名をわかりやすくする**: 意味のある名前を付けることで、コードの可読性を向上させます。
- **テストを記述する**: `pytest`などのテストフレームワークを使用して、コードの動作を保証しましょう。

## 推奨されるディレクトリ構成

以下は、ディレクトリ構成の例です。

```text
project_name/
├── README.md          # プロジェクトの概要
├── requirements.txt   # 必要なパッケージ
├── src/               # ソースコード
│   ├── __init__.py    # パッケージとして認識させるためのファイル
│   └── main.py        # エントリーポイント
├── tests/             # テストコード
│   ├── __init__.py
│   └── test_main.py
└── docs/              # ドキュメント (必要に応じて)
```
