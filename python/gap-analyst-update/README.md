# gap-analyst-update

Streamlit で乖離分析を行うアプリ（`app_updated.py`）。

## セットアップ

```bash
cd python/gap-analyst-update
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 実行方法

```bash
source .venv/bin/activate
streamlit run app_updated.py
```

## 補足

- `.venv/` は仮想環境として作成済みです。別環境で作り直す場合は `.venv/` を削除して上記手順を再実行してください。
- 依存ライブラリは `requirements.txt` にまとめています。
