import pandas as pd
import random
import numpy as np

# Excelファイルを読み込む
file_path = '/home/shuka/n-group/python/money-analysis/data-sample.xlsx'
df = pd.read_excel(file_path)

# ISS区分の選択肢
iss_options = ["研究", "建物", "システム", "その他"]

# 既存の行のISS区分をランダムに更新
for i in range(len(df)):
    df.at[i, 'ISS区分'] = random.choice(iss_options)

# 残りの90行を追加
new_rows = pd.DataFrame({
    'ISS区分': [random.choice(iss_options) for _ in range(90)]
})

# 既存のデータフレームと新しい行を結合
df = pd.concat([df, new_rows], ignore_index=True)

# 更新したデータフレームを保存
df.to_excel(file_path, index=False)

print(f"ISS区分の列に「研究」「建物」「システム」「その他」の4つの単語をランダムに100行分記入しました。")
print(f"ISS区分の内訳:")
print(df['ISS区分'].value_counts())
