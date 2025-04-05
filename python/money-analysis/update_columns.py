import pandas as pd
import random
import numpy as np

# Excelファイルを読み込む
file_path = '/home/shuka/n-group/python/money-analysis/data-sample.xlsx'
df = pd.read_excel(file_path)

# 部門の選択肢（a~h）
department_options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# 担当の選択肢（1~10）
staff_options = list(range(1, 11))

# 各行の部門と担当をランダムに設定
for i in range(len(df)):
    df.at[i, '部門'] = random.choice(department_options)
    df.at[i, '担当'] = random.choice(staff_options)

# 更新したデータフレームを保存
df.to_excel(file_path, index=False)

print(f"「部門」列にa~hのアルファベット、「担当」列に1~10の数字をランダムに100行分設定しました。")
print(f"\n部門の内訳:")
print(df['部門'].value_counts())
print(f"\n担当の内訳:")
print(df['担当'].value_counts())
