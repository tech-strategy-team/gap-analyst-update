import pandas as pd
import random
import numpy as np

# Excelファイルを読み込む
file_path = '/home/shuka/n-group/python/money-analysis/data-sample.xlsx'
df = pd.read_excel(file_path)

# 各行の修正計画_最終FB額と着地見込み額合計を設定
for i in range(len(df)):
    # 修正計画_最終FB額に1から2000までのランダムな数字を設定
    fb_amount = random.randint(1, 2000)
    df.at[i, '修正計画_最終FB額'] = fb_amount
    
    # 部門の値に基づいて係数を決定
    department = df.at[i, '部門']
    if department == 'a':
        # aの場合：0.6~0.9
        coefficient = random.uniform(0.6, 0.9)
    elif department == 'b':
        # bの場合：0.8~0.95
        coefficient = random.uniform(0.8, 0.95)
    elif department == 'c':
        # cの場合：0.9
        coefficient = 0.9
    else:
        # それ以外の場合：0.9~1.0
        coefficient = random.uniform(0.9, 1.0)
    
    # 着地見込み額合計を計算
    expected_amount = round(fb_amount * coefficient)
    df.at[i, '着地見込み額合計'] = expected_amount

# 更新したデータフレームを保存
df.to_excel(file_path, index=False)

print(f"「修正計画_最終FB額」の列に1から2000までの数字をランダムに設定しました。")
print(f"「着地見込み額合計」の列に、部門に応じた係数をかけた金額を設定しました。")
print("\n部門ごとの係数：")
print("- 部門a：0.6~0.9")
print("- 部門b：0.8~0.95")
print("- 部門c：0.9")
print("- その他：0.9~1.0")

# サンプルデータを表示
print("\nサンプルデータ（最初の10行）：")
print(df[['ISS区分', '部門', '担当', '修正計画_最終FB額', '着地見込み額合計']].head(10))
