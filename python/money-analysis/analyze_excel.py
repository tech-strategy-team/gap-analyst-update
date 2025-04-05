import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Excelファイルを読み込む
file_path = 'data-sample.xlsx'
df = pd.read_excel(file_path)

# データの基本情報を表示
print("データの基本情報:")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print("\n列名:")
print(df.columns.tolist())
print("\nデータ型:")
print(df.dtypes)
print("\n最初の5行:")
print(df.head())

# 欠損値の確認
print("\n欠損値の数:")
print(df.isnull().sum())

# 特徴量と目的変数の確認
features = ["ISS区分", "部", "部門", "担当", "投資_リース"]
target = "着地見込み額合計"

# 特徴量の種類を確認
print("\n特徴量の種類:")
for feature in features:
    if feature in df.columns:
        print(f"{feature}: {df[feature].nunique()} 種類")
        print(f"例: {df[feature].unique()[:5]}")
    else:
        print(f"{feature}: 列が存在しません")

# 目的変数の統計情報
if target in df.columns:
    print(f"\n目的変数 '{target}' の統計情報:")
    print(df[target].describe())
else:
    print(f"\n目的変数 '{target}' の列が存在しません")
