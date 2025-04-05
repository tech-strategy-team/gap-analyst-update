import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import matplotlib.pyplot as plt
import joblib

# Excelファイルを読み込む
file_path = 'data-sample.xlsx'
df = pd.read_excel(file_path)

# 特徴量と目的変数を定義
features = ["ISS区分", "部", "部門", "担当", "投資_リース"]
target = "着地見込み額合計"

# データの前処理
# 欠損値の確認と処理
print("欠損値の数:")
print(df[features + [target]].isnull().sum())

# 欠損値がある場合は処理する
df_clean = df.dropna(subset=features + [target])
print(f"欠損値除去後のデータ数: {df_clean.shape[0]}")

# 特徴量と目的変数を分離
X = df_clean[features]
y = df_clean[target]

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# カテゴリカル変数の前処理パイプラインを作成
categorical_features = [col for col in features if X[col].dtype == 'object']
numeric_features = [col for col in features if X[col].dtype != 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# モデルパイプラインを作成
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# グリッドサーチのパラメータ
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

# グリッドサーチを実行
print("グリッドサーチを実行中...")
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最適なパラメータを表示
print("最適なパラメータ:")
print(grid_search.best_params_)

# 最適なモデルを取得
best_model = grid_search.best_estimator_

# テストデータで予測
y_pred = best_model.predict(X_test)

# モデルの評価
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"平均二乗誤差 (MSE): {mse:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
print(f"決定係数 (R²): {r2:.2f}")

# 予測値と実際の値の比較グラフを作成
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('予測値 vs 実際の値')
plt.savefig('prediction_vs_actual.png')
print("予測値と実際の値の比較グラフを 'prediction_vs_actual.png' として保存しました。")

# 特徴量の重要度を計算
if hasattr(best_model['regressor'], 'feature_importances_'):
    # 特徴量名を取得
    ohe = best_model['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    
    # 数値特徴量の名前を追加
    feature_names = np.concatenate([cat_feature_names, np.array(numeric_features)])
    
    # 特徴量の重要度を取得
    importances = best_model['regressor'].feature_importances_
    
    # 特徴量の重要度をソート
    indices = np.argsort(importances)[::-1]
    
    # 上位20個の特徴量を表示
    top_n = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))
    plt.title('特徴量の重要度')
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特徴量の重要度グラフを 'feature_importance.png' として保存しました。")
    
    print("\n特徴量の重要度 (上位10):")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# モデルを保存
model_filename = 'prediction_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\nモデルを '{model_filename}' として保存しました。")

# 予測用のスクリプトを作成
predict_script = '''
import pandas as pd
import joblib
import sys
import os

def predict_from_excel(excel_file):
    """
    Excelファイルから予測を行う関数
    
    Parameters:
    -----------
    excel_file : str
        予測したいデータが含まれるExcelファイルのパス
    
    Returns:
    --------
    result_df : pandas.DataFrame
        予測結果を含むデータフレーム
    """
    # モデルを読み込む
    model_path = 'prediction_model.pkl'
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。")
        return None
    
    model = joblib.load(model_path)
    
    # 特徴量のリスト
    features = ["ISS区分", "部", "部門", "担当", "投資_リース"]
    
    # Excelファイルを読み込む
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"エラー: Excelファイルの読み込みに失敗しました。{e}")
        return None
    
    # 必要な特徴量が含まれているか確認
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"エラー: 以下の特徴量がデータに含まれていません: {missing_features}")
        return None
    
    # 予測を実行
    try:
        predictions = model.predict(df[features])
        
        # 結果をデータフレームに追加
        result_df = df.copy()
        result_df['予測_着地見込み額合計'] = predictions
        
        # 実際の値がある場合は差分を計算
        if '着地見込み額合計' in df.columns:
            result_df['差分'] = df['着地見込み額合計'] - predictions
            result_df['誤差率(%)'] = (result_df['差分'] / df['着地見込み額合計']) * 100
        
        return result_df
    except Exception as e:
        print(f"エラー: 予測の実行中にエラーが発生しました。{e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <Excelファイルのパス>")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    result_df = predict_from_excel(excel_file)
    
    if result_df is not None:
        # 結果を表示
        print("\\n予測結果 (最初の5行):")
        if '着地見込み額合計' in result_df.columns:
            print(result_df[['ISS区分', '部門', '担当', '着地見込み額合計', '予測_着地見込み額合計', '差分', '誤差率(%)']].head())
        else:
            print(result_df[['ISS区分', '部門', '担当', '予測_着地見込み額合計']].head())
        
        # 結果をExcelファイルに保存
        output_file = "prediction_results.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\\n予測結果を {output_file} に保存しました。")
'''

with open('predict.py', 'w', encoding='utf-8') as f:
    f.write(predict_script)

print("\n予測用スクリプト 'predict.py' を作成しました。")
print("使用方法: python predict.py <予測したいExcelファイルのパス>")

# 予測モデルの使用例
print("\n予測モデルの使用例:")
# テストデータの最初の5行を使用
example_data = X_test.head(5)
predictions = best_model.predict(example_data)

# 結果を表示
result_df = pd.DataFrame({
    '予測値': predictions,
    '実際の値': y_test.iloc[:5].values,
    '差分': y_test.iloc[:5].values - predictions,
    '誤差率(%)': ((y_test.iloc[:5].values - predictions) / y_test.iloc[:5].values) * 100
})
print(result_df)
