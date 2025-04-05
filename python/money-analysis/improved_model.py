import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os
import time

# 開始時間を記録
start_time = time.time()

print("複数のモデルを試して最適なモデルを選択します...")

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

# カテゴリカル変数と数値変数を分離
categorical_features = [col for col in features if X[col].dtype == 'object']
numeric_features = [col for col in features if X[col].dtype != 'object']

# 前処理パイプラインを作成
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# 試すモデルのリスト
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'SVR': SVR(),
    'MLP': MLPRegressor(random_state=42, max_iter=1000)
}

# 各モデルのハイパーパラメータグリッド
param_grids = {
    'RandomForest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 10]
    },
    'AdaBoost': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 1.0]
    },
    'ElasticNet': {
        'regressor__alpha': [0.1, 1.0, 10.0],
        'regressor__l1_ratio': [0.1, 0.5, 0.9]
    },
    'Lasso': {
        'regressor__alpha': [0.1, 1.0, 10.0]
    },
    'Ridge': {
        'regressor__alpha': [0.1, 1.0, 10.0]
    },
    'SVR': {
        'regressor__C': [0.1, 1.0, 10.0],
        'regressor__gamma': ['scale', 'auto'],
        'regressor__kernel': ['linear', 'rbf']
    },
    'MLP': {
        'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'regressor__learning_rate': ['constant', 'adaptive']
    }
}

# 結果を保存するための辞書
results = {}

# 各モデルを評価
for model_name, model in models.items():
    print(f"\n{model_name}モデルを評価中...")
    
    # パイプラインを作成
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # グリッドサーチを実行
    grid_search = GridSearchCV(
        pipeline,
        param_grids[model_name],
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    # モデルを訓練
    grid_search.fit(X_train, y_train)
    
    # 最適なモデルを取得
    best_model = grid_search.best_estimator_
    
    # テストデータで予測
    y_pred = best_model.predict(X_test)
    
    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 結果を表示
    print(f"最適なパラメータ: {grid_search.best_params_}")
    print(f"平均二乗誤差 (MSE): {mse:.2f}")
    print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
    print(f"決定係数 (R²): {r2:.2f}")
    
    # 結果を保存
    results[model_name] = {
        'model': best_model,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'best_params': grid_search.best_params_
    }

# 最も精度の高いモデルを選択（R²が最も高いモデル）
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_result = results[best_model_name]

print("\n===== 最適なモデル =====")
print(f"モデル: {best_model_name}")
print(f"平均二乗誤差 (MSE): {best_result['mse']:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {best_result['rmse']:.2f}")
print(f"決定係数 (R²): {best_result['r2']:.2f}")
print(f"最適なパラメータ: {best_result['best_params']}")

# 全モデルの比較グラフを作成
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
r2_scores = [results[model]['r2'] for model in model_names]
rmse_scores = [results[model]['rmse'] for model in model_names]

# R²スコアのグラフ
plt.subplot(1, 2, 1)
bars = plt.bar(model_names, r2_scores)
plt.title('各モデルのR²スコア比較')
plt.ylabel('R²スコア')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)  # R²は通常0〜1の範囲
# 最良のモデルを強調表示
best_index = model_names.index(best_model_name)
bars[best_index].set_color('red')

# RMSEスコアのグラフ
plt.subplot(1, 2, 2)
bars = plt.bar(model_names, rmse_scores)
plt.title('各モデルのRMSE比較')
plt.ylabel('RMSE')
plt.xticks(rotation=45, ha='right')
# 最良のモデルを強調表示
bars[best_index].set_color('red')

plt.tight_layout()
plt.savefig('model_comparison.png')
print("モデル比較グラフを 'model_comparison.png' として保存しました。")

# 最適なモデルを使用して予測値と実際の値の比較グラフを作成
best_model = best_result['model']
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title(f'予測値 vs 実際の値 ({best_model_name}モデル)')
plt.savefig('best_model_prediction.png')
print("最適なモデルの予測値と実際の値の比較グラフを 'best_model_prediction.png' として保存しました。")

# 最適なモデルを保存
model_filename = 'best_prediction_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\n最適なモデルを '{model_filename}' として保存しました。")

# 予測用のスクリプトを作成
predict_script = f'''
import pandas as pd
import joblib
import sys
import os
import numpy as np

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
    model_path = 'best_prediction_model.pkl'
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{{model_path}}' が見つかりません。")
        return None
    
    model = joblib.load(model_path)
    
    # 特徴量のリスト
    features = ["ISS区分", "部", "部門", "担当", "投資_リース"]
    
    # Excelファイルを読み込む
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"エラー: Excelファイルの読み込みに失敗しました。{{e}}")
        return None
    
    # 必要な特徴量が含まれているか確認
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"エラー: 以下の特徴量がデータに含まれていません: {{missing_features}}")
        return None
    
    # 予測を実行
    try:
        predictions = model.predict(df[features])
        
        # 結果をデータフレームに追加
        result_df = df.copy()
        result_df['予測_着地見込み額合計'] = predictions.astype(int)  # 整数に変換
        
        # 実際の値がある場合は差分を計算
        if '着地見込み額合計' in df.columns:
            result_df['差分'] = df['着地見込み額合計'] - result_df['予測_着地見込み額合計']
            result_df['誤差率(%)'] = (result_df['差分'] / df['着地見込み額合計']) * 100
        
        return result_df
    except Exception as e:
        print(f"エラー: 予測の実行中にエラーが発生しました。{{e}}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python improved_predict.py <Excelファイルのパス>")
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
        output_file = "improved_prediction_results.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\\n予測結果を {{output_file}} に保存しました。")
        
        # 予測の統計情報を表示（実際の値がある場合）
        if '着地見込み額合計' in result_df.columns:
            mae = np.abs(result_df['差分']).mean()
            rmse = np.sqrt((result_df['差分'] ** 2).mean())
            r2 = 1 - ((result_df['差分'] ** 2).sum() / ((result_df['着地見込み額合計'] - result_df['着地見込み額合計'].mean()) ** 2).sum())
            
            print("\\n予測の統計情報:")
            print(f"平均絶対誤差 (MAE): {{mae:.2f}}")
            print(f"平方根平均二乗誤差 (RMSE): {{rmse:.2f}}")
            print(f"決定係数 (R²): {{r2:.2f}}")
'''

with open('improved_predict.py', 'w', encoding='utf-8') as f:
    f.write(predict_script)

print("\n予測用スクリプト 'improved_predict.py' を作成しました。")
print("使用方法: python improved_predict.py <予測したいExcelファイルのパス>")

# 実行時間を表示
end_time = time.time()
execution_time = end_time - start_time
print(f"\n実行時間: {execution_time:.2f}秒")
