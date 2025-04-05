import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib
import time
import os

# 開始時間を記録
start_time = time.time()

print("MLPモデルの精度向上を試みます...")

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

# データの詳細な分析
print("\nデータの詳細な分析:")
print(f"目的変数の統計情報:\n{y.describe()}")

# 外れ値の検出
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ((y < lower_bound) | (y > upper_bound)).sum()
print(f"外れ値の数: {outliers}")

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# カテゴリカル変数と数値変数を分離
categorical_features = [col for col in features if X[col].dtype == 'object']
numeric_features = [col for col in features if X[col].dtype != 'object']

print(f"\nカテゴリカル特徴量: {categorical_features}")
print(f"数値特徴量: {numeric_features}")

# 前処理パイプラインを作成
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# MLPモデルのパイプラインを作成
mlp_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(random_state=42, max_iter=2000))
])

# ハイパーパラメータグリッド
param_grid = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'regressor__alpha': [0.0001, 0.001, 0.01],
    'regressor__learning_rate': ['constant', 'adaptive'],
    'regressor__activation': ['relu', 'tanh']
}

# グリッドサーチを実行
print("\nグリッドサーチを実行中...")
grid_search = GridSearchCV(
    mlp_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# モデルを訓練
grid_search.fit(X_train, y_train)

# 最適なモデルを取得
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# テストデータで予測
y_pred = best_model.predict(X_test)

# モデルの評価
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 結果を表示
print("\n===== 最適なモデル =====")
print(f"最適なパラメータ: {best_params}")
print(f"平均二乗誤差 (MSE): {mse:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
print(f"決定係数 (R²): {r2:.2f}")

# 予測値と実際の値の比較グラフを作成
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title('予測値 vs 実際の値 (Enhanced MLP)')
plt.savefig('enhanced_mlp_prediction.png')
print("予測値と実際の値の比較グラフを 'enhanced_mlp_prediction.png' として保存しました。")

# 特徴量の重要度を計算（MLPでは直接的な方法はないため、順列重要度を使用）
from sklearn.inspection import permutation_importance

print("\n特徴量の重要度を計算中...")
result = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# 特徴量の重要度をソート
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 特徴量の重要度グラフを作成
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('特徴量')
plt.ylabel('重要度')
plt.title('特徴量の重要度')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('enhanced_mlp_feature_importance.png')
print("特徴量の重要度グラフを 'enhanced_mlp_feature_importance.png' として保存しました。")

# 最適なモデルを保存
model_filename = 'enhanced_mlp_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\n最適なモデルを '{model_filename}' として保存しました。")

# 予測用のスクリプトを作成
predict_script = '''
import pandas as pd
import numpy as np
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
    model_path = 'enhanced_mlp_model.pkl'
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
        result_df['予測_着地見込み額合計'] = predictions.astype(int)  # 整数に変換
        
        # 実際の値がある場合は差分を計算
        if '着地見込み額合計' in df.columns:
            result_df['差分'] = df['着地見込み額合計'] - result_df['予測_着地見込み額合計']
            result_df['誤差率(%)'] = (result_df['差分'] / df['着地見込み額合計']) * 100
        
        return result_df
    except Exception as e:
        print(f"エラー: 予測の実行中にエラーが発生しました。{e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python enhanced_mlp_predict.py <Excelファイルのパス>")
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
        output_file = "enhanced_mlp_prediction_results.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\\n予測結果を {output_file} に保存しました。")
        
        # 予測の統計情報を表示（実際の値がある場合）
        if '着地見込み額合計' in result_df.columns:
            mae = np.abs(result_df['差分']).mean()
            rmse = np.sqrt((result_df['差分'] ** 2).mean())
            r2 = 1 - ((result_df['差分'] ** 2).sum() / ((result_df['着地見込み額合計'] - result_df['着地見込み額合計'].mean()) ** 2).sum())
            
            print("\\n予測の統計情報:")
            print(f"平均絶対誤差 (MAE): {mae:.2f}")
            print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
            print(f"決定係数 (R²): {r2:.2f}")
'''

with open('enhanced_mlp_predict.py', 'w', encoding='utf-8') as f:
    f.write(predict_script)

print("\n予測用スクリプト 'enhanced_mlp_predict.py' を作成しました。")
print("使用方法: python enhanced_mlp_predict.py <予測したいExcelファイルのパス>")

# 実行時間を表示
end_time = time.time()
execution_time = end_time - start_time
print(f"\n実行時間: {execution_time:.2f}秒")
