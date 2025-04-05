import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import joblib
import time
import os

# 開始時間を記録
start_time = time.time()

print("特徴量エンジニアリングを強化したモデルを作成します...")

# Excelファイルを読み込む
file_path = 'data-sample.xlsx'
df = pd.read_excel(file_path)

# データの詳細情報を表示
print("\nデータの詳細情報:")
print(f"データ形状: {df.shape}")
print(f"カラム: {df.columns.tolist()}")
print("\nデータ型:")
print(df.dtypes)

# 特徴量と目的変数を定義
features = ["ISS区分", "部", "部門", "担当", "投資_リース"]
target = "着地見込み額合計"

# データの前処理
# 欠損値の確認と処理
print("\n欠損値の数:")
print(df[features + [target]].isnull().sum())

# 欠損値がある場合は処理する
df_clean = df.dropna(subset=features + [target])
print(f"欠損値除去後のデータ数: {df_clean.shape[0]}")

# 特徴量と目的変数を分離
X = df_clean[features]
y = df_clean[target]

# データの詳細な分析
print("\n目的変数の統計情報:")
print(y.describe())

# 外れ値の検出
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ((y < lower_bound) | (y > upper_bound)).sum()
print(f"外れ値の数: {outliers}")

# カテゴリカル変数と数値変数を分離
categorical_features = [col for col in features if X[col].dtype == 'object']
numeric_features = [col for col in features if X[col].dtype != 'object']

print(f"\nカテゴリカル特徴量: {categorical_features}")
print(f"数値特徴量: {numeric_features}")

# カテゴリカル変数の値を確認
for col in categorical_features:
    print(f"\n{col}の値の分布:")
    print(X[col].value_counts())

# 数値変数の統計情報
for col in numeric_features:
    print(f"\n{col}の統計情報:")
    print(X[col].describe())

# 特徴量エンジニアリング関数
def create_engineered_features(X):
    """
    特徴量エンジニアリングを行う関数
    
    Parameters:
    -----------
    X : pandas.DataFrame
        元の特徴量
    
    Returns:
    --------
    X_new : pandas.DataFrame
        エンジニアリングされた特徴量
    """
    X_new = X.copy()
    
    # カテゴリカル変数の組み合わせを作成
    if "ISS区分" in X.columns and "部門" in X.columns:
        X_new["ISS区分_部門"] = X["ISS区分"] + "_" + X["部門"]
    
    # 数値特徴量の変換
    if "担当" in X.columns:
        X_new["担当_log"] = np.log1p(X["担当"])  # log(1+x)変換
        X_new["担当_squared"] = X["担当"] ** 2  # 二乗変換
    
    return X_new

# 特徴量エンジニアリングを適用
X_engineered = create_engineered_features(X)

print("\n特徴量エンジニアリング後の特徴量:")
print(X_engineered.columns.tolist())

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

# カテゴリカル変数と数値変数を分離（エンジニアリング後）
categorical_features_eng = [col for col in X_engineered.columns if X_engineered[col].dtype == 'object']
numeric_features_eng = [col for col in X_engineered.columns if X_engineered[col].dtype != 'object']

print(f"\nエンジニアリング後のカテゴリカル特徴量: {categorical_features_eng}")
print(f"エンジニアリング後の数値特徴量: {numeric_features_eng}")

# 前処理パイプラインを作成
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_eng),
        ('num', RobustScaler(), numeric_features_eng)  # RobustScalerを使用して外れ値の影響を軽減
    ])

# MLPモデルのパイプラインを作成
mlp_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(random_state=42, max_iter=5000))  # イテレーション数を増やす
])

# Ridgeモデルのパイプラインを作成
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(random_state=42))
])

# RandomForestモデルのパイプラインを作成
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# MLPモデルのハイパーパラメータグリッド
mlp_param_grid = {
    'regressor__hidden_layer_sizes': [(100,), (100, 50), (100, 100), (200, 100, 50)],
    'regressor__alpha': [0.0001, 0.001, 0.01],
    'regressor__learning_rate': ['constant', 'adaptive'],
    'regressor__activation': ['relu', 'tanh'],
    'regressor__solver': ['adam', 'sgd']
}

# Ridgeモデルのハイパーパラメータグリッド
ridge_param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# RandomForestモデルのハイパーパラメータグリッド
rf_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

# 交差検証の設定
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# グリッドサーチを実行（MLPモデル）
print("\nMLPモデルのグリッドサーチを実行中...")
mlp_grid_search = GridSearchCV(
    mlp_pipeline,
    mlp_param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# モデルを訓練
mlp_grid_search.fit(X_train, y_train)

# 最適なMLPモデルを取得
best_mlp_model = mlp_grid_search.best_estimator_
best_mlp_params = mlp_grid_search.best_params_

# テストデータで予測
y_pred_mlp = best_mlp_model.predict(X_test)

# MLPモデルの評価
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# 結果を表示
print("\n===== MLPモデルの結果 =====")
print(f"最適なパラメータ: {best_mlp_params}")
print(f"平均二乗誤差 (MSE): {mse_mlp:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse_mlp:.2f}")
print(f"決定係数 (R²): {r2_mlp:.2f}")

# グリッドサーチを実行（Ridgeモデル）
print("\nRidgeモデルのグリッドサーチを実行中...")
ridge_grid_search = GridSearchCV(
    ridge_pipeline,
    ridge_param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# モデルを訓練
ridge_grid_search.fit(X_train, y_train)

# 最適なRidgeモデルを取得
best_ridge_model = ridge_grid_search.best_estimator_
best_ridge_params = ridge_grid_search.best_params_

# テストデータで予測
y_pred_ridge = best_ridge_model.predict(X_test)

# Ridgeモデルの評価
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# 結果を表示
print("\n===== Ridgeモデルの結果 =====")
print(f"最適なパラメータ: {best_ridge_params}")
print(f"平均二乗誤差 (MSE): {mse_ridge:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse_ridge:.2f}")
print(f"決定係数 (R²): {r2_ridge:.2f}")

# グリッドサーチを実行（RandomForestモデル）
print("\nRandomForestモデルのグリッドサーチを実行中...")
rf_grid_search = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# モデルを訓練
rf_grid_search.fit(X_train, y_train)

# 最適なRandomForestモデルを取得
best_rf_model = rf_grid_search.best_estimator_
best_rf_params = rf_grid_search.best_params_

# テストデータで予測
y_pred_rf = best_rf_model.predict(X_test)

# RandomForestモデルの評価
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 結果を表示
print("\n===== RandomForestモデルの結果 =====")
print(f"最適なパラメータ: {best_rf_params}")
print(f"平均二乗誤差 (MSE): {mse_rf:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse_rf:.2f}")
print(f"決定係数 (R²): {r2_rf:.2f}")

# アンサンブルモデルの作成
print("\nアンサンブルモデルを作成中...")

# VotingRegressorの作成
voting_regressor = VotingRegressor([
    ('mlp', best_mlp_model),
    ('ridge', best_ridge_model),
    ('rf', best_rf_model)
])

# アンサンブルモデルを訓練
voting_regressor.fit(X_train, y_train)

# テストデータで予測
y_pred_voting = voting_regressor.predict(X_test)

# アンサンブルモデルの評価
mse_voting = mean_squared_error(y_test, y_pred_voting)
rmse_voting = np.sqrt(mse_voting)
r2_voting = r2_score(y_test, y_pred_voting)

# 結果を表示
print("\n===== アンサンブルモデルの結果 =====")
print(f"平均二乗誤差 (MSE): {mse_voting:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse_voting:.2f}")
print(f"決定係数 (R²): {r2_voting:.2f}")

# 全モデルの比較
models = {
    'MLP': {'model': best_mlp_model, 'mse': mse_mlp, 'rmse': rmse_mlp, 'r2': r2_mlp},
    'Ridge': {'model': best_ridge_model, 'mse': mse_ridge, 'rmse': rmse_ridge, 'r2': r2_ridge},
    'RandomForest': {'model': best_rf_model, 'mse': mse_rf, 'rmse': rmse_rf, 'r2': r2_rf},
    'Ensemble': {'model': voting_regressor, 'mse': mse_voting, 'rmse': rmse_voting, 'r2': r2_voting}
}

# 最も精度の高いモデルを選択（R²が最も高いモデル）
best_model_name = max(models, key=lambda x: models[x]['r2'])
best_result = models[best_model_name]

print("\n===== 最適なモデル =====")
print(f"モデル: {best_model_name}")
print(f"平均二乗誤差 (MSE): {best_result['mse']:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {best_result['rmse']:.2f}")
print(f"決定係数 (R²): {best_result['r2']:.2f}")

# 全モデルの比較グラフを作成
plt.figure(figsize=(12, 6))
model_names = list(models.keys())
r2_scores = [models[model]['r2'] for model in model_names]
rmse_scores = [models[model]['rmse'] for model in model_names]

# R²スコアのグラフ
plt.subplot(1, 2, 1)
bars = plt.bar(model_names, r2_scores)
plt.title('各モデルのR²スコア比較')
plt.ylabel('R²スコア')
plt.ylim(0, 1)  # R²は通常0〜1の範囲
# 最良のモデルを強調表示
best_index = model_names.index(best_model_name)
bars[best_index].set_color('red')

# RMSEスコアのグラフ
plt.subplot(1, 2, 2)
bars = plt.bar(model_names, rmse_scores)
plt.title('各モデルのRMSE比較')
plt.ylabel('RMSE')
# 最良のモデルを強調表示
bars[best_index].set_color('red')

plt.tight_layout()
plt.savefig('feature_engineering_model_comparison.png')
print("モデル比較グラフを 'feature_engineering_model_comparison.png' として保存しました。")

# 最適なモデルを使用して予測値と実際の値の比較グラフを作成
best_model = best_result['model']

# 予測を実行
y_pred = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title(f'予測値 vs 実際の値 ({best_model_name}モデル)')
plt.savefig('feature_engineering_model_prediction.png')
print("最適なモデルの予測値と実際の値の比較グラフを 'feature_engineering_model_prediction.png' として保存しました。")

# 特徴量の重要度を計算（RandomForestモデルの場合）
if best_model_name == 'RandomForest':
    # 特徴量名を取得
    feature_names = best_rf_model.named_steps['preprocessor'].get_feature_names_out()
    
    # 特徴量の重要度を取得
    importances = best_rf_model.named_steps['regressor'].feature_importances_
    
    # 特徴量の重要度をソート
    indices = np.argsort(importances)[::-1]
    
    # 上位20個の特徴量を表示
    top_n = min(20, len(feature_names))
    plt.figure(figsize=(12, 8))
    plt.title('特徴量の重要度')
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_engineering_importance.png')
    print("特徴量の重要度グラフを 'feature_engineering_importance.png' として保存しました。")
    
    print("\n特徴量の重要度 (上位10):")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
else:
    # 特徴量の重要度を計算（MLPでは直接的な方法はないため、順列重要度を使用）
    from sklearn.inspection import permutation_importance
    
    print("\n特徴量の重要度を計算中...")
    result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # 特徴量の重要度をソート
    feature_importance = pd.DataFrame({
        'feature': X_engineered.columns,
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
    plt.savefig('feature_engineering_importance.png')
    print("特徴量の重要度グラフを 'feature_engineering_importance.png' として保存しました。")
    
    print("\n特徴量の重要度:")
    print(feature_importance)

# 最適なモデルと特徴量エンジニアリング関数を保存
model_data = {
    'model': best_model,
    'feature_engineering_func': create_engineered_features,
    'model_name': best_model_name
}

model_filename = 'feature_engineering_model.pkl'
joblib.dump(model_data, model_filename)
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
    model_path = 'feature_engineering_model.pkl'
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{model_path}' が見つかりません。")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_engineering_func = model_data['feature_engineering_func']
    model_name = model_data['model_name']
    
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
        # 特徴量エンジニアリングを適用
        X = df[features]
        X_engineered = feature_engineering_func(X)
        
        # 予測を実行
        predictions = model.predict(X_engineered)
        
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
        print("使用方法: python feature_engineering_predict.py <Excelファイルのパス>")
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
        output_file = "feature_engineering_prediction_results.xlsx"
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

with open('feature_engineering_predict.py', 'w', encoding='utf-8') as f:
    f.write(predict_script)

print("\n予測用スクリプト 'feature_engineering_predict.py' を作成しました。")
print("使用方法: python feature_engineering_predict.py <予測したいExcelファイルのパス>")

# 実行時間を表示
end_time = time.time()
execution_time = end_time - start_time
print(f"\n実行時間: {execution_time:.2f}秒")
