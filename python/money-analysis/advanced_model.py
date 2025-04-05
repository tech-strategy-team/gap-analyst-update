import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import time
import os

# 開始時間を記録
start_time = time.time()

print("高度な特徴量エンジニアリングと高度なモデルを使用して精度向上を試みます...")

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

# 外れ値を除外するかどうかを決定（今回は除外しない）
# df_no_outliers = df_clean[~((y < lower_bound) | (y > upper_bound))]
# X = df_no_outliers[features]
# y = df_no_outliers[target]
# print(f"外れ値除去後のデータ数: {len(y)}")

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# カテゴリカル変数と数値変数を分離
categorical_features = [col for col in features if X[col].dtype == 'object']
numeric_features = [col for col in features if X[col].dtype != 'object']

print(f"\nカテゴリカル特徴量: {categorical_features}")
print(f"数値特徴量: {numeric_features}")

# 特徴量エンジニアリング関数
def create_feature_engineering_pipeline(poly_degree=2, interaction_only=False):
    # カテゴリカル変数の前処理
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 数値変数の前処理
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False))
    ])
    
    # 前処理パイプラインを作成
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ])
    
    return preprocessor

# 高度なモデルの定義
def get_advanced_models():
    models = {
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        'XGBoost_with_poly': Pipeline([
            ('preprocessor', create_feature_engineering_pipeline(poly_degree=2, interaction_only=True)),
            ('regressor', xgb.XGBRegressor(random_state=42, n_jobs=-1))
        ]),
        'LightGBM_with_poly': Pipeline([
            ('preprocessor', create_feature_engineering_pipeline(poly_degree=2, interaction_only=True)),
            ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1))
        ])
    }
    return models

# 各モデルのハイパーパラメータグリッド
def get_param_grids():
    param_grids = {
        'XGBoost': {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1]
        },
        'LightGBM': {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        },
        'XGBoost_with_poly': {
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5],
            'regressor__n_estimators': [100, 200],
            'regressor__min_child_weight': [1, 3],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0],
            'regressor__gamma': [0, 0.1]
        },
        'LightGBM_with_poly': {
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5],
            'regressor__n_estimators': [100, 200],
            'regressor__num_leaves': [31, 50],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0],
            'regressor__reg_alpha': [0, 0.1],
            'regressor__reg_lambda': [0, 0.1]
        }
    }
    return param_grids

# 前処理パイプラインを作成
preprocessor = create_feature_engineering_pipeline(poly_degree=2, interaction_only=True)

# 高度なモデルを取得
models = get_advanced_models()
param_grids = get_param_grids()

# 結果を保存するための辞書
results = {}

# 各モデルを評価
for model_name, model in models.items():
    print(f"\n{model_name}モデルを評価中...")
    
    # XGBoostとLightGBMの基本モデルの場合は前処理を適用
    if model_name in ['XGBoost', 'LightGBM']:
        # 前処理を適用
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # グリッドサーチを実行
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # モデルを訓練
        grid_search.fit(X_train_transformed, y_train)
        
        # 最適なモデルを取得
        best_model = grid_search.best_estimator_
        
        # テストデータで予測
        y_pred = best_model.predict(X_test_transformed)
        
    else:  # パイプラインを使用するモデルの場合
        # グリッドサーチを実行
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
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
        'best_params': grid_search.best_params_,
        'is_pipeline': model_name not in ['XGBoost', 'LightGBM'],
        'preprocessor': preprocessor if model_name in ['XGBoost', 'LightGBM'] else None
    }

# アンサンブルモデルの作成
print("\nアンサンブルモデルを作成中...")

# 上位2つのモデルを選択
top_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:2]
top_model_names = [name for name, _ in top_models]
print(f"アンサンブルに使用するモデル: {top_model_names}")

# VotingRegressorの作成
estimators = []
for name, result in results.items():
    if name in top_model_names:
        if result['is_pipeline']:
            estimators.append((name, result['model']))
        else:
            # 前処理を含むパイプラインを作成
            pipeline = Pipeline([
                ('preprocessor', result['preprocessor']),
                ('regressor', result['model'])
            ])
            estimators.append((name, pipeline))

voting_regressor = VotingRegressor(estimators=estimators)
voting_regressor.fit(X_train, y_train)
y_pred_voting = voting_regressor.predict(X_test)
mse_voting = mean_squared_error(y_test, y_pred_voting)
rmse_voting = np.sqrt(mse_voting)
r2_voting = r2_score(y_test, y_pred_voting)

print(f"Voting Regressor の結果:")
print(f"平均二乗誤差 (MSE): {mse_voting:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {rmse_voting:.2f}")
print(f"決定係数 (R²): {r2_voting:.2f}")

# 結果を保存
results['VotingRegressor'] = {
    'model': voting_regressor,
    'mse': mse_voting,
    'rmse': rmse_voting,
    'r2': r2_voting,
    'best_params': None,
    'is_pipeline': False,
    'preprocessor': None
}

# 最も精度の高いモデルを選択（R²が最も高いモデル）
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_result = results[best_model_name]

print("\n===== 最適なモデル =====")
print(f"モデル: {best_model_name}")
print(f"平均二乗誤差 (MSE): {best_result['mse']:.2f}")
print(f"平方根平均二乗誤差 (RMSE): {best_result['rmse']:.2f}")
print(f"決定係数 (R²): {best_result['r2']:.2f}")
if best_result['best_params']:
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
plt.savefig('advanced_model_comparison.png')
print("モデル比較グラフを 'advanced_model_comparison.png' として保存しました。")

# 最適なモデルを使用して予測値と実際の値の比較グラフを作成
best_model = best_result['model']

# 予測を実行
if best_result['is_pipeline']:
    y_pred = best_model.predict(X_test)
else:
    if best_model_name == 'VotingRegressor':
        y_pred = best_model.predict(X_test)
    else:
        X_test_transformed = best_result['preprocessor'].transform(X_test)
        y_pred = best_model.predict(X_test_transformed)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('実際の値')
plt.ylabel('予測値')
plt.title(f'予測値 vs 実際の値 ({best_model_name}モデル)')
plt.savefig('advanced_model_prediction.png')
print("最適なモデルの予測値と実際の値の比較グラフを 'advanced_model_prediction.png' として保存しました。")

# 最適なモデルを保存
model_filename = 'advanced_prediction_model.pkl'
joblib.dump(best_result, model_filename)
print(f"\n最適なモデルを '{model_filename}' として保存しました。")

# 予測用のスクリプトを作成
predict_script = f'''
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
    model_path = 'advanced_prediction_model.pkl'
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイル '{{model_path}}' が見つかりません。")
        return None
    
    model_result = joblib.load(model_path)
    model = model_result['model']
    is_pipeline = model_result['is_pipeline']
    preprocessor = model_result['preprocessor']
    
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
        if is_pipeline:
            predictions = model.predict(df[features])
        else:
            if model.__class__.__name__ == 'VotingRegressor':
                predictions = model.predict(df[features])
            else:
                X_transformed = preprocessor.transform(df[features])
                predictions = model.predict(X_transformed)
        
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
        print("使用方法: python advanced_predict.py <Excelファイルのパス>")
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
        output_file = "advanced_prediction_results.xlsx"
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

with open('advanced_predict.py', 'w', encoding='utf-8') as f:
    f.write(predict_script)

print("\n予測用スクリプト 'advanced_predict.py' を作成しました。")
print("使用方法: python advanced_predict.py <予測したいExcelファイルのパス>")

# 実行時間を表示
end_time = time.time()
execution_time = end_time - start_time
print(f"\n実行時間: {execution_time:.2f}秒")

# READMEファイルの更新内容を作成
readme_update = f'''
# 高度な予測モデル

さらなる精度向上のために、以下のアプローチを実装しました：

1. 特徴量エンジニアリング
   - 多項式特徴量の生成（2次の多項式）
   - 特徴量間の交互作用の考慮
   - 数値特徴量の標準化

2. 高度なモデルの使用
   - XGBoost: 勾配ブースティングの高度な実装
   - LightGBM: 高速で効率的な勾配ブースティングフレームワーク
   - 多項式特徴量を組み合わせたXGBoostとLightGBM

3. アンサンブル手法
   - 上位2つのモデルを組み合わせたVotingRegressor

## 使用方法

以下のコマンドを実行して高度な予測モデルを作成します：

```bash
python advanced_model.py
```

新しいデータに対する予測は以下のコマンドで実行できます：

```bash
python advanced_predict.py <予測したいExcelファイルのパス>
```

予測結果は `advanced_prediction_results.xlsx` として保存されます。
'''

print("\nREADMEファイルに追加する内容を作成しました。")
print("READMEファイルを更新してください。")
