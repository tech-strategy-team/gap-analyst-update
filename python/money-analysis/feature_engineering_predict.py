
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
        print("\n予測結果 (最初の5行):")
        if '着地見込み額合計' in result_df.columns:
            print(result_df[['ISS区分', '部門', '担当', '着地見込み額合計', '予測_着地見込み額合計', '差分', '誤差率(%)']].head())
        else:
            print(result_df[['ISS区分', '部門', '担当', '予測_着地見込み額合計']].head())
        
        # 結果をExcelファイルに保存
        output_file = "feature_engineering_prediction_results.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\n予測結果を {output_file} に保存しました。")
        
        # 予測の統計情報を表示（実際の値がある場合）
        if '着地見込み額合計' in result_df.columns:
            mae = np.abs(result_df['差分']).mean()
            rmse = np.sqrt((result_df['差分'] ** 2).mean())
            r2 = 1 - ((result_df['差分'] ** 2).sum() / ((result_df['着地見込み額合計'] - result_df['着地見込み額合計'].mean()) ** 2).sum())
            
            print("\n予測の統計情報:")
            print(f"平均絶対誤差 (MAE): {mae:.2f}")
            print(f"平方根平均二乗誤差 (RMSE): {rmse:.2f}")
            print(f"決定係数 (R²): {r2:.2f}")
