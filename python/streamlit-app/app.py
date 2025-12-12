import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="CSV Data Visualizer",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 CSV Data Visualizer")
st.markdown("CSVファイルをアップロードして、データを可視化します。")

# サイドバー
st.sidebar.header("設定")

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=["csv"],
    help="CSVファイルを選択してください"
)

# メイン処理
if uploaded_file is not None:
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(uploaded_file)

        # データ概要を表示
        st.header("📋 データ概要")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("行数", len(df))
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            st.metric("欠損値", df.isnull().sum().sum())

        # データプレビュー
        st.subheader("データプレビュー")
        st.dataframe(df.head(10), use_container_width=True)

        # 基本統計量
        if st.checkbox("基本統計量を表示"):
            st.subheader("基本統計量")
            st.dataframe(df.describe(), use_container_width=True)

        # グラフ作成セクション
        st.header("📈 グラフ作成")

        # 数値列とカテゴリ列を取得
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_columns = df.columns.tolist()

        if len(numeric_columns) > 0:
            # グラフタイプ選択
            graph_type = st.sidebar.selectbox(
                "グラフタイプ",
                ["折れ線グラフ", "棒グラフ", "散布図", "ヒストグラム", "箱ひげ図", "円グラフ"]
            )

            if graph_type == "折れ線グラフ":
                st.subheader("折れ線グラフ")
                x_col = st.selectbox("X軸", all_columns, key="line_x")
                y_col = st.selectbox("Y軸", numeric_columns, key="line_y")

                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} の推移")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "棒グラフ":
                st.subheader("棒グラフ")
                x_col = st.selectbox("X軸", all_columns, key="bar_x")
                y_col = st.selectbox("Y軸", numeric_columns, key="bar_y")

                fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} ごとの {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "散布図":
                st.subheader("散布図")
                x_col = st.selectbox("X軸", numeric_columns, key="scatter_x")
                y_col = st.selectbox("Y軸", numeric_columns, key="scatter_y")
                color_col = st.selectbox("色分け（オプション）", ["なし"] + all_columns, key="scatter_color")

                if color_col == "なし":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "ヒストグラム":
                st.subheader("ヒストグラム")
                col = st.selectbox("列を選択", numeric_columns, key="hist_col")
                bins = st.slider("ビン数", 5, 100, 30)

                fig = px.histogram(df, x=col, nbins=bins, title=f"{col} の分布")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "箱ひげ図":
                st.subheader("箱ひげ図")
                y_col = st.selectbox("Y軸", numeric_columns, key="box_y")
                x_col = st.selectbox("X軸（カテゴリ、オプション）", ["なし"] + all_columns, key="box_x")

                if x_col == "なし":
                    fig = px.box(df, y=y_col, title=f"{y_col} の箱ひげ図")
                else:
                    fig = px.box(df, x=x_col, y=y_col, title=f"{x_col} ごとの {y_col}")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "円グラフ":
                st.subheader("円グラフ")
                names_col = st.selectbox("カテゴリ列", all_columns, key="pie_names")
                values_col = st.selectbox("値列", numeric_columns, key="pie_values")

                fig = px.pie(df, names=names_col, values=values_col, title=f"{names_col} の割合")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("数値列が見つかりません。グラフを作成するには数値データが必要です。")

        # データダウンロード
        st.header("💾 データのダウンロード")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="処理済みデータをダウンロード",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.info("CSVファイルの形式を確認してください。")

else:
    # ファイルがアップロードされていない場合
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください。")

    # サンプルCSVの説明
    st.header("📝 使い方")
    st.markdown("""
    1. 左のサイドバーから「CSVファイルをアップロード」をクリック
    2. CSVファイルを選択
    3. データが自動的に読み込まれ、グラフが表示されます
    4. グラフタイプや軸を選択して、様々な可視化を試せます

    ### サンプルデータ
    `data/sample.csv` にサンプルデータが用意されています。
    """)
