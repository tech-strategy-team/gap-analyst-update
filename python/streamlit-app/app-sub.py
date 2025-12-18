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

        # 金額サマリーセクション
        st.header("💰 金額サマリー")

        # 必要な列が存在するか確認
        required_columns = ['開発計画金額', '着地見込み額', '実績']
        if all(col in df.columns for col in required_columns):
            # 全体の合計
            st.subheader("📊 全体サマリー")
            total_plan = df['開発計画金額'].sum()
            total_forecast = df['着地見込み額'].sum()
            total_actual = df['実績'].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("開発計画金額 合計", f"{total_plan:,.1f}")
            with col2:
                st.metric("着地見込み額 合計", f"{total_forecast:,.1f}")
            with col3:
                st.metric("実績 合計", f"{total_actual:,.1f}")

            # 区切り線
            st.markdown("---")

            # ISS区分ごとの集計
            if 'ISS区分' in df.columns:
                st.subheader("📈 ISS区分別サマリー")
                iss_summary = df.groupby('ISS区分')[required_columns].sum().reset_index()
                iss_summary = iss_summary.sort_values('開発計画金額', ascending=False)

                # ISS区分別の表示
                st.dataframe(
                    iss_summary.style.format({
                        '開発計画金額': '{:,.1f}',
                        '着地見込み額': '{:,.1f}',
                        '実績': '{:,.1f}'
                    }),
                    use_container_width=True
                )

                # 組替有無による内訳がある場合
                if '組替有無' in df.columns:
                    # 各ISS区分ごとにグラフを作成
                    iss_breakdown = df.groupby(['ISS区分', '組替有無'])[required_columns].sum().reset_index()

                    # ISS区分のリストを取得
                    iss_categories = sorted(iss_breakdown['ISS区分'].unique())

                    # 2列レイアウトでグラフを表示
                    for i in range(0, len(iss_categories), 2):
                        cols = st.columns(2)

                        for col_idx, iss_cat in enumerate(iss_categories[i:i+2]):
                            with cols[col_idx]:
                                # 該当するISS区分のデータを抽出
                                iss_cat_data = iss_breakdown[iss_breakdown['ISS区分'] == iss_cat]

                                # 長形式に変換
                                iss_data = []
                                for _, row in iss_cat_data.iterrows():
                                    iss_data.append({
                                        '金額種別': '開発計画金額',
                                        '組替有無': row['組替有無'],
                                        '金額': row['開発計画金額']
                                    })
                                    iss_data.append({
                                        '金額種別': '着地見込み額',
                                        '組替有無': row['組替有無'],
                                        '金額': row['着地見込み額']
                                    })
                                    iss_data.append({
                                        '金額種別': '実績',
                                        '組替有無': row['組替有無'],
                                        '金額': row['実績']
                                    })

                                iss_df_cat = pd.DataFrame(iss_data)

                                # 棒グラフを作成
                                fig_iss_cat = px.bar(
                                    iss_df_cat,
                                    x='金額種別',
                                    y='金額',
                                    color='組替有無',
                                    title=f'{iss_cat}',
                                    barmode='stack',
                                    color_discrete_map={'組替済': '#5470c6', '組替無': '#91cc75'},
                                    category_orders={'金額種別': ['開発計画金額', '着地見込み額', '実績']},
                                    text='金額'
                                )

                                # 数値表示とレイアウト調整
                                fig_iss_cat.update_traces(texttemplate='%{text:,.0f}', textposition='inside')

                                # 各カテゴリの合計を計算して注釈として追加
                                totals = iss_df_cat.groupby('金額種別')['金額'].sum()
                                for idx, category in enumerate(['開発計画金額', '着地見込み額', '実績']):
                                    total = totals.get(category, 0)
                                    fig_iss_cat.add_annotation(
                                        x=idx,
                                        y=total,
                                        text=f'計: {total:,.0f}',
                                        showarrow=False,
                                        yshift=10,
                                        font=dict(size=10, color='black', family='Arial Black')
                                    )

                                fig_iss_cat.update_layout(
                                    xaxis_title='',
                                    yaxis_title='金額',
                                    height=400,
                                    showlegend=True
                                )

                                st.plotly_chart(fig_iss_cat, use_container_width=True)

                    # 区切り線
                    st.markdown("---")
                else:
                    # 組替有無がない場合は従来のグラフ
                    fig_iss = go.Figure()
                    fig_iss.add_trace(go.Bar(
                        name='開発計画金額',
                        x=iss_summary['ISS区分'],
                        y=iss_summary['開発計画金額'],
                        marker_color='lightblue'
                    ))
                    fig_iss.add_trace(go.Bar(
                        name='着地見込み額',
                        x=iss_summary['ISS区分'],
                        y=iss_summary['着地見込み額'],
                        marker_color='orange'
                    ))
                    fig_iss.add_trace(go.Bar(
                        name='実績',
                        x=iss_summary['ISS区分'],
                        y=iss_summary['実績'],
                        marker_color='green'
                    ))
                    fig_iss.update_layout(
                        title='ISS区分別 金額比較',
                        barmode='group',
                        xaxis_title='ISS区分',
                        yaxis_title='金額'
                    )
                    st.plotly_chart(fig_iss, use_container_width=True)

            # 部門ごとの集計
            if '部門' in df.columns:
                st.subheader("🏢 部門別サマリー")
                dept_summary = df.groupby('部門')[required_columns].sum().reset_index()
                dept_summary = dept_summary.sort_values('開発計画金額', ascending=False)

                # 部門別の表示
                st.dataframe(
                    dept_summary.style.format({
                        '開発計画金額': '{:,.1f}',
                        '着地見込み額': '{:,.1f}',
                        '実績': '{:,.1f}'
                    }),
                    use_container_width=True
                )

                # 組替有無による内訳がある場合
                if '組替有無' in df.columns:
                    # 各部門ごとにグラフを作成
                    dept_breakdown = df.groupby(['部門', '組替有無'])[required_columns].sum().reset_index()

                    # 部門のリストを取得（開発計画金額の降順）
                    dept_total = df.groupby('部門')['開発計画金額'].sum().sort_values(ascending=False)
                    dept_categories = dept_total.index.tolist()

                    # 2列レイアウトでグラフを表示
                    for i in range(0, len(dept_categories), 2):
                        cols = st.columns(2)

                        for col_idx, dept in enumerate(dept_categories[i:i+2]):
                            with cols[col_idx]:
                                # 該当する部門のデータを抽出
                                dept_cat_data = dept_breakdown[dept_breakdown['部門'] == dept]

                                # 長形式に変換
                                dept_data = []
                                for _, row in dept_cat_data.iterrows():
                                    dept_data.append({
                                        '金額種別': '開発計画金額',
                                        '組替有無': row['組替有無'],
                                        '金額': row['開発計画金額']
                                    })
                                    dept_data.append({
                                        '金額種別': '着地見込み額',
                                        '組替有無': row['組替有無'],
                                        '金額': row['着地見込み額']
                                    })
                                    dept_data.append({
                                        '金額種別': '実績',
                                        '組替有無': row['組替有無'],
                                        '金額': row['実績']
                                    })

                                dept_df_cat = pd.DataFrame(dept_data)

                                # 棒グラフを作成
                                fig_dept_cat = px.bar(
                                    dept_df_cat,
                                    x='金額種別',
                                    y='金額',
                                    color='組替有無',
                                    title=f'{dept}',
                                    barmode='stack',
                                    color_discrete_map={'組替済': '#5470c6', '組替無': '#91cc75'},
                                    category_orders={'金額種別': ['開発計画金額', '着地見込み額', '実績']},
                                    text='金額'
                                )

                                # 数値表示とレイアウト調整
                                fig_dept_cat.update_traces(texttemplate='%{text:,.0f}', textposition='inside')

                                # 各カテゴリの合計を計算して注釈として追加
                                totals = dept_df_cat.groupby('金額種別')['金額'].sum()
                                for idx, category in enumerate(['開発計画金額', '着地見込み額', '実績']):
                                    total = totals.get(category, 0)
                                    fig_dept_cat.add_annotation(
                                        x=idx,
                                        y=total,
                                        text=f'計: {total:,.0f}',
                                        showarrow=False,
                                        yshift=10,
                                        font=dict(size=10, color='black', family='Arial Black')
                                    )

                                fig_dept_cat.update_layout(
                                    xaxis_title='',
                                    yaxis_title='金額',
                                    height=400,
                                    showlegend=True
                                )

                                st.plotly_chart(fig_dept_cat, use_container_width=True)

                    # 区切り線
                    st.markdown("---")
                else:
                    # 組替有無がない場合は従来のグラフ
                    fig_dept = go.Figure()
                    fig_dept.add_trace(go.Bar(
                        name='開発計画金額',
                        x=dept_summary['部門'],
                        y=dept_summary['開発計画金額'],
                        marker_color='lightblue'
                    ))
                    fig_dept.add_trace(go.Bar(
                        name='着地見込み額',
                        x=dept_summary['部門'],
                        y=dept_summary['着地見込み額'],
                        marker_color='orange'
                    ))
                    fig_dept.add_trace(go.Bar(
                        name='実績',
                        x=dept_summary['部門'],
                        y=dept_summary['実績'],
                        marker_color='green'
                    ))
                    fig_dept.update_layout(
                        title='部門別 金額比較',
                        barmode='group',
                        xaxis_title='部門',
                        yaxis_title='金額'
                    )
                    st.plotly_chart(fig_dept, use_container_width=True)

            # 差額分析
            st.subheader("⚠️ 差額分析")
            df['差額'] = df['開発計画金額'] - df['着地見込み額']
            df_with_diff = df[df['差額'] != 0].copy()

            if len(df_with_diff) > 0:
                st.write(f"差額があるレコード数: **{len(df_with_diff)}** 件")

                # 差額の大きい順にソート
                df_with_diff = df_with_diff.sort_values('差額', ascending=False, key=abs)

                # 表示する列を選択
                display_columns = ['施策番号', 'ISS区分', '施策名', '部門', '担当者',
                                   '開発計画金額', '着地見込み額', '差額']
                available_display_columns = [col for col in display_columns if col in df_with_diff.columns]

                # 差額データを表示
                st.dataframe(
                    df_with_diff[available_display_columns].style.format({
                        '開発計画金額': '{:,.1f}',
                        '着地見込み額': '{:,.1f}',
                        '差額': '{:,.1f}'
                    }),
                    use_container_width=True
                )

                # 差額の統計
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("差額の平均", f"{df_with_diff['差額'].mean():,.1f}")
                with col2:
                    st.metric("差額の最大値", f"{df_with_diff['差額'].max():,.1f}")
                with col3:
                    st.metric("差額の最小値", f"{df_with_diff['差額'].min():,.1f}")
            else:
                st.success("全てのレコードで開発計画金額と着地見込み額が一致しています。")
        else:
            st.warning("必要な列（開発計画金額、着地見込み額、実績）が見つかりません。")

        # データダウンロード
        st.header("💾 データのダウンロード")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="処理済みデータをダウンロード",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

        # データ概要を表示（最後に移動）
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
    3. データが自動的に読み込まれ、金額サマリーが表示されます
    4. 全体・ISS区分別・部門別の集計結果を確認できます
    5. 差額分析で開発計画金額と着地見込み額の差を確認できます

    ### サンプルデータ
    `data/sample.csv` にサンプルデータが用意されています。
    """)
