"""
Streamlitアプリ: 開発計画 vs 着地見込み 差額分析

このアプリは、施策ごとの「開発計画金額」と「着地見込み額」の乖離を可視化し、
乖離が閾値以上の施策をアラートとして抽出します。
月次データ（開発計画_4月～3月、着地見込み計画_4月～3月）がある場合は、
月次推移（累計）を可視化できます。
また、ISS区分別／部門別に、開発計画金額・着地見込み額・実績を横並びにし、
組替有無の内訳を積み上げ棒グラフで表示します。

単位:
- 金額は「百万円」単位に扱います（例: 20 = 2,000万円）

実行方法:
    streamlit run app.py

Example:
    1) 画面左のサイドバーからCSVをアップロード
    2) フィルタ・集計軸を選択
    3) KPI / アラート / 月次推移 / 明細を確認
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


import plotly.express as px
# -----------------------------
# 設定
# -----------------------------
MONTHS: List[str] = ["4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月", "1月", "2月", "3月"]
DEV_MONTH_COLS: List[str] = [f"開発計画_{m}" for m in MONTHS]
LAND_MONTH_COLS: List[str] = [f"着地見込み計画_{m}" for m in MONTHS]

REQUIRED_COLS: List[str] = [
    "施策番号",
    "ISS区分",
    "施策名",
    "部門",
    "担当者",
    "開発計画金額",
    "着地見込み額",
    "実績",
    "組替有無",
]

GROUP_MAP = {
    "ISS区分": ["ISS区分"],
    "部門": ["部門"],
    "担当者": ["担当者"],
    "ISS区分×部門": ["ISS区分", "部門"],
}


@dataclass(frozen=True)
class AppConfig:
    """アプリの動作設定"""
    alert_threshold_million_yen: float = 20.0  # 乖離アラート閾値（百万円）


# -----------------------------
# ユーティリティ
# -----------------------------


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    """
    CSVを読み込みます。文字化け対策でutf-8-sigを優先し、失敗したらutf-8を試します。
    """
    if uploaded_file is None:
        raise ValueError("CSVファイルが指定されていません。")
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding="utf-8")


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """必須列の不足をチェックします。"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0), missing


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """乖離などの派生列を追加します。"""
    out = df.copy()
    out["乖離"] = out["開発計画金額"].astype(float) - out["着地見込み額"].astype(float)
    out["乖離絶対値"] = out["乖離"].abs()
    out["乖離率"] = np.where(
        out["開発計画金額"].astype(float) == 0,
        np.nan,
        out["乖離"] / out["開発計画金額"].astype(float),
    )
    return out


def apply_filters(
    df: pd.DataFrame,
    iss: List[str],
    dept: List[str],
    owner: List[str],
    kumikae: List[str],
) -> pd.DataFrame:
    """サイドバーで指定されたフィルタを適用します。"""
    out = df.copy()
    if iss:
        out = out[out["ISS区分"].isin(iss)]
    if dept:
        out = out[out["部門"].isin(dept)]
    if owner:
        out = out[out["担当者"].isin(owner)]
    if kumikae:
        out = out[out["組替有無"].isin(kumikae)]
    return out


def group_summary(df: pd.DataFrame, group_cols: List[str], threshold: float) -> pd.DataFrame:
    """集計軸に応じたサマリを作成します。"""
    g = df.groupby(group_cols, dropna=False, as_index=False).agg(
        施策数=("施策番号", "count"),
        開発計画金額=("開発計画金額", "sum"),
        着地見込み額=("着地見込み額", "sum"),
        実績=("実績", "sum"),
        乖離=("乖離", "sum"),
        乖離絶対値=("乖離絶対値", "sum"),
        アラート件数=("乖離絶対値", lambda s: int((s >= threshold).sum())),
    )
    # 見やすい順に
    g = g.sort_values(["乖離絶対値", "施策数"], ascending=[False, False]).reset_index(drop=True)
    return g


def measure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    施策名が重複する場合に数値を合算したサマリを返します。
    ISS区分/部門/担当者/組替有無などは代表値（先頭行）を採用します。
    """
    numeric_cols = [c for c in ["開発計画金額", "着地見込み額", "実績"] if c in df.columns]
    representative_cols = [c for c in ["施策番号", "ISS区分", "部門", "担当者", "組替有無"] if c in df.columns]

    agg_map: dict[str, str] = {c: "first" for c in representative_cols}
    agg_map.update({c: "sum" for c in numeric_cols})

    g = df.groupby("施策名", dropna=False).agg(agg_map).reset_index()

    if "開発計画金額" in g.columns and "着地見込み額" in g.columns:
        g["乖離"] = g["開発計画金額"].astype(float) - g["着地見込み額"].astype(float)
        g["乖離絶対値"] = g["乖離"].abs()

    ordered_cols = [
        "施策番号",
        "ISS区分",
        "施策名",
        "部門",
        "担当者",
        "開発計画金額",
        "着地見込み額",
        "乖離",
        "乖離絶対値",
        "実績",
        "組替有無",
    ]
    existing_cols = [c for c in ordered_cols if c in g.columns]
    g = g[existing_cols]

    g = g.sort_values("乖離絶対値", ascending=False, na_position="last").reset_index(drop=True) if "乖離絶対値" in g.columns else g
    return g


def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """選択スコープの月次累計を返します。"""
    dev = df[DEV_MONTH_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    land = df[LAND_MONTH_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=0)

    # 実績列が存在する場合は合計を計算
    total_actual = pd.to_numeric(df["実績"], errors="coerce").sum() if "実績" in df.columns else 0

    out = pd.DataFrame(
        {
            "月": pd.Categorical(MONTHS, categories=MONTHS, ordered=True),
            "開発計画": dev.values,
            "着地見込み計画": land.values,
        }
    )
    out["開発計画_累計"] = out["開発計画"].cumsum()
    out["着地見込み計画_累計"] = out["着地見込み計画"].cumsum()

    # 実績の累計を追加（全期間の実績を12等分して累計）
    monthly_actual = total_actual / 12
    out["実績_累計"] = [monthly_actual * (i + 1) for i in range(12)]

    return out


def style_alerts(df: pd.DataFrame, threshold: float):
    """アラート行を強調するためのスタイル定義。"""
    def _row_style(row):
        if float(row.get("乖離絶対値", 0)) >= threshold:
            return ["background-color: rgba(255, 0, 0, 0.08)"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)



def render_category_summary_and_charts(
    df: pd.DataFrame,
    category_col: str,
    money_cols: List[str] | None = None,
) -> None:
    """カテゴリ別サマリーとグラフを描画します。

    app-sub.py 側の「ISS区分別サマリー」「部門別サマリー」の表示ロジックを、
    app.py 側に移植するための共通関数です。

    表示仕様:
    - サマリ: category_col 単位で、開発計画金額 / 着地見込み額 / 実績 を合計
    - グラフ: 「組替有無」がある場合、カテゴリごとに
        x=金額種別 開発計画金額 着地見込み額 実績
        color=組替有無 の stacked bar を表示
      「組替有無」がない場合は、カテゴリ全体の grouped bar を表示

    Args:
        df: フィルタ後データ
        category_col: 集計軸の列名 例 ISS区分 部門
        money_cols: 金額列名のリスト。省略時は ['開発計画金額','着地見込み額','実績'] を使用

    Example:
        render_category_summary_and_charts(df_f, "ISS区分")
    """

    if money_cols is None:
        money_cols = ["開発計画金額", "着地見込み額", "実績"]

    if category_col not in df.columns:
        st.warning(f"'{category_col}' 列が存在しないため、このサマリは表示できません。")
        return

    # ---- サマリ作成 ----
    summary = (
        df.groupby(category_col, dropna=False)[money_cols]
        .sum()
        .reset_index()
        .sort_values("開発計画金額", ascending=False)
        .reset_index(drop=True)
    )

    # 見やすいフォーマット
    try:
        st.dataframe(
            summary.style.format({c: "{:,.1f}" for c in money_cols}),
            use_container_width=True,
            height=260,
        )
    except Exception as e:
        st.warning(f"テーブルのスタイル適用中にエラーが発生しました: {e}")
        # Styler 非対応環境へのフォールバック
        st.dataframe(summary, use_container_width=True, height=260)

    selected_categories = summary[category_col].tolist()

    # ---- 組替有無の内訳がある場合: カテゴリごとに stacked bar ----
    if "組替有無" in df.columns:
        breakdown = (
            df.groupby([category_col, "組替有無"], dropna=False)[money_cols]
            .sum()
            .reset_index()
        )

        st.markdown("**内訳 グラフ**")
        st.caption("各カテゴリごとに、開発計画金額・着地見込み額・実績を横並びにし、組替有無の内訳を積み上げで表示します。")

        for i in range(0, len(selected_categories), 2):
            cols = st.columns(2)
            for col_idx, cat in enumerate(selected_categories[i : i + 2]):
                with cols[col_idx]:
                    cat_data = breakdown[breakdown[category_col] == cat]

                    # 長形式へ変換
                    long_df = cat_data.melt(
                        id_vars=["組替有無"],
                        value_vars=money_cols,
                        var_name="金額種別",
                        value_name="金額",
                    )

                    fig = px.bar(
                        long_df,
                        x="金額種別",
                        y="金額",
                        color="組替有無",
                        barmode="stack",
                        text="金額",
                        category_orders={"金額種別": money_cols},
                        title=str(cat),
                    )
                    fig.update_traces(texttemplate="%{text:,.0f}", textposition="inside")

                    # 各金額種別の合計を注記
                    totals = long_df.groupby("金額種別")["金額"].sum()
                    for idx, m in enumerate(money_cols):
                        total = float(totals.get(m, 0))
                        fig.add_annotation(
                            x=idx,
                            y=total,
                            text=f"計: {total:,.0f}",
                            showarrow=False,
                            yshift=10,
                        )

                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="金額",
                        height=380,
                        showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---- 組替有無が無い場合: カテゴリ全体の grouped bar ----
    else:
        st.markdown("**金額比較 グラフ**")
        long = summary.melt(id_vars=[category_col], value_vars=money_cols, var_name="金額種別", value_name="金額")
        long = long[long[category_col].isin(selected_categories)]

        fig = px.bar(
            long,
            x=category_col,
            y="金額",
            color="金額種別",
            barmode="group",
            category_orders={category_col: selected_categories, "金額種別": money_cols},
        )
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title="金額",
            height=420,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    """アプリのエントリポイント"""
    st.set_page_config(page_title="差額分析アプリ", layout="wide")
    st.title("月次の差額分析アプリ")
    st.caption("開発計画金額と着地見込み額の乖離を、施策・区分・部門などで可視化します。単位は百万円です。")
    st.info("開発計画金額と着地見込み額の乖離が設定したアラート閾値以上の施策一覧がアラート施策一覧として表示されます。")

    config = AppConfig()

    with st.sidebar:
        st.header("データ読み込み")
        uploaded = st.file_uploader("CSVをアップロード", type=["csv"])

        st.divider()
        st.header("設定")
        threshold = st.number_input(
            "アラート閾値 乖離絶対値 百万円",
            min_value=0.0,
            value=float(config.alert_threshold_million_yen),
            step=1.0,
        )

    if uploaded is None:
        st.info("左のサイドバーからCSVをアップロードしてください")
        st.stop()

    df_raw = load_csv(uploaded)

    ok, missing = validate_schema(df_raw)
    if not ok:
        st.error(f"必須列が不足しています: {', '.join(missing)}")
        st.stop()

    df = add_derived_columns(df_raw)

    # -----------------------------
    # フィルタ UI
    # -----------------------------
    with st.sidebar:
        st.divider()
        st.header("フィルタ")
        iss_list = sorted([x for x in df["ISS区分"].dropna().unique().tolist()])
        dept_list = sorted([x for x in df["部門"].dropna().unique().tolist()])
        owner_list = sorted([x for x in df["担当者"].dropna().unique().tolist()])
        kumikae_list = sorted([x for x in df["組替有無"].dropna().unique().tolist()])

        iss_sel = st.multiselect("ISS区分", iss_list, default=[])
        dept_sel = st.multiselect("部門", dept_list, default=[])
        owner_sel = st.multiselect("担当者", owner_list, default=[])
        kumikae_sel = st.multiselect("組替有無", kumikae_list, default=[])

        st.divider()
        st.header("集計軸")
        group_mode = st.selectbox(
            "集計の切り口",
            ["施策単位", "ISS区分", "部門", "担当者", "ISS区分×部門"],
            index=0,
        )

    df_f = apply_filters(df, iss_sel, dept_sel, owner_sel, kumikae_sel)

    # -----------------------------
    # KPI
    # -----------------------------
    total_plan = float(df_f["開発計画金額"].sum())
    total_land = float(df_f["着地見込み額"].sum())
    total_actual = float(df_f["実績"].sum())
    total_gap_land = total_plan - total_land
    total_gap_actual = total_plan - total_actual
    n_measures = int(df_f.shape[0])
    n_alerts = int((df_f["乖離絶対値"] >= threshold).sum())

    # 進捗率計算
    progress_land = (total_land / total_plan * 100) if total_plan != 0 else 0
    progress_actual = (total_actual / total_plan * 100) if total_plan != 0 else 0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("施策数", f"{n_measures:,}")
    c2.metric("開発計画 合計", f"{total_plan:,.0f}")
    c3.metric("着地見込み 合計", f"{total_land:,.0f}", delta=f"進捗率 {progress_land:.1f}%")
    c4.metric("実績 合計", f"{total_actual:,.0f}", delta=f"進捗率 {progress_actual:.1f}%")
    c5.metric("乖離 合計（着地見込み）", f"{total_gap_land:,.0f}")
    c6.metric("乖離 合計（実績）", f"{total_gap_actual:,.0f}")
    c7.metric(f"アラート件数 乖離≥{threshold:.0f}", f"{n_alerts:,}")

    # -----------------------------
    # アラート一覧
    # -----------------------------
    st.subheader("アラート施策一覧")
    alerts_df = df_f[df_f["乖離絶対値"] >= threshold].copy()
    if alerts_df.empty:
        st.success("現在のスコープではアラート対象の施策はありません。")
    else:
        alerts_view = alerts_df[
            ["施策番号", "ISS区分", "施策名", "部門", "担当者", "開発計画金額", "着地見込み額", "乖離", "乖離絶対値", "実績", "組替有無"]
        ].sort_values("乖離絶対値", ascending=False).reset_index(drop=True)
        st.dataframe(style_alerts(alerts_view, threshold), use_container_width=True, height=320)

    # -----------------------------
    # 集計表
    # -----------------------------
    st.subheader("概況サマリ")
    if group_mode == "施策単位":
        # 施策名が重複する場合、数値を合算して表示
        summary_df = measure_summary(df_f)
    else:
        group_cols = GROUP_MAP.get(group_mode)
        if group_cols:
            summary_df = group_summary(df_f, group_cols, threshold)
        else:
            summary_df = pd.DataFrame()

    st.dataframe(summary_df, use_container_width=True, height=320)


    # -----------------------------
    # ISS区分サマリー＆グラフ（app-sub.py から移植）
    # -----------------------------
    st.subheader("ISS区分サマリーとグラフ")
    render_category_summary_and_charts(df_f, "ISS区分")

    # -----------------------------
    # 部門別サマリー＆グラフ（app-sub.py から移植）
    # -----------------------------
    st.subheader("部門別サマリーとグラフ")
    render_category_summary_and_charts(df_f, "部門")

    # -----------------------------
    # 月次推移
    # -----------------------------
    st.subheader("月次推移")

    # 月次列が存在するかチェック
    has_monthly = all(c in df_f.columns for c in DEV_MONTH_COLS) and all(c in df_f.columns for c in LAND_MONTH_COLS)

    if has_monthly:
        ms = monthly_series(df_f)

        st.markdown("**月別 累計**")
        st.line_chart(ms.set_index("月")[["開発計画_累計", "着地見込み計画_累計", "実績_累計"]])

        st.markdown("**月次テーブル**")
        st.dataframe(ms, use_container_width=True, height=260)
    else:
        st.warning("月次データ（開発計画_4月～3月、着地見込み計画_4月～3月）がCSVに含まれていません。")

    # -----------------------------
    # 明細表示とダウンロード
    # -----------------------------
    st.subheader("明細データ")
    st.dataframe(df_f, use_container_width=True, height=360)

    # ダウンロード
    csv_bytes = df_f.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "フィルタ後データをCSVでダウンロード",
        data=csv_bytes,
        file_name="filtered_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
