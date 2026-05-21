import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Divvy Bike Share Analysis",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ヘルパー関数
# ============================================================
def insight_box(title, body):
    with st.container(border=True):
        st.markdown(f"#### 💡 {title}")
        st.markdown(body)

def cluster_badge(cid):
    colors = {0: "🟠", 1: "🔴", 2: "🔵"}
    names  = {0: "通勤＋レジャー混在型", 1: "通勤特化型", 2: "標準型"}
    return f"{colors[cid]} {names[cid]}"

# ============================================================
# データ読み込み
# ============================================================
@st.cache_data
def load_data():
    daily    = pd.read_csv("data/processed/daily.csv",
                           parse_dates=["date"])
    station  = pd.read_csv("data/processed/station_map_data.csv")
    profiles = pd.read_csv("data/processed/cluster_profiles.csv")
    return daily, station, profiles

daily, station, profiles = load_data()

CLUSTER_NAMES  = {0: "通勤＋レジャー混在型", 1: "通勤特化型", 2: "標準型"}
CLUSTER_COLORS = {0: "#f97316", 1: "#ef4444", 2: "#3b82f6"}
CLUSTER_EN     = {0: "Mixed", 1: "Commute-focused", 2: "Standard"}

# ============================================================
# サイドバー
# ============================================================
st.sidebar.title("🚲 Divvy Analysis")
st.sidebar.caption("Chicago Bike Share · 2025.05 – 2026.04")

page = st.sidebar.radio(
    "ページを選択",
    ["📊 Overview", "🔍 Demand Factors", "🗺️ Station Strategy"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("""
**データソース**
- [Divvy Trip Data](https://divvy-tripdata.s3.amazonaws.com/index.html)
- [Open-Meteo API](https://open-meteo.com)

**分析期間**  
2025年5月1日 〜 2026年4月30日  
総乗車件数：約500万件  
分析ステーション数：692駅
""")

# ============================================================
# PAGE 1: Overview
# ============================================================
if page == "📊 Overview":
    st.title("📊 Overview")
    st.caption("シェアサイクルの利用動向を俯瞰する")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総乗車件数（概算）",
                f"{daily['ride_count'].sum()/1e6:.1f}M")
    col2.metric("1日平均利用件数",
                f"{daily['ride_count'].mean():,.0f}")
    col3.metric("最多利用日",
                daily.loc[daily['ride_count'].idxmax(),
                          'date'].strftime('%Y/%m/%d'))
    col4.metric("一時利用比率（平均）",
                f"{daily['casual_ratio'].mean():.1%}")

    st.divider()

    # 月別利用件数
    st.subheader("月別平均利用件数")
    monthly = daily.groupby("month")["ride_count"].mean().reset_index()
    month_map = {5:"5月",6:"6月",7:"7月",8:"8月",9:"9月",10:"10月",
                 11:"11月",12:"12月",1:"1月",2:"2月",3:"3月",4:"4月"}
    month_order = [month_map[m]
                   for m in [5,6,7,8,9,10,11,12,1,2,3,4]]
    monthly["month_label"] = monthly["month"].map(month_map)
    monthly["month_label"] = pd.Categorical(
        monthly["month_label"], categories=month_order, ordered=True
    )
    monthly = monthly.sort_values("month_label")

    fig_monthly = px.bar(
        monthly, x="month_label", y="ride_count",
        labels={"month_label":"月","ride_count":"平均利用件数"},
        color_discrete_sequence=["#667eea"]
    )
    fig_monthly.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#eeeeee"),
        showlegend=False, height=350
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ヒートマップ
    st.subheader("曜日×時間帯ヒートマップ")
    st.caption("平日通勤型（二山）と週末レジャー型（一山）の差を確認できます")
    heatmap_data = profiles.pivot_table(
        index="is_weekend", columns="hour",
        values="ratio", aggfunc="mean"
    )
    heatmap_data.index = ["平日", "週末"]
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h}時" for h in range(24)],
        y=["平日", "週末"],
        colorscale="YlOrRd"
    ))
    fig_heat.update_layout(
        height=220, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 散布図
    st.subheader("気温と利用件数の関係")
    fig_scatter = px.scatter(
        daily, x="temp", y="ride_count",
        color="is_weekend",
        color_discrete_map={0:"#667eea", 1:"#f97316"},
        labels={"temp":"日平均気温 (℃)",
                "ride_count":"利用件数",
                "is_weekend":""},
    )
    fig_scatter.for_each_trace(lambda t: t.update(
        name="平日" if t.name == "0" else "週末"
    ))
    fig_scatter.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#eeeeee"),
        xaxis=dict(gridcolor="#eeeeee"), height=380
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    insight_box("Business Insight", """
利用件数は **月・曜日より気温との相関（r=0.90）が圧倒的に高い**。
需要予測の第一変数として気象予報を活用することで、翌日の配置計画精度を大幅に向上できる可能性がある。

また、平日は8時・17〜18時の通勤ピークが明確な一方、週末は昼間のなだらかな利用に変わる。
**1日の合計件数は平日・週末でほぼ同じでも、再配置の最繁忙時間帯は曜日によって異なる**点に注意が必要。
    """)


# ============================================================
# PAGE 2: Demand Factors
# ============================================================
elif page == "🔍 Demand Factors":
    st.title("🔍 Demand Factors")
    st.caption("利用量を動かす要因を定量化する")

    @st.cache_data
    def run_regression(daily):
        features = ["temp","precip","wind",
                    "is_weekend","is_holiday","month"]
        X = daily[features].copy()
        y = daily["ride_count"]
        scaler = StandardScaler()
        num_cols = ["temp","precip","wind","month"]
        X[num_cols] = scaler.fit_transform(X[num_cols])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression().fit(X_tr, y_tr)
        r2_tr = r2_score(y_tr, model.predict(X_tr))
        r2_te = r2_score(y_te, model.predict(X_te))
        coef_df = pd.DataFrame({
            "変数": features,
            "係数": model.coef_,
            "abs": np.abs(model.coef_)
        }).sort_values("abs", ascending=True)
        y_pred = model.predict(X_te)
        return r2_tr, r2_te, coef_df, y_te, y_pred

    r2_train, r2_test, coef_df, y_te, y_pred = run_regression(daily)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test R²",  f"{r2_test:.4f}")
    col2.metric("Train R²", f"{r2_train:.4f}")
    col3.metric("過学習の差", f"{r2_train - r2_test:.4f}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("各変数の影響度（回帰係数）")
        var_labels = {
            "temp":"気温","precip":"降水量","wind":"風速",
            "is_weekend":"週末フラグ",
            "is_holiday":"祝日フラグ","month":"月"
        }
        coef_df["変数名"] = coef_df["変数"].map(var_labels)
        coef_df["color"] = coef_df["係数"].apply(
            lambda x: "#667eea" if x > 0 else "#ef4444"
        )
        fig_coef = go.Figure(go.Bar(
            x=coef_df["係数"], y=coef_df["変数名"],
            orientation="h", marker_color=coef_df["color"]
        ))
        fig_coef.add_vline(x=0, line_color="black", line_width=1)
        fig_coef.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#eeeeee", title="係数（標準化済み）"),
            height=320, margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_right:
        st.subheader("予測値 vs 実測値")
        lim = [0, daily["ride_count"].max() * 1.05]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_te, y=y_pred, mode="markers",
            marker=dict(color="#667eea", opacity=0.5, size=6),
            name="予測値"
        ))
        fig_pred.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="#ef4444", width=1.5, dash="dash"),
            name="完全予測ライン"
        ))
        fig_pred.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#eeeeee", title="実測値"),
            yaxis=dict(gridcolor="#eeeeee", title="予測値"),
            height=320, margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # 需要シミュレーター
    st.divider()
    st.subheader("🎛️ 需要シミュレーター")
    st.caption("気象条件を変えて推定利用件数を確認できます")

    col1, col2, col3, col4 = st.columns(4)
    sim_temp    = col1.slider("気温 (℃)", -20, 35, 20)
    sim_precip  = col2.slider("降水量 (mm)", 0, 50, 0)
    sim_wind    = col3.slider("風速 (km/h)", 0, 60, 15)
    sim_weekend = col4.selectbox("曜日", ["平日", "週末"])

    @st.cache_data
    def get_model(daily):
        features = ["temp","precip","wind",
                    "is_weekend","is_holiday","month"]
        X = daily[features].copy()
        y = daily["ride_count"]
        scaler = StandardScaler()
        num_cols = ["temp","precip","wind","month"]
        X[num_cols] = scaler.fit_transform(X[num_cols])
        model = LinearRegression().fit(X, y)
        return scaler, model, num_cols

    scaler2, model2, num_cols2 = get_model(daily)

    sim_input = pd.DataFrame([{
        "temp": sim_temp, "precip": sim_precip,
        "wind": sim_wind,
        "is_weekend": 1 if sim_weekend == "週末" else 0,
        "is_holiday": 0,
        "month": daily["month"].mean()
    }])
    sim_input[num_cols2] = scaler2.transform(sim_input[num_cols2])
    sim_pred = model2.predict(sim_input)[0]
    avg = daily["ride_count"].mean()

    col_a, col_b = st.columns([1, 2])
    col_a.metric(
        "推定利用件数",
        f"{max(0, sim_pred):,.0f} 件",
        delta=f"{(sim_pred-avg)/avg:+.1%} vs 平均"
    )

    insight_box("Business Insight", f"""
気温の係数（+7,060）は他変数の4〜10倍と突出しており、
**需要予測の第一変数は気温**であることが定量的に確認された。
Test R²=0.874と説明力が高く過学習もないため、
このモデルは翌日の気象予報を入力とした **需要の事前推計ツール** として実運用への応用が期待できる。

週末フラグの係数（−51）はほぼゼロだが、
これは「合計件数が同じ」という意味であり、時間帯別の分布は大きく異なる（Overviewページ参照）。
    """)


# ============================================================
# PAGE 3: Station Strategy
# ============================================================
elif page == "🗺️ Station Strategy":
    st.title("🗺️ Station Strategy")
    st.caption("利用パターンによるステーション分類と空間分布")

    cluster_summary = station.groupby("cluster").agg(
        駅数=("start_station_name","count"),
        平均利用件数=("total_count","mean")
    ).reset_index()

    col1, col2, col3 = st.columns(3)
    for col, cid in zip([col1, col2, col3], [1, 0, 2]):
        row = cluster_summary[
            cluster_summary["cluster"] == cid
        ].iloc[0]
        col.metric(
            cluster_badge(cid),
            f"{int(row['駅数'])}駅",
            delta=f"平均{row['平均利用件数']:,.0f}件/年"
        )

    st.divider()

    # 時間帯プロファイル
    st.subheader("時間帯プロファイル")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        selected_clusters = st.multiselect(
            "表示するクラスタ",
            options=list(CLUSTER_NAMES.values()),
            default=list(CLUSTER_NAMES.values())
        )
    with col_f2:
        show_weekend = st.radio(
            "平日 / 週末", ["平日","週末","両方"], horizontal=True
        )

    fig_profile = go.Figure()
    weekend_map = {"平日":[False],"週末":[True],"両方":[False,True]}
    line_styles  = {False:"solid", True:"dash"}
    line_labels  = {False:"平日", True:"週末"}

    for cid, cname in CLUSTER_NAMES.items():
        if cname not in selected_clusters:
            continue
        for is_we in weekend_map[show_weekend]:
            sub = profiles[
                (profiles["cluster"] == cid) &
                (profiles["is_weekend"] == is_we)
            ].sort_values("hour")
            fig_profile.add_trace(go.Scatter(
                x=sub["hour"], y=sub["ratio"],
                mode="lines",
                name=f"{cname}（{line_labels[is_we]}）",
                line=dict(
                    color=CLUSTER_COLORS[cid],
                    dash=line_styles[is_we],
                    width=2.5
                )
            ))

    fig_profile.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            gridcolor="#eeeeee", title="時間帯",
            tickvals=list(range(0,24,2)),
            ticktext=[f"{h}時" for h in range(0,24,2)]
        ),
        yaxis=dict(gridcolor="#eeeeee", title="利用比率"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    st.divider()

    # 地図
    st.subheader("ステーション空間分布")
    map_filter = st.multiselect(
        "表示するクラスタ",
        options=list(CLUSTER_NAMES.values()),
        default=list(CLUSTER_NAMES.values()),
        key="map_filter"
    )
    station["cluster_name"] = station["cluster"].map(CLUSTER_NAMES)
    station_filtered = station[
        station["cluster_name"].isin(map_filter)
    ]

    fig_map = px.scatter_mapbox(
        station_filtered,
        lat="lat", lon="lng",
        color="cluster_name",
        color_discrete_map={
            v: CLUSTER_COLORS[k] for k, v in CLUSTER_NAMES.items()
        },
        size="total_count",
        size_max=18,
        hover_name="start_station_name",
        hover_data={"total_count":True,
                    "cluster_name":True,
                    "lat":False,"lng":False},
        labels={"cluster_name":"クラスタ",
                "total_count":"総利用件数"},
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat":41.88,"lon":-87.63},
        height=520
    )
    fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("円の大きさは総利用件数を表します。ホバーで詳細を確認できます。")

    # 上位ステーションテーブル
    st.subheader("クラスタ別・利用件数上位ステーション")
    tab1, tab2, tab3 = st.tabs(
        [CLUSTER_NAMES[1], CLUSTER_NAMES[0], CLUSTER_NAMES[2]]
    )
    for tab, cid in zip([tab1,tab2,tab3],[1,0,2]):
        with tab:
            top = station[station["cluster"]==cid]\
                .nlargest(10,"total_count")\
                [["start_station_name","total_count"]]\
                .rename(columns={
                    "start_station_name":"ステーション名",
                    "total_count":"総利用件数"
                })
            top["総利用件数"] = top["総利用件数"].apply(
                lambda x: f"{int(x):,}"
            )
            st.dataframe(top, use_container_width=True,
                         hide_index=True)

    insight_box("Business Insight", """
K-meansクラスタリングは **時間帯データのみ** を入力としたにもかかわらず、
地図上でループ地区（CBD）への集中という地理的に意味のある空間パターンが現れた。

🔴 **通勤特化型** はCBDに集中。夕方に自転車が流入し翌朝には不足するパターンが予想され、
夜間〜早朝の再配置が優先課題となる。

🔵 **標準型** の高利用ステーション（Navy Pier等）は観光需要が主で週末に需要が高まるため、
週末昼間帯の重点補充が有効と考えられる。
    """)
