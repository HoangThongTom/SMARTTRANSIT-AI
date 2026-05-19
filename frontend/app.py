"""
SmartTransit AI — Streamlit Dashboard
Giao diện trực quan hóa giao thông TP.HCM theo thời gian thực.
"""

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API = f"{BACKEND_URL}/api/v1"

st.set_page_config(
    page_title="SmartTransit AI — TP.HCM",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; font-weight: 700; }
    .metric-card p  { margin: 0.2rem 0 0; color: #666; font-size: 0.85rem; }

    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #e94560);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }

    .prediction-box {
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        font-weight: 600;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }

    section[data-testid="stSidebar"] { background-color: #0f3460 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_realtime(district=None):
    params = {"district": district} if district else {}
    try:
        r = requests.get(f"{API}/realtime", params=params, timeout=5)
        return r.json() if r.ok else None
    except:
        return None


@st.cache_data(ttl=300)
def fetch_segments():
    try:
        r = requests.get(f"{API}/segments", timeout=5)
        return r.json() if r.ok else None
    except:
        return None


@st.cache_data(ttl=60)
def fetch_hotspots(limit=10):
    try:
        r = requests.get(f"{API}/hotspots", params={"limit": limit}, timeout=5)
        return r.json() if r.ok else None
    except:
        return None


@st.cache_data(ttl=300)
def fetch_hourly_stats(dow=None):
    params = {"day_of_week": dow} if dow is not None else {}
    try:
        r = requests.get(f"{API}/stats/hourly", params=params, timeout=5)
        return r.json() if r.ok else None
    except:
        return None


def call_predict(payload: dict):
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=5)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.ok
    except:
        return False


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚌 SmartTransit AI")
    st.markdown("---")

    status = check_backend()
    st.markdown(
        f"**Trạng thái API:** {'🟢 Online' if status else '🔴 Offline'}"
    )

    st.markdown("---")
    page = st.radio(
        "📋 Chọn trang",
        ["🗺️ Bản đồ thời gian thực", "📊 Phân tích thống kê", "🤖 Dự báo AI", "📍 Điểm kẹt xe"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🔍 Bộ lọc")
    district_options = ["Tất cả", "Q1", "Q3", "Q5", "BT", "TB", "GV", "PN", "TD"]
    selected_district = st.selectbox("Quận/Huyện", district_options)
    if selected_district == "Tất cả":
        selected_district = None

    st.markdown("---")
    if st.button("🔄 Làm mới dữ liệu"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    ---
    **Chú thích màu:**
    - 🟢 Thông thoáng (>45 km/h)
    - 🟡 Chậm (20–45 km/h)
    - 🟠 Kẹt nhẹ (8–20 km/h)
    - 🔴 Kẹt nặng (<8 km/h)
    """)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🚌 SmartTransit AI — Giao thông TP.HCM</h1>
    <p>Cập nhật lúc: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')} &nbsp;|&nbsp; Dự báo thông minh bằng Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: BẢN ĐỒ THỜI GIAN THỰC
# ─────────────────────────────────────────────────────────────
if "Bản đồ" in page:
    data = fetch_realtime(selected_district)

    if data:
        summary = data.get("summary", {})
        total = data.get("total_segments", 0)

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card" style="border-color:#27ae60">
                <h3 style="color:#27ae60">{summary.get('thong_thoang', 0)}</h3>
                <p>✅ Thông thoáng</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card" style="border-color:#f39c12">
                <h3 style="color:#f39c12">{summary.get('cham', 0)}</h3>
                <p>🟡 Chậm</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card" style="border-color:#e67e22">
                <h3 style="color:#e67e22">{summary.get('ket_nhe', 0)}</h3>
                <p>🟠 Kẹt nhẹ</p></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card" style="border-color:#c0392b">
                <h3 style="color:#c0392b">{summary.get('ket_nang', 0)}</h3>
                <p>🔴 Kẹt nặng</p></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Folium Map
        m = folium.Map(
            location=[10.8, 106.70],
            zoom_start=13,
            tiles="CartoDB dark_matter"
        )

        color_map = {0: "#27ae60", 1: "#f39c12", 2: "#e67e22", 3: "#c0392b"}
        label_map = {0: "Thông thoáng", 1: "Chậm", 2: "Kẹt nhẹ", 3: "Kẹt nặng"}

        for seg in data.get("segments", []):
            lat = seg.get("lat_start")
            lon = seg.get("lon_start")
            if not lat or not lon:
                continue

            lvl = seg.get("congestion_level", 0) or 0
            color = color_map.get(lvl, "#27ae60")
            speed = seg.get("avg_speed_kmh")
            speed_str = f"{speed:.1f} km/h" if speed else "N/A"

            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + lvl * 3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(f"""
                    <b>{seg.get('street_name', '')}</b><br>
                    Quận: {seg.get('district_code', '')}<br>
                    Tình trạng: <b style="color:{color}">{label_map.get(lvl, '')}</b><br>
                    Tốc độ TB: {speed_str}
                """, max_width=200),
                tooltip=f"{seg.get('street_name', '')} — {label_map.get(lvl, '')}",
            ).add_to(m)

        col_map, col_list = st.columns([3, 1])
        with col_map:
            st.markdown("### 🗺️ Bản đồ giao thông TP.HCM")
            st_folium(m, width=None, height=500)

        with col_list:
            st.markdown("### 📋 Danh sách đường")
            df_seg = pd.DataFrame(data.get("segments", []))
            if not df_seg.empty:
                df_display = df_seg[["street_name", "status_label"]].copy()
                df_display.columns = ["Đường", "Tình trạng"]
                st.dataframe(df_display, hide_index=True, height=480)
    else:
        st.error("⚠️ Không kết nối được Backend API. Kiểm tra lại `docker-compose up backend`")


# ─────────────────────────────────────────────────────────────
# PAGE: PHÂN TÍCH THỐNG KÊ
# ─────────────────────────────────────────────────────────────
elif "Thống kê" in page:
    st.markdown("## 📊 Phân tích Giao thông TP.HCM")

    dow_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
    selected_dow = st.selectbox("Chọn ngày trong tuần", range(7), format_func=lambda x: dow_names[x])

    data = fetch_hourly_stats(selected_dow)
    if data and data.get("stats"):
        df = pd.DataFrame(data["stats"])
        df_day = df[df["day_of_week"] == selected_dow].sort_values("hour")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df_day, x="hour", y="avg_congestion",
                title=f"Mức kẹt xe trung bình theo giờ — {dow_names[selected_dow]}",
                labels={"hour": "Giờ", "avg_congestion": "Mức kẹt (0-3)"},
                color="avg_congestion",
                color_continuous_scale=["#27ae60", "#f39c12", "#e67e22", "#c0392b"],
                range_color=[0, 3],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Be Vietnam Pro",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.line(
                df_day, x="hour", y="avg_speed",
                title=f"Tốc độ trung bình theo giờ — {dow_names[selected_dow]}",
                labels={"hour": "Giờ", "avg_speed": "Tốc độ TB (km/h)"},
                markers=True,
                color_discrete_sequence=["#0f3460"],
            )
            # Vùng giờ cao điểm
            fig2.add_vrect(x0=6.5, x1=8.5, fillcolor="#e74c3c", opacity=0.15,
                           annotation_text="Cao điểm sáng", annotation_position="top left")
            fig2.add_vrect(x0=16, x1=19, fillcolor="#e74c3c", opacity=0.15,
                           annotation_text="Cao điểm chiều", annotation_position="top left")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        # Heatmap tất cả các ngày
        st.markdown("### 🌡️ Heatmap Kẹt xe — Cả tuần")
        df_all = pd.DataFrame(fetch_hourly_stats().get("stats", [])) if fetch_hourly_stats() else pd.DataFrame()
        if not df_all.empty:
            pivot = df_all.pivot_table(values="avg_congestion", index="day_of_week", columns="hour")
            fig3 = px.imshow(
                pivot,
                labels={"x": "Giờ trong ngày", "y": "Ngày", "color": "Mức kẹt"},
                y=dow_names,
                color_continuous_scale=["#27ae60", "#f39c12", "#e67e22", "#c0392b"],
                aspect="auto",
                title="Heatmap: Mức độ kẹt xe theo giờ và ngày"
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Chưa có dữ liệu thống kê. Đảm bảo đã chạy mock_data.py")


# ─────────────────────────────────────────────────────────────
# PAGE: DỰ BÁO AI
# ─────────────────────────────────────────────────────────────
elif "Dự báo" in page:
    st.markdown("## 🤖 Dự báo Kẹt xe bằng AI")
    st.markdown("Nhập thông tin bên dưới, AI sẽ dự báo mức độ giao thông tức thì.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            seg_options = {
                "Q1_NTH_001 — Nguyễn Trãi Q1": "Q1_NTH_001",
                "Q1_LLO_001 — Lê Lợi Q1": "Q1_LLO_001",
                "Q3_CMT8_001 — CMT8 Q3": "Q3_CMT8_001",
                "BT_XVNT_001 — Xô Viết Nghệ Tĩnh": "BT_XVNT_001",
                "TB_CH_001 — Cộng Hòa Tân Bình": "TB_CH_001",
                "TD_PVD_001 — Phạm Văn Đồng Thủ Đức": "TD_PVD_001",
                "GV_NK_001 — Nguyễn Kiệm Gò Vấp": "GV_NK_001",
            }
            selected_seg_label = st.selectbox("🛣️ Đoạn đường", list(seg_options.keys()))
            segment_id = seg_options[selected_seg_label]

        with col2:
            hour = st.slider("🕐 Giờ cần dự báo", 0, 23, datetime.now().hour)
            dow_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
            day_of_week = st.selectbox("📅 Ngày trong tuần", range(7), format_func=lambda x: dow_names[x])

        with col3:
            weather = st.selectbox("🌧️ Thời tiết", ["clear", "rain", "heavy_rain", "fog"],
                                   format_func=lambda x: {"clear": "☀️ Nắng", "rain": "🌧 Mưa nhẹ",
                                                           "heavy_rain": "⛈️ Mưa lớn", "fog": "🌫️ Sương mù"}[x])
            month = st.number_input("📆 Tháng", 1, 12, datetime.now().month)
            is_holiday = st.checkbox("🎉 Ngày lễ / nghỉ")

        submitted = st.form_submit_button("🔮 Dự báo ngay", use_container_width=True)

    if submitted:
        payload = {
            "segment_id": segment_id,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": int(month),
            "weather_condition": weather,
            "is_holiday": is_holiday,
            "lanes": 4,
        }

        with st.spinner("🤖 AI đang phân tích..."):
            result = call_predict(payload)

        if "error" in result:
            st.error(f"❌ Lỗi: {result['error']}")
        else:
            level = result.get("congestion_level", 0)
            color = result.get("congestion_color", "#27ae60")
            label = result.get("congestion_label", "")
            confidence = result.get("confidence", 0)
            recommendation = result.get("recommendation", "")
            speed = result.get("estimated_speed_kmh", 0)

            st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, {color}dd, {color}99)">
                Kết quả dự báo: <b>{label}</b> &nbsp;|&nbsp; Tốc độ ước tính: ~{speed} km/h
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🎯 Độ chính xác (confidence)", f"{confidence*100:.1f}%")
                st.markdown(f"**💡 Khuyến nghị:** {recommendation}")

            with col_b:
                probs = result.get("probabilities", {})
                if probs:
                    fig = go.Figure(go.Bar(
                        x=list(probs.values()),
                        y=list(probs.keys()),
                        orientation="h",
                        marker_color=["#27ae60", "#f39c12", "#e67e22", "#c0392b"],
                    ))
                    fig.update_layout(
                        title="Xác suất từng mức kẹt",
                        xaxis_title="Xác suất",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=200,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE: ĐIỂM KẸT XE
# ─────────────────────────────────────────────────────────────
elif "Điểm kẹt" in page:
    st.markdown("## 📍 Top Điểm Kẹt Xe TP.HCM")
    top_n = st.slider("Số lượng điểm hiển thị", 5, 30, 10)
    data = fetch_hotspots(top_n)

    if data and data.get("hotspots"):
        df = pd.DataFrame(data["hotspots"])

        # Bản đồ hotspots
        m = folium.Map(location=[10.8, 106.70], zoom_start=13, tiles="CartoDB dark_matter")
        for i, row in df.iterrows():
            if pd.notna(row.get("lat_start")) and pd.notna(row.get("lon_start")):
                lvl = int(row.get("congestion_level", 0))
                colors = {0: "#27ae60", 1: "#f39c12", 2: "#e67e22", 3: "#c0392b"}
                folium.Marker(
                    location=[row["lat_start"], row["lon_start"]],
                    popup=f"#{i+1} {row['street_name']} ({row['district_code']}) — {row['status_label']}",
                    tooltip=f"#{i+1} {row['street_name']}",
                    icon=folium.Icon(color=["green","orange","orange","red"][lvl], icon="info-sign")
                ).add_to(m)
        st_folium(m, width=None, height=400)

        # Table
        st.markdown("### 📋 Bảng xếp hạng")
        df_display = df[["street_name", "district_code", "status_label", "avg_speed_kmh"]].copy()
        df_display.index = range(1, len(df_display) + 1)
        df_display.columns = ["Đường", "Quận", "Tình trạng", "Tốc độ TB (km/h)"]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("Không có dữ liệu hotspots. Kiểm tra backend API.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#999; font-size:0.8rem'>SmartTransit AI v1.0 — Hệ thống dự báo giao thông TP.HCM | "
    "Powered by Random Forest + FastAPI + Streamlit</center>",
    unsafe_allow_html=True
)
