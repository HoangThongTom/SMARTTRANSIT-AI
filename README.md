# 🚌 SmartTransit AI — Hệ thống Dự báo Kẹt xe TP.HCM

Hệ thống AI dự báo tình trạng giao thông TP.HCM theo thời gian thực, sử dụng Machine Learning để phân tích dữ liệu lịch sử và đưa ra cảnh báo kẹt xe.

---

## 🏗️ Kiến trúc hệ thống

```
smarttransit-ai/
├── database/          # Giai đoạn 1: Schema + Mock Data TP.HCM
├── ai_core/           # Giai đoạn 2: AI Model + Training Pipeline
├── backend/           # Giai đoạn 3: FastAPI REST API
└── frontend/          # Giai đoạn 3: Streamlit Dashboard
```

## 🚀 Khởi động nhanh (Docker)

```bash
# Clone project
git clone <repo-url>
cd smarttransit-ai

# Copy file cấu hình môi trường
cp .env.example .env

# Build & chạy toàn bộ hệ thống
docker-compose up --build

# Hoặc chạy từng service
docker-compose up db          # PostgreSQL
docker-compose up backend     # FastAPI :8000
docker-compose up frontend    # Streamlit :8501
```

## 🔧 Chạy thủ công (không Docker)

### 1. Cài dependencies
```bash
pip install -r requirements.txt
```

### 2. Khởi động PostgreSQL & tạo schema
```bash
psql -U postgres -c "CREATE DATABASE smarttransit;"
psql -U postgres -d smarttransit -f database/schema.sql
```

### 3. Đổ mock data TP.HCM
```bash
python database/mock_data.py
```

### 4. Train AI model
```bash
python ai_core/src/data_processing.py
python ai_core/src/train_pipeline.py
```

### 5. Khởi động Backend API
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Khởi động Frontend
```bash
cd frontend
streamlit run app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Health check |
| GET | `/api/v1/segments` | Danh sách đoạn đường |
| POST | `/api/v1/predict` | Dự báo mức độ kẹt xe |
| GET | `/api/v1/realtime` | Trạng thái giao thông hiện tại |
| GET | `/api/v1/hotspots` | Top điểm kẹt xe nặng nhất |
| GET | `/api/v1/history/{segment_id}` | Lịch sử một đoạn đường |

### Ví dụ request dự báo:
```json
POST /api/v1/predict
{
  "segment_id": "Q1_NTH_001",
  "hour": 17,
  "day_of_week": 2,
  "month": 6,
  "is_holiday": false,
  "weather_condition": "rain"
}
```

---

## 🗺️ Dữ liệu Mock — Các tuyến đường TP.HCM

- **Quận 1**: Nguyễn Trãi, Lê Lợi, Đồng Khởi, Đinh Tiên Hoàng
- **Quận 3**: Võ Văn Tần, Nam Kỳ Khởi Nghĩa, Cách Mạng Tháng 8
- **Bình Thạnh**: Đinh Bộ Lĩnh, Xô Viết Nghệ Tĩnh, Bạch Đằng
- **Tân Bình**: Hoàng Văn Thụ, Cộng Hòa, Trường Chinh
- **Gò Vấp**: Nguyễn Kiệm, Lê Văn Thọ, Phạm Văn Chiêu

---

## 🤖 AI Model

- **Thuật toán**: Random Forest Classifier (sklearn)
- **Features**: giờ, ngày tuần, tháng, tốc độ TB, mưa, ngày lễ, mật độ xe
- **Labels**: `0=Thông thoáng`, `1=Chậm`, `2=Kẹt nhẹ`, `3=Kẹt nặng`
- **Accuracy mục tiêu**: ≥ 85% trên tập test

---

## 🛠️ Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| Database | PostgreSQL 15 |
| AI/ML | Scikit-learn, Pandas, NumPy |
| Backend | FastAPI, SQLAlchemy, Uvicorn |
| Frontend | Streamlit, Folium, Plotly |
| DevOps | Docker, Docker Compose |

---

## 📊 Giao diện Dashboard

- **Bản đồ nhiệt** — Hiển thị điểm kẹt xe theo màu sắc
- **Biểu đồ theo giờ** — Mức độ ùn tắc theo khung giờ
- **Dự báo thông minh** — Nhập thông tin, AI trả kết quả ngay
- **Top điểm đen** — 10 đoạn đường kẹt nhất

---

## 👥 Đội phát triển

Dự án SmartTransit AI — Giải pháp giao thông thông minh cho TP.HCM 🇻🇳
