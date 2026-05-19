# 🚌 SmartTransit AI — Hệ thống Dự báo Kẹt xe TP.HCM

Hệ thống AI dự báo tình trạng giao thông TP.HCM theo thời gian thực, ứng dụng học máy phân tích chu kỳ thời gian đô thị đô thị và lịch sử trượt vận tốc để cảnh báo mức độ ùn tắc giao thông.

---

## 🏗️ Kiến trúc hệ thống


```

smarttransit-ai/
├── database/         # Schema PostgreSQL + Mock Data Simulator TP.HCM
├── ai_core/          # Core xử lý dữ liệu (Pandas) + Pipeline huấn luyện (Scikit-learn)
├── backend/          # REST API (FastAPI) + Công cụ suy luận (Predictor)
└── frontend/         # Streamlit Web Dashboard trực quan hóa (Folium/Plotly)

```

## 🚀 Khởi động nhanh bằng Docker

```bash
# Clone project
git clone <repo-url>
cd smarttransit-ai

# Copy file cấu hình môi trường
cp .env.example .env

# Build và chạy toàn bộ hệ thống (Hạ tầng tự động chạy Seeder và Trainer)
docker-compose up --build -d

# Kiểm tra các service đang chạy ngầm
docker-compose ps

```

## 🔧 Khởi động thủ công để phát triển (Local Development)

### 1. Cài đặt các thư viện hệ thống

```bash
pip install -r requirements.txt

```

### 2. Kích hoạt Database và nạp dữ liệu lịch sử

```bash
# Đảm bảo container postgres_db đang chạy ngầm
docker-compose up -d db

# Chạy script sinh dữ liệu 6 tháng giả lập thực tế TP.HCM (173,760 bản ghi)
docker-compose up seeder

```

### 3. Huấn luyện Mô hình AI Core

```bash
# Chạy pipeline xử lý dữ liệu và fit thuật toán học máy
docker-compose up trainer

```

### 4. Khởi chạy Backend API (FastAPI)

Đứng tại thư mục gốc của dự án, thiết lập đường dẫn tìm kiếm package và kích hoạt server:

```powershell
# Trên PowerShell (Windows)
$env:PYTHONPATH="backend"; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

```

### 5. Khởi chạy Giao diện (Streamlit Dashboard)

```bash
cd frontend
streamlit run app.py

```

---

## 📡 Danh sách Cổng API Lõi

| Method | Endpoint | Quyền truy cập | Mô tả chức năng |
| --- | --- | --- | --- |
| GET | `/health` | Public | Kiểm tra sức khỏe hệ thống và kết nối DB |
| GET | `/api/v1/segments` | Public | Lấy danh sách 20 đoạn đường kèm tọa độ GPS |
| POST | `/api/v1/predict` | Public | Đẩy thông số bối cảnh, AI trả kết quả dự báo ùn tắc |
| POST | `/api/v1/predict/batch` | Public | Dự báo lưu lượng lớn hàng loạt đoạn đường cùng lúc |
| GET | `/api/v1/hotspots` | Public | Trích xuất top các điểm đen giao thông kẹt nặng nhất |

### Cấu trúc dữ liệu yêu cầu dự báo (`POST /api/v1/predict`):

```json
{
  "segment_id": "Q1_NTH_001",
  "hour": 17,
  "day_of_week": 2,
  "month": 6,
  "is_holiday": false,
  "weather_condition": "rain",
  "avg_speed_kmh": null,
  "vehicle_count": null,
  "lanes": 4
}

```

---

## 🤖 Kiến trúc AI Model Core

Mô hình phân loại đa lớp giải quyết bài toán ùn tắc đô thị bằng cách triệt tiêu hiện tượng rò rỉ dữ liệu (Target Leakage), không học các biến hệ quả hiện tại mà tập trung học sâu các đặc trưng chu kỳ bối cảnh và độ trễ lịch sử:

* **Thuật toán chính**: Random Forest Classifier (Scikit-learn) kết hợp xử lý chuẩn hóa `StandardScaler`.
* **Hệ thống Đặc trưng (Features)**:
* *Thời gian lượng giác*: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`.
* *Bối cảnh đô thị*: `is_holiday`, `is_weekend`, `is_rush_hour`, `weather_encoded`, `lanes`.
* *Độ trễ vận tốc lịch sử*: `speed_lag1h`, `speed_lag24h`, `speed_rolling3h`.


* **Phân loại Nhãn Đầu ra (Labels)**:
* `0`: Thông thoáng
* `1`: Chậm
* `2`: Kẹt nhẹ
* `3`: Kẹt nặng


* **Hiệu năng thực nghiệm**: Đạt **97.45% Accuracy** và **0.9746 F1-Score** trên tập dữ liệu kiểm thử độc lập (Test Split).

---

## 🛠️ Công nghệ Sử dụng (Tech Stack)

* **Database Layer**: PostgreSQL 18.
* **Data & AI Pipeline**: Scikit-learn, Pandas, NumPy, PyArrow.
* **Backend Service**: FastAPI, SQLAlchemy ORM, Uvicorn ASGI Server.
* **Frontend Dashboard**: Streamlit, Folium Map Engine, Plotly Component.
* **DevOps**: Docker, Docker Compose Containerization.

---

## 📊 Tính năng trên Web Dashboard

* **Interactive Map** — Bản đồ số vẽ các đoạn đường màu Xanh/Vàng/Đỏ trực quan theo thời gian thực.
* **Congestion Trends** — Biểu đồ phân tích xu hướng kẹt xe theo từng khung giờ bằng Plotly.
* **Smart Form Inference** — Giao diện form điều chỉnh giả lập bối cảnh để thử nghiệm khả năng phán đoán của AI.
