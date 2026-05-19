FROM python:3.11-slim

WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết cho Healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Sử dụng context từ thư mục gốc để nạp requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit folium streamlit-folium plotly pandas requests loguru psycopg2-binary

# Nạp mã nguồn ứng dụng frontend
COPY frontend/ /app/frontend/

WORKDIR /app/frontend

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]