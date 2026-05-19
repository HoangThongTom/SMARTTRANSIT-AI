"""
SmartTransit AI — FastAPI Backend
REST API cho hệ thống dự báo kẹt xe TP.HCM
"""

import os
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from loguru import logger

from app.config import get_db, check_db_connection, MODEL_PATH, SCALER_PATH
from app.Predictor import predictor, PredictionInput, PredictionOutput

# ─────────────────────────────────────────────────────────────
# LIFESPAN (startup / shutdown)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy khi server start và shutdown."""
    logger.info("🚀 SmartTransit API đang khởi động...")

    # Kiểm tra DB
    if not check_db_connection():
        logger.warning("⚠️  Database chưa kết nối được — một số endpoint sẽ không hoạt động")

    # Load AI model
    metrics_path = MODEL_PATH.replace("traffic_model.joblib", "metrics.json")
    scaler_path  = MODEL_PATH.replace("traffic_model.joblib", "scaler.joblib")
    try:
        predictor.load(MODEL_PATH, scaler_path, metrics_path)
        logger.success("✅ AI Model loaded thành công")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  {e} — Hãy chạy train_pipeline.py trước")

    logger.success("✅ SmartTransit API sẵn sàng!")
    yield
    logger.info("👋 API đang tắt...")


# ─────────────────────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartTransit AI",
    description="🚌 API dự báo tình trạng kẹt xe TP.HCM bằng Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# ROOT & HEALTH
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "name": "SmartTransit AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.is_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/model-info", tags=["System"])
def model_info():
    """Thông tin về AI model đang chạy."""
    return predictor.model_info


# ─────────────────────────────────────────────────────────────
# ROAD SEGMENTS
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/segments", tags=["Roads"])
def get_segments(
    district: Optional[str] = Query(None, description="Lọc theo quận: Q1, Q3, BT..."),
    db: Session = Depends(get_db)
):
    """Danh sách tất cả đoạn đường."""
    query = text("""
        SELECT rs.*, d.name as district_name
        FROM road_segments rs
        JOIN districts d ON rs.district_code = d.code
        WHERE (:district IS NULL OR rs.district_code = :district)
        ORDER BY rs.district_code, rs.street_name
    """)
    result = db.execute(query, {"district": district}).mappings().all()
    return {"count": len(result), "segments": [dict(r) for r in result]}


@app.get("/api/v1/segments/{segment_id}", tags=["Roads"])
def get_segment(segment_id: str, db: Session = Depends(get_db)):
    """Chi tiết một đoạn đường."""
    result = db.execute(
        text("SELECT * FROM road_segments WHERE segment_id = :sid"),
        {"sid": segment_id}
    ).mappings().first()
    if not result:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy segment: {segment_id}")
    return dict(result)


# ─────────────────────────────────────────────────────────────
# REAL-TIME TRAFFIC
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/realtime", tags=["Traffic"])
def get_realtime_traffic(
    district: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Trạng thái giao thông hiện tại (từ view current_traffic_status)."""
    query = """
        SELECT *
        FROM current_traffic_status
        WHERE (:district IS NULL OR district_code = :district)
        ORDER BY congestion_level DESC, district_code
    """
    result = db.execute(text(query), {"district": district}).mappings().all()
    data = [dict(r) for r in result]

    # Đếm theo mức độ
    summary = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in data:
        lvl = row.get("congestion_level") or 0
        summary[lvl] = summary.get(lvl, 0) + 1

    return {
        "timestamp": datetime.now().isoformat(),
        "total_segments": len(data),
        "summary": {
            "thong_thoang": summary[0],
            "cham": summary[1],
            "ket_nhe": summary[2],
            "ket_nang": summary[3],
        },
        "segments": data,
    }


@app.get("/api/v1/hotspots", tags=["Traffic"])
def get_hotspots(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Top N đoạn đường kẹt xe nặng nhất."""
    query = text("""
        SELECT segment_id, street_name, district_code,
               congestion_level, avg_speed_kmh, status_label, status_color,
               lat_start, lon_start, lat_end, lon_end
        FROM current_traffic_status
        ORDER BY congestion_level DESC, avg_speed_kmh ASC
        LIMIT :limit
    """)
    result = db.execute(query, {"limit": limit}).mappings().all()
    return {"hotspots": [dict(r) for r in result]}


# ─────────────────────────────────────────────────────────────
# HISTORICAL DATA
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/history/{segment_id}", tags=["Traffic"])
def get_history(
    segment_id: str,
    hours: int = Query(24, ge=1, le=168, description="Số giờ lịch sử cần lấy"),
    db: Session = Depends(get_db)
):
    """Lịch sử giao thông của một đoạn đường."""
    query = text("""
        SELECT recorded_at, hour, avg_speed_kmh, vehicle_count,
               congestion_level, weather_condition
        FROM traffic_records
        WHERE segment_id = :sid
          AND recorded_at >= NOW() - INTERVAL '1 hour' * :hours
        ORDER BY recorded_at DESC
        LIMIT 200
    """)
    result = db.execute(query, {"sid": segment_id, "hours": hours}).mappings().all()
    return {
        "segment_id": segment_id,
        "hours_requested": hours,
        "records": [dict(r) for r in result]
    }


@app.get("/api/v1/stats/hourly", tags=["Analytics"])
def get_hourly_stats(
    day_of_week: Optional[int] = Query(None, ge=0, le=6),
    db: Session = Depends(get_db)
):
    """Thống kê kẹt xe theo giờ trong ngày."""
    query = text("""
        SELECT hour, day_of_week, avg_congestion, avg_speed, record_count
        FROM hourly_congestion_stats
        WHERE (:dow IS NULL OR day_of_week = :dow)
        ORDER BY day_of_week, hour
    """)
    result = db.execute(query, {"dow": day_of_week}).mappings().all()
    return {"stats": [dict(r) for r in result]}


# ─────────────────────────────────────────────────────────────
# AI PREDICTION
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/predict", response_model=PredictionOutput, tags=["AI Prediction"])
def predict_congestion(inp: PredictionInput):
    """
    Dự báo mức độ kẹt xe cho một đoạn đường.

    - **segment_id**: Mã đoạn đường (VD: Q1_NTH_001)
    - **hour**: Giờ cần dự báo (0-23)
    - **day_of_week**: Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)
    - **weather_condition**: Thời tiết (clear/rain/heavy_rain/fog)
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="AI Model chưa sẵn sàng. Vui lòng train model trước: python ai_core/src/train_pipeline.py"
        )
    try:
        result = predictor.predict(inp)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", tags=["AI Prediction"])
def predict_batch(inputs: List[PredictionInput]):
    """Dự báo nhiều đoạn đường cùng lúc (tối đa 50)."""
    if len(inputs) > 50:
        raise HTTPException(status_code=400, detail="Tối đa 50 segments mỗi batch")
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="AI Model chưa sẵn sàng")

    results = []
    for inp in inputs:
        try:
            results.append(predictor.predict(inp).model_dump())
        except Exception as e:
            results.append({"segment_id": inp.segment_id, "error": str(e)})

    return {"count": len(results), "predictions": results}


# ─────────────────────────────────────────────────────────────
# DISTRICTS
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/districts", tags=["Roads"])
def get_districts(db: Session = Depends(get_db)):
    """Danh sách quận/huyện."""
    result = db.execute(text("SELECT * FROM districts ORDER BY code")).mappings().all()
    return {"districts": [dict(r) for r in result]}


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from app.config import API_HOST, API_PORT
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
