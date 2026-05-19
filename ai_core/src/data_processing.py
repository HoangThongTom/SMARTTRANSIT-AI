"""
SmartTransit AI — Data Processing Pipeline
Clean, transform và tạo features từ dữ liệu giao thông thô.
"""

import os
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://smartuser:smartpass123@localhost:5432/smarttransit")
OUTPUT_TRAIN = "ai_core/models/train_data.parquet"
OUTPUT_TEST  = "ai_core/models/test_data.parquet"

FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "is_holiday", "is_weekend", "is_rush_hour",
    "weather_encoded", "vehicle_count",
    "hour_sin", "hour_cos",          # Cyclic encoding cho giờ
    "dow_sin", "dow_cos",            # Cyclic encoding cho ngày
    "month_sin", "month_cos",        # Cyclic encoding cho tháng
    "speed_lag1h", "speed_lag24h",   # Lag features
    "speed_rolling3h",               # Rolling average
]

TARGET_COL = "congestion_level"

WEATHER_MAP = {
    "clear": 0, "fog": 1, "rain": 2, "heavy_rain": 3
}


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_raw_data(engine) -> pd.DataFrame:
    logger.info("📥 Load dữ liệu từ PostgreSQL...")
    query = """
        SELECT
            tr.segment_id,
            tr.recorded_at,
            tr.hour,
            tr.day_of_week,
            tr.month,
            tr.is_holiday,
            tr.weather_condition,
            tr.avg_speed_kmh,
            tr.vehicle_count,
            tr.congestion_level,
            rs.district_code,
            rs.road_type,
            rs.lanes
        FROM traffic_records tr
        JOIN road_segments rs ON tr.segment_id = rs.segment_id
        ORDER BY tr.segment_id, tr.recorded_at
    """
    df = pd.read_sql(query, engine)
    logger.success(f"  ✅ Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: DATA CLEANING
# ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🧹 Làm sạch dữ liệu...")
    original_len = len(df)

    # 2.1 Xử lý missing values
    df["avg_speed_kmh"].fillna(df["avg_speed_kmh"].median(), inplace=True)
    df["vehicle_count"].fillna(df["vehicle_count"].median(), inplace=True)
    df["weather_condition"].fillna("clear", inplace=True)

    # 2.2 Loại bỏ outliers (speed < 0 hoặc > 150 km/h)
    before = len(df)
    df = df[(df["avg_speed_kmh"] >= 0) & (df["avg_speed_kmh"] <= 120)]
    df = df[(df["vehicle_count"] >= 0) & (df["vehicle_count"] <= 2000)]
    logger.info(f"  Loại bỏ {before - len(df):,} outliers")

    # 2.3 Đảm bảo kiểu dữ liệu
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df["is_holiday"] = df["is_holiday"].astype(bool)
    df["congestion_level"] = df["congestion_level"].astype(int)

    logger.success(f"  ✅ Còn lại {len(df):,}/{original_len:,} rows sau cleaning")
    return df


# ─────────────────────────────────────────────────────────────
# BƯỚC 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("⚙️  Feature Engineering...")
    df = df.copy()
    df = df.sort_values(["segment_id", "recorded_at"])

    # 3.1 Boolean features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"] >= 6) & (df["hour"] <= 8)) |
        ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)

    # 3.2 Weather encoding
    df["weather_encoded"] = df["weather_condition"].map(WEATHER_MAP).fillna(0).astype(int)

    # 3.3 Cyclic encoding (tránh discontinuity 23→0)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 3.4 Lag features (speed của 1 giờ và 24 giờ trước)
    df["speed_lag1h"]  = df.groupby("segment_id")["avg_speed_kmh"].shift(2)   # 2 records = 1h
    df["speed_lag24h"] = df.groupby("segment_id")["avg_speed_kmh"].shift(48)  # 48 records = 24h

    # 3.5 Rolling average (3 giờ qua)
    df["speed_rolling3h"] = df.groupby("segment_id")["avg_speed_kmh"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )

    # 3.6 District encoding
    district_dummies = pd.get_dummies(df["district_code"], prefix="dist", drop_first=True)
    df = pd.concat([df, district_dummies], axis=1)

    # 3.7 Road type encoding
    road_dummies = pd.get_dummies(df["road_type"], prefix="road", drop_first=True)
    df = pd.concat([df, road_dummies], axis=1)

    # Điền NaN từ lag features bằng median
    for col in ["speed_lag1h", "speed_lag24h", "speed_rolling3h"]:
        df[col].fillna(df["avg_speed_kmh"].median(), inplace=True)

    # Boolean → int
    df["is_holiday"] = df["is_holiday"].astype(int)

    logger.success(f"  ✅ {len(df.columns)} features sau engineering")
    return df


# ─────────────────────────────────────────────────────────────
# BƯỚC 4: TRAIN/TEST SPLIT (theo thời gian - không random)
# ─────────────────────────────────────────────────────────────
def time_based_split(df: pd.DataFrame, test_ratio=0.2):
    """Chia dữ liệu theo thời gian: 80% train, 20% test gần nhất."""
    logger.info("✂️  Time-based train/test split...")
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    logger.success(f"  Train: {len(train):,} | Test: {len(test):,}")
    return train, test


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: EXPORT
# ─────────────────────────────────────────────────────────────
def export_data(train: pd.DataFrame, test: pd.DataFrame):
    os.makedirs("ai_core/models", exist_ok=True)
    train.to_parquet(OUTPUT_TRAIN, index=False)
    test.to_parquet(OUTPUT_TEST, index=False)
    logger.success(f"  ✅ Saved: {OUTPUT_TRAIN}, {OUTPUT_TEST}")

    # Stats
    logger.info("📊 Phân phối nhãn:")
    for name, data in [("TRAIN", train), ("TEST", test)]:
        dist = data[TARGET_COL].value_counts(normalize=True).sort_index()
        for lvl, pct in dist.items():
            labels = {0: "Thông thoáng", 1: "Chậm", 2: "Kẹt nhẹ", 3: "Kẹt nặng"}
            logger.info(f"  {name} | Level {lvl} ({labels[lvl]}): {pct*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run():
    logger.info("=" * 55)
    logger.info("  SmartTransit AI — Data Processing Pipeline")
    logger.info("=" * 55)

    engine = create_engine(DATABASE_URL)
    df_raw = load_raw_data(engine)
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    train, test = time_based_split(df_features)
    export_data(train, test)

    logger.success("✅ Pipeline hoàn thành! Sẵn sàng train model.")
    return train, test


if __name__ == "__main__":
    run()
