"""
SmartTransit AI — Mock Data Generator
Tạo dữ liệu giả lập giao thông TP.HCM để train AI và test hệ thống.
"""

import os
import random
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from loguru import logger

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://smartuser:smartpass123@localhost:5432/smarttransit")
MONTHS_HISTORY = 6          # Sinh 6 tháng dữ liệu lịch sử
RECORDS_PER_SEGMENT = 2160  # ~1 record/giờ trong 90 ngày

# ─────────────────────────────────────────────────────────────
# DỮ LIỆU CÁC QUẬN/HUYỆN TP.HCM
# ─────────────────────────────────────────────────────────────
DISTRICTS = [
    {"code": "Q1",  "name": "Quận 1",       "area_km2": 7.73,  "population": 146000},
    {"code": "Q3",  "name": "Quận 3",       "area_km2": 4.92,  "population": 188000},
    {"code": "Q5",  "name": "Quận 5",       "area_km2": 4.27,  "population": 171000},
    {"code": "BT",  "name": "Bình Thạnh",   "area_km2": 20.76, "population": 495000},
    {"code": "TB",  "name": "Tân Bình",     "area_km2": 22.37, "population": 487000},
    {"code": "GV",  "name": "Gò Vấp",       "area_km2": 19.74, "population": 658000},
    {"code": "PN",  "name": "Phú Nhuận",    "area_km2": 4.88,  "population": 163000},
    {"code": "TD",  "name": "Thủ Đức",      "area_km2": 211.6, "population": 1100000},
]

# ─────────────────────────────────────────────────────────────
# DỮ LIỆU ĐƯỜNG TP.HCM (có tọa độ thực)
# ─────────────────────────────────────────────────────────────
ROAD_SEGMENTS = [
    # QUẬN 1
    {"segment_id": "Q1_NTH_001", "district": "Q1", "street": "Nguyễn Trãi",
     "from_p": "Nguyễn Trãi - Trần Hưng Đạo", "to_p": "Nguyễn Trãi - Phạm Ngũ Lão",
     "length": 0.8, "lanes": 4, "type": "primary",
     "lat_s": 10.7671, "lon_s": 106.6838, "lat_e": 10.7638, "lon_e": 106.6901},
    {"segment_id": "Q1_LLO_001", "district": "Q1", "street": "Lê Lợi",
     "from_p": "Lê Lợi - Nam Kỳ Khởi Nghĩa", "to_p": "Lê Lợi - Pasteur",
     "length": 0.5, "lanes": 4, "type": "primary",
     "lat_s": 10.7731, "lon_s": 106.6988, "lat_e": 10.7745, "lon_e": 106.7021},
    {"segment_id": "Q1_DKH_001", "district": "Q1", "street": "Đồng Khởi",
     "from_p": "Đồng Khởi - Lê Lợi", "to_p": "Đồng Khởi - Nguyễn Du",
     "length": 0.4, "lanes": 4, "type": "primary",
     "lat_s": 10.7745, "lon_s": 106.7021, "lat_e": 10.7774, "lon_e": 106.7035},
    {"segment_id": "Q1_DTH_001", "district": "Q1", "street": "Đinh Tiên Hoàng",
     "from_p": "Đinh Tiên Hoàng - Đinh Lễ", "to_p": "Đinh Tiên Hoàng - Lê Thánh Tôn",
     "length": 0.6, "lanes": 2, "type": "secondary",
     "lat_s": 10.7750, "lon_s": 106.6985, "lat_e": 10.7763, "lon_e": 106.7002},

    # QUẬN 3
    {"segment_id": "Q3_VVT_001", "district": "Q3", "street": "Võ Văn Tần",
     "from_p": "Võ Văn Tần - CMT8", "to_p": "Võ Văn Tần - Phạm Ngọc Thạch",
     "length": 0.9, "lanes": 4, "type": "primary",
     "lat_s": 10.7795, "lon_s": 106.6932, "lat_e": 10.7812, "lon_e": 106.6975},
    {"segment_id": "Q3_NKKN_001", "district": "Q3", "street": "Nam Kỳ Khởi Nghĩa",
     "from_p": "NKKN - Lý Tự Trọng", "to_p": "NKKN - Nguyễn Thị Minh Khai",
     "length": 1.1, "lanes": 4, "type": "primary",
     "lat_s": 10.7745, "lon_s": 106.6988, "lat_e": 10.7812, "lon_e": 106.6975},
    {"segment_id": "Q3_CMT8_001", "district": "Q3", "street": "Cách Mạng Tháng 8",
     "from_p": "CMT8 - Nguyễn Thị Minh Khai", "to_p": "CMT8 - Điện Biên Phủ",
     "length": 1.3, "lanes": 4, "type": "primary",
     "lat_s": 10.7812, "lon_s": 106.6925, "lat_e": 10.7901, "lon_e": 106.6889},

    # BÌNH THẠNH
    {"segment_id": "BT_DBL_001", "district": "BT", "street": "Đinh Bộ Lĩnh",
     "from_p": "Đinh Bộ Lĩnh - Phan Đăng Lưu", "to_p": "Đinh Bộ Lĩnh - Nơ Trang Long",
     "length": 1.5, "lanes": 4, "type": "primary",
     "lat_s": 10.8023, "lon_s": 106.7103, "lat_e": 10.8156, "lon_e": 106.7012},
    {"segment_id": "BT_XVNT_001", "district": "BT", "street": "Xô Viết Nghệ Tĩnh",
     "from_p": "XVNT - Đinh Bộ Lĩnh", "to_p": "XVNT - Nguyễn Xí",
     "length": 2.1, "lanes": 6, "type": "primary",
     "lat_s": 10.8023, "lon_s": 106.7103, "lat_e": 10.7923, "lon_e": 106.7234},
    {"segment_id": "BT_BD_001", "district": "BT", "street": "Bạch Đằng",
     "from_p": "Bạch Đằng - Nguyễn Hữu Cảnh", "to_p": "Bạch Đằng - Đinh Bộ Lĩnh",
     "length": 1.8, "lanes": 4, "type": "primary",
     "lat_s": 10.7923, "lon_s": 106.7178, "lat_e": 10.8023, "lon_e": 106.7103},

    # TÂN BÌNH
    {"segment_id": "TB_HVT_001", "district": "TB", "street": "Hoàng Văn Thụ",
     "from_p": "Hoàng Văn Thụ - Phan Đình Phùng", "to_p": "Hoàng Văn Thụ - Nguyễn Văn Trỗi",
     "length": 2.0, "lanes": 4, "type": "primary",
     "lat_s": 10.7934, "lon_s": 106.6623, "lat_e": 10.8056, "lon_e": 106.6712},
    {"segment_id": "TB_CH_001", "district": "TB", "street": "Cộng Hòa",
     "from_p": "Cộng Hòa - Hoàng Văn Thụ", "to_p": "Cộng Hòa - Sân bay TSN",
     "length": 2.5, "lanes": 6, "type": "primary",
     "lat_s": 10.8056, "lon_s": 106.6712, "lat_e": 10.8189, "lon_e": 106.6656},
    {"segment_id": "TB_TC_001", "district": "TB", "street": "Trường Chinh",
     "from_p": "Trường Chinh - Âu Cơ", "to_p": "Trường Chinh - Cộng Hòa",
     "length": 3.2, "lanes": 6, "type": "primary",
     "lat_s": 10.7956, "lon_s": 106.6567, "lat_e": 10.8056, "lon_e": 106.6712},

    # GÒ VẤP
    {"segment_id": "GV_NK_001", "district": "GV", "street": "Nguyễn Kiệm",
     "from_p": "Nguyễn Kiệm - Phan Văn Trị", "to_p": "Nguyễn Kiệm - Hoàng Minh Giám",
     "length": 2.8, "lanes": 4, "type": "primary",
     "lat_s": 10.8145, "lon_s": 106.6789, "lat_e": 10.8212, "lon_e": 106.6923},
    {"segment_id": "GV_LVT_001", "district": "GV", "street": "Lê Văn Thọ",
     "from_p": "Lê Văn Thọ - Nguyễn Kiệm", "to_p": "Lê Văn Thọ - Phạm Văn Chiêu",
     "length": 1.7, "lanes": 4, "type": "secondary",
     "lat_s": 10.8267, "lon_s": 106.6845, "lat_e": 10.8312, "lon_e": 106.6712},
    {"segment_id": "GV_PVC_001", "district": "GV", "street": "Phạm Văn Chiêu",
     "from_p": "Phạm Văn Chiêu - Lê Văn Thọ", "to_p": "Phạm Văn Chiêu - Quang Trung",
     "length": 2.3, "lanes": 4, "type": "secondary",
     "lat_s": 10.8312, "lon_s": 106.6712, "lat_e": 10.8389, "lon_e": 106.6823},

    # PHÚ NHUẬN
    {"segment_id": "PN_PCT_001", "district": "PN", "street": "Phan Xích Long",
     "from_p": "Phan Xích Long - Hoa Sứ", "to_p": "Phan Xích Long - Nguyễn Văn Trỗi",
     "length": 1.2, "lanes": 2, "type": "secondary",
     "lat_s": 10.8034, "lon_s": 106.6912, "lat_e": 10.8056, "lon_e": 106.7012},
    {"segment_id": "PN_NVT_001", "district": "PN", "street": "Nguyễn Văn Trỗi",
     "from_p": "NV Trỗi - Hoàng Diệu 2", "to_p": "NV Trỗi - Hoàng Văn Thụ",
     "length": 2.4, "lanes": 4, "type": "primary",
     "lat_s": 10.8034, "lon_s": 106.6912, "lat_e": 10.7934, "lon_e": 106.6823},

    # THỦ ĐỨC
    {"segment_id": "TD_PVD_001", "district": "TD", "street": "Phạm Văn Đồng",
     "from_p": "Phạm Văn Đồng - Nguyễn Xí", "to_p": "Phạm Văn Đồng - Gò Dưa",
     "length": 4.5, "lanes": 8, "type": "primary",
     "lat_s": 10.8234, "lon_s": 106.7234, "lat_e": 10.8567, "lon_e": 106.7456},
    {"segment_id": "TD_VNP_001", "district": "TD", "street": "Võ Nguyên Giáp",
     "from_p": "VNG - Bình Chiểu", "to_p": "VNG - Linh Đông",
     "length": 3.8, "lanes": 6, "type": "primary",
     "lat_s": 10.8456, "lon_s": 106.7523, "lat_e": 10.8712, "lon_e": 106.7689},
]

# ─────────────────────────────────────────────────────────────
# MÔ HÌNH KẸT XE THEO GIỜ (thực tế TP.HCM)
# ─────────────────────────────────────────────────────────────
HOURLY_CONGESTION_PATTERN = {
    0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05,
    4: 0.08, 5: 0.12,
    6: 0.45,   # Bắt đầu giờ cao điểm sáng
    7: 0.85,   # Đỉnh sáng
    8: 0.75,
    9: 0.55,
    10: 0.35, 11: 0.30,
    12: 0.45,  # Giờ trưa
    13: 0.40, 14: 0.30, 15: 0.35,
    16: 0.65,  # Bắt đầu giờ cao điểm chiều
    17: 0.90,  # Đỉnh chiều (kẹt nhất!)
    18: 0.85,
    19: 0.65,
    20: 0.45, 21: 0.35,
    22: 0.20, 23: 0.10,
}

# Weekend thấp hơn ngày thường ~30%
WEEKEND_MULTIPLIER = 0.65

# ─────────────────────────────────────────────────────────────
# HÀM TÍNH MỨC ĐỘ KẸT XE
# ─────────────────────────────────────────────────────────────
def get_congestion_level(congestion_prob: float, is_rain: bool) -> tuple:
    """
    Trả về (congestion_level, avg_speed, vehicle_count)
    """
    rain_boost = 0.15 if is_rain else 0.0
    prob = min(1.0, congestion_prob + rain_boost + random.uniform(-0.1, 0.1))

    if prob < 0.2:
        level = 0
        speed = random.uniform(45, 70)
        base_vehicles = random.randint(50, 200)
    elif prob < 0.5:
        level = 1
        speed = random.uniform(20, 45)
        base_vehicles = random.randint(200, 500)
    elif prob < 0.75:
        level = 2
        speed = random.uniform(8, 20)
        base_vehicles = random.randint(400, 800)
    else:
        level = 3
        speed = random.uniform(2, 8)
        base_vehicles = random.randint(600, 1200)
    # Thêm ngẫu nhiên cho lượng xe (để có phân phối thực tế hơn)
    noise = random.randint(-120, 120)
    vehicles = max(20, base_vehicles + noise )   

    return level, round(speed, 2), vehicles


def get_weather(month: int, hour: int) -> str:
    """Mô phỏng thời tiết TP.HCM theo mùa."""
    # Mùa mưa: tháng 5-11, hay mưa chiều
    is_rain_season = month in [5, 6, 7, 8, 9, 10, 11]
    is_rain_hour = hour in range(14, 20)  # Hay mưa chiều

    if is_rain_season and is_rain_hour and random.random() < 0.40:
        return "heavy_rain" if random.random() < 0.3 else "rain"
    elif is_rain_season and random.random() < 0.15:
        return "rain"
    elif not is_rain_season and random.random() < 0.05:
        return "fog"
    return "clear"


# ─────────────────────────────────────────────────────────────
# CÁC NGÀY LỄ VIỆT NAM 2024
# ─────────────────────────────────────────────────────────────
HOLIDAYS_2024 = [
    datetime(2024, 1, 1),   # Tết Dương lịch
    datetime(2024, 2, 8),   # Mùng 1 Tết Giáp Thìn
    datetime(2024, 2, 9),   # Mùng 2 Tết
    datetime(2024, 2, 10),  # Mùng 3 Tết
    datetime(2024, 2, 11),  # Mùng 4 Tết
    datetime(2024, 2, 12),  # Mùng 5 Tết
    datetime(2024, 4, 18),  # Giỗ Tổ Hùng Vương
    datetime(2024, 4, 30),  # Ngày Giải phóng
    datetime(2024, 5, 1),   # Quốc tế Lao động
    datetime(2024, 9, 2),   # Quốc khánh
]


def is_holiday(dt: datetime) -> bool:
    return dt.date() in [h.date() for h in HOLIDAYS_2024]


# ─────────────────────────────────────────────────────────────
# HÀM CHÍNH: INSERT DATA
# ─────────────────────────────────────────────────────────────
def seed_database():
    logger.info("🚀 Bắt đầu đổ mock data SmartTransit AI...")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    try:
        # 1. Insert Districts
        logger.info("📍 Insert dữ liệu Quận/Huyện...")
        for d in DISTRICTS:
            cur.execute("""
                INSERT INTO districts (code, name, area_km2, population)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO NOTHING
            """, (d["code"], d["name"], d["area_km2"], d["population"]))
        conn.commit()
        logger.success(f"  ✅ {len(DISTRICTS)} quận/huyện")

        # 2. Insert Road Segments
        logger.info("🛣️  Insert dữ liệu đoạn đường...")
        for seg in ROAD_SEGMENTS:
            cur.execute("""
                INSERT INTO road_segments
                    (segment_id, district_code, street_name, from_point, to_point,
                     length_km, lanes, road_type, lat_start, lon_start, lat_end, lon_end)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (segment_id) DO NOTHING
            """, (
                seg["segment_id"], seg["district"], seg["street"],
                seg["from_p"], seg["to_p"], seg["length"], seg["lanes"],
                seg["type"], seg["lat_s"], seg["lon_s"], seg["lat_e"], seg["lon_e"]
            ))
        conn.commit()
        logger.success(f"  ✅ {len(ROAD_SEGMENTS)} đoạn đường")

        # 3. Insert Traffic Records
        logger.info("📊 Sinh dữ liệu lịch sử giao thông (6 tháng)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * MONTHS_HISTORY)

        total_records = 0
        batch = []
        BATCH_SIZE = 500

        current_date = start_date
        while current_date <= end_date:
            for seg in ROAD_SEGMENTS:
                # Mỗi 30 phút một record
                for hour in range(0, 24):
                    for minute in [0, 30]:
                        dt = current_date.replace(hour=hour, minute=minute, second=0)
                        dow = dt.weekday()  # 0=Monday
                        is_weekend = dow >= 5

                        base_prob = HOURLY_CONGESTION_PATTERN[hour]
                        if is_weekend:
                            base_prob *= WEEKEND_MULTIPLIER

                        weather = get_weather(dt.month, hour)
                        is_rain = "rain" in weather
                        level, speed, vehicles = get_congestion_level(base_prob, is_rain)
                        holiday = is_holiday(dt)

                        batch.append((
                            seg["segment_id"], dt, hour, dow, dt.month,
                            holiday, weather, speed, vehicles, level
                        ))

                        if len(batch) >= BATCH_SIZE:
                            cur.executemany("""
                                INSERT INTO traffic_records
                                    (segment_id, recorded_at, hour, day_of_week, month,
                                     is_holiday, weather_condition, avg_speed_kmh, vehicle_count,
                                     congestion_level)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, batch)
                            conn.commit()
                            total_records += len(batch)
                            batch.clear()

            current_date += timedelta(days=1)

        # Insert phần còn lại
        if batch:
            cur.executemany("""
                INSERT INTO traffic_records
                    (segment_id, recorded_at, hour, day_of_week, month,
                     is_holiday, weather_condition, avg_speed_kmh, vehicle_count,
                     congestion_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, batch)
            conn.commit()
            total_records += len(batch)

        logger.success(f"  ✅ {total_records:,} records giao thông")

        # 4. Insert Special Events
        logger.info("📅 Insert sự kiện đặc biệt...")
        events = [
            ("Tết Nguyên Đán Giáp Thìn", "holiday",
             datetime(2024, 2, 8), datetime(2024, 2, 14), "TP.HCM", 3),
            ("Đua xe F1 Night Race", "sport",
             datetime(2024, 11, 15), datetime(2024, 11, 17), "Quận 1", 3),
            ("Hội chợ Du lịch Quốc tế TPHCM", "concert",
             datetime(2024, 9, 5), datetime(2024, 9, 8), "Quận 1", 2),
            ("Marathon TP.HCM 2024", "sport",
             datetime(2024, 10, 27), datetime(2024, 10, 27), "Quận 1-3", 2),
        ]
        for ev in events:
            cur.execute("""
                INSERT INTO special_events (event_name, event_type, start_time, end_time, affected_area, impact_level)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, ev)
        conn.commit()
        logger.success(f"  ✅ {len(events)} sự kiện đặc biệt")

        # Summary
        cur.execute("SELECT COUNT(*) FROM traffic_records")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM road_segments")
        segments = cur.fetchone()[0]
        logger.success(f"""
╔══════════════════════════════════════════╗
║     🎉 MOCK DATA HOÀN THÀNH!           ║
╠══════════════════════════════════════════╣
║  Quận/Huyện  : {len(DISTRICTS)} quận              ║
║  Đoạn đường  : {segments} đường              ║
║  Records     : {total:>12,} bản ghi  ║
║  Thời gian   : 6 tháng lịch sử        ║
╚══════════════════════════════════════════╝
        """)

    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Lỗi: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    seed_database()
