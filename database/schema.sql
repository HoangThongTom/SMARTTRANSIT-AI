-- ============================================================
-- SmartTransit AI — PostgreSQL Schema
-- Hệ thống dự báo kẹt xe TP.HCM
-- ============================================================

-- Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────
-- BẢNG 1: Quận/Huyện
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS districts (
    id          SERIAL PRIMARY KEY,
    code        VARCHAR(10) UNIQUE NOT NULL,       -- VD: Q1, Q3, BT
    name        VARCHAR(100) NOT NULL,              -- VD: Quận 1
    area_km2    NUMERIC(6,2),
    population  INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
-- BẢNG 2: Đoạn đường (Road Segments)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS road_segments (
    id              SERIAL PRIMARY KEY,
    segment_id      VARCHAR(30) UNIQUE NOT NULL,   -- VD: Q1_NTH_001
    district_code   VARCHAR(10) REFERENCES districts(code),
    street_name     VARCHAR(200) NOT NULL,
    from_point      VARCHAR(200),                  -- Từ giao lộ nào
    to_point        VARCHAR(200),                  -- Đến giao lộ nào
    length_km       NUMERIC(5,3),
    lanes           INTEGER DEFAULT 2,
    road_type       VARCHAR(30),                   -- primary, secondary, residential
    lat_start       NUMERIC(10,7),
    lon_start       NUMERIC(10,7),
    lat_end         NUMERIC(10,7),
    lon_end         NUMERIC(10,7),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
-- BẢNG 3: Dữ liệu lịch sử giao thông (Historical Traffic)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS traffic_records (
    id                  BIGSERIAL PRIMARY KEY,
    segment_id          VARCHAR(30) REFERENCES road_segments(segment_id),
    recorded_at         TIMESTAMPTZ NOT NULL,
    hour                SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    day_of_week         SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0=Mon
    month               SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    is_holiday          BOOLEAN DEFAULT FALSE,
    weather_condition   VARCHAR(20) DEFAULT 'clear',  -- clear, rain, heavy_rain, fog
    avg_speed_kmh       NUMERIC(5,2),                 -- Tốc độ trung bình (km/h)
    vehicle_count       INTEGER,                       -- Số lượng xe đếm được
    congestion_level    SMALLINT NOT NULL,             -- 0=OK, 1=Slow, 2=Jam, 3=HeavyJam
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Index cho query nhanh hơn
CREATE INDEX IF NOT EXISTS idx_traffic_segment ON traffic_records(segment_id);
CREATE INDEX IF NOT EXISTS idx_traffic_time ON traffic_records(recorded_at);
CREATE INDEX IF NOT EXISTS idx_traffic_hour ON traffic_records(hour);
CREATE INDEX IF NOT EXISTS idx_traffic_congestion ON traffic_records(congestion_level);

-- ─────────────────────────────────────────────────────────────
-- BẢNG 4: Dự báo AI (AI Predictions)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL PRIMARY KEY,
    segment_id          VARCHAR(30) REFERENCES road_segments(segment_id),
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    target_datetime     TIMESTAMPTZ NOT NULL,
    predicted_level     SMALLINT NOT NULL,             -- 0-3
    confidence_score    NUMERIC(4,3),                  -- 0.000 - 1.000
    model_version       VARCHAR(20) DEFAULT 'v1.0',
    input_features      JSONB,                         -- Lưu input để debug
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_segment ON predictions(segment_id);
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(target_datetime);

-- ─────────────────────────────────────────────────────────────
-- BẢNG 5: Sự kiện đặc biệt (Events)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS special_events (
    id              SERIAL PRIMARY KEY,
    event_name      VARCHAR(200) NOT NULL,
    event_type      VARCHAR(50),                   -- holiday, concert, sport, accident
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ NOT NULL,
    affected_area   VARCHAR(100),
    impact_level    SMALLINT DEFAULT 1,            -- 1=Low, 2=Medium, 3=High
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────
-- VIEW: Trạng thái giao thông hiện tại
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW current_traffic_status AS
SELECT
    rs.segment_id,
    rs.street_name,
    rs.district_code,
    rs.lat_start,
    rs.lon_start,
    rs.lat_end,
    rs.lon_end,
    tr.avg_speed_kmh,
    tr.vehicle_count,
    tr.congestion_level,
    tr.weather_condition,
    tr.recorded_at,
    CASE tr.congestion_level
        WHEN 0 THEN 'Thông thoáng'
        WHEN 1 THEN 'Chậm'
        WHEN 2 THEN 'Kẹt nhẹ'
        WHEN 3 THEN 'Kẹt nặng'
    END AS status_label,
    CASE tr.congestion_level
        WHEN 0 THEN '#27ae60'
        WHEN 1 THEN '#f39c12'
        WHEN 2 THEN '#e67e22'
        WHEN 3 THEN '#c0392b'
    END AS status_color
FROM road_segments rs
LEFT JOIN LATERAL (
    SELECT * FROM traffic_records
    WHERE segment_id = rs.segment_id
    ORDER BY recorded_at DESC
    LIMIT 1
) tr ON true;

-- ─────────────────────────────────────────────────────────────
-- VIEW: Thống kê kẹt xe theo giờ (cho biểu đồ)
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW hourly_congestion_stats AS
SELECT
    hour,
    day_of_week,
    AVG(congestion_level)::NUMERIC(3,2)   AS avg_congestion,
    AVG(avg_speed_kmh)::NUMERIC(5,2)      AS avg_speed,
    COUNT(*)                               AS record_count
FROM traffic_records
GROUP BY hour, day_of_week
ORDER BY day_of_week, hour;

COMMENT ON TABLE traffic_records IS 'Dữ liệu lịch sử đo đạc giao thông từng đoạn đường TP.HCM';
COMMENT ON TABLE road_segments IS 'Các đoạn đường TP.HCM được phân đoạn để theo dõi';
COMMENT ON TABLE predictions IS 'Kết quả dự báo từ AI model';
