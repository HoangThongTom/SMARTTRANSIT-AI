"""
SmartTransit AI — AI Predictor
Load model đã train và thực hiện dự báo kẹt xe.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────────────────────────
# SCHEMA INPUT
# ─────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    segment_id: str = Field(..., example="Q1_NTH_001")
    hour: int = Field(..., ge=0, le=23, example=17)
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="0=Thứ 2, 6=Chủ nhật")
    month: int = Field(..., ge=1, le=12, example=6)
    is_holiday: bool = Field(False, example=False)
    weather_condition: str = Field("clear", example="rain", description="clear/rain/heavy_rain/fog")
    avg_speed_kmh: Optional[float] = Field(None, ge=0, le=120, example=25.5)
    vehicle_count: Optional[int] = Field(None, ge=0, example=450)
    lanes: int = Field(4, ge=1, le=10, example=4)


class PredictionOutput(BaseModel):
    segment_id: str
    congestion_level: int
    congestion_label: str
    congestion_color: str
    confidence: float
    probabilities: dict
    estimated_speed_kmh: float
    recommendation: str


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
WEATHER_MAP = {"clear": 0, "fog": 1, "rain": 2, "heavy_rain": 3}

CONGESTION_LABELS = {
    0: "Thông thoáng",
    1: "Chậm",
    2: "Kẹt nhẹ",
    3: "Kẹt nặng"
}

CONGESTION_COLORS = {
    0: "#27ae60",   # Xanh lá
    1: "#f39c12",   # Vàng
    2: "#e67e22",   # Cam
    3: "#c0392b"    # Đỏ
}

SPEED_ESTIMATES = {
    0: (45, 70),    # Thông thoáng
    1: (20, 45),    # Chậm
    2: (8, 20),     # Kẹt nhẹ
    3: (2, 8)       # Kẹt nặng
}

RECOMMENDATIONS = {
    0: "✅ Đường thông thoáng, có thể di chuyển bình thường.",
    1: "🟡 Giao thông chậm. Nên xuất phát sớm 10-15 phút.",
    2: "🟠 Kẹt xe nhẹ! Hãy cân nhắc tuyến đường thay thế.",
    3: "🔴 Kẹt xe nặng! Nên đổi tuyến đường hoặc chờ thêm 30-60 phút."
}


# ─────────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────────────────────
class TrafficPredictor:
    """Singleton để load model 1 lần dùng nhiều lần."""

    _instance = None
    _model = None
    _scaler = None
    _metrics = None
    _feature_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path: str, scaler_path: str, metrics_path: str = None):
        """Load model, scaler và metadata."""
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model không tìm thấy: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler không tìm thấy: {scaler_path}")

        self._model  = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)

        if metrics_path and Path(metrics_path).exists():
            with open(metrics_path) as f:
                self._metrics = json.load(f)
                self._feature_names = self._metrics.get("feature_names", [])

        logger.success(f"✅ Model loaded: {model_path.name}")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    def _build_feature_vector(self, inp: PredictionInput) -> np.ndarray:
        """Chuyển đổi input thành vector features khớp với training."""
        is_weekend   = int(inp.day_of_week >= 5)
        is_rush_hour = int(
            (6 <= inp.hour <= 8) or (16 <= inp.hour <= 19)
        )
        weather_enc  = WEATHER_MAP.get(inp.weather_condition, 0)

        # Giả sử speed nếu không có
        if inp.avg_speed_kmh is None:
            # Ước lượng dựa trên giờ
            hour_prob = self._estimate_congestion_prob(inp.hour, inp.day_of_week)
            if hour_prob < 0.3:
                inp.avg_speed_kmh = 55.0
            elif hour_prob < 0.6:
                inp.avg_speed_kmh = 30.0
            else:
                inp.avg_speed_kmh = 12.0

        vehicle_count = inp.vehicle_count or 300

        # Cyclic encoding
        hour_sin  = np.sin(2 * np.pi * inp.hour / 24)
        hour_cos  = np.cos(2 * np.pi * inp.hour / 24)
        dow_sin   = np.sin(2 * np.pi * inp.day_of_week / 7)
        dow_cos   = np.cos(2 * np.pi * inp.day_of_week / 7)
        month_sin = np.sin(2 * np.pi * inp.month / 12)
        month_cos = np.cos(2 * np.pi * inp.month / 12)

        features = [
            inp.hour, inp.day_of_week, inp.month,
            int(inp.is_holiday), is_weekend, is_rush_hour,
            weather_enc, inp.avg_speed_kmh, vehicle_count,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            month_sin, month_cos,
            inp.avg_speed_kmh,   # speed_lag1h (dùng current)
            inp.avg_speed_kmh,   # speed_lag24h
            inp.avg_speed_kmh,   # speed_rolling3h
            inp.lanes,
        ]

        # Pad thêm zeros cho district/road one-hot nếu model expect nhiều features hơn
        n_expected = self._model.n_features_in_
        while len(features) < n_expected:
            features.append(0)

        return np.array(features[:n_expected]).reshape(1, -1)

    def _estimate_congestion_prob(self, hour: int, dow: int) -> float:
        """Ước lượng xác suất kẹt theo giờ (backup khi không có speed)."""
        pattern = {
            6: 0.45, 7: 0.85, 8: 0.75, 9: 0.55,
            10: 0.35, 11: 0.30, 12: 0.45, 13: 0.40,
            14: 0.30, 15: 0.35, 16: 0.65, 17: 0.90,
            18: 0.85, 19: 0.65, 20: 0.45, 21: 0.35,
        }
        base = pattern.get(hour, 0.10)
        if dow >= 5:  # Weekend
            base *= 0.65
        return base

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        """Thực hiện dự báo."""
        if not self.is_loaded:
            raise RuntimeError("Model chưa được load. Gọi predictor.load() trước.")

        X = self._build_feature_vector(inp)
        X_scaled = self._scaler.transform(X)

        predicted_level = int(self._model.predict(X_scaled)[0])
        probabilities = self._model.predict_proba(X_scaled)[0]
        confidence = float(probabilities[predicted_level])

        # Ước tính tốc độ dựa vào level
        speed_min, speed_max = SPEED_ESTIMATES[predicted_level]
        estimated_speed = (speed_min + speed_max) / 2

        return PredictionOutput(
            segment_id=inp.segment_id,
            congestion_level=predicted_level,
            congestion_label=CONGESTION_LABELS[predicted_level],
            congestion_color=CONGESTION_COLORS[predicted_level],
            confidence=round(confidence, 3),
            probabilities={
                CONGESTION_LABELS[i]: round(float(p), 3)
                for i, p in enumerate(probabilities)
            },
            estimated_speed_kmh=round(estimated_speed, 1),
            recommendation=RECOMMENDATIONS[predicted_level],
        )

    @property
    def model_info(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "model_type": type(self._model).__name__ if self._model else None,
            "n_features": getattr(self._model, "n_features_in_", None),
            "metrics": self._metrics,
        }


# Global singleton
predictor = TrafficPredictor()
