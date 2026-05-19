"""
SmartTransit AI — Model Training Pipeline
Train, evaluate và lưu model dự báo kẹt xe.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TRAIN_PATH   = "ai_core/models/train_data.parquet"
TEST_PATH    = "ai_core/models/test_data.parquet"
MODEL_PATH   = "ai_core/models/traffic_model.joblib"
SCALER_PATH  = "ai_core/models/scaler.joblib"
METRICS_PATH = "ai_core/models/metrics.json"

TARGET_COL = "congestion_level"

FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "is_holiday", "is_weekend", "is_rush_hour",
    "weather_encoded", "avg_speed_kmh", "vehicle_count",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "speed_lag1h", "speed_lag24h",
    "speed_rolling3h", "lanes",
]

CONGESTION_LABELS = {
    0: "Thông thoáng",
    1: "Chậm",
    2: "Kẹt nhẹ",
    3: "Kẹt nặng"
}


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: LOAD PROCESSED DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    logger.info("📥 Load dữ liệu đã xử lý...")
    train = pd.read_parquet(TRAIN_PATH)
    test  = pd.read_parquet(TEST_PATH)

    # Lấy các feature columns tồn tại trong dataframe
    available_features = [f for f in FEATURE_COLS if f in train.columns]
    # Thêm các district/road one-hot columns nếu có
    extra_cols = [c for c in train.columns if c.startswith("dist_") or c.startswith("road_")]
    all_features = available_features + extra_cols

    X_train = train[all_features]
    y_train = train[TARGET_COL]
    X_test  = test[all_features]
    y_test  = test[TARGET_COL]

    logger.success(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"  Features: {len(all_features)} cột")
    return X_train, y_train, X_test, y_test, all_features


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: SCALING
# ─────────────────────────────────────────────────────────────
def scale_features(X_train, X_test):
    logger.info("⚖️  Chuẩn hóa features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    logger.success(f"  ✅ Scaler saved: {SCALER_PATH}")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────
# BƯỚC 3: TRAIN RANDOM FOREST (model chính)
# ─────────────────────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    logger.info("🌳 Train Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    logger.success("  ✅ Random Forest trained!")
    return rf


# ─────────────────────────────────────────────────────────────
# BƯỚC 4: HYPERPARAMETER TUNING (optional, tốn thời gian)
# ─────────────────────────────────────────────────────────────
def tune_model(X_train, y_train, quick=True):
    """
    quick=True: chỉ thử vài tham số để nhanh
    quick=False: grid search đầy đủ
    """
    logger.info("🔧 Hyperparameter Tuning...")

    if quick:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [15, 20, None],
            "min_samples_leaf": [1, 2],
        }
    else:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="f1_weighted",
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    logger.success(f"  Best params: {grid_search.best_params_}")
    logger.success(f"  Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names):
    logger.info("📊 Đánh giá model trên tập Test...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    logger.info(f"\n{'='*50}")
    logger.info(f"  Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  F1 (weighted)  : {f1_w:.4f}")
    logger.info(f"  F1 (macro)     : {f1_macro:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=list(CONGESTION_LABELS.values()))}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        logger.info("  Top 10 features quan trọng nhất:")
        for name, imp in feat_imp[:10]:
            bar = "█" * int(imp * 100)
            logger.info(f"    {name:<25} {bar} ({imp:.4f})")

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "trained_at": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
    }


# ─────────────────────────────────────────────────────────────
# BƯỚC 6: CROSS VALIDATION
# ─────────────────────────────────────────────────────────────
def cross_validate(model, X_train, y_train):
    logger.info("🔄 Cross Validation (5-fold)...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)
    logger.success(f"  CV Scores: {scores}")
    logger.success(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


# ─────────────────────────────────────────────────────────────
# BƯỚC 7: SAVE MODEL
# ─────────────────────────────────────────────────────────────
def save_model(model, metrics: dict):
    os.makedirs("ai_core/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.success(f"""
╔══════════════════════════════════════════╗
║     🎉 MODEL TRAINING HOÀN THÀNH!     ║
╠══════════════════════════════════════════╣
║  Model    : {MODEL_PATH:<28} ║
║  Accuracy : {metrics['accuracy']*100:.2f}%                        ║
║  F1 Score : {metrics['f1_weighted']:.4f}                       ║
╚══════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run(tune=False):
    logger.info("=" * 55)
    logger.info("  SmartTransit AI — Training Pipeline")
    logger.info("=" * 55)

    # Load
    X_train, y_train, X_test, y_test, feature_names = load_data()

    # Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # Train
    if tune:
        model = tune_model(X_train_s, y_train, quick=True)
    else:
        model = train_random_forest(X_train_s, y_train)

    # Cross-validate
    cv_mean, cv_std = cross_validate(model, X_train_s, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test_s, y_test, feature_names)
    metrics["cv_mean"] = round(float(cv_mean), 4)
    metrics["cv_std"]  = round(float(cv_std), 4)

    # Save
    save_model(model, metrics)

    return model, metrics


if __name__ == "__main__":
    run(tune=False)
