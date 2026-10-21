import os
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
BASE = Path(__file__).resolve().parent.parent / "models"
FILES = {"random_forest": "random_forest.joblib", "xgboost": "xgboost.joblib"}
_cache = {}

def get_model(name):
    # Lazy load: only the requested model is read, then reused while the instance stays warm.
    if name not in _cache:
        _cache[name] = joblib.load(BASE / FILES[name])
    return _cache[name]

FEATURES = ["brand","model_name","variant_trim","vehicle_age_years","km_driven","fuel_type","transmission_type",
            "seller_type","owner_type","city","mileage_kmpl","engine_cc","max_power_bhp","seats","specs_imputed"]
NUMERIC = {"vehicle_age_years","km_driven","mileage_kmpl","engine_cc","max_power_bhp","seats","specs_imputed"}

@app.before_request
def check_key():
    key = os.environ.get("MODEL_API_KEY")
    if key and request.path != "/health" and request.headers.get("x-api-key") != key:
        return jsonify({"error": "Unauthorized."}), 401

@app.get("/")
def home():
    return jsonify({"message": "Indian Used Car Price Prediction API", "models": list(FILES)})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_available": list(FILES)})

@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        name = data.get("model", "random_forest")
        if name not in FILES:
            return jsonify({"error": "Invalid model.", "available_models": list(FILES)}), 400
        unknown = set(data) - set(FEATURES) - {"model"}
        if unknown:
            return jsonify({"error": "Unknown input fields.", "fields": sorted(unknown)}), 400
        row = {}
        for col in FEATURES:
            v = data.get(col)
            if v is None:
                row[col] = None
            elif col in NUMERIC:
                if isinstance(v, bool):
                    return jsonify({"error": f"{col} must be numeric."}), 400
                try:
                    n = float(v)
                except (TypeError, ValueError):
                    return jsonify({"error": f"{col} must be numeric."}), 400
                if not pd.notna(n) or abs(n) == float("inf"):
                    return jsonify({"error": f"{col} must be a finite number."}), 400
                row[col] = n
            elif isinstance(v, str):
                row[col] = v
            else:
                return jsonify({"error": f"{col} must be a string."}), 400
        price = float(get_model(name).predict(pd.DataFrame([row], columns=FEATURES))[0])
        return jsonify({"model": name, "predicted_price_inr": round(price, 2), "currency": "INR"})
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500
