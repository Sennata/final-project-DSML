import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_model.joblib')

print(f"Loading model from: {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Define standard feature column names required by the pipeline
FEATURE_COLS = [
    'Ukuran',
    'Kelayakan (Grade)',
    'Kondisi',
    'Tipe Kain',
    'Tipe Baju',
    'Merk (Merk/Non-Merk',
    'Polosan/Gambar (Y/N)',
    'Gender (M/F/U)'
]

# Model benchmark metadata from Final_Project_DSML.ipynb
MODEL_METRICS = {
    "model_name": "Random Forest Regressor (100 Estimators)",
    "mae": 13024.72,
    "rmse": 19038.40,
    "r2_score": 0.5652
}

# Categorical options extracted from the trained model
FEATURE_OPTIONS = {
    "ukuran": [
        {"value": "S", "label": "S (Small)"},
        {"value": "M", "label": "M (Medium)"},
        {"value": "L", "label": "L (Large)"},
        {"value": "XL", "label": "XL (Extra Large)"},
        {"value": "All Size", "label": "All Size (Semua Ukuran)"}
    ],
    "grade": [
        {"value": "A", "label": "Grade A (Kondisi Sangat Bagus / Like New)"},
        {"value": "B", "label": "Grade B (Kondisi Bagus / Pemakaian Wajar)"},
        {"value": "C", "label": "Grade C (Kondisi Cukup / Ada Minus)"}
    ],
    "kondisi": [
        "-",
        "Ada noda kecil",
        "Ada noda",
        "Ada lubang kecil + noda",
        "Warna sedikit pudar",
        "Warna pudar",
        "Warna pudar, leher molor",
        "Kancing beda 1",
        "Karet melar",
        "Kerah melar",
        "Kusut",
        "Melar dan ada noda",
        "Sablon mengelupas",
        "Gambar mulai hilang",
        "Gambar pudar",
        "kain kasar",
        "NEW"
    ],
    "tipe_kain": [
        "Katun",
        "Polyester",
        "Poliester",
        "Rayon",
        "Flanel",
        "Baby Terry",
        "Knit",
        "Linen",
        "Scuba",
        "American/Japan Drill"
    ],
    "tipe_baju": [
        {"value": "Kaos", "label": "Kaos (T-Shirt)"},
        {"value": "Kerah", "label": "Kerah / Polo / Kemeja"}
    ],
    "merk": [
        "Non-Merk",
        "Merk (Uniqlo)",
        "Merk (Giordano)",
        "Merk (The Excecutive)",
        "Merk (Avenue)",
        "Merk (Colorbox)",
        "Merk (Hush Puppies)",
        "Merk (Levistrauss)",
        "Merk (Aerostreet)",
        "Merk (Mareema)",
        "Merk (The Monseur)",
        "Merk 3Second",
        "Merk Adidas",
        "Merk Nike",
        "Merk Puma",
        "Merk Champion",
        "Merk Converse",
        "Merk Eiger",
        "Merk Nevada",
        "Merk Greenlight",
        "Merk Moutley",
        "Merk New Balance",
        "Merk Planet Surf",
        "Merk Batik Keris"
    ],
    "polosan": [
        {"value": "Polos", "label": "Polos"},
        {"value": "Gambar", "label": "Gambar / Sablon"},
        {"value": "Motif Garis-garis", "label": "Motif Garis-garis"},
        {"value": "Kotak-kotak", "label": "Kotak-kotak"},
        {"value": "Gambar (Batik)", "label": "Batik"},
        {"value": "Motif polkadot", "label": "Polkadot"}
    ],
    "gender": [
        {"value": "M", "label": "Pria (Men)"},
        {"value": "F", "label": "Wanita (Women)"},
        {"value": "U", "label": "Unisex"}
    ]
}

def format_rupiah(amount):
    """Format numeric value into Indonesian Rupiah format"""
    try:
        val = int(round(amount))
        return f"Rp {val:,}".replace(",", ".")
    except Exception:
        return f"Rp {amount}"

@app.route('/')
def home():
    return render_template('index.html', options=FEATURE_OPTIONS, metrics=MODEL_METRICS)

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({
        "status": "success",
        "options": FEATURE_OPTIONS,
        "metrics": MODEL_METRICS
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input JSON data provided"}), 400

        # Extract features with sensible defaults
        ukuran = data.get('ukuran', 'M').strip()
        grade = data.get('grade', 'B').strip()
        kondisi = data.get('kondisi', '-').strip()
        tipe_kain = data.get('tipe_kain', 'Katun').strip()
        tipe_baju = data.get('tipe_baju', 'Kaos').strip()
        merk = data.get('merk', 'Non-Merk').strip()
        polosan = data.get('polosan', 'Polos').strip()
        gender = data.get('gender', 'U').strip()

        # Build DataFrame with exact matching columns
        input_dict = {
            'Ukuran': ukuran,
            'Kelayakan (Grade)': grade,
            'Kondisi': kondisi,
            'Tipe Kain': tipe_kain,
            'Tipe Baju': tipe_baju,
            'Merk (Merk/Non-Merk': merk,
            'Polosan/Gambar (Y/N)': polosan,
            'Gender (M/F/U)': gender
        }

        df_input = pd.DataFrame([input_dict], columns=FEATURE_COLS)

        # Run prediction through the pipeline
        pred_value = float(model.predict(df_input)[0])

        # Floor prediction to at least 0 (or reasonable minimum for thrift clothes)
        pred_value = max(0.0, pred_value)

        # Calculate estimated range based on MAE
        mae = MODEL_METRICS["mae"]
        low_range = max(5000.0, pred_value - mae)
        high_range = pred_value + mae

        return jsonify({
            "status": "success",
            "prediction": round(pred_value, 2),
            "formatted_price": format_rupiah(pred_value),
            "range": {
                "min": round(low_range, 2),
                "max": round(high_range, 2),
                "formatted_min": format_rupiah(low_range),
                "formatted_max": format_rupiah(high_range)
            },
            "input_received": input_dict,
            "model_info": MODEL_METRICS
        })

    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Clothing Price Prediction Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
