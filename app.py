"""
Smart Fracing System — Flask API v_FINAL

/optimize matches Cell 41 (Widget) in smart_system.ipynb EXACTLY:
  - Runs in ORIGINAL units space (NOT transformed)
  - OPTIMIZED_ORIGINAL_FEATURES = ['Stage Spacing','Lateral Length','Injection Rate','Percentage of LG','Proppant Loading']
  - OPT_BOUNDS = FEATURE_BOUNDS from data min/max in original units
  - x0 = DEFAULT_VALUES = df[col].mean() in original units
  - SLSQP: 1 + 8 random starts, rng.uniform per-feature, seed=42, maxiter=1500, ftol=1e-9
  - DE: strategy='best1bin', maxiter=300, popsize=20, tol=1e-3, polish=True, seed=42
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import warnings

app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('smart_fracing_system.html')

print("Loading ANN model...")
model = joblib.load(os.path.join(BASE_DIR, 'ANN_model.pkl'))
print("Loading Quantile Transformer...")
qt = joblib.load(os.path.join(BASE_DIR, 'quantile_transformer.pkl'))
print("Models loaded!")

# ── Feature order from x_train.columns ───────────────────────────────────────
MODEL_FEATURES = [
    'Stage Spacing', 'Well Spacing', 'Thickness', 'Injection Rate',
    'Water Saturation', 'Pressure Gradient',
    'log_Proppant', 'log_Lateral', 'log_ISIP',
    'QT_Porosity', 'QT_Percentage_of_LG'
]

ORIGINAL_INPUT_FEATURES = [
    'Stage Spacing', 'Well Spacing', 'Thickness', 'Injection Rate',
    'Water Saturation', 'Pressure Gradient', 'Proppant Loading',
    'Lateral Length', 'ISIP', 'Porosity', 'Percentage of LG',
]

# Matches OPTIMIZED_ORIGINAL_FEATURES in Cell 38
OPTIMIZED_ORIGINAL_FEATURES = [
    'Stage Spacing', 'Lateral Length', 'Injection Rate',
    'Percentage of LG', 'Proppant Loading',
]

# Matches FIXED_ORIGINAL_FEATURES in Cell 38
FIXED_ORIGINAL_FEATURES = [
    'Well Spacing', 'Thickness', 'Porosity', 'ISIP',
    'Water Saturation', 'Pressure Gradient',
]

# ── DEFAULT_VALUES = df[col].mean() — exact from EUR_dataset.csv (506 rows) ──
# Matches DEFAULT_VALUES in Cell 38
DEFAULT_VALUES = {
    'Stage Spacing':     147.640316,
    'Well Spacing':      820.158103,
    'Thickness':         162.365613,
    'Injection Rate':    63.079051,
    'Water Saturation':  19.213439,
    'Pressure Gradient': 0.930257,
    'Proppant Loading':  2567.065217,
    'Lateral Length':    8153.086957,
    'ISIP':              7010.490119,
    'Porosity':          7.337549,
    'Percentage of LG':  64.845455,
}

# ── OPT_BOUNDS = FEATURE_BOUNDS from data min/max in original units ───────────
# Matches OPT_BOUNDS = [FEATURE_BOUNDS[col] for col in OPTIMIZED_ORIGINAL_FEATURES]
OPT_BOUNDS = {
    'Stage Spacing':     (140.0,   330.0),
    'Lateral Length':    (4500.0,  11500.0),
    'Injection Rate':    (55.0,    80.0),
    'Percentage of LG':  (15.0,    95.0),
    'Proppant Loading':  (1100.0,  3200.0),
}

# ── Core transform: matches original_to_model_input() in Cell 38 ─────────────
def original_to_model_input(params):
    """Convert original-unit dict to model input DataFrame."""
    porosity = float(params['Porosity'])
    pct_lg   = float(params['Percentage of LG'])
    qt_vals  = qt.transform([[porosity, pct_lg]])
    row = {
        'Stage Spacing':        float(params['Stage Spacing']),
        'Well Spacing':         float(params['Well Spacing']),
        'Thickness':            float(params['Thickness']),
        'Injection Rate':       float(params['Injection Rate']),
        'Water Saturation':     float(params['Water Saturation']),
        'Pressure Gradient':    float(params['Pressure Gradient']),
        'log_Proppant':         np.log1p(float(params['Proppant Loading'])),
        'log_Lateral':          np.log1p(float(params['Lateral Length'])),
        'log_ISIP':             np.log1p(float(params['ISIP'])),
        'QT_Porosity':          float(qt_vals[0, 0]),
        'QT_Percentage_of_LG':  float(qt_vals[0, 1]),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)

def predict_eur(params):
    """Matches predict_eur_from_original() in Cell 38."""
    return float(model.predict(original_to_model_input(params))[0])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/debug', methods=['GET'])
def debug():
    test = {
        'Stage Spacing': 200, 'Well Spacing': 1000, 'Thickness': 150,
        'Injection Rate': 65, 'Water Saturation': 20.0, 'Pressure Gradient': 0.85,
        'Proppant Loading': 2000, 'Lateral Length': 8000, 'ISIP': 6500,
        'Porosity': 7.0, 'Percentage of LG': 50.0
    }
    try:
        return jsonify({'test_predicted_eur': round(predict_eur(test), 6)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        missing = [k for k in ORIGINAL_INPUT_FEATURES if k not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400
        return jsonify({'predicted_eur': round(predict_eur(data), 6)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True)
        rows = data.get('rows', [])
        if not rows:
            return jsonify({'error': 'No rows provided'}), 400
        results = []
        for row in rows:
            missing = [k for k in ORIGINAL_INPUT_FEATURES if k not in row]
            if missing:
                return jsonify({'error': f'Missing fields in row: {missing}'}), 400
            result = {k: row[k] for k in ORIGINAL_INPUT_FEATURES}
            result['Predicted_EUR'] = round(predict_eur(row), 6)
            results.append(result)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Matches Cell 41 (Widget) run_optimizer() EXACTLY.
    Runs in ORIGINAL units space.

    SLSQP:
      x0 = DEFAULT_VALUES for OPTIMIZED_ORIGINAL_FEATURES
      starts = [x0] + 8 random (rng.uniform per-feature, seed=42)
      maxiter=1500, ftol=1e-9

    DE:
      strategy='best1bin', maxiter=300, popsize=20,
      tol=1e-3, atol=1e-6, mutation=(0.5,1.0), recombination=0.7,
      seed=42, polish=True
    """
    try:
        from scipy.optimize import minimize, differential_evolution

        data   = request.get_json(force=True)
        # Cast every incoming fixed value to float to avoid JSON string/int mismatches
        fixed  = {k: float(v) for k, v in data.get('fixed', {}).items()}
        # Fill any missing fixed features with DEFAULT_VALUES (exact df[col].mean())
        for f in FIXED_ORIGINAL_FEATURES:
            if f not in fixed:
                fixed[f] = DEFAULT_VALUES[f]
        method = data.get('method', 'SLSQP')

        # OPT_BOUNDS as list of tuples — matches [FEATURE_BOUNDS[col] for col in OPTIMIZED_ORIGINAL_FEATURES]
        opt_bounds_list = [OPT_BOUNDS[f] for f in OPTIMIZED_ORIGINAL_FEATURES]
        lo = np.array([b[0] for b in opt_bounds_list])
        hi = np.array([b[1] for b in opt_bounds_list])

        def objective_original(x_original_opt):
            """Matches objective_original() in Cell 41."""
            params = dict(fixed)
            params.update({k: float(v) for k, v in zip(OPTIMIZED_ORIGINAL_FEATURES, x_original_opt)})
            return -predict_eur(params)

        if method == 'DE':
            # ── Exact Cell 41 DE ──────────────────────────────────────────────
            res = differential_evolution(
                objective_original,
                opt_bounds_list,
                strategy='best1bin',
                maxiter=300,
                popsize=20,
                tol=1e-3,
                atol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                workers=1,
            )

        else:
            # ── Exact Cell 41 SLSQP ───────────────────────────────────────────
            # x0 = DEFAULT_VALUES for OPTIMIZED_ORIGINAL_FEATURES
            x0 = np.array([DEFAULT_VALUES[f] for f in OPTIMIZED_ORIGINAL_FEATURES], dtype=float)

            # starts = [x0] + 8 random — rng.uniform per-feature, seed=42
            rng = np.random.default_rng(42)
            starts = [x0]
            for _ in range(8):
                starts.append(np.array([
                    rng.uniform(low=OPT_BOUNDS[f][0], high=OPT_BOUNDS[f][1])
                    for f in OPTIMIZED_ORIGINAL_FEATURES
                ], dtype=float))

            best_res, best_fun = None, np.inf
            for start in starts:
                res = minimize(
                    objective_original,
                    x0=start,
                    method='SLSQP',
                    bounds=opt_bounds_list,
                    options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False}
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            res = best_res

        if res is None:
            return jsonify({'error': 'Optimization failed'}), 500

        # Build optimized dict in original units
        optimized = {
            k: float(res.x[i])
            for i, k in enumerate(OPTIMIZED_ORIGINAL_FEATURES)
        }

        # Final EUR prediction with exact optimized values
        best_eur = predict_eur({**fixed, **optimized})

        return jsonify({
            'success':   bool(res.success),
            'method':    method,
            'best_eur':  round(best_eur, 6),
            'optimized': {k: round(v, 4) for k, v in optimized.items()},
            'n_evals':   int(res.nfev) if hasattr(res, 'nfev') else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
