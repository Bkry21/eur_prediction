"""
Smart Fracing System — Flask API
Matches smart_system.ipynb EXACTLY:
  - log1p transforms
  - DataFrame with named columns in training order
  - DATA_MEANS = exact df[col].mean() from EUR_dataset.csv (506 rows)
  - SLSQP: x0=DATA_MEANS + 8 random starts (seed=42, per-feature uniform), maxiter=1500, ftol=1e-9
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

# ── Exact feature order from x_train.columns ─────────────────────────────────
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

OPTIMIZED_ORIGINAL_FEATURES = [
    'Stage Spacing', 'Lateral Length', 'Injection Rate',
    'Percentage of LG', 'Proppant Loading',
]

FIXED_ORIGINAL_FEATURES = [
    'Well Spacing', 'Thickness', 'Porosity', 'ISIP',
    'Water Saturation', 'Pressure Gradient',
]

# ── EXACT data means from EUR_dataset.csv (506 rows) — matches DEFAULT_VALUES ─
DATA_MEANS = {
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

# ── Core transform: matches original_to_model_input() in notebook ─────────────
def original_to_model_input(params):
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
    return float(model.predict(original_to_model_input(params))[0])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/debug', methods=['GET'])
def debug():
    test_params = {
        'Stage Spacing': 200, 'Well Spacing': 1000, 'Thickness': 150,
        'Injection Rate': 65, 'Water Saturation': 20.0, 'Pressure Gradient': 0.85,
        'Proppant Loading': 2000, 'Lateral Length': 8000, 'ISIP': 6500,
        'Porosity': 7.0, 'Percentage of LG': 50.0
    }
    try:
        df_in = original_to_model_input(test_params)
        eur = float(model.predict(df_in)[0])
        return jsonify({
            'test_predicted_eur': round(eur, 6),
            'model_features_order': MODEL_FEATURES,
            'data_means': DATA_MEANS,
        })
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
        eur = predict_eur(data)
        return jsonify({'predicted_eur': round(eur, 6)})
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
            eur = predict_eur(row)
            result = {k: row[k] for k in ORIGINAL_INPUT_FEATURES}
            result['Predicted_EUR'] = round(eur, 6)
            results.append(result)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Matches notebook run_optimizer() exactly:
    SLSQP: x0 = DATA_MEANS, then 8 random starts (seed=42, per-feature uniform)
    DE:    strategy='best1bin', maxiter=300, popsize=20, tol=1e-3, polish=True
    """
    try:
        from scipy.optimize import minimize, differential_evolution

        data     = request.get_json(force=True)
        fixed    = data.get('fixed', {})
        bounds   = data.get('bounds', {})
        method   = data.get('method', 'SLSQP')

        opt_keys     = [k for k in bounds.keys() if k not in fixed]
        lo           = np.array([bounds[k][0] for k in opt_keys], dtype=float)
        hi           = np.array([bounds[k][1] for k in opt_keys], dtype=float)
        scipy_bounds = list(zip(lo.tolist(), hi.tolist()))

        def objective(x):
            params = dict(fixed)
            for i, k in enumerate(opt_keys):
                params[k] = float(x[i])
            return -predict_eur(params)

        if method == 'DE':
            # ── Exact notebook DE ─────────────────────────────────────────
            res = differential_evolution(
                objective,
                scipy_bounds,
                seed=42,
                strategy='best1bin',
                maxiter=300,
                popsize=20,
                tol=1e-3,
                atol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=True,
                workers=1,
            )

        else:
            # ── Exact notebook SLSQP ──────────────────────────────────────
            # x0 = DEFAULT_VALUES (exact data means) for optimized features
            x0 = np.array([DATA_MEANS[k] for k in opt_keys], dtype=float)
            x0 = np.clip(x0, lo, hi)

            # 8 random starts — matches notebook: per-feature uniform, seed=42
            rng = np.random.default_rng(42)
            starts = [x0]
            for _ in range(8):
                start = np.array([
                    rng.uniform(low=bounds[k][0], high=bounds[k][1])
                    for k in opt_keys
                ], dtype=float)
                starts.append(start)

            best_res, best_fun = None, np.inf
            for start in starts:
                try:
                    r = minimize(
                        objective, start,
                        method='SLSQP',
                        bounds=scipy_bounds,
                        options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False}
                    )
                    if r.fun is not None and not np.isnan(r.fun) and r.fun < best_fun:
                        best_fun = r.fun
                        best_res = r
                except Exception:
                    pass
            res = best_res

        if res is None:
            return jsonify({'error': 'Optimization failed'}), 500

        optimized = {k: float(res.x[i]) for i, k in enumerate(opt_keys)}
        best_eur  = predict_eur({**fixed, **optimized})

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
