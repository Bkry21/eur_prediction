"""
Smart Fracing System — Flask API
SLSQP matches Cell 29-31 in smart_system.ipynb EXACTLY:
  - Optimizer runs in TRANSFORMED space (log_Lateral, log_Proppant, QT_Percentage_of_LG)
  - x0 = X_all[completion_features].mean() in transformed space
  - bounds = X_all[completion_features].min/max() in transformed space
  - 1 + 12 random starts, seed=42, rng.uniform(lb, ub) — same as make_starts(n_random=12)
  - maxiter=2000, ftol=1e-9
  - Result is inverse-transformed back to original units for display
DE matches Cell 34:
  - strategy='best1bin', maxiter=800, popsize=25, init='sobol', polish=True
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

# Completion features in TRANSFORMED space — matches Cell 29 completion_features
COMPLETION_FEATURES_TRANSFORMED = [
    'Stage Spacing',         # unchanged
    'log_Lateral',           # log1p(Lateral Length)
    'Injection Rate',        # unchanged
    'QT_Percentage_of_LG',  # qt.transform col 1
    'log_Proppant',          # log1p(Proppant Loading)
]

# Fixed features in TRANSFORMED space — matches Cell 29 fixed_scenario_features
FIXED_FEATURES_TRANSFORMED = [
    'Well Spacing',      # unchanged
    'Thickness',         # unchanged
    'QT_Porosity',       # qt.transform col 0
    'log_ISIP',          # log1p(ISIP)
    'Water Saturation',  # unchanged
    'Pressure Gradient', # unchanged
]

# ── Transformed-space bounds from X_all[completion_features].min/max() ────────
# Computed from EUR_dataset.csv (506 rows) + transforms
TRANSFORMED_BOUNDS = {
    'Stage Spacing':        (140.0,      330.0),
    'log_Lateral':          (8.412055,   9.350189),   # log1p([4500, 11500])
    'Injection Rate':       (55.0,       80.0),
    'QT_Percentage_of_LG': None,   # computed at runtime from qt
    'log_Proppant':         (7.003974,   8.071219),   # log1p([1100, 3200])
}

# ── Transformed-space means from X_all[completion_features].mean() ────────────
# These are the exact x0 used by make_starts() in Cell 29
TRANSFORMED_MEANS = {
    'Stage Spacing':        147.640316,
    'log_Lateral':          8.999594,    # mean of log1p(Lateral Length)
    'Injection Rate':       63.079051,
    'QT_Percentage_of_LG': None,         # computed at runtime from qt
    'log_Proppant':         7.835841,    # mean of log1p(Proppant Loading)
}

def get_qt_pct_lg_bounds_and_mean():
    """Compute QT_Percentage_of_LG bounds/mean by transforming Pct LG data range."""
    # Transform the min and max of Percentage of LG through the qt
    pct_min = qt.transform([[7.337549, 15.0]])[0, 1]   # min Pct LG = 15
    pct_max = qt.transform([[7.337549, 95.0]])[0, 1]   # max Pct LG = 95
    pct_mean = qt.transform([[7.337549, 64.845455]])[0, 1]  # mean Pct LG
    return float(min(pct_min, pct_max)), float(max(pct_min, pct_max)), float(pct_mean)

# Precompute at startup
_qt_pct_lb, _qt_pct_ub, _qt_pct_mean = get_qt_pct_lg_bounds_and_mean()
TRANSFORMED_BOUNDS['QT_Percentage_of_LG'] = (_qt_pct_lb, _qt_pct_ub)
TRANSFORMED_MEANS['QT_Percentage_of_LG']  = _qt_pct_mean
print(f"QT_Percentage_of_LG bounds: [{_qt_pct_lb:.6f}, {_qt_pct_ub:.6f}], mean: {_qt_pct_mean:.6f}")

# ── Core transform: original → model input ────────────────────────────────────
def original_to_model_input(params):
    """Matches original_to_model_input() in notebook Cell 38."""
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

# ── Transformed-space predict (used by SLSQP/DE optimizer) ────────────────────
def make_input_row_transformed(x_transformed, fixed_transformed):
    """
    Matches make_input_row() in Cell 29 — builds model input from:
      x_transformed: array of completion feature values in TRANSFORMED space
      fixed_transformed: dict of fixed feature values in TRANSFORMED space
    """
    row = {}
    row.update(fixed_transformed)
    row.update({k: float(v) for k, v in zip(COMPLETION_FEATURES_TRANSFORMED, x_transformed)})
    return pd.DataFrame([row], columns=MODEL_FEATURES)

def predict_eur_transformed(x_transformed, fixed_transformed):
    """Matches predict_eur() in Cell 29."""
    df_row = make_input_row_transformed(x_transformed, fixed_transformed)
    return float(model.predict(df_row)[0])

def inverse_transform_result(x_opt, qt_porosity_val):
    """
    Inverse-transforms the optimizer result back to original units.
    Matches Cell 32/35 inverse transform logic exactly.
    qt_porosity_val: the actual QT_Porosity from fixed_transformed
    """
    result = {}
    qt_pct_lg_val = None
    for i, feat in enumerate(COMPLETION_FEATURES_TRANSFORMED):
        val = float(x_opt[i])
        if feat == 'log_Lateral':
            result['Lateral Length'] = np.expm1(val)
        elif feat == 'log_Proppant':
            result['Proppant Loading'] = np.expm1(val)
        elif feat == 'QT_Percentage_of_LG':
            qt_pct_lg_val = val
        else:
            result[feat] = val   # Stage Spacing, Injection Rate unchanged
    # Inverse transform QT features together — matches notebook Cell 32
    # qt.inverse_transform([[QT_Porosity, QT_Percentage_of_LG]])
    if qt_pct_lg_val is not None:
        inv = qt.inverse_transform([[qt_porosity_val, qt_pct_lg_val]])
        result['Percentage of LG'] = float(inv[0, 1])
    return result

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
        eur = predict_eur(test_params)
        return jsonify({
            'test_predicted_eur': round(eur, 6),
            'transformed_bounds': TRANSFORMED_BOUNDS,
            'transformed_means': TRANSFORMED_MEANS,
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
    Runs optimizer in TRANSFORMED space — matches Cell 29-31 (SLSQP) and Cell 34 (DE).

    Fixed features are transformed to model space:
      QT_Porosity = qt.transform([[Porosity, mean_PctLG]])[0,0]  — only Porosity matters
      log_ISIP    = log1p(ISIP)
      others      = unchanged

    Completion features are optimized in transformed space:
      Stage Spacing, Injection Rate — unchanged
      log_Lateral     bounds/means in log space
      log_Proppant    bounds/means in log space
      QT_Percentage_of_LG  bounds/means in QT space

    Result is inverse-transformed back to original units.
    """
    try:
        from scipy.optimize import minimize, differential_evolution

        data   = request.get_json(force=True)
        fixed  = data.get('fixed', {})   # original-unit fixed values from HTML
        method = data.get('method', 'SLSQP')

        # ── Build fixed_transformed dict (matches scenario_fixed_values in Cell 29) ──
        porosity_val = float(fixed['Porosity'])
        # For QT_Porosity we need both Porosity and Pct LG, but Pct LG is being optimized.
        # Use mean Pct LG as placeholder — same as X_all["QT_Porosity"].mean() logic
        qt_fixed = qt.transform([[porosity_val, 64.845455]])  # 64.845455 = mean Pct LG
        qt_porosity = float(qt_fixed[0, 0])

        fixed_transformed = {
            'Well Spacing':      float(fixed['Well Spacing']),
            'Thickness':         float(fixed['Thickness']),
            'QT_Porosity':       qt_porosity,
            'log_ISIP':          np.log1p(float(fixed['ISIP'])),
            'Water Saturation':  float(fixed['Water Saturation']),
            'Pressure Gradient': float(fixed['Pressure Gradient']),
        }

        # ── Transformed-space bounds (matches bounds_list in Cell 29) ─────────────
        lb = np.array([TRANSFORMED_BOUNDS[f][0] for f in COMPLETION_FEATURES_TRANSFORMED])
        ub = np.array([TRANSFORMED_BOUNDS[f][1] for f in COMPLETION_FEATURES_TRANSFORMED])
        scipy_bounds = list(zip(lb.tolist(), ub.tolist()))

        def objective_min(x):
            """Matches objective_min() in Cell 29."""
            return -predict_eur_transformed(x, fixed_transformed)

        if method == 'DE':
            # ── Exact Cell 34 DE settings ─────────────────────────────────────────
            res = differential_evolution(
                objective_min,
                scipy_bounds,
                strategy='best1bin',
                maxiter=800,
                popsize=25,
                tol=1e-3,
                atol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                init='sobol',
                polish=True,
                workers=1,
            )

        else:
            # ── Exact Cell 31 SLSQP settings ──────────────────────────────────────
            # x0 = X_all[completion_features].mean() in TRANSFORMED space
            x0 = np.array([TRANSFORMED_MEANS[f] for f in COMPLETION_FEATURES_TRANSFORMED])

            # make_starts(n_random=12, seed=42): rng.uniform(lb, ub) per-feature
            rng = np.random.default_rng(42)
            starts = [x0]
            for _ in range(12):
                starts.append(rng.uniform(lb, ub))

            best_res, best_fun = None, np.inf
            for start in starts:
                try:
                    r = minimize(
                        objective_min, start,
                        method='SLSQP',
                        bounds=scipy_bounds,
                        options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
                    )
                    if r.fun is not None and not np.isnan(r.fun) and r.fun < best_fun:
                        best_fun = r.fun
                        best_res = r
                except Exception:
                    pass
            res = best_res

        if res is None:
            return jsonify({'error': 'Optimization failed'}), 500

        # ── Inverse-transform result to original units ────────────────────────────
        optimized_original = inverse_transform_result(res.x, fixed_transformed['QT_Porosity'])
        best_eur = -float(res.fun)

        return jsonify({
            'success':   bool(res.success),
            'method':    method,
            'best_eur':  round(best_eur, 6),
            'optimized': {k: round(float(v), 4) for k, v in optimized_original.items()},
            'n_evals':   int(res.nfev) if hasattr(res, 'nfev') else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
