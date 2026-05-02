"""
Smart Fracing System - Flask API
Run: python app.py
Open browser at: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Serve the HTML frontend (no CORS needed!) ─────────────────────────────────
@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'smart_fracing_system.html'))

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading ANN model...")
model = joblib.load(os.path.join(BASE_DIR, 'ANN_model.pkl'))

print("Loading Quantile Transformer...")
qt = joblib.load(os.path.join(BASE_DIR, 'quantile_transformer.pkl'))

print("Models loaded successfully!")


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_eur(params):
    porosity = float(params['Porosity'])
    pct_lg   = float(params['Percentage of LG'])

    qt_feats    = qt.transform([[porosity, pct_lg]])
    qt_porosity = qt_feats[0, 0]
    qt_pct_lg   = qt_feats[0, 1]

    X = np.array([[
        float(params['Stage Spacing']),
        float(params['Well Spacing']),
        float(params['Thickness']),
        float(params['Injection Rate']),
        float(params['Water Saturation']),
        float(params['Pressure Gradient']),
        np.log(float(params['Proppant Loading'])),
        np.log(float(params['Lateral Length'])),
        np.log(float(params['ISIP'])),
        qt_porosity,
        qt_pct_lg
    ]])

    eur = model.predict(X)[0]
    return float(eur)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required = ['Stage Spacing','Well Spacing','Thickness','Injection Rate',
                    'Water Saturation','Pressure Gradient','Proppant Loading',
                    'Lateral Length','ISIP','Porosity','Percentage of LG']
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f'Missing: {missing}'}), 400
        eur = predict_eur(data)
        return jsonify({'predicted_eur': round(eur, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True)
        rows = data.get('rows', [])
        results = []
        for row in rows:
            eur = predict_eur(row)
            results.append({**row, 'Predicted_EUR': round(eur, 4)})
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        from scipy.optimize import minimize, differential_evolution
        data    = request.get_json(force=True)
        fixed   = data.get('fixed', {})
        bounds  = data.get('bounds', {})
        method  = data.get('method', 'SLSQP')
        opt_keys = list(bounds.keys())
        lo = [bounds[k][0] for k in opt_keys]
        hi = [bounds[k][1] for k in opt_keys]

        def objective(x):
            params = {**fixed}
            for i, k in enumerate(opt_keys):
                params[k] = x[i]
            return -predict_eur(params)

        scipy_bounds = list(zip(lo, hi))
        if method == 'DE':
            res = differential_evolution(objective, scipy_bounds, seed=42, maxiter=300, workers=1)
        else:
            x0 = [(l+h)/2 for l,h in scipy_bounds]
            res = minimize(objective, x0, method='SLSQP', bounds=scipy_bounds)

        optimized = {k: round(float(res.x[i]), 4) for i, k in enumerate(opt_keys)}
        best_eur  = predict_eur({**fixed, **optimized})
        return jsonify({'success': bool(res.success), 'method': method,
                        'best_eur': round(best_eur, 4), 'optimized': optimized})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
import os
if __name__ == '__main__':
    print("\n>>> Open browser at: http://localhost:5000 <<<\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
