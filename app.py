"""
Smart Fracing System - Flask API (Production Ready)
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
import warnings
app = Flask(__name__)
CORS(app)

# تجاهل التحذيرات
warnings.filterwarnings('ignore')

# الحصول على المسار الأساسي للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Serve the HTML frontend ──────────────────────────────────────────────────
@app.route('/')
def index():
    # تم تغيير send_file إلى render_template لأنها الطريقة الأفضل في Flask
    # تأكد أن ملف الـ HTML موجود داخل مجلد اسمه templates
    return render_template('smart_fracing_system.html')

# ── Load models once at startup ───────────────────────────────────────────────
# استخدام os.path.join لضمان عمل المسارات على السيرفر بشكل صحيح
print("Loading ANN model...")
model_path = os.path.join(BASE_DIR, 'ANN_model.pkl')
model = joblib.load(model_path)

print("Loading Quantile Transformer...")
qt_path = os.path.join(BASE_DIR, 'quantile_transformer.pkl')
qt = joblib.load(qt_path)

print("Models loaded successfully!")


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_eur(params):
    porosity = float(params['Porosity'])
    pct_lg   = float(params['Percentage of LG'])

    # التحويل باستخدام Quantile Transformer
    qt_feats    = qt.transform([[porosity, pct_lg]])
    qt_porosity = qt_feats[0, 0]
    qt_pct_lg   = qt_feats[0, 1]

    # تجهيز المصفوفة للتنبؤ
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


@app.route('/static/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)


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
        if not rows:
            return jsonify({'error': 'No rows provided'}), 400

        required = ['Stage Spacing','Well Spacing','Thickness','Injection Rate',
                    'Water Saturation','Pressure Gradient','Proppant Loading',
                    'Lateral Length','ISIP','Porosity','Percentage of LG']

        results = []
        for row in rows:
            missing = [k for k in required if k not in row]
            if missing:
                return jsonify({'error': f'Missing fields in row: {missing}'}), 400
            eur = predict_eur(row)
            result = {k: row[k] for k in required}
            result['Predicted_EUR'] = round(eur, 4)
            results.append(result)

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # استيراد scipy هنا لضمان عملها داخل السيرفر
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
            res = differential_evolution(
                objective, scipy_bounds, seed=42, maxiter=150,
                popsize=12, tol=1e-6, workers=1, mutation=(0.5, 1.2), recombination=0.8
            )
        else:
            # Try multiple starting points and keep the best result
            best_res = None
            starts = [
                [(l+h)/2 for l,h in scipy_bounds],
                [l + (h-l)*0.25 for l,h in scipy_bounds],
                [l + (h-l)*0.75 for l,h in scipy_bounds],
            ]
            for x0 in starts:
                try:
                    r = minimize(objective, x0, method='SLSQP', bounds=scipy_bounds,
                                 options={'ftol': 1e-9, 'maxiter': 300})
                    if best_res is None or r.fun < best_res.fun:
                        best_res = r
                except Exception:
                    pass
            res = best_res

        optimized = {k: round(float(res.x[i]), 4) for i, k in enumerate(opt_keys)}
        best_eur  = predict_eur({**fixed, **optimized})
        
        return jsonify({
            'success': bool(res.success), 
            'method': method,
            'best_eur': round(best_eur, 4), 
            'optimized': optimized
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run (Production Config) ──────────────────────────────────────────────────
if __name__ == '__main__':
    # قراءة البورت من السيرفر (ضروري جداً لـ Render)
    port = int(os.environ.get("PORT", 5000))
    # تشغيل التطبيق على 0.0.0.0 ليكون متاحاً خارجياً
    app.run(host='0.0.0.0', port=port, debug=False)
