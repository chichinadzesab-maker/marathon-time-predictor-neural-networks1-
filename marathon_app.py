"""
marathon_app.py — Flask backend for marathon finish time prediction.
Run: python marathon_app.py
Then open: http://localhost:5000
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load saved model ───────────────────────────────────────
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

# Reconstruct scalers
scaler_X_mean  = data['scaler_X_mean']
scaler_X_scale = data['scaler_X_scale']
scaler_y_mean  = data['scaler_y_mean']
scaler_y_scale = data['scaler_y_scale']

# FC-FFNN weights
ffnn_W      = data['ffnn_W']
ffnn_b      = data['ffnn_b']

# RBF weights
rbf_centers = data['rbf_centers']
rbf_W       = data['rbf_W']
rbf_width   = data['rbf_width']

FEATURES = [
    'age', 'running_experience_months', 'previous_marathon_count',
    'motivation_level', 'weekly_mileage_km', 'runs_per_week',
    'long_run_distance_km', 'speed_work_sessions_per_week',
    'rest_days_per_week', 'training_adherence_pct',
    'consecutive_weeks_no_miss', 'resting_heart_rate_bpm',
    'bmi', 'recovery_score', 'injury_count',
    'training_streak_days', 'missed_workout_pct',
    'goal_completion_rate', 'mental_preparation_score'
]

# ── Model inference ────────────────────────────────────────
def relu(x): return np.maximum(0, x)

def ffnn_predict(X):
    cur = X
    for i, (w, b) in enumerate(zip(ffnn_W, ffnn_b)):
        z = cur @ w + b
        cur = relu(z) if i < len(ffnn_W)-1 else z
    return cur

def rbf_predict(X):
    D   = np.linalg.norm(X[:,None] - rbf_centers[None], axis=2)
    Phi = np.exp(-D**2 / (2 * rbf_width**2))
    Phi_b = np.hstack([Phi, np.ones((len(Phi), 1))])
    return Phi_b @ rbf_W

def scale_input(x_raw):
    return (x_raw - scaler_X_mean) / scaler_X_scale

def unscale_output(y_s):
    return float((y_s * scaler_y_scale + scaler_y_mean).flatten()[0])

# ── Riegel formula for distance conversion ─────────────────
def riegel(marathon_minutes, target_km, marathon_km=42.195):
    return marathon_minutes * (target_km / marathon_km) ** 1.06

def format_time(minutes):
    minutes = max(1, float(minutes))
    h = int(minutes // 60)
    m = int(minutes % 60)
    s = int((minutes * 60) % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

# ── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form.get(f, 0)) for f in FEATURES]
        X_raw  = np.array(values).reshape(1, -1)
        X_s    = scale_input(X_raw)

        # FC-FFNN prediction
        ffnn_s   = ffnn_predict(X_s)
        ffnn_min = unscale_output(ffnn_s)

        # RBF prediction
        rbf_s    = rbf_predict(X_s)
        rbf_min  = unscale_output(rbf_s)

        # Average of both models
        avg_min  = (ffnn_min + rbf_min) / 2

        distances = [5, 8, 10, 15, 21.1, 42.195]
        dist_labels = ['5 km', '8 km', '10 km', '15 km', 'Half Marathon', 'Full Marathon']

        results = {
            'ffnn_marathon': format_time(ffnn_min),
            'rbf_marathon':  format_time(rbf_min),
            'distances': []
        }

        for km, label in zip(distances, dist_labels):
            t = riegel(avg_min, km)
            results['distances'].append({
                'label': label,
                'time':  format_time(t),
                'km':    km
            })

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Starting Marathon Predictor...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
