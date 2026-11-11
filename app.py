"""
Octagon Oracle — Flask Web Application
Run: python app.py  (after running train_model.py)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ──────────────────────────────────────────────
# Load model artifacts
# ──────────────────────────────────────────────
MODEL_PATH    = os.path.join('model', 'ufc_model.pkl')
SCALER_PATH   = os.path.join('model', 'scaler.pkl')
FEATURES_PATH = os.path.join('model', 'feature_cols.pkl')

model        = None
scaler       = None
feature_cols = None

def load_artifacts():
    global model, scaler, feature_cols
    if os.path.exists(MODEL_PATH):
        model        = joblib.load(MODEL_PATH)
        scaler       = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        print("[OK] Model loaded successfully.")
    else:
        print("[WARNING] Model not found. Run train_model.py first.")

load_artifacts()

# ──────────────────────────────────────────────
# Helper: build feature vector from form inputs
# ──────────────────────────────────────────────
def build_feature_vector(a, b):
    """
    a, b are dicts with keys:
      wins, losses, ko_wins, sub_wins, dec_wins,
      str_acc, td_acc, sub_att, ctrl,
      height, reach, age, stance
    """
    def safe(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    a_total = safe(a['wins']) + safe(a['losses'])
    b_total = safe(b['wins']) + safe(b['losses'])
    a_ratio = safe(a['wins']) / a_total if a_total > 0 else 0.5
    b_ratio = safe(b['wins']) / b_total if b_total > 0 else 0.5

    vec = {
        'str_acc_diff':        safe(a['str_acc'])   - safe(b['str_acc']),
        'td_acc_diff':         safe(a['td_acc'])     - safe(b['td_acc']),
        'sub_att_diff':        safe(a['sub_att'])    - safe(b['sub_att']),
        'ctrl_diff':           safe(a['ctrl'])       - safe(b['ctrl']),
        'sig_str_att_diff':    0.0,
        'sig_str_landed_diff': 0.0,
        'td_att_diff':         0.0,
        'td_landed_diff':      0.0,
        'win_diff':            safe(a['wins'])       - safe(b['wins']),
        'loss_diff':           safe(a['losses'])     - safe(b['losses']),
        'ko_diff':             safe(a['ko_wins'])    - safe(b['ko_wins']),
        'sub_diff':            safe(a['sub_wins'])   - safe(b['sub_wins']),
        'dec_diff':            safe(a['dec_wins'])   - safe(b['dec_wins']),
        'height_diff':         safe(a['height'])     - safe(b['height']),
        'reach_diff':          safe(a['reach'])      - safe(b['reach']),
        'age_diff':            safe(a['age'])        - safe(b['age']),
        'opp_str_diff':        0.0,
        'win_ratio_diff':      a_ratio - b_ratio,
        'experience_diff':     (safe(a['wins']) + safe(a['losses'])) -
                               (safe(b['wins']) + safe(b['losses'])),
        'stance_match':        1.0 if a['stance'] == b['stance'] else 0.0,
        'R_is_southpaw':       1.0 if a['stance'] == 'Southpaw' else 0.0,
        'B_is_southpaw':       1.0 if b['stance'] == 'Southpaw' else 0.0,
    }
    return vec


def predict_winner(fighter_a, fighter_b, fighter_a_name, fighter_b_name):
    """Returns dict with prediction details."""
    if model is None:
        return {'error': 'Model not loaded. Run train_model.py first.'}

    raw_vec = build_feature_vector(fighter_a, fighter_b)

    row = np.array([[raw_vec.get(f, 0.0) for f in feature_cols]])

    model_name = type(model).__name__
    if 'Logistic' in model_name:
        row = scaler.transform(row)

    proba = model.predict_proba(row)[0]   # [P(Blue), P(Red)]
    pred  = np.argmax(proba)              # 0=Blue, 1=Red

    winner_name = fighter_a_name if pred == 1 else fighter_b_name
    winner_prob = float(proba[pred]) * 100

    return {
        'winner':        winner_name,
        'winner_corner': 'Red' if pred == 1 else 'Blue',
        'red_prob':      round(float(proba[1]) * 100, 1),
        'blue_prob':     round(float(proba[0]) * 100, 1),
        'confidence':    round(winner_prob, 1),
        'fighter_a':     fighter_a_name,
        'fighter_b':     fighter_b_name,
        'stats_a':       fighter_a,
        'stats_b':       fighter_b,
    }

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    def get_fighter(prefix):
        return {
            'wins':     request.form.get(f'{prefix}_wins',     0),
            'losses':   request.form.get(f'{prefix}_losses',   0),
            'ko_wins':  request.form.get(f'{prefix}_ko_wins',  0),
            'sub_wins': request.form.get(f'{prefix}_sub_wins', 0),
            'dec_wins': request.form.get(f'{prefix}_dec_wins', 0),
            'str_acc':  request.form.get(f'{prefix}_str_acc',  0),
            'td_acc':   request.form.get(f'{prefix}_td_acc',   0),
            'sub_att':  request.form.get(f'{prefix}_sub_att',  0),
            'ctrl':     request.form.get(f'{prefix}_ctrl',     0),
            'height':   request.form.get(f'{prefix}_height',   170),
            'reach':    request.form.get(f'{prefix}_reach',    170),
            'age':      request.form.get(f'{prefix}_age',      28),
            'stance':   request.form.get(f'{prefix}_stance',   'Orthodox'),
        }

    fighter_a      = get_fighter('a')
    fighter_b      = get_fighter('b')
    fighter_a_name = request.form.get('fighter_a_name', 'Fighter A').strip() or 'Fighter A'
    fighter_b_name = request.form.get('fighter_b_name', 'Fighter B').strip() or 'Fighter B'

    result = predict_winner(fighter_a, fighter_b, fighter_a_name, fighter_b_name)

    if 'error' in result:
        return render_template('predict.html', error=result['error'])

    return render_template('result.html', result=result)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint."""
    data           = request.get_json(force=True)
    fighter_a      = data.get('fighter_a', {})
    fighter_b      = data.get('fighter_b', {})
    fighter_a_name = data.get('fighter_a_name', 'Fighter A')
    fighter_b_name = data.get('fighter_b_name', 'Fighter B')
    result = predict_winner(fighter_a, fighter_b, fighter_a_name, fighter_b_name)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)