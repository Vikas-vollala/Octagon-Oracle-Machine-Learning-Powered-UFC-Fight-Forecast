# ⬡ Octagon Oracle — UFC Fight Predictor

> Machine Learning–powered UFC fight outcome forecasting with a Flask web interface.

---

## 📋 Overview

Octagon Oracle predicts UFC fight outcomes using historical fight data from the
[Rajeev Warrier UFC Dataset (Kaggle)](https://www.kaggle.com/datasets/rajeevw/ufcdata).

The system trains three models (Logistic Regression, Random Forest, XGBoost), evaluates
them with 5-fold cross-validation, and deploys the best-performing model via a Flask
web application with interactive visualizations.

**Typical accuracy: ~70–72% on held-out test data**

---

## 🗂️ Project Structure

```
octagon_oracle/
├── app.py                  ← Flask web application
├── train_model.py          ← Model training script
├── requirements.txt        ← Python dependencies
├── README.md
│
├── dataset/
│   └── preprocessed_data.csv   ← ⬅ Place Kaggle dataset here
│
├── model/                  ← Created after training
│   ├── ufc_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── model_info.txt
│
├── notebook/
│   └── ufc_prediction.ipynb    ← Full EDA + training notebook
│
├── templates/
│   ├── index.html          ← Home page
│   ├── predict.html        ← Fighter input form
│   ├── result.html         ← Prediction results
│   └── about.html          ← Project info
│
└── static/
    └── css/
        └── style.css       ← UFC dark theme
```

---

## ⚡ Quick Start

### 1. Get the Dataset
Download `preprocessed_data.csv` from Kaggle:
```
https://www.kaggle.com/datasets/rajeevw/ufcdata
```
Place it at: `dataset/preprocessed_data.csv`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Engineer 20+ differential features
- Train and evaluate 3 models with cross-validation
- Save the best model to `model/`

### 4. Run the Web App
```bash
python app.py
```
Open: **http://localhost:5000**

### 5. (Optional) Explore the Notebook
```bash
jupyter notebook notebook/ufc_prediction.ipynb
```

---

## 🤖 ML Pipeline

| Step | Details |
|------|---------|
| **Data** | 5,000+ UFC fights, filtered to Red/Blue outcomes |
| **Features** | 20+ differential stats (Red − Blue): strikes, takedowns, record, physical |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Evaluation** | 5-fold stratified CV + held-out test set |
| **Selection** | Best model by AUC score |
| **Serving** | Flask REST API + HTML/JS frontend |

---

## 📊 Key Features Engineered

- `win_ratio_diff` — career win percentage differential
- `str_acc_diff` — significant strike accuracy differential
- `td_acc_diff` — takedown accuracy differential
- `experience_diff` — total fights experience differential
- `reach_diff`, `height_diff`, `age_diff` — physical differentials
- `ko_diff`, `sub_diff`, `dec_diff` — finish method differentials
- `stance_match`, `R_is_southpaw`, `B_is_southpaw` — stance features
- `wc_*` — weight class one-hot encoding
- `title_bout` — championship fight flag

---

## 🌐 Web Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/predict` | GET | Fighter input form |
| `/predict` | POST | Submit prediction |
| `/about` | GET | Project info |
| `/api/predict` | POST | JSON API endpoint |

### JSON API Example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fighter_a_name": "Jon Jones",
    "fighter_b_name": "Stipe Miocic",
    "fighter_a": {"wins":20,"losses":1,"ko_wins":10,"sub_wins":3,"dec_wins":7,
                  "str_acc":55,"td_acc":42,"sub_att":0.8,"ctrl":150,
                  "height":193,"reach":215,"age":36,"stance":"Orthodox"},
    "fighter_b": {"wins":20,"losses":4,"ko_wins":12,"sub_wins":2,"dec_wins":6,
                  "str_acc":51,"td_acc":38,"sub_att":0.4,"ctrl":90,
                  "height":193,"reach":196,"age":41,"stance":"Orthodox"}
  }'
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. MMA outcomes are inherently
unpredictable. Do not use this for gambling or financial decisions.

---

## 📄 License

MIT — free to use, modify, and distribute.