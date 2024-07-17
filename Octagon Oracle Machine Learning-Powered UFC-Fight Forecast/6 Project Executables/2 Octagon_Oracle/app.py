from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

model = joblib.load('model (1).pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect data from the form
        takedown_attempts = float(request.form['stat1'])
        strikes_landed = float(request.form['stat2'])
        strikes_absorbed = float(request.form['stat3'])
        control_time = float(request.form['stat4'])
        submission_attempts = float(request.form['stat5'])

        # Cap takedown_attempts at 15 if greater than 20
        if takedown_attempts > 20:
            takedown_attempts = 15

        # Here you would add your machine learning model prediction logic
        # For simplicity, we are using a placeholder logic
        prediction = "Blue" if (strikes_landed > strikes_absorbed) else "Red"
        if 15 <= takedown_attempts <= 20:  # Condition for the range 15 to 20
            prediction = "Draw"

        return render_template('result.html', prediction=prediction)

    return render_template('predict.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
