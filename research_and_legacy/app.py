import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
app = Flask(__name__)

# --- CHANGE 1: Load the TUNED model ONCE (Global Scope) ---
# We load it here so it stays in memory. We don't reload it inside the function.
try:
    model = pickle.load(open('model_tuned.pkl', 'rb'))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model_tuned.pkl' not found. Make sure you exported the tuned model.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # 1. Extract data by NAME to ensure correct order
            # The order inside this list MUST match your X_train columns exactly
            feature_vector = [
                float(request.form['age']),
                float(request.form['w_class']),       # Model expects Workclass 2nd
                float(request.form['edu']),
                float(request.form['martial_stat']),
                float(request.form['occup']),
                float(request.form['relation']),
                float(request.form['race']),
                float(request.form['sex']),
                float(request.form['c_gain']),        # Capital Gain is 9th in standard Adult dataset
                float(request.form['c_loss']),
                float(request.form['hours_per_week']),
                float(request.form['country'])
            ]

            # 2. Reshape
            to_predict = np.array(feature_vector).reshape(1, 12)

            # 3. Predict
            prediction = model.predict(to_predict)

            # 4. Result Logic
            if int(prediction[0]) == 1:
                result_text = 'Prediction: Income is >50K'
            else:
                result_text = 'Prediction: Income is <=50K'

            return render_template("index.html", prediction_text=result_text)

        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)