import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)
model = load('model.pkl')

# Ensure these are the same feature names used during training
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    form_features = [float(x) for x in request.form.values()]
    
    # Ensure features are in a DataFrame with the correct columns
    final_features = pd.DataFrame([form_features], columns=feature_names)
    
    # Predict using the preloaded model
    prediction = model.predict(final_features)
    
    # The model may return a 2D array, so we need to extract the first element
    output = prediction[0]
    
    if output == 1:
        prediction_text = "You have a high chance of having a cardiovascular disease."
    else:
        prediction_text = "You have a low chance of having a cardiovascular disease."
    
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
