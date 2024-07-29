from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Define the feature names
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    form_features = [float(x) for x in request.form.values()]
    print(f"Form features: {form_features}")
    
    # Ensure features are in a DataFrame with the correct columns
    final_features = pd.DataFrame([form_features], columns=feature_names)
    print(f"Final features DataFrame: {final_features}")
    
    # Preprocess the features
    final_features = preprocessor.transform(final_features)
    print(f"Preprocessed features: {final_features}")
    
    # Reshape the data to fit the RNN model
    final_features = np.expand_dims(final_features, axis=2)
    print(f"Final features for RNN: {final_features.shape}")
    
    # Predict using the preloaded model
    prediction = model.predict(final_features)
    print(f"Raw prediction: {prediction}")
    
    # The model may return a 2D array, so we need to extract the first element
    output = np.argmax(prediction, axis=1)[0]
    print(f"Prediction output: {output}")
    
    if output == 1:
        prediction_text = "You have a high chance of having a cardiovascular disease."
    else:
        prediction_text = "You have a low chance of having a cardiovascular disease."
    
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
