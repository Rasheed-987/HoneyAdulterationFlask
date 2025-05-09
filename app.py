import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins (or use origins=["http://localhost:5173"])

# Load the ANN model
MODEL_PATH = 'ANN.h5'  # Update this path to where your model is stored
model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            input_data = df.values  # Ensure that the input is in the right shape for the model
                   
            predictions = model.predict(input_data)
            
            return jsonify({'predictions': predictions.tolist()}), 200
        
        except Exception as e:
            return jsonify({'error': f'Error processing the CSV file: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

        return jsonify({'predictions': predictions.tolist()}), 200
        



if __name__ == '__main__':
    app.run(debug=True)
