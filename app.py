from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib
from google.cloud import storage
import os
from io import BytesIO

app = Flask(__name__)
bucket_name = 'nutricare'
model_filename = 'Klasifikasi_Obesitas.h5'
storage_client = storage.Client.from_service_account_json('cloud-storage.json')

temp_model_path = 'download/temp_model.h5'

# Memeriksa apakah file sudah ada
if not os.path.exists(temp_model_path):
    # Jika belum ada, lakukan pengunduhan
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_filename)

    model_file = BytesIO()
    blob.download_to_file(model_file)
    model_file.seek(0)

    # Menyimpan file ke lokasi lokal
    with open(temp_model_path, 'wb') as temp_model_file:
        temp_model_file.write(model_file.read())
    
# Load model using TensorFlow
model = keras.models.load_model(temp_model_path)

# Load MinMaxScaler
loaded_mm_scaler = joblib.load('mm_scaler.joblib')

# Load LabelEncoder
loaded_le_target = joblib.load('le_target.joblib')

def preprocess_input(Tinggi, Berat):
    # Calculate BMI
    bmi = Berat / ((Tinggi / 100) ** 2)
    input_data = loaded_mm_scaler.transform([[bmi]])

    return np.array(input_data)

def make_prediction(model, input_data):
    # Make prediction
    prediction = model.predict(input_data)

    # If your model output is one-hot encoded, convert it to a numerical label
    predicted_label = np.argmax(prediction)

    # Convert numerical label back to original label
    predicted_label_original = loaded_le_target.inverse_transform([predicted_label])[0]
    
    return predicted_label_original

@app.route('/predict-obesitas', methods=['POST'])
def predict_obesitas():
    tinggi = request.form.get('tinggi')
    berat = request.form.get('berat')

    try:
        tinggi = float(tinggi)
        berat = float(berat)
    except ValueError:
        return jsonify({'status': 'ERROR', 'message': 'Invalid numeric values for tinggi or berat.'}), 400

    if tinggi and berat:
        input_data = preprocess_input(tinggi, berat)

        model_predict = make_prediction(model, input_data)
        
        return jsonify({'status': 'SUCCESS','tinggi':tinggi, 'berat':berat, 'result': model_predict}), 200
    else:
        return jsonify({'status': 'ERROR', 'message': 'No file provided.'}), 400
    
if __name__ == '__main__':
    app.run(debug=True)