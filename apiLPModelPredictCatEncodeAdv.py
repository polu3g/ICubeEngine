from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

app = Flask(__name__)

# Load model and encoder at the start for efficiency
MODEL_PATH = 'lp_model.keras'
CATEGORIES_PATH = 'categories.txt'
TRAIN_DATA_PATH = 'lp_train_exp_alphabet_adv.csv'

# Load and prepare the model
try:
    restored_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    restored_model = None

# Load categories
def read_categories(file_path):
    try:
        with open(file_path, 'r') as file:
            categories = [line.strip() for line in file.readlines()]
        return categories
    except Exception as e:
        print(f"Error reading categories: {e}")
        return []

CATEGORIES = read_categories(CATEGORIES_PATH)

# Load and prepare the training data for the encoder
try:
    lp_train = pd.read_csv(
        TRAIN_DATA_PATH,
        names=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals", "WebsiteThemeType"],
        encoding='ISO-8859-1'
    )

    lp_features = lp_train.copy()
    lp_labels = lp_features.pop('WebsiteThemeType')

    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])
except Exception as e:
    print(f"Error preparing training data or encoder: {e}")
    one_hot_encoder = None

@app.route('/')
def index():
    return 'Welcome to the Python Web Service!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing fields in input data"}), 400

        input_data = [data[field] for field in required_fields]

        print(input_data)
        
        input_data_df = pd.DataFrame([input_data], columns=required_fields)
        
        if one_hot_encoder is None or restored_model is None:
            return jsonify({"error": "Model or encoder is not properly loaded"}), 500

        encoded_input_data = one_hot_encoder.transform(input_data_df)
        encoded_input_data = np.array(encoded_input_data)

        y_pred = restored_model.predict(encoded_input_data)
        predicted_class = CATEGORIES[int(np.round(y_pred[0, 0]))]

        prediction = [{'predicted_web_theme_color_base': predicted_class}, {"tensorVersion": tf.__version__}]
        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
