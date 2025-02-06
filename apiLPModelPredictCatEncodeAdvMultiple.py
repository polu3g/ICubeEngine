from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

app = Flask(__name__)

# Load model and encoder at the start for efficiency
MODEL_PATH = 'lp_model.keras'
CATEGORIES_WEBSITE_THEME_PATH = 'categories_website_theme.txt'
CATEGORIES_PRICING_STRATEGY_PATH = 'categories_pricing_strategy.txt'
TRAIN_DATA_PATH = 'lp_train_exp_alphabet_adv_multiple.csv'

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

CATEGORIES_WEBSITE_THEME = read_categories(CATEGORIES_WEBSITE_THEME_PATH)
CATEGORIES_PRICING_STRATEGY = read_categories(CATEGORIES_PRICING_STRATEGY_PATH)

# Load and prepare the training data for the encoder
try:
    lp_train = pd.read_csv(
        TRAIN_DATA_PATH,
        names=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals", "WebsiteThemeType", "PricingStrategy"],
        encoding='ISO-8859-1'
    )

    lp_features = lp_train.copy()
    lp_labels_website_theme = lp_features.pop('WebsiteThemeType')
    lp_labels_pricing_strategy = lp_features.pop('PricingStrategy')

    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])

    # Label encoders for the output labels
    le_website_theme = LabelEncoder()
    le_website_theme.fit(lp_labels_website_theme)
    le_pricing_strategy = LabelEncoder()
    le_pricing_strategy.fit(lp_labels_pricing_strategy)

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

        input_data_df = pd.DataFrame([input_data], columns=required_fields)
        
        if one_hot_encoder is None or restored_model is None:
            return jsonify({"error": "Model or encoder is not properly loaded"}), 500

        encoded_input_data = one_hot_encoder.transform(input_data_df)
        encoded_input_data = np.array(encoded_input_data)

        y_pred = restored_model.predict(encoded_input_data)
        y_pred_website_theme = y_pred[0]
        y_pred_pricing_strategy = y_pred[1]

        predicted_website_theme = CATEGORIES_WEBSITE_THEME[int(np.argmax(y_pred_website_theme))]
        predicted_pricing_strategy = CATEGORIES_PRICING_STRATEGY[int(np.argmax(y_pred_pricing_strategy))]

        prediction = {
            'predicted_web_theme_color_base': predicted_website_theme,
            'pred_pricing_strategy_name': predicted_pricing_strategy,
            "iCubeVersion": tf.__version__
        }
        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)    

