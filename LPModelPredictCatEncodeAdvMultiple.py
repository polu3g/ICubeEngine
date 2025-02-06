
 
# Author: Pallab Chakrabarti
# Email: pallab.chakrabarti@hp.com
# Date: 2024-05-03

# This script trains a neural network model to predict the website theme color based on various features.
# It includes data preprocessing steps, model training, evaluation, and prediction.

# Dependencies:
# - pandas
# - numpy
# - scikit-learn
# - tensorflow
# - matplotlib

# Ensure you have the required CSV file `lp_train_exp_alphabet.csv` in the same directory.

# The input data is provided as a list of strings representing the categorical features.
# This input data is converted into a DataFrame with the same structure as the training data.
# The DataFrame is then one-hot encoded using the same OneHotEncoder used during training.
# The encoded input data is converted to a NumPy array and passed to the model for prediction.
# Validation Split: The data is split into training and validation sets using train_test_split.
# Model Architecture: Added another dense layer to increase model complexity.
# Training with Validation: The model is trained with validation data, and the loss is plotted to check for overfitting.
# Plotting Training & Validation Loss: This helps in visualizing and diagnosing if the model is overfitting or underfitting.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os


# One-hot encoder for categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Load and preprocess data
lp_train = pd.read_csv(
    "lp_train_exp_alphabet_adv_multiple.csv",
    names=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals", "WebsiteThemeType", "PricingStrategy"],
    encoding='ISO-8859-1'
)

lp_features = lp_train.copy()


lp_labels_website_theme = lp_features.pop('WebsiteThemeType')
lp_labels_pricing_strategy = lp_features.pop('PricingStrategy')

# Encode labels
le_website_theme = LabelEncoder()
le_website_theme.fit(lp_labels_website_theme)
lp_labels_website_theme_enc = le_website_theme.transform(lp_labels_website_theme)

le_pricing_strategy = LabelEncoder()
le_pricing_strategy.fit(lp_labels_pricing_strategy)
lp_labels_pricing_strategy_enc = le_pricing_strategy.transform(lp_labels_pricing_strategy)

# One-hot encode categorical features
encoded_features = one_hot_encoder.fit_transform(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])


#################################### Model Restoration ####################################

restored_model = tf.keras.models.load_model('lp_model.keras')

# Check its architecture
restored_model.summary()

# Encode the input data
# input_data = ["New York City, USA",	"Winter",	"April",	"Percentage Off (e.g., 20% off)",	"Percentage Discount",	"Standard Shipping",	"New Year's Day (Global)"]
# input_data = ["Rome, Italy",	"Winter",	"April",	"Buy More Save More (e.g., buy 2 get 1 free)",	"Limited-Time Offers",	"Same-Day Delivery",	"Independence Day (Nigeria)"]
input_data = ["San Francisco, USA",	"Autumn (Fall)",	"May",	"Exclusive Member Sales",	"Percentage Discount",	"Flat-Rate Shipping",	"Día de los Muertos (Mexico)"]

############ ["City",         "Season",       "Month",  "Promotions",                                   "Offers",               "ShippingOptions",    "LocalFestivals"]

input_data_df = pd.DataFrame([input_data], columns=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"])

encoded_input_data = one_hot_encoder.transform(input_data_df)
encoded_input_data = np.array(encoded_input_data)

y_pred = restored_model.predict(encoded_input_data)

y_pred_website_theme = y_pred[0]
y_pred_pricing_strategy = y_pred[1]

# Generate arg maxes for predictions
def read_categories(file_path):
    with open(file_path, 'r') as file:
        categories = [line.strip() for line in file.readlines()]
    return categories

# Path to the file containing the categories
file_path_website_theme = 'categories_website_theme.txt'
file_path_pricing_strategy = 'categories_pricing_strategy.txt'

RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"

# Read the categories into lists
CATEGORIES_WEBSITE_THEME = read_categories(file_path_website_theme)
CATEGORIES_PRICING_STRATEGY = read_categories(file_path_pricing_strategy)

pred_website_theme_name = CATEGORIES_WEBSITE_THEME[int(np.argmax(y_pred_website_theme[0]))]
pred_pricing_strategy_name = CATEGORIES_PRICING_STRATEGY[int(np.argmax(y_pred_pricing_strategy[0]))]

print("Predicted Website Theme Type and Pricing Strategy")
print(f"input_data => {input_data}")

print(f"{RED_COLOR}Website Theme Type: {pred_website_theme_name}{RESET_COLOR}")
print(f"{RED_COLOR}Pricing Strategy: {pred_pricing_strategy_name}{RESET_COLOR}")

