
 
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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

lp_train = pd.read_csv(
    "lp_train_exp_alphabet_adv.csv",
    names=["City", "Season", "Month", "Promotions", "Offers",
           "ShippingOptions", "LocalFestivals", "WebsiteThemeType"],
    encoding='ISO-8859-1'
)

lp_features = lp_train.copy()
lp_labels = lp_features.pop('WebsiteThemeType')
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = one_hot_encoder.fit_transform(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])


#################################### Model Restoration ####################################

restored_model = tf.keras.models.load_model('lp_model.keras')

# Check its architecture
restored_model.summary()

# Encode the input data
input_data = ["New York City, USA",	"Dry Season",	"January",	"Percentage Off (e.g., 20% off)",	"Percentage Discount",	"Standard Shipping",	"New Year's Day (Global)"]

# input_data = ['Minneapolis, USA', 'Early Winter', 'October', 'Refer a Friend Discounts', 'Percentage Discount', 'In-Store Pickup', "New Year's Day (Global)"]
############ ["City",         "Season",       "Month",  "Promotions",                                   "Offers",               "ShippingOptions",    "LocalFestivals"]

print(input_data)

input_data_df = pd.DataFrame([input_data], columns=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"])

encoded_input_data = one_hot_encoder.transform(input_data_df)
encoded_input_data = np.array(encoded_input_data)

y_pred = restored_model.predict(encoded_input_data)


# Generate arg maxes for predictions
# Read the categories from a file
def read_categories(file_path):
    with open(file_path, 'r') as file:
        categories = [line.strip() for line in file.readlines()]
    return categories

# Path to the file containing the categories
file_path = 'categories.txt'

# Read the categories into a list
CATEGORIES = read_categories(file_path)

pred_name = CATEGORIES[int(np.round(y_pred[0, 0]))]

print("Web-theme color base")

RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"
print(f"{RED_COLOR}{pred_name}{RESET_COLOR}")



