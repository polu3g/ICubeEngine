
 
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
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# File path
file_path = "lp_model.keras"

# Check if the file exists
if os.path.exists(file_path):
    # Remove the file
    os.remove(file_path)
    print(f"{file_path} has been successfully removed.")
else:
    print(f"{file_path} does not exist.")


# Load and preprocess data
lp_train = pd.read_csv(
    "lp_train_exp_alphabet_adv.csv",
    names=["City", "Season", "Month", "Promotions", "Offers",
           "ShippingOptions", "LocalFestivals", "WebsiteThemeType"],
    encoding='ISO-8859-1'
)

lp_features = lp_train.copy()
lp_labels = lp_features.pop('WebsiteThemeType')

# Encode labels
le = LabelEncoder()
le.fit(lp_labels)
lp_labels_enc = le.transform(lp_labels)

# One-hot encode categorical features
encoded_features = one_hot_encoder.fit_transform(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])

encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]))

lp_features = pd.concat([lp_features.drop(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"], axis=1), encoded_df], axis=1)

lp_features = np.array(lp_features)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(lp_features, lp_labels_enc, test_size=0.2, random_state=42)


# Define the number of folds
n_splits = 5  # You can adjust this value

# Initialize the KFold splitter
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate over the splits
for train_index, test_index in kf.split(lp_features):
    X_train, X_test = lp_features[train_index], lp_features[test_index]
    y_train, y_test = lp_labels_enc[train_index], lp_labels_enc[test_index]
    # Now, train your model on X_train and y_train, and evaluate on X_test and y_test


# Model setup
lp_model = tf.keras.Sequential([
  layers.Dense(256, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout(0.3),
  layers.Dense(128, activation='relu'),
  layers.BatchNormalization(),
  layers.Dropout(0.3),
  layers.Dense(1)
])

# Compile the model
lp_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 metrics=['mean_absolute_error'])

# Callbacks for early stopping and learning rate reduction
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Train the model with validation
history = lp_model.fit(X_train, y_train, epochs=1000, 
                       validation_data=(X_val, y_val),
                       callbacks=[early_stopping, reduce_lr])

# Train the model without early stop
""" history = lp_model.fit(X_train, y_train, epochs=100, 
                       validation_data=(X_val, y_val)) """

# Save the model
lp_model.save('lp_model.keras')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#################################### Model Restoration ####################################

restored_model = tf.keras.models.load_model('lp_model.keras')

# Check its architecture
restored_model.summary()

# Encode the input data
# input_data = ["Atlanta, USA", "Summer", "February", "Points Programs (earn points for purchases)", "Buy One Get One (BOGO)", "In-Store Pickup", "Notting Hill Carnival (UK)"]

############ ["City",         "Season",       "Month",  "Promotions",                                   "Offers",               "ShippingOptions",    "LocalFestivals"]
input_data = ["New York City, USA",	"Dry Season",	"January",	"Percentage Off (e.g., 20% off)",	"Percentage Discount",	"Standard Shipping",	"New Year's Day (Global)"]

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



