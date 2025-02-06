
 
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

# One-hot encoder for categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# File path for model
file_path = "lp_model.keras"

# Check if the model file exists and remove it if it does
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been successfully removed.")
else:
    print(f"{file_path} does not exist.")

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

encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]))

lp_features = pd.concat([lp_features.drop(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"], axis=1), encoded_df], axis=1)

lp_features = np.array(lp_features)

# Split the data
X_train, X_val, y_train_website_theme, y_val_website_theme, y_train_pricing_strategy, y_val_pricing_strategy = train_test_split(
    lp_features, lp_labels_website_theme_enc, lp_labels_pricing_strategy_enc, test_size=0.2, random_state=42
)

# Define the number of folds
n_splits = 5

# Initialize the KFold splitter
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate over the splits (example purpose, you can integrate KFold in the training process)
for train_index, test_index in kf.split(lp_features):
    X_train_fold, X_test_fold = lp_features[train_index], lp_features[test_index]
    y_train_website_theme_fold, y_test_website_theme_fold = lp_labels_website_theme_enc[train_index], lp_labels_website_theme_enc[test_index]
    y_train_pricing_strategy_fold, y_test_pricing_strategy_fold = lp_labels_pricing_strategy_enc[train_index], lp_labels_pricing_strategy_enc[test_index]
    # You can now train your model on X_train and evaluate on X_test

# Model setup
input_shape = X_train.shape[1]
inputs = layers.Input(shape=(input_shape,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Output layers
website_theme_output = layers.Dense(len(le_website_theme.classes_), activation='softmax', name='website_theme_output')(x)
pricing_strategy_output = layers.Dense(len(le_pricing_strategy.classes_), activation='softmax', name='pricing_strategy_output')(x)

# Define the model with the two outputs
lp_model = tf.keras.Model(inputs=inputs, outputs=[website_theme_output, pricing_strategy_output])

# Compile the model
lp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'website_theme_output': 'sparse_categorical_crossentropy',
        'pricing_strategy_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'website_theme_output': 'accuracy',
        'pricing_strategy_output': 'accuracy'
    }
)

# Callbacks for early stopping and learning rate reduction
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=40, min_lr=0.00001)

# Train the model with validation
history = lp_model.fit(
    X_train, 
    {'website_theme_output': y_train_website_theme, 'pricing_strategy_output': y_train_pricing_strategy},
    epochs=1000,
    validation_data=(X_val, {'website_theme_output': y_val_website_theme, 'pricing_strategy_output': y_val_pricing_strategy}),
    callbacks=[early_stopping, reduce_lr]
)

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
input_data = ["New York City, USA", "Dry Season", "January", "Percentage Off (e.g., 20% off)", "Percentage Discount", "Standard Shipping", "New Year's Day (Global)"]
input_data_df = pd.DataFrame([input_data], columns=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"])

encoded_input_data = one_hot_encoder.transform(input_data_df)
encoded_input_data = np.array(encoded_input_data)

# Predict both labels
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

# Read the categories into lists
CATEGORIES_WEBSITE_THEME = read_categories(file_path_website_theme)
CATEGORIES_PRICING_STRATEGY = read_categories(file_path_pricing_strategy)

pred_website_theme_name = CATEGORIES_WEBSITE_THEME[int(np.argmax(y_pred_website_theme[0]))]
pred_pricing_strategy_name = CATEGORIES_PRICING_STRATEGY[int(np.argmax(y_pred_pricing_strategy[0]))]

print("Predicted Website Theme Type and Pricing Strategy")
print(f"input_data => {input_data}")

RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"
print(f"{RED_COLOR}Website Theme Type: {pred_website_theme_name}{RESET_COLOR}")
print(f"{RED_COLOR}Pricing Strategy: {pred_pricing_strategy_name}{RESET_COLOR}")
