
 
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
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

# Load the data with a different encoding
lp_train = pd.read_csv(
    "lp_train_exp_alphabet.csv",
    names=["City", "Season", "Month", "Promotions", "Offers",
           "ShippingOptions", "LocalFestivals", "WebsiteThemeType"],
    encoding='ISO-8859-1'  # Try using 'ISO-8859-1' encoding
)

lp_train.head()

lp_features = lp_train.copy()
lp_labels = lp_features.pop('WebsiteThemeType')

# Encode labels
le = LabelEncoder()
le.fit(lp_labels)
lp_labels_enc = le.transform(lp_labels)

# One-hot encode categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Ignore unknown categories during transform
encoded_features = one_hot_encoder.fit_transform(lp_features[["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]])

encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"]))

lp_features.drop(["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"], axis=1, inplace=True)
lp_features = pd.concat([lp_features, encoded_df], axis=1)

lp_features = np.array(lp_features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(lp_features, lp_labels_enc, test_size=0.2, random_state=42)

# Model setup
lp_model = tf.keras.Sequential([
  layers.Dense(256, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

# Compile the model
lp_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(), metrics=['mean_absolute_error'])

# Train the model with validation
history = lp_model.fit(X_train, y_train, epochs=100, 
                       validation_data=(X_val, y_val))

# Save the entire model as a SavedModel.
lp_model.save('lp_model.keras')

# Plot training & validation loss values
import matplotlib.pyplot as plt

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
input_data = [ "Rome, Italy",	"Early Winter",	"April",	"Buy More Save More (e.g., buy 2 get 1 free)",	"Limited-Time Offers",	"Same-Day Delivery",	"Independence Day (Nigeria)" ]

input_data_df = pd.DataFrame([input_data], columns=["City", "Season", "Month", "Promotions", "Offers", "ShippingOptions", "LocalFestivals"])

encoded_input_data = one_hot_encoder.transform(input_data_df)
encoded_input_data = np.array(encoded_input_data)

y_pred = restored_model.predict(encoded_input_data)

# Generate arg maxes for predictions
CATEGORIES = [
"AliceBlue","AntiqueWhite","Aqua","Aquamarine","Azure","Beige","Bisque","Black","BlanchedAlmond","Blue",
"BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson",
"Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen",
"DarkOrange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkTurquoise",
"DarkViolet","DeepPink","DeepSkyBlue","DimGray","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia",
"Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo",
"Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan",
"LightGoldenRodYellow","LightGray","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray",
"LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon","MediumAquaMarine","MediumBlue",
"MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed",
"MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange",
"OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink",
"Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell",
"Sienna","Silver","SkyBlue","SlateBlue","SlateGray","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato",
"Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen","RebeccaPurple","LightSalmon","Salmon",
"DarkSalmon","LightCoral","IndianRed","Crimson","FireBrick","DarkRed","Red","Pink","LightPink","HotPink","DeepPink",
"MediumVioletRed","PaleVioletRed","Coral","Tomato","OrangeRed","DarkOrange","Orange","Gold","Yellow","LightYellow",
"LemonChiffon","LightGoldenRodYellow","PapayaWhip","Moccasin","PeachPuff","PaleGoldenRod","Khaki","DarkKhaki","Lavender",
"Thistle","Plum","Violet","Orchid","Fuchsia","Magenta","MediumOrchid","MediumPurple","RebeccaPurple","BlueViolet",
"DarkViolet","DarkOrchid","DarkMagenta","Purple","Indigo","SlateBlue","DarkSlateBlue","MediumSlateBlue","GreenYellow",
"Chartreuse","LawnGreen","Lime","SpringGreen","MediumSpringGreen","LightGreen","PaleGreen","DarkSeaGreen","MediumSeaGreen",
"SeaGreen","ForestGreen","Green","DarkGreen","YellowGreen","OliveDrab","Olive","DarkOliveGreen","MediumAquaMarine",
"DarkCyan","Teal","Aqua","Cyan","LightCyan","PaleTurquoise","AquaMarine"]

pred_name = CATEGORIES[int(np.round(y_pred[0, 0]))]

print("Web-theme color base")

print(pred_name)

