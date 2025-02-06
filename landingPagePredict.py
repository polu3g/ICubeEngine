import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scalar= MinMaxScaler()

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)

lp_train = pd.read_csv(
    "lp_train_exp.csv",
    names=["City", "Season", "Month", "Promotions", "Offers",
           "ShippingOptions", "LocalFestivals", "WebsiteThemeType"])

lp_train.head()

lp_features = lp_train.copy()
lp_labels = lp_features.pop('WebsiteThemeType')

le = LabelEncoder()
le.fit(lp_labels)
lp_labels_enc = le.transform(lp_labels)


lp_features = np.array(lp_features)

# print(lp_features) 


lp_model = tf.keras.Sequential([
  layers.Dense(256, activation='relu'),
  layers.Dense(1)
])

# Compile the model
lp_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam(),metrics=['mean_absolute_error'])

# train
lp_model.fit(lp_features, lp_labels_enc, epochs=50, 
                    validation_split=0.2)

# Save the entire model as a SavedModel.
lp_model.save('lp_model.keras')

#################################### Model Restoration ####################################

restored_model = tf.keras.models.load_model('lp_model.keras')

# Check its architecture
restored_model.summary()

# Split dataset into training and testing sets  
# X_train, X_test, y_train, y_test = train_test_split(lp_features, lp_labels, test_size=0.2, random_state=42)  

# 53,	53,	53,	53,	53,	53,	53	GreenYellow (target- WebsiteThemeColor)
# Input data excluding target
a = [ 53,	53,	53,	53,	53,	53,	53]
a = np.array(a)  
a = np.expand_dims(a, 0) 

y_pred = restored_model.predict(a)

# print(np.round(y_pred[0, 0]))

# Generate arg maxes for predictions
CATEGORIES = [
"AliceBlue","AntiqueWhite","Aqua","Aquamarine","Azure","Beige","Bisque","Black","BlanchedAlmond","Blue","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","DarkOrange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen","RebeccaPurple","LightSalmon","Salmon","DarkSalmon","LightCoral","IndianRed","Crimson","FireBrick","DarkRed","Red","Pink","LightPink","HotPink","DeepPink","MediumVioletRed","PaleVioletRed","Coral","Tomato","OrangeRed","DarkOrange","Orange","Gold","Yellow","LightYellow","LemonChiffon","LightGoldenRodYellow","PapayaWhip","Moccasin","PeachPuff","PaleGoldenRod","Khaki","DarkKhaki","Lavender","Thistle","Plum","Violet","Orchid","Fuchsia","Magenta","MediumOrchid","MediumPurple","RebeccaPurple","BlueViolet","DarkViolet","DarkOrchid","DarkMagenta","Purple","Indigo","SlateBlue","DarkSlateBlue","MediumSlateBlue","GreenYellow","Chartreuse","LawnGreen","Lime","SpringGreen","MediumSpringGreen","LightGreen","PaleGreen","DarkSeaGreen","MediumSeaGreen","SeaGreen","ForestGreen","Green","DarkGreen","YellowGreen","OliveDrab","Olive","DarkOliveGreen","MediumAquaMarine","DarkCyan","Teal","Aqua","Cyan","LightCyan","PaleTurquoise","AquaMarine"]
pred_name = CATEGORIES[int(np.round(y_pred[0, 0]))]
print ("Web-theme color base")
print(pred_name)
