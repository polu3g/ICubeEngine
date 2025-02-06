import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# Read data using Pandas DataFrame
lp_train = pd.read_csv("lp_train_exp.csv", names=["City", "Season", "Month", "Promotions", "Offers",
                                                  "ShippingOptions", "LocalFestivals", "WebsiteThemeType"])

# Get unique values for both WebsiteThemeType and LocalFestivals
unique_website_types = lp_train['WebsiteThemeType'].unique()
unique_festivals = lp_train['LocalFestivals'].unique()

# Combine unique values to ensure all classes are present
all_unique_classes = np.unique(np.concatenate([unique_website_types, unique_festivals]))

# Split features and labels
lp_labels = lp_train[['WebsiteThemeType', 'LocalFestivals']]
lp_features = lp_train.drop(['WebsiteThemeType', 'LocalFestivals'], axis=1)

# Encode labels using LabelEncoder with all unique classes
le_website = LabelEncoder()
le_website.fit(all_unique_classes)
lp_labels['WebsiteThemeType'] = le_website.transform(lp_labels['WebsiteThemeType'])

le_festival = LabelEncoder()
le_festival.fit(all_unique_classes)
lp_labels['LocalFestivals'] = le_festival.transform(lp_labels['LocalFestivals'])

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
lp_features_scaled = scaler.fit_transform(lp_features)

# Define the model
lp_model = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Output layer for WebsiteThemeType and LocalFestivals
])

# Compile the model
lp_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['mean_absolute_error'])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(lp_features_scaled, lp_labels, test_size=0.2, random_state=42)

# Train the model
lp_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Save the trained model
lp_model.save('lp_model.keras')

# Load the saved model
restored_model = tf.keras.models.load_model('lp_model.keras')

# Generate predictions on a sample input
sample_input = np.array([[150, 150, 150, 150, 150, 150]])  # Example input data
sample_input_scaled = scaler.transform(sample_input)
predictions = restored_model.predict(sample_input_scaled)

# Decode the predicted labels
predicted_website_index = int(np.round(predictions[0, 0]))
predicted_festival_index = int(np.round(predictions[0, 1]))

predicted_website = le_website.inverse_transform([predicted_website_index])[0]
predicted_festival = le_festival.inverse_transform([predicted_festival_index])[0]

print("Predicted WebsiteThemeType:", predicted_website)
print("Predicted LocalFestivals:", predicted_festival)
