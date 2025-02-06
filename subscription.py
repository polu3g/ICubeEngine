from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Placeholder for encoders, scalers, and models
le_model = LabelEncoder()
le_cartridge = LabelEncoder()
scaler = StandardScaler()

# Pre-trained model placeholder
model = Sequential()

# Load data and train the model (replace this with the actual model loading process)
def train_model():
    global le_model, le_cartridge, scaler, model
    
    # Replace with actual file path
    file_path = "printer_subscription_data.xlsx"
    data = pd.read_excel(file_path)
    
    # Preprocess
    data['model_encoded'] = le_model.fit_transform(data['model_name'])
    data['cartridge_encoded'] = le_cartridge.fit_transform(data['cartridge_name'])
    
    data['carbon_footprint_savings_percent'] = (
        (data['baseline_carbon_footprint'] - data['carbon_footprint']) / data['baseline_carbon_footprint']
    ) * 100

    features = ['model_encoded', 'cartridge_encoded', 'daily_usage']
    X = data[features]
    targets = ['savings', 'carbon_footprint_savings_percent']
    y = data[targets]

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define and compile the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(len(targets))
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Train the model initially
train_model()

# REST API Endpoint
@app.route('/predict/savings', methods=['POST'])
def predict_savings():
    try:
        # Read JSON input
        input_data = request.get_json()

        # Convert input JSON to DataFrame
        new_df = pd.DataFrame(input_data)

        # Add unseen labels to the encoder
        new_labels_model = new_df['model_name'].unique()
        new_labels_cartridge = new_df['cartridge_name'].unique()
        
        le_model.classes_ = np.append(le_model.classes_, np.setdiff1d(new_labels_model, le_model.classes_))
        le_cartridge.classes_ = np.append(le_cartridge.classes_, np.setdiff1d(new_labels_cartridge, le_cartridge.classes_))
        
        # Transform the new labels
        new_df['model_encoded'] = le_model.transform(new_df['model_name'])
        new_df['cartridge_encoded'] = le_cartridge.transform(new_df['cartridge_name'])
        
        # Prepare the input for prediction
        new_X = new_df[['model_encoded', 'cartridge_encoded', 'daily_usage']]
        new_X_scaled = scaler.transform(new_X)

        # Make predictions
        predictions = model.predict(new_X_scaled)
        predicted_savings = predictions[:, 0].flatten()  # First column: savings
        predicted_footprint_savings_percent = predictions[:, 1].flatten()  # Second column: carbon savings percentage

        # Create response JSON
        response = []
        for i in range(len(new_df)):
            response.append({
                "model_name": new_df['model_name'].iloc[i],
                "cartridge_name": new_df['cartridge_name'].iloc[i],
                "daily_usage": new_df['daily_usage'].iloc[i],
                "predicted_savings": f"${predicted_savings[i]:.2f}",
                "predicted_carbon_footprint_savings_percent": f"{predicted_footprint_savings_percent[i]:.2f}%"
            })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
