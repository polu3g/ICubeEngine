import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import keras_tuner as kt


# Load the data
data_file = "saliva_glucose_data.csv"  # Replace with the actual path
data = pd.read_csv(data_file)

# Inspect the data
print("Data Sample:\n", data.head())

# Normalize features and targets separately
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Separate features and target variable
X = data[['saliva_glucose']].values
y = data[['blood_glucose']].values

# Normalize independently
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


# Create Keras Model for Keras Tuner
def build_model(hp):
    # Define the model architecture dynamically
    model = Sequential()
    model.add(Dense(hp.Int('units_1', min_value=64, max_value=256, step=32), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('units_2', min_value=32, max_value=128, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model


# Initialize Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="my_dir",
    project_name="glucose_prediction"
)

# Perform hyperparameter search
tuner.search(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Print the best hyperparameters found
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters found: {best_hps.get_config()}")

# Train the best model using the found hyperparameters
model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test data
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# Make predictions
predictions = model.predict(X_test)

# Rescale predictions and actual values for comparison
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()

# Display some results
results = pd.DataFrame({
    'Actual blood glucose': y_test_rescaled,
    'Predicted blood glucose': predictions_rescaled
})
print("Sample Predictions:\n", results.head())

# Plot actual vs predicted values
plt.scatter(y_test_rescaled, predictions_rescaled)
plt.xlabel('Actual Blood Glucose')
plt.ylabel('Predicted Blood Glucose')
plt.title('Actual vs Predicted Blood Glucose')
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
print(f"RMSE (Root Mean Squared Error): {rmse}")
