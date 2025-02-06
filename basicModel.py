import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

# Data
X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y = np.array([[2], [4], [6], [8], [10]], dtype=np.float32)

X_train, X_test = X[:4], X[4:]
y_train, y_test = y[:4], y[4:]

# Model
model = Sequential([Dense(units=1, input_shape=[1],kernel_initializer=RandomUniform(minval=-1.0, maxval=1.0))])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train
model.fit(X_train, y_train, epochs=1000, verbose=0)
print("Training complete!")

# After training the model
weights, biases = model.layers[0].get_weights()

print("Weights:", weights)
print("Biases:", biases)

# Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Mean Absolute Error: {mae}")

# Predict for 5 random numbers
random_inputs = np.random.uniform(low=10, high=50, size=(5, 1)).astype(np.float32)  # Generate 5 random numbers between 0 and 10
predictions = model.predict(random_inputs)

# Print predictions
for i, input_val in enumerate(random_inputs):
    print(f"Prediction for input {input_val[0]:.2f}: {predictions[i][0]:.2f}")