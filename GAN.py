import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import random

# Step 1: Load the existing data
try:
    data = pd.read_csv('printer-sales-data.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('printer-sales-data.csv', encoding='latin1')

# Step 2: Preprocess the data
label_encoders = {}
for column in ['state', 'model', 'subscription']:
    if column in data.columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Step 3: Train a generative model
X = data.drop('price', axis=1).values
y = data['price'].values

# Convert to float arrays
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)

# Create a simple neural network model
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=50, batch_size=16, verbose=2)

# Plot the loss per epoch
plt.plot(history.history['loss'])
plt.title('Model Loss per Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Step 4: Generate synthetic data
num_samples = 1000
synthetic_data = []

# Generate synthetic data based on the learned model
for _ in range(num_samples):
    synthetic_sample = {}
    for column in ['state', 'model', 'subscription']:
        if column in label_encoders:
            synthetic_sample[column] = random.choice(data[column].unique())
            synthetic_sample[column] = label_encoders[column].inverse_transform([synthetic_sample[column]])[0]

    synthetic_input = np.array([[label_encoders['state'].transform([synthetic_sample['state']])[0],
                                 label_encoders['model'].transform([synthetic_sample['model']])[0],
                                 label_encoders['subscription'].transform([synthetic_sample['subscription']])[0]]])
    synthetic_input = synthetic_input.astype(float)
    synthetic_sample['price'] = model.predict(synthetic_input)[0][0]

    synthetic_data.append(synthetic_sample)

# Step 5: Save the synthetic data to a CSV file
synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df.to_csv('synthetic_printer-sales-data.csv', index=False, encoding='utf-8')

print("Synthetic data generated and saved to 'synthetic_printer-sales-data.csv'")
