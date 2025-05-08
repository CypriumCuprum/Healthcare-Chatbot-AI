import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the synthetic data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic_health_data.csv')
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop('disease', axis=1).values
y = df['disease'].values

# Encode the target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

# Save the encoder classes for future use
np.save(os.path.join(os.path.dirname(__file__), 'class_names.npy'), encoder.classes_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple DNN model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save(os.path.join(os.path.dirname(__file__), 'health_model.h5'))
print("Model saved successfully!")

# Check accuracy for each class
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print metrics for each class
for i, class_name in enumerate(encoder.classes_):
    class_indices = np.where(y_test == i)[0]
    class_acc = np.mean(y_pred_classes[class_indices] == i)
    print(f"Accuracy for {class_name}: {class_acc:.4f}") 