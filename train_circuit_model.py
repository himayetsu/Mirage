import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Load feedback data and extract image patches
def load_feedback_images(feedback_file, image_path):
    if not os.path.exists(feedback_file):
        return np.array([]), np.array([]), {}
    
    data = pd.read_csv(feedback_file, names=['x', 'y', 'label'])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    images, labels = [], []
    label_map = {label: i for i, label in enumerate(data['label'].unique())}
    
    for _, row in data.iterrows():
        x, y = int(row['x']), int(row['y'])
        label = row['label']
        
        patch = image[y-15:y+15, x-15:x+15]  # Extract small patch
        if patch.shape == (30, 30):
            images.append(patch)
            labels.append(label_map[label])
    
    return np.array(images), np.array(labels), label_map

# Build CNN model
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Retrain CNN using feedback data
def retrain_model(feedback_file, image_path, model_output):
    X, y, label_map = load_feedback_images(feedback_file, image_path)
    if len(X) == 0:
        print("No new feedback data available. Skipping retraining.")
        return

    X = X.reshape(-1, 30, 30, 1) / 255.0  # Normalize and reshape
    y = to_categorical(y, num_classes=len(label_map))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(len(label_map))
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    model.save(model_output)
    print(f"Model retrained and saved to {model_output}")

if __name__ == "__main__":
    retrain_model("feedback.csv", "test_circuit.png", "circuit_model.h5")
