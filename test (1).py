import numpy as np
import pandas as pd
import joblib
import os
import time

print("Sleep Stage Classifier Test Script")
print("=================================")

# Check if model file exists
model_path = "sleep_stage_classifier.joblib"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please make sure your model file is in the current directory")
    exit(1)

# Try to load the model
print("Loading model...")
try:
    start_time = time.time()
    model = joblib.load(model_path)
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define test data (simulated)
print("Creating test data...")
# Create a simple test with data representing different sleep stages
test_data = [
    # Format: [acc_x_mean, acc_x_std, acc_y_mean, acc_y_std, acc_z_mean, acc_z_std, hr_mean, hr_std, movement_intensity]
    # Deep sleep test case (low heart rate, minimal movement)
    [0.01, 0.01, 0.01, 0.01, 9.8, 0.01, 48.0, 0.5, 0.01],
    
    # Light sleep test case (moderate heart rate, some movement)
    [0.05, 0.05, 0.05, 0.05, 9.8, 0.02, 60.0, 1.0, 0.05],
    
    # REM sleep test case (variable heart rate, some movement)
    [0.03, 0.07, 0.03, 0.07, 9.8, 0.03, 75.0, 2.0, 0.08],
    
    # Awake test case (high heart rate, significant movement)
    [0.2, 0.15, 0.2, 0.15, 9.8, 0.1, 85.0, 3.0, 0.2]
]

# Convert to numpy array
test_data = np.array(test_data)

# Make predictions
print("Making predictions...")
start_time = time.time()
predictions = model.predict(test_data)
predict_time = time.time() - start_time

# Map predictions to sleep stage names
label_names = ['Wake', 'NREM Stage 1', 'NREM Stage 2', 'NREM Stage 3', 'REM']
predicted_stages = [label_names[pred] for pred in predictions]

# Display results
print(f"\nPrediction complete in {predict_time:.3f} seconds")
print("\nTest Results:")
print("-----------------------")
for i, (features, stage) in enumerate(zip(test_data, predicted_stages)):
    hr = features[6]  # Heart rate is at index 6
    movement = features[8]  # Movement intensity is at index 8
    print(f"Sample {i+1}: Heart Rate: {hr:.1f} BPM, Movement: {movement:.3f}, Predicted: {stage}")

print("\nTest completed successfully")
print("Your model is working on the Raspberry Pi!")
print("You can now implement the full live inference system")