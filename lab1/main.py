import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D,
    Flatten,
    Dense,
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def extract_ppg_signal(frame_folder_path):
    frame_files = sorted(
        [f for f in os.listdir(frame_folder_path) if f.endswith((".jpg", ".png"))]
    )
    red_channel = []
    green_channel = []
    blue_channel = []
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        # BGR
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        blue_channel.append(b_mean)
        green_channel.append(g_mean)
        red_channel.append(r_mean)
    # Take average of the 3 channels
    ppg_signal = (
        np.array(red_channel) + np.array(green_channel) + np.array(blue_channel)
    ) / 3
    return ppg_signal


def get_windows_and_labels(ppg_signal, gt_df, window_sec=5, fps=25):
    count_gt = len(gt_df.index)
    count_windows = count_gt // window_sec
    windows = []
    labels = []
    window_size = window_sec * fps
    for i in range(count_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = ppg_signal[start_idx:end_idx]
        windows.append(window)
        # Get ground truth for the window
        gt_start = i * window_sec
        gt_end = gt_start + window_sec - 1
        gt_values = gt_df.loc[gt_start:gt_end, "SpO2"]
        label = np.mean(gt_values)  # truth value for this window
        labels.append(label)
    return windows, labels


def load_data(data_path):
    X = []  # input values in CNN
    y = []  # truth output values for CNN
    subjects = os.listdir(data_path)
    for subject in subjects:
        face_path = os.path.join(data_path, subject, "Face")
        gt_path = os.path.join(data_path, subject, "gt_SpO2.csv")
        gt_df = pd.read_csv(gt_path)  # dataframe with SpO2 values
        # Extract PPG signal
        ppg_signal = extract_ppg_signal(face_path)
        # Break down signals into windows
        windows, labels = get_windows_and_labels(ppg_signal, gt_df)
        X.extend(windows)
        y.extend(labels)
    return np.array(X), np.array(y)


# Paths to the data directories
train_path = "Task/Train"
validation_path = "Task/Validation"
test_path = "Task/Test"

# Pre-process the dataset
print("Loading training data...")
X_train, y_train = load_data(train_path)
print("Loading validation data...")
X_val, y_val = load_data(validation_path)
print("Loading test data...")
X_test, y_test = load_data(test_path)

# Reshape data for CNN input
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(125, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
model.add(AveragePooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)

# Summary of the model
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE (Loss): {test_loss}")
print(f"Test MAE (Metric): {test_mae}")

# Plot training & validation loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss (MSE)")
plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mean_absolute_error"], label="Train MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Val MAE")
plt.title("Model Metric")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend()

plt.tight_layout()
plt.show()
