#!/usr/bin/env python3
# Mushroom Classification - Main Program
# Based on the Jupyter notebook "secondapril.ipynb"

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train")
TEST_PATH = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULT_PATH = os.path.join(OUTPUT_DIR, "result.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.keras")

# Model parameters
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 80

# Class mapping
CLASS_MAPPING = {
    'bào ngư xám + trắng': 1,
    'linh chi trắng': 3,
    'nấm mỡ': 0,
    'Đùi gà Baby (cắt ngắn)': 2
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Preprocessing Functions
def preprocess_image(image, image_size=IMAGE_SIZE):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image / 255.0

def load_and_preprocess_image(path, image_size=IMAGE_SIZE):
    image = tf.io.read_file(path)
    return preprocess_image(image, image_size)

def load_and_preprocess_from_path_label(path, label):
    image = load_and_preprocess_image(path)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

# Training and Validation Data Pipeline
def train_val_data_pipeline():
    data_root = pathlib.Path(TRAIN_PATH)
    all_image_paths = np.array([str(path) for path in data_root.glob('*/*') if path.is_file()])
    all_labels = np.array([CLASS_MAPPING[path.parent.name] for path in data_root.glob('*/*')])
    
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
    dataset = dataset.shuffle(len(all_image_paths), seed=0)
    
    train_size = int(0.9 * len(all_image_paths))
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)
    
    train_data = (train_data
                  .map(load_and_preprocess_from_path_label, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(BATCH_SIZE)
                  .repeat()
                  .prefetch(tf.data.AUTOTUNE))
    
    val_data = (val_data
                .map(load_and_preprocess_from_path_label, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    
    return train_data, val_data, all_image_paths, all_labels

# Testing Data Pipeline
def test_data_pipeline():
    test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH) if f.endswith('.jpg')]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
    test_dataset = (test_dataset
                    .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))
    return test_dataset, test_files

# Model Definition
def build_model():
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# TTA Prediction (Test Time Augmentation)
def predict_with_tta(model, test_dataset, test_files, n_iter=2):
    predictions = []
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True
    )
    
    base_pred = model.predict(test_dataset)
    predictions.append(base_pred)
    
    for _ in range(n_iter - 1):
        tta_generator = tta_datagen.flow_from_dataframe(
            pd.DataFrame({'filename': test_files}),
            x_col='filename',
            y_col=None,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=False
        )
        preds = model.predict(tta_generator, steps=len(tta_generator))
        predictions.append(preds)
    
    return np.mean(predictions, axis=0)

def train_and_predict():
    # Load all training data
    train_data_base, _, all_image_paths, all_labels = train_val_data_pipeline()
    total_samples = len(all_image_paths)
    
    print(f"\nTraining on full dataset ({total_samples} samples)...")
    train_dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
    train_dataset = (train_dataset
                     .map(load_and_preprocess_from_path_label, num_parallel_calls=tf.data.AUTOTUNE)
                     .shuffle(total_samples, seed=0)
                     .batch(BATCH_SIZE)
                     .repeat()
                     .prefetch(tf.data.AUTOTUNE))
    train_steps = total_samples // BATCH_SIZE + (1 if total_samples % BATCH_SIZE else 0)
    
    # Load test data (no labels, just for prediction)
    test_dataset, test_files = test_data_pipeline()
    
    # Train final model or load it if already exists
    if os.path.exists(MODEL_PATH):
        final_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded final model from {MODEL_PATH}")
    else:
        final_model = build_model()
        
        # Train the model
        final_model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            epochs=EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='loss', save_best_only=True, verbose=1)
            ],
            verbose=1
        )
    
    # Generate test set predictions
    print("\nGenerating test set predictions...")
    test_pred = predict_with_tta(final_model, test_dataset, test_files)
    test_pred_classes = np.argmax(test_pred, axis=1)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': [os.path.splitext(os.path.basename(f))[0] for f in test_files],
        'type': test_pred_classes
    })
    submission.to_csv(RESULT_PATH, index=False)
    print(f"Submission file created at: {RESULT_PATH}")
    
    return submission

def main():
    print("=== Mushroom Classification ===")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class mapping: {CLASS_MAPPING}")
    
    # Check if the data directories exist
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training data directory not found at {TRAIN_PATH}")
        return
        
    if not os.path.exists(TEST_PATH):
        print(f"Error: Test data directory not found at {TEST_PATH}")
        return
    
    # Train model and generate predictions
    submission = train_and_predict()
    
    print("\nDone!")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    main()