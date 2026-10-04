import os
import glob
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Set fixed seeds
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'cnn_dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
CLASSES = ['New_Sealed', 'Like_New', 'Minor_Wear', 'Damaged']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

print("=" * 80)
print("TRAINING HIEN RETURNED-PRODUCT CONDITION CNN CLASSIFIER")
print("=" * 80)

def load_split_data(split_name, shuffle=False):
    """Load image split directly into normalized NumPy arrays."""
    X, y = [], []
    for cls_name in CLASSES:
        cls_idx = CLASS_TO_IDX[cls_name]
        pattern = os.path.join(DATA_DIR, split_name, cls_name, '*.png')
        files = sorted(glob.glob(pattern))
        for f in files:
            img = Image.open(f).convert('RGB').resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(cls_idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    if shuffle:
        p = np.random.permutation(len(X))
        X, y = X[p], y[p]
        
    return X, y

print("\n[1/5] Ingesting Synthetic Dataset...")
X_train, y_train = load_split_data('train', shuffle=True)
X_val, y_val = load_split_data('validation', shuffle=True)
X_test, y_test = load_split_data('test', shuffle=False)

print(f" - Train Split : {len(X_train)} samples, shape {X_train.shape}")
print(f" - Val Split   : {len(X_val)} samples, shape {X_val.shape}")
print(f" - Test Split  : {len(X_test)} samples, shape {X_test.shape}")

y_train_cat = keras.utils.to_categorical(y_train, num_classes=4)
y_val_cat = keras.utils.to_categorical(y_val, num_classes=4)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=4)

# -------------------------------------------------------------
# 2. Build High-Performance Lightweight CNN
# -------------------------------------------------------------
print("\n[2/5] Constructing Lightweight Multi-Scale CNN Architecture...")

def build_lightweight_cnn():
    inputs = keras.Input(shape=(128, 128, 3), name='image_input')
    
    # Layer 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Layer 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Layer 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global Pooling & Dense Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation='softmax', name='condition_probabilities')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='hien_condition_cnn')
    return model

model = build_lightweight_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------------
# 3. Model Training
# -------------------------------------------------------------
print("\n[3/5] Training CNN on Train Split...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=12,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------------
# 4. Held-Out Test Evaluation
# -------------------------------------------------------------
print("\n[4/5] Evaluating Real Performance Metrics on Test Set (360 Images)...")

y_pred_probs = model.predict(X_test, batch_size=32, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred).tolist()

print("\n" + "=" * 60)
print("TEST SET BENCHMARK RESULTS:")
print("-" * 60)
print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"Precision     : {prec * 100:.2f}%")
print(f"Recall        : {rec * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(np.array(cm))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))
print("=" * 60)

# -------------------------------------------------------------
# 5. Save Artifacts & Metadata
# -------------------------------------------------------------
print("\n[5/5] Saving Trained Model and Metrics...")

model_save_path = os.path.join(MODEL_DIR, 'hien_condition_cnn.keras')
model.save(model_save_path)
print(f" -> Model saved to: {model_save_path}")

classes_save_path = os.path.join(MODEL_DIR, 'cnn_class_names.json')
with open(classes_save_path, 'w') as f:
    json.dump(CLASSES, f, indent=2)
print(f" -> Class names saved to: {classes_save_path}")

metrics_save_path = os.path.join(MODEL_DIR, 'cnn_metrics.json')
metrics_data = {
    "model_name": "HIEN-Condition-CNN (Convolutional Neural Network)",
    "architecture": "3-Block Conv2D + MaxPooling + GlobalAveragePooling + Dense(64) + Softmax(4)",
    "classes": CLASSES,
    "input_resolution": [128, 128, 3],
    "dataset_splits": {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "total_samples": len(X_train) + len(X_val) + len(X_test)
    },
    "test_metrics": {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "per_class": {
            cls_name: {
                "precision": round(float(precision_score(y_test == i, y_pred == i, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test == i, y_pred == i, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test == i, y_pred == i, zero_division=0)), 4)
            }
            for i, cls_name in enumerate(CLASSES)
        }
    }
}

with open(metrics_save_path, 'w') as f:
    json.dump(metrics_data, f, indent=2)
print(f" -> CNN Metrics saved to: {metrics_save_path}")

print("\nCNN Training and Evaluation Completed Successfully!")
