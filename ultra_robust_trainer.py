"""
LETTER TRAINER
==============
Trains a Dense neural network on collected letter landmark data.
Saves: models/letters_model.h5 + models/letters_labels.pkl
"""

import numpy as np
import json
import os
import pickle
from glob import glob

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("Run: pip install tensorflow scikit-learn")
    exit()

DATA_DIR   = "sign_data/letters"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_KEYS = ["features", "landmarks", "keypoints", "data", "hand_landmarks"]
LABEL_KEYS   = ["label", "letter", "class", "sign", "gesture"]


def extract_features(data, fallback_label):
    """Extract feature vector and label from a JSON dict, regardless of key names."""
    # Features
    feat = None
    for key in FEATURE_KEYS:
        if key in data:
            feat = data[key]
            break
    if feat is None:
        for v in data.values():
            if isinstance(v, (list, np.ndarray)):
                feat = v
                break

    # Label
    lbl = None
    for key in LABEL_KEYS:
        if key in data:
            lbl = str(data[key])
            break
    if lbl is None:
        lbl = fallback_label

    return feat, lbl


def load_data():
    X, y = [], []

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Data directory not found: {DATA_DIR}")
        print("Run your collector script first.")
        return None, None, None

    print("\nLoading letter data...")
    skipped = 0
    for letter_folder in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, letter_folder)
        if not os.path.isdir(path):
            continue

        files = glob(os.path.join(path, "*.json"))
        if not files:
            continue

        folder_count = 0
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [WARN] Could not read {fp}: {e}")
                skipped += 1
                continue

            feat, lbl = extract_features(data, fallback_label=letter_folder)

            if feat is None:
                print(f"  [WARN] No feature array found in {fp}, skipping.")
                skipped += 1
                continue

            X.append(feat)
            y.append(lbl)
            folder_count += 1

        print(f"  {letter_folder}: {folder_count} samples loaded")

    if skipped:
        print(f"\n  [!] Skipped {skipped} files due to errors.")

    if not X:
        print("[ERROR] No valid samples found.")
        return None, None, None

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y)
    labels = sorted(list(set(y)))

    print(f"\n  Total : {len(X_arr)} samples")
    print(f"  Classes: {len(labels)}  →  {labels}")
    print(f"  Feature dim: {X_arr.shape[1]}")
    return X_arr, y_arr, labels


def build_model(input_dim, num_classes):
    print("\nBuilding model...")
    inp = layers.Input(shape=(input_dim,))

    x = layers.Dense(512, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"  Parameters: {model.count_params():,}")
    return model


def train():
    print("=" * 60)
    print("  LETTER MODEL TRAINER")
    print("=" * 60)

    X, y, labels = load_data()
    if X is None:
        return

    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_model(X.shape[1], len(labels))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "best_letter.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
    ]

    print("\nTraining...\n")
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=120,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  TEST ACCURACY : {acc * 100:.2f}%")
    print(f"  TEST LOSS     : {loss:.4f}")
    print(f"{'=' * 60}")

    model_path  = os.path.join(MODELS_DIR, "letters_model.h5")
    labels_path = os.path.join(MODELS_DIR, "letters_labels.pkl")
    model.save(model_path)
    with open(labels_path, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"\n  Saved model  → {model_path}")
    print(f"  Saved labels → {labels_path}")

    # Per-class accuracy report
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print("\n  Per-class accuracy on test set:")
    print(f"  {'Letter':<10} {'Samples':>7}  {'Accuracy':>8}")
    print(f"  {'-'*30}")
    for i, lbl in enumerate(encoder.classes_):
        mask = y_test == i
        n = mask.sum()
        if n > 0:
            cls_acc = (y_pred[mask] == i).mean() * 100
            print(f"  {lbl:<10} {n:>7}  {cls_acc:>7.1f}%")

    print("\n  Done. Next: python asl_system.py")


if __name__ == "__main__":
    train()