"""
LSTM Attribution Model — Deep learning model for conversion prediction.

Architecture: Masking → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(sigmoid)
Trained on padded journey sequences to predict conversion probability.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress TF warnings for cleaner output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_attribution.keras")


def build_lstm_model(max_seq_length, n_features, mask_value=-1.0):
    """
    Build LSTM model for binary conversion prediction.

    Architecture:
        Masking → LSTM(64, return_sequences) → Dropout(0.3) →
        LSTM(32) → Dropout(0.3) → Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Masking(mask_value=mask_value, input_shape=(max_seq_length, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def compute_class_weights(y_train):
    """Compute class weights to handle imbalanced conversion data."""
    n_total = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_total - n_pos

    # Balanced weighting
    weight_neg = n_total / (2 * n_neg)
    weight_pos = n_total / (2 * n_pos)

    return {0: weight_neg, 1: weight_pos}


def get_callbacks():
    """Return training callbacks: EarlyStopping + ReduceLROnPlateau."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the LSTM model with class weighting and callbacks.

    Returns: training history
    """
    class_weights = compute_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")

    callbacks = get_callbacks()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def save_model(model, path=None):
    """Save trained model to disk."""
    if path is None:
        path = MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"\nModel saved to: {path}")


def load_model(path=None):
    """Load trained model from disk."""
    if path is None:
        path = MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    return keras.models.load_model(path)


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics."""
    # Keras 3 returns dict from evaluate with return_dict=True
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    print(f"\nTest Set Evaluation:")
    for name, value in results.items():
        print(f"  {name:12s} {value:.4f}")

    return results
