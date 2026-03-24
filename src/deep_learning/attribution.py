"""
LSTM Gradient-Based Attribution — Extract per-channel attribution weights.

Uses tf.GradientTape to compute gradients of the conversion prediction
with respect to input features, then aggregates by channel to produce
attribution weights matching the format from src/attribution/models.py.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deep_learning.sequence_prep import CHANNEL_ENCODING, PAD_VALUE, encode_channels


def compute_gradients_for_journeys(model, X):
    """
    Compute gradients of model output w.r.t. input for each sample.

    Returns: gradients array with same shape as X
    """
    X_tensor = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
    gradients = tape.gradient(predictions, X_tensor)
    return gradients.numpy()


def compute_lstm_attribution(model, df, max_len, pad_value=PAD_VALUE, batch_size=512):
    """
    Extract per-channel attribution using gradient-based method.

    Process:
    1. Prepare sequences for converting journeys only
    2. Compute gradients of prediction w.r.t. input
    3. Take absolute gradient magnitude at each timestep
    4. Map timesteps back to channels and aggregate
    5. Normalize to sum to 100%

    Returns: DataFrame with columns ['channel', 'lstm_deep_learning']
             matching the format from src/attribution/models.py
    """
    from deep_learning.sequence_prep import prepare_sequences

    # Prepare all sequences (we need the full X to match indices)
    X, y, _, _ = prepare_sequences(df, max_len=max_len)

    # Filter to converting journeys only
    converting_mask = y == 1.0
    X_conv = X[converting_mask]

    if len(X_conv) == 0:
        print("  Warning: No converting journeys found.")
        return _empty_attribution()

    print(f"  Computing gradients for {len(X_conv)} converting journeys...")

    # Compute gradients in batches to manage memory
    all_grads = []
    for i in range(0, len(X_conv), batch_size):
        batch = X_conv[i:i + batch_size]
        grads = compute_gradients_for_journeys(model, batch)
        all_grads.append(grads)
    gradients = np.concatenate(all_grads, axis=0)

    # Aggregate gradient magnitude per channel
    # Use the L2 norm across features at each timestep as the importance score
    grad_magnitude = np.sqrt(np.sum(gradients ** 2, axis=-1))  # shape: [n_conv, max_len]

    # Build reverse mapping: channel_id → channel_name
    id_to_channel = {v: k for k, v in CHANNEL_ENCODING.items()}
    n_channels = len(CHANNEL_ENCODING)

    # Accumulate importance per channel across all converting journeys
    channel_importance = {ch: 0.0 for ch in CHANNEL_ENCODING}

    converting_df = df[df["has_conversion"] == 1].reset_index(drop=True)

    for i in range(len(X_conv)):
        if i >= len(converting_df):
            break
        channels = converting_df.iloc[i]["channel_list"]
        seq_len = min(len(channels), max_len)

        # If journey was truncated, align with the right (most recent) end
        offset = max(0, len(channels) - max_len)

        for j in range(seq_len):
            ch_name = channels[offset + j] if (offset + j) < len(channels) else "other"
            if ch_name not in channel_importance:
                ch_name = "other"
            channel_importance[ch_name] += grad_magnitude[i, j]

    # Normalize to proportions
    total = sum(channel_importance.values())
    if total > 0:
        channel_importance = {k: v / total for k, v in channel_importance.items()}

    # Filter out channels with zero attribution and build DataFrame
    result = pd.DataFrame([
        {"channel": ch, "lstm_deep_learning": weight}
        for ch, weight in channel_importance.items()
        if weight > 0
    ])

    if result.empty:
        return _empty_attribution()

    result = result.sort_values("lstm_deep_learning", ascending=False).reset_index(drop=True)
    return result


def _empty_attribution():
    """Return empty attribution DataFrame as fallback."""
    return pd.DataFrame(columns=["channel", "lstm_deep_learning"])


def run_lstm_attribution_pipeline(df):
    """
    Full pipeline: load saved model, compute attribution.
    Used by the agent tool.

    Returns: DataFrame with columns ['channel', 'lstm_deep_learning']
    """
    from deep_learning.lstm_model import load_model, MODEL_PATH
    from deep_learning.sequence_prep import get_max_sequence_length

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained LSTM model found at {MODEL_PATH}. "
            "Run 'python src/deep_learning/train.py' first."
        )

    model = load_model()
    max_len = get_max_sequence_length(df)

    return compute_lstm_attribution(model, df, max_len)


if __name__ == "__main__":
    from attribution.data_prep import extract_journeys
    from deep_learning.lstm_model import load_model
    from deep_learning.sequence_prep import get_max_sequence_length

    print("=" * 60)
    print("LSTM Gradient-Based Attribution")
    print("=" * 60)

    df = extract_journeys()
    model = load_model()
    max_len = get_max_sequence_length(df)

    attribution = compute_lstm_attribution(model, df, max_len)
    print(f"\nLSTM Attribution (normalized to 100%):")
    for _, row in attribution.iterrows():
        bar = "█" * int(row["lstm_deep_learning"] * 100)
        print(f"  {row['channel']:<20} {row['lstm_deep_learning']*100:5.1f}%  {bar}")
