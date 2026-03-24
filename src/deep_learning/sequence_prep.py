"""
Sequence Preparation — Convert journey data to LSTM-ready sequences.

Encodes channels as integers, pads sequences to equal length,
adds positional features, and splits into train/test sets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MAX_SEQ_LENGTH

# Fixed channel encoding — matches the 5 channels in GA4 sample data
CHANNEL_ENCODING = {
    "organic_search": 0,
    "paid_search": 1,
    "direct": 2,
    "referral": 3,
    "other": 4,
    "social": 5,
    "email": 6,
    "display": 7,
    "affiliate": 8,
}

# Padding value — must be different from any valid channel ID
PAD_VALUE = -1


def get_max_sequence_length(df, percentile=95):
    """Calculate max sequence length from 95th percentile of journey lengths."""
    p95 = int(np.percentile(df["journey_length"], percentile))
    return min(p95, MAX_SEQ_LENGTH)


def encode_channels(channel_list):
    """Encode a list of channel names to integer IDs."""
    return [CHANNEL_ENCODING.get(ch, CHANNEL_ENCODING["other"]) for ch in channel_list]


def prepare_sequences(df, max_len=None):
    """
    Convert journey DataFrame to LSTM-ready sequences.

    Returns:
        X: array of shape [n_samples, max_len, 2] — features are (channel_id, position_in_journey)
        y: array of shape [n_samples] — binary conversion label
        max_len: the sequence length used for padding
        channel_encoding: dict mapping channel names to integer IDs
    """
    if max_len is None:
        max_len = get_max_sequence_length(df)

    n_samples = len(df)
    n_features = 2  # channel_id, position_in_journey

    X = np.full((n_samples, max_len, n_features), PAD_VALUE, dtype=np.float32)
    y = np.array(df["has_conversion"].values, dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        channels = row["channel_list"]
        encoded = encode_channels(channels)
        seq_len = min(len(encoded), max_len)

        # Truncate from the left (keep most recent touchpoints) if too long
        if len(encoded) > max_len:
            encoded = encoded[-max_len:]

        journey_len = len(channels)
        for j in range(seq_len):
            # Feature 1: channel ID (normalized to 0-1 range)
            X[i, j, 0] = encoded[j] / max(len(CHANNEL_ENCODING) - 1, 1)
            # Feature 2: position in journey (0.0 = start, 1.0 = end)
            if len(encoded) > 1:
                original_pos = (len(channels) - seq_len + j) if len(channels) > max_len else j
                X[i, j, 1] = original_pos / (journey_len - 1)
            else:
                X[i, j, 1] = 1.0

    return X, y, max_len, CHANNEL_ENCODING


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test with stratification by conversion label.

    Returns: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_lstm_data(df):
    """
    Full pipeline: journey DataFrame → train/test LSTM sequences.

    Returns dict with X_train, X_test, y_train, y_test, max_len, channel_encoding.
    """
    X, y, max_len, channel_encoding = prepare_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    n_channels = len(CHANNEL_ENCODING)
    n_converting = int(y.sum())
    n_total = len(y)

    print(f"\nSequence Preparation Summary:")
    print(f"  Total samples: {n_total}")
    print(f"  Converting: {n_converting} ({n_converting/n_total*100:.1f}%)")
    print(f"  Max sequence length: {max_len}")
    print(f"  Features per timestep: 2 (channel_id_norm, position)")
    print(f"  Channels encoded: {n_channels}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train conversion rate: {y_train.mean()*100:.1f}%")
    print(f"  y_test conversion rate:  {y_test.mean()*100:.1f}%")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "max_len": max_len,
        "channel_encoding": channel_encoding,
        "pad_value": PAD_VALUE,
    }


if __name__ == "__main__":
    from attribution.data_prep import extract_journeys

    print("=" * 60)
    print("LSTM Sequence Preparation Test")
    print("=" * 60)

    df = extract_journeys()
    data = prepare_lstm_data(df)

    print(f"\nSample X_train[0]:")
    print(data["X_train"][0])
    print(f"Label: {data['y_train'][0]}")
