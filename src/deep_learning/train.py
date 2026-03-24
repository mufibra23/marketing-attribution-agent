"""
Training Script — Load data, train LSTM, save model, print metrics.

Usage: python src/deep_learning/train.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress TF warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    print("=" * 60)
    print("LSTM Attribution Model — Training")
    print("=" * 60)

    # Step 1: Load journey data from BigQuery
    print("\n[1/4] Loading journey data from BigQuery...")
    from attribution.data_prep import extract_journeys
    df = extract_journeys()

    # Step 2: Prepare LSTM sequences
    print("\n[2/4] Preparing LSTM sequences...")
    from deep_learning.sequence_prep import prepare_lstm_data
    data = prepare_lstm_data(df)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    max_len = data["max_len"]
    n_features = X_train.shape[2]

    # Step 3: Build and train model
    print("\n[3/4] Building and training LSTM model...")
    from deep_learning.lstm_model import (
        build_lstm_model, train_model, save_model, evaluate_model
    )

    model = build_lstm_model(max_len, n_features, mask_value=data["pad_value"])
    model.summary()

    history = train_model(model, X_train, y_train, X_test, y_test)

    # Step 4: Evaluate and save
    print("\n[4/4] Evaluating and saving model...")
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model)

    # Print training summary
    best_epoch = len(history.history["loss"])
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Epochs trained: {best_epoch}")
    print(f"  Best val AUC:   {max(history.history.get('val_auc', [0])):.4f}")
    for name, value in metrics.items():
        print(f"  Test {name}: {value:.4f}")
    print(f"  Model saved to: models/lstm_attribution.keras")

    # Step 5: Run attribution extraction as verification
    print(f"\n{'=' * 60}")
    print("Running LSTM Attribution Extraction (verification)...")
    print(f"{'=' * 60}")
    from deep_learning.attribution import compute_lstm_attribution
    attribution_df = compute_lstm_attribution(model, df, max_len, data["pad_value"])
    print(f"\nLSTM Attribution Results:")
    for _, row in attribution_df.iterrows():
        print(f"  {row['channel']:<20} {row['lstm_deep_learning']*100:5.1f}%")


if __name__ == "__main__":
    main()
