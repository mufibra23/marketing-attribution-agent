"""
Attribution Model Runner — runs 6 rule-based + data-driven models on user journey data.

Uses DP6's marketing_attribution_models (MAM) library.
"""
import pandas as pd
import numpy as np
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GCP_PROJECT


def build_mam_input(df):
    """Convert journey DataFrame to MAM's path-based input format."""
    mam_df = pd.DataFrame({
        "path": df["journey_path"],
        "conversion": df["has_conversion"].astype(bool),
    })
    return mam_df


def run_all_models(df):
    """
    Run all 6 attribution models + Shapley on journey data.

    Returns dict with 'results' (comparison DataFrame), 'markov_transition_matrix', etc.
    """
    from marketing_attribution_models import MAM

    print("Building MAM input data...")
    mam_input = build_mam_input(df)

    n_channels = len(set(
        ch.strip()
        for path in mam_input["path"]
        for ch in path.split(" > ")
    ))
    print(f"MAM input: {len(mam_input)} journeys, {n_channels} channels")

    # Create MAM object
    print("\nInitializing MAM attribution model...")
    attributions = MAM(
        mam_input,
        group_channels=False,
        channels_colname="path",
        journey_with_conv_colname="conversion",
        path_separator=" > ",
    )

    # --- Run all models ---
    print("\n[1/7] Running First-click attribution...")
    first_click_raw = attributions.attribution_first_click()

    print("[2/7] Running Last-click attribution...")
    last_click_raw = attributions.attribution_last_click()

    print("[3/7] Running Linear attribution...")
    linear_raw = attributions.attribution_linear()

    print("[4/7] Running Time-decay attribution...")
    time_decay_raw = attributions.attribution_time_decay()

    print("[5/7] Running Position-based attribution...")
    position_raw = attributions.attribution_position_based()

    print("[6/7] Running Markov chain attribution (this takes a moment)...")
    markov_raw = attributions.attribution_markov()

    print("[7/7] Running Shapley value attribution...")
    try:
        shapley_raw = attributions.attribution_shapley(size=4, order=False)
        has_shapley = True
    except Exception as e:
        print(f"  Shapley failed: {e}")
        shapley_raw = None
        has_shapley = False

    # --- Debug: inspect what MAM actually returned ---
    print("\n  DEBUG — Inspecting MAM output structure:")
    sample = first_click_raw[0]
    print(f"  Type: {type(sample)}")
    if isinstance(sample, pd.DataFrame):
        print(f"  Columns: {list(sample.columns)}")
        print(f"  Dtypes:\n{sample.dtypes}")
        print(f"  Head:\n{sample.head()}")
    elif isinstance(sample, pd.Series):
        print(f"  Index: {list(sample.index[:5])}")
        print(f"  Values (first 3): {list(sample.values[:3])}")
        print(f"  Value type: {type(sample.values[0])}")

    # --- Extract channel-level attribution from each model ---
    transition_matrix = markov_raw[3] if len(markov_raw) > 3 else None

    # Build comparison table manually based on what we see
    comparison = extract_all_attributions(
        first_click_raw, last_click_raw, linear_raw,
        time_decay_raw, position_raw, markov_raw,
        shapley_raw if has_shapley else None
    )

    print("\n[OK] All models complete!")

    return {
        "results": comparison,
        "markov_transition_matrix": transition_matrix,
        "model_names": list(comparison.columns[1:]),  # all except 'channel'
    }


def extract_attribution_from_result(raw_tuple, model_name):
    """
    Extract per-channel attribution from a MAM result tuple.

    MAM returns tuples where:
      [0] = per-journey results (one row per journey — NOT what we want)
      [1] = per-channel aggregated attribution (one row per channel)
    For rule-based models, [1] is a Series with channel names as index.
    For Markov, [1] is a DataFrame with 'channels' and attribution columns.
    """
    result = raw_tuple[1]

    if isinstance(result, pd.Series):
        return pd.DataFrame({
            "channel": [str(x) for x in result.index],
            model_name: [float(x) for x in result.values],
        })

    if isinstance(result, pd.DataFrame):
        # Find channel column (usually 'channels')
        channel_col = result.columns[0]

        # Find the best numeric value column
        value_col = None
        for col in result.columns[1:]:
            sample_val = result[col].iloc[0]
            if isinstance(sample_val, (int, float, np.integer, np.floating)):
                if value_col is None or "attribution" in str(col).lower():
                    value_col = col

        if value_col is None:
            print(f"  Warning: No scalar numeric column found for {model_name}")
            return None

        out = result[[channel_col, value_col]].copy()
        out.columns = ["channel", model_name]
        out["channel"] = out["channel"].astype(str)
        out[model_name] = pd.to_numeric(out[model_name], errors="coerce").fillna(0)
        return out

    return None


def extract_all_attributions(first_click, last_click, linear, time_decay,
                              position, markov, shapley):
    """Combine all model results into one comparison table."""
    models = {
        "first_click": first_click,
        "last_click": last_click,
        "linear": linear,
        "time_decay": time_decay,
        "position_based": position,
        "markov": markov,
    }

    comparison = None

    for model_name, raw_tuple in models.items():
        model_data = extract_attribution_from_result(raw_tuple, model_name)
        if model_data is None:
            continue

        # Normalize to proportions
        total = model_data[model_name].sum()
        if total > 0:
            model_data[model_name] = model_data[model_name] / total

        if comparison is None:
            comparison = model_data
        else:
            comparison = comparison.merge(model_data, on="channel", how="outer")

    # Handle Shapley separately (different structure)
    if shapley is not None:
        try:
            shap_df = shapley[0]
            if isinstance(shap_df, pd.DataFrame) and "combinations" in shap_df.columns:
                # Filter single-channel rows only
                single = shap_df[~shap_df["combinations"].str.contains(">", na=False)].copy()
                if len(single) > 0:
                    shap_cols = [c for c in single.columns if "shapley" in c.lower()]
                    if shap_cols:
                        shap_data = single[["combinations", shap_cols[0]]].copy()
                        shap_data.columns = ["channel", "shapley"]
                        shap_data["channel"] = shap_data["channel"].str.strip()
                        shap_data["shapley"] = pd.to_numeric(shap_data["shapley"], errors="coerce").fillna(0)
                        total = shap_data["shapley"].sum()
                        if total > 0:
                            shap_data["shapley"] = shap_data["shapley"] / total
                        if comparison is not None:
                            comparison = comparison.merge(shap_data, on="channel", how="outer")
        except Exception as e:
            print(f"  Shapley extraction failed: {e}")

    if comparison is not None:
        comparison = comparison.fillna(0)
        sort_col = "markov" if "markov" in comparison.columns else comparison.columns[1]
        comparison = comparison.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return comparison


def print_comparison_table(comparison):
    """Pretty print the attribution comparison table."""
    if comparison is None or comparison.empty:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("ATTRIBUTION MODEL COMPARISON (normalized to 100%)")
    print("=" * 100)

    display_df = comparison.copy()
    for col in display_df.columns:
        if col != "channel":
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%")

    print(display_df.to_string(index=False))

    # Model disagreement
    print("\n" + "-" * 100)
    print("MODEL DISAGREEMENT (max - min attribution per channel)")
    print("-" * 100)

    numeric_cols = [c for c in comparison.columns if c != "channel"]
    for _, row in comparison.iterrows():
        values = [row[c] for c in numeric_cols if pd.notna(row[c])]
        if len(values) > 0:
            spread = max(values) - min(values)
            bar = "█" * int(spread * 200)
            print(f"  {row['channel']:20s} spread: {spread*100:5.1f}%  {bar}")

    # Key insight: Markov vs Last-click
    if "markov" in comparison.columns and "last_click" in comparison.columns:
        print("\n" + "-" * 100)
        print("KEY INSIGHT: Markov vs Last-Click (where GA4 default gets it wrong)")
        print("-" * 100)
        for _, row in comparison.iterrows():
            delta = (row["markov"] - row["last_click"]) * 100
            direction = "▲ UNDERVALUED by last-click" if delta > 1 else (
                "▼ OVERVALUED by last-click" if delta < -1 else "≈ Similar"
            )
            print(f"  {row['channel']:20s} Markov: {row['markov']*100:5.1f}%  "
                  f"Last-click: {row['last_click']*100:5.1f}%  "
                  f"Delta: {delta:+5.1f}%  {direction}")


if __name__ == "__main__":
    print("=" * 60)
    print("Marketing Attribution Agent - Model Runner Test")
    print("=" * 60)

    from data_prep import extract_journeys

    df = extract_journeys()
    output = run_all_models(df)
    print_comparison_table(output["results"])

    if output["markov_transition_matrix"] is not None:
        print("\n--- Markov Transition Matrix ---")
        print(output["markov_transition_matrix"].round(3))

    print(f"\n[OK] {len(output['model_names'])} models ran successfully: {output['model_names']}")
