"""
LangChain tools for the Marketing Attribution Agent.

These are the tools the LLM can decide to call during the ReAct loop.
Each tool wraps a function from our attribution pipeline.
"""
import json
import sys
import os
from langchain_core.tools import tool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Global cache so we don't re-query BigQuery every time
_cache = {
    "journey_df": None,
    "attribution_output": None,
}


@tool
def query_bigquery_journeys() -> str:
    """
    Query BigQuery to extract user journey data from the GA4 sample dataset.
    Returns a summary of the journey data including total journeys,
    conversion rate, average journey length, and channel distribution.
    Call this FIRST before running attribution models.
    """
    from attribution.data_prep import extract_journeys

    if _cache["journey_df"] is not None:
        df = _cache["journey_df"]
        return (
            f"Journey data already loaded: {len(df)} journeys, "
            f"{df['has_conversion'].sum()} conversions "
            f"({df['has_conversion'].mean()*100:.1f}%), "
            f"avg length {df['journey_length'].mean():.1f} touchpoints. "
            f"Channels: {sorted(set(ch for chs in df['channel_list'] for ch in chs))}"
        )

    df = extract_journeys()
    _cache["journey_df"] = df

    # Channel distribution
    all_channels = [ch for chs in df["channel_list"] for ch in chs]
    from collections import Counter
    channel_counts = Counter(all_channels)
    total_touchpoints = sum(channel_counts.values())
    channel_dist = {k: f"{v/total_touchpoints*100:.1f}%" for k, v in channel_counts.most_common()}

    return (
        f"Extracted {len(df)} user journeys from GA4 sample dataset.\n"
        f"Converting journeys: {df['has_conversion'].sum()} ({df['has_conversion'].mean()*100:.1f}%)\n"
        f"Non-converting: {len(df) - df['has_conversion'].sum()}\n"
        f"Average journey length: {df['journey_length'].mean():.1f} touchpoints\n"
        f"Channel distribution (% of all touchpoints): {json.dumps(channel_dist, indent=2)}\n"
        f"Unique channels: {sorted(set(ch for chs in df['channel_list'] for ch in chs))}"
    )


@tool
def run_attribution_models() -> str:
    """
    Run all 7 attribution models on the journey data:
    first-click, last-click, linear, time-decay, position-based, Markov chain, and Shapley values.
    Returns a comparison table showing each channel's attribution weight per model.
    You MUST call query_bigquery_journeys first to load the data.
    """
    from attribution.models import run_all_models

    if _cache["journey_df"] is None:
        return "ERROR: No journey data loaded. Call query_bigquery_journeys first."

    output = run_all_models(_cache["journey_df"])
    _cache["attribution_output"] = output

    comparison = output["results"]

    # Format as readable table
    lines = ["Attribution Model Comparison (normalized to 100%):\n"]
    lines.append(f"{'Channel':<20} " + " ".join(f"{col:>14}" for col in comparison.columns if col != "channel"))
    lines.append("-" * 120)

    for _, row in comparison.iterrows():
        vals = " ".join(
            f"{row[col]*100:13.1f}%" for col in comparison.columns if col != "channel"
        )
        lines.append(f"{row['channel']:<20} {vals}")

    # Add Markov vs Last-click insight
    if "markov" in comparison.columns and "last_click" in comparison.columns:
        lines.append("\nKey Insight — Markov vs Last-Click:")
        for _, row in comparison.iterrows():
            delta = (row["markov"] - row["last_click"]) * 100
            if abs(delta) > 1:
                direction = "UNDERVALUED" if delta > 0 else "OVERVALUED"
                lines.append(
                    f"  {row['channel']}: {direction} by last-click by {abs(delta):.1f} percentage points"
                )

    # Add transition matrix info
    if output.get("markov_transition_matrix") is not None:
        tm = output["markov_transition_matrix"]
        lines.append("\nMarkov Removal Effects (higher = more important):")
        if hasattr(tm, "iterrows"):
            for _, row in tm.iterrows():
                for col in tm.columns:
                    lines.append(f"  {row.name}: {row[col]:.3f}")
        else:
            lines.append(f"  {tm}")

    return "\n".join(lines)


@tool
def compare_channels(channel_name: str) -> str:
    """
    Get detailed attribution analysis for a specific channel.
    Shows how all 7 models rate this channel, where models agree/disagree,
    and a budget recommendation.
    Args:
        channel_name: The channel to analyze (e.g. 'organic_search', 'paid_search', 'direct', 'referral', 'other')
    """
    if _cache["attribution_output"] is None:
        return "ERROR: No attribution results. Call run_attribution_models first."

    comparison = _cache["attribution_output"]["results"]
    channel_name_lower = channel_name.lower().strip()

    # Find the channel row
    match = comparison[comparison["channel"].str.lower() == channel_name_lower]
    if match.empty:
        available = ", ".join(comparison["channel"].tolist())
        return f"Channel '{channel_name}' not found. Available channels: {available}"

    row = match.iloc[0]
    numeric_cols = [c for c in comparison.columns if c != "channel"]
    values = {col: row[col] for col in numeric_cols}

    max_model = max(values, key=values.get)
    min_model = min(values, key=values.get)
    spread = (values[max_model] - values[min_model]) * 100

    lines = [
        f"Detailed Analysis for: {row['channel']}",
        f"{'=' * 50}",
    ]

    for model, val in values.items():
        bar = "█" * int(val * 100)
        lines.append(f"  {model:<18} {val*100:5.1f}%  {bar}")

    lines.append(f"\nModel Agreement:")
    lines.append(f"  Highest rating: {max_model} ({values[max_model]*100:.1f}%)")
    lines.append(f"  Lowest rating:  {min_model} ({values[min_model]*100:.1f}%)")
    lines.append(f"  Spread: {spread:.1f} percentage points")

    if spread > 5:
        lines.append(f"\n⚠️ HIGH UNCERTAINTY — models disagree significantly on this channel.")
        lines.append(f"  This suggests {row['channel']} plays different roles at different")
        lines.append(f"  journey stages (e.g., strong at initiating but weak at closing).")
    else:
        lines.append(f"\n✅ LOW UNCERTAINTY — models mostly agree on this channel's value.")

    return "\n".join(lines)


@tool
def get_budget_recommendation() -> str:
    """
    Generate a budget reallocation recommendation based on Markov chain attribution.
    Compares data-driven (Markov) attribution with the GA4 default (last-click)
    to identify channels that are over-funded or under-funded.
    Call run_attribution_models first.
    """
    if _cache["attribution_output"] is None:
        return "ERROR: No attribution results. Call run_attribution_models first."

    comparison = _cache["attribution_output"]["results"]

    if "markov" not in comparison.columns or "last_click" not in comparison.columns:
        return "ERROR: Markov or Last-click results missing."

    lines = [
        "BUDGET REALLOCATION RECOMMENDATION",
        "Based on Markov chain (data-driven) vs Last-click (GA4 default)",
        "=" * 60,
        "",
        f"{'Channel':<20} {'Last-Click':>12} {'Markov':>12} {'Action':>20} {'Delta':>10}",
        "-" * 76,
    ]

    for _, row in comparison.iterrows():
        lc = row["last_click"] * 100
        mk = row["markov"] * 100
        delta = mk - lc

        if delta > 2:
            action = "↑ INCREASE BUDGET"
        elif delta < -2:
            action = "↓ DECREASE BUDGET"
        else:
            action = "→ MAINTAIN"

        lines.append(f"{row['channel']:<20} {lc:11.1f}% {mk:11.1f}% {action:>20} {delta:+9.1f}%")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("• Channels where Markov > Last-click are being UNDERVALUED — they assist")
    lines.append("  conversions earlier in the journey but don't get credit at the end.")
    lines.append("• Channels where Markov < Last-click are being OVERVALUED — they happen")
    lines.append("  to be the last touch before purchase but aren't truly driving conversions.")
    lines.append("")
    lines.append("⚠️ This is based on the GA4 sample dataset (demo data).")
    lines.append("For production use, connect to your actual GA4 property.")

    return "\n".join(lines)


# List of all tools for the agent
ALL_TOOLS = [
    query_bigquery_journeys,
    run_attribution_models,
    compare_channels,
    get_budget_recommendation,
]
