"""
Marketing Attribution Agent — Streamlit Dashboard.

5 tabs: Overview, Attribution Models, Channel Deep Dive, LSTM Deep Learning, AI Agent Chat.
"""
import sys
import os

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(
    page_title="Marketing Attribution Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached) — prefers local CSVs, falls back to BigQuery/MAM
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data(show_spinner="Loading journey data...")
def load_journey_data():
    csv_path = os.path.join(DATA_DIR, "journey_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["channel_list"] = df["channel_list"].apply(lambda x: x.split("|"))
        return df
    from attribution.data_prep import extract_journeys
    return extract_journeys()


@st.cache_data(show_spinner="Loading attribution results...")
def run_models(_df):
    csv_path = os.path.join(DATA_DIR, "attribution_results.csv")
    if os.path.exists(csv_path):
        return {"results": pd.read_csv(csv_path)}
    from attribution.models import run_all_models
    return run_all_models(_df)


@st.cache_data(show_spinner="Loading LSTM attribution results...")
def load_lstm_results():
    csv_path = os.path.join(DATA_DIR, "lstm_results.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def get_data():
    """Load all data into session state once."""
    if "journey_df" not in st.session_state:
        st.session_state.journey_df = load_journey_data()
    if "attribution_output" not in st.session_state:
        st.session_state.attribution_output = run_models(st.session_state.journey_df)
    return st.session_state.journey_df, st.session_state.attribution_output


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
def tab_overview(df):
    st.header("Journey Data Overview")
    st.caption("Source: GA4 Obfuscated Sample Ecommerce (BigQuery public dataset)")

    n_total = len(df)
    n_conv = int(df["has_conversion"].sum())
    n_non = n_total - n_conv
    conv_rate = n_conv / n_total * 100
    avg_len = df["journey_length"].mean()
    total_rev = df["conversion_value"].sum()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Journeys", f"{n_total:,}")
    c2.metric("Conversions", f"{n_conv:,}")
    c3.metric("Conversion Rate", f"{conv_rate:.1f}%")
    c4.metric("Avg Touchpoints", f"{avg_len:.1f}")
    c5.metric("Total Revenue", f"${total_rev:,.0f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Channel distribution
        all_channels = [ch for chs in df["channel_list"] for ch in chs]
        ch_counts = Counter(all_channels)
        ch_df = pd.DataFrame(ch_counts.items(), columns=["Channel", "Touchpoints"])
        ch_df = ch_df.sort_values("Touchpoints", ascending=False)
        fig = px.bar(
            ch_df, x="Channel", y="Touchpoints",
            title="Channel Frequency (all touchpoints)",
            color="Channel",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Journey length distribution
        fig = px.histogram(
            df, x="journey_length", nbins=min(20, df["journey_length"].nunique()),
            title="Journey Length Distribution",
            labels={"journey_length": "Touchpoints per Journey"},
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Conversion funnel by channel (first touch)
        first_touch = df.copy()
        first_touch["first_channel"] = first_touch["channel_list"].apply(lambda x: x[0])
        ft_stats = first_touch.groupby("first_channel").agg(
            journeys=("has_conversion", "count"),
            conversions=("has_conversion", "sum"),
        ).reset_index()
        ft_stats["conv_rate"] = ft_stats["conversions"] / ft_stats["journeys"] * 100
        ft_stats = ft_stats.sort_values("journeys", ascending=False)
        fig = px.bar(
            ft_stats, x="first_channel", y=["journeys", "conversions"],
            title="First-Touch Channel: Journeys vs Conversions",
            barmode="group",
            labels={"first_channel": "First Channel", "value": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        # Last touch channel
        last_touch = df.copy()
        last_touch["last_channel"] = last_touch["channel_list"].apply(lambda x: x[-1])
        lt_stats = last_touch.groupby("last_channel").agg(
            journeys=("has_conversion", "count"),
            conversions=("has_conversion", "sum"),
        ).reset_index()
        lt_stats["conv_rate"] = lt_stats["conversions"] / lt_stats["journeys"] * 100
        lt_stats = lt_stats.sort_values("journeys", ascending=False)
        fig = px.bar(
            lt_stats, x="last_channel", y=["journeys", "conversions"],
            title="Last-Touch Channel: Journeys vs Conversions",
            barmode="group",
            labels={"last_channel": "Last Channel", "value": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sample journeys
    with st.expander("Sample Converting Journeys"):
        sample = df[df["has_conversion"] == 1].head(10)[["journey_path", "journey_length", "conversion_value"]]
        st.dataframe(sample, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Attribution Models
# ---------------------------------------------------------------------------
def tab_attribution(output):
    st.header("Attribution Model Comparison")
    comparison = output["results"]
    model_cols = [c for c in comparison.columns if c != "channel"]

    # Heatmap
    heat_data = comparison.set_index("channel")[model_cols]
    fig = px.imshow(
        heat_data.values,
        x=model_cols,
        y=heat_data.index.tolist(),
        color_continuous_scale="Blues",
        aspect="auto",
        title="Attribution Heatmap (normalized to 100%)",
        labels={"color": "Attribution"},
    )
    fig.update_traces(text=np.round(heat_data.values * 100, 1), texttemplate="%{text}%")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Grouped bar chart
        melted = comparison.melt(id_vars="channel", value_vars=model_cols,
                                  var_name="Model", value_name="Attribution")
        melted["Attribution %"] = melted["Attribution"] * 100
        fig = px.bar(
            melted, x="channel", y="Attribution %", color="Model",
            barmode="group",
            title="Attribution by Channel & Model",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Radar chart for all channels
        try:
            fig = go.Figure()
            channels = comparison["channel"].tolist()
            for model in model_cols:
                vals = (comparison[model] * 100).tolist()
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=channels + [channels[0]],
                    name=model,
                    fill="toself",
                    opacity=0.6,
                ))
            fig.update_layout(
                title="Radar: Model Agreement per Channel",
                polar_radialaxis_visible=True,
                polar_radialaxis_suffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.subheader("Model Agreement per Channel")
            radar_df = comparison.set_index("channel")[model_cols] * 100
            radar_df = radar_df.round(1).astype(str) + "%"
            st.dataframe(radar_df, use_container_width=True)

    # Comparison table
    st.subheader("Full Results Table")
    display = comparison.copy()
    for c in model_cols:
        display[c] = display[c].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Model disagreement
    st.subheader("Model Disagreement (spread per channel)")
    spreads = []
    for _, row in comparison.iterrows():
        vals = [row[c] for c in model_cols]
        spreads.append({
            "Channel": row["channel"],
            "Min %": min(vals) * 100,
            "Max %": max(vals) * 100,
            "Spread (pp)": (max(vals) - min(vals)) * 100,
        })
    spread_df = pd.DataFrame(spreads).sort_values("Spread (pp)", ascending=False)
    fig = px.bar(
        spread_df, x="Channel", y="Spread (pp)",
        title="Model Disagreement — Higher = More Uncertainty",
        color="Spread (pp)",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Markov vs Last-click
    if "markov" in comparison.columns and "last_click" in comparison.columns:
        st.subheader("Markov vs Last-Click (GA4 Default)")
        st.caption("Positive delta = channel is UNDERVALUED by last-click attribution")
        delta_df = comparison[["channel"]].copy()
        delta_df["Markov %"] = comparison["markov"] * 100
        delta_df["Last-Click %"] = comparison["last_click"] * 100
        delta_df["Delta (pp)"] = delta_df["Markov %"] - delta_df["Last-Click %"]

        fig = px.bar(
            delta_df, x="channel", y="Delta (pp)",
            title="Markov minus Last-Click (positive = undervalued by GA4)",
            color="Delta (pp)",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Budget recommendation table
        st.subheader("Budget Reallocation Recommendation")
        budget = delta_df.copy()
        budget["Action"] = budget["Delta (pp)"].apply(
            lambda d: "INCREASE budget" if d > 2 else ("DECREASE budget" if d < -2 else "Maintain")
        )
        st.dataframe(budget, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Channel Deep Dive
# ---------------------------------------------------------------------------
def tab_channel_deep_dive(df, output):
    st.header("Channel Deep Dive")
    comparison = output["results"]
    model_cols = [c for c in comparison.columns if c != "channel"]
    channels = comparison["channel"].tolist()

    selected = st.selectbox("Select a channel", channels)
    row = comparison[comparison["channel"] == selected].iloc[0]

    vals = {m: row[m] * 100 for m in model_cols}
    max_model = max(vals, key=vals.get)
    min_model = min(vals, key=vals.get)
    spread = vals[max_model] - vals[min_model]

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest Attribution", f"{vals[max_model]:.1f}%", f"{max_model}")
    c2.metric("Lowest Attribution", f"{vals[min_model]:.1f}%", f"{min_model}")
    c3.metric("Spread", f"{spread:.1f} pp",
              "High uncertainty" if spread > 5 else "Low uncertainty")

    col_l, col_r = st.columns(2)

    with col_l:
        # Bar chart for this channel across models
        ch_df = pd.DataFrame({"Model": list(vals.keys()), "Attribution %": list(vals.values())})
        ch_df = ch_df.sort_values("Attribution %", ascending=True)
        fig = px.bar(
            ch_df, x="Attribution %", y="Model", orientation="h",
            title=f"{selected} — Attribution Across Models",
            color="Attribution %",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Pie chart — this channel's share per model
        fig = go.Figure(data=[go.Pie(
            labels=list(vals.keys()),
            values=list(vals.values()),
            hole=0.4,
        )])
        fig.update_layout(title=f"{selected} — Model Weight Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Journey-level stats for this channel
    st.subheader(f"Journey Stats for '{selected}'")
    contains_ch = df[df["channel_list"].apply(lambda x: selected in x)]
    n_journeys = len(contains_ch)
    n_conv_ch = int(contains_ch["has_conversion"].sum())
    conv_rate_ch = n_conv_ch / n_journeys * 100 if n_journeys > 0 else 0

    # Position analysis
    positions = {"First": 0, "Middle": 0, "Last": 0}
    for _, r in contains_ch.iterrows():
        chs = r["channel_list"]
        if chs[0] == selected:
            positions["First"] += 1
        if chs[-1] == selected:
            positions["Last"] += 1
        for ch in chs[1:-1]:
            if ch == selected:
                positions["Middle"] += 1
                break

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Journeys with Channel", f"{n_journeys:,}")
    c2.metric("Conversion Rate", f"{conv_rate_ch:.1f}%")
    c3.metric("As First Touch", f"{positions['First']:,}")
    c4.metric("As Last Touch", f"{positions['Last']:,}")

    pos_df = pd.DataFrame(positions.items(), columns=["Position", "Count"])
    fig = px.pie(pos_df, names="Position", values="Count",
                 title=f"'{selected}' — Position in Journey")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: LSTM Deep Learning
# ---------------------------------------------------------------------------
def tab_lstm(df):
    st.header("LSTM Deep Learning Attribution")
    st.caption("Gradient-based attribution from a trained LSTM conversion prediction model")

    lstm_df = load_lstm_results()
    if lstm_df is None:
        st.warning("No pre-computed LSTM results found. Run locally and commit `data/lstm_results.csv`.")
        st.code("python src/deep_learning/train.py\npython -c \"\nimport sys; sys.path.insert(0,'src')\nfrom attribution.data_prep import extract_journeys\nfrom deep_learning.attribution import run_lstm_attribution_pipeline\ndf = extract_journeys()\nresult = run_lstm_attribution_pipeline(df)\nresult.to_csv('data/lstm_results.csv', index=False)\n\"", language="bash")
        return

    st.session_state.lstm_attribution = lstm_df

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(
            lstm_df, x="channel", y="lstm_deep_learning",
            title="LSTM Attribution (gradient-based)",
            labels={"lstm_deep_learning": "Attribution Weight", "channel": "Channel"},
            color="lstm_deep_learning",
            color_continuous_scale="Plasma",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure(data=[go.Pie(
            labels=lstm_df["channel"].tolist(),
            values=lstm_df["lstm_deep_learning"].tolist(),
            hole=0.4,
        )])
        fig.update_layout(title="LSTM Attribution Share")
        st.plotly_chart(fig, use_container_width=True)

    # Compare LSTM with statistical models
    if "attribution_output" in st.session_state:
        comparison = st.session_state.attribution_output["results"]
        if "markov" in comparison.columns:
            st.subheader("LSTM vs Statistical Models")
            merged = comparison[["channel", "markov", "last_click"]].merge(
                lstm_df, on="channel", how="outer"
            ).fillna(0)
            melted = merged.melt(
                id_vars="channel",
                value_vars=["markov", "last_click", "lstm_deep_learning"],
                var_name="Model", value_name="Attribution",
            )
            melted["Attribution %"] = melted["Attribution"] * 100
            fig = px.bar(
                melted, x="channel", y="Attribution %", color="Model",
                barmode="group",
                title="LSTM vs Markov vs Last-Click",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Display table
    st.subheader("LSTM Attribution Values")
    display = lstm_df.copy()
    display["lstm_deep_learning"] = display["lstm_deep_learning"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.info("Method: tf.GradientTape on LSTM conversion model. "
            "Gradients computed from converting journeys only, aggregated by channel.")


# ---------------------------------------------------------------------------
# Tab 5: AI Agent Chat
# ---------------------------------------------------------------------------
def tab_agent_chat(df, output):
    st.header("AI Agent Chat")
    st.caption("Ask questions about the attribution results — powered by Gemini + LangGraph")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Set GOOGLE_API_KEY in your .env file to enable the AI agent.")
        return

    # Initialize agent
    if "agent_graph" not in st.session_state:
        with st.spinner("Initializing LangGraph agent..."):
            # Populate the tools cache so agent doesn't re-query BigQuery
            from agent.tools import _cache
            _cache["journey_df"] = df
            _cache["attribution_output"] = output
            if "lstm_attribution" in st.session_state:
                _cache["lstm_attribution"] = st.session_state.lstm_attribution

            from agent.graph import create_agent, precompute_attribution, chat as agent_chat
            graph, memory = create_agent()
            st.session_state.agent_graph = graph
            st.session_state.agent_memory = memory
            st.session_state.agent_chat_fn = agent_chat

            # Build the precomputed data string for context
            comparison = output["results"]
            model_cols = [c for c in comparison.columns if c != "channel"]
            lines = [
                f"DATA: {len(df)} journeys, {int(df['has_conversion'].sum())} conversions "
                f"({df['has_conversion'].mean()*100:.1f}%), avg {df['journey_length'].mean():.1f} touchpoints.",
                "",
                "ATTRIBUTION (normalized):",
            ]
            for _, row in comparison.iterrows():
                vals = "  ".join(f"{col}:{row[col]*100:.1f}%" for col in model_cols)
                lines.append(f"  {row['channel']}: {vals}")
            st.session_state.precomputed_str = "\n".join(lines)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested prompts
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "Analyze the attribution results and give me actionable insights",
            "Which channels are undervalued by last-click attribution?",
            "Give me a budget reallocation recommendation",
            "Compare the organic_search channel across all models",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"suggest_{i}"):
                st.session_state.pending_prompt = s
                st.rerun()

    # Check for pending prompt from suggestion buttons
    pending = st.session_state.pop("pending_prompt", None)

    prompt = st.chat_input("Ask about your attribution data...")
    if pending:
        prompt = pending

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    is_first = len(st.session_state.chat_history) == 1
                    precomp = st.session_state.precomputed_str if is_first else None
                    response = st.session_state.agent_chat_fn(
                        st.session_state.agent_graph,
                        prompt,
                        thread_id="streamlit_session",
                        precomputed_data=precomp,
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Agent error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("📊 Marketing Attribution Agent")
    st.markdown("Multi-model attribution analysis on GA4 sample ecommerce data")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This dashboard runs **7 statistical attribution models** "
            "and **1 LSTM deep learning model** on Google Analytics 4 "
            "sample ecommerce data from BigQuery."
        )
        st.markdown("**Models:**")
        st.markdown(
            "1. First-Click\n2. Last-Click\n3. Linear\n4. Time-Decay\n"
            "5. Position-Based\n6. Markov Chain\n7. Shapley Value\n8. LSTM (Deep Learning)"
        )
        st.divider()
        st.markdown("**Stack:** LangGraph + Gemini + TensorFlow + BigQuery")

        if st.button("Clear Cache & Reload"):
            st.cache_data.clear()
            for key in ["journey_df", "attribution_output", "lstm_attribution",
                        "agent_graph", "agent_memory", "chat_history"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Load data
    df, output = get_data()

    # Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "Overview",
        "Attribution Models",
        "Channel Deep Dive",
        "LSTM Deep Learning",
        "AI Agent Chat",
    ])

    with t1:
        tab_overview(df)
    with t2:
        tab_attribution(output)
    with t3:
        tab_channel_deep_dive(df, output)
    with t4:
        tab_lstm(df)
    with t5:
        tab_agent_chat(df, output)


if __name__ == "__main__":
    main()
