"""
LangGraph Agent Graph — Marketing Attribution Agent.

Smart architecture: pre-computes BigQuery data + attribution models (no LLM needed),
then uses a single LLM call for analysis. Follow-up questions use lightweight tools.
"""
import os
import sys
import time
import re

# Fix Windows console encoding for unicode characters
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GOOGLE_API_KEY
from agent.state import AttributionState
from agent.tools import ALL_TOOLS, _cache

SYSTEM_PROMPT = """You are a Marketing Attribution Analyst agent. You help marketers understand which marketing channels truly drive conversions by comparing multiple attribution models (7 statistical + 1 LSTM deep learning).

You have access to these tools for follow-up questions:
1. compare_channels — Deep dive into a specific channel's attribution across all models
2. get_budget_recommendation — Generate budget reallocation advice
3. run_lstm_attribution — Run LSTM deep learning attribution (gradient-based)

When analyzing attribution data provided to you:
- Explain results in business terms, not technical jargon
- Always give specific, actionable recommendations with percentages
- Flag channels where models disagree strongly — this signals uncertainty
- When comparing Markov vs last-click, explain WHY they differ: Markov captures the full journey, while last-click only credits the final touchpoint

IMPORTANT: This uses the GA4 sample ecommerce dataset (demo data from Google). Always mention this.

Be concise but insightful. Lead with the business insight, then support with data."""

FOLLOWUP_TOOLS = [t for t in ALL_TOOLS if t.name in ("compare_channels", "get_budget_recommendation", "run_lstm_attribution")]


def precompute_attribution():
    """Run BigQuery extraction + all 7 attribution models. No LLM needed."""
    from attribution.data_prep import extract_journeys
    from attribution.models import run_all_models

    if _cache.get("attribution_output") is not None:
        print("  Using cached attribution results.")
        output = _cache["attribution_output"]
        df = _cache["journey_df"]
    else:
        print("  Loading journey data from BigQuery...")
        df = extract_journeys()
        _cache["journey_df"] = df

        print("  Running 7 attribution models...")
        output = run_all_models(df)
        _cache["attribution_output"] = output

    comparison = output["results"]

    lines = [
        f"DATA LOADED: {len(df)} user journeys, {df['has_conversion'].sum()} conversions "
        f"({df['has_conversion'].mean()*100:.1f}%), avg {df['journey_length'].mean():.1f} touchpoints.",
        f"Channels: {sorted(set(ch for chs in df['channel_list'] for ch in chs))}",
        "",
        "ATTRIBUTION RESULTS (normalized to 100%):",
        f"{'Channel':<20} {'1st-click':>10} {'Last-click':>11} {'Linear':>8} {'Time-dec':>9} {'Position':>9} {'Markov':>8} {'Shapley':>9}",
        "-" * 95,
    ]

    model_cols = [c for c in comparison.columns if c != "channel"]
    for _, row in comparison.iterrows():
        vals = " ".join(f"{row[c]*100:9.1f}%" for c in model_cols)
        lines.append(f"{row['channel']:<20} {vals}")

    if "markov" in comparison.columns and "last_click" in comparison.columns:
        lines.append("")
        lines.append("MARKOV vs LAST-CLICK DELTAS:")
        for _, row in comparison.iterrows():
            delta = (row["markov"] - row["last_click"]) * 100
            if abs(delta) > 0.5:
                direction = "UNDERVALUED" if delta > 0 else "OVERVALUED"
                lines.append(f"  {row['channel']}: {direction} by last-click by {abs(delta):.1f}pp")

    if output.get("markov_transition_matrix") is not None:
        tm = output["markov_transition_matrix"]
        lines.append("")
        lines.append("MARKOV REMOVAL EFFECTS (higher = removing channel hurts conversions more):")
        if hasattr(tm, "iterrows"):
            for idx, row in tm.iterrows():
                for col in tm.columns:
                    lines.append(f"  {idx}: {row[col]:.3f}")

    # LSTM deep learning attribution
    try:
        from deep_learning.attribution import run_lstm_attribution_pipeline
        print("  Running LSTM deep learning attribution...")
        lstm_df = run_lstm_attribution_pipeline(df)
        _cache["lstm_attribution"] = lstm_df

        lines.append("")
        lines.append("LSTM DEEP LEARNING ATTRIBUTION (gradient-based, from converting journeys):")
        for _, row in lstm_df.iterrows():
            lines.append(f"  {row['channel']:<20} {row['lstm_deep_learning']*100:5.1f}%")
    except FileNotFoundError:
        lines.append("")
        lines.append("LSTM MODEL: Not trained yet. Run 'python src/deep_learning/train.py' to enable.")
    except Exception as e:
        lines.append("")
        lines.append(f"LSTM MODEL: Error — {e}")

    return "\n".join(lines)


def create_agent():
    """Build and return the LangGraph agent graph with memory."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(FOLLOWUP_TOOLS)

    def agent_node(state: AttributionState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = 30
                    match = re.search(r"retry in (\d+\.?\d*)", error_str.lower())
                    if match:
                        wait_time = max(float(match.group(1)) + 5, 15)
                    if attempt < max_retries - 1:
                        print(f"\n  Rate limited. Waiting {wait_time:.0f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                raise

    tool_node = ToolNode(FOLLOWUP_TOOLS)

    def should_continue(state: AttributionState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AttributionState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    return compiled, memory


def chat(graph, user_message: str, thread_id: str = "default", precomputed_data: str = None):
    config = {"configurable": {"thread_id": thread_id}}

    if precomputed_data:
        enriched_message = (
            f"{user_message}\n\n"
            f"--- PRE-COMPUTED ATTRIBUTION DATA ---\n"
            f"{precomputed_data}\n"
            f"--- END DATA ---\n\n"
            f"Please analyze this attribution data and provide actionable insights."
        )
    else:
        enriched_message = user_message

    result = graph.invoke(
        {"messages": [HumanMessage(content=enriched_message)]},
        config=config,
    )

    last_message = result["messages"][-1]
    return last_message.content


if __name__ == "__main__":
    print("=" * 60)
    print("Marketing Attribution Agent — Interactive Chat")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    print("Pre-computing attribution data...\n")
    precomputed = precompute_attribution()
    print("\nData ready. Starting agent.\n")

    agent_graph, memory = create_agent()
    thread_id = "test_session_1"
    first_message = True

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nAgent thinking...\n")
        try:
            if first_message:
                response = chat(agent_graph, user_input, thread_id, precomputed_data=precomputed)
                first_message = False
            else:
                response = chat(agent_graph, user_input, thread_id)
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
