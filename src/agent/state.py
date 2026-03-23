"""
Agent state schema for the Marketing Attribution Agent.

Defines the shared state that flows through every node in the LangGraph graph.
"""
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages


class AttributionState(TypedDict):
    """State that persists across all agent nodes."""

    # Conversation history — LangGraph's add_messages reducer appends new messages
    messages: Annotated[list, add_messages]

    # Data loaded from BigQuery (serialized as dict for state compatibility)
    journey_data_loaded: bool

    # Attribution results from all models (serialized as dict)
    attribution_results: Optional[dict]

    # Whether models have been run in this conversation
    models_run: bool
