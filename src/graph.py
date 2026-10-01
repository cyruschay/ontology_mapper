"""Graph wiring.

The shape of this graph is the point: the ontology preference order is an edge, the
tool budget is state, human approval is a resumable interrupt, and network retry is
declared on the one node that touches the network. None of that depends on the model
behaving well.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from . import nodes
from .ols import OlsError
from .state import MapperState
from .tools import TOOLS

# Only the retrieval node retries: OLS intermittently 500s, while a retried LLM node
# would silently re-bill.
OLS_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0,
                        retry_on=(OlsError,))


def _tool_error(exc):
    """Hand a failed lookup back to the model as text instead of ending the run."""
    return (f"That lookup failed ({type(exc).__name__}: {exc}). "
            "Decide from the candidates you already have, or try a different lookup.")


def build(checkpointer=None):
    """Compile the agent. Pass a checkpointer to make interrupts resumable."""
    graph = StateGraph(MapperState)

    graph.add_node("normalize", nodes.normalize)
    graph.add_node("alias_lookup", nodes.alias_lookup)
    graph.add_node("plan_tiers", nodes.plan_tiers)
    graph.add_node("next_tier", nodes.next_tier)
    graph.add_node("retrieve", nodes.retrieve, retry=OLS_RETRY)
    graph.add_node("adjudicate", nodes.adjudicate)
    # Prebuilt executor: it matches tool_calls to tools, injects graph state via
    # ToolRuntime, applies Command updates, and turns a tool exception into a
    # ToolMessage the model can read and recover from.
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=_tool_error))
    graph.add_node("decide", nodes.decide)
    graph.add_node("widen", nodes.widen, retry=OLS_RETRY)
    graph.add_node("verify", nodes.verify)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("persist", nodes.persist)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "alias_lookup")
    graph.add_conditional_edges("alias_lookup", nodes.route_after_alias,
                                {"persist": "persist", "plan_tiers": "plan_tiers"})
    graph.add_edge("plan_tiers", "retrieve")
    graph.add_edge("retrieve", "adjudicate")
    graph.add_conditional_edges("adjudicate", nodes.route_tools,
                                {"tools": "tools", "decide": "decide"})
    graph.add_edge("tools", "adjudicate")
    graph.add_conditional_edges("decide", nodes.route_after_decide,
                                {"next_tier": "next_tier", "widen": "widen",
                                 "verify": "verify"})
    graph.add_edge("widen", "decide")
    graph.add_edge("next_tier", "retrieve")
    graph.add_conditional_edges("verify", nodes.route_after_verify,
                                {"human_review": "human_review", "persist": "persist"})
    graph.add_edge("human_review", "persist")
    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
