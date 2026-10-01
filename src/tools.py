"""Traversal tools exposed to the adjudicating model.

Real LangChain tools, bound with bind_tools and executed by langgraph's ToolNode,
rather than a hand-rolled request/dispatch loop. ToolRuntime injects the graph
state, so a tool resolves the current ontology and the candidate IRIs itself.
"""

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from . import ols


def _context(runtime):
    """Current ontology and everything selectable in this tier."""
    state = runtime.state
    ontology = state["ontologies"][state["tier"]]
    return state, ontology, _selectable(state)


def _selectable(state):
    """Search hits plus anything traversal has surfaced so far."""
    return list(state.get("candidates") or []) + list(state.get("discovered") or [])


def _find(candidates, curie):
    return next((c for c in candidates if c["curie"] == curie), None)


def _record(found, runtime, text):
    """Emit the tool result and add the traversed entries to the selectable set."""
    return Command(update={
        "discovered": [c.model_dump() for c in found],
        "messages": [ToolMessage(text, tool_call_id=runtime.tool_call_id)],
    })


@tool
def list_parents(curie: str, tool_runtime: ToolRuntime) -> Command:
    """List the direct parents (broader concepts) of an entry, by its CURIE.

    Use this to check whether an entry is narrower than the query term. Any parent
    returned becomes selectable as your final answer.
    """
    _, ontology, candidates = _context(tool_runtime)
    match = _find(candidates, curie)
    if not match:
        return _record([], tool_runtime, f"{curie} is not an entry you have seen.")
    found = ols.list_parents(match.get("ontology") or ontology, match["iri"])
    text = (f"parents of {curie} (selectable): "
            + (", ".join(f"{c.curie} {c.label}" for c in found) or "(none)"))
    return _record(found, tool_runtime, text)


@tool
def list_children(curie: str, tool_runtime: ToolRuntime) -> Command:
    """List the direct children (narrower concepts) of an entry, by its CURIE.

    Use this to check whether an entry is broader than the query term. Any child
    returned becomes selectable as your final answer.
    """
    _, ontology, candidates = _context(tool_runtime)
    match = _find(candidates, curie)
    if not match:
        return _record([], tool_runtime, f"{curie} is not an entry you have seen.")
    found = ols.list_children(match.get("ontology") or ontology, match["iri"])[:15]
    text = (f"children of {curie} (selectable): "
            + (", ".join(f"{c.curie} {c.label}" for c in found) or "(none)"))
    return _record(found, tool_runtime, text)


@tool
def get_entity(curie: str, tool_runtime: ToolRuntime) -> str:
    """Fetch the full label, synonyms and definition of a candidate, by its CURIE.

    Use this when the candidate list is too terse to judge an exact synonym match.
    """
    _, ontology, candidates = _context(tool_runtime)
    match = _find(candidates, curie)
    if not match:
        return f"{curie} is not an entry you have seen."
    entity = ols.get_entity(match.get("ontology") or ontology, match["iri"])
    return (f"{curie}: label={entity['label']!r} synonyms={entity['synonyms'][:8]} "
            f"definition={(entity['definition'] or '')[:300]!r}")


@tool
def next_page(tool_runtime: ToolRuntime) -> Command:
    """Retrieve the next page of candidates from the current ontology.

    Use this when none of the candidates shown is a plausible match but more may exist.
    """
    from .nodes import retrieve  # local import: nodes imports this module

    state, ontology, _ = _context(tool_runtime)
    page = state.get("page", 0) + 1
    try:
        fresh = retrieve({**state, "page": page})
    except ols.OlsError as exc:
        return Command(update={"messages": [ToolMessage(
            f"next_page failed, keeping the current candidates: {exc}",
            tool_call_id=tool_runtime.tool_call_id)]})

    listed = "\n".join(f"- {c['curie']} | {c['label']}" for c in fresh["candidates"])
    return Command(update={
        "page": page,
        "candidates": fresh["candidates"],
        "retrieval_errors": fresh["retrieval_errors"],
        "messages": [ToolMessage(
            f"page {page} of {ontology}:\n{listed or '(no further candidates)'}",
            tool_call_id=tool_runtime.tool_call_id)],
    })


TOOLS = [list_parents, list_children, get_entity, next_page]
