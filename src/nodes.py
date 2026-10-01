"""One function per graph node. Each returns a state delta, never mutates state."""

from langchain_core.messages import (HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import tools_condition
from langgraph.types import interrupt

from . import ols, tables, tiers
from .llm import MODEL, get_llm
from .state import Candidate, MatchDecision, Verification
from .tools import TOOLS

TOOL_BUDGET = 6          # hard ceiling on traversal calls per run, enforced in the graph
CONFIDENCE_FLOOR = 0.7   # below this, a human decides
ROWS = 10                # candidates per source per tier

OUTCOME = {
    "exact": "match", "semantic": "match",
    "parent": "close", "child": "close",
    "none": "no match",
}


# --------------------------------------------------------------------------- input

def normalize(state):
    """Collapse whitespace so the alias table keys on a stable string."""
    return {"normalized_term": " ".join(state["term"].split())}


def alias_lookup(state):
    """Exact hit against past adjudicated decisions short-circuits the whole LLM path."""
    hit = tables.lookup_alias(state["normalized_term"], state["target_ontology_class"])
    if not hit:
        return {"annotation": "no alias hit"}
    return {
        "decision": {
            "relation": hit["flag"], "curie": hit["term_id"],
            "label": hit["standardized_term"], "confidence": 1.0,
            "reasoning": f"alias table hit, originally decided by {hit['decided_by']}",
            "ontology": hit["ontology"],
        },
        "flag": hit["flag"],
        "outcome": OUTCOME.get(hit["flag"], "match"),
        "decided_by": "alias",
        "annotation": f"alias table hit from {hit['time']}",
    }


def plan_tiers(state):
    """Resolve the entity class to its deterministic ontology order."""
    return {
        "ontologies": tiers.ontologies_for(state["target_ontology_class"]),
        "tier": 0, "page": 0, "retrieval_errors": [], "messages": [],
        "candidates": [], "discovered": None, "widened": False,
    }


def next_tier(state):
    """Escalate to the next ontology, discarding the exhausted tier's candidates.

    The message history is cleared too: lookups against the previous ontology would
    only mislead the model about which candidates are in scope.
    """
    return {"tier": state["tier"] + 1, "page": 0, "candidates": [], "discovered": None,
            "widened": False, "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


# ----------------------------------------------------------------------- retrieval

def retrieve(state):
    """Hybrid retrieval: lexical always, embedding where the ontology supports it.

    Lexical is the load-bearing path (it works for every ontology). Embedding is
    best-effort: the endpoint is intermittently unavailable and absent for some
    ontologies, so its failure degrades the run rather than ending it.
    """
    ontology = state["ontologies"][state["tier"]]
    page = state.get("page", 0)
    errors = list(state.get("retrieval_errors", []))

    lexical = ols.search_lexical(state["normalized_term"], ontology,
                                 rows=ROWS, start=page * ROWS)

    embedded = []
    if tiers.use_embeddings(ontology):
        try:
            embedded = ols.search_embedding(state["normalized_term"], ontology,
                                            size=ROWS, page=page)
        except ols.OlsError as exc:
            errors.append(f"embedding {ontology}: {exc}")

    return {"candidates": _merge(lexical, embedded), "retrieval_errors": errors}


def _merge(lexical, embedded):
    """Union by CURIE, keeping lexical order first and tagging overlap as 'both'."""
    merged = {}
    for candidate in lexical:
        if candidate.curie:
            merged[candidate.curie] = candidate
    for candidate in embedded:
        if not candidate.curie:
            continue
        if candidate.curie in merged:
            merged[candidate.curie].source = "both"
            merged[candidate.curie].score = candidate.score
        else:
            merged[candidate.curie] = candidate
    return [c.model_dump() for c in merged.values()]


# -------------------------------------------------------------------- adjudication

def _render(candidates):
    lines = []
    for c in candidates:
        parts = [f"- {c['curie']} | {c['label']}"]
        if c.get("synonyms"):
            parts.append(f"  synonyms: {', '.join(c['synonyms'][:6])}")
        if c.get("definition"):
            parts.append(f"  definition: {c['definition'][:240]}")
        parts.append(f"  retrieved_by: {c['source']}")
        lines.append("\n".join(parts))
    return "\n".join(lines) if lines else "(no candidates)"


# One glossary, shared by the adjudicator and the reviewer. They previously carried
# separate wordings and drifted on direction: a reviewer called an entry "parent"
# while its own prose said the entry was more specific, which is 'child'. A false
# disagreement like that corrupts the approval signal, so the definition lives once.
RELATIONS = (
    "A relation describes the ENTRY relative to the QUERY TERM:\n"
    "  exact    - the entry's label or one of its synonyms IS the query term\n"
    "  semantic - a different label denoting the same concept\n"
    "  parent   - the entry is BROADER / more general than the query term\n"
    "  child    - the entry is NARROWER / more specific than the query term\n"
    "  none     - nothing offered denotes the query term"
)

SYSTEM = (
    "You map a biomedical term onto an entry in one ontology.\n"
    f"{RELATIONS}\n"
    "Use the lookup tools whenever the candidate list is too terse to judge a "
    "hierarchy relation or an exact synonym. Entries returned by a lookup are "
    "selectable too, even when they belong to a different ontology than the one "
    "being searched - if the best answer is a parent you discovered, choose it. "
    "Never name an identifier you have not actually seen."
)


def _question(state):
    ontology = state["ontologies"][state["tier"]]
    return (f"QUERY TERM: {state['normalized_term']}\n"
            f"CONTEXT: {state.get('optional_context') or '(none)'}\n"
            f"ENTITY CLASS: {state['target_ontology_class']}\n"
            f"ONTOLOGY: {ontology}\n\n"
            f"CANDIDATES:\n{_render(state.get('candidates', []))}")


def adjudicate(state):
    """Let the model look things up. Tool calls are real, bound tool calls."""
    messages = state.get("messages") or [SystemMessage(SYSTEM), HumanMessage(_question(state))]
    reply = get_llm().bind_tools(TOOLS).invoke(messages)
    # Seed the history on the first pass so ToolNode and the next turn see the prompt.
    new = ([SystemMessage(SYSTEM), HumanMessage(_question(state)), reply]
           if not state.get("messages") else [reply])
    return {"messages": new}


def _seen(state):
    """Every entry the model is allowed to choose, keyed by CURIE."""
    entries = list(state.get("candidates") or []) + list(state.get("discovered") or [])
    return {e["curie"]: e for e in entries if e.get("curie")}


def decide(state):
    """Collapse the finished conversation into one validated verdict.

    The chosen CURIE is checked against what was actually retrieved or traversed, so
    a hallucinated identifier becomes a 'none' with an explanation rather than a row
    in the decision table. The ontology is taken from the chosen entry, which may
    differ from the tier being searched when the answer came from a traversal.
    """
    llm = get_llm().with_structured_output(MatchDecision)
    decision = llm.invoke(list(state["messages"]) + [HumanMessage(
        "Give your final verdict on the query term now, based only on the entries "
        "you have seen - candidates or lookup results.")])

    verdict = decision.model_dump()
    seen = _seen(state)
    tier_ontology = state["ontologies"][state["tier"]]

    if verdict["relation"] == "none" or not verdict.get("curie"):
        verdict["ontology"] = tier_ontology
        return {"decision": verdict, "decided_by": MODEL}

    chosen = seen.get(verdict["curie"])
    if not chosen:
        return {
            "decision": {**verdict, "relation": "none", "curie": None, "label": None,
                         "ontology": tier_ontology, "confidence": 0.0},
            "decided_by": MODEL,
            "annotation": f"rejected unseen identifier {verdict['curie']}",
        }

    verdict["label"] = chosen.get("label") or verdict.get("label")
    verdict["ontology"] = chosen.get("ontology") or ols.ontology_of(verdict["curie"])
    return {"decision": verdict, "decided_by": MODEL}


def widen(state):
    """Traverse up from a 'child' verdict, once, before accepting it.

    A 'child' verdict means the chosen entry is narrower than the query, so the term
    the query actually denotes may be its parent - and that parent is often in another
    ontology (PR:000050032 'immunoglobulin complex (human)' has GO:0019814
    'immunoglobulin complex' as its parent, an exact match for the bare query). Those
    parents are added to the selectable set and the verdict is taken again.
    """
    decision = state["decision"]
    chosen = _seen(state).get(decision.get("curie") or "")
    if not chosen:
        return {"widened": True}

    try:
        parents = ols.list_parents(chosen.get("ontology") or decision["ontology"],
                                   chosen["iri"])
    except ols.OlsError as exc:
        return {"widened": True,
                "messages": [HumanMessage(f"(could not widen: {exc})")]}

    if not parents:
        return {"widened": True}

    listed = "\n".join(
        f"- {p.curie} | {p.label}" + (f" | synonyms: {', '.join(p.synonyms[:5])}"
                                      if p.synonyms else "")
        for p in parents)
    return {
        "widened": True,
        "discovered": [p.model_dump() for p in parents],
        "messages": [HumanMessage(
            f"You judged {decision['curie']} to be narrower than "
            f"{state['normalized_term']!r}. These are its direct parents, and they "
            f"are selectable:\n{listed}\n"
            "If one of them denotes the query term exactly or semantically, choose it "
            "instead. Otherwise keep your original answer.")],
    }


# -------------------------------------------------------------------- verification

def verify(state):
    """An independent second opinion that never sees the first reasoning."""
    decision = state["decision"]
    if decision["relation"] == "none" or not decision.get("curie"):
        return {"verification": {"confirmed": True, "relation": "none",
                                 "issue": None}}

    match = _seen(state).get(decision["curie"])
    ontology = decision.get("ontology") or state["ontologies"][state["tier"]]

    detail = f"{decision['curie']} | {decision.get('label')}"
    if match:
        try:
            entity = ols.get_entity(ontology, match["iri"])
            parents = ols.list_parents(ontology, match["iri"])
            detail = (f"CURIE: {entity['curie'] or decision['curie']}\n"
                      f"LABEL: {entity['label']}\n"
                      f"SYNONYMS: {entity['synonyms'][:10]}\n"
                      f"DEFINITION: {entity['definition']}\n"
                      f"PARENTS: {[f'{p.curie} {p.label}' for p in parents]}")
        except ols.OlsError as exc:
            detail += f"\n(could not fetch full detail: {exc})"

    prompt = (
        "Independently review a proposed ontology mapping. You have not seen the "
        "reasoning behind it.\n\n"
        f"{RELATIONS}\n\n"
        f"QUERY TERM: {state['normalized_term']}\n"
        f"CONTEXT: {state.get('optional_context') or '(none)'}\n"
        f"ENTITY CLASS: {state['target_ontology_class']}\n"
        f"CLAIMED RELATION: {decision['relation']}\n\n"
        f"PROPOSED ENTRY:\n{detail}\n\n"
        "Does the claimed relation hold? 'exact' requires the query to equal the "
        "label or one of the synonyms exactly.\n"
        "Set confirmed=true when the claimed relation is right, and put that same "
        "value in `relation`. Set confirmed=false only when a DIFFERENT relation "
        "applies, put the one you would assign in `relation`, and make `issue` say "
        "why - your issue text and your relation must agree, using the directions "
        "above. If the entry is more specific than the query term, that is 'child', "
        "not 'parent'."
    )
    result = get_llm().with_structured_output(Verification).invoke(prompt)
    return {"verification": result.model_dump()}


# ------------------------------------------------------------------ human in loop

def human_review(state):
    """Pause for a person. Resumable: the checkpointer holds the run until approval."""
    decision = state["decision"]
    verification = state.get("verification") or {}

    answer = interrupt({
        "question": "Approve this ontology mapping?",
        "term": state["normalized_term"],
        "ontology_class": state["target_ontology_class"],
        "proposed": {
            "curie": decision.get("curie"), "label": decision.get("label"),
            "relation": decision["relation"], "confidence": decision["confidence"],
            "reasoning": decision["reasoning"],
        },
        "verification": verification,
        "reply_with": {"approved": True, "by": "your-name",
                       "relation": "(optional override)"},
    })

    approved = bool(answer.get("approved"))
    by = answer.get("by") or "unknown"
    relation = answer.get("relation") or decision["relation"]
    if not approved:
        relation = "none"

    return {
        "decision": {**decision, "relation": relation},
        "flag": relation,
        "outcome": OUTCOME[relation],
        "decided_by": f"HumanReviewer:{by}",
        "annotation": ("approved" if approved else "rejected")
                      + f" after {MODEL} proposed {decision['relation']}",
    }


# ------------------------------------------------------------------------ persist

def persist(state):
    """Log every decision; promote confirmed matches into the alias table."""
    decision = state.get("decision") or {}
    relation = state.get("flag") or decision.get("relation") or "none"
    # The decision carries the ontology it was made against, including on alias hits.
    ontology = decision.get("ontology") or ""
    if not ontology and state.get("ontologies"):
        ontology = state["ontologies"][state["tier"]]

    annotation = state.get("annotation") or ""
    verification = state.get("verification") or {}
    if verification.get("issue"):
        annotation = f"{annotation}; reviewer: {verification['issue']}".strip("; ")
    if state.get("retrieval_errors"):
        annotation = f"{annotation}; {len(state['retrieval_errors'])} retrieval error(s)".strip("; ")

    tables.log_decision(
        term=state["normalized_term"],
        standardized_term=decision.get("label"),
        term_id=decision.get("curie"),
        ontology=ontology,
        flag=relation,
        annotation=annotation,
        decided_by=state.get("decided_by") or MODEL,
    )

    # Only adjudicated same-concept matches become aliases; parent/child never do.
    if relation in ("exact", "semantic") and decision.get("curie") \
            and state.get("decided_by") != "alias":
        tables.upsert_alias(
            term=state["normalized_term"],
            ontology_class=state["target_ontology_class"],
            standardized_term=decision.get("label") or "",
            term_id=decision["curie"],
            ontology=ontology,
            flag=relation,
            decided_by=state.get("decided_by") or MODEL,
        )

    return {"flag": relation, "outcome": OUTCOME[relation], "annotation": annotation}


# ------------------------------------------------------------------------ routing

def route_after_alias(state):
    return "persist" if state.get("decision") else "plan_tiers"


def route_tools(state):
    """Run the model's requested tools, unless the budget is spent.

    Wraps the prebuilt tools_condition so the routing logic stays theirs, and adds
    the one thing it cannot know: a hard ceiling on lookups per tier.
    """
    used = sum(isinstance(m, ToolMessage) for m in state.get("messages", []))
    if used >= TOOL_BUDGET:
        return "decide"
    return "tools" if tools_condition(state) == "tools" else "decide"


def route_after_decide(state):
    """Escalate on nothing found; widen once on a 'child'; otherwise review."""
    decision = state["decision"]
    if decision["relation"] == "none" and state["tier"] + 1 < len(state["ontologies"]):
        return "next_tier"
    if decision["relation"] == "child" and not state.get("widened"):
        return "widen"
    return "verify"


def route_after_verify(state):
    """A person decides on exact claims, reviewer disagreement, or low confidence."""
    decision = state["decision"]
    verification = state.get("verification") or {}
    if decision["relation"] == "none":
        return "persist"
    if (decision["relation"] == "exact"
            or not verification.get("confirmed", True)
            or decision.get("confidence", 0) < CONFIDENCE_FLOOR):
        return "human_review"
    return "persist"
