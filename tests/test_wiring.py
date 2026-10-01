"""Graph-wiring tests with the model and the API stubbed.

These assert the properties that LangGraph is here to guarantee - deterministic tier
escalation, a hard tool budget, a resumable human interrupt, and correct persistence -
without needing an API key or a reachable OLS.

Run: python tests/test_wiring.py
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from src import nodes, ols, tables
from src.graph import build
from src.state import Candidate, MatchDecision, Verification


class FakeLlm:
    """Stands in for the chat model on both paths the graph uses.

    bind_tools() -> returns AIMessages, optionally carrying tool_calls, like a
    tool-calling model. with_structured_output() -> returns scripted Pydantic objects.
    """

    def __init__(self, decisions, verification=None, always_call=None):
        self.decisions = list(decisions)
        self.verification = verification or Verification(confirmed=True, relation="semantic")
        self.always_call = always_call   # a tool name to request on every turn
        self.calls = {"decision": 0, "verification": 0, "turns": 0}
        self._mode = None

    def bind_tools(self, _tools):
        self._mode = "tools"
        return self

    def with_structured_output(self, model):
        self._mode = model
        return self

    def invoke(self, _input):
        if self._mode == "tools":
            self.calls["turns"] += 1
            if self.always_call:
                return AIMessage(content="", tool_calls=[{
                    "name": self.always_call,
                    "args": {"curie": "PR:000050308"},
                    "id": f"call_{self.calls['turns']}",
                    "type": "tool_call",
                }])
            return AIMessage(content="no further lookups needed")
        if self._mode is Verification:
            self.calls["verification"] += 1
            return self.verification
        self.calls["decision"] += 1
        # Repeat the final scripted decision once the script runs out.
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]


def stub(monkey_llm, candidates_by_ontology):
    """Point nodes at the fake model and fake retrieval."""
    nodes.get_llm = lambda *a, **k: monkey_llm
    nodes.ols.search_lexical = lambda term, ontology, **k: [
        Candidate(**c) for c in candidates_by_ontology.get(ontology, [])
    ]
    nodes.ols.search_embedding = lambda term, ontology, **k: []
    nodes.ols.get_entity = lambda ontology, iri: {
        "curie": "X:1", "label": "stub", "definition": "d", "synonyms": ["IgG"],
        "has_parents": True, "has_children": True,
    }
    nodes.ols.list_parents = lambda ontology, iri: []
    nodes.ols.list_children = lambda ontology, iri: []


def fresh_tables():
    for path in (tables.ALIASES, tables.DECISIONS):
        path.unlink(missing_ok=True)


def run(term, ontology_class="protein", resume=None):
    app = build()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = app.invoke({"term": term, "optional_context": None,
                        "target_ontology_class": ontology_class}, config=thread)
    if state.get("__interrupt__") and resume is not None:
        state = app.invoke(Command(resume=resume), config=thread)
    return state, app, thread


PR_HIT = {"curie": "PR:000050308", "iri": "http://x/PR_000050308",
          "label": "IgG4 immunoglobulin complex (human)", "ontology": "pr"}
GO_HIT = {"curie": "GO:0071735", "iri": "http://x/GO_0071735",
          "label": "IgG immunoglobulin complex", "ontology": "go"}
DOID_HIT = {"curie": "DOID:0080356", "iri": "http://x/DOID_0080356",
            "label": "IgG4-related disease", "ontology": "doid"}


def test_tier_escalation():
    """'none' in pr must escalate to go - and the logged ontology must be go."""
    fresh_tables()
    llm = FakeLlm([
        MatchDecision(relation="none", confidence=0.9, reasoning="nothing in pr"),
        MatchDecision(relation="semantic", curie="GO:0071735",
                      label="IgG immunoglobulin complex", confidence=0.9,
                      reasoning="same concept"),
    ])
    stub(llm, {"pr": [], "go": [GO_HIT]})
    state, _, _ = run("IgG")

    assert state["tier"] == 1, state["tier"]
    assert state["outcome"] == "match", state["outcome"]
    row = tables._read(tables.DECISIONS)[-1]
    assert row["ontology"] == "go", row
    assert row["term_id"] == "GO:0071735", row
    print("  tier escalation: pr -> go, logged ontology=go")


def test_no_match_writes_no_alias():
    """Exhausting every tier must log 'none' and leave the alias table untouched."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="none", confidence=0.95, reasoning="absent")])
    stub(llm, {})
    state, _, _ = run("asdfzzz")

    assert state["flag"] == "none", state["flag"]
    assert state["outcome"] == "no match", state["outcome"]
    assert state["tier"] == 2, state["tier"]          # walked pr -> go -> ncit
    assert not tables.ALIASES.exists(), "no-match must not create an alias"
    assert tables._read(tables.DECISIONS)[-1]["flag"] == "none"
    print("  no match: all 3 tiers walked, flag=none, no alias written")


def test_exact_match_requires_human_and_writes_alias():
    """An 'exact' claim must interrupt, and approval must be attributed to the human."""
    fresh_tables()
    llm = FakeLlm(
        [MatchDecision(relation="exact", curie="PR:000050308",
                       label="IgG4 immunoglobulin complex (human)", confidence=0.99,
                       reasoning="label matches")],
        verification=Verification(confirmed=True, relation="exact"),
    )
    stub(llm, {"pr": [PR_HIT]})

    app = build()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = app.invoke({"term": "IgG", "optional_context": None,
                         "target_ontology_class": "protein"}, config=thread)
    assert paused.get("__interrupt__"), "exact claim must pause for a human"
    assert paused["__interrupt__"][0].value["proposed"]["relation"] == "exact"

    state = app.invoke(Command(resume={"approved": True, "by": "cyrus"}), config=thread)
    assert state["decided_by"] == "HumanReviewer:cyrus", state["decided_by"]
    assert state["outcome"] == "match"
    alias = tables.lookup_alias("IgG", "protein")
    assert alias and alias["term_id"] == "PR:000050308", alias
    assert alias["decided_by"] == "HumanReviewer:cyrus", alias
    print("  exact match: interrupted, resumed as HumanReviewer:cyrus, alias written")


def test_human_rejection_downgrades_to_none():
    """Rejecting a proposal must record 'none' and write no alias."""
    fresh_tables()
    llm = FakeLlm(
        [MatchDecision(relation="exact", curie="PR:000050308", label="wrong",
                       confidence=0.99, reasoning="claimed")],
        verification=Verification(confirmed=True, relation="exact"),
    )
    stub(llm, {"pr": [PR_HIT]})
    state, _, _ = run("IgG", resume={"approved": False, "by": "cyrus"})

    assert state["flag"] == "none", state["flag"]
    assert state["outcome"] == "no match"
    assert not tables.ALIASES.exists(), "a rejected proposal must not become an alias"
    print("  human rejection: downgraded to none, no alias written")


def test_tool_budget_is_capped():
    """A model that asks for a tool every turn must still terminate at the budget."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="semantic", curie="PR:000050308",
                                 label="x", confidence=0.9, reasoning="r")],
                  always_call="list_parents")
    stub(llm, {"pr": [PR_HIT]})
    state, _, _ = run("IgG", resume={"approved": True, "by": "cyrus"})

    executed = sum(isinstance(m, ToolMessage) for m in state["messages"])
    assert executed == nodes.TOOL_BUDGET, executed
    assert state["decision"]["relation"] in ("semantic", "exact")
    print(f"  tool budget: {executed} lookups executed then forced to decide, "
          f"cap={nodes.TOOL_BUDGET}")


def test_tool_error_becomes_a_message():
    """A failing lookup must come back as a ToolMessage, not end the run."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="child", curie="PR:000050308", label="x",
                                 confidence=0.9, reasoning="r")],
                  always_call="list_parents")
    stub(llm, {"pr": [PR_HIT]})
    nodes.ols.list_parents = lambda *a, **k: (_ for _ in ()).throw(
        ols.OlsError("parents: 500 Server Error"))
    state, _, _ = run("IgG", resume={"approved": True, "by": "cyrus"})

    errors = [m for m in state["messages"]
              if isinstance(m, ToolMessage) and "failed" in str(m.content)]
    assert errors, [str(m.content)[:60] for m in state["messages"]]
    assert state["outcome"] == "close", state["outcome"]
    print(f"  tool error: surfaced as ToolMessage, run completed ({state['outcome']})")


def test_unseen_identifier_is_rejected():
    """A CURIE the model never saw must not reach the decision table."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="exact", curie="PR:999999999",
                                 label="invented", confidence=0.99,
                                 reasoning="hallucinated")])
    stub(llm, {"pr": [PR_HIT]})
    state, _, _ = run("IgG")

    assert state["flag"] == "none", state["flag"]
    assert state["decision"]["curie"] is None, state["decision"]
    row = tables._read(tables.DECISIONS)[-1]
    assert "rejected unseen identifier PR:999999999" in row["annotation"], row
    assert not tables.ALIASES.exists()
    print("  unseen identifier: rejected, logged as none, no alias")


def test_discovered_parent_is_selectable():
    """A parent surfaced by traversal must be selectable, with its own ontology."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="parent", curie="GO:0071735",
                                 label="IgG immunoglobulin complex", confidence=0.9,
                                 reasoning="the discovered parent is the right level")],
                  always_call="list_parents")
    stub(llm, {"pr": [PR_HIT]})
    # Traversal of a PR term returns a GO parent, as the live API does.
    nodes.ols.list_parents = lambda ontology, iri: [Candidate(
        curie="GO:0071735", iri="http://x/GO_0071735",
        label="IgG immunoglobulin complex", ontology="go", source="hierarchy")]
    state, _, _ = run("IgG", resume={"approved": True, "by": "cyrus"})

    assert state["decision"]["curie"] == "GO:0071735", state["decision"]
    assert state["decision"]["ontology"] == "go", state["decision"]
    assert state["outcome"] == "close", state["outcome"]
    row = tables._read(tables.DECISIONS)[-1]
    assert row["ontology"] == "go", row      # not pr, the tier that was searched
    print("  discovered parent: selected GO:0071735 while searching pr, logged ontology=go")


def test_parallel_lookups_accumulate():
    """Several tool calls in one step must all land, not crash the channel.

    The live model issues lookups in parallel, and a plain last-value channel raises
    InvalidUpdateError when two tools write it in the same step.
    """
    fresh_tables()

    class TwoCalls(FakeLlm):
        def invoke(self, _input):
            if self._mode == "tools":
                self.calls["turns"] += 1
                if self.calls["turns"] == 1:
                    return AIMessage(content="", tool_calls=[
                        {"name": "list_parents", "args": {"curie": "PR:000050308"},
                         "id": "a", "type": "tool_call"},
                        {"name": "list_parents", "args": {"curie": "PR:000050305"},
                         "id": "b", "type": "tool_call"}])
                return AIMessage(content="done")
            return super().invoke(_input)

    llm = TwoCalls([MatchDecision(relation="parent", curie="PR:000050032",
                                  label="immunoglobulin complex", confidence=0.9,
                                  reasoning="r")],
                   verification=Verification(confirmed=True, relation="parent"))
    stub(llm, {"pr": [PR_HIT, {"curie": "PR:000050305", "iri": "http://x/b",
                               "label": "IgG1", "ontology": "pr"}]})
    nodes.ols.list_parents = lambda ontology, iri: [Candidate(
        curie="GO:0071735" if iri.endswith("PR_000050308") else "PR:000050032",
        iri=iri + "_p", label="a parent", source="hierarchy",
        ontology="go" if iri.endswith("PR_000050308") else "pr")]
    state, _, _ = run("IgG", resume={"approved": True, "by": "cyrus"})

    found = sorted(d["curie"] for d in state.get("discovered") or [])
    assert found == ["GO:0071735", "PR:000050032"], found
    assert state["decision"]["curie"] == "PR:000050032", state["decision"]
    print(f"  parallel lookups: both entries kept {found}")


def test_child_verdict_widens_to_parent():
    """A 'child' verdict must re-examine the entry's parents, once, and may switch.

    Mirrors the real case: PR has no species-agnostic 'immunoglobulin complex', but
    PR:000050032 (human) has GO:0019814 as its parent, which is an exact match.
    """
    fresh_tables()
    llm = FakeLlm([
        MatchDecision(relation="child", curie="PR:000050032",
                      label="immunoglobulin complex (human)", confidence=0.92,
                      reasoning="the human form is narrower"),
        MatchDecision(relation="exact", curie="GO:0019814",
                      label="immunoglobulin complex", confidence=0.97,
                      reasoning="the parent label is the query term"),
    ], verification=Verification(confirmed=True, relation="exact"))
    stub(llm, {"pr": [{"curie": "PR:000050032", "iri": "http://x/PR_000050032",
                       "label": "immunoglobulin complex (human)", "ontology": "pr"}]})
    nodes.ols.list_parents = lambda ontology, iri: [Candidate(
        curie="GO:0019814", iri="http://x/GO_0019814",
        label="immunoglobulin complex", ontology="go", source="hierarchy")]
    state, _, _ = run("immunoglobulin complex", resume={"approved": True, "by": "cyrus"})

    assert state["widened"] is True, state.get("widened")
    assert state["decision"]["curie"] == "GO:0019814", state["decision"]
    assert state["decision"]["ontology"] == "go", state["decision"]
    assert state["outcome"] == "match", state["outcome"]
    print("  child widening: PR:000050032 (child) -> GO:0019814 (exact) via parents")


def test_widening_happens_at_most_once():
    """A model that keeps saying 'child' must not loop through widening."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="child", curie="PR:000050032",
                                 label="immunoglobulin complex (human)",
                                 confidence=0.9, reasoning="still narrower")],
                  verification=Verification(confirmed=True, relation="child"))
    stub(llm, {"pr": [{"curie": "PR:000050032", "iri": "http://x/PR_000050032",
                       "label": "immunoglobulin complex (human)", "ontology": "pr"}]})
    calls = {"n": 0}

    def counting_parents(ontology, iri):
        calls["n"] += 1
        return [Candidate(curie="GO:0019814", iri="http://x/GO_0019814",
                          label="immunoglobulin complex", ontology="go",
                          source="hierarchy")]

    nodes.ols.list_parents = counting_parents
    state, _, _ = run("immunoglobulin complex", resume={"approved": True, "by": "cyrus"})

    assert state["decision"]["relation"] == "child", state["decision"]
    assert state["outcome"] == "close", state["outcome"]
    # one widening lookup, plus the one verify fetches for its detail block
    assert calls["n"] <= 2, calls["n"]
    print(f"  widening bounded: kept 'child', {calls['n']} parent lookups, no loop")


def test_verify_glossary_is_shared():
    """The reviewer and the adjudicator must define relations identically."""
    assert nodes.RELATIONS in nodes.SYSTEM
    assert "NARROWER" in nodes.RELATIONS and "BROADER" in nodes.RELATIONS
    print("  glossary: one RELATIONS block, used by both prompts")


def test_low_confidence_triggers_review():
    """Confidence under the floor must route to a human even when verified."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="semantic", curie="PR:000050308", label="x",
                                 confidence=0.4, reasoning="unsure")])
    stub(llm, {"pr": [PR_HIT]})
    app = build()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    paused = app.invoke({"term": "IgG", "optional_context": None,
                         "target_ontology_class": "protein"}, config=thread)
    assert paused.get("__interrupt__"), "low confidence must pause"
    print(f"  low confidence: {0.4} < {nodes.CONFIDENCE_FLOOR} paused for review")


def test_retrieval_error_is_recorded_not_raised():
    """An embedding outage must degrade the run, not end it."""
    fresh_tables()
    llm = FakeLlm([MatchDecision(relation="semantic", curie="DOID:0080356",
                                 label="IgG4-related disease", confidence=0.9,
                                 reasoning="r")])
    stub(llm, {"doid": [DOID_HIT]})
    nodes.ols.search_embedding = lambda *a, **k: (_ for _ in ()).throw(
        ols.OlsError("llm_search: 500 Server Error"))
    state, _, _ = run("IgG", "disease", resume={"approved": True, "by": "cyrus"})

    assert state["retrieval_errors"], "the embedding failure should be recorded"
    assert "500" in state["retrieval_errors"][0]
    assert state["tier"] == 0, "lexical still matched, so no escalation"
    assert state["decision"]["curie"] == "DOID:0080356", state["decision"]
    assert tables._read(tables.DECISIONS)[-1]["annotation"].endswith("retrieval error(s)")
    print(f"  retrieval error: degraded to lexical, matched on tier 0, "
          f"{len(state['retrieval_errors'])} error(s) logged")


if __name__ == "__main__":
    fresh_tables()
    for test in (test_tier_escalation, test_no_match_writes_no_alias,
                 test_exact_match_requires_human_and_writes_alias,
                 test_human_rejection_downgrades_to_none,
                 test_tool_budget_is_capped,
                 test_tool_error_becomes_a_message,
                 test_unseen_identifier_is_rejected,
                 test_discovered_parent_is_selectable,
                 test_parallel_lookups_accumulate,
                 test_child_verdict_widens_to_parent,
                 test_widening_happens_at_most_once,
                 test_verify_glossary_is_shared,
                 test_low_confidence_triggers_review,
                 test_retrieval_error_is_recorded_not_raised):
        print(f"{test.__name__}:")
        test()
    fresh_tables()
    print("\nall wiring tests passed")
