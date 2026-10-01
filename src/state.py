"""The typed contract shared by every node in the graph."""

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

OntologyClass = Literal["protein", "disease", "specimen", "organism"]
Relation = Literal["exact", "semantic", "parent", "child", "none"]
Outcome = Literal["match", "close", "no match"]


def merge_discovered(left, right):
    """Reducer for traversal results, deduped by CURIE.

    Needed because the model issues tool calls in parallel: several tools write this
    channel in one step, which a plain last-value channel rejects outright. Passing
    None resets the channel, which is how a new tier discards the previous one's finds.
    """
    if right is None:
        return []
    merged = {entry["curie"]: entry for entry in (left or [])}
    for entry in right or []:
        merged.setdefault(entry["curie"], entry)
    return list(merged.values())


class Candidate(BaseModel):
    """One retrieved ontology entry, normalized across both OLS search APIs."""

    curie: str
    iri: str
    label: str
    ontology: str
    synonyms: list[str] = Field(default_factory=list)
    definition: Optional[str] = None
    source: Literal["lexical", "embedding", "both", "hierarchy"] = "lexical"
    score: Optional[float] = None


class MatchDecision(BaseModel):
    """The adjudicating model's structured verdict.

    Carries no tool-request field: traversal happens through real bound tools, so
    by the time this is produced the model has already finished looking things up.
    """

    relation: Relation = Field(description="How the best candidate relates to the query")
    curie: Optional[str] = Field(None, description="CURIE of the chosen candidate, null if relation is none")
    label: Optional[str] = Field(None, description="Label of the chosen candidate")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="One or two sentences justifying the verdict")


class Verification(BaseModel):
    """An independent second opinion on a claimed match."""

    confirmed: bool = Field(description="True if the claimed relation holds")
    relation: Relation = Field(description="The relation this reviewer would assign")
    issue: Optional[str] = Field(None, description="What is wrong, if not confirmed")


class MapperState(TypedDict, total=False):
    """Everything the graph carries. Checkpointed, so all values stay JSON-friendly."""

    # input
    term: str
    normalized_term: str
    optional_context: Optional[str]
    target_ontology_class: OntologyClass

    # tier plan
    ontologies: list[str]
    tier: int
    page: int

    # retrieval
    candidates: list[dict]
    # Entries surfaced by traversal rather than search. Selectable, and they carry
    # their own ontology, which may differ from the tier being searched.
    discovered: Annotated[list[dict], merge_discovered]
    retrieval_errors: list[str]

    # adjudication: the tool-calling conversation lives here, appended by the
    # add_messages reducer and consumed by ToolNode
    messages: Annotated[list[AnyMessage], add_messages]
    decision: Optional[dict]
    verification: Optional[dict]
    # Set once a 'child' verdict has been re-examined against its parents, so the
    # widening step cannot loop.
    widened: bool

    # outcome
    flag: Relation
    outcome: Outcome
    decided_by: str
    annotation: str
