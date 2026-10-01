"""Command line entry point.

    python -m src.cli --term IgG --class protein
    python -m src.cli --term IgG --class protein --approve cyrus
"""

import argparse
import json
import uuid

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .graph import build
from .llm import LlmUnavailable
from .ols import OlsError
from .state import MapperState


def _report(state):
    decision = state.get("decision") or {}
    print(f"\nterm       : {state.get('normalized_term')}")
    print(f"outcome    : {state.get('outcome')}  (flag={state.get('flag')})")
    print(f"identifier : {decision.get('curie') or '-'}")
    print(f"label      : {decision.get('label') or '-'}")
    ontology = decision.get("ontology")
    if not ontology and state.get("ontologies"):
        ontology = state["ontologies"][state.get("tier", 0)]
    print(f"ontology   : {ontology or '-'}")
    print(f"decided_by : {state.get('decided_by')}")
    print(f"confidence : {decision.get('confidence')}")
    print(f"reasoning  : {decision.get('reasoning')}")
    if state.get("verification"):
        print(f"review     : {state['verification']}")
    lookups = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
    if lookups:
        print(f"lookups    : {len(lookups)}")
        for message in lookups:
            print(f"  - {str(message.content).splitlines()[0][:100]}")
    if state.get("retrieval_errors"):
        print("retrieval errors:")
        for line in state["retrieval_errors"]:
            print(f"  - {line}")


def main():
    parser = argparse.ArgumentParser(description="Map a term onto an ontology identifier.")
    parser.add_argument("--term", required=True)
    parser.add_argument("--class", dest="ontology_class", required=True,
                        choices=["protein", "disease", "specimen", "organism"])
    parser.add_argument("--context", default=None, help="optional disambiguating context")
    parser.add_argument("--approve", metavar="NAME",
                        help="auto-approve any human checkpoint as NAME (for development)")
    parser.add_argument("--thread", default=None, help="reuse a thread id to resume a run")
    args = parser.parse_args()

    agent = build()
    thread = {"configurable": {"thread_id": args.thread or str(uuid.uuid4())}}

    initial: MapperState = {
        "term": args.term,
        "optional_context": args.context,
        "target_ontology_class": args.ontology_class,
    }
    try:
        state = agent.invoke(initial, config=thread)
    except LlmUnavailable as exc:
        raise SystemExit(f"error: {exc}")
    except OlsError as exc:
        raise SystemExit(f"error: the ontology service failed after retries: {exc}")

    # An interrupt surfaces as __interrupt__ rather than raising.
    while state.get("__interrupt__"):
        payload = state["__interrupt__"][0].value
        if args.approve:
            state = agent.invoke(Command(resume={"approved": True, "by": args.approve}),
                                 config=thread)
            continue
        print("\n--- HUMAN APPROVAL NEEDED ---")
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nresume with: --thread {thread['configurable']['thread_id']} --approve <name>")
        return

    _report(state)


if __name__ == "__main__":
    main()
