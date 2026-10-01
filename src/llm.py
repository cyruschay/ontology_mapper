"""Chat model factory.

One place to change the model. Imported lazily so the rest of the package stays
importable (and the OLS client testable) on a machine without langchain-openai
installed or an API key set.
"""

import os
from pathlib import Path

MODEL = "gpt-6-luna"  # rejects a temperature parameter, so none is passed

ENV_FILE = Path(__file__).parents[1] / ".env"


class LlmUnavailable(RuntimeError):
    """Raised when the model cannot be constructed, with the fix in the message."""


def _load_env():
    """Read OPENAI_API_KEY out of .env when it has not been exported.

    Deliberately stdlib: the agent should run in any environment, with or without
    python-dotenv installed.
    """
    if os.environ.get("OPENAI_API_KEY") or not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def get_llm(model=MODEL):
    """A chat model ready for .bind_tools() or .with_structured_output().

    Pinned to the Responses API: on /v1/chat/completions this model refuses
    function tools unless reasoning_effort is 'none'. Going through /v1/responses
    keeps the reasoning that the adjudication depends on, and in testing produced
    four parallel lookups where the non-reasoning path produced one.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LlmUnavailable(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise LlmUnavailable(
            f"OPENAI_API_KEY is not set, and none was found in {ENV_FILE}."
        )

    # No temperature: gpt-6-luna rejects the parameter.
    return ChatOpenAI(model=model, use_responses_api=True)
