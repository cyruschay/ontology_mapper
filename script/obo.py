"""Minimal streaming primitives for reading .obo ontology files."""

from pathlib import Path

PARENT_DIR = Path(__file__).parents[1] / "data" / "ontology"


def resolve(name) -> Path:
    """Accept a path or a bare filename living in data/ontology."""
    path = Path(name)
    return path if path.exists() else PARENT_DIR / name


def iter_terms(path, fields):
    """Yield each [Term] stanza of an .obo file as a dict of field -> list of values."""
    term = None
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line.startswith("["):
                if term:
                    yield term
                term = {} if line == "[Term]" else None
            elif term is not None and ": " in line:
                key, value = line.split(": ", 1)
                if key in fields:
                    term.setdefault(key, []).append(value)
    if term:
        yield term


def synonyms(term, scope="EXACT"):
    """Return the synonym texts of a term, keeping only the given scope (None for all).

    A synonym line looks like: synonym: "surfer's eye" EXACT [] {comment="..."}
    """
    texts = []
    for line in term.get("synonym", []):
        parts = line.split('"')
        if len(parts) < 3:
            continue
        text, rest = parts[1], parts[2].split()
        if scope is None or (rest and rest[0] == scope):
            texts.append(text)
    return texts


def parents(term):
    """Return the (id, name) of each is_a parent; name falls back to the id.

    A line looks like: is_a: UBERON:0011216 {source="cjm"} ! organ system subdivision
    The trailing qualifier block is dropped so the same parent is never split in two.
    """
    pairs = []
    for line in term.get("is_a", []):
        parent_id, _, name = line.partition(" ! ")
        parent_id = parent_id.split("{")[0].strip()
        pairs.append((parent_id, name.strip() or parent_id))
    return pairs
