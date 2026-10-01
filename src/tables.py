"""The alias table and the decision log, as TSVs under data/tables/.

The alias table is the memory that makes repeat runs free: an exact string hit
short-circuits the whole LLM path. The decision log is append-only provenance for
every AI or human verdict.

Plain csv rather than pandas: the tables are small key-value records, and pandas
in this environment emits NumPy ABI warnings on import for no benefit here.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

TABLE_DIR = Path(__file__).parents[1] / "data" / "tables"
ALIASES = TABLE_DIR / "aliases.tsv"
DECISIONS = TABLE_DIR / "decisions.tsv"

ALIAS_COLUMNS = ["term_lower", "term", "standardized_term", "term_id", "ontology",
                 "ontology_class", "flag", "decided_by", "time"]

# Exactly the fields specified for the decision table.
DECISION_COLUMNS = ["term", "standardized_term", "term_id", "ontology", "flag",
                    "annotation", "decided_by", "time"]


def now():
    """UTC timestamp, so logs from different machines stay comparable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_atomic(rows, columns, path):
    """Write via a temp file then replace, so a crash cannot truncate the table."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def lookup_alias(term, ontology_class):
    """Exact (case-insensitive) alias hit, or None."""
    key = term.strip().lower()
    for row in _read(ALIASES):
        if row.get("term_lower") == key and row.get("ontology_class") == ontology_class:
            return row
    return None


def upsert_alias(term, ontology_class, standardized_term, term_id, ontology, flag, decided_by):
    """Record an adjudicated match so future runs skip retrieval entirely."""
    key = term.strip().lower()
    rows = [r for r in _read(ALIASES)
            if not (r.get("term_lower") == key and r.get("ontology_class") == ontology_class)]
    row = {
        "term_lower": key, "term": term, "standardized_term": standardized_term,
        "term_id": term_id, "ontology": ontology, "ontology_class": ontology_class,
        "flag": flag, "decided_by": decided_by, "time": now(),
    }
    rows.append(row)
    _write_atomic(rows, ALIAS_COLUMNS, ALIASES)
    return row


def log_decision(term, standardized_term, term_id, ontology, flag, annotation, decided_by):
    """Append one row to the decision log. Every AI decision passes through here."""
    row = {
        "term": term, "standardized_term": standardized_term or "", "term_id": term_id or "",
        "ontology": ontology or "", "flag": flag, "annotation": annotation or "",
        "decided_by": decided_by, "time": now(),
    }
    _write_atomic(_read(DECISIONS) + [row], DECISION_COLUMNS, DECISIONS)
    return row
