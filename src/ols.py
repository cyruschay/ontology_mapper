"""The only module that talks to the EBI Ontology Lookup Service.

Every quirk of the two OLS search APIs is absorbed here so the rest of the graph
sees one normalized Candidate shape:

* lexical search leaks imported classes unless isDefiningOntology=true
* `obo_id` is empty for PR terms, so CURIEs come from `short_form`
* embedding search returns `label` as a list and a zeroed score when filtered,
  but a string and a real score when unfiltered
* traversal endpoints need the IRI double-URL-encoded
"""

from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .state import Candidate

BASE = "https://www.ebi.ac.uk/ols4/api"
EMBEDDING_MODEL = "llama-embed-nemotron-8b_pca512"  # the only model that embeds live text

# Measured: a cold lexical query can take >60s or 500 before succeeding on retry, so
# the load-bearing path stays patient.
TIMEOUT = 45

# The embedding search is best-effort: it is missing for some ontologies and goes
# unresponsive for minutes at a time. Waiting on it the way we wait on lexical search
# turned a degraded endpoint into a 6-minute stall per call, so it gets a short fuse
# and a single attempt - the run degrades to lexical instead.
EMBEDDING_TIMEOUT = 12

FIELDS = "iri,label,short_form,obo_id,synonym,description,ontology_name"


class OlsError(RuntimeError):
    """Raised on any transport or protocol failure, for the retry policy to catch."""


def _session(total=3):
    """A session with transport-level retry, since OLS intermittently 500s."""
    retry = Retry(
        total=total,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _session()
# No transport retry: one short attempt, then fall back to lexical.
EMBEDDING_SESSION = _session(total=0)


def _get(path, params, session=None, timeout=None):
    session = session or SESSION
    try:
        response = session.get(f"{BASE}/{path}", params=params,
                               timeout=timeout or TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise OlsError(f"{path}: {exc}") from exc


def _curie(short_form, obo_id=None):
    """PR terms come back with an empty obo_id, so derive the CURIE from short_form."""
    if obo_id:
        return obo_id
    return short_form.replace("_", ":", 1) if short_form else ""


def _text(value):
    """Flatten the several shapes OLS uses for one text field.

    Ontology-filtered results wrap label/definition in a list; definitions may be
    reification objects of the form {"type": ["reification"], "value": "..."}.
    """
    if isinstance(value, list):
        return _text(value[0]) if value else None
    if isinstance(value, dict):
        return value.get("value")
    return value


def ontology_of(curie):
    """The OLS ontology id a CURIE belongs to.

    Verified against every ontology in the tier lists: the lowercased CURIE prefix
    is the OLS ontology id (PR -> pr, NCBITaxon -> ncbitaxon, GO -> go).
    """
    prefix = curie.split(":", 1)[0]
    return prefix.lower() if prefix else ""


def _encode_iri(iri):
    """Traversal endpoints need the IRI encoded twice."""
    return quote(quote(iri, safe=""), safe="")


def search_lexical(term, ontology, rows=10, start=0):
    """Solr-backed search. Works for every ontology, including those without embeddings."""
    data = _get("search", {
        "q": term,
        "ontology": ontology,
        "isDefiningOntology": "true",  # `local=true` is broken upstream and returns 0
        "rows": rows,
        "start": start,
        "fieldList": FIELDS,
    })
    docs = data.get("response", {}).get("docs", [])
    candidates = []
    for doc in docs:
        short_form = doc.get("short_form", "")
        # Defence in depth: the filter above should already exclude imported classes.
        if short_form and not short_form.lower().startswith(ontology.lower()):
            continue
        candidates.append(Candidate(
            curie=_curie(short_form, doc.get("obo_id")),
            iri=doc.get("iri", ""),
            label=_text(doc.get("label")) or "",
            ontology=doc.get("ontology_name", ontology),
            synonyms=doc.get("synonym") or [],
            definition=_text(doc.get("description")),
            source="lexical",
        ))
    return candidates


def search_embedding(term, ontology, size=10, page=0):
    """Vector search. Returns [] for ontologies with no stored embeddings (e.g. PR)."""
    data = _get("v2/entities/llm_search", {
        "q": term,
        "model": EMBEDDING_MODEL,
        "ontologyId": ontology,
        "size": size,
        "page": page,
    }, session=EMBEDDING_SESSION, timeout=EMBEDDING_TIMEOUT)
    candidates = []
    for element in data.get("elements", []):
        candidates.append(Candidate(
            curie=element.get("curie", ""),
            iri=element.get("iri", ""),
            label=_text(element.get("label")) or "",
            ontology=element.get("ontologyId", ontology),
            synonyms=[],
            definition=_text(element.get("definition")),
            source="embedding",
            score=element.get("distance") or None,  # zeroed when ontology-filtered
        ))
    return candidates


def _terms(path):
    data = _get(path, {})
    return data.get("_embedded", {}).get("terms", [])


def list_parents(ontology, iri):
    """Direct is_a parents of a term. Parents may be defined in another ontology."""
    return [_traversed(t) for t in
            _terms(f"ontologies/{ontology}/terms/{_encode_iri(iri)}/parents")]


def list_children(ontology, iri):
    """Direct children of a term. Children may be defined in another ontology."""
    return [_traversed(t) for t in
            _terms(f"ontologies/{ontology}/terms/{_encode_iri(iri)}/children")]


def _traversed(term):
    """Build a Candidate from a traversal result, tagged as hierarchy-sourced."""
    curie = _curie(term.get("short_form", ""), term.get("obo_id"))
    return Candidate(
        curie=curie,
        iri=term.get("iri", ""),
        label=_text(term.get("label")) or "",
        ontology=ontology_of(curie),
        definition=_text(term.get("description")),
        source="hierarchy",
    )


def get_entity(ontology, iri):
    """Full detail for one class, used by the verification node."""
    data = _get(f"v2/ontologies/{ontology}/classes/{_encode_iri(iri)}", {})
    return {
        "curie": data.get("curie", ""),
        "label": _text(data.get("label")) or "",
        "definition": _text(data.get("definition")),
        # v2 synonyms can also be reification objects, so unwrap each one.
        "synonyms": [t for t in (_text(v) for v in (data.get("synonym") or [])) if t],
        "has_parents": data.get("hasDirectParents"),
        "has_children": data.get("hasDirectChildren"),
    }
