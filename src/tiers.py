"""Ontology preference order per entity class.

The order is deterministic graph data, not something the LLM chooses. A query
escalates to the next tier only when the current one yields no match.
"""

PREFERENCE = {
    # PR embeds 155,814 UniProt accessions as PR:P08069-style ids, so it covers
    # the UniProt tier directly; GO adds complexes and activities; NCIT is the catch-all.
    "protein": ["pr", "go", "ncit"],
    # MONDO is the merged disease ontology and catches what DOID lacks; EFO carries
    # trait/GWAS phrasing.
    "disease": ["doid", "mondo", "efo", "ncit"],
    # UBERON for anatomy, CL for cell types, BTO for tissue and cell-line phrasing.
    "specimen": ["uberon", "cl", "bto", "ncit"],
    # Taxonomy alone, as specified, and lexical only: see NO_EMBEDDINGS below.
    "organism": ["ncbitaxon"],
}

# Ontologies to skip embedding search for, measured against the live API:
#   pr        - the request times out rather than answering, so skipping it saves a
#               90s stall plus retries on every protein query
#   ncbitaxon - the request succeeds but returns 0 results, so there is nothing to merge
# Skipped rather than called and discarded, so a run never pays for a request that
# cannot help. Re-check if OLS adds embeddings for these.
NO_EMBEDDINGS = {"pr", "ncbitaxon"}


def ontologies_for(ontology_class):
    """The ordered tier list for an entity class."""
    try:
        return list(PREFERENCE[ontology_class])
    except KeyError:
        raise ValueError(
            f"unknown ontology_class {ontology_class!r}; expected one of {sorted(PREFERENCE)}"
        ) from None


def use_embeddings(ontology):
    """Whether an embedding search is worth issuing for this ontology."""
    return ontology not in NO_EMBEDDINGS
