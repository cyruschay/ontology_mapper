import argparse

import obo

# Fields kept for each matched term.
FIELDS = ("id", "name", "def", "synonym", "is_a", "xref")


def matches(term, query, exact):
    """Check the query against the term id, name and synonyms."""
    names = term.get("id", []) + term.get("name", []) + obo.synonyms(term, scope=None)
    names = [n.lower() for n in names]
    return query in names if exact else any(query in n for n in names)


def query_ontology(path, query, exact):
    return [t for t in obo.iter_terms(path, FIELDS) if matches(t, query, exact)]


def main():
    parser = argparse.ArgumentParser(description="Find matching terms in an ontology.")
    parser.add_argument("term", help="term to look for (case-insensitive)")
    parser.add_argument("ontology", help="ontology file, e.g. doid.obo")
    parser.add_argument("--exact", action="store_true", help="require an exact match")
    args = parser.parse_args()

    path = obo.resolve(args.ontology)
    results = query_ontology(path, args.term.lower(), args.exact)
    print(f"{len(results)} match(es) for {args.term!r} in {path.name}")
    for term in results:
        print()
        for field in FIELDS:
            for value in term.get(field, []):
                print(f"{field}: {value}")


if __name__ == "__main__":
    main()
