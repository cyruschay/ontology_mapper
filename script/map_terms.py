"""Find ontology candidates for a list of terms and render them as a sub-taxonomy."""

import argparse

import obo

FIELDS = ("id", "name", "synonym", "is_a", "is_obsolete")

MAX_LEAVES = 10  # a node with more terminal leaves than this gets its leaves cut
KEEP = 5         # how many terminal leaves survive the cut

# Identifier prefixes to ignore, per ontology file. These stanzas are database
# cross-references (genes, organisms) rather than classes worth mapping onto.
BLACKLIST = {
    "pr.obo": ("HGNC", "MGI", "RGD", "ZFIN", "FlyBase", "WormBase", "EcoGene",
               "dictyBase", "SGD", "PomBase", "NCBITaxon", "NCBIGene")
    }


def blocked(identifier, prefixes):
    """True if an identifier carries a blacklisted prefix."""
    return identifier.split(":", 1)[0] in prefixes


def find_matches(path, queries, include_obsolete=False, prefixes=()):
    """Match each query against term names and EXACT synonyms.

    Exact matches win; the partial (substring) pass only applies to queries that
    had no exact match. Returns {query: [term]}.
    """
    exact = {q: [] for q in queries}
    partial = {q: [] for q in queries}
    lowered = [(q, q.lower()) for q in queries]

    for term in obo.iter_terms(path, FIELDS):
        if term.get("is_obsolete") and not include_obsolete:
            continue
        if blocked(term["id"][0], prefixes):
            continue
        labels = [n.lower() for n in term.get("name", [])]
        labels += [s.lower() for s in obo.synonyms(term, scope="EXACT")]

        for query, needle in lowered:
            if any(needle == label for label in labels):
                exact[query].append(term)
            elif any(needle in label for label in labels):
                partial[query].append(term)

    return {q: exact[q] or partial[q] for q in queries}


def build_tree(path, matches, prefixes=()):
    """Splice the matches into one graph: matched nodes plus one layer of parents.

    The parent layer is retrieved first, then edges are composed over the whole
    node set in a second pass, so retrieved nodes that are linked *through* an
    added parent end up in the same sub-graph instead of separate ones.
    """
    names = {}
    for hits in matches.values():
        for term in hits:
            names[term["id"][0]] = term.get("name", [term["id"][0]])[0]
            for parent_id, parent_name in obo.parents(term):
                if not blocked(parent_id, prefixes):
                    names.setdefault(parent_id, parent_name)

    children = {}
    for term in obo.iter_terms(path, ("id", "name", "is_a")):
        term_id = term["id"][0]
        if term_id not in names:
            continue
        for parent_id, _ in obo.parents(term):
            if parent_id in names:
                children.setdefault(parent_id, set()).add(term_id)

    has_parent = {child for kids in children.values() for child in kids}
    roots = sorted(set(names) - has_parent)
    return names, children, roots


def count_leaves(node_id, children, cache):
    """Number of terminal leaves in a node's sub-graph."""
    if node_id not in cache:
        kids = children.get(node_id, ())
        cache[node_id] = sum(count_leaves(k, children, cache) for k in kids) or 1
    return cache[node_id]


def render(path, queries, matches, prefixes=()):
    """Render the spliced sub-taxonomy as indented text for an LLM prompt."""
    names, children, roots = build_tree(path, matches, prefixes)
    cache = {}

    def walk(node_id, depth, seen):
        lines = [f"{'  ' * depth}{node_id}  {names[node_id]}"]
        cut = count_leaves(node_id, children, cache) > MAX_LEAVES
        shown, skipped = 0, False

        for child in sorted(children.get(node_id, ())):
            if child in seen:
                continue
            if cut and not children.get(child):  # a terminal leaf of this node
                if shown >= KEEP:
                    skipped = True
                    continue
                shown += 1
            lines += walk(child, depth + 1, seen | {child})

        if skipped:
            lines.append(f"{'  ' * (depth + 1)}...")
        return lines

    blocks = [f"ONTOLOGY: {path.name}\nQUERIES: {', '.join(queries)}"]
    blocks += ["\n".join(walk(root, 0, {root})) for root in roots]

    missing = [q for q, hits in matches.items() if not hits]
    if missing:
        blocks.append("\n".join(f"NO MATCH: {q}" for q in missing))
    return "\n\n".join(blocks)


def main():
    parser = argparse.ArgumentParser(description="Map terms onto an ontology sub-taxonomy.")
    parser.add_argument("ontology", help="ontology file, e.g. doid.obo")
    parser.add_argument("terms", nargs="*", help="terms to map (case-insensitive)")
    parser.add_argument("--file", help="file with one term per line")
    parser.add_argument("--include-obsolete", action="store_true", help="keep obsolete terms")
    args = parser.parse_args()

    queries = list(args.terms)
    if args.file:
        queries += [line.strip() for line in open(args.file) if line.strip()]
    if not queries:
        parser.error("provide at least one term or --file")

    path = obo.resolve(args.ontology)
    prefixes = BLACKLIST.get(path.name, ())
    matches = find_matches(path, queries, args.include_obsolete, prefixes)
    print(render(path, queries, matches, prefixes))


if __name__ == "__main__":
    main()
