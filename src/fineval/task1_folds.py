"""Leakage-resistant folds for low-resource financial NER.

The released GRFinNER rows have no issuer or source-document identifier, but
the same named entity often occurs in several nearby excerpts.  Ordinary row
K-fold therefore leaks exact company/person names into held-out predictions.
These helpers connect rows sharing a normalized organization or person and
assign whole connected components to balanced folds.

Locations are deliberately not linkage keys: generic surfaces such as Greece,
Athens, or Cyprus would collapse unrelated issuers into one giant component.
They are still counted when balancing label load.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
import unicodedata


LINK_TYPES = frozenset({"ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ"})
LEGAL_WORDS = frozenset({
    "a", "e", "ae", "s", "sa", "ltd", "limited", "plc", "inc", "llc",
    "bv", "sarl", "company", "corporation", "corp", "group", "holding",
    "holdings", "bank", "α", "ε", "αε", "αβεε", "αεβε", "αεεαπ",
    "ικε", "μαε", "ανωνυμη", "ανωνυμοσ", "εταιρεια", "εταιρειασ",
    "εταιρια", "εταιριασ", "ομιλοσ", "ομιλου", "τραπεζα", "συμμετοχων",
})


def normalize_entity_surface(surface: str) -> str:
    """Return an accent/punctuation-insensitive exact-entity key."""
    decomposed = unicodedata.normalize("NFD", str(surface).casefold())
    unaccented = "".join(
        character for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).replace("ς", "σ")
    words = re.findall(r"\w+", unaccented, flags=re.UNICODE)
    stripped = [word for word in words if word not in LEGAL_WORDS]
    # Do not erase a surface consisting solely of a legal abbreviation.
    return " ".join(stripped or words)


def _components(entity_rows, link_types=LINK_TYPES):
    parent = list(range(len(entity_rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner: dict[tuple[str, str], int] = {}
    for row_index, entities in enumerate(entity_rows):
        for surface, kind in set(entities):
            if kind not in link_types:
                continue
            normalized = normalize_entity_surface(surface)
            if len(normalized) < 3:
                continue
            key = kind, normalized
            if key in owner:
                union(row_index, owner[key])
            else:
                owner[key] = row_index

    components = defaultdict(list)
    for row_index in range(len(entity_rows)):
        components[find(row_index)].append(row_index)
    return list(components.values())


def assign_entity_disjoint_folds(entity_rows, n_splits: int = 5, seed: int = 7):
    """Assign balanced folds while keeping exact linked entities together."""
    entity_rows = [list(row) for row in entity_rows]
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if len(entity_rows) < n_splits:
        raise ValueError("fewer rows than folds")

    components = _components(entity_rows)
    type_names = sorted({kind for row in entity_rows for _, kind in row})
    type_totals = Counter(kind for row in entity_rows for _, kind in row)
    target_types = {
        kind: type_totals[kind] / n_splits for kind in type_names
    }

    def stable_tie(rows) -> int:
        digest = hashlib.sha256(
            (str(seed) + ":" + ",".join(map(str, rows))).encode()
        ).digest()
        return int.from_bytes(digest[:8], "big")

    component_stats = []
    for rows in components:
        counts = Counter(
            kind for index in rows for _, kind in set(entity_rows[index])
        )
        component_stats.append((rows, counts))
    component_stats.sort(
        key=lambda item: (-len(item[0]), -sum(item[1].values()), stable_tie(item[0]))
    )

    fold_rows = [0] * n_splits
    fold_types = [Counter() for _ in range(n_splits)]
    assignments = [-1] * len(entity_rows)
    for rows, counts in component_stats:
        candidates = []
        for fold in range(n_splits):
            # Greedy longest-component scheduling: the least-loaded fold wins.
            # Label burden only breaks equal row loads.  A target-deviation
            # objective would incorrectly pack one fold up to its target before
            # opening the remaining folds.
            type_cost = sum(
                (
                    (fold_types[fold][kind] + counts[kind] - target_types[kind])
                    / max(1.0, target_types[kind])
                ) ** 2
                for kind in type_names
            )
            candidates.append((fold_rows[fold], type_cost, fold))
        _, _, selected = min(candidates)
        for row_index in rows:
            assignments[row_index] = selected
        fold_rows[selected] += len(rows)
        fold_types[selected].update(counts)

    if any(value < 0 for value in assignments):
        raise AssertionError("not every row received a fold")
    return assignments


def audit_entity_disjoint_folds(entity_rows, assignments, link_types=LINK_TYPES):
    """Report fold balance and any exact linked-entity leakage."""
    entity_rows = [list(row) for row in entity_rows]
    if len(entity_rows) != len(assignments):
        raise ValueError("row/assignment length mismatch")
    folds_by_key = defaultdict(set)
    for fold, entities in zip(assignments, entity_rows):
        for surface, kind in set(entities):
            if kind not in link_types:
                continue
            normalized = normalize_entity_surface(surface)
            if len(normalized) >= 3:
                folds_by_key[(kind, normalized)].add(int(fold))
    leakage = [
        {"type": kind, "surface_key": surface, "folds": sorted(folds)}
        for (kind, surface), folds in folds_by_key.items() if len(folds) > 1
    ]
    return {
        "rows_by_fold": dict(sorted(Counter(map(int, assignments)).items())),
        "linked_entity_keys": len(folds_by_key),
        "leaking_entity_keys": leakage,
    }
