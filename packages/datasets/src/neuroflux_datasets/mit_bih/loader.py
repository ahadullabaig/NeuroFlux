"""Subject-held-out train/test splitter for MIT-BIH records.

Provides a deterministic, **record-wise** (subject-wise) split that
guarantees no information leakage across the train/test boundary.

Per ``PLAN.md`` line 277: *"never by time-window within subject — this
leaks information."*  Every time window from a given record lands
entirely in either the train set or the test set.
"""

from __future__ import annotations

import numpy as np


def train_test_split(
    record_ids: list[str],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split MIT-BIH record IDs into train and test sets.

    The split is **record-wise** (equivalently subject-wise for MIT-BIH,
    where each record corresponds to a distinct patient).  This prevents
    information leakage that would occur if time windows from the same
    record appeared in both sets.

    The split is fully deterministic for a given *seed* and
    *record_ids* list (order-independent — IDs are sorted internally).

    Args:
        record_ids: List of MIT-BIH record identifiers
            (e.g. ``["100", "101", "102", ...]``).
        test_fraction: Fraction of records to assign to the test set.
            Must be in the open interval ``(0, 1)``.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(train_ids, test_ids)`` of disjoint, sorted lists.

    Raises:
        ValueError: If *record_ids* is empty or *test_fraction* is
            not in ``(0, 1)``.
    """
    if not record_ids:
        raise ValueError("record_ids must be non-empty.")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}.")

    # Sort to make the split order-independent.
    sorted_ids = sorted(record_ids)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted_ids).tolist()

    n_test = max(1, round(len(shuffled) * test_fraction))
    n_test = min(n_test, len(shuffled) - 1)  # keep ≥1 in train

    test_ids: list[str] = sorted(shuffled[:n_test])
    train_ids: list[str] = sorted(shuffled[n_test:])

    return train_ids, test_ids
