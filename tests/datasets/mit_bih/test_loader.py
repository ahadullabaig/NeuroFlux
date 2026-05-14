"""Tests for MIT-BIH subject-held-out train/test splitter."""

from __future__ import annotations

import pytest
from neuroflux_datasets.mit_bih.loader import train_test_split

# A representative subset of the 48 MIT-BIH record IDs.
_SAMPLE_IDS: list[str] = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
]


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------


class TestDeterminism:
    """Split must be fully reproducible given the same inputs."""

    def test_same_seed_same_split(self) -> None:
        """Identical seed + IDs → identical split."""
        train1, test1 = train_test_split(_SAMPLE_IDS, seed=42)
        train2, test2 = train_test_split(_SAMPLE_IDS, seed=42)

        assert train1 == train2
        assert test1 == test2

    def test_different_seed_different_split(self) -> None:
        """Different seeds produce different splits."""
        _, test1 = train_test_split(_SAMPLE_IDS, seed=0)
        _, test2 = train_test_split(_SAMPLE_IDS, seed=999)

        assert test1 != test2

    def test_order_independent(self) -> None:
        """Shuffled input order produces the same split (sorted internally)."""
        shuffled = list(reversed(_SAMPLE_IDS))
        train1, test1 = train_test_split(_SAMPLE_IDS, seed=7)
        train2, test2 = train_test_split(shuffled, seed=7)

        assert train1 == train2
        assert test1 == test2


# -----------------------------------------------------------------------
# No leakage
# -----------------------------------------------------------------------


class TestNoLeakage:
    """Train and test sets must be strictly disjoint."""

    def test_disjoint_sets(self) -> None:
        """train ∩ test = ∅."""
        train, test = train_test_split(_SAMPLE_IDS, seed=0)
        overlap = set(train) & set(test)
        assert overlap == set(), f"Leakage detected: {overlap}"

    def test_union_equals_input(self) -> None:
        """train ∪ test = input (no records lost)."""
        train, test = train_test_split(_SAMPLE_IDS, seed=0)
        assert sorted(train + test) == sorted(_SAMPLE_IDS)

    def test_no_duplicates_in_train(self) -> None:
        """Train set contains no duplicate IDs."""
        train, _ = train_test_split(_SAMPLE_IDS, seed=0)
        assert len(train) == len(set(train))

    def test_no_duplicates_in_test(self) -> None:
        """Test set contains no duplicate IDs."""
        _, test = train_test_split(_SAMPLE_IDS, seed=0)
        assert len(test) == len(set(test))


# -----------------------------------------------------------------------
# Fraction sizing
# -----------------------------------------------------------------------


class TestFractionSizing:
    """Test fraction controls the approximate size of each set."""

    def test_default_fraction_is_20_percent(self) -> None:
        """Default test_fraction=0.2 → ~20% of records in test."""
        train, test = train_test_split(_SAMPLE_IDS)
        # 20 records × 0.2 = 4 test records
        assert len(test) == 4
        assert len(train) == 16

    def test_custom_fraction(self) -> None:
        """test_fraction=0.3 → ~30% of records in test."""
        train, test = train_test_split(_SAMPLE_IDS, test_fraction=0.3)
        # 20 × 0.3 = 6
        assert len(test) == 6
        assert len(train) == 14

    def test_at_least_one_in_each_set(self) -> None:
        """Even with extreme fractions, both sets are non-empty."""
        # Very small fraction
        train, test = train_test_split(_SAMPLE_IDS, test_fraction=0.01)
        assert len(train) >= 1
        assert len(test) >= 1

        # Very large fraction
        train, test = train_test_split(_SAMPLE_IDS, test_fraction=0.99)
        assert len(train) >= 1
        assert len(test) >= 1

    def test_two_records_splits_one_each(self) -> None:
        """Minimum viable split: 2 records → 1 train, 1 test."""
        train, test = train_test_split(["100", "101"], test_fraction=0.5)
        assert len(train) == 1
        assert len(test) == 1


# -----------------------------------------------------------------------
# Output format
# -----------------------------------------------------------------------


class TestOutputFormat:
    """Return values are sorted lists of strings."""

    def test_returns_sorted_lists(self) -> None:
        """Both train and test lists are sorted."""
        train, test = train_test_split(_SAMPLE_IDS, seed=0)
        assert train == sorted(train)
        assert test == sorted(test)

    def test_returns_strings(self) -> None:
        """All elements are strings."""
        train, test = train_test_split(_SAMPLE_IDS, seed=0)
        for rid in train + test:
            assert isinstance(rid, str)


# -----------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------


class TestErrorHandling:
    """Invalid inputs raise clear errors."""

    def test_empty_ids_raises(self) -> None:
        """Empty record_ids raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            train_test_split([])

    def test_fraction_zero_raises(self) -> None:
        """test_fraction=0 raises ValueError."""
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_SAMPLE_IDS, test_fraction=0.0)

    def test_fraction_one_raises(self) -> None:
        """test_fraction=1 raises ValueError."""
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_SAMPLE_IDS, test_fraction=1.0)

    def test_fraction_negative_raises(self) -> None:
        """Negative test_fraction raises ValueError."""
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(_SAMPLE_IDS, test_fraction=-0.5)
