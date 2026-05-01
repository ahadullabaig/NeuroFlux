import numpy as np
import pytest
from neuroflux_core.metrics.accuracy import (
    classification_accuracy,
    winner_take_all,
)


def test_winner_take_all_basic() -> None:
    assert winner_take_all(np.array([1, 5, 3, 2])) == 1
    assert winner_take_all(np.array([10.0, 0.0, 0.0])) == 0


def test_winner_take_all_ties_pick_first() -> None:
    # numpy argmax returns first max
    assert winner_take_all(np.array([3, 3, 3])) == 0


def test_winner_take_all_validates_shape() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        winner_take_all(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="non-empty"):
        winner_take_all(np.array([]))


def test_classification_accuracy() -> None:
    p = np.array([0, 1, 2, 1])
    t = np.array([0, 1, 1, 1])
    assert classification_accuracy(p, t) == 0.75


def test_classification_accuracy_perfect() -> None:
    p = np.array([1, 2, 3])
    assert classification_accuracy(p, p) == 1.0


def test_classification_accuracy_validates() -> None:
    with pytest.raises(ValueError, match="same shape"):
        classification_accuracy(np.array([1, 2]), np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="non-empty"):
        classification_accuracy(np.array([]), np.array([]))
