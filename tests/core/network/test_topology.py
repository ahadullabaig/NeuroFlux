import numpy as np
import pytest
from neuroflux_core.network.topology import init_weights


def test_uniform_scheme_bounds_and_dtype() -> None:
    rng = np.random.default_rng(0)
    w = init_weights((50, 50), scheme="uniform", rng=rng, w_min=0.0, w_max=1.0)
    assert w.shape == (50, 50)
    assert w.dtype == np.float32
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0)


def test_gaussian_scheme_clipped_to_bounds() -> None:
    rng = np.random.default_rng(0)
    w = init_weights(
        (100, 100), scheme="gaussian", rng=rng, w_min=0.2, w_max=0.8, std=0.5
    )
    assert np.all(w >= 0.2)
    assert np.all(w <= 0.8)


def test_sparse_scheme_density() -> None:
    rng = np.random.default_rng(0)
    w = init_weights(
        (100, 100), scheme="sparse", rng=rng, w_min=0.1, w_max=0.9, density=0.2
    )
    nonzero_frac = np.mean(w != 0.0)
    assert 0.15 < nonzero_frac < 0.25


def test_scaled_scheme_bounds() -> None:
    rng = np.random.default_rng(0)
    w = init_weights(
        (50, 50), scheme="scaled", rng=rng, w_min=0.0, w_max=1.0, scale=0.3
    )
    assert np.all(w >= 0.0)
    assert np.all(w <= 0.3 + 1e-6)


def test_reproducibility() -> None:
    a = init_weights((10, 10), scheme="uniform", rng=np.random.default_rng(7))
    b = init_weights((10, 10), scheme="uniform", rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_invalid_scheme() -> None:
    with pytest.raises(ValueError, match="Unknown init scheme"):
        init_weights((5, 5), scheme="bogus")  # type: ignore[arg-type]


def test_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        init_weights((5, 5), w_min=1.0, w_max=1.0)


def test_invalid_density() -> None:
    with pytest.raises(ValueError, match="density"):
        init_weights((5, 5), scheme="sparse", density=0.0)


def test_invalid_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        init_weights((5, 5), scheme="scaled", scale=1.5)
