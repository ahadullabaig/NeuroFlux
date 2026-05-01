"""Connection-pattern initialization helpers for synaptic weight matrices."""

from typing import Literal

import numpy as np

InitScheme = Literal["uniform", "gaussian", "sparse", "scaled"]


def init_weights(
    shape: tuple[int, int],
    scheme: InitScheme = "scaled",
    rng: np.random.Generator | None = None,
    w_min: float = 0.0,
    w_max: float = 1.0,
    *,
    mean: float | None = None,
    std: float | None = None,
    density: float = 0.1,
    scale: float = 0.3,
) -> np.ndarray:
    """
    Initialize a (n_pre, n_post) weight matrix according to a named scheme.

    Schemes:
        "uniform": Uniform[w_min, w_max].
        "gaussian": Normal(mean, std), clipped to [w_min, w_max].
            Defaults: mean=midpoint, std=range/6.
        "sparse": Uniform[w_min, w_max] with `density` fraction of nonzero entries
            (rest set to 0).
        "scaled": Uniform[w_min, w_min + scale*(w_max - w_min)] — small initial values
            recommended for STDP regimes (default scheme).

    Args:
        shape: (n_pre, n_post).
        scheme: One of "uniform", "gaussian", "sparse", "scaled".
        rng: Optional Generator for reproducibility.
        w_min: Lower weight bound.
        w_max: Upper weight bound.
        mean: Mean for gaussian (default: midpoint of [w_min, w_max]).
        std: Std for gaussian (default: range/6).
        density: Nonzero fraction for sparse.
        scale: Initial range fraction for scaled.

    Returns:
        Float32 weight matrix of given shape.
    """
    if w_max <= w_min:
        raise ValueError("w_max must be strictly greater than w_min")

    gen = rng if rng is not None else np.random.default_rng()

    if scheme == "uniform":
        w = gen.uniform(w_min, w_max, size=shape)
    elif scheme == "gaussian":
        m = mean if mean is not None else (w_min + w_max) / 2.0
        s = std if std is not None else (w_max - w_min) / 6.0
        w = gen.normal(m, s, size=shape)
        w = np.clip(w, w_min, w_max)
    elif scheme == "sparse":
        if not 0.0 < density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        w = gen.uniform(w_min, w_max, size=shape)
        mask = gen.random(shape) < density
        w = np.where(mask, w, 0.0)
    elif scheme == "scaled":
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        upper = w_min + scale * (w_max - w_min)
        w = gen.uniform(w_min, upper, size=shape)
    else:
        raise ValueError(f"Unknown init scheme: {scheme!r}")

    return w.astype(np.float32)
