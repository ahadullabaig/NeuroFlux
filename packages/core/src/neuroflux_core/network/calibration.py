"""
Per-layer synaptic gain calibration.

The `g_syn` parameter scales the weighted dot-product input current to
mV-equivalent values. Its default (30.0) is implicitly tuned for a 32-wide
input layer: change the layer widths and the same gain either saturates the
network or silences it, because the fan-in changes but the gain does not.
Any experiment that varies network width therefore confounds "capacity" with
"operating point" unless gain is recalibrated per layer.

This is a known and solved problem in the ANN-to-SNN conversion literature —
weight normalisation and threshold balancing (Diehl et al., IJCNN 2015),
robust percentile normalisation (Rueckauer et al., Front. Neurosci. 11:682,
2017), and threshold-dependent BatchNorm (Zheng et al., AAAI 2021). Nothing
here is novel; it applies established practice to the unsupervised-STDP
setting, where there is no reference activation distribution to normalise
against and so the calibration has to be measured directly.

See `docs/related_work.md` §C and `docs/paper_roadmap.md` R4/M1b.
"""

import numpy as np

from neuroflux_core.network.feedforward import FeedForwardNetwork


def measure_layer_rates(
    network: FeedForwardNetwork,
    input_spikes: np.ndarray,
    dt: float,
) -> list[float]:
    """
    Mean firing rate of each neuron group, in Hz.

    Runs the network forward with plasticity disabled and resets its state
    first, so the result depends only on the weights, the gains and the input.

    Args:
        network: The network to measure. Not modified.
        input_spikes: Boolean array of shape (T, n_input).
        dt: Timestep in ms.

    Returns:
        One mean rate in Hz per neuron group, ordered input-side first.

    Raises:
        ValueError: If `input_spikes` is not 2-D or its width does not match
            the network's input layer.
    """
    if input_spikes.ndim != 2:
        raise ValueError(
            f"input_spikes must be 2-D (T, n_input), got shape {input_spikes.shape}."
        )
    if input_spikes.shape[1] != network.layers[0]:
        raise ValueError(
            f"input_spikes has width {input_spikes.shape[1]} but the input layer "
            f"is {network.layers[0]}."
        )

    n_steps = input_spikes.shape[0]
    counts = np.zeros(len(network.neurons), dtype=np.float64)

    network.reset_state()
    for t in range(n_steps):
        all_spikes = network.step(input_spikes[t], dt, training=False)
        # all_spikes[0] is the input; neuron groups start at index 1.
        for layer_idx, spikes in enumerate(all_spikes[1:]):
            counts[layer_idx] += float(np.count_nonzero(spikes))

    duration_s = n_steps * dt / 1000.0
    return [
        float(counts[i] / (network.neurons[i].n_neurons * duration_s))
        for i in range(len(network.neurons))
    ]


def calibrate_gain(
    network: FeedForwardNetwork,
    input_spikes: np.ndarray,
    dt: float,
    target_rate_hz: float = 40.0,
    tol_hz: float = 0.5,
    max_iter: int = 30,
    bounds: tuple[float, float] = (1e-3, 5.0e3),
) -> list[float]:
    """
    Calibrate each layer's synaptic gain so every layer fires near a target rate.

    Layers are calibrated input-side first and then frozen, because a layer's
    firing rate is monotone in its own gain only when the layers feeding it are
    held fixed. Search is geometric bisection (bisection on log gain), which
    suits a parameter spanning several orders of magnitude across widths.

    The calibrated gains are applied to `network` in place and also returned.
    Plasticity is disabled throughout, so weights are untouched.

    Args:
        network: Network to calibrate. Modified in place via `set_layer_gains`.
        input_spikes: Boolean array of shape (T, n_input) — a representative
            input, ideally the same signal the experiment will run on.
        dt: Timestep in ms.
        target_rate_hz: Firing rate each layer is driven toward.
        tol_hz: Stop early once a layer is within this many Hz of target.
        max_iter: Bisection steps per layer.
        bounds: (low, high) gain bracket. Both must be strictly positive.

    Returns:
        One calibrated gain per neuron group.

    Raises:
        ValueError: If `target_rate_hz` is not positive, or `bounds` is not a
            valid strictly-positive increasing bracket.

    Note:
        A layer whose rate cannot reach the target within `bounds` returns the
        closest achievable gain rather than raising. Check the achieved rates
        with `measure_layer_rates` — a layer that stays far from target usually
        means the input is too sparse to drive it at any gain.
    """
    if target_rate_hz <= 0.0:
        raise ValueError(f"target_rate_hz must be positive, got {target_rate_hz}.")
    low_bound, high_bound = bounds
    if not 0.0 < low_bound < high_bound:
        raise ValueError(f"bounds must satisfy 0 < low < high, got {bounds}.")

    gains = list(network.layer_gains)

    for layer_idx in range(len(network.neurons)):
        low, high = low_bound, high_bound

        for _ in range(max_iter):
            mid = float(np.sqrt(low * high))
            trial = list(gains)
            trial[layer_idx] = mid
            network.set_layer_gains(trial)

            rate = measure_layer_rates(network, input_spikes, dt)[layer_idx]

            if abs(rate - target_rate_hz) < tol_hz:
                low, high = mid, mid
                break
            if rate < target_rate_hz:
                low = mid
            else:
                high = mid

        gains[layer_idx] = float(np.sqrt(low * high))
        network.set_layer_gains(gains)

    return gains
