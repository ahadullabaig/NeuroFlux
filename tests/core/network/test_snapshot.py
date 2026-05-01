"""Snapshot regression tests — pin exact bit-level outputs for canonical configs.

Per PLAN.md §"Test strategy": "Outputs of canonical configs are snapshot-tested.
Any change must be deliberate and documented." If these change, update the
expected values *only* after confirming the change is intentional.
"""

import hashlib

import numpy as np
from neuroflux_core import Network, Simulator


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_canonical_small_network_snapshot() -> None:
    """Layer [10,5], uniform init, seed=42, signal=0.6, dt=1.0, 200 ms training."""
    net = Network.feedforward(layers=[10, 5], seed=42, init_scheme="uniform")
    sim = Simulator(net, dt=1.0, seed=42)
    signal = np.full(10, 0.6)
    out = sim.run(signal, duration_ms=200.0, training=True)

    w0 = sim.network.synapses[0].weights

    expected_w0_sha = "a5e3b453743a8689a0dd97bd1f27806f25150fc49af3220f5c168d1750ceb2a2"
    expected_out_sha = (
        "50f8638646b0c69f4727ffae2b2e02eba67059cbfa205c0129ec41f4b9b425b4"
    )
    expected_w0_sum = 26.8343830109
    expected_spike_total = 4

    assert _sha256(w0) == expected_w0_sha, (
        "Layer-0 weight matrix changed unexpectedly. If intentional, update "
        "the snapshot."
    )
    assert _sha256(np.packbits(out.astype(np.uint8))) == expected_out_sha, (
        "Output spike train changed unexpectedly. If intentional, update."
    )
    assert abs(float(w0.sum()) - expected_w0_sum) < 1e-5
    assert int(out.sum()) == expected_spike_total


def test_canonical_snapshot_reproduces_across_runs() -> None:
    """Same seed → identical snapshot, twice in a row."""

    def runit() -> tuple[str, int]:
        net = Network.feedforward(layers=[10, 5], seed=42, init_scheme="uniform")
        sim = Simulator(net, dt=1.0, seed=42)
        out = sim.run(np.full(10, 0.6), duration_ms=200.0, training=True)
        return _sha256(sim.network.synapses[0].weights), int(out.sum())

    a = runit()
    b = runit()
    assert a == b
