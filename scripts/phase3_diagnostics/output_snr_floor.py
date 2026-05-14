"""Output SNR floor diagnostic for Phase 3.

Tests the hypothesis from ``docs/phase3_findings.md`` §4.1: the SNN
imposes a fixed output quality floor (~10-12 dB) regardless of input
quality, because the rate-coded 8-neuron population decoder has a
bounded information-theoretic capacity.

The protocol:

1. For each cached MIT-BIH record, extract a clean 10 s segment via
   :func:`neuroflux_datasets.mit_bih.annotator.extract_clean_segments`.
2. Feed the **clean** signal (no flicker, no Gaussian noise) into the
   default-config SNN (``[32, 16, 8]``) for 50 epochs of STDP training.
3. Run a final eval pass and measure
   ``calculate_snr(clean_input, snn_reconstruction)``.  This is the
   intrinsic *output SNR* — the quality ceiling above which the network
   cannot reconstruct.
4. Aggregate per-record and overall; persist results.

Run::

    uv run python scripts/phase3_diagnostics/output_snr_floor.py

Outputs:
- ``results/phase3_diagnostics/output_snr_floor.csv``
- ``results/phase3_diagnostics/output_snr_floor_summary.json``
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from neuroflux_core.decoders.population import PopulationDecoder
from neuroflux_core.encoders.rate import PoissonRateEncoder
from neuroflux_core.io.config import LIFConfig, STDPConfig
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_core.network.feedforward import FeedForwardNetwork, Network
from neuroflux_core.network.simulator import Simulator
from neuroflux_datasets.mit_bih.annotator import extract_clean_segments

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "mit_bih"
OUT_DIR = REPO_ROOT / "results" / "phase3_diagnostics"

RECORD_IDS = [
    "100",
    "101",
    "103",
    "105",
    "108",
    "112",
    "113",
    "119",
    "200",
    "207",
    "212",
    "214",
    "228",
    "231",
    "232",
    "234",
]
SEEDS = list(range(5))
DURATION_S = 10.0
FS = 250.0
DT = 1000.0 / FS
EPOCHS = 50
NETWORK_LAYERS = [32, 16, 8]


def _reset_transient_state(sim: Simulator) -> None:
    assert isinstance(sim.network, FeedForwardNetwork)
    sim.time = 0.0
    sim.encoder.reset_state()
    sim.decoder.reset_state()
    for n in sim.network.neurons:
        n.v.fill(n.config.v_rest)
        n.refractory_time.fill(0.0)
    for s in sim.network.synapses:
        s.trace_pre.fill(0.0)
        s.trace_post.fill(0.0)


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    """Run the SNR-floor measurement for one (record, seed) pair."""
    record_id = task["record_id"]
    seed = task["seed"]

    segments = extract_clean_segments(
        record_id,
        CACHE_DIR,
        duration_s=DURATION_S,
        fs=FS,
    )
    if not segments:
        return {
            "record_id": record_id,
            "seed": seed,
            "status": "no_clean_segment",
            "output_snr_db": None,
        }

    clean = segments[seed % len(segments)]
    n_input = NETWORK_LAYERS[0]

    v_min = float(np.min(clean))
    v_max = float(np.max(clean))
    if v_max <= v_min:
        v_max = v_min + 1.0

    net = Network.feedforward(
        layers=NETWORK_LAYERS,
        neuron=LIFConfig(),
        plasticity=STDPConfig(),
        seed=seed,
    )
    encoder = PoissonRateEncoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=100.0,
        rng=np.random.default_rng(seed),
    )
    decoder = PopulationDecoder(
        v_min=v_min,
        v_max=v_max,
        max_firing_rate_hz=100.0,
    )
    sim = Simulator(net, dt=DT, seed=seed, encoder=encoder, decoder=decoder)
    assert isinstance(sim.network, FeedForwardNetwork)

    # The same tiling that run_single_experiment uses.
    tiled = np.tile(clean, (n_input, 1)).T
    duration_ms = DURATION_S * 1000.0

    # Train on the clean signal.
    for _ in range(EPOCHS):
        sim.run(tiled, duration_ms=duration_ms, training=True)
        _reset_transient_state(sim)

    # Final eval pass.
    out_spikes = sim.run(tiled, duration_ms=duration_ms, training=False)
    reconstructed = sim.decode(out_spikes)
    output_snr = calculate_snr(clean, reconstructed)

    return {
        "record_id": record_id,
        "seed": seed,
        "status": "ok",
        "output_snr_db": float(output_snr),
        "clean_min": v_min,
        "clean_max": v_max,
    }


def main() -> int:
    if not CACHE_DIR.is_dir():
        print(f"ERROR: cache dir missing at {CACHE_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [{"record_id": rid, "seed": seed} for rid in RECORD_IDS for seed in SEEDS]
    print(
        f"Output SNR floor diagnostic: {len(tasks)} measurements "
        f"({len(RECORD_IDS)} records x {len(SEEDS)} seeds, "
        f"{EPOCHS} epochs each, network={NETWORK_LAYERS})"
    )

    t0 = time.time()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor() as exe:
        futures = {exe.submit(_run_one, t): t for t in tasks}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
            if len(results) % 10 == 0:
                print(f"  completed {len(results)} / {len(tasks)}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    # CSV
    csv_path = OUT_DIR / "output_snr_floor.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"Per-row CSV → {csv_path}")

    # Summary
    ok_rows = [r for r in results if r["status"] == "ok"]
    by_record: dict[str, list[float]] = {}
    for r in ok_rows:
        by_record.setdefault(r["record_id"], []).append(r["output_snr_db"])

    summary = {
        "n_total": len(results),
        "n_ok": len(ok_rows),
        "epochs_per_run": EPOCHS,
        "network_layers": NETWORK_LAYERS,
        "duration_s": DURATION_S,
        "fs": FS,
        "mean_output_snr_db": (
            statistics.mean(r["output_snr_db"] for r in ok_rows) if ok_rows else None
        ),
        "stdev_output_snr_db": (
            statistics.stdev(r["output_snr_db"] for r in ok_rows)
            if len(ok_rows) > 1
            else None
        ),
        "min_output_snr_db": min((r["output_snr_db"] for r in ok_rows), default=None),
        "max_output_snr_db": max((r["output_snr_db"] for r in ok_rows), default=None),
        "per_record": {
            rid: {
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            for rid, vals in by_record.items()
        },
    }
    summary_path = OUT_DIR / "output_snr_floor_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {summary_path}")

    if summary["mean_output_snr_db"] is not None:
        print()
        print(
            f"Headline: mean output SNR on clean signal = "
            f"{summary['mean_output_snr_db']:.2f} dB "
            f"(stdev {summary['stdev_output_snr_db']:.2f}, "
            f"range [{summary['min_output_snr_db']:.2f}, "
            f"{summary['max_output_snr_db']:.2f}])"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
