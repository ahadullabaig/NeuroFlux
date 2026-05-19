"""Phase 3A — decision gate.

Three pre-locked gates that decide whether the dual-channel-first pivot
narrative is supported by the data.  Thresholds were locked in the plan
file before any sweep was run, so the gate cannot be goalpost-shifted
after the fact.

* **Gate A — dual-channel viability.**  At input SNR = 0 dB, the parallel
  passive-SNN R-peak channel must beat the best classical filter's rpeak
  by at least 0.10.  If it doesn't, classical filters already preserve
  R-peaks well enough on their own — the parallel channel doesn't earn
  its keep, and the dual-channel pitch collapses.

* **Gate B — SNN amplitude niche.**  At input SNR = 0 dB, there must
  exist *some* (epochs, tau) snn_trained config whose mean ΔSNR exceeds
  the best classical ΔSNR.  If not, the SNN has no amplitude regime to
  call its own — drop it from any routed pipeline.

* **Gate C — implementation sanity.**  At input SNR = 20 dB, Butterworth
  notch ΔSNR must be positive.  Otherwise the baseline implementation is
  broken and nothing downstream is trustworthy.

Run::

    uv run python scripts/phase3a/gate_check.py
"""

from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

DB_PATH = Path("results/phase3_full_sweep.db")
GATE_A_THRESHOLD = 0.10  # rpeak gap required at 0 dB
LOW_SNR = 0.0
HIGH_SNR = 20.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _query_floats(
    conn: sqlite3.Connection, sql: str, params: tuple[float | str, ...]
) -> list[float]:
    cursor = conn.execute(sql, params)
    return [float(r[0]) for r in cursor.fetchall() if r[0] is not None]


def _gate_a(conn: sqlite3.Connection) -> tuple[bool, float, float, str]:
    passive_rpeak = _mean(
        _query_floats(
            conn,
            "SELECT rpeak_integrity FROM runs "
            "WHERE method = 'snn_passive' AND input_snr = ?",
            (LOW_SNR,),
        )
    )
    classical_best = max(
        _mean(
            _query_floats(
                conn,
                "SELECT rpeak_integrity FROM runs WHERE method = ? AND input_snr = ?",
                (method, LOW_SNR),
            )
        )
        for method in ("butterworth_notch", "lms")
    )
    gap = passive_rpeak - classical_best
    passed = gap >= GATE_A_THRESHOLD
    detail = (
        f"passive_rpeak={passive_rpeak:.4f}, "
        f"best_classical_rpeak={classical_best:.4f}, gap={gap:+.4f}"
    )
    return passed, passive_rpeak, classical_best, detail


def _gate_b(conn: sqlite3.Connection) -> tuple[bool, float, float, str]:
    # Iterate over every (epochs, tau) cell that snn_trained covers.
    cursor = conn.execute(
        "SELECT DISTINCT epochs, stdp_tau FROM runs "
        "WHERE method = 'snn_trained' AND input_snr = ?",
        (LOW_SNR,),
    )
    configs = list(cursor.fetchall())

    best_snn = float("-inf")
    best_cfg: tuple[int, float] | None = None
    for epochs, tau in configs:
        deltas = _query_floats(
            conn,
            "SELECT delta_snr FROM runs "
            "WHERE method = 'snn_trained' "
            "AND input_snr = ? AND epochs = ? AND stdp_tau = ?",
            (LOW_SNR, epochs, tau),
        )
        m = _mean(deltas)
        if m > best_snn:
            best_snn = m
            best_cfg = (int(epochs), float(tau))

    classical_best = max(
        _mean(
            _query_floats(
                conn,
                "SELECT delta_snr FROM runs WHERE method = ? AND input_snr = ?",
                (method, LOW_SNR),
            )
        )
        for method in ("butterworth_notch", "lms")
    )

    passed = best_snn > classical_best
    cfg_str = f"epochs={best_cfg[0]}, tau={best_cfg[1]}" if best_cfg else "n/a"
    detail = (
        f"best_snn_delta={best_snn:+.4f} ({cfg_str}), "
        f"best_classical_delta={classical_best:+.4f}"
    )
    return passed, best_snn, classical_best, detail


def _gate_c(conn: sqlite3.Connection) -> tuple[bool, float, str]:
    bw_delta = _mean(
        _query_floats(
            conn,
            "SELECT delta_snr FROM runs "
            "WHERE method = 'butterworth_notch' AND input_snr = ?",
            (HIGH_SNR,),
        )
    )
    passed = bw_delta > 0.0
    detail = f"butterworth_delta@20dB={bw_delta:+.4f}"
    return passed, bw_delta, detail


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(
            f"Database not found at {DB_PATH}. Run the Phase 3A sweep first."
        )

    conn = sqlite3.connect(str(DB_PATH))
    try:
        a_pass, _, _, a_detail = _gate_a(conn)
        b_pass, _, _, b_detail = _gate_b(conn)
        c_pass, _, c_detail = _gate_c(conn)
    finally:
        conn.close()

    def status(p: bool) -> str:
        return "PASS" if p else "FAIL"

    print()
    print(f"GATE A (dual-channel viability):  {status(a_pass)}  ({a_detail})")
    print(f"GATE B (snn amplitude niche):     {status(b_pass)}  ({b_detail})")
    print(f"GATE C (implementation sanity):   {status(c_pass)}  ({c_detail})")
    print()

    if not c_pass:
        raise SystemExit(
            "Gate C failed — baseline implementation is broken. Halt before "
            "interpreting Gates A/B."
        )


if __name__ == "__main__":
    main()
