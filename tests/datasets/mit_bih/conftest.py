"""Shared fixtures for MIT-BIH downloader tests.

All ``wfdb`` access is mocked here so that tests never hit the network.
The ``mock_wfdb`` fixture monkeypatches ``wfdb.dl_database`` to create
dummy ``.hea`` and ``.dat`` files in the target directory.

Additional fixtures mock ``wfdb.rdrecord`` and ``wfdb.rdann`` for the
annotator / corruption-pipeline tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def _fake_dl_database(
    db_dir: str,
    dl_dir: str,
    records: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Simulate ``wfdb.dl_database`` by writing stub files."""
    dl_path = Path(dl_dir)
    dl_path.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = []
    for rid in records:
        (dl_path / f"{rid}.hea").write_text(
            f"{rid} 2 360 650000\n"
            f"{rid}.dat 212 200 11 1024 0 0 0 MLII\n"
            f"{rid}.dat 212 200 11 1024 0 0 0 V5\n"
        )
        (dl_path / f"{rid}.dat").write_bytes(
            b"\x00" * 256  # deterministic stub payload
        )


@pytest.fixture()
def mock_wfdb(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch ``wfdb.dl_database`` so no network I/O occurs.

    Also stubs ``_load_manifest`` to return an empty record map. This keeps
    caching / provenance tests independent of the committed manifest's real
    checksums — tests that explicitly exercise checksum verification override
    this stub locally (see ``TestDownloadRecordChecksum``).

    Returns the mock object so tests can assert call counts / args.
    """
    import wfdb  # type: ignore[import-untyped]

    mock = MagicMock(side_effect=_fake_dl_database)
    monkeypatch.setattr(wfdb, "dl_database", mock)
    monkeypatch.setattr(
        "neuroflux_datasets.mit_bih.downloader._load_manifest",
        lambda: {"records": {}},
    )
    return mock


# ---------------------------------------------------------------------------
# Annotator mocks — rdrecord + rdann
# ---------------------------------------------------------------------------

# Duration of the fake record in seconds (at 360 Hz).
_FAKE_DURATION_S: float = 60.0
_FAKE_FS: float = 360.0
_FAKE_N_SAMPLES: int = int(_FAKE_DURATION_S * _FAKE_FS)  # 21600

# Position (sample index) of a single artifact marker in the fake record.
# Placed at 25 s — this lets tests verify that windows overlapping this
# position are correctly excluded.
_ARTIFACT_SAMPLE: int = int(25.0 * _FAKE_FS)  # 9000


@dataclass
class _FakeRecord:
    """Minimal stand-in for ``wfdb.Record`` with a ``.p_signal`` array."""

    p_signal: np.ndarray
    fs: float = _FAKE_FS


@dataclass
class _FakeAnnotation:
    """Minimal stand-in for ``wfdb.Annotation``."""

    sample: np.ndarray
    symbol: list[str] = field(default_factory=list)


def _build_fake_signal(n_samples: int, fs: float) -> np.ndarray:
    """Generate a 2-channel ECG-like signal for testing.

    Channel 0: 1 Hz sine (simulates MLII lead).
    Channel 1: 2 Hz sine (simulates V5 lead).
    """
    t = np.arange(n_samples) / fs
    ch0 = np.sin(2.0 * np.pi * 1.0 * t)
    ch1 = np.sin(2.0 * np.pi * 2.0 * t)
    return np.column_stack([ch0, ch1])


@pytest.fixture()
def mock_rdrecord(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch ``wfdb.rdrecord`` to return a fake single-channel record.

    The fake record is a 60 s, 360 Hz sine wave (1 Hz on channel 0).
    The ``channels`` kwarg is honoured: ``channels=[0]`` returns only
    column 0, matching real ``wfdb.rdrecord`` behaviour.
    """
    import wfdb

    full_signal = _build_fake_signal(_FAKE_N_SAMPLES, _FAKE_FS)

    def _fake_rdrecord(
        record_name: str,
        channels: list[int] | None = None,
        **kwargs: Any,
    ) -> _FakeRecord:
        if channels is not None:
            sig = full_signal[:, channels]
        else:
            sig = full_signal
        return _FakeRecord(p_signal=sig, fs=_FAKE_FS)

    mock = MagicMock(side_effect=_fake_rdrecord)
    monkeypatch.setattr(wfdb, "rdrecord", mock)
    return mock


@pytest.fixture()
def mock_rdann(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch ``wfdb.rdann`` to return fake annotations.

    Beat annotations:
    - Normal beats ('N') every 1 s (360 samples apart), total 60.
    - One artifact marker ('~') at sample 9000 (25 s mark).
    """
    import wfdb

    # Normal beats at 1 s intervals
    beat_positions = np.arange(0, _FAKE_N_SAMPLES, int(_FAKE_FS))
    beat_symbols = ["N"] * len(beat_positions)

    # Insert artifact at 25 s
    all_positions = np.append(beat_positions, _ARTIFACT_SAMPLE)
    all_symbols = beat_symbols + ["~"]

    # Sort by sample position (required by wfdb convention)
    sort_idx = np.argsort(all_positions)
    sorted_positions = all_positions[sort_idx]
    sorted_symbols = [all_symbols[i] for i in sort_idx]

    def _fake_rdann(
        record_name: str,
        extension: str,
        **kwargs: Any,
    ) -> _FakeAnnotation:
        return _FakeAnnotation(
            sample=sorted_positions.copy(),
            symbol=list(sorted_symbols),
        )

    mock = MagicMock(side_effect=_fake_rdann)
    monkeypatch.setattr(wfdb, "rdann", mock)
    return mock


@pytest.fixture()
def mock_rdann_clean(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Like ``mock_rdann`` but with NO artifact markers.

    All annotations are normal beats ('N'). Useful for tests that
    need to verify behaviour on a fully clean record.
    """
    import wfdb

    beat_positions = np.arange(0, _FAKE_N_SAMPLES, int(_FAKE_FS))
    beat_symbols = ["N"] * len(beat_positions)

    def _fake_rdann_clean(
        record_name: str,
        extension: str,
        **kwargs: Any,
    ) -> _FakeAnnotation:
        return _FakeAnnotation(
            sample=beat_positions.copy(),
            symbol=list(beat_symbols),
        )

    mock = MagicMock(side_effect=_fake_rdann_clean)
    monkeypatch.setattr(wfdb, "rdann", mock)
    return mock
