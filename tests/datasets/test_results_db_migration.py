"""Tests for ResultsDB schema migration (Phase 2 → Phase 3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from neuroflux_datasets.cli import ResultsDB, RunResult


class TestResultsDBMigration:
    """Verify that old 8-column Phase 2 databases upgrade cleanly."""

    def _create_old_db(self, db_path: Path) -> None:
        """Create a Phase 2-era 8-column database with one row."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT,
                flicker_freq REAL,
                input_snr REAL,
                epochs INTEGER,
                stdp_tau REAL,
                seed INTEGER,
                delta_snr REAL,
                convergence_speed INTEGER
            )
        """)
        conn.execute(
            """
            INSERT INTO runs
            (signal_type, flicker_freq, input_snr, epochs,
             stdp_tau, seed, delta_snr, convergence_speed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ecg", 50.0, 10.0, 5, 20.0, 42, 4.5, 3),
        )
        conn.commit()
        conn.close()

    def test_old_db_gains_new_columns(self, tmp_path: Path) -> None:
        """Opening an old DB adds dataset, record_id, rpeak_integrity columns."""
        db_path = tmp_path / "old.db"
        self._create_old_db(db_path)

        # Open with the new ResultsDB — triggers _migrate()
        db = ResultsDB(db_path)

        cursor = db.conn.execute("PRAGMA table_info(runs)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "dataset" in columns
        assert "record_id" in columns
        assert "rpeak_integrity" in columns

    def test_old_rows_readable_after_migration(self, tmp_path: Path) -> None:
        """Pre-existing rows are still readable with correct values."""
        db_path = tmp_path / "old.db"
        self._create_old_db(db_path)

        db = ResultsDB(db_path)

        cursor = db.conn.execute(
            "SELECT signal_type, flicker_freq, delta_snr, dataset, "
            "record_id, rpeak_integrity FROM runs"
        )
        row = cursor.fetchone()

        assert row[0] == "ecg"
        assert row[1] == 50.0
        assert row[2] == 4.5
        # New columns get defaults / NULL for old rows
        assert row[3] == "synthetic"  # DEFAULT 'synthetic'
        assert row[4] is None  # record_id NULL
        assert row[5] is None  # rpeak_integrity NULL

    def test_insert_after_migration(self, tmp_path: Path) -> None:
        """New rows with all 11 columns can be inserted after migration."""
        db_path = tmp_path / "old.db"
        self._create_old_db(db_path)

        db = ResultsDB(db_path)

        new_result = RunResult(
            signal_type="ecg",
            flicker_freq=60.0,
            input_snr=15.0,
            epochs=10,
            stdp_tau=25.0,
            seed=99,
            delta_snr=5.0,
            convergence_speed=4,
            dataset="mit-bih",
            record_id="100",
            rpeak_integrity=0.95,
        )
        db.insert_result(new_result)

        cursor = db.conn.execute(
            "SELECT dataset, record_id, rpeak_integrity FROM runs WHERE seed = 99"
        )
        row = cursor.fetchone()
        assert row[0] == "mit-bih"
        assert row[1] == "100"
        assert row[2] == 0.95

    def test_fresh_db_has_all_columns(self, tmp_path: Path) -> None:
        """A brand-new database has all 11 columns from the start."""
        db = ResultsDB(tmp_path / "fresh.db")

        cursor = db.conn.execute("PRAGMA table_info(runs)")
        columns = {row[1] for row in cursor.fetchall()}

        expected = {
            "id",
            "signal_type",
            "flicker_freq",
            "input_snr",
            "epochs",
            "stdp_tau",
            "seed",
            "delta_snr",
            "convergence_speed",
            "dataset",
            "record_id",
            "rpeak_integrity",
        }
        assert columns == expected

    def test_double_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Running _migrate() twice does not error or duplicate columns."""
        db_path = tmp_path / "old.db"
        self._create_old_db(db_path)

        db = ResultsDB(db_path)
        # Manually call _migrate again — should be a no-op
        db._migrate()

        cursor = db.conn.execute("PRAGMA table_info(runs)")
        columns = [row[1] for row in cursor.fetchall()]
        # No duplicates
        assert len(columns) == len(set(columns))

    def test_phase2_run_result_defaults(self, tmp_path: Path) -> None:
        """RunResult without new fields uses correct defaults."""
        db = ResultsDB(tmp_path / "test.db")

        # Phase 2-style RunResult — only the original 8 fields
        result = RunResult(
            signal_type="sine",
            flicker_freq=50.0,
            input_snr=10.0,
            epochs=3,
            stdp_tau=20.0,
            seed=0,
            delta_snr=3.0,
            convergence_speed=2,
        )
        db.insert_result(result)

        cursor = db.conn.execute("SELECT dataset, record_id, rpeak_integrity FROM runs")
        row = cursor.fetchone()
        assert row[0] == "synthetic"
        assert row[1] is None
        assert row[2] is None
