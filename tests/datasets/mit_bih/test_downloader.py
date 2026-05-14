"""Tests for MIT-BIH downloader: caching, checksum verification, provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from neuroflux_datasets.mit_bih.downloader import (
    IntegrityError,
    _load_manifest,
    compute_checksum,
    download_record,
    verify_records,
)

# -----------------------------------------------------------------------
# compute_checksum
# -----------------------------------------------------------------------


class TestComputeChecksum:
    """Unit tests for the SHA-256 checksum utility."""

    def test_known_content(self, tmp_path: Path) -> None:
        """Checksum of known bytes matches hashlib reference."""
        payload = b"neuroflux-test-content"
        f = tmp_path / "test.bin"
        f.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()
        assert compute_checksum(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file produces the well-known SHA-256 of empty input."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        assert compute_checksum(f) == expected

    def test_large_file(self, tmp_path: Path) -> None:
        """Checksum works correctly for files larger than the read buffer."""
        # 64 KiB > default 8 KiB chunk size, exercises the chunked read
        payload = b"\xab" * 65536
        f = tmp_path / "large.bin"
        f.write_bytes(payload)

        expected = hashlib.sha256(payload).hexdigest()
        assert compute_checksum(f) == expected


# -----------------------------------------------------------------------
# download_record — caching behaviour
# -----------------------------------------------------------------------


class TestDownloadRecordCaching:
    """Verify that download_record caches and skips re-download."""

    def test_first_download_creates_files(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
    ) -> None:
        """First call downloads and creates .hea + .dat files."""
        result = download_record("100", tmp_path)

        assert (tmp_path / "100.hea").exists()
        assert (tmp_path / "100.dat").exists()
        assert result == tmp_path.resolve()
        mock_wfdb.assert_called_once()

    def test_second_call_skips_download(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
    ) -> None:
        """Second call with cached files does not re-download."""
        download_record("100", tmp_path)
        mock_wfdb.reset_mock()

        download_record("100", tmp_path)
        mock_wfdb.assert_not_called()

    def test_creates_cache_directory(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
    ) -> None:
        """Cache directory is created if it does not exist."""
        cache = tmp_path / "deep" / "nested" / "cache"
        download_record("101", cache)

        assert cache.exists()
        assert (cache / "101.hea").exists()


# -----------------------------------------------------------------------
# download_record — checksum verification
# -----------------------------------------------------------------------


class TestDownloadRecordChecksum:
    """Verify SHA-256 integrity checking on cache hit and post-download."""

    def test_cache_hit_valid_checksum_skips_download(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cached file with matching checksum is accepted without download."""
        # Pre-populate cache
        dat_content = b"\x00" * 256
        (tmp_path / "100.hea").write_text("header")
        (tmp_path / "100.dat").write_bytes(dat_content)
        expected_hash = hashlib.sha256(dat_content).hexdigest()

        # Patch manifest to include checksum
        manifest: dict[str, Any] = {
            "records": {
                "100": {"dat_checksum": expected_hash},
            }
        }
        monkeypatch.setattr(
            "neuroflux_datasets.mit_bih.downloader._load_manifest",
            lambda: manifest,
        )

        download_record("100", tmp_path)
        mock_wfdb.assert_not_called()

    def test_cache_hit_checksum_mismatch_redownloads(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cached file with wrong checksum triggers re-download."""
        # Pre-populate cache with "wrong" content
        (tmp_path / "100.hea").write_text("header")
        (tmp_path / "100.dat").write_bytes(b"wrong-content")

        manifest: dict[str, Any] = {
            "records": {
                "100": {
                    "dat_checksum": hashlib.sha256(b"\x00" * 256).hexdigest(),
                },
            }
        }
        monkeypatch.setattr(
            "neuroflux_datasets.mit_bih.downloader._load_manifest",
            lambda: manifest,
        )

        download_record("100", tmp_path)
        # Should have re-downloaded
        mock_wfdb.assert_called_once()

    def test_post_download_checksum_mismatch_raises(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fresh download whose checksum doesn't match raises IntegrityError."""
        # Manifest expects a checksum that won't match the stub payload
        manifest: dict[str, Any] = {
            "records": {
                "100": {"dat_checksum": "badhash"},
            }
        }
        monkeypatch.setattr(
            "neuroflux_datasets.mit_bih.downloader._load_manifest",
            lambda: manifest,
        )

        with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
            download_record("100", tmp_path)

    def test_no_manifest_checksum_trusts_cache(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When manifest has no checksum, cached files are trusted."""
        (tmp_path / "100.hea").write_text("header")
        (tmp_path / "100.dat").write_bytes(b"anything")

        manifest: dict[str, Any] = {"records": {"100": {"dat_checksum": None}}}
        monkeypatch.setattr(
            "neuroflux_datasets.mit_bih.downloader._load_manifest",
            lambda: manifest,
        )

        download_record("100", tmp_path)
        mock_wfdb.assert_not_called()


# -----------------------------------------------------------------------
# download_record — provenance
# -----------------------------------------------------------------------


class TestDownloadRecordProvenance:
    """Verify provenance sidecar is written on fresh download."""

    def test_provenance_file_created(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
    ) -> None:
        """A .provenance.json sidecar is created after download."""
        download_record("100", tmp_path)

        prov_path = tmp_path / "100.provenance.json"
        assert prov_path.exists()

        prov = json.loads(prov_path.read_text())
        assert prov["record_id"] == "100"
        assert "download_timestamp" in prov
        assert prov["database"] == "mitdb"
        assert "physionet.org" in prov["source_url"]

    def test_provenance_includes_checksum(
        self,
        tmp_path: Path,
        mock_wfdb: MagicMock,
    ) -> None:
        """Provenance records the computed dat_checksum."""
        download_record("100", tmp_path)

        prov = json.loads((tmp_path / "100.provenance.json").read_text())
        assert prov["dat_checksum"] is not None
        # Must match the actual file
        actual = compute_checksum(tmp_path / "100.dat")
        assert prov["dat_checksum"] == actual


# -----------------------------------------------------------------------
# verify_records
# -----------------------------------------------------------------------


class TestVerifyRecords:
    """Tests for the header-only cache check."""

    def test_all_missing(self, tmp_path: Path) -> None:
        """All IDs returned when cache dir is empty."""
        missing = verify_records(["100", "101", "102"], tmp_path)
        assert missing == ["100", "101", "102"]

    def test_all_present(self, tmp_path: Path) -> None:
        """Empty list returned when all records are cached."""
        for rid in ["100", "101"]:
            (tmp_path / f"{rid}.hea").write_text("h")
            (tmp_path / f"{rid}.dat").write_bytes(b"d")

        missing = verify_records(["100", "101"], tmp_path)
        assert missing == []

    def test_partial(self, tmp_path: Path) -> None:
        """Only missing IDs are returned."""
        (tmp_path / "100.hea").write_text("h")
        (tmp_path / "100.dat").write_bytes(b"d")

        missing = verify_records(["100", "101"], tmp_path)
        assert missing == ["101"]

    def test_partial_files_count_as_missing(self, tmp_path: Path) -> None:
        """A record with only .hea (no .dat) is still missing."""
        (tmp_path / "100.hea").write_text("h")
        # No .dat file

        missing = verify_records(["100"], tmp_path)
        assert missing == ["100"]

    def test_nonexistent_cache_dir(self, tmp_path: Path) -> None:
        """Non-existent cache directory means everything is missing."""
        missing = verify_records(["100"], tmp_path / "does_not_exist")
        assert missing == ["100"]


# -----------------------------------------------------------------------
# _load_manifest
# -----------------------------------------------------------------------


class TestLoadManifest:
    """Tests for manifest loading."""

    def test_manifest_loads_successfully(self) -> None:
        """The committed manifest.json can be parsed."""
        manifest = _load_manifest()
        assert "records" in manifest
        assert "100" in manifest["records"]
        assert "234" in manifest["records"]

    def test_manifest_has_all_48_records(self) -> None:
        """Manifest contains all 48 standard MIT-BIH records."""
        manifest = _load_manifest()
        assert len(manifest["records"]) == 48
