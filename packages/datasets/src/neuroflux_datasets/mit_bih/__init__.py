"""MIT-BIH Arrhythmia Database integration for NeuroFlux.

Provides downloading, integrity verification, preprocessing, and
ground-truth annotation pipelines for the PhysioNet MIT-BIH dataset.
"""

from neuroflux_datasets.mit_bih.annotator import (
    extract_clean_segments,
    make_pair,
)
from neuroflux_datasets.mit_bih.downloader import (
    IntegrityError,
    compute_checksum,
    download_record,
    verify_records,
)
from neuroflux_datasets.mit_bih.loader import (
    train_test_split,
)
from neuroflux_datasets.mit_bih.preprocessor import (
    normalize,
    resample_signal,
)

__all__ = [
    "IntegrityError",
    "compute_checksum",
    "download_record",
    "extract_clean_segments",
    "make_pair",
    "normalize",
    "resample_signal",
    "train_test_split",
    "verify_records",
]
