"""Smoke tests: verify all packages are importable after installation."""

import importlib


def test_neuroflux_core_importable() -> None:
    importlib.import_module("neuroflux_core")


def test_neuroflux_datasets_importable() -> None:
    importlib.import_module("neuroflux_datasets")


def test_neuroflux_baselines_importable() -> None:
    importlib.import_module("neuroflux_baselines")
