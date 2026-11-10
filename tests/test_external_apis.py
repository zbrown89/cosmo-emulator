"""Smoke tests for external APIs used by the emulator.

These tests intentionally check the public interfaces imported by
theory.py. They do not run CLASS, generate spectra, or require
Abacus data. A failure can indicate upstream API drift and must be
compared with the package versions installed on NERSC.
"""


def test_cosmoprimo_and_desilike_core_imports():
    from cosmoprimo import Cosmology
    from desilike import build
    from desilike.base import get_params
    from desilike.theories.primordial_cosmology import CosmoprimoCosmology
    from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

    assert callable(Cosmology)
    assert callable(build)
    assert callable(get_params)
    assert callable(CosmoprimoCosmology)
    assert callable(DirectSpectrum2Template)


def test_project_velocileptors_class_is_available():
    """Check the exact theory class name currently imported by theory.py."""
    from desilike.theories.galaxy_clustering.full_shape import (
        LPTVelocileptorsTracerSpectrum2Poles,
    )

    assert callable(LPTVelocileptorsTracerSpectrum2Poles)


def test_project_stochastic_settings_contract():
    """Check the settings keys consumed by TheoryManager initialization."""
    from desilike.theories.galaxy_clustering.full_shape import (
        get_physical_stochastic_settings,
    )

    settings = get_physical_stochastic_settings("LRG")

    assert {"fsat", "sigv", "rvir"} <= set(settings)
