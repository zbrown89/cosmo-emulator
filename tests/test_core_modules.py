"""Fast unit tests for project-owned modules.

These tests use synthetic inputs only. They do not need Abacus files,
CLASS, Velocileptors, or a NERSC filesystem.

cosmology.py imports cosmoprimo at module import time even though its
catalogue utilities do not need it. A tiny import stub keeps these
pure-data tests independent of the external cosmology stack. The
separate API tests exercise the real cosmoprimo/desilike installation.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cosmology_module(monkeypatch):
    """Import cosmology.py without initializing its optional CLASS backend."""
    stub = types.ModuleType("cosmoprimo")
    stub.Cosmology = object
    monkeypatch.setitem(sys.modules, "cosmoprimo", stub)
    sys.modules.pop("cosmology", None)
    module = importlib.import_module("cosmology")
    yield module
    sys.modules.pop("cosmology", None)


def _catalogue():
    return pd.DataFrame(
        {
            "root": ["abacus_cosm002", "abacus_cosm000", "abacus_cosm001"],
            "notes": ["two", "zero", "one"],
            "omega_b": [0.0224, 0.02237, 0.02242],
            "omega_cdm": [0.12, 0.12, 0.1134],
            "h": [0.6278, 0.6736, 0.7030],
            "A_s": [2.1e-9, 2.083e-9, 2.15e-9],
            "n_s": [0.9649, 0.9649, 0.9638],
            "alpha_s": [0.0, 0.0, 0.0],
            "N_ur": [2.0328, 2.0328, 2.0328],
            "w0_fld": [-0.7, -1.0, -1.0],
            "wa_fld": [-0.5, 0.0, 0.0],
            "sigma8_cb": [0.8116, 0.8114, 0.7802],
            "sigma8_m": [0.8080, 0.8080, 0.7770],
        }
    )


def test_read_catalogue_sorts_and_filters_ids(cosmology_module, tmp_path):
    path = tmp_path / "cosmologies.csv"
    _catalogue().to_csv(path, index=False)

    result = cosmology_module.read_abacus_cosmologies(
        path, cosmology_ids=[2, 0]
    )

    assert result["cosmo_id"].tolist() == [0, 2]
    assert result["root"].tolist() == [
        "abacus_cosm000", "abacus_cosm002"
    ]
    assert np.isfinite(
        result[cosmology_module.COSMOLOGY_PARAMETERS].to_numpy()
    ).all()


def test_read_catalogue_rejects_missing_requested_id(
    cosmology_module, tmp_path
):
    path = tmp_path / "cosmologies.csv"
    _catalogue().to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing cosmology IDs"):
        cosmology_module.read_abacus_cosmologies(
            path, cosmology_ids=[999]
        )


def test_read_catalogue_rejects_duplicate_ids(
    cosmology_module, tmp_path
):
    data = _catalogue()
    data.loc[2, "root"] = "abacus_cosm000"
    path = tmp_path / "cosmologies.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate cosmology IDs"):
        cosmology_module.read_abacus_cosmologies(path)


def test_read_catalogue_rejects_nonfinite_values(
    cosmology_module, tmp_path
):
    data = _catalogue()
    data.loc[0, "omega_b"] = np.nan
    path = tmp_path / "cosmologies.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Non-finite"):
        cosmology_module.read_abacus_cosmologies(path)


def test_cosmology_dictionary_and_metadata(cosmology_module):
    row = _catalogue().iloc[1].copy()
    row["cosmo_id"] = 0

    params = cosmology_module.get_cosmology_dict(row)
    metadata = cosmology_module.get_cosmology_metadata(row)

    assert set(params) == set(cosmology_module.COSMOLOGY_PARAMETERS)
    assert all(isinstance(value, float) for value in params.values())
    assert metadata["cosmo_id"] == 0
    assert metadata["root"] == "abacus_cosm000"
    assert metadata["h_reference"] == pytest.approx(0.6736)
    assert metadata["A_s_reference"] == pytest.approx(2.083e-9)


def test_cosmology_summary_is_available_at_module_level(cosmology_module):
    # This public helper should be importable independently of metadata extraction.
    assert callable(cosmology_module.print_cosmology_summary)


def test_latin_hypercube_is_reproducible_and_respects_bounds():
    from sampling import latin_hypercube

    bounds = {"x": (-1.0, 1.0), "y": (10.0, 20.0)}
    first = latin_hypercube(bounds, n_samples=12, seed=123)
    second = latin_hypercube(bounds, n_samples=12, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["x", "y"]
    assert first.shape == (12, 2)
    assert first["x"].between(-1.0, 1.0).all()
    assert first["y"].between(10.0, 20.0).all()


@pytest.mark.parametrize(
    "bounds,n_samples",
    [
        ({}, 5),
        ({"x": (0.0, 1.0)}, 0),
        ({"x": (1.0, 1.0)}, 5),
        ({"x": (2.0, 1.0)}, 5),
        ({"x": (0.0, np.inf)}, 5),
    ],
)
def test_latin_hypercube_rejects_invalid_inputs(bounds, n_samples):
    from sampling import latin_hypercube

    with pytest.raises(ValueError):
        latin_hypercube(bounds, n_samples=n_samples, seed=1)


def test_pipeline_a_reuses_same_nuisance_design_for_each_cosmology():
    from sampling import make_pipeline_a_design

    cosmologies = pd.DataFrame(
        {
            "cosmo_id": [130, 131],
            "omega_b": [0.0223, 0.0224],
        }
    )
    bounds = {"b1": (1.0, 3.0), "alpha0": (-5.0, 5.0)}

    design = make_pipeline_a_design(
        cosmologies,
        n_nuisance=5,
        nuisance_bounds=bounds,
        seed=42,
    )

    assert len(design) == 10
    assert design["cosmo_id"].value_counts().to_dict() == {
        130: 5, 131: 5
    }
    nuisance_columns = list(bounds)
    first = design.loc[design.cosmo_id == 130, nuisance_columns].reset_index(drop=True)
    second = design.loc[design.cosmo_id == 131, nuisance_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_pipeline_a_requires_cosmology_id():
    from sampling import make_pipeline_a_design

    with pytest.raises(ValueError, match="cosmo_id"):
        make_pipeline_a_design(
            pd.DataFrame({"omega_b": [0.0224]}),
            n_nuisance=2,
            nuisance_bounds={"b1": (1.0, 3.0)},
            seed=1,
        )


def test_pipeline_b_combines_parameter_spaces_and_is_reproducible():
    from sampling import make_pipeline_b_design

    cosmology_bounds = {"omega_b": (0.021, 0.024)}
    nuisance_bounds = {"b1": (1.0, 3.0)}

    first = make_pipeline_b_design(
        8, cosmology_bounds, nuisance_bounds, seed=17
    )
    second = make_pipeline_b_design(
        8, cosmology_bounds, nuisance_bounds, seed=17
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (8, 2)
    assert set(first.columns) == {"omega_b", "b1"}
    assert first["omega_b"].between(0.021, 0.024).all()
    assert first["b1"].between(1.0, 3.0).all()


def test_pipeline_b_rejects_overlapping_parameter_names():
    from sampling import make_pipeline_b_design

    with pytest.raises(ValueError, match="Overlapping"):
        make_pipeline_b_design(
            5,
            {"b1": (0.0, 1.0)},
            {"b1": (1.0, 2.0)},
            seed=1,
        )


def test_nuisance_lhs_is_reproducible_and_respects_bounds():
    from nuisance import (
        FIXED_NUISANCE,
        NUISANCE_BOUNDS,
        NUISANCE_PARAMETERS,
        sample_nuisance_lhs,
    )

    first = sample_nuisance_lhs(16, seed=23)
    second = sample_nuisance_lhs(16, seed=23)

    pd.testing.assert_frame_equal(first, second)
    assert first["sample_id"].tolist() == list(range(16))
    assert set(NUISANCE_PARAMETERS).issubset(first.columns)

    for name, (lower, upper) in NUISANCE_BOUNDS.items():
        assert first[name].between(lower, upper).all()

    for name, value in FIXED_NUISANCE.items():
        assert (first[name] == value).all()


def test_nuisance_dictionary_has_expected_nested_structure():
    from nuisance import (
        ALL_NUISANCE_PARAMETERS,
        sample_nuisance_lhs,
        get_nuisance_dict,
    )

    sample = sample_nuisance_lhs(1, seed=5).iloc[0]
    result = get_nuisance_dict(sample)

    assert set(result) == {"bias", "eft", "stochastic"}
    assert set(result["bias"]) == {"b1", "b2", "bs", "b3"}
    assert set(result["eft"]) == {"alpha0", "alpha2", "alpha4", "alpha6"}
    assert set(result["stochastic"]) == {"sn0", "sn2", "sn4"}

    flattened = (
        set(result["bias"])
        | set(result["eft"])
        | set(result["stochastic"])
    )
    assert flattened == set(ALL_NUISANCE_PARAMETERS)
