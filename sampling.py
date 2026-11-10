
"""
sampling.py

Latin Hypercube sampling utilities for the power-spectrum
emulator project.

Supports:

Pipeline A:
    Fixed Abacus cosmologies, sampled nuisance parameters.

Pipeline B:
    Joint sampling of cosmological and nuisance parameters.

All parameter bounds are supplied explicitly.
"""

import numpy as np
import pandas as pd

from scipy.stats import qmc


# =========================================================
# GENERIC LATIN HYPERCUBE
# =========================================================

def latin_hypercube(
    bounds,
    n_samples,
    seed=None,
):
    """
    Generate a Latin Hypercube design.

    Parameters
    ----------
    bounds : dict
        Dictionary mapping parameter names to
        (lower, upper) bounds.

    n_samples : int
        Number of samples.

    seed : int, optional
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Sampled parameter values.
    """

    if not bounds:
        raise ValueError('No parameter bounds supplied.')

    if n_samples < 1:
        raise ValueError('n_samples must be positive.')

    names = list(bounds.keys())

    limits = np.asarray(
        [bounds[name] for name in names],
        dtype=float,
    )

    if limits.shape != (len(names), 2):
        raise ValueError('Every parameter needs two bounds.')

    if not np.isfinite(limits).all():
        raise ValueError('Parameter bounds must be finite.')

    if np.any(limits[:, 1] <= limits[:, 0]):
        raise ValueError('Upper bounds must exceed lower bounds.')

    sampler = qmc.LatinHypercube(
        d=len(names),
        seed=seed,
    )

    unit_samples = sampler.random(
        n=n_samples
    )

    samples = qmc.scale(
        unit_samples,
        limits[:, 0],
        limits[:, 1],
    )

    return pd.DataFrame(
        samples,
        columns=names,
    )


# =========================================================
# PARAMETER-SPECIFIC SAMPLING
# =========================================================

def sample_nuisance_parameters(
    n_samples,
    nuisance_bounds,
    seed=None,
):
    """Generate an LHS design for nuisance parameters."""

    return latin_hypercube(
        bounds=nuisance_bounds,
        n_samples=n_samples,
        seed=seed,
    )


def sample_cosmological_parameters(
    n_samples,
    cosmology_bounds,
    seed=None,
):
    """Generate an LHS design for cosmological parameters."""

    return latin_hypercube(
        bounds=cosmology_bounds,
        n_samples=n_samples,
        seed=seed,
    )


# =========================================================
# PIPELINE A
# =========================================================

def make_pipeline_a_design(
    cosmologies,
    n_nuisance,
    nuisance_bounds,
    seed=None,
):
    """
    Generate a training design anchored to Abacus
    cosmologies.

    Each cosmology receives the same nuisance design.

    Reusing the same nuisance design makes comparisons
    between cosmologies easier and reduces sampling noise
    when examining cosmology-dependent behaviour.

    Parameters
    ----------
    cosmologies : pandas.DataFrame
        Selected Abacus cosmologies.

    n_nuisance : int
        Number of nuisance configurations per cosmology.

    nuisance_bounds : dict
        Bounds for nuisance parameters.

    seed : int, optional
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Combined cosmology-nuisance design.
    """

    if 'cosmo_id' not in cosmologies.columns:
        raise ValueError(
            'cosmologies must contain cosmo_id.'
        )

    nuisance_samples = sample_nuisance_parameters(
        n_samples=n_nuisance,
        nuisance_bounds=nuisance_bounds,
        seed=seed,
    )

    designs = []

    for _, cosmology in cosmologies.iterrows():

        cosmo_block = pd.DataFrame(
            [
                cosmology.to_dict()
            ] * n_nuisance
        ).reset_index(drop=True)

        nuisance_block = nuisance_samples.copy()

        designs.append(
            pd.concat(
                [cosmo_block, nuisance_block],
                axis=1,
            )
        )

    return pd.concat(
        designs,
        ignore_index=True,
    )


# =========================================================
# PIPELINE B
# =========================================================

def make_pipeline_b_design(
    n_samples,
    cosmology_bounds,
    nuisance_bounds,
    seed=None,
):
    """
    Generate a joint cosmology-nuisance LHS design.

    Cosmological and nuisance parameters are sampled
    jointly in one Latin Hypercube.

    Parameters
    ----------
    n_samples : int
        Total number of training samples.

    cosmology_bounds : dict
        Bounds for independent cosmological parameters.

    nuisance_bounds : dict
        Bounds for nuisance parameters.

    seed : int, optional
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Joint parameter design.
    """

    overlap = (
        set(cosmology_bounds)
        & set(nuisance_bounds)
    )

    if overlap:
        raise ValueError(
            f'Overlapping parameter names: {overlap}'
        )

    combined_bounds = {
        **cosmology_bounds,
        **nuisance_bounds,
    }

    return latin_hypercube(
        bounds=combined_bounds,
        n_samples=n_samples,
        seed=seed,
    )
