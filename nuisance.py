
"""
Nuisance parameter sampling for the power spectrum emulator.

Parameters are sampled in the standard basis used by the
current LPTVelocileptors theory implementation.

b3 is fixed to zero.
"""

import numpy as np
import pandas as pd

from scipy.stats import qmc


# ---------------------------------------------------------
# SAMPLING CONFIGURATION
# ---------------------------------------------------------

# Parameters sampled during nuisance-parameter design generation.
# These are emulator training ranges, not necessarily inference priors.
NUISANCE_BOUNDS = {
    # Bias
    "b1": (1.0, 3.0),
    "b2": (-5.0, 5.0),
    "bs": (-5.0, 5.0),

    # EFT counterterms
    "alpha0": (-50.0, 50.0),
    "alpha2": (-50.0, 50.0),
    "alpha4": (-50.0, 50.0),

    # Stochastic terms
    "sn0": (-1.5, 1.5),
    "sn2": (-20.0, 20.0),
    "sn4": (-650.0, 650.0),
}

# Parameters held fixed during the initial emulator experiment.
# They are passed to the theory calculation but are not sampled.
FIXED_NUISANCE = {
    "b3": 0.0,
    "alpha6": 0.0,
}

NUISANCE_PARAMETERS = list(NUISANCE_BOUNDS.keys())

ALL_NUISANCE_PARAMETERS = (
    list(NUISANCE_BOUNDS.keys())
    + list(FIXED_NUISANCE.keys())
)


# ---------------------------------------------------------
# LATIN HYPERCUBE SAMPLING
# ---------------------------------------------------------

def sample_nuisance_lhs(
    n_samples,
    seed=42,
):
    """
    Generate nuisance parameter samples using Latin
    hypercube sampling.

    Parameters
    ----------
    n_samples : int
        Number of nuisance realizations.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per nuisance realization.
        Includes fixed parameters such as b3.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    names = list(NUISANCE_BOUNDS.keys())

    lower = np.array([
        NUISANCE_BOUNDS[name][0]
        for name in names
    ])

    upper = np.array([
        NUISANCE_BOUNDS[name][1]
        for name in names
    ])

    sampler = qmc.LatinHypercube(
        d=len(names),
        seed=seed,
    )

    unit_samples = sampler.random(
        n=n_samples
    )

    physical_samples = qmc.scale(
        unit_samples,
        lower,
        upper,
    )

    samples = pd.DataFrame(
        physical_samples,
        columns=names,
    )

    # Add fixed nuisance parameters
    for name, value in FIXED_NUISANCE.items():
        samples[name] = value

    samples.insert(
        0,
        "sample_id",
        np.arange(n_samples),
    )

    return samples


# ---------------------------------------------------------
# PARAMETER DICTIONARY
# ---------------------------------------------------------

def get_nuisance_dict(sample):
    """
    Convert a nuisance-parameter sample into the dictionaries
    expected by TheoryManager.evaluate().

    Sampled parameters must be present in `sample`.
    Fixed parameters are automatically inserted from FIXED_NUISANCE.
    """

    # Start with the sampled values.
    values = {name: float(sample[name]) for name in NUISANCE_BOUNDS}

    # Add fixed parameters.
    values.update(FIXED_NUISANCE)

    return {
        "bias": {
            "b1": values["b1"],
            "b2": values["b2"],
            "bs": values["bs"],
            "b3": values["b3"],
        },
        "eft": {
            "alpha0": values["alpha0"],
            "alpha2": values["alpha2"],
            "alpha4": values["alpha4"],
            "alpha6": values["alpha6"],
        },
        "stochastic": {
            "sn0": values["sn0"],
            "sn2": values["sn2"],
            "sn4": values["sn4"],
        },
    }