import re
import numpy as np
import pandas as pd
from pathlib import Path

from cosmoprimo import Cosmology
from scipy.optimize import brentq

# Independent cosmological parameters
COSMOLOGY_PARAMETERS = [
    'omega_b',
    'omega_cdm',
    'sigma8_cb',
    'n_s',
    'alpha_s',
    'N_ur',
    'w0_fld',
    'wa_fld',
]

# Parameters required by the Boltzmann solver
DERIVED_PARAMETERS = [
    'h',
    'A_s',
]

def read_abacus_cosmologies(
    cosmology_file,
    cosmology_ids=None,
):
    """
    Read and optionally filter the Abacus cosmology catalogue.

    Parameters
    ----------
    cosmology_file : str or Path
        Path to cosmologies.csv.

    cosmology_ids : list of int, optional
        Explicit cosmology IDs to select.
        If None, return the entire catalogue.

    Returns
    -------
    pandas.DataFrame
        Cleaned cosmology catalogue.
    """

    cosmologies = pd.read_csv(cosmology_file)

    # Clean column names and string fields
    cosmologies.columns = cosmologies.columns.str.strip()

    cosmologies['root'] = (
        cosmologies['root'].astype(str).str.strip()
    )

    # Extract cosmology IDs
    cosmologies['cosmo_id'] = pd.to_numeric(
        cosmologies['root'].str.extract(
            r'cosm(\d+)', expand=False
        ),
        errors='coerce'
    )

    cosmologies = cosmologies.dropna(
        subset=['cosmo_id']
    ).copy()

    cosmologies['cosmo_id'] = (
        cosmologies['cosmo_id'].astype(int)
    )

    # Check required columns
    required_columns = (
        COSMOLOGY_PARAMETERS
        + DERIVED_PARAMETERS
    )

    missing_columns = [
        col for col in required_columns
        if col not in cosmologies.columns
    ]

    if missing_columns:
        raise ValueError(
            f'Missing cosmology columns: {missing_columns}'
        )

    # Convert numerical columns explicitly
    for column in required_columns:
        cosmologies[column] = pd.to_numeric(
            cosmologies[column],
            errors='raise'
        )

    # Check for duplicate IDs
    if cosmologies['cosmo_id'].duplicated().any():
        raise ValueError('Duplicate cosmology IDs found.')

    # Check numerical validity
    if not np.isfinite(
        cosmologies[required_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            'Non-finite cosmological parameters found.'
        )

    # Optional cosmology selection
    if cosmology_ids is not None:

        cosmology_ids = list(cosmology_ids)

        found_ids = set(cosmologies['cosmo_id'])
        missing_ids = set(cosmology_ids) - found_ids

        if missing_ids:
            raise ValueError(
                f'Missing cosmology IDs: {sorted(missing_ids)}'
            )

        cosmologies = cosmologies[
            cosmologies['cosmo_id'].isin(cosmology_ids)
        ].copy()

    cosmologies = cosmologies.sort_values(
        'cosmo_id'
    ).reset_index(drop=True)

    print(f'Loaded {len(cosmologies)} cosmologies.')

    return cosmologies


def get_cosmology_dict(cosmology_row):
    """
    Extract the eight independent cosmological parameters
    and associated derived/reference quantities.

    Parameters
    ----------
    cosmology_row : pandas.Series or dict

    Returns
    -------
    dict
    """

    return {
        name: float(cosmology_row[name])
        for name in COSMOLOGY_PARAMETERS
    }


def get_cosmology_metadata(cosmology_row):
    """
    Extract additional cosmology information useful
    for validation and bookkeeping.
    """

    return {
        'cosmo_id': int(cosmology_row['cosmo_id']),
        'root': str(cosmology_row['root']),
        'h_reference': float(cosmology_row['h']),
        'A_s_reference': float(cosmology_row['A_s']),
        'sigma8_m': float(cosmology_row['sigma8_m']),
        'N_ncdm': 1,
        'omega_ncdm': 0.00064420,
    }

    
def print_cosmology_summary(cosmologies):
    """
    Print a compact summary of the selected cosmologies.
    """

    columns = [
        'root',
        'omega_b',
        'omega_cdm',
        'sigma8_cb',
        'n_s',
        'alpha_s',
        'N_ur',
        'w0_fld',
        'wa_fld',
        'h',
    ]

    print(cosmologies[columns].to_string(index=False))

# ---------------------------------------------------------
# COSMOLOGY CONFIGURATION
# ---------------------------------------------------------

THETA_S_100 = 1.042143

TAU_REIO = 0.0544
T_CMB = 2.7255

NEUTRINO_PARAMS = {
    'N_ncdm': 1,
    'omega_ncdm': 0.00064420,
    'T_ncdm_over_cmb': 0.71611,
}


# ---------------------------------------------------------
# BUILD CLASS COSMOLOGY
# ---------------------------------------------------------

def _build_class_cosmology(cosmo_params, h, A_s):

    params = {
        'omega_b': float(cosmo_params['omega_b']),
        'omega_cdm': float(cosmo_params['omega_cdm']),
        'h': float(h),
        'A_s': float(A_s),

        'n_s': float(cosmo_params['n_s']),
        'alpha_s': float(cosmo_params['alpha_s']),
        'N_ur': float(cosmo_params['N_ur']),

        'w0_fld': float(cosmo_params['w0_fld']),
        'wa_fld': float(cosmo_params['wa_fld']),

        'tau_reio': TAU_REIO,
        'T_cmb': T_CMB,

        'recombination': 'HyRec',

        **NEUTRINO_PARAMS,
    }

    return Cosmology(engine='CLASS', **params)


def get_h_from_theta_star(
    cosmo_params,
    theta_s_100=THETA_S_100,
    A_s_reference=2.083e-9,
    h_limits=(0.4, 1.0),
):
    """
    Solve for h using the fixed acoustic scale.

    A_s is temporarily fixed because theta_star is
    effectively independent of the primordial amplitude.
    """

    def theta_residual(h):
        trial_cosmo = _build_class_cosmology(
            cosmo_params,
            h=float(h),
            A_s=A_s_reference,
        )
    
        return (
            float(trial_cosmo.get_thermodynamics().theta_star)
            - theta_s_100 / 100.0
        )
    
    
    h = brentq(
        theta_residual,
        *h_limits,
        xtol=1e-10,
        rtol=1e-10,
    )
    
    return float(h)


def _evaluate_sigma8_cb(cosmo_params, h, A_s):

    cosmo = _build_class_cosmology(
        cosmo_params,
        h=h,
        A_s=A_s,
    )

    return float(cosmo.get_fourier().sigma8_cb)


def get_As_from_sigma8(
    cosmo_params,
    h,
    A_s_bounds=(0.1e-9, 1.0e-8),
    rtol=1e-8,
):
    from scipy.optimize import brentq

    sigma8_target = cosmo_params["sigma8_cb"]

    def residual(A_s):
        return _evaluate_sigma8_cb(cosmo_params, h, A_s) - sigma8_target

    f_low = residual(A_s_bounds[0])
    f_high = residual(A_s_bounds[1])

    if f_low * f_high > 0:
        raise ValueError(
            "Could not bracket A_s solution for sigma8_cb.\n"
            f"  Target sigma8_cb: {sigma8_target:.8f}\n"
            f"  A_s lower bound: {A_s_bounds[0]:.3e}, residual: {f_low:+.8f}\n"
            f"  A_s upper bound: {A_s_bounds[1]:.3e}, residual: {f_high:+.8f}\n"
            f"  Cosmology: {cosmo_params}"
        )

    return brentq(residual, *A_s_bounds, rtol=rtol)


def get_cosmology_params(cosmo_params):

    # Step 1: Solve h
    h = get_h_from_theta_star(cosmo_params)

    # Step 2: Solve A_s at the recovered h
    A_s = get_As_from_sigma8(
        cosmo_params,
        h=h,
    )

    return {
        **{
            key: float(value)
            for key, value in cosmo_params.items()
        },

        'h': h,
        'A_s': A_s,

        'tau_reio': TAU_REIO,
        'T_cmb': T_CMB,

        **NEUTRINO_PARAMS,
    }

    