"""
theory.py

Power-spectrum theory initialization and evaluation.

The current DESI desilike refactor-jax theory class imports
velocileptors.LPT.lpt_rsd_jax, which is absent from the shared
velocileptors installation on Perlmutter.  This module therefore
uses the installed FFTW implementation directly.

The public TheoryManager.evaluate() interface and the project's
standard-basis nuisance conventions are preserved.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from desilike import build
from desilike.theories.primordial_cosmology import CosmoprimoCosmology
from desilike.theories.galaxy_clustering.template import DirectSpectrum2Template

from .cosmology import get_cosmology_params


# =========================================================
# FIDUCIAL THEORY SETTINGS
# =========================================================

Z_FID = 0.8
TRACER_FID = "LRG"
NBAR_FID = 4.56e-4

ELL_FID = [0, 2, 4]

K_MIN = 0.02
K_MAX = 0.20
DELTA_K = 0.005

K_FID = np.arange(K_MIN + DELTA_K / 2, K_MAX, DELTA_K)

# The standard-basis table combiner uses nd=1e-4, matching the
# desilike velocileptors table convention.  It is not the tracer
# number density nbar.
TABLE_ND = 1.0e-4

ALL_TABLE_ELL = (0, 2, 4)


# =========================================================
# PARAMETER DEFINITIONS
# =========================================================

BIAS_PARAMETERS = ("b1", "b2", "bs", "b3")
EFT_PARAMETERS = ("alpha0", "alpha2", "alpha4", "alpha6")
STOCHASTIC_PARAMETERS = ("sn0", "sn2", "sn4")

NUISANCE_PARAMETERS = (
    BIAS_PARAMETERS + EFT_PARAMETERS + STOCHASTIC_PARAMETERS
)



def _velocileptors_kvec(k, boost_prec=2):
    """Internal velocileptors evaluation grid used by desilike."""
    k = np.asarray(k, dtype="f8")
    return np.concatenate([
        [min(0.0005, k[0])],
        np.geomspace(0.0015, 0.025, 10 * boost_prec, endpoint=True),
        np.arange(
            0.03,
            max(0.5, k[-1]) + 0.015 / boost_prec,
            0.01 / boost_prec,
        ),
    ])

def _standard_bias_monomials(bias, eft, stochastic, nd=TABLE_ND):
    """Return the 19 coefficients matching the velocileptors PT-table columns."""
    b1 = float(bias["b1"])
    b2 = float(bias["b2"])
    bs = float(bias["bs"])
    b3 = float(bias["b3"])

    alpha0 = float(eft["alpha0"])
    alpha2 = float(eft["alpha2"])
    alpha4 = float(eft["alpha4"])
    alpha6 = float(eft["alpha6"])

    sn0 = float(stochastic["sn0"])
    sn2 = float(stochastic["sn2"])
    sn4 = float(stochastic["sn4"])

    return np.asarray(
        [
            1.0,
            b1,
            b1**2,
            b2,
            b1 * b2,
            b2**2,
            bs,
            b1 * bs,
            b2 * bs,
            bs**2,
            b3,
            b1 * b3,
            alpha0,
            alpha2,
            alpha4,
            alpha6,
            sn0 / nd,
            sn2 / nd,
            sn4 / nd,
        ],
        dtype=float,
    )


# =========================================================
# THEORY MANAGER
# =========================================================

class TheoryManager:
    """
    Manage cosmology-dependent velocileptors PT tables and evaluate
    spectra for standard-basis nuisance parameters.

    The theory calculation uses velocileptors.LPT.lpt_rsd_fftw
    directly, avoiding the unavailable lpt_rsd_jax import in the
    current shared desilike installation.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of initialized cosmologies retained in memory.
        Uses least-recently-used eviction.
    z : float
        Redshift.
    k : array
        Output wavenumber bin centres.
    ells : list or tuple
        Requested multipoles; any subset of (0, 2, 4) is supported.
    prior_basis : str
        This manager implements the standard nuisance basis.
    nbar : float
        Retained as tracer metadata for compatibility with existing
        project configuration. Standard-basis stochastic coefficients
        use TABLE_ND, as in the desilike PT-table combiner.
    tracer : str
        Retained as tracer metadata for compatibility.
    mu : int
        Number of positive Gauss-Legendre angular nodes used to build
        the multipole table. Matches desilike's LPT default of 4.
    """

    def __init__(
        self,
        max_cache_size=8,
        z=Z_FID,
        k=K_FID,
        ells=ELL_FID,
        prior_basis="standard",
        nbar=NBAR_FID,
        tracer=TRACER_FID,
        mu=4,
    ):
        self.max_cache_size = int(max_cache_size)
        if self.max_cache_size < 1:
            raise ValueError("max_cache_size must be at least 1.")

        self.z = float(z)
        self.k = np.asarray(k, dtype=float).copy()
        self.ells = tuple(int(ell) for ell in ells)

        if not self.ells:
            raise ValueError("At least one multipole must be requested.")
        if any(ell not in ALL_TABLE_ELL for ell in self.ells):
            raise ValueError(
                f"Supported multipoles are {ALL_TABLE_ELL}; got {self.ells}."
            )
        if self.k.ndim != 1 or not np.all(np.isfinite(self.k)):
            raise ValueError("k must be a finite one-dimensional array.")
        if np.any(np.diff(self.k) <= 0):
            raise ValueError("k must be strictly increasing.")

        self.prior_basis = str(prior_basis).lower()
        if self.prior_basis != "standard":
            raise NotImplementedError(
                "This FFTW TheoryManager currently supports prior_basis='standard' "
                "only. The project's nuisance designs and emulator comparisons "
                "use the standard basis."
            )

        self.nbar = float(nbar)
        self.tracer = tracer
        self.mu = int(mu)
        if self.mu < 1:
            raise ValueError("mu must be a positive integer.")

        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    # -----------------------------------------------------
    # CACHE KEY
    # -----------------------------------------------------

    def _make_cache_key(self, cosmo_params):
        cosmology_key = tuple(
            sorted((key, float(value)) for key, value in cosmo_params.items())
        )
        theory_key = (
            self.z,
            tuple(self.k),
            self.ells,
            self.prior_basis,
            self.nbar,
            self.tracer,
            self.mu,
        )
        return cosmology_key, theory_key

    # -----------------------------------------------------
    # INITIALIZE FFTW THEORY TABLE
    # -----------------------------------------------------

    def _initialize_theory(self, cosmo_params):
        # Convert the eight independent parameters into the complete
        # CLASS-compatible cosmology used by the existing project.
        params = get_cosmology_params(cosmo_params)
        cosmo = CosmoprimoCosmology(fiducial=params)

        # Match the k-grid and PT options of the current desilike
        # LPTVelocileptorsPTSpectrum2Poles implementation.
        k_table = _velocileptors_kvec(self.k)
        kin = np.geomspace(
            min(1.0e-4, k_table[0] / 2.0),
            max(2.0, k_table[-1] * 2.0),
            500,
        )

        template = DirectSpectrum2Template(z=self.z, cosmo=cosmo)
        template.update(k=kin)

        # Build only the cosmology/template dependency. This evaluates
        # the CLASS-backed linear spectrum and growth/AP quantities,
        # without instantiating the missing JAX velocileptors calculator.
        template_pipe = build(template)
        template_pipe()

        from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

        pt_options = {
            "use_Pzel": False,
            "kIR": 0.2,
            "cutoff": 10,
            "extrap_min": -4,
            "extrap_max": 2,
            "N": 800,
            "jn": 5,
        }

        linear_k = np.asarray(template.k, dtype=float)
        linear_pk = np.asarray(template.pk_dd, dtype=float)
        if linear_pk.ndim != 1:
            linear_pk = np.squeeze(linear_pk)
        if linear_k.ndim != 1 or linear_pk.shape != linear_k.shape:
            raise ValueError(
                "Unexpected linear-spectrum shapes from DirectSpectrum2Template: "
                f"k={linear_k.shape}, pk_dd={linear_pk.shape}."
            )

        pt = LPT_RSD(linear_k, linear_pk, **pt_options)

        # Growth rate f = fsigma8 / sigma8 from the initialized template.
        f = float(template.fsigma8 / template.sigma8)
        qpar = float(np.asarray(template.qpar))
        qper = float(np.asarray(template.qper))

        # FFTW make_pltable is the reference implementation for the
        # JAX bias_table. Its default nmax=5 and ngauss=4 match desilike.
        pt.make_pltable(
            f,
            kv=k_table,
            apar=qpar,
            aperp=qper,
            ngauss=self.mu,
        )

        # Column order: 12 bias monomials, 4 EFT terms, 3 stochastic terms.
        pktable = np.stack(
            [pt.p0ktable, pt.p2ktable, pt.p4ktable],
            axis=0,
        ).astype(float, copy=False)

        expected_table_shape = (3, len(k_table), 19)
        if pktable.shape != expected_table_shape:
            raise ValueError(
                "Unexpected velocileptors PT-table shape: "
                f"{pktable.shape}; expected {expected_table_shape}."
            )

        state = SimpleNamespace(
            template=template,
            pt=pt,
            k_table=k_table,
            pktable=pktable,
            table_ells=ALL_TABLE_ELL,
        )
        return state, template_pipe

    # -----------------------------------------------------
    # GET THEORY
    # -----------------------------------------------------

    def get_theory(self, cosmo_params):
        """Retrieve an initialized cosmology/PT table from the LRU cache."""
        key = self._make_cache_key(cosmo_params)

        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.cache_misses += 1
        theory, pipe = self._initialize_theory(cosmo_params)
        self._cache[key] = (theory, pipe)

        if len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        return theory, pipe

    # -----------------------------------------------------
    # EVALUATE POWER SPECTRUM
    # -----------------------------------------------------

    def evaluate(self, cosmo_params, bias, eft, stochastic):
        """
        Evaluate P_ell(k) for one cosmology and nuisance combination.

        Returns
        -------
        numpy.ndarray
            Power spectrum with shape (n_ells, n_k), in the requested
            order of self.ells.
        """
        parameter_sets = {
            "bias": (bias, BIAS_PARAMETERS),
            "eft": (eft, EFT_PARAMETERS),
            "stochastic": (stochastic, STOCHASTIC_PARAMETERS),
        }

        for label, (values, required) in parameter_sets.items():
            if not isinstance(values, dict):
                raise TypeError(f"{label} must be a dictionary.")
            missing = set(required) - set(values)
            if missing:
                raise ValueError(
                    f"Missing {label} parameters: {sorted(missing)}"
                )

        theory, _ = self.get_theory(cosmo_params)

        monomials = _standard_bias_monomials(
            bias=bias,
            eft=eft,
            stochastic=stochastic,
        )

        # Contract the 19 PT-table columns with the standard-basis
        # nuisance monomials. The stochastic coefficients are normalized
        # by TABLE_ND, matching desilike's tablevel combiner.
        poles_table = np.einsum(
            "lki,i->lk",
            theory.pktable,
            monomials,
            optimize=True,
        )

        # Interpolate from the native PT grid to the requested output bins.
        interpolate_poles = scipy_interp1d(
            theory.k_table,
            poles_table,
            axis=1,
            kind="cubic",
            bounds_error=True,
            assume_sorted=True,
        )
        all_poles = np.asarray(interpolate_poles(self.k), dtype=float)

        ell_indices = [
            theory.table_ells.index(ell) for ell in self.ells
        ]
        result = all_poles[ell_indices]

        expected_shape = (len(self.ells), len(self.k))
        if result.shape != expected_shape:
            raise ValueError(
                f"Unexpected power-spectrum shape: {result.shape}; "
                f"expected {expected_shape}."
            )
        if not np.isfinite(result).all():
            raise ValueError("Theory evaluation produced non-finite values.")

        return result

    # -----------------------------------------------------
    # CACHE MANAGEMENT
    # -----------------------------------------------------

    def clear_cache(self):
        """Remove all cached cosmology/PT-table objects."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self):
        """Return cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
