
"""
theory.py

Power-spectrum theory initialization and evaluation.

Uses desilike's LPTVelocileptorsTracerSpectrum2Poles
with cosmoprimo cosmologies.

Initialized theory objects are cached in memory so that
multiple nuisance-parameter evaluations at the same
cosmology do not require repeated initialization.
"""

from collections import OrderedDict
import numpy as np

from desilike import build
from desilike.base import get_params

from desilike.theories.primordial_cosmology import (
    CosmoprimoCosmology,
)

from desilike.theories.galaxy_clustering.template import (
    DirectSpectrum2Template,
)

from desilike.theories.galaxy_clustering.full_shape import (
    LPTVelocileptorsTracerSpectrum2Poles,
    get_physical_stochastic_settings,
)

from .cosmology import get_cosmology_params


# =========================================================
# FIDUCIAL THEORY SETTINGS
# =========================================================

Z_FID = 0.8

TRACER_FID = 'LRG'

NBAR_FID = 4.56e-4

ELL_FID = [0, 2, 4]

K_MIN = 0.02
K_MAX = 0.20
DELTA_K = 0.005

K_FID = np.arange(
    K_MIN + DELTA_K / 2,
    K_MAX,
    DELTA_K,
)


# =========================================================
# PARAMETER DEFINITIONS
# =========================================================

BIAS_PARAMETERS = (
    'b1',
    'b2',
    'bs',
    'b3',
)

EFT_PARAMETERS = (
    'alpha0',
    'alpha2',
    'alpha4',
    'alpha6',
)

STOCHASTIC_PARAMETERS = (
    'sn0',
    'sn2',
    'sn4',
)

NUISANCE_PARAMETERS = (
    BIAS_PARAMETERS
    + EFT_PARAMETERS
    + STOCHASTIC_PARAMETERS
)


# =========================================================
# THEORY MANAGER
# =========================================================

class TheoryManager:
    """
    Manage initialized PT theory objects and their pipelines.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of initialized theories retained
        in memory. Uses least-recently-used eviction.

    z : float
        Redshift.

    k : array
        Wavenumber bin centres.

    ells : list
        Multipoles to calculate.

    prior_basis : str
        EFT prior basis passed to desilike.

    nbar : float
        Tracer number density.

    tracer : str
        Tracer type used for stochastic settings.
    """

    def __init__(
        self,
        max_cache_size=8,
        z=Z_FID,
        k=K_FID,
        ells=ELL_FID,
        prior_basis='standard',
        nbar=NBAR_FID,
        tracer=TRACER_FID,
    ):

        self.max_cache_size = int(max_cache_size)

        if self.max_cache_size < 1:
            raise ValueError(
                'max_cache_size must be at least 1.'
            )

        self.z = float(z)
        self.k = np.asarray(k, dtype=float).copy()
        self.ells = tuple(ells)

        self.prior_basis = prior_basis
        self.nbar = float(nbar)
        self.tracer = tracer

        self._cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0

    # -----------------------------------------------------
    # CACHE KEY
    # -----------------------------------------------------

    def _make_cache_key(self, cosmo_params):

        cosmology_key = tuple(
            sorted(
                (key, float(value))
                for key, value in cosmo_params.items()
            )
        )

        theory_key = (
            self.z,
            tuple(self.k),
            self.ells,
            self.prior_basis,
            self.nbar,
            self.tracer,
        )

        return cosmology_key, theory_key

    # -----------------------------------------------------
    # INITIALIZE THEORY
    # -----------------------------------------------------

    def _initialize_theory(self, cosmo_params):

        # Convert independent parameters into a complete
        # CLASS-compatible cosmology.

        params = get_cosmology_params(cosmo_params)

        cosmo = CosmoprimoCosmology(
            fiducial=params
        )

        template = DirectSpectrum2Template(
            z=self.z,
            cosmo=cosmo,
        )

        # Tracer-specific stochastic settings.

        settings = get_physical_stochastic_settings(
            self.tracer
        )

        theory = LPTVelocileptorsTracerSpectrum2Poles(
            k=self.k,
            ells=list(self.ells),
            template=template,
            prior_basis=self.prior_basis,
            fsat=settings['fsat'],
            sigv=settings['sigv'],
            rvir=settings['rvir'],
            nbar=self.nbar,
        )

        pipe = build(theory)

        return theory, pipe

    # -----------------------------------------------------
    # GET THEORY
    # -----------------------------------------------------

    def get_theory(self, cosmo_params):
        """
        Retrieve an initialized theory from the cache.

        Initializes and caches the theory if necessary.
        """

        key = self._make_cache_key(cosmo_params)

        # Cache hit
        if key in self._cache:

            self.cache_hits += 1

            self._cache.move_to_end(key)

            return self._cache[key]

        # Cache miss
        self.cache_misses += 1

        theory, pipe = self._initialize_theory(
            cosmo_params
        )

        self._cache[key] = (theory, pipe)

        # LRU eviction
        if len(self._cache) > self.max_cache_size:

            self._cache.popitem(last=False)

        return theory, pipe

    # -----------------------------------------------------
    # EVALUATE POWER SPECTRUM
    # -----------------------------------------------------

    def evaluate(
        self,
        cosmo_params,
        bias,
        eft,
        stochastic,
    ):
        """
        Evaluate the power spectrum for a given cosmology
        and nuisance-parameter combination.

        Returns
        -------
        numpy.ndarray
            Power spectrum with shape (n_ells, n_k).
        """

        theory, pipe = self.get_theory(
            cosmo_params
        )

        # Check nuisance parameter dictionaries.

        parameter_sets = {
            'bias': (bias, BIAS_PARAMETERS),
            'eft': (eft, EFT_PARAMETERS),
            'stochastic': (
                stochastic,
                STOCHASTIC_PARAMETERS,
            ),
        }

        for label, (values, required) in parameter_sets.items():

            missing = set(required) - set(values)

            if missing:
                raise ValueError(
                    f'Missing {label} parameters: '
                    f'{sorted(missing)}'
                )

        # Assign bias parameters.

        for name in BIAS_PARAMETERS:
            getattr(theory, name).value = float(bias[name])

        # Assign EFT parameters.

        for name in EFT_PARAMETERS:
            getattr(theory, name).value = float(eft[name])

        # Assign stochastic parameters.

        for name in STOCHASTIC_PARAMETERS:
            getattr(theory, name).value = float(
                stochastic[name]
            )

        # Evaluate pipeline using current parameter values.

        defaults = {
            param.name: param._value
            for param in get_params(theory)
        }

        result = np.asarray(pipe(defaults))

        expected_shape = (
            len(self.ells),
            len(self.k),
        )

        if result.shape != expected_shape:
            raise ValueError(
                f'Unexpected power-spectrum shape: '
                f'{result.shape}; expected {expected_shape}.'
            )

        return result

    # -----------------------------------------------------
    # CACHE MANAGEMENT
    # -----------------------------------------------------

    def clear_cache(self):
        """Remove all cached theory objects."""

        self._cache.clear()

        self.cache_hits = 0
        self.cache_misses = 0

    def cache_info(self):
        """Return cache statistics."""

        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
        }
