
"""
dataset.py

Generate and save PT power-spectrum training datasets.

Spectra are saved in NumPy chunks, with corresponding
parameter metadata stored as CSV files.
"""

import json
import numpy as np
import pandas as pd

from pathlib import Path
from time import perf_counter

from .theory import (
    TheoryManager,
    BIAS_PARAMETERS,
    EFT_PARAMETERS,
    STOCHASTIC_PARAMETERS,
)


# =========================================================
# BUILD ABACUS-ANCHORED TRAINING DESIGN
# =========================================================

def build_anchor_training_design(
    anchors,
    nuisance_samples,
    cosmology_parameters,
):
    """
    Construct the full Cartesian product of Abacus
    cosmology anchors and nuisance parameter samples.

    Every cosmology receives the same nuisance design.

    Parameters
    ----------
    anchors : pandas.DataFrame
        Abacus emulator-grid cosmologies.

    nuisance_samples : pandas.DataFrame
        Flat nuisance parameter design.

    cosmology_parameters : list
        Independent cosmological parameter names.

    Returns
    -------
    pandas.DataFrame
        Complete training design.
    """

    if not isinstance(anchors, pd.DataFrame):
        raise TypeError('anchors must be a DataFrame.')

    if not isinstance(nuisance_samples, pd.DataFrame):
        raise TypeError('nuisance_samples must be a DataFrame.')

    if len(anchors) == 0 or len(nuisance_samples) == 0:
        raise ValueError('Input designs cannot be empty.')

    # Required columns
    required_cosmo = list(cosmology_parameters)

    required_nuisance = (
        list(BIAS_PARAMETERS)
        + list(EFT_PARAMETERS)
        + list(STOCHASTIC_PARAMETERS)
    )

    for column in required_cosmo:
        if column not in anchors.columns:
            raise ValueError(
                f'Missing cosmological parameter: {column}'
            )

    for column in required_nuisance:
        if column not in nuisance_samples.columns:
            raise ValueError(
                f'Missing nuisance parameter: {column}'
            )

    # Keep cosmological metadata where available
    metadata_columns = [
        column for column in ['root', 'cosmo_id']
        if column in anchors.columns
    ]

    anchor_columns = (
        required_cosmo + metadata_columns
    )

    anchors = anchors[anchor_columns].copy()

    nuisance_samples = nuisance_samples[
        required_nuisance
    ].copy()

    # Unique temporary key for Cartesian product
    anchors['_merge_key'] = 1
    nuisance_samples['_merge_key'] = 1

    samples = pd.merge(
        anchors,
        nuisance_samples,
        on='_merge_key',
        how='inner',
        sort=False,
    )

    samples = samples.drop(
        columns='_merge_key'
    )

    # Group by cosmology to maximize theory-cache reuse
    if 'cosmo_id' in samples.columns:
        samples = samples.sort_values(
            ['cosmo_id']
        ).reset_index(drop=True)

    expected_size = (
        len(anchors) * len(nuisance_samples)
    )

    if len(samples) != expected_size:
        raise RuntimeError(
            f'Unexpected design size: {len(samples)} '
            f'(expected {expected_size}).'
        )

    print('Anchor training design')
    print('=' * 55)
    print(f'Cosmologies:       {len(anchors)}')
    print(f'Nuisance samples:  {len(nuisance_samples)}')
    print(f'Total spectra:     {len(samples)}')
    print('=' * 55)

    return samples


# =========================================================
# GENERATE DATASET
# =========================================================

def generate_dataset(
    samples,
    manager,
    output_dir,
    cosmology_parameters,
    chunk_size=100,
    label='training',
):
    """
    Generate a power-spectrum dataset.

    Parameters
    ----------
    samples : pandas.DataFrame
        Input parameter design.

    manager : TheoryManager
        Initialized theory manager.

    output_dir : str or Path
        Output directory.

    cosmology_parameters : list
        Names of independent cosmological parameters.

    chunk_size : int
        Number of spectra saved per chunk.

    label : str
        Dataset label for bookkeeping.

    Returns
    -------
    dict
        Dataset generation summary.
    """

    if not isinstance(samples, pd.DataFrame):
        raise TypeError('samples must be a DataFrame.')

    if len(samples) == 0:
        raise ValueError('No samples supplied.')

    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Required columns
    required_columns = (
        list(cosmology_parameters)
        + list(BIAS_PARAMETERS)
        + list(EFT_PARAMETERS)
        + list(STOCHASTIC_PARAMETERS)
    )

    missing = [
        name for name in required_columns
        if name not in samples.columns
    ]

    if missing:
        raise ValueError(
            f'Missing input columns: {missing}'
        )

    # Ensure unique columns
    if samples.columns.duplicated().any():
        raise ValueError(
            'Duplicate parameter columns found.'
        )

    n_total = len(samples)

    n_ells = len(manager.ells)
    n_k = len(manager.k)

    spectra_buffer = []
    metadata_buffer = []

    chunk_number = 0
    n_completed = 0

    total_start = perf_counter()

    print(f'Generating dataset: {label}')
    print(f'Total spectra: {n_total}')
    print(f'Chunk size: {chunk_size}')
    print('=' * 55)

    # -----------------------------------------------------
    # MAIN GENERATION LOOP
    # -----------------------------------------------------

    for sample_index, (_, row) in enumerate(
        samples.iterrows()
    ):

        cosmo_params = {
            name: float(row[name])
            for name in cosmology_parameters
        }

        bias = {
            name: float(row[name])
            for name in BIAS_PARAMETERS
        }

        eft = {
            name: float(row[name])
            for name in EFT_PARAMETERS
        }

        stochastic = {
            name: float(row[name])
            for name in STOCHASTIC_PARAMETERS
        }

        pk = manager.evaluate(
            cosmo_params=cosmo_params,
            bias=bias,
            eft=eft,
            stochastic=stochastic,
        )

        if pk.shape != (n_ells, n_k):
            raise ValueError(
                f'Unexpected spectrum shape at sample '
                f'{sample_index}: {pk.shape}'
            )

        if not np.isfinite(pk).all():
            raise ValueError(
                f'Non-finite spectrum at sample '
                f'{sample_index}.'
            )

        spectra_buffer.append(pk)

        metadata_buffer.append(
            row.to_dict()
        )

        n_completed += 1

        # -------------------------------------------------
        # SAVE CHUNK
        # -------------------------------------------------

        is_last_sample = (
            sample_index == n_total - 1
        )

        chunk_full = (
            len(spectra_buffer) >= chunk_size
        )

        if chunk_full or is_last_sample:

            spectra = np.stack(
                spectra_buffer,
                axis=0,
            )

            metadata = pd.DataFrame(
                metadata_buffer
            )

            spectra_path = output_dir / (
                f'spectra_{chunk_number:06d}.npy'
            )

            metadata_path = output_dir / (
                f'parameters_{chunk_number:06d}.csv'
            )

            np.save(
                spectra_path,
                spectra,
            )

            metadata.to_csv(
                metadata_path,
                index=False,
            )

            elapsed = perf_counter() - total_start

            print(
                f'Chunk {chunk_number:04d}: '
                f'{n_completed}/{n_total} spectra '
                f'({elapsed:.1f} s elapsed)'
            )

            spectra_buffer.clear()
            metadata_buffer.clear()

            chunk_number += 1

    # -----------------------------------------------------
    # SAVE RUN INFORMATION
    # -----------------------------------------------------

    elapsed = perf_counter() - total_start

    run_info = {
        'label': label,
        'n_samples': n_total,
        'n_chunks': chunk_number,
        'chunk_size': chunk_size,
        'n_ells': n_ells,
        'ells': list(manager.ells),
        'n_k': n_k,
        'k': manager.k.tolist(),
        'z': manager.z,
        'tracer': manager.tracer,
        'nbar': manager.nbar,
        'prior_basis': manager.prior_basis,
        'generation_time_seconds': elapsed,
        'cache_info': manager.cache_info(),
        'cosmology_parameters': list(
            cosmology_parameters
        ),
    }

    with open(
        output_dir / 'run_info.json',
        'w',
    ) as f:

        json.dump(
            run_info,
            f,
            indent=4,
        )

    print('=' * 55)
    print('Dataset generation complete.')
    print(f'Spectra generated: {n_completed}')
    print(f'Chunks saved: {chunk_number}')
    print(f'Total time: {elapsed:.2f} seconds')
    print('Cache:', manager.cache_info())

    return run_info
