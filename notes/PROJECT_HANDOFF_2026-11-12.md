# Anchored P(k) Emulator — Handoff (2026-11-12)

**Purpose:** Resume the factorized Velocileptors PT-coefficient emulator work without repeating already-resolved comparison-cell failures. This handoff supersedes the immediate-next-steps portion of `PROJECT_HANDOFF_2026-11-10.md`; the broader scientific motivation and design history in that document remain useful.

## 1. Current project state

- **Repository:** [`zbrown89/cosmo-emulator`](https://github.com/zbrown89/cosmo-emulator)
- **Working branch:** `test-connection`
- **NERSC project checkout:** `/global/u1/z/zbrown5/power_emulator/project`
- **Current completed PT-table dataset:** `velocileptors_pt_lhs_v2_restricted`
- **Dataset directory (absolute):** `/global/u1/z/zbrown5/power_emulator/project/training_data/velocileptors_pt_lhs_v2_restricted`
- **PT-table files:** `/global/u1/z/zbrown5/power_emulator/project/training_data/velocileptors_pt_lhs_v2_restricted/pt_tables/`
- The notebooks use `Path("./training_data")`; run them from the project checkout root, or adjust `DATA_ROOT` explicitly. Do not silently fall back to the older `velocileptors_pt_lhs_v1` dataset.

The current dataset is the restricted v2 LHS sample. It has been generated and used to train both the P0/P2/P4 and P0/P2 emulator variants. Do not assume a file count without checking the actual `cosmo_*.npz` files on NERSC.

## 2. Relevant notebooks

All paths below are relative to the repository root, on `test-connection`.

| Notebook | Purpose / state |
|---|---|
| [`00_theory_reconstruction.ipynb`](../notebooks/00_theory_reconstruction.ipynb) | Initial direct-theory / PT-table reconstruction work |
| [`02_generate_training_tables.ipynb`](../notebooks/02_generate_training_tables.ipynb) | PT-table dataset generation |
| [`03_train_pt_coefficient_emulator.ipynb`](../notebooks/03_train_pt_coefficient_emulator.ipynb) | P0/P2/P4 emulator training |
| [`04_train_p0_p2_pt_coefficient_emulator.ipynb`](../notebooks/04_train_p0_p2_pt_coefficient_emulator.ipynb) | P0/P2 emulator training; contains the corrected standard-basis comparison cells |
| [`05_compare_pca_dimensions_p024_vs_p02.ipynb`](../notebooks/05_compare_pca_dimensions_p024_vs_p02.ipynb) | First PCA-dimension comparison |
| [`06_standardized_pca_targets_dimension_sweep.ipynb`](../notebooks/06_standardized_pca_targets_dimension_sweep.ipynb) | Current experiment: standardized PCA targets, larger PCA sweep, three NN seeds |

Notebook 06 is the next item awaiting the user's NERSC run and results. Its experiment covers both target sets, PCA dimensions `[8, 12, 16, 24, 32, 48, 64, 80, 96]`, and network seeds `[11, 22, 33]` (54 runs total). It standardizes each PCA coordinate with a `StandardScaler` fitted only on training latent coordinates, then reverses that scaling before inverse PCA. It records PCA-only reconstruction, NN table and spectrum errors, per-multipole errors, per-PC errors, training/validation histories, and seed variability.

**Validation caveat:** notebook 06 uses the fixed holdout split for early stopping/model selection. Its holdout metrics are development diagnostics, not an untouched final-test estimate. After selecting a training recipe, use an independent cosmology test set for a final reported accuracy.

## 3. PCA investigations so far

### Notebook 05: unstandardized PCA-coordinate targets

The first sweep compared P0/P2 and P0/P2/P4 using the same six PCA dimensions: 8, 12, 16, 24, 32, 48.

| PCA components | PCA table RMS P0/P2 | PCA table RMS P0/P2/P4 | NN table RMS P0/P2 | NN table RMS P0/P2/P4 |
|---:|---:|---:|---:|---:|
| 8 | 0.011624 | 0.012180 | 0.036987 | 0.037071 |
| 12 | 0.004646 | 0.005372 | 0.119767 | 0.041079 |
| 16 | 0.002672 | 0.003108 | 0.095320 | 0.104974 |
| 24 | 0.001066 | 0.001215 | 0.219614 | 0.256755 |
| 32 | 0.000505 | 0.000561 | 0.213053 | 0.227285 |
| 48 | 0.000188 | 0.000230 | 0.223131 | 0.237382 |

The PCA-only reconstruction error kept decreasing strongly through 48 components; it had **not** saturated. Cumulative variance explained was already very high, but that did not guarantee accurate NN prediction of the latent coordinates.

The NN errors became much worse above 8 components and were non-monotonic. This suggested a training/target-conditioning problem rather than a PCA information limit. The working hypothesis was that PCA scores have very different variances and that unscaled multi-output MSE makes later coordinates difficult to learn.

### Notebook 06: current follow-up test

Notebook 06 tests that hypothesis by standardizing PCA scores before NN training and undoing the scaling before inverse PCA. It extends the sweep to 64, 80, and 96 components, and repeats each target/dimension combination with three random network seeds. Do not infer the outcome until the NERSC run has completed and its summary has been inspected.

Interpretation priorities:
1. Compare NN table error against PCA-only reconstruction error.
2. Prefer reconstructed-spectrum error as the practical performance metric.
3. Inspect train/validation histories to distinguish optimization/underfitting from overfitting.
4. Inspect per-PC errors to see whether later latent directions remain difficult.
5. Check variability across the three seeds before selecting a PCA dimension.
6. Remember the holdout/early-stopping caveat above.

## 4. P0/P2/P4 versus P0/P2

Both emulators use the same cosmological inputs and the same underlying completed v2 PT-table dataset. P0/P2 is formed by selecting the first two multipole slices from the stored P0/P2/P4 tables; it is not a separately generated theory dataset.

The initial comparison showed:
- P0/P2 has modestly lower PCA-only table reconstruction error at every tested dimension.
- At 8 PCs, NN table errors were almost identical for the two target families.
- At larger dimensions, both NN models performed poorly in the original unstandardized-target experiment.
- Whether P0/P2 genuinely needs fewer PCA dimensions or attains better final spectrum accuracy remains unresolved pending notebook 06.

The P0/P2/P4 table shape is `(3, 117, 19)`; P0/P2 is `(2, 117, 19)`. The native k grid and component ordering must be read from the saved NPZ files, not guessed.

## 5. CRITICAL: permanent prescription for the recurring growth-rate / comparison-cell bug

**Do not rebuild the final theory-comparison cell by guessing where a growth rate lives. The working comparison is already implemented in notebook 04. Reuse its standard-basis route.**

### Correct and preferred route: standard nuisance basis (no growth-rate extraction)

The successful comparison uses:

```python
from src.theory import TheoryManager, Z_FID, K_FID

ELL_COMPARE = [0, 2]  # use [0, 2, 4] for the P0/P2/P4 emulator

theory_manager = TheoryManager(
    max_cache_size=1,
    z=Z_FID,
    ells=ELL_COMPARE,
    prior_basis="standard",
)
```

Use the 11 standard nuisance coefficients in exactly this order:

```python
STANDARD_PARAMETER_ORDER = (
    "b1", "b2", "bs", "b3",
    "alpha0", "alpha2", "alpha4", "alpha6",
    "sn0", "sn2", "sn4",
)
standard_pars = np.asarray(
    [FIDUCIAL_NUISANCE[name] for name in STANDARD_PARAMETER_ORDER],
    dtype=float,
)
```

The direct-theory evaluation receives the same nuisance values in the usual nested dictionaries:

```python
nuisance_for_theory = {
    "bias": {k: FIDUCIAL_NUISANCE[k] for k in ("b1", "b2", "bs", "b3")},
    "eft": {k: FIDUCIAL_NUISANCE[k] for k in
            ("alpha0", "alpha2", "alpha4", "alpha6")},
    "stochastic": {k: FIDUCIAL_NUISANCE[k] for k in ("sn0", "sn2", "sn4")},
}
pk_true = np.asarray(
    theory_manager.evaluate(TEST_COSMOLOGY, **nuisance_for_theory),
    dtype=float,
)
```

The PT-table emulator prediction is then contracted **directly** with the standard coefficients:

```python
from desilike.theories.galaxy_clustering.full_shape import (
    _velocileptors_combine_bias_terms_spectrum2_poles,
)

native_pk = np.asarray(
    _velocileptors_combine_bias_terms_spectrum2_poles(
        pt_table_pred, standard_pars
    ),
    dtype=float,
)
pk_emulator = np.vstack([
    np.interp(K_FID, native_k_reference, row) for row in native_pk
])
```

Notebook 04's cell defining `reconstruct_spectrum_from_table()` adds input/output shape checks and returns both the output-grid and native-grid spectra. Its final comparison cell is the known-working reference implementation.

**In standard basis, do not:**
- Extract a growth rate.
- Call `get_theory()` just to obtain growth rate.
- Call `get_physical_stochastic_settings()`.
- Pass `fsat`, `sigv`, or `nbar` to the PT-table contraction.
- Call `_velocileptors_physical_to_standard`.
- Apply a physical-to-standard nuisance conversion a second time.

### If a physical-basis calculation is genuinely required

The exact growth-rate expression established in the project is:

```python
growth_rate = float(
    np.asarray(theory.pt.fsigma8) / np.asarray(theory.pt.sigma8)
)
```

That is, use the initialized theory object's PT calculator and the ratio `theory.pt.fsigma8 / theory.pt.sigma8`. Do **not** search guessed attributes such as `theory.f`, `theory.growth_rate`, or template attributes. The earlier loop over guessed attributes was wrong and produced:

`RuntimeError: Could not locate growth rate on initialized theory/template.`

The exact expression above is the only growth-rate access path confirmed in prior project work. But for the current emulator spectrum comparison, prefer the standard-basis route and avoid needing it at all.

### Other recurring comparison-cell failure to avoid

A later edit produced `NameError: name 'predict_pt_table' is not defined`. This happened because the comparison cell called a helper that was absent from the active notebook state. Before executing a comparison cell:
- Ensure the notebook has executed the cell defining `predict_pt_table(cosmology)`.
- Preserve the helper when copying/revising final comparison cells.
- In notebook 04, the helper uses the fitted `emulator`, `x_scaler_final`, `pca_final`, `component_scale_final`, and reshapes to `(2, len(native_k_reference), 19)`.
- Do not assume a function exists merely because a later cell calls it; verify its definition and run order.

The recurring CLASS warning

`Class did not read input parameter(s): sigma8_cb`

was observed alongside these failures. It is a warning, not the cause of the missing-growth-rate RuntimeError or the missing-`predict_pt_table` NameError. Diagnose the actual traceback separately; do not treat that warning as the stopping exception by default.

**Mandatory procedure for future sessions:** first inspect the known-working comparison cells in `notebooks/04_train_p0_p2_pt_coefficient_emulator.ipynb` (especially the reconstruction helper and final comparison cell). Preserve `prior_basis="standard"`, the exact 11-parameter ordering, the `predict_pt_table` definition, and native-to-output k interpolation. Do not substitute an improvised growth-rate search.

## 6. Current nuisance convention and spectrum reconstruction

The current working comparison uses a standard-basis nuisance dictionary. The demonstrated fiducial example in notebook 04 is:

```python
FIDUCIAL_NUISANCE = {
    "b1": 2.0, "b2": 0.0, "bs": 0.0, "b3": 0.0,
    "alpha0": 0.0, "alpha2": 0.0, "alpha4": 0.0, "alpha6": 0.0,
    "sn0": 0.0, "sn2": 0.0, "sn4": 0.0,
}
```

The true theory and emulator contraction must use the same nuisance values. The PT tables live on their native k grid; interpolate reconstructed spectra to `K_FID` only after native-grid contraction.

## 7. Next action when the user returns

The user will report the results from `06_standardized_pca_targets_dimension_sweep.ipynb`.

When they do:
1. Read the reported summary table and plots carefully; do not assume target scaling fixed the problem.
2. Compare both P0/P2 and P0/P2/P4 across dimensions and seeds.
3. Determine whether the NN errors approach the PCA-only floor and whether spectrum errors improve.
4. Use train/validation and per-PC diagnostics to explain the result.
5. Recommend the next experiment based on evidence. Keep the fixed-holdout caveat in mind.
6. Do not change the production training notebooks or declare a best PCA dimension until the results support that choice.

**The next expected input is the user's notebook 06 results.**
