# Anchored P(k) Emulator — Project Handoff

**Draft for review — November 10, 2026**

## 1. Project objective

The project investigates whether the 52 available AbacusSummit cosmologies adequately sample the relevant eight-dimensional cosmological parameter space for emulating galaxy-clustering power-spectrum multipoles.

The broader goal is to construct an accurate, computationally efficient emulator for redshift-space galaxy-clustering predictions, ultimately supporting full-shape inference using LRG data.

The central research question is whether the cosmological coverage provided by the existing Abacus anchor set is sufficient, or whether alternative cosmological designs can provide better emulator performance.

An important architectural insight has substantially simplified the problem: **cosmological and nuisance-parameter dependence can be factorized.** This allows the emulator to focus on the cosmology-dependent perturbation-theory components rather than learning the full nuisance-parameter dependence from a large collection of spectra.

The project is exploratory. Alternative designs, sample sizes, and emulator strategies should be evaluated rather than treated as settled before their performance is demonstrated.

## 2. Cosmological parameterization

The emulator uses eight independently sampled cosmological parameters:

1. \`omega_b\`
2. \`omega_cdm\`
3. \`sigma8_cb\`
4. \`n_s\`
5. \`alpha_s\`
6. \`N_ur\`
7. \`w0_fld\`
8. \`wa_fld\`

The Hubble parameter \`h\` is derived by enforcing a fixed CMB acoustic angular scale, \`theta_*\`.

The primordial amplitude \`A_s\` is subsequently derived by solving for the requested \`sigma8_cb\`.

The relevant conversion code lives in \`src/cosmology.py\`.

### Parameter-domain construction

The initial parameter-domain strategy uses the Abacus cosmologies c130–c181 (52 cosmologies).

For each sampled parameter, bounds are constructed from the Abacus minimum and maximum, expanded around the c130 parameter value by a factor of 1.25:

\[
p_{\min}=p_{\mathrm{c130}}-1.25(p_{\mathrm{c130}}-p_{\min,\mathrm{Abacus}})
\]

\[
p_{\max}=p_{\mathrm{c130}}+1.25(p_{\max,\mathrm{Abacus}}-p_{\mathrm{c130}})
\]

This ensures that all Abacus extrema lie inside the expanded parameter intervals.

This is the current broad design domain, not a guarantee that every point in its eight-dimensional rectangular volume is numerically realizable by the cosmology conversion pipeline. The adequacy and validity of the domain should be assessed empirically.

## 3. Theoretical factorization — central emulator architecture

A key result of the theory-reconstruction work is that the Velocileptors prediction can be expressed as

\[
P_\ell(k;\theta_{\rm cosmo},\theta_{\rm nuisance})
=
\sum_{i=1}^{19}
T_{\ell i}(k;\theta_{\rm cosmo})
M_i(\theta_{\rm nuisance}).
\]

Here:

- \(T_{\ell i}\) is the cosmology-dependent PT component table.
- \(M_i\) is the nuisance-dependent coefficient vector.
- There are 19 PT components.

This factorization is the central architectural decision for the emulator.

**The primary emulator target is the cosmology-dependent PT table, not a collection of spectra spanning the nuisance-parameter space.**

The intended workflow is:

1. Generate the native PT component tables at selected cosmologies.
2. Train an emulator to predict those tables as a function of cosmology.
3. At post-processing time, construct the nuisance-dependent coefficient vector.
4. Contract the predicted PT table with that vector to reconstruct the desired power-spectrum multipoles.

This removes the need to train over a large nuisance-parameter grid. Nuisance sampling is therefore not a central requirement of the PT-table emulator.

The pilot reconstruction demonstrated very close agreement with direct theory evaluation. A small residual remained, plausibly associated with interpolation differences. This was not considered a blocker and was deferred.

The native PT table is expected to have shape \((3,n_k,19)\), with the pilot and current implementation indicating \(n_k=117\). The native PT \(k\)-grid must be retrieved from the theory object and saved with the table; it must not be assumed to be the final output grid.

The actual component ordering should be obtained from the installed theory implementation, not inferred from guessed physical labels.

## 4. Research strategy and experimental designs

The broader research program includes several possible comparisons. These are research questions to investigate, not all simultaneously approved implementation requirements.

### A. Anchored versus unanchored cosmological designs

The key comparison is between:

- The 52 Abacus anchor cosmologies.
- 52 independently selected, space-filling cosmologies in the same eight-dimensional domain.

The purpose is to test whether the Abacus cosmological design provides an advantage over an unanchored space-filling design at equal cosmology count.

Unanchored designs may be repeated across random seeds to assess sensitivity to the particular sample.

The factorized architecture means the relevant training examples are cosmology-dependent PT tables. Nuisance parameters need not be independently sampled for the primary emulator comparison.

### B. Exploring larger unanchored designs

A natural follow-up is to investigate whether substantially larger unanchored cosmology samples improve emulator performance.

The current working experiment uses a 250-point scrambled Latin hypercube in the expanded eight-dimensional domain, with seed 20261110.

This is a convenient exploratory test, not an established optimal sample size or a final scientific design. It may be sufficient to cover the parameter space, or it may not. Its adequacy must be judged by emulator performance and validation.

The current generation notebook produces cosmology-dependent PT tables. Any reconstructed spectra are secondary post-processing products used for checks or downstream applications.

No separate 50-cosmology validation sample is part of the current notebook specification.

### C. Eventual 52-cosmology emulator

The eventual anchored 52-point emulator remains an important target. The larger unanchored experiment is a way to develop and test the factorized emulator workflow before returning to the more restrictive design.

The exact relationship between the larger-sample experiment, learning curves, repeated random designs, and the final anchored comparison remains open to refinement.

## 5. Theory configuration and nuisance dependence

The fiducial redshift is:

\[
z=0.8.
\]

The output multipoles are:

\[
\ell=[0,2,4].
\]

The native PT table has its own \(k\)-grid, which must be saved alongside the table.

A final spectrum may be reconstructed on the output grid used in the earlier anchored-spectrum design:

- \(k=0.020\) to \(0.200\ h\,\mathrm{Mpc}^{-1}\)
- \(\Delta k=0.005\ h\,\mathrm{Mpc}^{-1}\)
- 36 bins

Nuisance dependence is handled through the standard Velocileptors coefficient construction and analytical contraction with the PT components.

Earlier work established useful candidate nuisance settings and ranges for spectrum reconstruction, including bias, EFT, and stochastic parameters. These remain relevant for testing the reconstruction and for eventual inference, but they are not inputs to the primary PT-table emulator.

Previously considered physical settings include:

- \(f_{\rm sat}=0.13\)
- \(\sigma_v=6.19388\)
- \(r_{\rm vir}=0.46438\)

Earlier anchored-spectrum experiments also used a number density \(n_{\rm bar}=4.56\times10^{-4}\).

These are reference choices, not requirements to sample nuisance space during PT-table training. Any final nuisance choices should be documented in the context of the downstream spectrum reconstruction or inference application.

## 6. Emulator training philosophy

The emulator should be theory-blind once the training tables have been generated.

The training and validation pipeline should operate on precomputed PT tables and cosmological inputs. It should not call CLASS, Velocileptors, or \`TheoryManager\` during emulator fitting or prediction.

External theory comparisons should occur separately, after emulator predictions have been made.

Validation splits should be grouped by entire cosmology, not by individual \(k\)-bins, multipoles, or nuisance-reconstructed spectra. This avoids leakage of information from a cosmology represented in both training and validation.

PCA plus Gaussian processes has been a candidate architecture for compressing and emulating the PT-table outputs, but the final architecture and hyperparameters remain to be established empirically.

## 7. Software and repository state

**GitHub repository:** \`zbrown89/cosmo-emulator\`

**Branch:** \`test-connection\`

**NERSC working directory:**

\`/global/u1/z/zbrown5/power_emulator/project\`

Relevant source modules:

- \`src/cosmology.py\`
- \`src/theory.py\`

Relevant notebooks:

- \`notebooks/00_theory_reconstruction.ipynb\`
- \`notebooks/01_pt_table_pilot.ipynb\`
- \`notebooks/02_generate_training_tables.ipynb\`

Notebook 02 is configured to generate 250 cosmologies using a scrambled eight-dimensional Latin hypercube and save one compressed NPZ file per cosmology.

Expected output structure:

\`\`\`text
training_data/
└── velocileptors_pt_lhs_v1/
    ├── cosmology_design.csv
    ├── run_config.json
    ├── progress.csv
    └── pt_tables/
        ├── cosmo_0000.npz
        ├── cosmo_0001.npz
        └── ...
\`\`\`

Each NPZ file stores the PT table, native \(k\)-grid, multipoles, component labels, cosmology information, and metadata.

The component labels currently use generic indexed names. They are placeholders, not claims about the physical ordering of the 19 components.

Notebook URL:

https://github.com/zbrown89/cosmo-emulator/blob/test-connection/notebooks/02_generate_training_tables.ipynb

The notebook has been uploaded and downloaded to NERSC. Its current generation run has resumed after numerical bracket adjustments, but completion and full dataset validation remain outstanding.

## 8. Numerical and physical validity constraints

The cosmology conversion pipeline contains numerical solvers with finite search intervals and other implementation-specific operating limits.

For example:

- \(h\) is solved from the acoustic-scale condition.
- \(A_s\) is solved from the target \(\sigma_{8,\mathrm{cb}}\).

Some sampled cosmologies may fail if their derived solutions lie outside the solver's imposed search intervals.

The general lesson is to understand and respect the numerical and physical validity constraints built into \`src/cosmology.py\`.

When a failure occurs, distinguish between:

1. A valid physical solution that lies outside an unnecessarily restrictive numerical bracket.
2. A genuinely invalid or unsupported cosmological parameter combination.
3. A numerical failure of the underlying theory calculation.

Search intervals should be changed only when justified by the first case, with appropriate checks on the recovered solution. Arbitrarily expanding limits to force every LHS point to succeed is not an acceptable substitute for checking physical validity.

The recent \(A_s\) and \(h\) bracket failures are examples of this broader issue. Their specific fixes should be verified and documented in the source code.

## 9. Development workflow principles

- Develop and debug interactively in Jupyter before launching long jobs.
- Use SLURM for longer runs after the workflow has been validated.
- Avoid hard-coded paths to root or notes directories.
- Preserve the distinction between cosmology-dependent PT tables and nuisance-dependent spectra.
- Treat proposed experimental designs as proposals until explicitly approved.
- Check code changes against the current project specification before uploading.
- Keep handoff notes that distinguish established decisions, exploratory experiments, and unresolved questions.

## 10. Deferred work and open research questions

- Determine whether 250 cosmologies are sufficient for accurate emulation.
- Explore alternative unanchored sample sizes and designs.
- Return to the equal-size 52-versus-52 anchored/unanchored comparison.
- Assess the value of repeating unanchored designs across random seeds.
- Finalize the PT-table emulator architecture and hyperparameters.
- Establish robust validation against direct theory calculations.
- Reconstruct nuisance-dependent spectra for downstream inference.
- Perform eventual inference tests using LRG data and the EZmock covariance.

## 11. Immediate next steps

1. Finish the current exploratory PT-table generation run.
2. Verify all expected tables and metadata.
3. Inspect table shapes, finiteness, and native \(k\)-grid consistency.
4. Inspect derived \(h\) and \(A_s\) distributions and investigate suspicious values.
5. Establish the actual ordering of the 19 PT components from the installed implementation.
6. Build the first working emulator for the cosmology-dependent PT tables.
7. Evaluate its accuracy and determine whether the current design adequately covers the parameter space.
8. Use those results to inform the eventual anchored 52-point experiment.

## 12. Items requiring user review

1. Is the expanded Abacus-derived domain still the intended starting domain for exploratory unanchored designs?
2. Should the current 250-point LHS be treated simply as an initial experiment, with no assumption that it is the final training design?
3. Is PCA plus Gaussian processes still the preferred first emulator architecture to test?
4. Are the candidate nuisance and physical settings above worth retaining as reference choices for downstream spectrum reconstruction?
5. Is there any additional scientific motivation or intended inference application that should be stated more explicitly?

**End of draft.**
