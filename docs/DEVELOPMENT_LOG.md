# Development Log

This file is the durable record of implementation decisions and test results.
Record the exact test command, environment, result, and relevant commit. Do not
mark a test as passing unless its output has been observed.

## 2026-11-10 — Initial GitHub Actions API-test setup

**Branch:** test-connection

**Purpose:** Establish automated compatibility checks for the external
cosmology packages used by the emulator, especially cosmoprimo and desilike.

**Files added:**
- .github/workflows/api-tests.yml: GitHub Actions workflow using Python 3.11.
- tests/test_external_apis.py: import and API-contract smoke tests.

**Dependency installation:** The workflow installs the current development
version of desilike from the official cosmodesi/desilike GitHub repository.
Its declared dependencies include cosmoprimo and lsstypes. The workflow prints
pip freeze so the exact resolved package versions are visible in the Actions
log.

**Test scope:**
- Import the cosmoprimo and desilike symbols used by the project.
- Check the exact Velocileptors theory class name imported by theory.py.
- Check the stochastic-settings keys consumed by TheoryManager initialization.
- Compile all repository Python files to catch syntax errors.

**Important limitation:** These are API smoke tests, not a full theory
calculation. They do not initialize CLASS, execute a power-spectrum
calculation, or access NERSC data.

**Initial status:** Files added; GitHub Actions result not yet observed.
The upstream package version used by CI must later be compared with the
version installed in the NERSC cosmodesi environment. A failure against
upstream main is evidence of an API mismatch with that version, not by
itself proof that the NERSC environment is broken.


## 2026-11-10 — Core module unit tests

**File added:** `tests/test_core_modules.py`.

**Coverage added:**
- Cosmology catalogue sorting, ID filtering, missing IDs, duplicate IDs,
  non-finite values, dictionary construction, and metadata extraction.
- Latin Hypercube reproducibility, bounds, invalid inputs, Pipeline A
  shared nuisance design, and Pipeline B combined sampling.
- Nuisance LHS reproducibility, bounds, fixed parameters, and nested
  nuisance dictionary structure.
- A regression check that `print_cosmology_summary` is available at
  module scope.

**Test data:** Small synthetic pandas DataFrames and temporary CSV files.
No Abacus catalogue, CLASS calculation, or spectrum generation is needed.

**CI environment adjustment:** The first external API test run installed
DESI Python packages and passed Python compilation, but all three API
imports failed because the GitHub runner lacked a system MPI library
(`libmpi.so`). The workflow now installs OpenMPI runtime/development
packages before Python dependencies. This is an environment setup failure,
not yet evidence of an upstream Python API mismatch.

**Status:** Core tests and MPI-enabled external API tests have been added to
CI. Their results are pending; do not mark them as passing until observed.
