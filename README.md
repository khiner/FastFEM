# FastFEM

FastFEM is an Apple-Silicon finite-element library for tetrahedralization, modal analysis, and immersed finite-cell solves.
The project began with FEM code from [MeshEditor commit `b1dbf2c`](https://github.com/khiner/MeshEditor/commit/b1dbf2c94398e3287c4aa48e8e3c7786e5376829).

## Build

FastFEM requires macOS, C++23, and CMake 3.28 or newer.
Tests and the surface benchmark runner require Python 3.10 or newer.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure -j1
```

The optional standalone Metal toolchain lets CMake embed a precompiled metallib.
With `metal-tt`, CMake loads a binary pipeline archive instead of compiling production kernels at runtime:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Usage

`fastfem::Surface2Modes` converts a watertight surface mesh into a modal model:

```cpp
#include <FastFEM/Surface2Modes.h>

auto result = fastfem::Surface2Modes(
    positions, triangleIndices, material, excitationPositions, {1, 1, 1},
    fastfem::Discretization::FiniteCell
);
```

### Discretization

`fastfem::Discretization::FiniteCell` embeds the surface in a Cartesian Q2 background grid.
It integrates the enclosed volume without generating a tetrahedral mesh.
`fastfem::Discretization::Tet10` tetrahedralizes the enclosed volume and solves it with quadratic tetrahedral elements.

Set `config.Tetrahedralization.Refinement` to a `fastfem::TetRefinement` mode:

| Mode | Behavior |
| --- | --- |
| `None` (default) | Basic tetrahedralization and repair |
| `Quality` | Refine element shapes without a size constraint |
| `QualityAndResolution` | Subdivide the input surface and refine tetrahedra to the target resolution |

`config.Resolution` is a positive integer, defaulting to 12. It sets target spacing `h = L / Resolution`, where `L` is the longest input bounding-box extent:

- Tet10's `QualityAndResolution` mode subdivides edges to at most `h` after optional surface simplification, then tetrahedralizes with volume target `h³ / 6`. Subdivision preserves the piecewise-planar surface.
- Finite cell uses `ceil(extent / h)` cells per axis, with a minimum of one. Grid padding increases the spacing.

Uniform scaling preserves relative resolution. Equal resolution gives comparable target spacing, but degrees of freedom and modal accuracy can differ.
Tet10's `None` and `Quality` modes do not use the resolution target.

### Result

`Surface2Modes` returns `std::expected<fastfem::ModalResult, std::string>`.
An error contains a diagnostic string.

| Member | Contents |
| --- | --- |
| `Modes` | Audible frequencies, damping times, and mass-normalized shapes at the requested positions |
| `Mass` | Mass, center of mass, and principal inertia |
| `Summary` | Unfiltered eigenvalues and sampled shapes for `RescaleModes` |
| `Basis` | Optional opaque Tet10 warm-start basis requested through `SolveReuse::KeepBasis` |
| `SamplePointOfExcitation` | Index in `Modes.Positions` for each supplied excitation position |
| `Tetrahedra` | Tet10 volume points and tetrahedra. Empty for finite cell |

Access the generated Tet10 mesh through `result->Tetrahedra.Points` and `result->Tetrahedra.Tets`.
Pass `&result->Basis` as `SolveReuse::SeedBasis` for a compatible Tet10 solve.

### Material rescaling

`fastfem::RescaleModes` updates frequencies, damping times, and mass normalization after density or Young's modulus changes without solving again.
The geometry and Poisson ratio must remain unchanged.

```cpp
auto rescaled = fastfem::RescaleModes(result->Summary, result->Modes, updatedMaterial);
```

It returns `std::nullopt` if the summary cannot support the material change.

Run surface solves from the command line with `FastFEMModalSolve`:

```sh
./build/FastFEMModalSolve model.obj --discretization finite-cell
./build/FastFEMModalSolve model.obj --discretization tet10
./build/FastFEMModalSolve model.obj --discretization tet10 --refinement quality-and-resolution --resolution 12
```

## Solvers

FastFEM provides two discretizations with independent accuracy and certification checks.

### Tet10

`SolveTet10Modes` accepts a quadratic tetrahedral volume mesh and assembles stiffness and mass directly into a 3-by-3 block pencil.
It factors the pencil with relaxed-supernodal Cholesky and solves the FP64 generalized elastic eigenproblem.
A deterministic block shift-invert iteration extracts the modes, while Accelerate supplies fill-reducing ordering and dense BLAS/LAPACK kernels.

Pass a `SolveCache` to preserve the block pencil and symbolic factorization between compatible solves.
`SolveTet10Modes` can seed a guarded block-subspace re-solve with modes from a geometry-compatible prior solution.
Project-owned column-major matrices call Accelerate BLAS and LAPACK directly for dense algebra and certification.

### Finite cell

`SolveFiniteCellEigenpairs` operates on an implicit domain or a watertight triangle surface embedded in a Cartesian Q2 background grid.
Its matrix-free path combines:

- signed moment-fitted cut integration
- vectorized paired FP64 mass and shifted actions with exact packed cut operators
- a four-guard P1 block-subspace seed
- one packed localized multiplicative Metal correction with cooperative batch-eight local matrices
- a degree-four resident P1 multigrid cycle
- compact FP32 recurrence history whose exact FP64 actions overlap the Metal correction
- FP64 Ritz algebra and convergence checks
- precompiled Metal kernels and a binary pipeline archive when the installed toolchain supports them.

`SolveFiniteCellEigenpairs` evaluates exact physical FP64 residuals from the action panels computed during iteration.
It applies the same factor-free solver to every geometry and problem size.
An unconverged factor-free result selects the assembled FP64 eigensolver.
If neither solver converges within the supplied iteration budget, `SolveFiniteCellEigenpairs` returns an error.
Validation code calls `modal::finite_cell::CertifyEigenpairs` to recompute residuals and mass orthogonality independently.
Validation compares the factor-free solve against the assembled eigensolver and moment-fitted integration against uncompressed octree quadrature.

## Correctness corpus

The CTest suite includes:

- exact matrix actions, signed-moment equivalence, transfer adjointness, affine interpolation, and tetrahedralizer invariants
- longitudinal rod, Saint-Venant torsion, and Euler-Bernoulli asymptotic bar checks
- exact Lamb frequencies and eigenspaces for solid and concentric hollow spheres
- the exact traction-free torsional subset of finite circular cylinders
- plane-stress disk and Kirchhoff-Love thin-plate asymptotic checks
- conditioned finite-cell cases varying registration, Poisson ratio, fictitious stiffness, concavity, and hollow geometry
- a deterministic 60-object real watertight mesh gate
- 50 RealImpact modal solves
- 128-mode torus and 256-mode tapered-key fallback/determinism stresses
- the 110-object tetrahedralizer snapshot corpus when installed.

The test suite evaluates Tet10 and finite cell independently.
Tests use analytical frequencies and sampled eigenspaces where available.
Other cases measure physical residuals and mass orthogonality, compare against an assembled FP64 solve, and evaluate cluster-aware sampled-subspace MAC.
Cross-discretization agreement is a secondary diagnostic.

### Test suites

| CTest suite | Coverage |
| --- | --- |
| `FastFEMSurfaceBenchmarkDiagnostics` | Frequency and sampled eigenspace comparison math |
| `FastFEMSurface2Modes` | Public results, progress, refinement, and scale behavior |
| `FastFEMModalSolver` | Tet10 assembly, factorization reuse, tetrahedralization, and surface subdivision |
| `FastFEMFiniteCell` | Finite-cell integration, matrix actions, transfers, and small solver comparisons |
| `FastFEMAnalyticalModal` | Analytical spectra, eigenspaces, and convergence |
| `FastFEMFiniteCellRobustness` | Conditioning, registration, and high-mode fallback |
| `FastFEMFiniteCellRealMesh` | Finite-cell certification on a fixed surface corpus |
| `FastFEMFiniteCellRealImpact` | Finite-cell certification on the RealImpact scans |
| `FastFEMTetCorpus` | Tetrahedralization corpus snapshots |

## Performance benchmarks

### Surface-to-modes comparison

```sh
./script/BenchmarkSurface model.obj --resolution 12 --fem-modes 45 --repetitions 3 > comparison.json
```

Compares Tet10 and finite cell on the same watertight OBJ surface and resolution, with matching material, modal settings, and sample positions. Tet10 uses `QualityAndResolution`.
The surface must enclose one connected solid, such as `tests/fixtures/cube.obj`. Use `--solver PATH` to select a build.

Timings cover the full public `fastfem::Surface2Modes` call, including preparation, assembly, solve and fallback, sampling, and result construction. Process startup, OBJ loading, and JSON output are excluded.
Samples run in fresh processes with alternating route order and no warmup or reuse.

The JSON report includes commands, input hash, individual and median times, tetrahedron counts, undamped frequencies, frequency and mass differences, and sampled eigenspace overlap.
Frequencies exclude six rigid modes and bypass the retained frequency band. Overlap uses up to 64 input vertices and clusters modes within 1%, returning `null` for rank-deficient or potentially truncated clusters.
Failed, incomplete, timed-out, or inconsistent solves abort the benchmark.

### Tet10 factorization benchmark

This diagnostic measures Tet10 factorization and linear solves after tetrahedralizing an OBJ surface. Assembly time is reported separately.
Prefix the command with `/usr/bin/time -l` to report peak memory:

```sh
./build/FastFEMTet10CholeskyBenchmark --obj model.obj 5 16
```

## Optional corpora

```sh
./script/SetupTetCorpus --skip-realimpact --thingi10k
cmake -S . -B build -DFASTFEM_TET_CORPUS_DIR="$PWD/external/TetCorpus"
ctest --test-dir build -R 'FastFEMTetCorpus|FastFEMFiniteCellRealMesh|FastFEMFiniteCellRealImpact' --output-on-failure -j1
```

Pass `--realimpact-dir PATH` instead of `--skip-realimpact` to include the RealImpact corpus.
Set `FASTFEM_REALIMPACT_DATASET_DIR` to enable the RealImpact surface-simplification tests.
Tests report missing external corpora as skipped cases.

## Determinism

FastFEM constrains Accelerate sparse work to one thread before the first factorization.
Single-threaded sparse execution produces repeatable spectra, eigenspaces, certification, and solver selection for identical inputs on one machine.

```sh
cmake -S . -B build -DFASTFEM_PARALLEL_SPARSE=ON
```

The C++ API fixes the factorization policy before Accelerate initializes.
Enable parallel sparse execution only when throughput takes priority over bitwise repeatability across processes.
