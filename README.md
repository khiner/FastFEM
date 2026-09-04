# FastFEM

FastFEM is an Apple-Silicon finite-element library for tetrahedralization, modal analysis, and immersed finite-cell solves.
The project began with FEM code from [MeshEditor commit `b1dbf2c`](https://github.com/khiner/MeshEditor/commit/b1dbf2c94398e3287c4aa48e8e3c7786e5376829).

## Build

The project requires macOS, C++23, and CMake 3.28 or newer.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure -j1
```

The optional standalone Metal toolchain lets CMake embed a precompiled metallib.
When `metal-tt` is available, CMake also loads a binary pipeline archive instead of compiling production kernels at runtime:

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
| `Tetrahedra` | Tet10 volume points and tetrahedra; empty for finite cell |

Access the generated Tet10 mesh through `result->Tetrahedra.Points` and `result->Tetrahedra.Tets`.
Pass `&result->Basis` as `SolveReuse::SeedBasis` for a compatible later Tet10 solve.

### Material rescaling

`fastfem::RescaleModes` updates frequencies, damping times, and mass normalization after density or Young's modulus changes without solving again.
The geometry and Poisson ratio must remain unchanged.

```cpp
auto rescaled = fastfem::RescaleModes(result->Summary, result->Modes, updatedMaterial);
```

The function returns `std::nullopt` when the saved summary cannot support the requested material change.

`FastFEMModalSolve` exposes the same choice from the command line:

```sh
./build/FastFEMModalSolve model.obj --discretization finite-cell
./build/FastFEMModalSolve model.obj --discretization tet10
```

## Solvers

FastFEM provides two discretizations with independent accuracy and certification checks.

### Tet10

`SolveTet10Modes` accepts a quadratic tetrahedral volume mesh and assembles stiffness and mass directly into a 3-by-3 block pencil.
It factors the pencil with relaxed-supernodal Cholesky and solves the FP64 generalized elastic eigenproblem.
A deterministic block shift-invert iteration extracts the modes, while Accelerate supplies fill-reducing ordering and dense BLAS/LAPACK kernels.

Pass a `SolveCache` to preserve the block pencil and symbolic factorization between compatible solve calls.
`SolveTet10Modes` can seed a guarded block-subspace re-solve with modes from a geometry-compatible prior solution.
Project-owned column-major matrices call Accelerate BLAS and LAPACK directly for dense algebra and certification.

### Finite cell

`SolveFiniteCellEigenpairs` operates on an implicit domain or a watertight triangle surface embedded in a Cartesian Q2 background grid.
Its matrix-free path combines:

- signed moment-fitted cut integration;
- vectorized paired FP64 mass and shifted actions with exact packed cut operators;
- a four-guard P1 block-subspace seed;
- one packed localized multiplicative Metal correction with cooperative batch-eight local matrices;
- a degree-four resident P1 multigrid cycle;
- compact FP32 recurrence history whose exact FP64 actions overlap the Metal correction;
- FP64 Ritz algebra and convergence checks;
- precompiled Metal kernels and a binary pipeline archive when the installed toolchain supports them.

`SolveFiniteCellEigenpairs` evaluates exact physical FP64 residuals from the action panels computed during iteration.
It applies the same factor-free solver to every geometry and problem size.
An unconverged factor-free result selects the assembled FP64 eigensolver.
If neither solver converges within the supplied iteration budget, `SolveFiniteCellEigenpairs` returns an error.
Validation code calls `modal::finite_cell::CertifyEigenpairs` to recompute residuals and mass orthogonality independently.
Validation compares the factor-free solve against the assembled eigensolver and moment-fitted integration against uncompressed octree quadrature.

## Correctness corpus

The CTest suite includes:

- exact matrix actions, signed-moment equivalence, transfer adjointness, affine interpolation, and tetrahedralizer invariants;
- longitudinal rod, Saint-Venant torsion, and Euler-Bernoulli asymptotic bar checks;
- exact Lamb frequencies and eigenspaces for solid and concentric hollow spheres;
- the exact traction-free torsional subset of finite circular cylinders;
- plane-stress disk and Kirchhoff-Love thin-plate asymptotic checks;
- conditioned finite-cell cases varying registration, Poisson ratio, fictitious stiffness, concavity, and hollow geometry;
- a deterministic 60-object real watertight mesh gate;
- 128-mode torus and 256-mode tapered-key fallback/determinism stresses;
- the 110-object tetrahedralizer snapshot corpus when installed.

The test suite evaluates Tet10 and finite cell independently.
Tests use analytical frequencies and sampled analytical eigenspaces where they exist.
Other cases measure physical residuals and mass orthogonality, compare against an assembled FP64 solve, and evaluate cluster-aware sampled-subspace MAC.
Cross-discretization agreement provides a secondary diagnostic.

### Audio-scale corpus

```sh
./build/FastFEMFiniteCellAudioCorpus 6 128 all
```

### Tet10 factorization benchmark

The benchmark measures native factorization over structured or tetrahedralized Tet10 inputs with configurable repetition counts and panel widths.
`/usr/bin/time -l` reports peak memory:

```sh
./build/FastFEMTet10CholeskyBenchmark --tet 34 17 11 5 16
./build/FastFEMTet10CholeskyBenchmark --obj model.obj 5 16
```

The dedicated resolution-eight 256-mode tapered-key stress is part of `FastFEMFiniteCellRobustnessTest`.

## Optional corpora

```sh
./script/SetupTetCorpus --skip-realimpact --thingi10k
cmake -S . -B build -DFASTFEM_TET_CORPUS_DIR="$PWD/external/TetCorpus"
ctest --test-dir build -R 'FastFEMTetCorpus|FastFEMFiniteCellRealMesh' --output-on-failure -j1
```

Set `FASTFEM_REALIMPACT_DATASET_DIR` to enable optional RealImpact cases.
Tests report missing external corpora as skipped cases.

## Determinism

FastFEM constrains Accelerate sparse work to one thread before the first factorization.
Single-threaded sparse execution produces repeatable spectra, eigenspaces, certification, and solver selection for identical inputs on one machine.

```sh
cmake -S . -B build -DFASTFEM_PARALLEL_SPARSE=ON
```

The C++ API fixes the factorization policy before Accelerate initializes.
Enable parallel sparse execution only when throughput takes priority over bitwise repeatability across processes.
