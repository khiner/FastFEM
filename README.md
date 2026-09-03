# FastFEM

FastFEM is an Apple-Silicon finite-element library for tetrahedralization, modal analysis, and immersed finite-cell solves.
The project began with FEM code from [MeshEditor commit `b1dbf2c`](https://github.com/khiner/MeshEditor/commit/b1dbf2c94398e3287c4aa48e8e3c7786e5376829).

## Build

The project requires macOS, C++23, CMake 3.28 or newer, and a CMake-discoverable Eigen installation.

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

## Solvers

FastFEM provides two discretizations with independent accuracy and certification checks.

### Tet10

`mesh2modes` accepts a quadratic tetrahedral volume mesh and assembles stiffness and mass directly into a 3-by-3 block pencil.
It factors the pencil with relaxed-supernodal Cholesky and solves the FP64 generalized elastic eigenproblem.
A deterministic block shift-invert iteration extracts the modes, while Accelerate supplies fill-reducing ordering and dense BLAS/LAPACK kernels.

Pass a `SolveCache` to preserve the block pencil and symbolic factorization between compatible solve calls.
`mesh2modes` can seed a guarded block-subspace re-solve with modes from a geometry-compatible prior solution.
Scalar Eigen matrices provide mass actions and independent certification.

#### Usage

`src/audio/mesh2modes.h` declares the C++ API.
`FastFEMModalSolve` runs the complete watertight-surface-to-modal-model pipeline:

```sh
./build/FastFEMModalSolve model.obj
```

### Finite cell

`SolveFiniteCellBlock` operates on an implicit domain or a watertight triangle surface embedded in a Cartesian Q2 background grid.
Its matrix-free path combines:

- signed moment-fitted cut integration;
- vectorized paired FP64 mass and shifted actions with exact packed cut operators;
- a four-guard P1 block-subspace seed;
- one packed localized multiplicative Metal correction with cooperative batch-eight local matrices;
- a degree-four resident P1 multigrid cycle;
- compact FP32 recurrence history whose exact FP64 actions overlap the Metal correction;
- FP64 Ritz algebra and independent certification;
- precompiled Metal kernels and a binary pipeline archive when the installed toolchain supports them.

`SolveFiniteCellBlock` checks every result against physical FP64 residual and mass-orthogonality bounds.
It applies the same factor-free solver to every geometry and problem size.
Certification failure selects assembled FP64 Cholesky shift-invert.
If neither solver certifies within the supplied iteration budget, `SolveFiniteCellBlock` returns an error.
`src/audio/FiniteCellOracle.h` exposes the fixed four-guard Cholesky solver to validation code.

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
Other cases measure physical residuals and mass orthogonality, compare against an assembled FP64 oracle, and evaluate cluster-aware sampled-subspace MAC.
Cross-discretization agreement provides a secondary diagnostic.

### Audio-scale corpus

```sh
./build/FastFEMFiniteCellAudioCorpus 6 128 all
```

### Sparse-backend benchmark

The benchmark accepts structured or tetrahedralized Tet10 inputs, repetition counts, backend selection, and panel widths.
`/usr/bin/time -l` reports peak memory for one backend process:

```sh
./build/FastFEMBlockSparseBenchmark --tet 34 17 11 5 native 16
./build/FastFEMBlockSparseBenchmark --obj model.obj 5 all 16
```

The dedicated resolution-eight 256-mode tapered-key stress is part of `FastFEMFiniteCellConsolidationTest`.

## Optional corpora

```sh
./script/SetupTetCorpus --skip-realimpact --thingi10k
cmake -S . -B build -DFASTFEM_TET_CORPUS_DIR="$PWD/external/TetCorpus"
ctest --test-dir build -R 'FastFEMTetCorpus|FastFEMFiniteCellRealMeshConsolidation' --output-on-failure -j1
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
