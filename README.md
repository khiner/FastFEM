# FastFEM

FastFEM is an Apple-Silicon finite-element library for tetrahedralization, modal analysis, and immersed finite-cell solves. Its first FEM code was [copied from MeshEditor at `b1dbf2c`](https://github.com/khiner/MeshEditor/commit/b1dbf2c94398e3287c4aa48e8e3c7786e5376829).

## Build

The project requires macOS, C++23, CMake 3.28 or newer, and a CMake-discoverable Eigen installation.

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure -j1
```

The standalone Metal toolchain is optional. When it is available, CMake embeds a precompiled metallib and can load a binary pipeline archive instead of compiling production kernels at runtime:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Solvers

FastFEM provides two discretizations with independent accuracy and certification checks.

### Tet10

`mesh2modes` accepts a quadratic tetrahedral mesh and solves the generalized elastic eigenproblem in FP64. It supports reusable solve state and geometry-compatible eigenvector seeds for repeated modal synthesis after material or geometry changes.

The public entry point and its supporting configuration live in `src/audio/mesh2modes.h`. `FastFEMModalSolve` runs the complete watertight-surface-to-modal-model chain:

```sh
./build/FastFEMModalSolve model.obj
```

### Finite cell

`SolveFiniteCellBlock` operates directly on an implicit domain or a watertight triangle surface embedded in a Cartesian Q1/Q2 background grid. Its matrix-free path combines:

- signed moment-fitted cut integration;
- vectorized paired FP64 mass and shifted actions with exact packed cut operators;
- a four-guard P1 Spectra seed;
- one packed localized multiplicative Metal correction with cooperative batch-eight local matrices;
- a degree-four resident P1 multigrid cycle;
- compact FP32 recurrence history with FP64 Ritz algebra and certification;
- precompiled Metal kernels and a binary pipeline archive when the installed toolchain supports them.

Every returned result is checked with physical FP64 residuals and mass orthogonality. If the factor-free Metal attempt stagnates or misses that contract, the same API falls back to assembled FP64 Cholesky shift-invert. `SolveFiniteCellBlockCholesky` provides the assembled reference route used by correctness tests.

The default route does not branch on geometry or problem size. It attempts the factor-free route first and routes only on measured certification. If neither stage certifies within the supplied iteration budget, it throws instead of returning an unverified spectrum.

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

Neither Tet10 nor finite cell is treated as ground truth for the other. Tests use analytical frequencies and sampled analytical eigenspaces where they exist. Other cases require independently evaluated physical residuals, mass orthogonality, agreement with a same-discretization assembled FP64 oracle, and cluster-aware sampled-subspace MAC. Cross-discretization agreement is a secondary diagnostic.

The audio-scale corpus can also be run directly:

```sh
./build/FastFEMFiniteCellAudioCorpus 6 128 all
```

The dedicated resolution-eight 256-mode tapered-key stress is part of `FastFEMFiniteCellConsolidationTest`.

## Optional corpora

```sh
./script/SetupTetCorpus --skip-realimpact --thingi10k
cmake -S . -B build -DFASTFEM_TET_CORPUS_DIR="$PWD/external/TetCorpus"
ctest --test-dir build -R 'FastFEMTetCorpus|FastFEMFiniteCellRealMeshConsolidation' --output-on-failure -j1
```

`FASTFEM_REALIMPACT_DATASET_DIR` enables optional RealImpact cases. Missing external corpora report a skip rather than silently reducing a selected corpus.

## Determinism

FastFEM sets `VECLIB_MAXIMUM_THREADS=1` before the first Accelerate sparse factor. This makes spectra, eigenspaces, certification, and routing repeatable for the same inputs on the same machine. A host that explicitly accepts nondeterministic sparse reductions may opt in only at configure time:

```sh
cmake -S . -B build -DFASTFEM_PARALLEL_SPARSE=ON
```

There is no late mutable C++ switch, so a process cannot accidentally change factorization policy after Accelerate has initialized.
