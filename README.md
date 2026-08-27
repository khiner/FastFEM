# FastFEM

MeshEditor's tetrahedralization and modal finite-element solve chain, copied here for standalone experiments before this repository replaces the in-tree copy.

The library requires macOS, C++23, CMake 3.28 or newer, and a CMake-discoverable Eigen installation. Clone its pinned dependencies and build the solver tests with:

```sh
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

`FastFEMModalSolve` runs the complete surface-to-modal-model chain on an OBJ mesh:

```sh
./build/FastFEMModalSolve model.obj
```

Set `FASTFEM_REALIMPACT_DATASET_DIR` at configure time to enable the optional RealImpact cases. They skip when the dataset is absent.

The fast test target includes synthetic tetrahedralizer cases. The deterministic corpus gate checks 110 OBJ meshes in base and quality modes against `tests/fixtures/TetCorpusSnapshot.txt`. Set up available corpora and point CMake at them with:

```sh
./script/SetupTetCorpus --skip-realimpact --thingi10k
cmake -S . -B build -DFASTFEM_TET_CORPUS_DIR="$PWD/external/TetCorpus"
ctest --test-dir build -R FastFEMTetCorpus --output-on-failure
```

Use `--realimpact-dir PATH` instead of `--skip-realimpact` to include the optional RealImpact dataset. The corpus test is reported as skipped when its root is absent and fails if a selected dataset or snapshot case is missing.
