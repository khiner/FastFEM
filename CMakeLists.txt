cmake_minimum_required(VERSION 3.28)

project(FastFEM VERSION 0.1.0 LANGUAGES CXX)

if(NOT APPLE)
    message(FATAL_ERROR "FastFEM currently requires Apple's Accelerate framework")
endif()

option(FASTFEM_BUILD_TESTS "Build FastFEM tests" ${PROJECT_IS_TOP_LEVEL})
option(FASTFEM_BUILD_TOOLS "Build FastFEM command-line tools" ${PROJECT_IS_TOP_LEVEL})
set(FASTFEM_REALIMPACT_DATASET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/RealImpact/dataset" CACHE PATH "Optional RealImpact dataset directory")
set(FASTFEM_TET_CORPUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/TetCorpus" CACHE PATH "Tetrahedralizer OBJ corpus root")

find_package(Eigen3 CONFIG REQUIRED)

function(fastfem_require_submodule path)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${path}/CMakeLists.txt")
        message(FATAL_ERROR "Missing ${path}; run: git submodule update --init --recursive")
    endif()
endfunction()

if(NOT TARGET Spectra AND NOT TARGET Spectra::Spectra)
    fastfem_require_submodule(lib/spectra)
    block()
        set(BUILD_TESTS OFF)
        set(BUILD_EXAMPLES OFF)
        add_subdirectory(lib/spectra EXCLUDE_FROM_ALL)
    endblock()
endif()
if(TARGET Spectra::Spectra)
    set(FASTFEM_SPECTRA_TARGET Spectra::Spectra)
else()
    set(FASTFEM_SPECTRA_TARGET Spectra)
endif()

if(NOT TARGET meshoptimizer AND NOT TARGET meshoptimizer::meshoptimizer)
    set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
    set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
    fastfem_require_submodule(lib/meshoptimizer)
    add_subdirectory(lib/meshoptimizer EXCLUDE_FROM_ALL)
endif()
if(TARGET meshoptimizer::meshoptimizer)
    set(FASTFEM_MESHOPTIMIZER_TARGET meshoptimizer::meshoptimizer)
else()
    set(FASTFEM_MESHOPTIMIZER_TARGET meshoptimizer)
endif()

find_library(FASTFEM_ACCELERATE_FRAMEWORK Accelerate REQUIRED)

add_library(FastFEM
    src/audio/CholeskyShiftInvert.cpp
    src/audio/mesh2modes.cpp
    src/mesh/Tetrahedralize.cpp
    src/mesh/Tets.cpp
    src/numeric/Predicates.cpp
)
add_library(FastFEM::FastFEM ALIAS FastFEM)

target_compile_features(FastFEM PUBLIC cxx_std_23)
target_include_directories(FastFEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(FastFEM
    PUBLIC
        Eigen3::Eigen
        ${FASTFEM_SPECTRA_TARGET}
    PRIVATE
        ${FASTFEM_MESHOPTIMIZER_TARGET}
        ${FASTFEM_ACCELERATE_FRAMEWORK}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(FastFEM PRIVATE
        -Wall
        -Wextra
        -Wno-missing-field-initializers
        -Wno-elaborated-enum-base
        -fno-rtti
    )
endif()

if(FASTFEM_BUILD_TESTS OR FASTFEM_BUILD_TOOLS)
    if(NOT TARGET tinyobjloader AND NOT TARGET tinyobjloader::tinyobjloader)
        set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE BOOL "" FORCE)
        set(TINYOBJLOADER_WITH_PYTHON OFF CACHE BOOL "" FORCE)
        fastfem_require_submodule(lib/tinyobjloader)
        add_subdirectory(lib/tinyobjloader EXCLUDE_FROM_ALL)
    endif()
    if(TARGET tinyobjloader::tinyobjloader)
        set(FASTFEM_TINYOBJLOADER_TARGET tinyobjloader::tinyobjloader)
    else()
        set(FASTFEM_TINYOBJLOADER_TARGET tinyobjloader)
    endif()
endif()

if(FASTFEM_BUILD_TOOLS)
    add_executable(FastFEMModalSolve tools/ModalSolveTool.cpp)
    target_link_libraries(FastFEMModalSolve PRIVATE FastFEM::FastFEM ${FASTFEM_TINYOBJLOADER_TARGET})
    target_include_directories(FastFEMModalSolve PRIVATE tests)
endif()

if(FASTFEM_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET Boost::ut)
        set(BOOST_UT_DISABLE_MODULE ON CACHE BOOL "" FORCE)
        set(BOOST_UT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(BOOST_UT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BOOST_UT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BOOST_UT_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        fastfem_require_submodule(lib/ut)
        add_subdirectory(lib/ut EXCLUDE_FROM_ALL)
    endif()

    add_executable(FastFEMModalSolverTest tests/ModalSolverTest.cpp)
    target_compile_definitions(FastFEMModalSolverTest PRIVATE
        REALIMPACT_DATASET_DIR="${FASTFEM_REALIMPACT_DATASET_DIR}"
    )
    target_include_directories(FastFEMModalSolverTest PRIVATE tests)
    target_link_libraries(FastFEMModalSolverTest PRIVATE
        FastFEM::FastFEM
        Boost::ut
        ${FASTFEM_TINYOBJLOADER_TARGET}
    )
    add_test(NAME FastFEMModalSolver COMMAND FastFEMModalSolverTest)

    add_executable(FastFEMTetCorpusTest tests/TetCorpusTest.cpp)
    target_compile_definitions(FastFEMTetCorpusTest PRIVATE
        FASTFEM_TET_CORPUS_DIR="${FASTFEM_TET_CORPUS_DIR}"
        FASTFEM_TET_SNAPSHOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/TetCorpusSnapshot.txt"
    )
    target_include_directories(FastFEMTetCorpusTest PRIVATE tests)
    target_link_libraries(FastFEMTetCorpusTest PRIVATE
        FastFEM::FastFEM
        ${FASTFEM_TINYOBJLOADER_TARGET}
    )
    add_test(NAME FastFEMTetCorpus COMMAND FastFEMTetCorpusTest)
    set_tests_properties(FastFEMTetCorpus PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endif()
