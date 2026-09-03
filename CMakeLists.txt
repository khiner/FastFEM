cmake_minimum_required(VERSION 3.28)

project(FastFEM VERSION 0.1.0 LANGUAGES CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "FastFEM currently requires Apple's Accelerate framework")
endif()

option(FASTFEM_BUILD_TESTS "Build FastFEM tests" ${PROJECT_IS_TOP_LEVEL})
option(FASTFEM_BUILD_TOOLS "Build FastFEM command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(FASTFEM_PARALLEL_SPARSE "Allow nondeterministic multithreaded Accelerate sparse execution" OFF)
set(FASTFEM_REALIMPACT_DATASET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/RealImpact/dataset" CACHE PATH "Optional RealImpact dataset directory")
set(FASTFEM_TET_CORPUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/TetCorpus" CACHE PATH "Tetrahedralizer OBJ corpus root")

find_package(Eigen3 CONFIG REQUIRED)

function(fastfem_require_submodule path)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${path}/.git")
        message(FATAL_ERROR "Missing ${path}; run: git submodule update --init --recursive")
    endif()
endfunction()

if(NOT TARGET Spectra AND NOT TARGET Spectra::Spectra)
    fastfem_require_submodule(lib/spectra)
    block()
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
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
find_library(FASTFEM_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(FASTFEM_METAL_FRAMEWORK Metal REQUIRED)

add_library(FastFEM
    src/audio/BlockSparseCholesky.cpp
    src/audio/CholeskyShiftInvert.cpp
    src/audio/FiniteCell.cpp
    src/audio/FiniteCellBlockEigensolver.cpp
    src/audio/FiniteCellMetal.mm
    src/audio/SparseCholesky.cpp
    src/audio/Tet10Assembler.cpp
    src/audio/mesh2modes.cpp
    src/mesh/Tetrahedralize.cpp
    src/mesh/Tets.cpp
    src/numeric/Accelerate.cpp
    src/numeric/Predicates.cpp
)
add_library(FastFEM::FastFEM ALIAS FastFEM)

execute_process(COMMAND xcrun --sdk macosx --find metal RESULT_VARIABLE FASTFEM_METAL_RESULT OUTPUT_QUIET ERROR_QUIET)
execute_process(COMMAND xcrun --sdk macosx --find metallib RESULT_VARIABLE FASTFEM_METALLIB_RESULT OUTPUT_QUIET ERROR_QUIET)
execute_process(COMMAND xcrun --sdk macosx --find metal-tt RESULT_VARIABLE FASTFEM_METAL_TT_RESULT OUTPUT_QUIET ERROR_QUIET)
find_program(FASTFEM_XXD xxd)
if(FASTFEM_METAL_RESULT EQUAL 0 AND FASTFEM_METALLIB_RESULT EQUAL 0 AND FASTFEM_XXD)
    set(FASTFEM_METAL_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(FASTFEM_FINITE_CELL_METAL_SOURCE "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellKernels.metal")
    set(FASTFEM_FINITE_CELL_AIR "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellKernels.air")
    set(FASTFEM_FINITE_CELL_METALLIB "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellKernels.metallib")
    set(FASTFEM_FINITE_CELL_METALLIB_HEADER "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellMetallib.h")
    set(FASTFEM_FINITE_CELL_PIPELINES "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellPipelines.mtlp-json")
    set(FASTFEM_FINITE_CELL_BINARY_ARCHIVE "${FASTFEM_METAL_GENERATED_DIR}/FiniteCellPipelines.binary.metallib")
    set(FASTFEM_FINITE_CELL_ARCHIVE_OUTPUTS)
    set(FASTFEM_FINITE_CELL_ARCHIVE_COMMANDS)
    if(FASTFEM_METAL_TT_RESULT EQUAL 0)
        list(APPEND FASTFEM_FINITE_CELL_ARCHIVE_OUTPUTS "${FASTFEM_FINITE_CELL_BINARY_ARCHIVE}")
        list(APPEND FASTFEM_FINITE_CELL_ARCHIVE_COMMANDS
            COMMAND "${CMAKE_COMMAND}"
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/audio/FiniteCellMetal.mm
                -DOUTPUT=${FASTFEM_FINITE_CELL_PIPELINES}
                -DLIBRARY=${FASTFEM_FINITE_CELL_METALLIB}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PrepareMetalPipelineScript.cmake"
            COMMAND xcrun --sdk macosx metal-tt -target air64-apple-macos26.0 -gpu-family metal4
                "${FASTFEM_FINITE_CELL_PIPELINES}" -o "${FASTFEM_FINITE_CELL_BINARY_ARCHIVE}"
        )
    endif()
    add_custom_command(
        OUTPUT "${FASTFEM_FINITE_CELL_METALLIB_HEADER}" ${FASTFEM_FINITE_CELL_ARCHIVE_OUTPUTS}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${FASTFEM_METAL_GENERATED_DIR}"
        COMMAND "${CMAKE_COMMAND}"
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/src/audio/FiniteCellMetal.mm
            -DOUTPUT=${FASTFEM_FINITE_CELL_METAL_SOURCE}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExtractMetalSource.cmake"
        COMMAND xcrun --sdk macosx metal -c "${FASTFEM_FINITE_CELL_METAL_SOURCE}" -o "${FASTFEM_FINITE_CELL_AIR}"
        COMMAND xcrun --sdk macosx metallib "${FASTFEM_FINITE_CELL_AIR}" -o "${FASTFEM_FINITE_CELL_METALLIB}"
        ${FASTFEM_FINITE_CELL_ARCHIVE_COMMANDS}
        COMMAND "${FASTFEM_XXD}" -i -n FastFEMFiniteCellMetallib
            "${FASTFEM_FINITE_CELL_METALLIB}" "${FASTFEM_FINITE_CELL_METALLIB_HEADER}"
        DEPENDS
            src/audio/FiniteCellMetal.mm
            cmake/ExtractMetalSource.cmake
            cmake/PrepareMetalPipelineScript.cmake
        BYPRODUCTS
            "${FASTFEM_FINITE_CELL_METAL_SOURCE}"
            "${FASTFEM_FINITE_CELL_AIR}"
            "${FASTFEM_FINITE_CELL_METALLIB}"
            "${FASTFEM_FINITE_CELL_PIPELINES}"
        VERBATIM
    )
    target_sources(FastFEM PRIVATE "${FASTFEM_FINITE_CELL_METALLIB_HEADER}")
    target_include_directories(FastFEM PRIVATE "${FASTFEM_METAL_GENERATED_DIR}")
    target_compile_definitions(FastFEM PRIVATE FASTFEM_HAS_EMBEDDED_FINITE_CELL_METALLIB=1)
    if(FASTFEM_METAL_TT_RESULT EQUAL 0)
        target_compile_definitions(FastFEM PRIVATE FASTFEM_FINITE_CELL_BINARY_ARCHIVE_PATH="${FASTFEM_FINITE_CELL_BINARY_ARCHIVE}")
    endif()
else()
    message(WARNING "Metal compiler tools are unavailable; FastFEM will compile finite-cell kernels at runtime")
    target_compile_definitions(FastFEM PRIVATE FASTFEM_HAS_EMBEDDED_FINITE_CELL_METALLIB=0)
endif()

target_compile_features(FastFEM PUBLIC cxx_std_23)
target_compile_definitions(FastFEM PRIVATE FASTFEM_PARALLEL_SPARSE=$<BOOL:${FASTFEM_PARALLEL_SPARSE}>)
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
        ${FASTFEM_FOUNDATION_FRAMEWORK}
        ${FASTFEM_METAL_FRAMEWORK}
)
set_source_files_properties(src/audio/FiniteCellMetal.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)

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

    add_executable(FastFEMFiniteCellAudioCorpus tools/FiniteCellAudioCorpus.cpp)
    target_link_libraries(FastFEMFiniteCellAudioCorpus PRIVATE FastFEM::FastFEM)

    add_executable(FastFEMBlockSparseBenchmark tools/BlockSparseBenchmark.cpp)
    target_link_libraries(FastFEMBlockSparseBenchmark PRIVATE FastFEM::FastFEM ${FASTFEM_TINYOBJLOADER_TARGET})
    target_include_directories(FastFEMBlockSparseBenchmark PRIVATE tests)

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
    target_include_directories(FastFEMModalSolverTest PRIVATE tests tools)
    target_link_libraries(FastFEMModalSolverTest PRIVATE
        FastFEM::FastFEM
        Boost::ut
        ${FASTFEM_TINYOBJLOADER_TARGET}
    )
    add_test(NAME FastFEMModalSolver COMMAND FastFEMModalSolverTest)

    add_executable(FastFEMFiniteCellTest tests/FiniteCellTest.cpp)
    target_include_directories(FastFEMFiniteCellTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMFiniteCell COMMAND FastFEMFiniteCellTest)

    add_executable(FastFEMAnalyticalModalTest tests/AnalyticalModalTest.cpp)
    target_include_directories(FastFEMAnalyticalModalTest PRIVATE tests tools)
    target_link_libraries(FastFEMAnalyticalModalTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMAnalyticalModal COMMAND FastFEMAnalyticalModalTest)

    add_executable(FastFEMFiniteCellConsolidationTest tests/FiniteCellConsolidationTest.cpp)
    target_include_directories(FastFEMFiniteCellConsolidationTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellConsolidationTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMFiniteCellConsolidation COMMAND FastFEMFiniteCellConsolidationTest)

    add_executable(FastFEMFiniteCellRealMeshConsolidationTest tests/FiniteCellRealMeshConsolidationTest.cpp)
    target_compile_definitions(FastFEMFiniteCellRealMeshConsolidationTest PRIVATE
        FASTFEM_TET_CORPUS_DIR="${FASTFEM_TET_CORPUS_DIR}"
    )
    target_include_directories(FastFEMFiniteCellRealMeshConsolidationTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellRealMeshConsolidationTest PRIVATE FastFEM::FastFEM ${FASTFEM_TINYOBJLOADER_TARGET})
    add_test(
        NAME FastFEMFiniteCellRealMeshConsolidation
        COMMAND FastFEMFiniteCellRealMeshConsolidationTest --expected-fallbacks 3 --expected-stagnations 3
    )
    set_tests_properties(FastFEMFiniteCellRealMeshConsolidation PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

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
