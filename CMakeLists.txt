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
    src/audio/Tet10Cholesky.cpp
    src/audio/finite_cell/AccelerateShiftInvert.cpp
    src/audio/finite_cell/EigenpairCertification.cpp
    src/audio/FiniteCell.cpp
    src/audio/FiniteCellEigensolver.cpp
    src/audio/finite_cell/MetalOperations.mm
    src/audio/MassProperties.cpp
    src/audio/AccelerateSparseCholesky.cpp
    src/audio/surface2modes.cpp
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
                -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/audio/finite_cell/MetalOperations.mm
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
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/src/audio/finite_cell/MetalOperations.mm
            -DOUTPUT=${FASTFEM_FINITE_CELL_METAL_SOURCE}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExtractMetalSource.cmake"
        COMMAND xcrun --sdk macosx metal -c "${FASTFEM_FINITE_CELL_METAL_SOURCE}" -o "${FASTFEM_FINITE_CELL_AIR}"
        COMMAND xcrun --sdk macosx metallib "${FASTFEM_FINITE_CELL_AIR}" -o "${FASTFEM_FINITE_CELL_METALLIB}"
        ${FASTFEM_FINITE_CELL_ARCHIVE_COMMANDS}
        COMMAND "${FASTFEM_XXD}" -i -n FastFEMFiniteCellMetallib
            "${FASTFEM_FINITE_CELL_METALLIB}" "${FASTFEM_FINITE_CELL_METALLIB_HEADER}"
        DEPENDS
            src/audio/finite_cell/MetalOperations.mm
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
    PRIVATE
        ${FASTFEM_MESHOPTIMIZER_TARGET}
        ${FASTFEM_ACCELERATE_FRAMEWORK}
        ${FASTFEM_FOUNDATION_FRAMEWORK}
        ${FASTFEM_METAL_FRAMEWORK}
)
set_source_files_properties(src/audio/finite_cell/MetalOperations.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)

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

    add_executable(FastFEMTet10CholeskyBenchmark tools/Tet10CholeskyBenchmark.cpp)
    target_link_libraries(FastFEMTet10CholeskyBenchmark PRIVATE FastFEM::FastFEM ${FASTFEM_TINYOBJLOADER_TARGET})
    target_include_directories(FastFEMTet10CholeskyBenchmark PRIVATE tests)

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

    add_executable(FastFEMSurfaceModesTest tests/SurfaceModesTest.cpp)
    target_include_directories(FastFEMSurfaceModesTest PRIVATE tests tools)
    target_link_libraries(FastFEMSurfaceModesTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMSurfaceModes COMMAND FastFEMSurfaceModesTest)

    add_executable(FastFEMFiniteCellTest tests/FiniteCellTest.cpp)
    target_include_directories(FastFEMFiniteCellTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMFiniteCell COMMAND FastFEMFiniteCellTest)

    add_executable(FastFEMAnalyticalModalTest tests/AnalyticalModalTest.cpp)
    target_include_directories(FastFEMAnalyticalModalTest PRIVATE tests tools)
    target_link_libraries(FastFEMAnalyticalModalTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMAnalyticalModal COMMAND FastFEMAnalyticalModalTest)

    add_executable(FastFEMFiniteCellRobustnessTest tests/FiniteCellRobustnessTest.cpp)
    target_include_directories(FastFEMFiniteCellRobustnessTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellRobustnessTest PRIVATE FastFEM::FastFEM Boost::ut)
    add_test(NAME FastFEMFiniteCellRobustness COMMAND FastFEMFiniteCellRobustnessTest)

    add_executable(FastFEMFiniteCellRealMeshTest tests/FiniteCellRealMeshTest.cpp)
    target_compile_definitions(FastFEMFiniteCellRealMeshTest PRIVATE
        FASTFEM_TET_CORPUS_DIR="${FASTFEM_TET_CORPUS_DIR}"
    )
    target_include_directories(FastFEMFiniteCellRealMeshTest PRIVATE tests tools)
    target_link_libraries(FastFEMFiniteCellRealMeshTest PRIVATE FastFEM::FastFEM ${FASTFEM_TINYOBJLOADER_TARGET})
    add_test(
        NAME FastFEMFiniteCellRealMesh
        COMMAND FastFEMFiniteCellRealMeshTest --expected-fallbacks 3 --expected-stagnations 3
    )
    set_tests_properties(FastFEMFiniteCellRealMesh PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)

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
