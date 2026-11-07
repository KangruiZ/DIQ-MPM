option(ENABLE_CUSPARSE_SPGEMM "Use cuSPARSE SpGEMM for GPU sparse products" OFF)
option(ENABLE_AMGX "Enable the AMGX CUDA device solver backend" OFF)
get_filename_component(ZIRAN_DEFAULT_AMGX_ROOT
  "${ZIRAN_SOURCE_DIR}/../AMGX" ABSOLUTE)
set(AMGX_ROOT "${ZIRAN_DEFAULT_AMGX_ROOT}" CACHE PATH
  "AMGX source root (contains include/amgx_c.h)")
set(AMGX_LIBRARY "${AMGX_ROOT}/build-ziran-cuda12/libamgxsh.so" CACHE FILEPATH
  "AMGX shared library (libamgxsh.so)")

if(ENABLE_CUDA_STAGE2_PROBE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_library(ZIRAN_CUSPARSE_LIBRARY NAMES cusparse
    HINTS
      ${CUDAToolkit_LIBRARY_DIR}
      ${CUDAToolkit_ROOT}/lib
      ${CUDAToolkit_ROOT}/lib64
      ${CUDAToolkit_ROOT}/targets/x86_64-linux/lib)
  find_path(ZIRAN_CUSPARSE_INCLUDE_DIR cusparse.h
    HINTS
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CUDAToolkit_ROOT}/include
      ${CUDAToolkit_ROOT}/targets/x86_64-linux/include)

  if(ENABLE_AMGX)
    if(NOT AMGX_ROOT OR NOT AMGX_LIBRARY)
      message(FATAL_ERROR "ENABLE_AMGX requires AMGX_ROOT and AMGX_LIBRARY")
    endif()
    find_path(ZIRAN_AMGX_INCLUDE_DIR amgx_c.h
      HINTS ${AMGX_ROOT}/include)
    if(NOT ZIRAN_AMGX_INCLUDE_DIR OR NOT EXISTS "${AMGX_LIBRARY}")
      message(FATAL_ERROR "Could not find AMGX headers or AMGX_LIBRARY")
    endif()
  endif()
endif()

if(OPENVDB_FOUND)

  set(ZIRAN_COUPLING_COMPILE_OPTIONS
    -fPIC
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wcast-align
    -Wformat=2
    -Winit-self
    -Wmissing-include-dirs
    -Woverloaded-virtual
    -Wredundant-decls
    -march=native
    -fno-math-errno)
  set(ZIRAN_COUPLING_SOURCES
    main.cpp
    ../splitting/SplittingSimulationWeakFormFast.cpp
    ../splitting/SplittingSimulationSolidSystemFast.cpp
    ../splitting/SplittingBoundaryParticles.cpp)
  set(ZIRAN_COUPLING_TBB_MALLOC_LIBS tbbmalloc)
  if(NOT ENABLE_CUSPARSE_SPGEMM)
    list(APPEND ZIRAN_COUPLING_TBB_MALLOC_LIBS tbbmalloc_proxy)
  endif()

  if(ENABLE_CUDA_STAGE2_PROBE)
    add_ziran_executable(coupling ${ZIRAN_COUPLING_SOURCES} GpuSparseGridProbe.cu GpuOperatorProbe.cu GpuSolidImplicitProbe.cu GpuParticleLevelSet.cu)
    target_compile_definitions(coupling PRIVATE ENABLE_CUDA_STAGE2_PROBE)
  else()
    add_ziran_executable(coupling ${ZIRAN_COUPLING_SOURCES} GpuSparseGridProbe.cpp GpuOperatorProbe.cpp GpuSolidImplicitProbe.cpp GpuParticleLevelSet.cpp)
  endif()
  target_compile_options(coupling PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ZIRAN_COUPLING_COMPILE_OPTIONS};${ZIRAN_CXXFLAGS_LTO}>)
  target_link_libraries(coupling ziran mpm_core ${ZIRAN_COUPLING_TBB_MALLOC_LIBS} ${ZIRAN_CXXFLAGS_LTO})
  if(ENABLE_CUDA_STAGE2_PROBE)
    target_link_libraries(coupling CUDA::cudart)
    if(ENABLE_AMGX)
      target_compile_definitions(coupling PRIVATE ENABLE_AMGX)
      target_include_directories(coupling PRIVATE ${ZIRAN_AMGX_INCLUDE_DIR})
      target_link_libraries(coupling ${AMGX_LIBRARY})
    endif()
    if(ENABLE_CUSPARSE_SPGEMM AND ZIRAN_CUSPARSE_LIBRARY AND ZIRAN_CUSPARSE_INCLUDE_DIR)
      target_compile_definitions(coupling PRIVATE ZIRAN_ENABLE_CUSPARSE)
      target_include_directories(coupling PRIVATE ${ZIRAN_CUSPARSE_INCLUDE_DIR})
      target_link_libraries(coupling ${ZIRAN_CUSPARSE_LIBRARY})
    endif()
  endif()

  if(BUILD_TESTING)
    if(ENABLE_CUDA_STAGE2_PROBE)
      # GpuSparseGridProbe's fluid velocity-transfer API consumes the raw
      # DeviceVector view implemented by GpuOperatorProbe.
      add_ziran_executable(stage2_gpu_sparse_grid_probe_test GpuSparseGridProbeTest.cpp GpuSparseGridProbe.cu GpuOperatorProbe.cu)
      target_compile_definitions(stage2_gpu_sparse_grid_probe_test PRIVATE ENABLE_CUDA_STAGE2_PROBE)
    else()
      add_ziran_executable(stage2_gpu_sparse_grid_probe_test GpuSparseGridProbeTest.cpp GpuSparseGridProbe.cpp GpuOperatorProbe.cpp)
    endif()
    target_compile_options(stage2_gpu_sparse_grid_probe_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ZIRAN_COUPLING_COMPILE_OPTIONS};${ZIRAN_CXXFLAGS_LTO}>)
    target_link_libraries(stage2_gpu_sparse_grid_probe_test ziran mpm_core ${ZIRAN_COUPLING_TBB_MALLOC_LIBS} ${ZIRAN_CXXFLAGS_LTO})
    if(ENABLE_CUDA_STAGE2_PROBE)
      target_link_libraries(stage2_gpu_sparse_grid_probe_test CUDA::cudart)
    endif()
    add_test(NAME stage2_gpu_sparse_grid_probe_test COMMAND stage2_gpu_sparse_grid_probe_test)

    if(ENABLE_CUDA_STAGE2_PROBE)
      add_ziran_executable(stage3_gpu_operator_probe_test GpuOperatorProbeTest.cpp GpuOperatorProbe.cu)
      target_compile_definitions(stage3_gpu_operator_probe_test PRIVATE ENABLE_CUDA_STAGE2_PROBE)
    else()
      add_ziran_executable(stage3_gpu_operator_probe_test GpuOperatorProbeTest.cpp GpuOperatorProbe.cpp)
    endif()
    target_compile_options(stage3_gpu_operator_probe_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ZIRAN_COUPLING_COMPILE_OPTIONS};${ZIRAN_CXXFLAGS_LTO}>)
    target_include_directories(stage3_gpu_operator_probe_test PRIVATE ${EIGEN3_INCLUDE_DIR})
    if(ENABLE_CUDA_STAGE2_PROBE)
      target_link_libraries(stage3_gpu_operator_probe_test CUDA::cudart)
      if(ENABLE_CUSPARSE_SPGEMM AND ZIRAN_CUSPARSE_LIBRARY AND ZIRAN_CUSPARSE_INCLUDE_DIR)
        target_compile_definitions(stage3_gpu_operator_probe_test PRIVATE ZIRAN_ENABLE_CUSPARSE)
        target_include_directories(stage3_gpu_operator_probe_test PRIVATE ${ZIRAN_CUSPARSE_INCLUDE_DIR})
        target_link_libraries(stage3_gpu_operator_probe_test ${ZIRAN_CUSPARSE_LIBRARY})
      endif()
    else()
      target_link_libraries(stage3_gpu_operator_probe_test ${ZIRAN_CXXFLAGS_LTO})
    endif()
    add_test(NAME stage3_gpu_operator_probe_test COMMAND stage3_gpu_operator_probe_test)

    if(ENABLE_CUDA_STAGE2_PROBE)
      add_ziran_executable(solid_implicit_gpu_operator_test GpuSolidImplicitProbeTest.cpp GpuSolidImplicitProbe.cu GpuOperatorProbe.cu)
      target_compile_definitions(solid_implicit_gpu_operator_test PRIVATE ENABLE_CUDA_STAGE2_PROBE)
      target_link_libraries(solid_implicit_gpu_operator_test CUDA::cudart ziran ${ZIRAN_CXXFLAGS_LTO})
    else()
      add_ziran_executable(solid_implicit_gpu_operator_test GpuSolidImplicitProbeTest.cpp GpuSolidImplicitProbe.cpp GpuOperatorProbe.cpp)
      target_link_libraries(solid_implicit_gpu_operator_test ziran ${ZIRAN_CXXFLAGS_LTO})
    endif()
    target_compile_options(solid_implicit_gpu_operator_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${ZIRAN_COUPLING_COMPILE_OPTIONS};${ZIRAN_CXXFLAGS_LTO}>)
    # The Corotated A/B fixture directly instantiates the production
    # constitutive model to compare its projected dP/dF with the CUDA path.
    target_include_directories(solid_implicit_gpu_operator_test PRIVATE
      ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/Lib)
    add_test(NAME solid_implicit_gpu_operator_test COMMAND solid_implicit_gpu_operator_test)

    if(ENABLE_CUDA_STAGE2_PROBE AND ZIRAN_CUSPARSE_LIBRARY AND ZIRAN_CUSPARSE_INCLUDE_DIR)
      add_executable(cusparse_spgemm_smoke_test CusparseSpGemmSmoke.cu)
      target_include_directories(cusparse_spgemm_smoke_test PRIVATE ${ZIRAN_CUSPARSE_INCLUDE_DIR})
      target_link_libraries(cusparse_spgemm_smoke_test CUDA::cudart ${ZIRAN_CUSPARSE_LIBRARY})
      add_test(NAME cusparse_spgemm_smoke_test COMMAND cusparse_spgemm_smoke_test)
    endif()
  endif() # BUILD_TESTING
  
endif()
