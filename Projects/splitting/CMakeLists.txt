# The GPU probe options and AMGX/cusparse cache paths are declared by the
# coupling CMakeLists, which is configured first. Enabling the CUDA language and
# locating CUDAToolkit is directory-scoped, so it must be repeated here for the
# CUDA::cudart imported target to exist in this directory.
if(ENABLE_CUDA_STAGE2_PROBE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(OPENVDB_FOUND)
  set(ZIRAN_SPLITTING_SOURCES
    main.cpp
    SplittingSimulationWeakFormFast.cpp
    SplittingSimulationSolidSystemFast.cpp
    SplittingBoundaryParticles.cpp)

  # tbbmalloc_proxy overrides the global allocator; that collides with the
  # AMGX/CUDA runtime's load-time static initializers (which malloc before the
  # proxy is ready) and segfaults. Match coupling: drop the proxy whenever the
  # CUDA/cusparse backend is linked.
  set(ZIRAN_SPLITTING_TBB_MALLOC_LIBS tbbmalloc)
  if(NOT ENABLE_CUSPARSE_SPGEMM)
    list(APPEND ZIRAN_SPLITTING_TBB_MALLOC_LIBS tbbmalloc_proxy)
  endif()

  # The GPU solid probes live in Projects/coupling. When the CUDA build is
  # enabled (ENABLE_CUDA_STAGE2_PROBE, defined by the coupling CMakeLists which
  # is configured first) compile the real .cu kernels; otherwise fall back to
  # the CPU stubs so the host build still links.
  if(ENABLE_CUDA_STAGE2_PROBE)
    add_ziran_executable(splitting ${ZIRAN_SPLITTING_SOURCES}
      ../coupling/GpuSparseGridProbe.cu
      ../coupling/GpuOperatorProbe.cu
      ../coupling/GpuSolidImplicitProbe.cu)
    target_compile_definitions(splitting PRIVATE ENABLE_CUDA_STAGE2_PROBE)
  else()
    add_ziran_executable(splitting ${ZIRAN_SPLITTING_SOURCES}
      ../coupling/GpuSparseGridProbe.cpp
      ../coupling/GpuOperatorProbe.cpp
      ../coupling/GpuSolidImplicitProbe.cpp)
  endif()

  target_compile_options(splitting PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
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
    -fno-math-errno
    ${ZIRAN_CXXFLAGS_LTO}>)
  target_link_libraries(splitting ziran mpm_core ${ZIRAN_SPLITTING_TBB_MALLOC_LIBS} ${ZIRAN_CXXFLAGS_LTO})

  if(ENABLE_CUDA_STAGE2_PROBE)
    target_link_libraries(splitting CUDA::cudart)
    if(ENABLE_AMGX)
      target_compile_definitions(splitting PRIVATE ENABLE_AMGX)
      target_include_directories(splitting PRIVATE ${ZIRAN_AMGX_INCLUDE_DIR})
      target_link_libraries(splitting ${AMGX_LIBRARY})
    endif()
    if(ENABLE_CUSPARSE_SPGEMM AND ZIRAN_CUSPARSE_LIBRARY AND ZIRAN_CUSPARSE_INCLUDE_DIR)
      target_compile_definitions(splitting PRIVATE ZIRAN_ENABLE_CUSPARSE)
      target_include_directories(splitting PRIVATE ${ZIRAN_CUSPARSE_INCLUDE_DIR})
      target_link_libraries(splitting ${ZIRAN_CUSPARSE_LIBRARY})
    endif()
  endif()
endif()
