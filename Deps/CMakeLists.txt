find_package(TBB CONFIG QUIET)
if(TBB_FOUND AND TARGET TBB::tbb)
  add_library(tbb INTERFACE IMPORTED GLOBAL)
  set_target_properties(tbb PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbb)
  add_library(tbbmalloc INTERFACE IMPORTED GLOBAL)
  set_target_properties(tbbmalloc PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbbmalloc)
  add_library(tbbmalloc_proxy INTERFACE IMPORTED GLOBAL)
  set_target_properties(tbbmalloc_proxy PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbbmalloc_proxy)
else()
  find_package(TBB MODULE QUIET)
endif()

if(TBB_FOUND AND NOT TARGET tbb)
  add_library(tbb SHARED IMPORTED GLOBAL)
  set_target_properties(tbb PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    IMPORTED_LOCATION ${TBB_LIBRARIES}
    )
  add_library(tbbmalloc SHARED IMPORTED GLOBAL)
  set_target_properties(tbbmalloc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    IMPORTED_LOCATION ${TBB_MALLOC}
    )
  add_library(tbbmalloc_proxy SHARED IMPORTED GLOBAL)
  set_target_properties(tbbmalloc_proxy PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS}
    IMPORTED_LOCATION ${TBB_MALLOC_PROXY}
    )
endif()

find_package(Half QUIET)
if(HALF_FOUND)
  add_library(half SHARED IMPORTED GLOBAL)
  set_target_properties(half PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${HALF_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${HALF_INCLUDE_DIRS}
    IMPORTED_LOCATION ${HALF_LIBRARIES}
    )
else()
  add_library(half INTERFACE)
endif()

find_package(OpenVDB QUIET)
if(OPENVDB_FOUND AND TBB_FOUND)
  add_library(openvdb SHARED IMPORTED GLOBAL)
  set_target_properties(openvdb PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${OPENVDB_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${OPENVDB_INCLUDE_DIRS}
    IMPORTED_LOCATION ${OPENVDB_LIBRARIES}
    INTERFACE_COMPILE_DEFINITIONS "OPENVDB_USE_BLOSC;ZIRAN_WITH_VDB"
    )
endif()


find_package(Git)
# Fetch small third-party source deps at configure time (modern CMake).
include(FetchContent)

# Header-only libraries: download the source but do NOT add_subdirectory. The
# SOURCE_SUBDIR points at a directory with no CMakeLists.txt so that
# FetchContent_MakeAvailable only populates and skips the build of the upstream
# project; we expose the headers through an INTERFACE target ourselves.

FetchContent_Declare(amgcl
  GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
  GIT_TAG 28296c2a95894b3732a89b2f75f257a957b09857
  SOURCE_SUBDIR do-not-add-subdirectory)
FetchContent_MakeAvailable(amgcl)
add_library(amgcl INTERFACE)
target_include_directories(amgcl SYSTEM INTERFACE ${amgcl_SOURCE_DIR})
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  target_compile_options(amgcl INTERFACE ${OpenMP_CXX_FLAGS})
  target_link_libraries(amgcl INTERFACE
          $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
          )
endif ()

if (ENABLE_AMGCL_CUDA)
  FetchContent_Declare(vexcl
          GIT_REPOSITORY https://github.com/ddemidov/vexcl
          GIT_TAG 09b64779f5c22a743549fbc45d106f69fc86de2a)
  FetchContent_MakeAvailable(vexcl)
endif()

# partio: patched (compile fixes for the modern toolchain) and built as a
# subproject. FetchContent_MakeAvailable applies PATCH_COMMAND once and calls
# add_subdirectory on the populated source.
FetchContent_Declare(partio
  GIT_REPOSITORY https://github.com/wdas/partio.git
  GIT_TAG v1.1.0
  PATCH_COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/partio.patch)
FetchContent_MakeAvailable(partio)
set_target_properties(partio PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${partio_SOURCE_DIR}/src/lib"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${partio_SOURCE_DIR}/src/lib"
    INTERFACE_LINK_LIBRARIES "z"
)

FetchContent_Declare(tick
  GIT_REPOSITORY https://github.com/pfultz2/Tick.git
  GIT_TAG b82af5461e95ddc6b652ac2456976cc912ab2f01
  SOURCE_SUBDIR do-not-add-subdirectory)
FetchContent_MakeAvailable(tick)
add_library(Tick INTERFACE)
target_include_directories(Tick SYSTEM INTERFACE ${tick_SOURCE_DIR})

find_package(Metis)
if(METIS_FOUND)
  add_library(metis SHARED IMPORTED GLOBAL)
  if(METIS_INCLUDE_DIRS)
  set_target_properties(metis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${METIS_INCLUDE_DIRS}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${METIS_INCLUDE_DIRS}
    IMPORTED_LOCATION ${METIS_LIBRARIES}
    )
  else()
  set_target_properties(metis PROPERTIES
    IMPORTED_LOCATION ${METIS_LIBRARIES}
    )
  endif()
endif()

# allow visibility in more files
set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
set(HALF_FOUND ${HALF_FOUND} PARENT_SCOPE)
set(OPENVDB_FOUND ${OPENVDB_FOUND} PARENT_SCOPE)
set(METIS_FOUND ${METIS_FOUND} PARENT_SCOPE)
