if(OPENVDB_FOUND)
  find_package(Threads)
  set(MPM_INSTANTIATIONS
    instantiations/MpmForceAssembly3d.cpp
    instantiations/MpmParticleRange3d.cpp
    instantiations/MpmSimulationBase3d.cpp)
  set(MPM_FORCE_SOURCES
    Force/MpmPinningForce.cpp
    Force/MpmMultiTargetPinningForce.cpp)
  add_library(mpm_core STATIC ${MPM_INSTANTIATIONS} ${MPM_FORCE_SOURCES})
  target_compile_options(mpm_core PUBLIC "-Wno-unused-but-set-parameter")
  target_compile_options(mpm_core PUBLIC "-Wno-unused-but-set-variable")
  target_compile_options(mpm_core PUBLIC "-Wno-unused-variable")
  target_link_libraries(mpm_core spgrid partio ziran openvdb half tbb metis ${CMAKE_THREAD_LIBS_INIT})

endif()

