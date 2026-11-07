find_package(Eigen3 REQUIRED)
add_library(eigen INTERFACE IMPORTED)

find_package(Threads)

# Generate the ConstitutiveModel.h umbrella header by globbing every model header
# in Physics/ConstitutiveModel/. Adding a model header there is picked up
# automatically (CONFIGURE_DEPENDS re-globs on build); no manual include edit.
file(GLOB ZIRAN_CM_HEADERS CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ConstitutiveModel/*.h)
set(ZIRAN_CONSTITUTIVE_MODEL_INCLUDES "")
foreach(_h ${ZIRAN_CM_HEADERS})
  get_filename_component(_name ${_h} NAME)
  # HyperelasticConstitutiveModel.h is included first (foundational) in the .in.
  if(NOT _name STREQUAL "HyperelasticConstitutiveModel.h")
    string(APPEND ZIRAN_CONSTITUTIVE_MODEL_INCLUDES "#include <Ziran/${_h}>\n")
  endif()
endforeach()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ConstitutiveModel/ConstitutiveModel.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h
  @ONLY)

# Every *.cpp in Physics/ConstitutiveModel/ is a constitutive model translation
# unit; glob them so a new model's .cpp is compiled automatically (no edit here).
file(GLOB ZIRAN_CONSTITUTIVE_MODEL_SOURCES CONFIGURE_DEPENDS RELATIVE
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/Physics/ConstitutiveModel/*.cpp)

add_library(ziran STATIC
  CS/DataStructure/Box.cpp
  CS/DataStructure/DataArrayBase.cpp
  CS/Util/Active.cpp
  CS/Util/ErrorContext.cpp
  CS/Util/FloatingPointExceptions.cpp
  CS/Util/Logging.cpp
  CS/Util/Profiler.cpp
  CS/Util/PrettyPrinting.cpp
  CS/Util/RandomNumber.cpp
  CS/Util/SignalHandler.cpp
  CS/Util/Signals.cpp
  CS/Util/StackTrace.cpp
  CS/Util/Timer.cpp
  Math/Geometry/AnalyticLevelSet.cpp
  Math/Geometry/CollisionObject.cpp
  Math/Geometry/SimplexElements.cpp
  Math/Geometry/Grid.cpp
  Math/Geometry/ObjIO.cpp
  Math/Geometry/PolyIO.cpp
  Math/Geometry/SimplexMesh.cpp
  Math/Geometry/Tetrahedron.cpp
  Math/Geometry/Triangle.cpp
  Math/Geometry/Segment.cpp
  Math/Geometry/VdbLevelSet.cpp
  ${ZIRAN_CONSTITUTIVE_MODEL_SOURCES}
  Physics/LagrangianForce/Inertia.cpp
)
target_compile_options(ziran PRIVATE ${ZIRAN_CXXFLAGS_LTO})
target_include_directories(ziran PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# generated ConstitutiveModel.h umbrella (build tree); the source copy is removed
target_include_directories(ziran PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_include_directories(ziran SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_options(ziran PUBLIC "-Wno-unused-but-set-parameter")
target_compile_options(ziran PUBLIC "-Wno-return-type")

if (ENABLE_AMGCL_CUDA)
  target_link_libraries(ziran PUBLIC eigen half amgcl metis Tick tbb ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${ZIRAN_CXXFLAGS_LTO} VexCL::CUDA)
else()
  target_link_libraries(ziran PUBLIC eigen half amgcl metis Tick tbb ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${ZIRAN_CXXFLAGS_LTO})
endif()
if(OPENVDB_FOUND)
  target_link_libraries(ziran PRIVATE openvdb)
endif()

set(ZIRAN_DATA_DIR ${ZIRAN_SOURCE_DIR}/Data)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CS/Util/DataDir.h.in ${CMAKE_CURRENT_SOURCE_DIR}/CS/Util/DataDir.h @ONLY)
