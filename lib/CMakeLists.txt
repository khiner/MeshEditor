add_subdirectory(ut)
add_subdirectory(nativefiledialog-extended)
add_subdirectory(VulkanMemoryAllocator)
add_subdirectory(tetgen)
add_subdirectory(lunasvg)
add_subdirectory(tinyobjloader)
add_subdirectory(tinyply)

set(FASTGLTF_ENABLE_KHR_PHYSICS_RIGID_BODIES ON CACHE BOOL "" FORCE)
add_subdirectory(fastgltf)

# JoltPhysics: disable GPU compute (Hair/SoftBody GPU features not needed for rigid body physics)
set(JPH_USE_DX12 OFF CACHE BOOL "" FORCE)
set(JPH_USE_VK OFF CACHE BOOL "" FORCE)
set(JPH_USE_MTL OFF CACHE BOOL "" FORCE)
set(JPH_USE_CPU_COMPUTE OFF CACHE BOOL "" FORCE)
add_subdirectory(JoltPhysics/Build)

#=================== faust ===================
# See [the docs](https://faustdoc.grame.fr/manual/embedding/#using-libfaust-with-the-llvm-backend) for general help.

set(INCLUDE_EXECUTABLE off CACHE BOOL "Include runtime executable" FORCE)
set(INCLUDE_OSC off CACHE BOOL "Include Faust OSC library" FORCE)
set(INCLUDE_HTTP off CACHE BOOL "Include Faust HTTPD library" FORCE)
set(INCLUDE_WASM_GLUE off CACHE BOOL "Include wasm glue targets" FORCE)
set(INCLUDE_EMCC off CACHE BOOL "Include emcc targets" FORCE)
set(INCLUDE_STATIC on CACHE BOOL "Include static Faust library" FORCE)
set(LLVM_BACKEND COMPILER STATIC DYNAMIC CACHE STRING "Include LLVM backend" FORCE)

add_subdirectory(faust/build EXCLUDE_FROM_ALL)

set(FAUST_TARGET_LIB "staticlib")
target_compile_definitions(${FAUST_TARGET_LIB} PUBLIC LLVM_BUILD_UNIVERSAL=1)
add_library(faustlib ALIAS ${FAUST_TARGET_LIB})
