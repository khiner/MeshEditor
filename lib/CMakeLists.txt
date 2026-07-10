add_subdirectory(ut)
add_subdirectory(nativefiledialog-extended)
add_subdirectory(VulkanMemoryAllocator SYSTEM)
add_subdirectory(tetgen)
target_compile_options(tetgen PRIVATE -w)
add_subdirectory(lunasvg)
add_subdirectory(tinyobjloader)
add_subdirectory(tinyply)

# fastgltf links this simdjson::simdjson target
add_subdirectory(simdjson)
target_compile_options(simdjson PRIVATE -w)

# libwebp: build only the codec libraries (no tools/extras).
set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
add_subdirectory(libwebp)

set(FASTGLTF_ENABLE_KHR_PHYSICS_RIGID_BODIES ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_KHR_AUDIO_MODAL ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_INSTALL OFF CACHE BOOL "" FORCE) # We provide our own simdjson target.
add_subdirectory(fastgltf)
target_compile_options(fastgltf PRIVATE -w)

# JoltPhysics: disable GPU compute (Hair/SoftBody GPU features not needed for rigid body physics)
set(JPH_USE_DX12 OFF CACHE BOOL "" FORCE)
set(JPH_USE_VK OFF CACHE BOOL "" FORCE)
set(JPH_USE_MTL OFF CACHE BOOL "" FORCE)
set(JPH_USE_CPU_COMPUTE OFF CACHE BOOL "" FORCE)
add_subdirectory(JoltPhysics/Build)
