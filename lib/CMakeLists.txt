cmake_diagnostic(SET CMD_DEPRECATED IGNORE)
set_property(DIRECTORY PROPERTY SYSTEM TRUE)

add_subdirectory(ut)
add_subdirectory(lunasvg)
add_subdirectory(meshoptimizer)
add_subdirectory(tinyobjloader)
add_subdirectory(tinyply)

set(FASTFEM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FASTFEM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(FastFEM)

set(RBP_METAL_CPP_IMPL OFF CACHE BOOL "" FORCE)
set(RBP_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory(RigidBodyPhysics)

# fastgltf links this simdjson::simdjson target
add_subdirectory(simdjson)
target_compile_options(simdjson PRIVATE -w)

# fastgltf links this simdutf::simdutf target, built with only base64 like fastgltf's own download.
set(SIMDUTF_TESTS OFF CACHE BOOL "" FORCE)
set(SIMDUTF_TOOLS OFF CACHE BOOL "" FORCE)
set(SIMDUTF_ICONV OFF CACHE BOOL "" FORCE)
add_subdirectory(simdutf)
target_compile_options(simdutf PRIVATE -w)
target_compile_definitions(simdutf PUBLIC
    SIMDUTF_FEATURE_BASE64=1
    SIMDUTF_FEATURE_DETECT_ENCODING=0
    SIMDUTF_FEATURE_ASCII=0
    SIMDUTF_FEATURE_LATIN1=0
    SIMDUTF_FEATURE_UTF8=0
    SIMDUTF_FEATURE_UTF16=0
    SIMDUTF_FEATURE_UTF32=0)

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
set(FASTGLTF_ENABLE_KHR_AUDIO_RIGID_BODIES ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_INSTALL OFF CACHE BOOL "" FORCE) # We provide our own simdjson target.
add_subdirectory(fastgltf)
target_compile_options(fastgltf PRIVATE -w)
