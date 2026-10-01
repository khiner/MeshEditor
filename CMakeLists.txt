cmake_minimum_required(VERSION 4.4)

include(lib/FastFEM/cmake/Compiler.cmake)
project(MeshEditor LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "MeshEditor targets macOS on Apple Silicon.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Emit build/compile_commands.json for clangd.
set(CMAKE_CXX_FLAGS "")
set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
add_subdirectory(lib)
set(CMAKE_CXX_FLAGS "-stdlib=libc++")
set(CMAKE_C_FLAGS "")
target_compile_options(meshoptimizer PRIVATE -O3)

# Link full zstd compression and decompression support.
add_library(basisu_transcoder STATIC
    lib/basis_universal/transcoder/basisu_transcoder.cpp
    lib/basis_universal/zstd/zstd.c
)
target_include_directories(basisu_transcoder SYSTEM PUBLIC lib/basis_universal/transcoder)
target_compile_options(basisu_transcoder PRIVATE -O2 -w -fno-strict-aliasing $<$<COMPILE_LANGUAGE:CXX>:-include exception>)

# Suppress warnings from bundled libraries.
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp)
set_source_files_properties(${LIB_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

# Symlink resources next to the executable.
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")
if(NOT EXISTS ${RESOURCE_DEST})
    file(CREATE_LINK "${CMAKE_SOURCE_DIR}/res" ${RESOURCE_DEST} SYMBOLIC)
endif()

# Put editable shader sources, compiled libraries, and shared GPU headers under one runtime path.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
# Prune links whose source shader was deleted.
file(GLOB SHADER_LINKS "${CMAKE_BINARY_DIR}/shaders/*")
foreach(dst IN LISTS SHADER_LINKS)
    if(IS_SYMLINK "${dst}" AND NOT EXISTS "${dst}")
        file(REMOVE "${dst}")
    endif()
endforeach()
file(GLOB SHADER_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/shaders/*")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shader-libraries")
set(METAL_LIBRARIES)
foreach(src IN LISTS SHADER_SOURCE_FILES)
    get_filename_component(name "${src}" NAME)
    set(dst "${CMAKE_BINARY_DIR}/shaders/${name}")
    if(NOT EXISTS "${dst}")
        file(CREATE_LINK "${src}" "${dst}" SYMBOLIC)
    endif()
    if(name MATCHES "\\.metal$")
        get_filename_component(stem "${src}" NAME_WE)
        set(air "${CMAKE_BINARY_DIR}/shader-libraries/${stem}.air")
        set(library "${CMAKE_BINARY_DIR}/shaders/${stem}.metallib")
        add_custom_command(OUTPUT "${air}"
            COMMAND xcrun metal -w -c -std=metal3.2
                -I "${CMAKE_SOURCE_DIR}/src/shaders" -I "${CMAKE_SOURCE_DIR}/src"
                -MMD -MF "${air}.d" "${src}" -o "${air}"
            DEPENDS "${src}"
            DEPFILE "${air}.d"
            VERBATIM)
        add_custom_command(OUTPUT "${library}"
            COMMAND xcrun metallib "${air}" -o "${library}"
            DEPENDS "${air}"
            VERBATIM)
        list(APPEND METAL_LIBRARIES "${library}")
    endif()
endforeach()
add_custom_target(MeshEditorShaderLibraries DEPENDS ${METAL_LIBRARIES})
set(GPU_HEADERS_LINK "${CMAKE_BINARY_DIR}/shaders/gpu")
if(NOT EXISTS ${GPU_HEADERS_LINK})
    file(CREATE_LINK "${CMAKE_SOURCE_DIR}/src/gpu" ${GPU_HEADERS_LINK} SYMBOLIC)
endif()

# Enable sustained-contact audio with `SURFACE_AUDIO=1` or the cache variable.
# The default links the modal-impact entry points in src/audio/SurfaceContactAbsent.cpp.
option(MESHEDITOR_SURFACE_AUDIO "Build the surface-contact audio model" $ENV{SURFACE_AUDIO})

option(MESHEDITOR_APP "Build the MeshEditor app" ON)
option(MESHEDITOR_VALIDATE_ACTIONS "Validate replay and round-trip after every committed action" $ENV{VALIDATE_ACTIONS})
option(QUIET "Disable Timer prints" OFF)
set(RESTORE_SESSION_RETAIN 5 CACHE STRING "Max number of inactive unnamed projects to retain")
if(MESHEDITOR_VALIDATE_ACTIONS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "MESHEDITOR_VALIDATE_ACTIONS needs a Debug build")
endif()

include(cmake/Libraries.cmake)

add_executable(MeshEditorPipelineArchive tools/PipelineArchive.cpp)
mesheditor_compile_policy(MeshEditorPipelineArchive COLD)
target_link_libraries(MeshEditorPipelineArchive PRIVATE MeshEditorRender)
set(PIPELINE_ARCHIVE "${CMAKE_BINARY_DIR}/cache/Pipelines.mtl4a")
set(PIPELINE_ARCHIVE_STAMP "${CMAKE_BINARY_DIR}/cache/Pipelines.stamp")
file(GLOB_RECURSE PIPELINE_SHADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/shaders/*.metal" "${CMAKE_SOURCE_DIR}/src/gpu/*.h")
add_custom_command(OUTPUT "${PIPELINE_ARCHIVE_STAMP}"
    BYPRODUCTS "${PIPELINE_ARCHIVE}"
    COMMAND MeshEditorPipelineArchive "${CMAKE_BINARY_DIR}/shaders" "${PIPELINE_ARCHIVE}" "${PIPELINE_ARCHIVE_STAMP}"
    DEPENDS MeshEditorPipelineArchive ${PIPELINE_SHADER_FILES}
        src/metal/Shader.cpp src/mesh/MeshPipelines.cpp src/render/Pipelines.cpp
    COMMENT "Precompiling Metal pipelines for cold startup"
    VERBATIM)
add_custom_target(MeshEditorPrecompilePipelines DEPENDS "${PIPELINE_ARCHIVE_STAMP}")
add_dependencies(MeshEditorPrecompilePipelines MeshEditorShaderLibraries)

if(MESHEDITOR_APP)
    add_executable(MeshEditor src/main.cpp)
    target_link_libraries(MeshEditor PRIVATE MeshEditorUi)
    add_dependencies(MeshEditor MeshEditorPrecompilePipelines)
    rbp_copy_shaders(MeshEditor)
    mesheditor_compile_policy(MeshEditor COLD)
    target_compile_definitions(MeshEditor PRIVATE
        GLTF_SAMPLE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/external/glTF-Sample-Assets"
        GLTF_PHYSICS_DIR="${CMAKE_SOURCE_DIR}/external/glTF_Physics"
        GLTF_PHYSICAL_AUDIO_DIR="${CMAKE_SOURCE_DIR}/glTF_PhysicalAudio/samples"
        $<$<BOOL:${MESHEDITOR_VALIDATE_ACTIONS}>:VALIDATE_ACTIONS>)
    set_target_properties(MeshEditor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

add_subdirectory(tests)
