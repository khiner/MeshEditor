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

add_compile_options(-Wno-missing-field-initializers)

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

add_executable(gpu_schema_gen gpu/GenerateGpuSchema.cpp)
target_include_directories(gpu_schema_gen PRIVATE src)

set(GPU_SCHEMA_FILE "${CMAKE_SOURCE_DIR}/gpu/GpuSchema.yaml")
# Regenerate the build graph when schema types change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GPU_SCHEMA_FILE}")

set(GPU_SCHEMA_GENERATED_HEADERS
    "${CMAKE_BINARY_DIR}/gpu/BindlessBindings.h"
)
set(GPU_SCHEMA_GENERATED_MSL
    "${CMAKE_BINARY_DIR}/shaders/BindlessBindings.metal"
)

file(STRINGS "${GPU_SCHEMA_FILE}" GPU_SCHEMA_LINES)
set(GPU_SCHEMA_IN_STRUCTS OFF)
set(GPU_SCHEMA_IN_ENUMS OFF)
foreach(line IN LISTS GPU_SCHEMA_LINES)
    if(line MATCHES "^[ \t]*bindings:[ \t]*$")
        set(GPU_SCHEMA_IN_STRUCTS OFF)
        set(GPU_SCHEMA_IN_ENUMS OFF)
        continue()
    endif()
    if(line MATCHES "^[ \t]*enums:[ \t]*$")
        set(GPU_SCHEMA_IN_STRUCTS OFF)
        set(GPU_SCHEMA_IN_ENUMS ON)
        continue()
    endif()
    if(line MATCHES "^[ \t]*structs:[ \t]*$")
        set(GPU_SCHEMA_IN_STRUCTS ON)
        set(GPU_SCHEMA_IN_ENUMS OFF)
        continue()
    endif()
    if(GPU_SCHEMA_IN_STRUCTS AND line MATCHES "^  - name:[ \t]*([A-Za-z_][A-Za-z0-9_]*)")
        set(GPU_STRUCT_NAME "${CMAKE_MATCH_1}")
        list(APPEND GPU_SCHEMA_GENERATED_HEADERS "${CMAKE_BINARY_DIR}/gpu/${GPU_STRUCT_NAME}.h")
        list(APPEND GPU_SCHEMA_GENERATED_MSL "${CMAKE_BINARY_DIR}/shaders/${GPU_STRUCT_NAME}.metal")
    endif()
    if(GPU_SCHEMA_IN_ENUMS AND line MATCHES "^  - name:[ \t]*([A-Za-z_][A-Za-z0-9_]*)")
        set(GPU_ENUM_NAME "${CMAKE_MATCH_1}")
        list(APPEND GPU_SCHEMA_GENERATED_HEADERS "${CMAKE_BINARY_DIR}/gpu/${GPU_ENUM_NAME}.h")
        list(APPEND GPU_SCHEMA_GENERATED_MSL "${CMAKE_BINARY_DIR}/shaders/${GPU_ENUM_NAME}.metal")
    endif()
endforeach()
list(REMOVE_DUPLICATES GPU_SCHEMA_GENERATED_HEADERS)
list(REMOVE_DUPLICATES GPU_SCHEMA_GENERATED_MSL)

set(GPU_SCHEMA_STAMP "${CMAKE_BINARY_DIR}/gpu/gpu_schema.stamp")
add_custom_command(
    OUTPUT ${GPU_SCHEMA_STAMP}
    BYPRODUCTS
        ${GPU_SCHEMA_GENERATED_HEADERS}
        ${GPU_SCHEMA_GENERATED_MSL}
    COMMAND $<TARGET_FILE:gpu_schema_gen>
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}
        gpu/GpuSchema.yaml
    COMMAND ${CMAKE_COMMAND} -E touch ${GPU_SCHEMA_STAMP}
    DEPENDS gpu_schema_gen ${GPU_SCHEMA_FILE}
    COMMENT "Generating GPU schema outputs"
)
add_custom_target(generate_gpu_schema DEPENDS ${GPU_SCHEMA_STAMP})

# Symlink resources next to the executable.
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")
if(NOT EXISTS ${RESOURCE_DEST})
    file(CREATE_LINK "${CMAKE_SOURCE_DIR}/res" ${RESOURCE_DEST} SYMBOLIC)
endif()

# Put source and generated shaders under one runtime path.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
# Prune links whose source shader was deleted.
file(GLOB SHADER_LINKS "${CMAKE_BINARY_DIR}/shaders/*")
foreach(dst IN LISTS SHADER_LINKS)
    if(IS_SYMLINK "${dst}" AND NOT EXISTS "${dst}")
        file(REMOVE "${dst}")
    endif()
endforeach()
file(GLOB SHADER_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/shaders/*")
foreach(src IN LISTS SHADER_SOURCE_FILES)
    get_filename_component(name "${src}" NAME)
    set(dst "${CMAKE_BINARY_DIR}/shaders/${name}")
    if(NOT EXISTS "${dst}")
        file(CREATE_LINK "${src}" "${dst}" SYMBOLIC)
    endif()
endforeach()

# Enable sustained-contact audio with `SURFACE_AUDIO=1` or the cache variable.
# The default links the modal-impact entry points in src/audio/SurfaceContactAbsent.cpp.
option(MESHEDITOR_SURFACE_AUDIO "Build the surface-contact audio model" $ENV{SURFACE_AUDIO})

option(MESHEDITOR_APP "Build the MeshEditor app" ON)
option(MESHEDITOR_VALIDATE_ACTIONS "Validate replay and round-trip after every committed action" $ENV{VALIDATE_ACTIONS})
option(QUIET "Disable Timer prints" OFF)
set(RESTORE_SESSION_RETAIN 5 CACHE STRING "Max number of scratch sessions to keep")
if(MESHEDITOR_VALIDATE_ACTIONS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "MESHEDITOR_VALIDATE_ACTIONS needs a Debug build")
endif()

include(cmake/Libraries.cmake)

if(MESHEDITOR_APP)
    add_executable(MeshEditor src/main.cpp)
    target_link_libraries(MeshEditor PRIVATE MeshEditorUi)
    mesheditor_compile_policy(MeshEditor COLD)
    target_compile_definitions(MeshEditor PRIVATE
        GLTF_SAMPLE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/external/glTF-Sample-Assets"
        GLTF_PHYSICS_DIR="${CMAKE_SOURCE_DIR}/external/glTF_Physics"
        GLTF_PHYSICAL_AUDIO_DIR="${CMAKE_SOURCE_DIR}/glTF_PhysicalAudio/samples"
        $<$<BOOL:${MESHEDITOR_VALIDATE_ACTIONS}>:VALIDATE_ACTIONS>)
    set_target_properties(MeshEditor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

add_subdirectory(tests)
