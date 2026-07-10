cmake_minimum_required(VERSION 3.20)

project(MeshEditor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_C_FLAGS "-O2")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

add_compile_options(-Wno-missing-field-initializers)

if(APPLE)
    enable_language(OBJC) # Needed for nativefiledialog
endif()

add_subdirectory(lib)

add_library(basisu_transcoder STATIC
    lib/basis_universal/transcoder/basisu_transcoder.cpp
    lib/basis_universal/zstd/zstddeclib.c
)
target_include_directories(basisu_transcoder SYSTEM PUBLIC lib/basis_universal/transcoder)
target_compile_options(basisu_transcoder PRIVATE -w -fno-strict-aliasing $<$<COMPILE_LANGUAGE:CXX>:-include exception>)

# Ignore lib warnings
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp)
set_source_files_properties(${LIB_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW3F REQUIRED fftw3f IMPORTED_TARGET)
find_package(Vulkan REQUIRED)

set(ImguiDir lib/imgui)
set(ImplotDir lib/implot)

add_executable(gpu_schema_gen gpu/GenerateGpuSchema.cpp)
target_include_directories(gpu_schema_gen PRIVATE src)

set(GPU_SCHEMA_FILE "${CMAKE_SOURCE_DIR}/gpu/GpuSchema.yaml")
# Regenerate build graph when schema shape changes (e.g. added/removed struct names).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GPU_SCHEMA_FILE}")

set(GPU_SCHEMA_GENERATED_HEADERS
    "${CMAKE_BINARY_DIR}/gpu/BindlessBindings.h"
)
set(GPU_SCHEMA_GENERATED_GLSL
    "${CMAKE_BINARY_DIR}/shaders/BindlessBindings.glsl"
)

# Mirror generator naming: each top-level struct emits <Name>.h and <Name>.glsl.
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
        list(APPEND GPU_SCHEMA_GENERATED_GLSL "${CMAKE_BINARY_DIR}/shaders/${GPU_STRUCT_NAME}.glsl")
    endif()
    if(GPU_SCHEMA_IN_ENUMS AND line MATCHES "^  - name:[ \t]*([A-Za-z_][A-Za-z0-9_]*)")
        set(GPU_ENUM_NAME "${CMAKE_MATCH_1}")
        list(APPEND GPU_SCHEMA_GENERATED_HEADERS "${CMAKE_BINARY_DIR}/gpu/${GPU_ENUM_NAME}.h")
        list(APPEND GPU_SCHEMA_GENERATED_GLSL "${CMAKE_BINARY_DIR}/shaders/${GPU_ENUM_NAME}.glsl")
    endif()
endforeach()
list(REMOVE_DUPLICATES GPU_SCHEMA_GENERATED_HEADERS)
list(REMOVE_DUPLICATES GPU_SCHEMA_GENERATED_GLSL)

set(GPU_SCHEMA_STAMP "${CMAKE_BINARY_DIR}/gpu/gpu_schema.stamp")
add_custom_command(
    OUTPUT ${GPU_SCHEMA_STAMP}
    BYPRODUCTS
        ${GPU_SCHEMA_GENERATED_HEADERS}
        ${GPU_SCHEMA_GENERATED_GLSL}
    COMMAND $<TARGET_FILE:gpu_schema_gen>
        ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}
        gpu/GpuSchema.yaml
    COMMAND ${CMAKE_COMMAND} -E touch ${GPU_SCHEMA_STAMP}
    DEPENDS gpu_schema_gen ${GPU_SCHEMA_FILE}
    COMMENT "Generating GPU schema outputs"
)
add_custom_target(generate_gpu_schema DEPENDS ${GPU_SCHEMA_STAMP})

include_directories(
    ${ImguiDir}
    ${ImguiDir}/backends
    ${ImplotDir}
    ${CMAKE_BINARY_DIR}
    SYSTEM lib/imspinner
    SYSTEM lib/glm
    SYSTEM lib/entt/src
    SYSTEM lib/miniaudio/extras/miniaudio_split
    SYSTEM lib/tetgen
    SYSTEM lib/spectra/include
    SYSTEM lib/tinyobjloader
    SYSTEM lib/tinyply/source
    SYSTEM lib/lunasvg/plutovg/source
    SYSTEM lib/JoltPhysics
    SYSTEM lib/zpp_bits
    SYSTEM lib/readerwriterqueue
    SYSTEM lib/simdjson/include
    SYSTEM lib/libwebp/src

    SYSTEM PkgConfig::FFTW3F
    src
    ${Vulkan_INCLUDE_DIRS} # Last: may shadow vendored headers
)

# Symlink resources next to the executable.
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")
if(NOT EXISTS ${RESOURCE_DEST})
    file(CREATE_LINK "${CMAKE_SOURCE_DIR}/res" ${RESOURCE_DEST} SYMBOLIC)
endif()

# Symlink each source shader into build/shaders. Generated GLSL (from the GPU schema step)
# is also written here, giving a single runtime shader path. Add new files → reconfigure.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
file(GLOB SHADER_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/shaders/*")
foreach(src IN LISTS SHADER_SOURCE_FILES)
    get_filename_component(name "${src}" NAME)
    set(dst "${CMAKE_BINARY_DIR}/shaders/${name}")
    if(NOT EXISTS "${dst}")
        file(CREATE_LINK "${src}" "${dst}" SYMBOLIC)
    endif()
endforeach()

find_library(SHADERC_LIB shaderc_combined $ENV{VULKAN_SDK}/lib)
message(STATUS "Found shaderc in: ${SHADERC_LIB}")

option(MESHEDITOR_APP "Build the MeshEditor app" ON)
if(MESHEDITOR_APP)
    # Vendored SDL3 from main for pixel-precise macOS scroll deltas.
    # Drop and restore `find_package(SDL3 REQUIRED)` once a release ships past this commit.
    include(FetchContent)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_DISABLE_INSTALL_DOCS ON CACHE BOOL "" FORCE)
    # SDL provides windowing, events, and gamepad input.
    set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    set(SDL_TRAY OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL3
        URL https://github.com/libsdl-org/SDL/archive/85b36937a985c12dee7edc4dd6e1efcfaa508c99.tar.gz
        URL_HASH SHA256=e7663dad48549c7fa2571f65f30c478d49879e14c23569cfc10ddd8644dc6982
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(SDL3)

    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

    add_executable(${PROJECT_NAME}
        ${ImguiDir}/imgui_demo.cpp
        ${ImguiDir}/imgui_draw.cpp
        ${ImguiDir}/imgui_tables.cpp
        ${ImguiDir}/imgui_widgets.cpp
        ${ImguiDir}/imgui.cpp
        ${ImguiDir}/backends/imgui_impl_vulkan.cpp
        ${ImguiDir}/backends/imgui_impl_sdl3.cpp
        ${ImplotDir}/implot.cpp
        ${ImplotDir}/implot_items.cpp
        ${ImplotDir}/implot_demo.cpp
        lib/miniaudio/extras/miniaudio_split/miniaudio.c
        ${SOURCES}
    )
    add_dependencies(${PROJECT_NAME} generate_gpu_schema)

    if(APPLE)
        target_include_directories(${PROJECT_NAME} PRIVATE /opt/homebrew/include/eigen3)
        target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate")
    else()
        target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/eigen3)
    endif()

    target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 ${Vulkan_LIBRARIES} GPUOpen::VulkanMemoryAllocator ${SHADERC_LIB} nfd tetgen PkgConfig::FFTW3F lunasvg fastgltf::fastgltf simdjson::simdjson basisu_transcoder Jolt webp)

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
        # Replay tests: File->Replay->Current writes a fixture here when replay diverges from the live scene state.
        target_compile_definitions(${PROJECT_NAME} PRIVATE REPLAY_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/replay")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_BUILD)
    endif()

    # Use staging buffers for CPU->GPU transfers instead of direct-mapped memory.
    # Enable this for discrete GPUs.
    option(MVK_FORCE_STAGED_TRANSFERS "Force staging buffer transfers (for discrete GPUs or testing)" OFF)
    if(MVK_FORCE_STAGED_TRANSFERS)
        target_compile_definitions(${PROJECT_NAME} PRIVATE MVK_FORCE_STAGED_TRANSFERS)
    endif()

    option(QUIET "Disable Timer prints" OFF)
    if(QUIET)
        target_compile_definitions(${PROJECT_NAME} PRIVATE QUIET)
    endif()

    # Max number of .actions replay logs to retain in the replay/ dir.
    set(REPLAY_LOG_RETAIN 5 CACHE STRING "Max number of .actions replay logs to keep")
    target_compile_definitions(${PROJECT_NAME} PRIVATE REPLAY_LOG_RETAIN=${REPLAY_LOG_RETAIN})

    # glTF sample repos (git submodules under external/) populate File submenus with their models.
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        GLTF_SAMPLE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/external/glTF-Sample-Assets"
        GLTF_PHYSICS_DIR="${CMAKE_SOURCE_DIR}/external/glTF_Physics")

    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-elaborated-enum-base -fno-rtti)
endif()

add_definitions(-DTETLIBRARY)
add_definitions(-DIMGUI_DEFINE_MATH_OPERATORS -DGLM_ENABLE_EXPERIMENTAL -DZPP_BITS_AUTODETECT_MEMBERS_MODE=1)
add_definitions(-DMA_NO_ENGINE) # Reduce MA build size

add_subdirectory(tests)
