cmake_minimum_required(VERSION 3.20)

project(MeshEditor LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "MeshEditor targets macOS on Apple Silicon.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Emit build/compile_commands.json for clangd.
set(CMAKE_CXX_FLAGS "-O2")
set(CMAKE_C_FLAGS "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

add_compile_options(-Wno-missing-field-initializers)

add_subdirectory(lib)

# Full zstd (compress + decompress).
add_library(basisu_transcoder STATIC
    lib/basis_universal/transcoder/basisu_transcoder.cpp
    lib/basis_universal/zstd/zstd.c
)
target_include_directories(basisu_transcoder SYSTEM PUBLIC lib/basis_universal/transcoder)
target_compile_options(basisu_transcoder PRIVATE -w -fno-strict-aliasing $<$<COMPILE_LANGUAGE:CXX>:-include exception>)

# Ignore lib warnings
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp)
set_source_files_properties(${LIB_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW3F REQUIRED fftw3f IMPORTED_TARGET)
# The surface synthesis plans transforms from several threads at once, which the base library does
# not guard, and runs one large transform across them.
find_library(FFTW3F_THREADS fftw3f_threads REQUIRED)
set(EIGEN3_INCLUDE_DIR /opt/homebrew/include/eigen3)

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

include_directories(
    ${ImguiDir}
    ${ImguiDir}/backends
    ${ImplotDir}
    ${CMAKE_BINARY_DIR}
    SYSTEM lib/imspinner
    SYSTEM lib/glm
    SYSTEM lib/entt/src
    SYSTEM lib/spectra/include
    SYSTEM lib/tinyobjloader
    SYSTEM lib/tinyply/source
    SYSTEM lib/lunasvg/plutovg/source
    SYSTEM lib/JoltPhysics
    SYSTEM lib/zpp_bits
    SYSTEM lib/readerwriterqueue
    SYSTEM lib/simdjson/include
    SYSTEM lib/libwebp/src
    SYSTEM lib/basis_universal/zstd
    SYSTEM lib/metal-cpp

    SYSTEM PkgConfig::FFTW3F
    src
)

# Symlink resources next to the executable.
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")
if(NOT EXISTS ${RESOURCE_DEST})
    file(CREATE_LINK "${CMAKE_SOURCE_DIR}/res" ${RESOURCE_DEST} SYMBOLIC)
endif()

# Put source and generated shaders under one runtime path.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")
file(GLOB SHADER_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/shaders/*")
foreach(src IN LISTS SHADER_SOURCE_FILES)
    get_filename_component(name "${src}" NAME)
    set(dst "${CMAKE_BINARY_DIR}/shaders/${name}")
    if(NOT EXISTS "${dst}")
        file(CREATE_LINK "${src}" "${dst}" SYMBOLIC)
    endif()
endforeach()

# The surface-contact audio model (src/audio/surface/): two bodies resting or sliding on one another
# sounding from the roughness of the surfaces they slide over. Selected by `SURFACE_AUDIO=1` in the
# environment at configure time, or by setting the cache variable directly. Without it the build takes
# the core modal path alone - a collision driving the modes with one force pulse - and links the entry
# points in src/audio/SurfaceContactAbsent.cpp. See src/audio/SurfaceContact.h.
option(MESHEDITOR_SURFACE_AUDIO "Build the surface-contact audio model" $ENV{SURFACE_AUDIO})

# The .cpp files whichever way the option went, in three nested sets, so a target links only as far up
# as it reaches: the model off the registry, what reads the scene, and what draws the controls.
if(MESHEDITOR_SURFACE_AUDIO)
    set(SURFACE_MODEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/surface/SurfaceModel.cpp)
    set(SURFACE_SCENE_SOURCES ${SURFACE_MODEL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/surface/SurfaceAudio.cpp)
    set(SURFACE_AUDIO_SOURCES ${SURFACE_SCENE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/surface/SurfaceUi.cpp)
    add_compile_definitions(SURFACE_AUDIO)
else()
    set(SURFACE_MODEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/audio/SurfaceContactAbsent.cpp)
    set(SURFACE_SCENE_SOURCES ${SURFACE_MODEL_SOURCES})
    set(SURFACE_AUDIO_SOURCES ${SURFACE_MODEL_SOURCES})
endif()

option(MESHEDITOR_APP "Build the MeshEditor app" ON)
if(MESHEDITOR_APP)
    # The surface model and its stand-in both live under src/, so take whichever the option selected
    # and leave the other out.
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.mm)
    list(FILTER SOURCES EXCLUDE REGEX "src/audio/(surface/.*|SurfaceContactAbsent)\\.cpp$")
    list(APPEND SOURCES ${SURFACE_AUDIO_SOURCES})

    add_executable(${PROJECT_NAME}
        ${ImguiDir}/imgui_demo.cpp
        ${ImguiDir}/imgui_draw.cpp
        ${ImguiDir}/imgui_tables.cpp
        ${ImguiDir}/imgui_widgets.cpp
        ${ImguiDir}/imgui.cpp
        ${ImguiDir}/backends/imgui_impl_metal.mm
        ${ImguiDir}/backends/imgui_impl_osx.mm
        ${ImplotDir}/implot.cpp
        ${ImplotDir}/implot_items.cpp
        ${ImplotDir}/implot_demo.cpp
        ${SOURCES}
    )
    add_dependencies(${PROJECT_NAME} generate_gpu_schema)
    # The ImGui backends manage their own retains; do not enable ARC.
    set_source_files_properties(${ImguiDir}/backends/imgui_impl_metal.mm ${ImguiDir}/backends/imgui_impl_osx.mm PROPERTIES COMPILE_FLAGS "-w")
    set_source_files_properties(src/FileDialog.mm src/MacPlatform.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_IMPL_METAL_CPP)

    target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate" "-framework AppKit" "-framework AudioToolbox" "-framework AudioUnit" "-framework CoreAudio" "-framework CoreFoundation" "-framework GameController" "-framework Metal" "-framework Foundation" "-framework QuartzCore" "-framework UniformTypeIdentifiers")

    target_link_libraries(${PROJECT_NAME} PRIVATE meshoptimizer PkgConfig::FFTW3F ${FFTW3F_THREADS} lunasvg fastgltf::fastgltf simdjson::simdjson basisu_transcoder Jolt webp)

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
        # Replay tests: File->Replay->Current writes a fixture here when replay diverges from the live scene state.
        target_compile_definitions(${PROJECT_NAME} PRIVATE REPLAY_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/replay")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_BUILD)
    endif()

    option(QUIET "Disable Timer prints" OFF)
    if(QUIET)
        target_compile_definitions(${PROJECT_NAME} PRIVATE QUIET)
    endif()

    # Max number of scratch sessions to retain in the restore/ dir.
    set(RESTORE_SESSION_RETAIN 5 CACHE STRING "Max number of scratch sessions to keep")
    target_compile_definitions(${PROJECT_NAME} PRIVATE RESTORE_SESSION_RETAIN=${RESTORE_SESSION_RETAIN})

    # glTF sample repos populate File submenus with their models. The first two are git submodules
    # under external/, and the physical audio scenes are authored in this repo alongside the extension.
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        GLTF_SAMPLE_ASSETS_DIR="${CMAKE_SOURCE_DIR}/external/glTF-Sample-Assets"
        GLTF_PHYSICS_DIR="${CMAKE_SOURCE_DIR}/external/glTF_Physics"
        GLTF_PHYSICAL_AUDIO_DIR="${CMAKE_SOURCE_DIR}/glTF_PhysicalAudio/samples")

    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-elaborated-enum-base -fno-rtti)
endif()

add_definitions(-DTETLIBRARY)
add_definitions(-DIMGUI_DEFINE_MATH_OPERATORS -DGLM_ENABLE_EXPERIMENTAL -DZPP_BITS_AUTODETECT_MEMBERS_MODE=1)
add_definitions(-DMA_NO_ENGINE) # Reduce MA build size

add_subdirectory(tests)
