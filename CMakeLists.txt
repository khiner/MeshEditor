cmake_minimum_required(VERSION 3.20)

project(MeshEditor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-O2")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib")

add_compile_options(-Wno-missing-field-initializers)

if(APPLE)
    enable_language(OBJC) # Needed for nativefiledialog
endif()

set(STATIC_FAUST on CACHE BOOL "Build Static Faust library" FORCE)

add_subdirectory(lib)

# Ignore lib warnings
target_compile_options(staticlib PRIVATE -w) # Faust (can't use alias target)
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp)
set_source_files_properties(${LIB_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

find_package(SDL3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW3F REQUIRED fftw3f IMPORTED_TARGET)
find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

set(ImguiDir lib/imgui)
set(ImplotDir lib/implot)

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
foreach(line IN LISTS GPU_SCHEMA_LINES)
    if(line MATCHES "^[ \t]*structs:[ \t]*$")
        set(GPU_SCHEMA_IN_STRUCTS ON)
        continue()
    endif()
    if(GPU_SCHEMA_IN_STRUCTS AND line MATCHES "^  - name:[ \t]*([A-Za-z_][A-Za-z0-9_]*)")
        set(GPU_STRUCT_NAME "${CMAKE_MATCH_1}")
        list(APPEND GPU_SCHEMA_GENERATED_HEADERS "${CMAKE_BINARY_DIR}/gpu/${GPU_STRUCT_NAME}.h")
        list(APPEND GPU_SCHEMA_GENERATED_GLSL "${CMAKE_BINARY_DIR}/shaders/${GPU_STRUCT_NAME}.glsl")
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
add_dependencies(${PROJECT_NAME} generate_gpu_schema)

include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${ImguiDir}
    ${ImguiDir}/backends
    ${ImplotDir}
    ${CMAKE_BINARY_DIR}
    SYSTEM lib/imspinner
    SYSTEM lib/glm
    SYSTEM lib/entt/src
    SYSTEM lib/miniaudio/extras/miniaudio_split
    SYSTEM lib/tetgen
    SYSTEM lib/faust/architecture
    # Remaining faust includes for `drawschema.hh`
    SYSTEM lib/faust/compiler
    SYSTEM lib/faust/compiler/errors
    SYSTEM lib/faust/compiler/tlib
    SYSTEM lib/faust/compiler/boxes
    SYSTEM lib/spectra/include
    SYSTEM lib/tinyobjloader
    SYSTEM lib/tinyply/source

    SYSTEM PkgConfig::FFTW3F
    src
)

# Copy resources after building the project target.
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/res")
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${RESOURCE_DIR} ${RESOURCE_DEST}
    COMMENT "Copying resources to build directory"
)

# Delete SVG output directory.
set(SVG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-svg")
add_custom_target(remove_svg_output_dir
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${SVG_OUTPUT_DIR}
    COMMENT "Removing SVG output directory"
)
add_dependencies(${PROJECT_NAME} remove_svg_output_dir)

if(APPLE)
  target_include_directories(${PROJECT_NAME} PRIVATE /opt/homebrew/include/eigen3)
else()
  target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/eigen3)
endif()

find_library(SHADERC_LIB shaderc_combined $ENV{VULKAN_SDK}/lib)
message(STATUS "Found shaderc in: ${SHADERC_LIB}")

if(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 ${Vulkan_LIBRARIES} GPUOpen::VulkanMemoryAllocator ${SHADERC_LIB} nfd faustlib tetgen PkgConfig::FFTW3F lunasvg fastgltf::fastgltf)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_BUILD)
endif()

# Use staging buffers for CPU->GPU transfers instead of direct-mapped memory.
# Enable this for discrete GPUs.
option(MVK_FORCE_STAGED_TRANSFERS "Force staging buffer transfers (for discrete GPUs or testing)" OFF)
if(MVK_FORCE_STAGED_TRANSFERS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MVK_FORCE_STAGED_TRANSFERS)
endif()

# Disable lib warnings
# staticlib is faustlib (can't use alias target)
foreach(target IN LISTS disable_warning_targets)
    target_compile_options(${target} PRIVATE -w)
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-elaborated-enum-base)

add_definitions(-DTETLIBRARY)
add_definitions(-DIMGUI_DEFINE_MATH_OPERATORS -DGLM_ENABLE_EXPERIMENTAL)
add_definitions(-DMA_NO_ENGINE) # Reduce MA build size

# Copy shader files on every release build.
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shaders ${CMAKE_BINARY_DIR}/shaders
    )
    add_dependencies(${PROJECT_NAME} copy_shaders)
    add_dependencies(copy_shaders generate_drawdata_glsl)
endif()
