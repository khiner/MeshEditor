cmake_minimum_required(VERSION 3.20)

project(MeshEditor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -fexperimental-library")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib -lunwind -lc++abi")

add_compile_options(-Wno-missing-field-initializers)

if(APPLE)
    enable_language(OBJC) # Needed for nativefiledialog
endif()

set(STATIC_FAUST on CACHE BOOL "Build Static Faust library" FORCE)

add_subdirectory(lib)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW3F REQUIRED fftw3f IMPORTED_TARGET)
find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

set(IMGUI_DIR lib/imgui)
set(IMPLOT_DIR lib/implot)

add_executable(${PROJECT_NAME}
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot_demo.cpp
    lib/ImGuizmo/ImGuizmo.cpp
    lib/miniaudio/extras/miniaudio_split/miniaudio.c
    ${SOURCES}
)

set(INCLUDE_EXECUTABLE OFF CACHE STRING "Build the mesh2faust executable" FORCE)
set(Mesh2FaustDir ${CMAKE_CURRENT_SOURCE_DIR}/lib/faust/tools/physicalModeling/mesh2faust)
add_subdirectory(${Mesh2FaustDir})

include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
    SYSTEM lib/ImGuizmo
    SYSTEM lib/imspinner
    SYSTEM lib/glm
    SYSTEM lib/entt/src
    SYSTEM lib/miniaudio/extras/miniaudio_split
    SYSTEM lib/tetgen
    SYSTEM lib/faust/architecture
    SYSTEM PkgConfig::FFTW3F
    SYSTEM ${Mesh2FaustDir}/src
    src
)

set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/res")
set(RESOURCE_DEST "${CMAKE_BINARY_DIR}/res")

# Copy resources after building the project target.
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${RESOURCE_DIR} ${RESOURCE_DEST}
    COMMENT "Copying resources to build directory"
)

if(APPLE)
  target_include_directories(${PROJECT_NAME} PRIVATE /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)
else()
  target_include_directories(${PROJECT_NAME} PRIVATE /usr/include/eigen3)
endif()

find_library(SHADERC_LIB shaderc_combined $ENV{VULKAN_SDK}/lib)
find_library(SPIRV_CROSS_CORE_LIB spirv-cross-core $ENV{VULKAN_SDK}/lib)
message(STATUS "Found shaderc in: ${SHADERC_LIB}")
message(STATUS "Found spirv-cross-core in: ${SPIRV_CROSS_CORE_LIB}")

target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 ${Vulkan_LIBRARIES} GPUOpen::VulkanMemoryAllocator ${SHADERC_LIB} ${SPIRV_CROSS_CORE_LIB} nfd OpenMeshCore faustlib mesh2faust tetgen PkgConfig::FFTW3F)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_BUILD)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_definitions(${PROJECT_NAME} PRIVATE RELEASE_BUILD)
endif()

# Disable lib warnings
target_compile_options(staticlib PRIVATE -w) # Faust (can't use alias target)
target_compile_options(SDL3-shared PRIVATE -w)

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
endif()
