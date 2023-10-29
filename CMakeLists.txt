cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -stdlib=libc++ -fexperimental-library")

project(mesh2audio LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJC)
endif()

add_subdirectory(lib)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

set(IMGUI_DIR lib/imgui)

add_executable(${PROJECT_NAME}
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${SOURCES}
)
include_directories(
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${Vulkan_INCLUDE_DIRS}
    lib/glm
    src
)
find_library(SHADERC_LIB shaderc_combined $ENV{VULKAN_SDK}/lib)
message(STATUS "Found shaderc in: ${SHADERC_LIB}")
target_link_libraries(${PROJECT_NAME} PRIVATE SDL3::SDL3 ${Vulkan_LIBRARIES} ${SHADERC_LIB})

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-elaborated-enum-base)
