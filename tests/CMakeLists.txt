add_executable(MeshEditorTests
    RoundtripTest.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/GltfScene.cpp
    ${CMAKE_SOURCE_DIR}/src/armature/Armature.cpp
    ${CMAKE_SOURCE_DIR}/src/render/Bindless.cpp
    ${CMAKE_SOURCE_DIR}/src/render/GpuBufferOps.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/Entity.cpp
    ${CMAKE_SOURCE_DIR}/src/object/ExtrasMesh.cpp
    ${CMAKE_SOURCE_DIR}/src/image/ImageDecode.cpp
    ${CMAKE_SOURCE_DIR}/src/image/ImageEncode.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/Defaults.cpp
    ${CMAKE_SOURCE_DIR}/src/object/ObjectOps.cpp
    ${CMAKE_SOURCE_DIR}/src/selection/Selection.cpp
    ${CMAKE_SOURCE_DIR}/src/selection/SelectionOps.cpp
    ${CMAKE_SOURCE_DIR}/src/Stores.cpp
    ${CMAKE_SOURCE_DIR}/src/render/Textures.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/SceneGraph.cpp
    ${CMAKE_SOURCE_DIR}/src/TransformMath.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/Mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshStore.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/VulkanContext.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/FindMemoryType.cpp
)

add_dependencies(MeshEditorTests generate_gpu_schema)

target_include_directories(MeshEditorTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
    ${Vulkan_INCLUDE_DIRS}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/entt/src
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/fastgltf/deps/simdjson
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/VulkanMemoryAllocator/include
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/basis_universal/transcoder
)

target_link_libraries(MeshEditorTests PRIVATE
    fastgltf::fastgltf
    Boost::ut
    ${Vulkan_LIBRARIES}
    GPUOpen::VulkanMemoryAllocator
    basisu_transcoder
)

target_compile_definitions(MeshEditorTests PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD)
target_compile_options(MeshEditorTests PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base)
