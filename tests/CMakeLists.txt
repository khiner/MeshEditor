add_executable(MeshEditorTests
    RoundtripTest.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/EcsScene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/GltfScene.cpp
    ${CMAKE_SOURCE_DIR}/src/Armature.cpp
    ${CMAKE_SOURCE_DIR}/src/Bindless.cpp
    ${CMAKE_SOURCE_DIR}/src/Entity.cpp
    ${CMAKE_SOURCE_DIR}/src/ExtrasMesh.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageDecode.cpp
    ${CMAKE_SOURCE_DIR}/src/SceneDefaults.cpp
    ${CMAKE_SOURCE_DIR}/src/SceneOps.cpp
    ${CMAKE_SOURCE_DIR}/src/SceneStores.cpp
    ${CMAKE_SOURCE_DIR}/src/SceneTextures.cpp
    ${CMAKE_SOURCE_DIR}/src/SceneTree.cpp
    ${CMAKE_SOURCE_DIR}/src/TransformMath.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/Mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshStore.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/VulkanContext.cpp
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
