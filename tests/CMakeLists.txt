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
    ${CMAKE_SOURCE_DIR}/src/snapshot/SceneSnapshot.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SaveState.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/SceneSnapshotRoles.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/VulkanContext.cpp
    ${CMAKE_SOURCE_DIR}/src/vulkan/FindMemoryType.cpp
    ${CMAKE_SOURCE_DIR}/src/Paths.cpp
    ${CMAKE_SOURCE_DIR}/src/ProcessEvents.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/Viewport.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/ViewportRenderGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/ViewportOps.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/RenderExtent.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/ViewCamera.cpp
    ${CMAKE_SOURCE_DIR}/src/viewport/ViewportDisplay.cpp
    ${CMAKE_SOURCE_DIR}/src/render/Pipelines.cpp
    ${CMAKE_SOURCE_DIR}/src/render/Shader.cpp
    ${CMAKE_SOURCE_DIR}/src/selection/SelectionGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/selection/SelectionBitset.cpp
    ${CMAKE_SOURCE_DIR}/src/physics/PhysicsSystem.cpp
    ${CMAKE_SOURCE_DIR}/src/physics/PhysicsDebugDraw.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Core.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/RotationUi.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Action.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Selection.cpp
)

# PhysicsSystem.cpp subclasses Jolt GroupFilter; Jolt is built with -fno-rtti so this file must match.
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/physics/PhysicsSystem.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

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
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/JoltPhysics
)

target_link_libraries(MeshEditorTests PRIVATE
    fastgltf::fastgltf
    Boost::ut
    ${Vulkan_LIBRARIES}
    GPUOpen::VulkanMemoryAllocator
    basisu_transcoder
    ${SHADERC_LIB}
    Jolt
)

target_compile_definitions(MeshEditorTests PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}")
target_compile_options(MeshEditorTests PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base)

# Standalone: serializes every Action alternative through zpp::bits. Header-only (no GPU/registry),
# so it needs only the action headers (and the generated GPU schema headers they pull in).
add_executable(MeshEditorActionSerializeTest ActionSerializeTest.cpp)
add_dependencies(MeshEditorActionSerializeTest generate_gpu_schema)
target_include_directories(MeshEditorActionSerializeTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
    ${Vulkan_INCLUDE_DIRS}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/entt/src
)
target_link_libraries(MeshEditorActionSerializeTest PRIVATE Boost::ut)
target_compile_definitions(MeshEditorActionSerializeTest PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD)
target_compile_options(MeshEditorActionSerializeTest PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base)
