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
    ${CMAKE_SOURCE_DIR}/src/audio/ContactModel.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/ContactDynamics.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SceneSnapshot.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SaveState.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SnapshotRoles.cpp
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
    ${CMAKE_SOURCE_DIR}/src/viewport/ViewCameraOps.cpp
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

add_dependencies(MeshEditorTests generate_gpu_schema)

target_include_directories(MeshEditorTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/entt/src
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/simdjson/include
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/libwebp/src
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/VulkanMemoryAllocator/include
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/basis_universal/transcoder
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/JoltPhysics
    ${Vulkan_INCLUDE_DIRS} # Last: may shadow vendored headers
)

target_link_libraries(MeshEditorTests PRIVATE
    fastgltf::fastgltf
    simdjson::simdjson
    Boost::ut
    ${Vulkan_LIBRARIES}
    GPUOpen::VulkanMemoryAllocator
    basisu_transcoder
    ${SHADERC_LIB}
    Jolt
    webp
)

target_compile_definitions(MeshEditorTests PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}" MESHEDITOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(MeshEditorTests PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)

# Standalone: serializes every Action alternative through zpp::bits. Header-only (no GPU/registry),
# so it needs only the action headers (and the generated GPU schema headers they pull in).
add_executable(MeshEditorActionSerializeTest ActionSerializeTest.cpp)
add_dependencies(MeshEditorActionSerializeTest generate_gpu_schema)
target_include_directories(MeshEditorActionSerializeTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/entt/src
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(MeshEditorActionSerializeTest PRIVATE Boost::ut)
target_compile_definitions(MeshEditorActionSerializeTest PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD)
target_compile_options(MeshEditorActionSerializeTest PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)

# Default RealImpact dataset location for the modal solver test and bench: the corpus submodule.
# Overridable here (cmake -D) or at runtime with the REALIMPACT_DATASET_DIR env var.
set(REALIMPACT_DATASET_DIR "${CMAKE_SOURCE_DIR}/external/RealImpact/dataset" CACHE PATH "RealImpact dataset directory")

# Standalone: the modal FEM solver reads tet geometry directly, so it needs only mesh2modes.cpp and
# Tets.cpp with tetgen, meshoptimizer, Eigen, and Spectra.
add_executable(MeshEditorModalSolverTest ModalSolverTest.cpp ${CMAKE_SOURCE_DIR}/src/audio/mesh2modes.cpp ${CMAKE_SOURCE_DIR}/src/mesh/Tets.cpp)
target_include_directories(MeshEditorModalSolverTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/spectra/include
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/tetgen
)
target_link_libraries(MeshEditorModalSolverTest PRIVATE Boost::ut tetgen meshoptimizer "-framework Accelerate")
target_compile_definitions(MeshEditorModalSolverTest PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD TETLIBRARY REALIMPACT_DATASET_DIR="${REALIMPACT_DATASET_DIR}")
target_compile_options(MeshEditorModalSolverTest PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)

# Standalone: times each stage of the modal solve pipeline across RealImpact samples.
# RealImpact.cpp provides the object-name-to-material mapping.
add_executable(MeshEditorModalSolverBench ModalSolverBench.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/mesh2modes.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/RealImpact.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/Tets.cpp
)
target_include_directories(MeshEditorModalSolverBench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/spectra/include
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/tetgen
)
target_link_libraries(MeshEditorModalSolverBench PRIVATE tetgen meshoptimizer "-framework Accelerate")
target_compile_definitions(MeshEditorModalSolverBench PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD TETLIBRARY REALIMPACT_DATASET_DIR="${REALIMPACT_DATASET_DIR}")
target_compile_options(MeshEditorModalSolverBench PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)

# Standalone: the Hertz contact-time math is pure (no GPU/registry/mesh), so it needs only ContactModel.cpp.
add_executable(MeshEditorContactModelTest ContactModelTest.cpp ${CMAKE_SOURCE_DIR}/src/audio/ContactModel.cpp)
target_include_directories(MeshEditorContactModelTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
)
target_link_libraries(MeshEditorContactModelTest PRIVATE Boost::ut)
target_compile_definitions(MeshEditorContactModelTest PRIVATE GLM_ENABLE_EXPERIMENTAL DEBUG_BUILD)
target_compile_options(MeshEditorContactModelTest PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)
