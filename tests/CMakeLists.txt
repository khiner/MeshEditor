# Applies common warnings and definitions to test targets.
function(mesheditor_test_defaults target)
    target_compile_definitions(${target} PRIVATE DEBUG_BUILD)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti)
endfunction()

add_executable(MeshEditorTests
    RoundtripTest.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/GltfScene.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/SourceTexture.cpp
    ${CMAKE_SOURCE_DIR}/src/armature/Armature.cpp
    ${CMAKE_SOURCE_DIR}/src/render/GpuBufferOps.cpp
    ${CMAKE_SOURCE_DIR}/src/render/ClusterLod.cpp
    ${CMAKE_SOURCE_DIR}/src/render/GpuScene.cpp
    ${CMAKE_SOURCE_DIR}/src/render/VertexAdjacencyGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/render/MeshBatch.cpp
    ${CMAKE_SOURCE_DIR}/src/render/MeshConnectivityGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/render/VertexWeldGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/Profile.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/PassTimer.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/Entity.cpp
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
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshBvh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshStore.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/ContactModel.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/ContactDynamics.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/ModalAudio.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/Fft.cpp
    ${SURFACE_SCENE_SOURCES}
    ${CMAKE_SOURCE_DIR}/src/audio/ModalModelFile.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SceneSnapshot.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SaveState.cpp
    ${CMAKE_SOURCE_DIR}/src/snapshot/SnapshotRoles.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MetalContext.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Bindless.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Shader.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/RenderTarget.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/PassChain.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MslSource.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MetalCppImpl.cpp
    ${CMAKE_SOURCE_DIR}/src/File.cpp
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
    ${CMAKE_SOURCE_DIR}/src/selection/SelectionGpu.cpp
    ${CMAKE_SOURCE_DIR}/src/selection/SelectionBitset.cpp
    ${CMAKE_SOURCE_DIR}/src/physics/PhysicsSystem.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Core.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/RotationUi.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Action.cpp
    ${CMAKE_SOURCE_DIR}/src/action/Selection.cpp
)

add_dependencies(MeshEditorTests generate_gpu_schema)

target_include_directories(MeshEditorTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_include_directories(MeshEditorTests SYSTEM PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/lib/entt/src
    ${CMAKE_SOURCE_DIR}/lib/simdjson/include
    ${CMAKE_SOURCE_DIR}/lib/libwebp/src
    ${CMAKE_SOURCE_DIR}/lib/metal-cpp
    ${CMAKE_SOURCE_DIR}/lib/basis_universal/transcoder
    ${CMAKE_SOURCE_DIR}/lib/JoltPhysics
)

target_link_libraries(MeshEditorTests PRIVATE
    fastgltf::fastgltf
    simdjson::simdjson
    Boost::ut
    "-framework Accelerate"
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    basisu_transcoder
    meshoptimizer
    Jolt
    webp
)

mesheditor_test_defaults(MeshEditorTests)
target_compile_definitions(MeshEditorTests PRIVATE MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}" MESHEDITOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(MeshEditorActionSerializeTest ActionSerializeTest.cpp)
add_dependencies(MeshEditorActionSerializeTest generate_gpu_schema)
target_include_directories(MeshEditorActionSerializeTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_include_directories(MeshEditorActionSerializeTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/entt/src)
target_link_libraries(MeshEditorActionSerializeTest PRIVATE Boost::ut)
mesheditor_test_defaults(MeshEditorActionSerializeTest)

add_executable(MeshEditorModalSolveIntegrationTest ModalSolveIntegrationTest.cpp ${CMAKE_SOURCE_DIR}/src/audio/ModalSolve.cpp)
target_include_directories(MeshEditorModalSolveIntegrationTest PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(MeshEditorModalSolveIntegrationTest PRIVATE Boost::ut FastFEM::FastFEM)
mesheditor_test_defaults(MeshEditorModalSolveIntegrationTest)

add_executable(MeshEditorCompressTest CompressTest.cpp ${CMAKE_SOURCE_DIR}/src/Compress.cpp)
target_include_directories(MeshEditorCompressTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(MeshEditorCompressTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/basis_universal/zstd)
target_link_libraries(MeshEditorCompressTest PRIVATE Boost::ut basisu_transcoder)
mesheditor_test_defaults(MeshEditorCompressTest)

add_executable(MeshEditorAudioFileTest AudioFileTest.cpp ${CMAKE_SOURCE_DIR}/src/audio/AudioFile.cpp)
target_include_directories(MeshEditorAudioFileTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)
target_include_directories(MeshEditorAudioFileTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/entt/src)
target_link_libraries(MeshEditorAudioFileTest PRIVATE Boost::ut "-framework AudioToolbox" "-framework CoreFoundation")
mesheditor_test_defaults(MeshEditorAudioFileTest)

add_executable(MeshEditorContactCollectBench ContactCollectBench.cpp)
target_include_directories(MeshEditorContactCollectBench SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/JoltPhysics ${CMAKE_SOURCE_DIR}/lib/entt/src)
target_link_libraries(MeshEditorContactCollectBench PRIVATE Jolt)
mesheditor_test_defaults(MeshEditorContactCollectBench)

# Physics tests link the GPU-backed mesh store required by mesh collider support.
add_executable(MeshEditorContactCollectTest ContactCollectTest.cpp
    ${CMAKE_SOURCE_DIR}/src/physics/PhysicsSystem.cpp
    ${CMAKE_SOURCE_DIR}/src/scene/SceneGraph.cpp
    ${CMAKE_SOURCE_DIR}/src/TransformMath.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/Mesh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshBvh.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/MeshStore.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MetalContext.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Bindless.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/Shader.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/RenderTarget.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MslSource.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MetalCppImpl.cpp
    ${CMAKE_SOURCE_DIR}/src/Profile.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/PassTimer.cpp
)
add_dependencies(MeshEditorContactCollectTest generate_gpu_schema)
target_include_directories(MeshEditorContactCollectTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_include_directories(MeshEditorContactCollectTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/entt/src ${CMAKE_SOURCE_DIR}/lib/metal-cpp ${CMAKE_SOURCE_DIR}/lib/JoltPhysics)
target_link_libraries(MeshEditorContactCollectTest PRIVATE Boost::ut Jolt meshoptimizer "-framework Metal" "-framework Foundation" "-framework QuartzCore")
mesheditor_test_defaults(MeshEditorContactCollectTest)

add_executable(MeshEditorContactManifoldTest ContactManifoldTest.cpp)
target_include_directories(MeshEditorContactManifoldTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/JoltPhysics)
target_link_libraries(MeshEditorContactManifoldTest PRIVATE Boost::ut Jolt)
mesheditor_test_defaults(MeshEditorContactManifoldTest)

add_executable(MeshEditorClusterLodTest ClusterLodTest.cpp ${CMAKE_SOURCE_DIR}/src/render/ClusterLod.cpp)
add_dependencies(MeshEditorClusterLodTest generate_gpu_schema)
target_include_directories(MeshEditorClusterLodTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(MeshEditorClusterLodTest PRIVATE Boost::ut meshoptimizer)
mesheditor_test_defaults(MeshEditorClusterLodTest)

add_executable(MeshEditorMeshBvhTest MeshBvhTest.cpp ${CMAKE_SOURCE_DIR}/src/mesh/MeshBvh.cpp)
add_dependencies(MeshEditorMeshBvhTest generate_gpu_schema)
target_include_directories(MeshEditorMeshBvhTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(MeshEditorMeshBvhTest PRIVATE Boost::ut)
mesheditor_test_defaults(MeshEditorMeshBvhTest)


# Direct audio-render tests link the shared modal sources.
function(mesheditor_render_target target source)
    add_executable(${target} ${source} ${CMAKE_SOURCE_DIR}/src/audio/ModalAudio.cpp ${CMAKE_SOURCE_DIR}/src/audio/ContactModel.cpp ${CMAKE_SOURCE_DIR}/src/audio/Fft.cpp ${SURFACE_MODEL_SOURCES})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
    )
    target_include_directories(${target} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/entt/src)
    target_link_libraries(${target} PRIVATE Boost::ut "-framework Accelerate")
    mesheditor_test_defaults(${target})
endfunction()

mesheditor_render_target(MeshEditorModalRenderTest ModalRenderTest.cpp)

add_executable(MeshEditorContactModelTest ContactModelTest.cpp ${CMAKE_SOURCE_DIR}/src/audio/ContactModel.cpp)
target_include_directories(MeshEditorContactModelTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(MeshEditorContactModelTest PRIVATE Boost::ut)
mesheditor_test_defaults(MeshEditorContactModelTest)

# Builds surface-contact tests only when the model is enabled.
if(MESHEDITOR_SURFACE_AUDIO)
    mesheditor_render_target(MeshEditorSurfaceModelTest surface/SurfaceModelTest.cpp)
    mesheditor_render_target(MeshEditorSurfaceRenderTest surface/SurfaceRenderTest.cpp)
    mesheditor_render_target(MeshEditorSurfaceTrackPoolTest surface/SurfaceTrackPoolTest.cpp)
    mesheditor_render_target(MeshEditorContactSpringsScore surface/ContactSpringsScore.cpp)
endif()

# Compiles every shader source, including generated layout assertions.
add_executable(MeshEditorMslCompileTest
    MslCompileTest.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MslSource.cpp
    ${CMAKE_SOURCE_DIR}/src/metal/MetalCppImpl.cpp
)
add_dependencies(MeshEditorMslCompileTest generate_gpu_schema)
target_include_directories(MeshEditorMslCompileTest PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_include_directories(MeshEditorMslCompileTest SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/metal-cpp)
target_link_libraries(MeshEditorMslCompileTest PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")
set_target_properties(MeshEditorMslCompileTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
mesheditor_test_defaults(MeshEditorMslCompileTest)
