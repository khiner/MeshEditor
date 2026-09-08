# Each production source has one owner, shared by the app and tests.
add_library(mesheditor_headers INTERFACE)
target_include_directories(mesheditor_headers INTERFACE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_BINARY_DIR}")
target_include_directories(mesheditor_headers SYSTEM INTERFACE
    "${CMAKE_SOURCE_DIR}/lib/entt/src"
    "${CMAKE_SOURCE_DIR}/lib/FastFEM/include"
)
target_compile_definitions(mesheditor_headers INTERFACE
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release>:RELEASE_BUILD>
    $<$<BOOL:${MESHEDITOR_SURFACE_AUDIO}>:SURFACE_AUDIO>
    $<$<BOOL:${QUIET}>:QUIET>)

add_library(mesheditor_serialization INTERFACE)
target_include_directories(mesheditor_serialization SYSTEM INTERFACE "${CMAKE_SOURCE_DIR}/lib/zpp_bits")
target_compile_definitions(mesheditor_serialization INTERFACE ZPP_BITS_AUTODETECT_MEMBERS_MODE=1)

function(mesheditor_compile_policy target policy)
    if(policy STREQUAL "HOT")
        set(debug_optimization -O2)
    else()
        set(debug_optimization -O0)
    endif()
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base -fno-rtti
        "$<$<CONFIG:Debug>:${debug_optimization}>"
        "$<$<NOT:$<CONFIG:Debug>>:-O2>")
    set_target_properties(${target} PROPERTIES CXX_STANDARD 23 OBJCXX_STANDARD 23)
    target_link_libraries(${target} PUBLIC mesheditor_headers)
    add_dependencies(${target} generate_gpu_schema)
endfunction()

function(mesheditor_library target policy)
    add_library(${target} STATIC ${ARGN})
    mesheditor_compile_policy(${target} ${policy})
endfunction()

mesheditor_library(mesheditor_support COLD src/File.cpp src/Paths.cpp src/Compress.cpp)
target_include_directories(mesheditor_support SYSTEM PRIVATE lib/basis_universal/zstd)
target_link_libraries(mesheditor_support PRIVATE basisu_transcoder)

mesheditor_library(mesheditor_image_codecs HOT src/image/ImageDecode.cpp src/image/ImageEncode.cpp)
target_include_directories(mesheditor_image_codecs SYSTEM PRIVATE lib/lunasvg/plutovg/source)
target_link_libraries(mesheditor_image_codecs PRIVATE mesheditor_support webp)

add_library(mesheditor_imgui STATIC
    lib/imgui/imgui.cpp lib/imgui/imgui_draw.cpp lib/imgui/imgui_tables.cpp
    lib/imgui/imgui_widgets.cpp lib/imgui/imgui_demo.cpp
    lib/imgui/backends/imgui_impl_metal.mm lib/imgui/backends/imgui_impl_osx.mm)
target_include_directories(mesheditor_imgui SYSTEM PUBLIC lib/imgui lib/imgui/backends lib/metal-cpp)
target_compile_definitions(mesheditor_imgui PUBLIC IMGUI_DEFINE_MATH_OPERATORS IMGUI_IMPL_METAL_CPP)
target_compile_options(mesheditor_imgui PRIVATE -O2 -w)
target_link_libraries(mesheditor_imgui PRIVATE "-framework Metal" "-framework AppKit" "-framework QuartzCore" "-framework GameController")
add_library(mesheditor_implot STATIC lib/implot/implot.cpp lib/implot/implot_items.cpp lib/implot/implot_demo.cpp)
target_include_directories(mesheditor_implot SYSTEM PUBLIC lib/implot)
target_compile_options(mesheditor_implot PRIVATE -O2 -w)
# Numeric types used by the audio plots and the bundled ImPlot demo.
target_compile_definitions(mesheditor_implot PRIVATE "IMPLOT_CUSTOM_NUMERIC_TYPES=(ImS8)(ImS32)(ImU32)(float)(double)")
target_link_libraries(mesheditor_implot PUBLIC mesheditor_imgui)

mesheditor_library(MeshEditorMetal HOT
    src/Profile.cpp
    src/metal/Bindless.cpp
    src/metal/Buffer.cpp
    src/metal/Image.cpp
    src/metal/MetalContext.cpp
    src/metal/MetalCppImpl.cpp
    src/metal/MslSource.cpp
    src/metal/PassChain.cpp
    src/metal/PassTimer.cpp
    src/metal/RenderTarget.cpp
    src/metal/Shader.cpp
)

mesheditor_library(MeshEditorMesh HOT
    src/mesh/Mesh.cpp
    src/mesh/MeshBatch.cpp
    src/mesh/MeshBvh.cpp
    src/mesh/MeshConnectivityGpu.cpp
    src/mesh/MeshPipelines.cpp
    src/mesh/MeshStore.cpp
    src/mesh/MeshStores.cpp
    src/mesh/Primitives.cpp
    src/mesh/VertexAdjacencyGpu.cpp
    src/mesh/VertexWeldGpu.cpp
    src/selection/SelectionBitset.cpp
)

mesheditor_library(MeshEditorScene HOT
    src/TransformMath.cpp
    src/armature/Armature.cpp
    src/scene/Defaults.cpp
    src/scene/Entity.cpp
    src/scene/ObjectCreation.cpp
    src/scene/RotationUi.cpp
    src/scene/SceneGraph.cpp
    src/scene/SelectionQueries.cpp
    src/scene/SelectionState.cpp
    src/viewport/RenderExtent.cpp
    src/viewport/ViewCamera.cpp
    src/viewport/ViewCameraOps.cpp
    src/viewport/ViewportDisplay.cpp
)

mesheditor_library(MeshEditorRender HOT
    src/render/ClusterLod.cpp
    src/render/GpuBufferOps.cpp
    src/render/GpuScene.cpp
    src/render/MeshUpdates.cpp
    src/render/Pipelines.cpp
    src/render/RenderStores.cpp
    src/render/SceneUpdates.cpp
    src/render/Textures.cpp
    src/render/ViewportSubmission.cpp
    src/selection/SelectionGpu.cpp
    src/viewport/ViewportRenderGpu.cpp
)

mesheditor_library(MeshEditorPhysics HOT
    src/physics/ColliderUpdate.cpp
    src/physics/PhysicsStores.cpp
    src/physics/PhysicsSystem.cpp
)

mesheditor_library(MeshEditorAudio HOT
    src/audio/AudioDevice.cpp
    src/audio/AudioFile.cpp
    src/audio/AudioRender.cpp
    src/audio/AudioStores.cpp
    src/audio/ContactDynamics.cpp
    src/audio/ContactModel.cpp
    src/audio/Fft.cpp
    src/audio/FftAnalysis.cpp
    src/audio/ModalAudio.cpp
    src/audio/ModalModelFile.cpp
    src/audio/ModalSolve.cpp
)

mesheditor_library(MeshEditorAssets COLD
    src/assets/MaterialImport.cpp
    src/assets/MeshImport.cpp
    src/audio/RealImpact.cpp
    src/gltf/GltfScene.cpp
    src/gltf/GltfExport.cpp
    src/gltf/SourceTexture.cpp
)

mesheditor_library(MeshEditorEditor COLD
    src/ProcessEvents.cpp
    src/Stores.cpp
    src/action/Action.cpp
    src/action/ActionApply.cpp
    src/action/Audio.cpp
    src/action/Bone.cpp
    src/action/Core.cpp
    src/action/Io.cpp
    src/action/Log.cpp
    src/action/LogSerialize.cpp
    src/action/Object.cpp
    src/action/Physics.cpp
    src/action/Selection.cpp
    src/action/Timeline.cpp
    src/action/View.cpp
    src/editor/AudioExcitation.cpp
    src/editor/AudioIntegration.cpp
    src/editor/Timeline.cpp
    src/object/ObjectOps.cpp
    src/selection/Selection.cpp
    src/selection/SelectionOps.cpp
    src/snapshot/ArmatureSnapshot.cpp
    src/snapshot/AssetsSnapshot.cpp
    src/snapshot/AudioSnapshot.cpp
    src/snapshot/MeshSnapshot.cpp
    src/snapshot/PhysicsSnapshot.cpp
    src/snapshot/ReplayTestFixture.cpp
    src/snapshot/SaveState.cpp
    src/snapshot/SceneComponentsSnapshot.cpp
    src/snapshot/SceneSnapshot.cpp
    src/snapshot/SnapshotRoles.cpp
    src/snapshot/ViewportSnapshot.cpp
    src/viewport/Viewport.cpp
    src/viewport/ViewportOps.cpp
)

mesheditor_library(MeshEditorUi COLD
    src/VideoRecorder.cpp
    src/WorkspaceState.cpp
    src/animation/AnimationTimeline.cpp
    src/audio/AudioDeviceUi.cpp
    src/audio/AudioUi.cpp
    src/gizmo/TransformGizmo.cpp
    src/physics/PhysicsUi.cpp
    src/render/SvgResource.cpp
    src/scene/SceneControlsUi.cpp
    src/ui/FieldEdit.cpp
    src/ui/MacBackend.mm
    src/viewport/ViewportIcons.cpp
    src/viewport/ViewportPresent.cpp
    src/viewport/ViewportUi.cpp
)

mesheditor_library(MeshEditorPlatform COLD
    src/FileDialog.mm
    src/MacPlatform.mm
)

if(MESHEDITOR_SURFACE_AUDIO)
    target_sources(MeshEditorAudio PRIVATE src/audio/surface/SurfaceModel.cpp)
    target_sources(MeshEditorEditor PRIVATE src/audio/surface/SurfaceAudio.cpp)
    target_sources(MeshEditorUi PRIVATE src/audio/surface/SurfaceUi.cpp)
else()
    target_sources(MeshEditorAudio PRIVATE src/audio/SurfaceContactAbsent.cpp)
    target_sources(MeshEditorUi PRIVATE src/audio/SurfaceUiAbsent.cpp)
endif()

target_include_directories(MeshEditorMetal SYSTEM PUBLIC lib/metal-cpp)
target_link_libraries(MeshEditorMetal PUBLIC mesheditor_support PRIVATE "-framework Metal" "-framework Foundation" "-framework QuartzCore")
target_link_libraries(MeshEditorMesh PUBLIC MeshEditorMetal PRIVATE meshoptimizer mesheditor_serialization)
target_link_libraries(MeshEditorScene PUBLIC MeshEditorMesh)
target_link_libraries(MeshEditorRender PUBLIC MeshEditorScene PRIVATE meshoptimizer mesheditor_image_codecs basisu_transcoder)
target_link_libraries(MeshEditorPhysics PUBLIC MeshEditorScene PRIVATE Jolt)
target_link_libraries(MeshEditorAudio PUBLIC MeshEditorScene PRIVATE mesheditor_serialization FastFEM::FastFEM "-framework Accelerate" "-framework AudioToolbox" "-framework AudioUnit" "-framework CoreAudio")
target_link_libraries(MeshEditorAssets PUBLIC MeshEditorRender PRIVATE meshoptimizer MeshEditorAudio fastgltf::fastgltf simdjson::simdjson tinyobjloader tinyply mesheditor_image_codecs)
target_include_directories(MeshEditorAssets SYSTEM PRIVATE lib/tinyobjloader lib/tinyply/source)
target_include_directories(MeshEditorEditor SYSTEM PUBLIC lib/readerwriterqueue)
target_link_libraries(MeshEditorEditor PUBLIC mesheditor_serialization MeshEditorAssets MeshEditorPhysics MeshEditorAudio)
target_link_libraries(MeshEditorPlatform PUBLIC MeshEditorMetal PRIVATE "-framework AppKit" "-framework UniformTypeIdentifiers" "-framework GameController")
target_link_libraries(MeshEditorUi PUBLIC MeshEditorEditor MeshEditorPlatform mesheditor_implot PRIVATE lunasvg)
target_include_directories(MeshEditorUi SYSTEM PUBLIC lib/imspinner)
set_source_files_properties(src/FileDialog.mm src/MacPlatform.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
set_property(SOURCE src/snapshot/ReplayTestFixture.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "$<$<CONFIG:Debug>:REPLAY_FIXTURE_DIR=\"${CMAKE_SOURCE_DIR}/tests/replay\">")
set_property(SOURCE src/action/Log.cpp APPEND PROPERTY COMPILE_DEFINITIONS RESTORE_SESSION_RETAIN=${RESTORE_SESSION_RETAIN})

# Contact processing and modal input preparation run numerical loops inside the editor integration.
set_property(SOURCE src/editor/AudioIntegration.cpp src/audio/surface/SurfaceAudio.cpp
    APPEND PROPERTY COMPILE_OPTIONS -O2)
