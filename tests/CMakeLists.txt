# Test executables link the production libraries rather than recompiling their sources.
function(mesheditor_test target source)
    add_executable(${target} ${source})
    mesheditor_compile_policy(${target} COLD)
    target_link_libraries(${target} PRIVATE Boost::ut ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

mesheditor_test(MeshEditorTests RoundtripTest.cpp MeshEditorEditor
    fastgltf::fastgltf simdjson::simdjson basisu_transcoder meshoptimizer Jolt webp)
target_sources(MeshEditorTests PRIVATE SelectionTest.cpp)
target_compile_definitions(MeshEditorTests PRIVATE
    MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}"
    MESHEDITOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

mesheditor_test(MeshEditorActionSerializeTest ActionSerializeTest.cpp MeshEditorEditor)
mesheditor_test(MeshEditorModalSolveIntegrationTest ModalSolveIntegrationTest.cpp MeshEditorAudio FastFEM::FastFEM)
mesheditor_test(MeshEditorCompressTest CompressTest.cpp mesheditor_support)
mesheditor_test(MeshEditorAudioFileTest AudioFileTest.cpp MeshEditorAudio)
mesheditor_test(MeshEditorContactCollectBench ContactCollectBench.cpp MeshEditorPhysics Jolt)
target_compile_options(MeshEditorContactCollectBench PRIVATE -O2)
mesheditor_test(MeshEditorContactCollectTest ContactCollectTest.cpp MeshEditorPhysics Jolt)
mesheditor_test(MeshEditorContactManifoldTest ContactManifoldTest.cpp MeshEditorPhysics Jolt)
mesheditor_test(MeshEditorClusterLodTest ClusterLodTest.cpp MeshEditorRender meshoptimizer)
mesheditor_test(MeshEditorMeshBvhTest MeshBvhTest.cpp MeshEditorMesh)
mesheditor_test(MeshEditorModalRenderTest ModalRenderTest.cpp MeshEditorAudio)
mesheditor_test(MeshEditorContactModelTest ContactModelTest.cpp MeshEditorAudio)

if(MESHEDITOR_SURFACE_AUDIO)
    mesheditor_test(MeshEditorSurfaceModelTest surface/SurfaceModelTest.cpp MeshEditorAudio)
    mesheditor_test(MeshEditorSurfaceRenderTest surface/SurfaceRenderTest.cpp MeshEditorAudio)
    mesheditor_test(MeshEditorSurfaceTrackPoolTest surface/SurfaceTrackPoolTest.cpp MeshEditorAudio)
    mesheditor_test(MeshEditorContactSpringsScore surface/ContactSpringsScore.cpp MeshEditorAudio)
endif()

foreach(test MslCompile Transparency)
    set(target MeshEditor${test}Test)
    mesheditor_test(${target} ${test}Test.cpp MeshEditorMetal)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endforeach()
