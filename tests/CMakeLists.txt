# Test executables link the production libraries rather than recompiling their sources.
function(mesheditor_test target source)
    add_executable(${target} ${source})
    mesheditor_compile_policy(${target} HOT)
    target_link_libraries(${target} PRIVATE Boost::ut ${ARGN})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

mesheditor_test(MeshEditorTests RoundtripTest.cpp MeshEditorEditor
    fastgltf::fastgltf simdjson::simdjson basisu_transcoder meshoptimizer webp)
target_sources(MeshEditorTests PRIVATE SelectionTest.cpp ActionSerializeTest.cpp)
target_compile_definitions(MeshEditorTests PRIVATE
    MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}"
    MESHEDITOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

mesheditor_test(MeshEditorAudioTest SuiteMain.cpp MeshEditorAudio FastFEM::FastFEM)
target_sources(MeshEditorAudioTest PRIVATE AudioFileTest.cpp ContactModelTest.cpp ModalRenderTest.cpp ModalSolveIntegrationTest.cpp)
mesheditor_test(MeshEditorPhysicsTest SuiteMain.cpp MeshEditorPhysics rbp)
target_sources(MeshEditorPhysicsTest PRIVATE RbpBodyTest.cpp)
mesheditor_test(MeshEditorStoreTest SuiteMain.cpp MeshEditorProjectStore mesheditor_support)
target_sources(MeshEditorStoreTest PRIVATE CompressTest.cpp ProjectStoreTest.cpp StateTest.cpp)

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
add_dependencies(MeshEditorMslCompileTest MeshEditorShaderLibraries)

foreach(target IN ITEMS MeshEditorTests MeshEditorPhysicsTest)
    rbp_copy_shaders(${target})
endforeach()

mesheditor_test(MeshEditorProjectBench ProjectBench.cpp MeshEditorEditor)
target_compile_definitions(MeshEditorProjectBench PRIVATE MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}")
rbp_copy_shaders(MeshEditorProjectBench)
mesheditor_test(MeshEditorEditBench EditBench.cpp MeshEditorEditor)
add_dependencies(MeshEditorEditBench MeshEditorPrecompilePipelines)
target_compile_definitions(MeshEditorEditBench PRIVATE MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}")
rbp_copy_shaders(MeshEditorEditBench)
mesheditor_test(MeshEditorProjectTest ProjectTest.cpp MeshEditorUi)
target_sources(MeshEditorProjectTest PRIVATE ProcedureTest.cpp OperatorTest.cpp HistoryTest.cpp)
target_compile_definitions(MeshEditorProjectTest PRIVATE
    MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}" MESHEDITOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
rbp_copy_shaders(MeshEditorProjectTest)

mesheditor_test(MeshEditorGeometryTest GeometryOperationsTest.cpp MeshEditorEditor meshoptimizer)
target_sources(MeshEditorGeometryTest PRIVATE ClusterLodTest.cpp ElementArenaTest.cpp MetalBufferTest.cpp)
target_compile_definitions(MeshEditorGeometryTest PRIVATE MESHEDITOR_BUILD_DIR="${CMAKE_BINARY_DIR}")
rbp_copy_shaders(MeshEditorGeometryTest)
