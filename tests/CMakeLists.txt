add_executable(MeshEditorTests
    RoundtripTest.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf/GltfScene.cpp
    ${CMAKE_SOURCE_DIR}/src/TransformMath.cpp
)

target_include_directories(MeshEditorTests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/glm
    SYSTEM ${CMAKE_SOURCE_DIR}/lib/entt/src
)

target_link_libraries(MeshEditorTests PRIVATE fastgltf::fastgltf Boost::ut)
target_compile_definitions(MeshEditorTests PRIVATE GLM_ENABLE_EXPERIMENTAL)
target_compile_options(MeshEditorTests PRIVATE -Wall -Wextra -Wno-missing-field-initializers -Wno-elaborated-enum-base)
