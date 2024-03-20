set(SDL3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SDL)
add_subdirectory(${SDL3_DIR})

add_subdirectory(nativefiledialog-extended)
add_subdirectory(OpenMesh)
add_subdirectory(VulkanMemoryAllocator)
