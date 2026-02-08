#pragma once

#include "mesh/MeshData.h"
#include "numeric/mat4.h"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
struct SceneMeshData {
    MeshData Data;
    std::string Name;
};

struct SceneInstanceData {
    std::size_t MeshIndex{};
    std::optional<std::size_t> ParentInstanceIndex{};
    mat4 WorldTransform{1.f};
    std::string Name;
};

struct SceneData {
    std::vector<SceneMeshData> Meshes;
    std::vector<SceneInstanceData> Instances;
};

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path);
} // namespace gltf
