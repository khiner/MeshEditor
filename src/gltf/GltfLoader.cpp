#include "GltfLoader.h"

#include "numeric/vec3.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <format>
#include <functional>
#include <numeric>
#include <unordered_map>

namespace gltf {
namespace {
mat4 ToGlmMatrix(const fastgltf::math::fmat4x4 &matrix) {
    mat4 out{1.f};
    for (size_t c = 0; c < 4; ++c) {
        for (size_t r = 0; r < 4; ++r) out[c][r] = matrix[c][r];
    }
    return out;
}

void AppendPrimitive(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, MeshData &mesh_data) {
    if (primitive.type != fastgltf::PrimitiveType::Triangles) return;

    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return;
    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return;

    const uint32_t base_vertex = mesh_data.Positions.size();
    mesh_data.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, position_accessor,
        [&](fastgltf::math::fvec3 position, size_t index) {
            mesh_data.Positions[base_vertex + index] = vec3{position.x(), position.y(), position.z()};
        }
    );

    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor.has_value()) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(position_accessor.count);
        std::iota(indices.begin(), indices.end(), 0u);
    }

    if (indices.size() < 3) return;
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        mesh_data.Faces.push_back(
            {
                base_vertex + indices[i],
                base_vertex + indices[i + 1],
                base_vertex + indices[i + 2],
            }
        );
    }
}

std::expected<fastgltf::Asset, std::string> ParseAsset(const std::filesystem::path &path) {
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) {
        return std::unexpected{std::format("Failed to open glTF file '{}': {}", path.string(), fastgltf::getErrorMessage(gltf_file.error()))};
    }

    fastgltf::Parser parser{fastgltf::Extensions::KHR_mesh_quantization};
    constexpr auto options =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble |
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::GenerateMeshIndices;

    auto parsed = parser.loadGltf(gltf_file.get(), path.parent_path(), options);
    if (parsed.error() != fastgltf::Error::None) {
        return std::unexpected{std::format("Failed to parse glTF '{}': {}", path.string(), fastgltf::getErrorMessage(parsed.error()))};
    }

    return std::move(parsed.get());
}

std::optional<std::size_t> EnsureMeshData(
    const fastgltf::Asset &asset,
    std::size_t source_mesh_index,
    SceneData &scene_data,
    std::unordered_map<std::size_t, std::optional<std::size_t>> &mesh_index_map
) {
    if (auto it = mesh_index_map.find(source_mesh_index); it != mesh_index_map.end()) return it->second;

    const auto &source_mesh = asset.meshes[source_mesh_index];
    MeshData mesh_data;
    for (const auto &primitive : source_mesh.primitives) {
        AppendPrimitive(asset, primitive, mesh_data);
    }

    if (mesh_data.Positions.empty() || mesh_data.Faces.empty()) {
        mesh_index_map.emplace(source_mesh_index, std::nullopt);
        return std::nullopt;
    }

    const auto mesh_index = scene_data.Meshes.size();
    SceneMeshData scene_mesh;
    scene_mesh.Data = std::move(mesh_data);
    scene_mesh.Name = source_mesh.name.empty() ? std::format("Mesh{}", source_mesh_index) : std::string(source_mesh.name);
    scene_data.Meshes.emplace_back(std::move(scene_mesh));
    mesh_index_map.emplace(source_mesh_index, mesh_index);
    return mesh_index;
}
} // namespace

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path) {
    auto parsed_asset = ParseAsset(path);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", path.string())};
    const size_t scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", path.string())};

    SceneData scene_data;
    std::unordered_map<std::size_t, std::optional<std::size_t>> mesh_index_map;

    const auto &scene = asset.scenes[scene_index];
    const auto traverse =
        [&](std::size_t node_index, const fastgltf::math::fmat4x4 &parent_matrix, std::optional<std::size_t> nearest_parent_instance, const auto &self) -> void {
        if (node_index >= asset.nodes.size()) return;
        const auto &node = asset.nodes[node_index];
        const auto node_matrix = fastgltf::getTransformMatrix(node, parent_matrix);

        auto parent_instance = nearest_parent_instance;
        if (node.meshIndex.has_value() && *node.meshIndex < asset.meshes.size()) {
            if (auto mesh_index = EnsureMeshData(asset, *node.meshIndex, scene_data, mesh_index_map)) {
                SceneInstanceData instance;
                instance.MeshIndex = *mesh_index;
                instance.ParentInstanceIndex = nearest_parent_instance;
                instance.WorldTransform = ToGlmMatrix(node_matrix);
                if (!node.name.empty()) instance.Name = node.name;
                else if (!asset.meshes[*node.meshIndex].name.empty()) instance.Name = asset.meshes[*node.meshIndex].name;
                else instance.Name = std::format("Node{}", node_index);
                scene_data.Instances.emplace_back(std::move(instance));
                parent_instance = scene_data.Instances.size() - 1;
            }
        }

        // Temporary minimal mapping: transform-only glTF nodes are not created as MeshEditor entities yet.
        // Their transforms are folded into descendant world transforms, and parenting links are kept between mesh-bearing nodes.
        for (const auto child_index : node.children) {
            self(child_index, node_matrix, parent_instance, self);
        }
    };
    for (const auto root_index : scene.nodeIndices) {
        traverse(root_index, fastgltf::math::fmat4x4(1.f), std::nullopt, traverse);
    }

    if (scene_data.Meshes.empty() || scene_data.Instances.empty()) {
        return std::unexpected{std::format("glTF '{}' has no triangle primitives with POSITION data.", path.string())};
    }
    return scene_data;
}
} // namespace gltf
