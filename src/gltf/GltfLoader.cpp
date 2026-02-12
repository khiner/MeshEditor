#include "GltfLoader.h"

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <glm/gtx/matrix_decompose.hpp>

#include <format>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace gltf {
namespace {

mat4 ToGlmMatrix(const fastgltf::math::fmat4x4 &matrix) {
    mat4 out;
    for (auto c = 0; c < 4; ++c) {
        for (auto r = 0; r < 4; ++r) out[c][r] = matrix[c][r];
    }
    return out;
}

Transform MatrixToTransform(const mat4 &m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    if (!glm::decompose(m, scale, rotation, translation, skew, perspective)) return {};
    return {.P = translation, .R = glm::normalize(rotation), .S = scale};
}

mat4 ToMatrix(const Transform &t) {
    return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S);
}

Transform ToTransform(const fastgltf::TRS &trs) {
    const auto t = trs.translation;
    const auto r = trs.rotation;
    const auto s = trs.scale;
    return {.P = {t.x(), t.y(), t.z()}, .R = glm::normalize(quat{r.w(), r.x(), r.y(), r.z()}), .S = {s.x(), s.y(), s.z()}};
}

Transform GetLocalTransform(const fastgltf::Node &node) {
    if (const auto *trs = std::get_if<fastgltf::TRS>(&node.transform)) return ToTransform(*trs);
    return MatrixToTransform(ToGlmMatrix(std::get<fastgltf::math::fmat4x4>(node.transform)));
}

std::optional<uint32_t> ToIndex(std::size_t index, std::size_t upper_bound) {
    if (index >= upper_bound) return {};
    return index;
}
std::optional<uint32_t> ToIndex(const fastgltf::Optional<std::size_t> &index, std::size_t upper_bound) {
    if (!index) return {};
    return ToIndex(*index, upper_bound);
}

void AppendPrimitive(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, MeshData &mesh_data, std::optional<ArmatureDeformData> &deform_data) {
    if (primitive.type != fastgltf::PrimitiveType::Triangles) return;

    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return;
    const auto joints_it = primitive.findAttribute("JOINTS_0");
    const auto weights_it = primitive.findAttribute("WEIGHTS_0");
    const bool has_joints = joints_it != primitive.attributes.end();
    const bool has_weights = weights_it != primitive.attributes.end();
    if (has_joints != has_weights) {
        throw std::runtime_error{"glTF primitive has JOINTS_0 without WEIGHTS_0 (or vice versa)."};
    }

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return;

    const uint32_t base_vertex = static_cast<uint32_t>(mesh_data.Positions.size());
    mesh_data.Positions.resize(static_cast<std::size_t>(base_vertex) + position_accessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset, position_accessor,
        [&](fastgltf::math::fvec3 position, std::size_t index) {
            mesh_data.Positions[base_vertex + static_cast<uint32_t>(index)] = vec3{position.x(), position.y(), position.z()};
        }
    );

    if (has_joints && !deform_data) deform_data.emplace();
    const bool mesh_has_skin_data = deform_data && (!deform_data->Joints.empty() || !deform_data->Weights.empty());
    if (mesh_has_skin_data &&
        (deform_data->Joints.size() != static_cast<std::size_t>(base_vertex) ||
         deform_data->Weights.size() != static_cast<std::size_t>(base_vertex))) {
        throw std::runtime_error{"glTF primitive append encountered inconsistent skin channel sizes."};
    }

    if (has_joints || mesh_has_skin_data) {
        deform_data->Joints.resize(mesh_data.Positions.size(), {0, 0, 0, 0});
        deform_data->Weights.resize(mesh_data.Positions.size(), vec4{0.f, 0.f, 0.f, 0.f});
    }

    if (has_joints) {
        const auto &joints_accessor = asset.accessors[joints_it->accessorIndex];
        const auto &weights_accessor = asset.accessors[weights_it->accessorIndex];
        if (joints_accessor.count != position_accessor.count || weights_accessor.count != position_accessor.count) {
            throw std::runtime_error{
                std::format(
                    "glTF primitive skin attribute counts must match POSITION count (POSITION={}, JOINTS_0={}, WEIGHTS_0={}).",
                    position_accessor.count,
                    joints_accessor.count,
                    weights_accessor.count
                )
            };
        }

        std::vector<fastgltf::math::u16vec4> joints(position_accessor.count);
        fastgltf::copyFromAccessor<fastgltf::math::u16vec4>(asset, joints_accessor, joints.data());
        std::vector<fastgltf::math::fvec4> weights(position_accessor.count);
        fastgltf::copyFromAccessor<fastgltf::math::fvec4>(asset, weights_accessor, weights.data());

        for (uint32_t i = 0, count = static_cast<uint32_t>(joints.size()); i < count; ++i) {
            const auto &joint = joints[i];
            deform_data->Joints[base_vertex + i] = {joint.x(), joint.y(), joint.z(), joint.w()};
            const auto &weight = weights[i];
            deform_data->Weights[base_vertex + i] = {weight.x(), weight.y(), weight.z(), weight.w()};
        }
    }

    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(position_accessor.count);
        std::iota(indices.begin(), indices.end(), 0u);
    }
    if (indices.size() < 3) return;

    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        mesh_data.Faces.push_back({base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]});
    }
}

std::expected<fastgltf::Asset, std::string> ParseAsset(const std::filesystem::path &path) {
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) {
        return std::unexpected{std::format("Failed to open glTF file '{}': {}", path.string(), fastgltf::getErrorMessage(gltf_file.error()))};
    }

    fastgltf::Parser parser{fastgltf::Extensions::KHR_mesh_quantization};
    using fastgltf::Options;
    auto parsed = parser.loadGltf(gltf_file.get(), path.parent_path(), Options::DontRequireValidAssetMember | Options::AllowDouble | Options::LoadExternalBuffers | Options::GenerateMeshIndices | Options::DecomposeNodeMatrices);
    if (parsed.error() != fastgltf::Error::None) {
        return std::unexpected{std::format("Failed to parse glTF '{}': {}", path.string(), fastgltf::getErrorMessage(parsed.error()))};
    }

    return std::move(parsed.get());
}

std::optional<uint32_t> EnsureMeshData(const fastgltf::Asset &asset, uint32_t source_mesh_index, SceneData &scene_data, std::unordered_map<uint32_t, std::optional<uint32_t>> &mesh_index_map) {
    if (const auto it = mesh_index_map.find(source_mesh_index); it != mesh_index_map.end()) return it->second;
    if (static_cast<std::size_t>(source_mesh_index) >= asset.meshes.size()) {
        mesh_index_map.emplace(source_mesh_index, std::nullopt);
        return {};
    }

    const auto &source_mesh = asset.meshes[source_mesh_index];
    MeshData mesh_data;
    std::optional<ArmatureDeformData> mesh_deform_data;
    for (const auto &primitive : source_mesh.primitives) AppendPrimitive(asset, primitive, mesh_data, mesh_deform_data);
    if (mesh_data.Positions.empty() || mesh_data.Faces.empty()) {
        mesh_index_map.emplace(source_mesh_index, std::nullopt);
        return {};
    }

    const auto mesh_index = scene_data.Meshes.size();
    scene_data.Meshes.emplace_back(
        SceneMeshData{
            .Data = std::move(mesh_data),
            .DeformData = std::move(mesh_deform_data),
            .Name = source_mesh.name.empty() ? std::format("Mesh{}", source_mesh_index) : std::string(source_mesh.name),
        }
    );
    mesh_index_map.emplace(source_mesh_index, mesh_index);
    return mesh_index;
}

std::vector<std::optional<uint32_t>> BuildNodeParentTable(const fastgltf::Asset &asset) {
    std::vector<std::optional<uint32_t>> parents(asset.nodes.size(), std::nullopt);
    for (std::size_t parent_idx = 0; parent_idx < asset.nodes.size(); ++parent_idx) {
        const auto parent = static_cast<uint32_t>(parent_idx);
        for (const auto child_idx : asset.nodes[parent_idx].children) {
            const auto child = ToIndex(child_idx, asset.nodes.size());
            if (!child || parents[*child]) continue;
            parents[*child] = parent;
        }
    }
    return parents;
}

struct SceneTraversalData {
    std::vector<bool> InScene;
    std::vector<mat4> WorldTransforms;
};

SceneTraversalData TraverseSceneNodes(const fastgltf::Asset &asset, uint32_t scene_index) {
    SceneTraversalData traversal;
    traversal.InScene.assign(asset.nodes.size(), false);
    traversal.WorldTransforms.assign(asset.nodes.size(), I4);
    if (scene_index >= asset.scenes.size()) return traversal;

    const auto &scene = asset.scenes[scene_index];
    const auto traverse =
        [&](uint32_t node_index, const fastgltf::math::fmat4x4 &parent_matrix, const auto &self) -> void {
        if (static_cast<std::size_t>(node_index) >= asset.nodes.size()) return;

        const auto &node = asset.nodes[node_index];
        const auto world_matrix = fastgltf::getTransformMatrix(node, parent_matrix);
        traversal.InScene[node_index] = true;
        traversal.WorldTransforms[node_index] = ToGlmMatrix(world_matrix);
        for (const auto child_idx : node.children) {
            if (const auto child = ToIndex(child_idx, asset.nodes.size())) self(*child, world_matrix, self);
        }
    };

    for (const auto root_idx : scene.nodeIndices) {
        if (const auto root = ToIndex(root_idx, asset.nodes.size())) traverse(*root, fastgltf::math::fmat4x4(1.f), traverse);
    }

    return traversal;
}

std::optional<uint32_t> FindNearestJointAncestor(uint32_t node_index, const std::vector<std::optional<uint32_t>> &parents, const std::vector<bool> &is_joint_in_skin) {
    auto parent = parents[node_index];
    while (parent) {
        if (is_joint_in_skin[*parent]) return parent;
        parent = parents[*parent];
    }
    return {};
}

std::optional<uint32_t> FindNearestEmittedObjectAncestor(uint32_t node_index, const std::vector<std::optional<uint32_t>> &parents, const std::vector<bool> &is_object_emitted) {
    auto parent = parents[node_index];
    while (parent) {
        if (is_object_emitted[*parent]) return parent;
        parent = parents[*parent];
    }
    return {};
}

std::optional<uint32_t> ComputeCommonAncestor(const std::vector<uint32_t> &nodes, const std::vector<std::optional<uint32_t>> &parents) {
    if (nodes.empty()) return {};

    const auto build_root_path = [&](uint32_t node_index) {
        std::vector<uint32_t> path;
        path.emplace_back(node_index);
        while (parents[node_index]) {
            node_index = *parents[node_index];
            path.emplace_back(node_index);
        }
        std::reverse(path.begin(), path.end());
        return path;
    };

    auto common_path = build_root_path(nodes.front());
    for (uint32_t i = 1; i < nodes.size() && !common_path.empty(); ++i) {
        const auto path = build_root_path(nodes[i]);
        const auto common_count = static_cast<uint32_t>(std::min(common_path.size(), path.size()));

        uint32_t prefix = 0;
        while (prefix < common_count && common_path[prefix] == path[prefix]) ++prefix;
        common_path.resize(prefix);
    }

    if (common_path.empty()) return {};
    return common_path.back();
}

std::expected<Transform, std::string> ComputeJointRestLocal(
    uint32_t skin_index,
    uint32_t joint_node_index,
    std::optional<uint32_t> parent_joint_node_index,
    std::optional<uint32_t> anchor_node_index,
    const std::vector<std::optional<uint32_t>> &parents,
    const std::vector<Transform> &local_transforms
) {
    const auto rebased_parent_node_index = parent_joint_node_index ? parent_joint_node_index : anchor_node_index;
    if (!rebased_parent_node_index) return local_transforms[joint_node_index];
    if (*rebased_parent_node_index == joint_node_index) return Transform{};

    mat4 rebased_local{I4};
    auto current = joint_node_index;
    while (current != *rebased_parent_node_index) {
        rebased_local = ToMatrix(local_transforms[current]) * rebased_local;
        if (!parents[current]) {
            return std::unexpected{
                std::format(
                    "glTF skin {} joint node {} cannot be rebased to ancestor node {}.",
                    skin_index,
                    joint_node_index,
                    *rebased_parent_node_index
                )
            };
        }
        current = *parents[current];
    }
    return MatrixToTransform(rebased_local);
}

std::expected<std::vector<uint32_t>, std::string> BuildParentBeforeChildJointOrder(const std::vector<uint32_t> &source_joint_nodes, const std::unordered_map<uint32_t, std::optional<uint32_t>> &joint_parent_map, uint32_t skin_index) {
    std::vector<uint32_t> ordered;
    ordered.reserve(source_joint_nodes.size());

    std::unordered_map<uint32_t, uint8_t> state;
    state.reserve(source_joint_nodes.size());

    std::string error;
    const std::function<bool(uint32_t)> emit_joint = [&](uint32_t joint_node_index) {
        const auto current_state = state[joint_node_index];
        if (current_state == 2) return true;
        if (current_state == 1) {
            error = std::format("glTF skin {} has a cycle in joint ancestry at node {}.", skin_index, joint_node_index);
            return false;
        }

        state[joint_node_index] = 1;
        if (const auto it = joint_parent_map.find(joint_node_index);
            it != joint_parent_map.end() && it->second && !emit_joint(*it->second)) {
            return false;
        }

        state[joint_node_index] = 2;
        ordered.emplace_back(joint_node_index);
        return true;
    };

    for (const auto joint_node_index : source_joint_nodes) {
        if (!emit_joint(joint_node_index)) return std::unexpected{error};
    }

    return ordered;
}

std::vector<mat4> LoadInverseBindMatrices(
    const fastgltf::Asset &asset,
    const fastgltf::Skin &skin,
    uint32_t joint_count
) {
    std::vector<mat4> inverse_bind_matrices(joint_count, I4);
    if (!skin.inverseBindMatrices || *skin.inverseBindMatrices >= asset.accessors.size()) return inverse_bind_matrices;

    const auto &accessor = asset.accessors[*skin.inverseBindMatrices];
    if (accessor.type != fastgltf::AccessorType::Mat4 || accessor.count == 0) return inverse_bind_matrices;

    std::vector<fastgltf::math::fmat4x4> ibm(accessor.count);
    fastgltf::copyFromAccessor<fastgltf::math::fmat4x4>(asset, accessor, ibm.data());

    const auto count = std::min(ibm.size(), static_cast<std::size_t>(joint_count));
    for (std::size_t i = 0; i < count; ++i) inverse_bind_matrices[i] = ToGlmMatrix(ibm[i]);
    return inverse_bind_matrices;
}

std::string MakeNodeName(const fastgltf::Asset &asset, uint32_t node_index, std::optional<uint32_t> source_mesh_index = {}) {
    const auto &node = asset.nodes[node_index];
    if (!node.name.empty()) return std::string(node.name);

    if (source_mesh_index && static_cast<std::size_t>(*source_mesh_index) < asset.meshes.size()) {
        const auto &mesh_name = asset.meshes[*source_mesh_index].name;
        if (!mesh_name.empty()) return std::string(mesh_name);
    }

    return std::format("Node{}", node_index);
}
} // namespace

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path) {
    auto parsed_asset = ParseAsset(path);
    if (!parsed_asset) return std::unexpected{parsed_asset.error()};

    auto &asset = *parsed_asset;
    if (asset.scenes.empty()) return std::unexpected{std::format("glTF '{}' has no scenes.", path.string())};

    const auto scene_index = asset.defaultScene.value_or(0);
    if (scene_index >= asset.scenes.size()) return std::unexpected{std::format("glTF '{}' has invalid default scene index.", path.string())};

    SceneData scene_data;
    const auto parents = BuildNodeParentTable(asset);
    const auto traversal = TraverseSceneNodes(asset, static_cast<uint32_t>(scene_index));
    std::vector<Transform> local_transforms(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        local_transforms[node_index] = GetLocalTransform(asset.nodes[node_index]);
    }

    std::vector<bool> used_skin(asset.skins.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!traversal.InScene[node_index]) continue;
        if (const auto skin_index = ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size())) used_skin[*skin_index] = true;
    }

    std::vector<bool> is_joint(asset.nodes.size(), false);
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!used_skin[skin_index]) continue;
        const auto &skin = asset.skins[skin_index];
        for (const auto joint_idx : skin.joints) {
            if (const auto joint = ToIndex(joint_idx, asset.nodes.size())) is_joint[*joint] = true;
        }
    }

    std::unordered_map<uint32_t, std::optional<uint32_t>> mesh_index_map;
    scene_data.Nodes.resize(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        const auto &source_node = asset.nodes[node_index];
        auto &node = scene_data.Nodes[node_index];
        node.NodeIndex = node_index;
        node.ParentNodeIndex = parents[node_index];
        node.LocalTransform = local_transforms[node_index];
        node.InScene = traversal.InScene[node_index];
        node.WorldTransform = traversal.InScene[node_index] ? traversal.WorldTransforms[node_index] : I4;
        node.IsJoint = is_joint[node_index];
        node.SkinIndex = ToIndex(source_node.skinIndex, asset.skins.size());
        node.Name = MakeNodeName(asset, node_index, ToIndex(source_node.meshIndex, asset.meshes.size()));
        node.ChildrenNodeIndices.clear();
        node.ChildrenNodeIndices.reserve(source_node.children.size());
        for (const auto child_idx : source_node.children) {
            if (const auto child = ToIndex(child_idx, asset.nodes.size())) node.ChildrenNodeIndices.emplace_back(*child);
        }
        if (traversal.InScene[node_index]) {
            if (const auto source_mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size())) {
                node.MeshIndex = EnsureMeshData(asset, *source_mesh_index, scene_data, mesh_index_map);
            }
        }
    }

    std::vector<bool> is_object_emitted(asset.nodes.size(), false);
    for (uint32_t node_index = 0; node_index < scene_data.Nodes.size(); ++node_index) {
        if (const auto &node = scene_data.Nodes[node_index]; node.InScene) {
            // Joint nodes are bone-only unless they also carry renderable mesh data.
            is_object_emitted[node_index] = node.MeshIndex.has_value() || !node.IsJoint;
        }
    }

    std::vector<std::optional<uint32_t>> nearest_object_ancestor(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        nearest_object_ancestor[node_index] = FindNearestEmittedObjectAncestor(node_index, parents, is_object_emitted);
    }

    for (uint32_t node_index = 0; node_index < scene_data.Nodes.size(); ++node_index) {
        if (const auto &node = scene_data.Nodes[node_index]; is_object_emitted[node_index]) {
            const auto source_mesh_index = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size());
            scene_data.Objects.emplace_back(
                SceneObjectData{
                    .ObjectType = node.MeshIndex ? SceneObjectData::Type::Mesh : SceneObjectData::Type::Empty,
                    .NodeIndex = node.NodeIndex,
                    .ParentNodeIndex = nearest_object_ancestor[node.NodeIndex],
                    .WorldTransform = node.WorldTransform,
                    .MeshIndex = node.MeshIndex,
                    .SkinIndex = node.SkinIndex,
                    .Name = MakeNodeName(asset, node.NodeIndex, source_mesh_index),
                }
            );
        }
    }

    scene_data.Skins.reserve(asset.skins.size());
    for (uint32_t skin_index = 0; skin_index < asset.skins.size(); ++skin_index) {
        if (!used_skin[skin_index]) continue;
        const auto &skin = asset.skins[skin_index];

        std::vector<uint32_t> source_joint_nodes;
        source_joint_nodes.reserve(skin.joints.size());
        std::unordered_set<uint32_t> seen_joints;
        for (const auto joint_idx : skin.joints) {
            if (const auto joint = ToIndex(joint_idx, asset.nodes.size())) {
                if (const auto [it, inserted] = seen_joints.emplace(*joint); inserted) {
                    source_joint_nodes.emplace_back(*it);
                }
            }
        }
        if (source_joint_nodes.empty()) continue;

        std::vector<bool> is_joint_in_skin(asset.nodes.size(), false);
        for (const auto joint_node_index : source_joint_nodes) is_joint_in_skin[joint_node_index] = true;

        std::unordered_map<uint32_t, std::optional<uint32_t>> joint_parent_map;
        joint_parent_map.reserve(source_joint_nodes.size());
        for (const auto joint_node_index : source_joint_nodes) {
            joint_parent_map.emplace(joint_node_index, FindNearestJointAncestor(joint_node_index, parents, is_joint_in_skin));
        }

        auto ordered_joint_nodes = BuildParentBeforeChildJointOrder(source_joint_nodes, joint_parent_map, skin_index);
        if (!ordered_joint_nodes) return std::unexpected{ordered_joint_nodes.error()};

        const auto skeleton_node_index = ToIndex(skin.skeleton, asset.nodes.size());
        // Deterministic armature scene anchor: explicit skin.skeleton if present,
        // otherwise the computed joint ancestry root. Do not synthesize extra roots.
        SceneSkinData scene_skin{
            .SkinIndex = skin_index,
            .Name = skin.name.empty() ? std::format("Skin{}", skin_index) : std::string(skin.name),
            .SkeletonNodeIndex = skeleton_node_index,
            .AnchorNodeIndex = skeleton_node_index ? skeleton_node_index : ComputeCommonAncestor(*ordered_joint_nodes, parents),
        };
        scene_skin.Joints.reserve(ordered_joint_nodes->size());
        for (const auto joint_node_index : *ordered_joint_nodes) {
            const auto parent_joint_node_index = joint_parent_map.at(joint_node_index);
            auto rest_local = ComputeJointRestLocal(
                skin_index,
                joint_node_index,
                parent_joint_node_index,
                scene_skin.AnchorNodeIndex,
                parents,
                local_transforms
            );
            if (!rest_local) return std::unexpected{rest_local.error()};

            scene_skin.Joints.emplace_back(
                SkinJointData{
                    .JointNodeIndex = joint_node_index,
                    .ParentJointNodeIndex = parent_joint_node_index,
                    .RestLocal = *rest_local,
                    .Name = MakeNodeName(asset, joint_node_index),
                }
            );
        }
        scene_skin.InverseBindMatrices = LoadInverseBindMatrices(asset, skin, static_cast<uint32_t>(scene_skin.Joints.size()));
        if (scene_skin.AnchorNodeIndex && traversal.InScene[*scene_skin.AnchorNodeIndex]) {
            scene_skin.ParentObjectNodeIndex = nearest_object_ancestor[*scene_skin.AnchorNodeIndex];
        }
        scene_data.Skins.emplace_back(std::move(scene_skin));
    }

    // Parse animations
    for (uint32_t anim_index = 0; anim_index < asset.animations.size(); ++anim_index) {
        const auto &anim = asset.animations[anim_index];
        AnimationClipData clip;
        clip.Name = anim.name.empty() ? std::format("Animation{}", anim_index) : std::string(anim.name);
        float max_time = 0;

        for (const auto &channel : anim.channels) {
            if (!channel.nodeIndex || *channel.nodeIndex >= asset.nodes.size()) continue;
            if (channel.samplerIndex >= anim.samplers.size()) continue;

            AnimationPath target_path;
            uint32_t component_count;
            switch (channel.path) {
                case fastgltf::AnimationPath::Translation:
                    target_path = AnimationPath::Translation;
                    component_count = 3;
                    break;
                case fastgltf::AnimationPath::Rotation:
                    target_path = AnimationPath::Rotation;
                    component_count = 4;
                    break;
                case fastgltf::AnimationPath::Scale:
                    target_path = AnimationPath::Scale;
                    component_count = 3;
                    break;
                default: continue; // Skip weights and other unsupported paths
            }

            const auto &sampler = anim.samplers[channel.samplerIndex];
            if (sampler.inputAccessor >= asset.accessors.size() || sampler.outputAccessor >= asset.accessors.size()) continue;

            const auto &input_accessor = asset.accessors[sampler.inputAccessor];
            const auto &output_accessor = asset.accessors[sampler.outputAccessor];
            if (input_accessor.count == 0) continue;

            AnimationInterpolation interp;
            switch (sampler.interpolation) {
                case fastgltf::AnimationInterpolation::Step: interp = AnimationInterpolation::Step; break;
                case fastgltf::AnimationInterpolation::Linear: interp = AnimationInterpolation::Linear; break;
                case fastgltf::AnimationInterpolation::CubicSpline: interp = AnimationInterpolation::CubicSpline; break;
                default: interp = AnimationInterpolation::Linear; break;
            }

            std::vector<float> times(input_accessor.count);
            fastgltf::copyFromAccessor<float>(asset, input_accessor, times.data());

            std::vector<float> values(output_accessor.count * component_count);
            if (component_count == 4) {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(asset, output_accessor, [&](const auto &v, std::size_t i) {
                    const auto base = i * 4;
                    values[base] = v.x();
                    values[base + 1] = v.y();
                    values[base + 2] = v.z();
                    values[base + 3] = v.w();
                });
            } else {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, output_accessor, [&](const auto &v, std::size_t i) {
                    const auto base = i * 3;
                    values[base] = v.x();
                    values[base + 1] = v.y();
                    values[base + 2] = v.z();
                });
            }

            if (!times.empty()) max_time = std::max(max_time, times.back());

            clip.Channels.emplace_back(AnimationChannelData{
                .TargetNodeIndex = uint32_t(*channel.nodeIndex),
                .Target = target_path,
                .Interp = interp,
                .TimesSeconds = std::move(times),
                .Values = std::move(values),
            });
        }

        clip.DurationSeconds = max_time;
        if (!clip.Channels.empty()) scene_data.Animations.emplace_back(std::move(clip));
    }

    if (scene_data.Objects.empty() && scene_data.Skins.empty()) {
        return std::unexpected{std::format("glTF '{}' has no importable scene objects or skins.", path.string())};
    }
    return scene_data;
}
} // namespace gltf
