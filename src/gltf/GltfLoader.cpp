#include "GltfLoader.h"

#include "LightTypes.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace gltf {
namespace {
Transform MatrixToTransform(const mat4 &m) {
    vec3 scale, translation, skew;
    vec4 perspective;
    quat rotation;
    if (!glm::decompose(m, scale, rotation, translation, skew, perspective)) return {};
    return {.P = translation, .R = glm::normalize(rotation), .S = scale};
}

mat4 ToMatrix(const Transform &t) { return glm::translate(I4, t.P) * glm::mat4_cast(glm::normalize(t.R)) * glm::scale(I4, t.S); }

Transform ToTransform(const fastgltf::TRS &trs) {
    const auto t = trs.translation;
    const auto r = trs.rotation;
    const auto s = trs.scale;
    return {.P = {t.x(), t.y(), t.z()}, .R = glm::normalize(quat{r.w(), r.x(), r.y(), r.z()}), .S = {s.x(), s.y(), s.z()}};
}

std::optional<uint32_t> ToIndex(std::size_t index, std::size_t upper_bound) {
    if (index >= upper_bound) return {};
    return index;
}
std::optional<uint32_t> ToIndex(const fastgltf::Optional<std::size_t> &index, std::size_t upper_bound) {
    if (!index) return {};
    return ToIndex(*index, upper_bound);
}

// Appends positions/edges from a non-triangle primitive into the target MeshData,
// merging with any previously appended primitives of the same topology.
void AppendNonTrianglePrimitive(const fastgltf::Asset &asset, const fastgltf::Primitive &primitive, MeshData &target) {
    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return;

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return;

    const uint32_t base_vertex = target.Positions.size();
    target.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &target.Positions[base_vertex]);

    // Points have no indices to process — just positions
    if (primitive.type == fastgltf::PrimitiveType::Points) return;

    std::vector<uint32_t> indices;
    if (primitive.indicesAccessor) {
        const auto &index_accessor = asset.accessors[*primitive.indicesAccessor];
        indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<uint32_t>(asset, index_accessor, indices.data());
    } else {
        indices.resize(position_accessor.count);
        std::iota(indices.begin(), indices.end(), 0u);
    }

    // Offset indices by base_vertex for merging
    switch (primitive.type) {
        case fastgltf::PrimitiveType::Points:
        case fastgltf::PrimitiveType::Triangles:
        case fastgltf::PrimitiveType::TriangleStrip:
        case fastgltf::PrimitiveType::TriangleFan:
            break;
        case fastgltf::PrimitiveType::Lines:
            for (uint32_t i = 0; i + 1 < indices.size(); i += 2) target.Edges.push_back({base_vertex + indices[i], base_vertex + indices[i + 1]});
            break;
        case fastgltf::PrimitiveType::LineStrip:
            for (uint32_t i = 0; i + 1 < indices.size(); ++i) target.Edges.push_back({base_vertex + indices[i], base_vertex + indices[i + 1]});
            break;
        case fastgltf::PrimitiveType::LineLoop:
            for (uint32_t i = 0; i + 1 < indices.size(); ++i) target.Edges.push_back({base_vertex + indices[i], base_vertex + indices[i + 1]});
            if (indices.size() >= 2) target.Edges.push_back({base_vertex + indices.back(), base_vertex + indices.front()});
            break;
    }
}

std::expected<void, std::string> AppendPrimitive(
    const fastgltf::Asset &asset,
    const fastgltf::Primitive &primitive,
    MeshData &mesh_data,
    std::optional<ArmatureDeformData> &deform_data,
    std::optional<MorphTargetData> &morph_data
) {
    if (primitive.type != fastgltf::PrimitiveType::Triangles &&
        primitive.type != fastgltf::PrimitiveType::TriangleStrip &&
        primitive.type != fastgltf::PrimitiveType::TriangleFan) return {};

    const auto position_it = primitive.findAttribute("POSITION");
    if (position_it == primitive.attributes.end()) return {};
    const auto joints_it = primitive.findAttribute("JOINTS_0");
    const auto weights_it = primitive.findAttribute("WEIGHTS_0");
    const bool has_joints = joints_it != primitive.attributes.end();
    const bool has_weights = weights_it != primitive.attributes.end();
    if (has_joints != has_weights) return std::unexpected{"glTF primitive has JOINTS_0 without WEIGHTS_0 (or vice versa)."};

    const auto &position_accessor = asset.accessors[position_it->accessorIndex];
    if (position_accessor.count == 0) return {};

    const uint32_t base_vertex = mesh_data.Positions.size();
    mesh_data.Positions.resize(base_vertex + position_accessor.count);
    fastgltf::copyFromAccessor<vec3>(asset, position_accessor, &mesh_data.Positions[base_vertex]);

    // Parse NORMAL attribute
    const auto normal_it = primitive.findAttribute("NORMAL");
    const bool has_normals = normal_it != primitive.attributes.end();
    if (has_normals) {
        if (!mesh_data.Normals) {
            mesh_data.Normals.emplace();
            // Backfill zeros for vertices from earlier primitives that lacked normals
            mesh_data.Normals->resize(base_vertex, vec3{0.f});
        }
        const auto &normal_accessor = asset.accessors[normal_it->accessorIndex];
        mesh_data.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
        fastgltf::copyFromAccessor<vec3>(asset, normal_accessor, &(*mesh_data.Normals)[base_vertex]);
    } else if (mesh_data.Normals) {
        // Previous primitives had normals but this one doesn't — pad with zeros
        mesh_data.Normals->resize(base_vertex + position_accessor.count, vec3{0.f});
    }

    if (has_joints && !deform_data) deform_data.emplace();
    const bool mesh_has_skin_data = deform_data && (!deform_data->Joints.empty() || !deform_data->Weights.empty());
    if (mesh_has_skin_data &&
        (deform_data->Joints.size() != base_vertex ||
         deform_data->Weights.size() != base_vertex)) {
        return std::unexpected{"glTF primitive append encountered inconsistent skin channel sizes."};
    }

    if (has_joints || mesh_has_skin_data) {
        deform_data->Joints.resize(mesh_data.Positions.size(), uvec4{0});
        deform_data->Weights.resize(mesh_data.Positions.size(), vec4{0});
    }

    if (has_joints) {
        // Collect and validate all skin influence accessor pairs (JOINTS_n/WEIGHTS_n).
        std::vector<std::pair<const fastgltf::Accessor *, const fastgltf::Accessor *>> influence_accessors;
        for (uint32_t set_index = 0;; ++set_index) {
            const auto j_name = std::format("JOINTS_{}", set_index);
            const auto w_name = std::format("WEIGHTS_{}", set_index);
            const auto j_it = primitive.findAttribute(j_name);
            const auto w_it = primitive.findAttribute(w_name);
            if (j_it == primitive.attributes.end() && w_it == primitive.attributes.end()) break;
            if ((j_it == primitive.attributes.end()) != (w_it == primitive.attributes.end())) {
                return std::unexpected{std::format("glTF primitive has {} without {} (or vice versa).", j_name, w_name)};
            }

            const auto &j_acc = asset.accessors[j_it->accessorIndex];
            const auto &w_acc = asset.accessors[w_it->accessorIndex];
            if (j_acc.count != position_accessor.count || w_acc.count != position_accessor.count) {
                return std::unexpected{std::format(
                    "glTF primitive skin attribute counts must match POSITION count (POSITION={}, {}={}, {}={}).",
                    position_accessor.count, j_name, j_acc.count, w_name, w_acc.count
                )};
            }
            influence_accessors.emplace_back(&j_acc, &w_acc);
        }

        if (influence_accessors.size() == 1) {
            fastgltf::copyFromAccessor<uvec4>(asset, *influence_accessors.front().first, &deform_data->Joints[base_vertex]);
            fastgltf::copyFromAccessor<vec4>(asset, *influence_accessors.front().second, &deform_data->Weights[base_vertex]);
        } else {
            struct InfluenceSet {
                std::vector<uvec4> Joints;
                std::vector<vec4> Weights;
            };
            std::vector<InfluenceSet> sets(influence_accessors.size());
            for (std::size_t set_index = 0; set_index < sets.size(); ++set_index) {
                auto &s = sets[set_index];
                const auto [j_acc, w_acc] = influence_accessors[set_index];
                s.Joints.resize(position_accessor.count);
                fastgltf::copyFromAccessor<uvec4>(asset, *j_acc, s.Joints.data());
                s.Weights.resize(position_accessor.count);
                fastgltf::copyFromAccessor<vec4>(asset, *w_acc, s.Weights.data());
            }

            // Multiple influence sets: merge all, keep top 4 by weight, renormalize.
            // glTF 2.0 §3.7.3.1: implementations MAY support only 4 influences.
            // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skinned-mesh-attributes
            const auto total_influences = sets.size() * 4u;
            std::vector<std::pair<uint32_t, float>> all(total_influences);
            const auto top4_end = all.begin() + 4;
            const auto by_weight = [](const auto &a, const auto &b) { return a.second > b.second; };
            for (std::size_t i = 0; i < position_accessor.count; ++i) {
                uint32_t n = 0;
                for (const auto &s : sets) {
                    const auto &j = s.Joints[i];
                    const auto &w = s.Weights[i];
                    all[n++] = {j.x, w.x};
                    all[n++] = {j.y, w.y};
                    all[n++] = {j.z, w.z};
                    all[n++] = {j.w, w.w};
                }
                std::partial_sort(all.begin(), top4_end, all.begin() + n, by_weight);

                const float sum = all[0].second + all[1].second + all[2].second + all[3].second;
                const float inv = sum > 0.f ? 1.f / sum : 0.f;
                deform_data->Joints[base_vertex + i] = {all[0].first, all[1].first, all[2].first, all[3].first};
                deform_data->Weights[base_vertex + i] = {all[0].second * inv, all[1].second * inv, all[2].second * inv, all[3].second * inv};
            }
        }
    }

    // Parse morph targets (blend shapes) — append per-vertex deltas for this primitive
    if (!primitive.targets.empty()) {
        const uint32_t target_count = primitive.targets.size();
        const uint32_t prim_vertex_count = position_accessor.count;
        if (!morph_data) {
            morph_data.emplace();
            morph_data->TargetCount = target_count;
            // Backfill zeros for any vertices from earlier primitives
            morph_data->PositionDeltas.resize(std::size_t(target_count) * base_vertex, vec3{0.f});
        }
        if (morph_data->TargetCount != target_count) return std::unexpected{"glTF primitive morph target count mismatch between primitives of the same mesh."};

        // Append this primitive's deltas for each target (interleaved: all targets for this prim appended together)
        const auto prev_pos_size = morph_data->PositionDeltas.size();
        morph_data->PositionDeltas.resize(prev_pos_size + std::size_t(target_count) * prim_vertex_count, vec3{0.f});
        // Track whether any target in this primitive has normal deltas
        bool prim_has_normal_deltas = false;
        for (uint32_t t = 0; t < target_count; ++t) {
            if (primitive.findTargetAttribute(t, "NORMAL") != primitive.targets[t].end()) {
                prim_has_normal_deltas = true;
                break;
            }
        }
        if (prim_has_normal_deltas && morph_data->NormalDeltas.empty() && prev_pos_size > 0) {
            // Backfill zeros for position deltas from earlier primitives that had no normal deltas
            morph_data->NormalDeltas.resize(prev_pos_size, vec3{0.f});
        }
        if (prim_has_normal_deltas || !morph_data->NormalDeltas.empty()) {
            const auto prev_norm_size = morph_data->NormalDeltas.size();
            morph_data->NormalDeltas.resize(prev_norm_size + std::size_t(target_count) * prim_vertex_count, vec3{0.f});
        }
        for (uint32_t t = 0; t < target_count; ++t) {
            auto pos_it = primitive.findTargetAttribute(t, "POSITION");
            if (pos_it != primitive.targets[t].end()) {
                const auto &target_accessor = asset.accessors[pos_it->accessorIndex];
                if (target_accessor.count == prim_vertex_count) {
                    fastgltf::copyFromAccessor<vec3>(asset, target_accessor, &morph_data->PositionDeltas[prev_pos_size + std::size_t(t) * prim_vertex_count]);
                }
            }
            if (!morph_data->NormalDeltas.empty()) {
                auto norm_it = primitive.findTargetAttribute(t, "NORMAL");
                if (norm_it != primitive.targets[t].end()) {
                    const auto &norm_accessor = asset.accessors[norm_it->accessorIndex];
                    const auto prev_norm_size = morph_data->NormalDeltas.size() - std::size_t(target_count) * prim_vertex_count;
                    if (norm_accessor.count == prim_vertex_count) {
                        fastgltf::copyFromAccessor<vec3>(asset, norm_accessor, &morph_data->NormalDeltas[prev_norm_size + std::size_t(t) * prim_vertex_count]);
                    }
                }
            }
        }
    } else if (morph_data) {
        // Previous primitives had targets but this one doesn't — pad with zeros
        const uint32_t prim_vertex_count = position_accessor.count;
        morph_data->PositionDeltas.resize(morph_data->PositionDeltas.size() + std::size_t(morph_data->TargetCount) * prim_vertex_count, vec3{0.f});
        if (!morph_data->NormalDeltas.empty()) {
            morph_data->NormalDeltas.resize(morph_data->NormalDeltas.size() + std::size_t(morph_data->TargetCount) * prim_vertex_count, vec3{0.f});
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
    if (indices.size() < 3) return {};

    if (primitive.type == fastgltf::PrimitiveType::TriangleStrip) {
        for (uint32_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) {
                mesh_data.Faces.push_back({base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]});
            } else {
                mesh_data.Faces.push_back({base_vertex + indices[i + 1], base_vertex + indices[i], base_vertex + indices[i + 2]});
            }
        }
    } else if (primitive.type == fastgltf::PrimitiveType::TriangleFan) {
        for (uint32_t i = 1; i + 1 < indices.size(); ++i) {
            mesh_data.Faces.push_back({base_vertex + indices[0], base_vertex + indices[i], base_vertex + indices[i + 1]});
        }
    } else {
        for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
            mesh_data.Faces.push_back({base_vertex + indices[i], base_vertex + indices[i + 1], base_vertex + indices[i + 2]});
        }
    }
    return {};
}

std::expected<fastgltf::Asset, std::string> ParseAsset(const std::filesystem::path &path) {
    auto gltf_file = fastgltf::MappedGltfFile::FromPath(path);
    if (gltf_file.error() != fastgltf::Error::None) return std::unexpected{std::format("Failed to open glTF file '{}': {}", path.string(), fastgltf::getErrorMessage(gltf_file.error()))};

    fastgltf::Parser parser{fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::EXT_mesh_gpu_instancing | fastgltf::Extensions::KHR_lights_punctual};
    using fastgltf::Options;
    auto parsed = parser.loadGltf(gltf_file.get(), path.parent_path(), Options::DontRequireValidAssetMember | Options::AllowDouble | Options::LoadExternalBuffers | Options::GenerateMeshIndices | Options::DecomposeNodeMatrices);
    if (parsed.error() != fastgltf::Error::None) return std::unexpected{std::format("Failed to parse glTF '{}': {}", path.string(), fastgltf::getErrorMessage(parsed.error()))};

    return std::move(parsed.get());
}

std::expected<std::optional<uint32_t>, std::string> EnsureMeshData(const fastgltf::Asset &asset, uint32_t source_mesh_index, SceneData &scene_data, std::unordered_map<uint32_t, std::optional<uint32_t>> &mesh_index_map) {
    if (const auto it = mesh_index_map.find(source_mesh_index); it != mesh_index_map.end()) return it->second;

    const auto &source_mesh = asset.meshes[source_mesh_index];
    MeshData mesh_data;
    std::optional<ArmatureDeformData> mesh_deform_data;
    std::optional<MorphTargetData> mesh_morph_data;
    MeshData lines_data, points_data; // Merged across all line/point primitives
    // Track per-primitive vertex counts for morph target re-packing
    std::vector<uint32_t> prim_vertex_counts;
    for (const auto &primitive : source_mesh.primitives) {
        if (primitive.type == fastgltf::PrimitiveType::Points) {
            AppendNonTrianglePrimitive(asset, primitive, points_data);
            continue;
        }
        if (primitive.type == fastgltf::PrimitiveType::Lines || primitive.type == fastgltf::PrimitiveType::LineStrip || primitive.type == fastgltf::PrimitiveType::LineLoop) {
            AppendNonTrianglePrimitive(asset, primitive, lines_data);
            continue;
        }
        const uint32_t prev_vertex_count = mesh_data.Positions.size();
        if (auto append_result = AppendPrimitive(asset, primitive, mesh_data, mesh_deform_data, mesh_morph_data); !append_result) {
            return std::unexpected{std::move(append_result.error())};
        }
        prim_vertex_counts.emplace_back(mesh_data.Positions.size() - prev_vertex_count);
    }
    const bool has_triangle_data = !mesh_data.Positions.empty() && !mesh_data.Faces.empty();
    const bool has_lines = !lines_data.Positions.empty();
    const bool has_points = !points_data.Positions.empty();
    if (!has_triangle_data && !has_lines && !has_points) {
        mesh_index_map.emplace(source_mesh_index, std::nullopt);
        return std::optional<uint32_t>{};
    }

    // Re-pack morph target deltas from per-primitive chunks to per-target contiguous layout
    // Input layout:  [prim0: t0_verts, t1_verts, ...], [prim1: t0_verts, t1_verts, ...], ...
    // Output layout: [t0: all_verts], [t1: all_verts], ...
    if (mesh_morph_data && mesh_morph_data->TargetCount > 0 && prim_vertex_counts.size() > 1) {
        const uint32_t total_verts = mesh_data.Positions.size();
        const auto target_count = mesh_morph_data->TargetCount;
        std::vector<vec3> repacked(std::size_t(target_count) * total_verts, vec3{0.f});

        uint32_t src_offset = 0, dst_vert_offset = 0;
        for (const auto prim_verts : prim_vertex_counts) {
            for (uint32_t t = 0; t < target_count; ++t) {
                for (uint32_t v = 0; v < prim_verts; ++v) {
                    repacked[std::size_t(t) * total_verts + dst_vert_offset + v] = mesh_morph_data->PositionDeltas[src_offset + std::size_t(t) * prim_verts + v];
                }
            }
            src_offset += target_count * prim_verts;
            dst_vert_offset += prim_verts;
        }
        mesh_morph_data->PositionDeltas = std::move(repacked);

        if (!mesh_morph_data->NormalDeltas.empty()) {
            std::vector<vec3> repacked_normals(std::size_t(target_count) * total_verts, vec3{0.f});
            uint32_t norm_src_offset = 0, norm_dst_vert_offset = 0;
            for (const auto prim_verts : prim_vertex_counts) {
                for (uint32_t t = 0; t < target_count; ++t) {
                    for (uint32_t v = 0; v < prim_verts; ++v) {
                        repacked_normals[std::size_t(t) * total_verts + norm_dst_vert_offset + v] = mesh_morph_data->NormalDeltas[norm_src_offset + std::size_t(t) * prim_verts + v];
                    }
                }
                norm_src_offset += target_count * prim_verts;
                norm_dst_vert_offset += prim_verts;
            }
            mesh_morph_data->NormalDeltas = std::move(repacked_normals);
        }
    }

    // Read default morph target weights from mesh
    if (mesh_morph_data && !source_mesh.weights.empty()) {
        mesh_morph_data->DefaultWeights.resize(mesh_morph_data->TargetCount, 0.f);
        const auto copy_count = std::min(source_mesh.weights.size(), std::size_t(mesh_morph_data->TargetCount));
        std::copy_n(source_mesh.weights.begin(), copy_count, mesh_morph_data->DefaultWeights.begin());
    } else if (mesh_morph_data) {
        mesh_morph_data->DefaultWeights.assign(mesh_morph_data->TargetCount, 0.f);
    }

    const auto mesh_index = scene_data.Meshes.size();
    scene_data.Meshes.emplace_back(
        SceneMeshData{
            .Triangles = has_triangle_data ? std::optional{std::move(mesh_data)} : std::nullopt,
            .Lines = has_lines ? std::optional{std::move(lines_data)} : std::nullopt,
            .Points = has_points ? std::optional{std::move(points_data)} : std::nullopt,
            .DeformData = std::move(mesh_deform_data),
            .MorphData = std::move(mesh_morph_data),
            .Name = source_mesh.name.empty() ? std::format("Mesh{}", source_mesh_index) : std::string(source_mesh.name),
        }
    );
    mesh_index_map.emplace(source_mesh_index, mesh_index);
    return std::optional<uint32_t>{mesh_index};
}

std::vector<std::optional<uint32_t>> BuildNodeParentTable(const fastgltf::Asset &asset) {
    std::vector<std::optional<uint32_t>> parents(asset.nodes.size(), std::nullopt);
    for (std::size_t parent_idx = 0; parent_idx < asset.nodes.size(); ++parent_idx) {
        const uint32_t parent = parent_idx;
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

// DecomposeNodeMatrices guarantees TRS, so we can compose world transforms in glm space directly.
SceneTraversalData TraverseSceneNodes(const fastgltf::Asset &asset, const std::vector<Transform> &local_transforms, uint32_t scene_index) {
    const auto nodes_count = asset.nodes.size();
    SceneTraversalData traversal{.InScene = std::vector(nodes_count, false), .WorldTransforms = std::vector(nodes_count, I4)};
    if (scene_index >= asset.scenes.size()) return traversal;

    const auto &scene = asset.scenes[scene_index];
    const auto traverse =
        [&](uint32_t node_index, const mat4 &parent_world, const auto &self) {
            if (node_index >= nodes_count) return;

            const auto &node = asset.nodes[node_index];
            const auto world = parent_world * ToMatrix(local_transforms[node_index]);
            traversal.InScene[node_index] = true;
            traversal.WorldTransforms[node_index] = world;
            for (const auto child_idx : node.children) {
                if (const auto child = ToIndex(child_idx, nodes_count)) self(*child, world, self);
            }
        };
    for (const auto root_idx : scene.nodeIndices) {
        if (const auto root = ToIndex(root_idx, nodes_count)) traverse(*root, I4, traverse);
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
        const auto common_count = std::min(common_path.size(), path.size());
        std::size_t prefix = 0;
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
            return std::unexpected{std::format("glTF skin {} joint node {} cannot be rebased to ancestor node {}.", skin_index, joint_node_index, *rebased_parent_node_index)};
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

    const auto emit_joint = [&](uint32_t joint_node_index, const auto &self) -> std::expected<void, std::string> {
        const auto current_state = state[joint_node_index];
        if (current_state == 2) return {};
        if (current_state == 1) return std::unexpected{std::format("glTF skin {} has a cycle in joint ancestry at node {}.", skin_index, joint_node_index)};

        state[joint_node_index] = 1;
        if (const auto it = joint_parent_map.find(joint_node_index);
            it != joint_parent_map.end() && it->second) {
            if (auto parent_result = self(*it->second, self); !parent_result) return parent_result;
        }

        state[joint_node_index] = 2;
        ordered.emplace_back(joint_node_index);
        return {};
    };

    for (const auto joint_node_index : source_joint_nodes) {
        if (auto result = emit_joint(joint_node_index, emit_joint); !result) return std::unexpected{std::move(result.error())};
    }

    return ordered;
}

std::vector<mat4> LoadInverseBindMatrices(const fastgltf::Asset &asset, const fastgltf::Skin &skin, uint32_t joint_count) {
    std::vector<mat4> inverse_bind_matrices(joint_count, I4);
    if (!skin.inverseBindMatrices || *skin.inverseBindMatrices >= asset.accessors.size()) return inverse_bind_matrices;

    const auto &accessor = asset.accessors[*skin.inverseBindMatrices];
    if (accessor.type != fastgltf::AccessorType::Mat4 || accessor.count == 0) return inverse_bind_matrices;
    if (accessor.count <= joint_count) {
        fastgltf::copyFromAccessor<mat4>(asset, accessor, inverse_bind_matrices.data());
    } else {
        std::vector<mat4> ibm(accessor.count);
        fastgltf::copyFromAccessor<mat4>(asset, accessor, ibm.data());
        std::copy_n(ibm.begin(), joint_count, inverse_bind_matrices.begin());
    }
    return inverse_bind_matrices;
}

std::string MakeNodeName(const fastgltf::Asset &asset, uint32_t node_index, std::optional<uint32_t> source_mesh_index = {}) {
    const auto &node = asset.nodes[node_index];
    if (!node.name.empty()) return std::string(node.name);

    if (source_mesh_index && *source_mesh_index < asset.meshes.size()) {
        const auto &mesh_name = asset.meshes[*source_mesh_index].name;
        if (!mesh_name.empty()) return std::string(mesh_name);
    }

    return std::format("Node{}", node_index);
}
std::vector<Transform> ReadInstanceTransforms(const fastgltf::Asset &asset, const fastgltf::Node &node) {
    if (node.instancingAttributes.empty()) return {};

    const auto t_attr = node.findInstancingAttribute("TRANSLATION");
    const auto r_attr = node.findInstancingAttribute("ROTATION");
    const auto s_attr = node.findInstancingAttribute("SCALE");

    // Determine instance count from the first present accessor
    const uint32_t instance_count = t_attr != node.instancingAttributes.end() ? asset.accessors[t_attr->accessorIndex].count :
        r_attr != node.instancingAttributes.end()                             ? asset.accessors[r_attr->accessorIndex].count :
        s_attr != node.instancingAttributes.end()                             ? asset.accessors[s_attr->accessorIndex].count :
                                                                                0;
    if (instance_count == 0) return {};

    std::vector<Transform> transforms(instance_count);
    if (t_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[t_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec3>(asset, accessor, [&](const vec3 &v, auto i) { transforms[i].P = v; });
    }
    if (r_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[r_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec4>(asset, accessor, [&](const vec4 &v, auto i) {
            transforms[i].R = glm::normalize(quat{v.w, v.x, v.y, v.z});
        });
    }
    if (s_attr != node.instancingAttributes.end()) {
        const auto &accessor = asset.accessors[s_attr->accessorIndex];
        fastgltf::iterateAccessorWithIndex<vec3>(asset, accessor, [&](const vec3 &v, auto i) { transforms[i].S = v; });
    }

    return transforms;
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
    std::vector<Transform> local_transforms(asset.nodes.size());
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        // DecomposeNodeMatrices guarantees node.transform is always TRS.
        local_transforms[node_index] = ToTransform(std::get<fastgltf::TRS>(asset.nodes[node_index].transform));
    }
    const auto traversal = TraverseSceneNodes(asset, local_transforms, uint32_t(scene_index));

    std::vector<bool> used_skin(asset.skins.size(), false);
    for (uint32_t node_index = 0; node_index < asset.nodes.size(); ++node_index) {
        if (!traversal.InScene[node_index]) continue;
        if (const auto skin_index = ToIndex(asset.nodes[node_index].skinIndex, asset.skins.size())) used_skin[*skin_index] = true;
    }

    // Parse cameras
    scene_data.Cameras.reserve(asset.cameras.size());
    for (const auto &cam : asset.cameras) {
        auto projection = std::visit(
            [](const auto &source) -> CameraData {
                using Projection = std::decay_t<decltype(source)>;
                if constexpr (std::is_same_v<Projection, fastgltf::Camera::Perspective>) {
                    return Perspective{.FieldOfViewRad = source.yfov, .FarClip = source.zfar, .NearClip = source.znear, .AspectRatio = source.aspectRatio};
                } else {
                    return Orthographic{.Mag = {source.xmag, source.ymag}, .FarClip = source.zfar, .NearClip = source.znear};
                }
            },
            cam.camera
        );
        scene_data.Cameras.emplace_back(CameraData{std::move(projection)}, std::string{cam.name});
    }

    // Parse KHR_lights_punctual lights
    scene_data.Lights.reserve(asset.lights.size());
    for (const auto &light : asset.lights) {
        PunctualLight punctual_light{
            .Direction = {0.f, 0.f, -1.f},
            .Range = 0.f,
            .Color = {light.color.x(), light.color.y(), light.color.z()},
            .Intensity = light.intensity,
            .Position = {0.f, 0.f, 0.f},
            .InnerConeCos = 0.f,
            .OuterConeCos = 0.f,
            .Type = LightTypePoint,
        };
        switch (light.type) {
            case fastgltf::LightType::Directional:
                punctual_light.Type = LightTypeDirectional;
                break;
            case fastgltf::LightType::Point:
                punctual_light.Type = LightTypePoint;
                punctual_light.Range = light.range ? *light.range : 0.f;
                break;
            case fastgltf::LightType::Spot: {
                const float outer = light.outerConeAngle ? *light.outerConeAngle : std::numbers::pi_v<float> / 4.f;
                const float inner = std::clamp(light.innerConeAngle ? *light.innerConeAngle : 0.f, 0.f, outer);
                punctual_light.Type = LightTypeSpot;
                punctual_light.Range = light.range ? *light.range : 0.f;
                punctual_light.InnerConeCos = std::cos(inner);
                punctual_light.OuterConeCos = std::cos(outer);
                break;
            }
        }
        scene_data.Lights.emplace_back(punctual_light, std::string{light.name});
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
        const auto source_mesh_index = ToIndex(source_node.meshIndex, asset.meshes.size());
        auto mesh_index = std::optional<uint32_t>{};
        if (traversal.InScene[node_index] && source_mesh_index) {
            auto ensured_mesh = EnsureMeshData(asset, *source_mesh_index, scene_data, mesh_index_map);
            if (!ensured_mesh) return std::unexpected{std::move(ensured_mesh.error())};
            mesh_index = *ensured_mesh;
        }
        std::vector<uint32_t> children_node_indices;
        children_node_indices.reserve(source_node.children.size());
        for (const auto child_idx : source_node.children) {
            if (const auto child = ToIndex(child_idx, asset.nodes.size())) children_node_indices.emplace_back(*child);
        }
        scene_data.Nodes[node_index] = SceneNodeData{
            .NodeIndex = node_index,
            .ParentNodeIndex = parents[node_index],
            .ChildrenNodeIndices = std::move(children_node_indices),
            .LocalTransform = local_transforms[node_index],
            .WorldTransform = traversal.InScene[node_index] ? traversal.WorldTransforms[node_index] : I4,
            .InScene = traversal.InScene[node_index],
            .IsJoint = is_joint[node_index],
            .MeshIndex = mesh_index,
            .SkinIndex = ToIndex(source_node.skinIndex, asset.skins.size()),
            .CameraIndex = ToIndex(source_node.cameraIndex, asset.cameras.size()),
            .LightIndex = ToIndex(source_node.lightIndex, asset.lights.size()),
            .Name = MakeNodeName(asset, node_index, source_mesh_index),
        };
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

    // Precompute per-node instance transforms for EXT_mesh_gpu_instancing
    std::vector<std::vector<Transform>> node_instance_transforms(asset.nodes.size());
    for (uint32_t ni = 0; ni < asset.nodes.size(); ++ni) {
        if (traversal.InScene[ni]) node_instance_transforms[ni] = ReadInstanceTransforms(asset, asset.nodes[ni]);
    }

    for (uint32_t node_index = 0; node_index < scene_data.Nodes.size(); ++node_index) {
        if (const auto &node = scene_data.Nodes[node_index]; is_object_emitted[node_index]) {
            const auto source_mesh_index = ToIndex(asset.nodes[node_index].meshIndex, asset.meshes.size());
            if (const auto &instance_transforms = node_instance_transforms[node_index];
                !instance_transforms.empty() && node.MeshIndex) {
                // EXT_mesh_gpu_instancing: emit one object per instance with baked world transform
                const auto base_name = MakeNodeName(asset, node.NodeIndex, source_mesh_index);
                const auto &source_weights = asset.nodes[node_index].weights;
                auto node_weights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())};
                for (uint32_t i = 0; i < instance_transforms.size(); ++i) {
                    scene_data.Objects.emplace_back(
                        SceneObjectData{
                            .ObjectType = SceneObjectData::Type::Mesh,
                            .NodeIndex = node.NodeIndex,
                            .ParentNodeIndex = std::nullopt,
                            .WorldTransform = node.WorldTransform * ToMatrix(instance_transforms[i]),
                            .MeshIndex = node.MeshIndex,
                            .SkinIndex = node.SkinIndex,
                            .NodeWeights = node_weights,
                            .Name = base_name + "." + std::to_string(i),
                        }
                    );
                }
            } else {
                const auto &source_weights = asset.nodes[node_index].weights;
                const auto object_type = node.MeshIndex ? SceneObjectData::Type::Mesh :
                    node.CameraIndex                    ? SceneObjectData::Type::Camera :
                    node.LightIndex                     ? SceneObjectData::Type::Light :
                                                          SceneObjectData::Type::Empty;
                scene_data.Objects.emplace_back(
                    SceneObjectData{
                        .ObjectType = object_type,
                        .NodeIndex = node.NodeIndex,
                        .ParentNodeIndex = nearest_object_ancestor[node.NodeIndex],
                        .WorldTransform = node.WorldTransform,
                        .MeshIndex = node.MeshIndex,
                        .SkinIndex = node.SkinIndex,
                        .CameraIndex = node.CameraIndex,
                        .LightIndex = node.LightIndex,
                        .NodeWeights = source_weights.empty() ? std::optional<std::vector<float>>{} : std::optional{std::vector<float>(source_weights.begin(), source_weights.end())},
                        .Name = MakeNodeName(asset, node.NodeIndex, source_mesh_index),
                    }
                );
            }
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
            auto rest_local = ComputeJointRestLocal(skin_index, joint_node_index, parent_joint_node_index, scene_skin.AnchorNodeIndex, parents, local_transforms);
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
        scene_skin.InverseBindMatrices = LoadInverseBindMatrices(asset, skin, uint32_t(scene_skin.Joints.size()));
        if (scene_skin.AnchorNodeIndex && traversal.InScene[*scene_skin.AnchorNodeIndex]) {
            scene_skin.ParentObjectNodeIndex = nearest_object_ancestor[*scene_skin.AnchorNodeIndex];
        }
        scene_data.Skins.emplace_back(std::move(scene_skin));
    }

    // Parse animations
    for (uint32_t anim_index = 0; anim_index < asset.animations.size(); ++anim_index) {
        const auto &anim = asset.animations[anim_index];
        AnimationClipData clip{.Name = anim.name.empty() ? std::format("Animation{}", anim_index) : std::string(anim.name), .Channels = {}};
        float max_time = 0;
        struct ChannelTargetSpec {
            AnimationPath Path;
            std::size_t ComponentCount;
        };
        for (const auto &channel : anim.channels) {
            if (!channel.nodeIndex || *channel.nodeIndex >= asset.nodes.size()) continue;
            if (channel.samplerIndex >= anim.samplers.size()) continue;

            const auto target_spec = [&]() -> std::optional<ChannelTargetSpec> {
                switch (channel.path) {
                    case fastgltf::AnimationPath::Translation: return ChannelTargetSpec{.Path = AnimationPath::Translation, .ComponentCount = 3};
                    case fastgltf::AnimationPath::Rotation: return ChannelTargetSpec{.Path = AnimationPath::Rotation, .ComponentCount = 4};
                    case fastgltf::AnimationPath::Scale: return ChannelTargetSpec{.Path = AnimationPath::Scale, .ComponentCount = 3};
                    case fastgltf::AnimationPath::Weights: {
                        // Look up morph target count from the target node's mesh
                        const auto &target_node = asset.nodes[*channel.nodeIndex];
                        if (!target_node.meshIndex || *target_node.meshIndex >= asset.meshes.size()) return std::nullopt;
                        const auto component_count = asset.meshes[*target_node.meshIndex].primitives.empty() ? 0 : asset.meshes[*target_node.meshIndex].primitives[0].targets.size();
                        if (component_count == 0) return std::nullopt;
                        return ChannelTargetSpec{.Path = AnimationPath::Weights, .ComponentCount = component_count};
                    }
                }
                return std::nullopt;
            }();
            if (!target_spec) continue;

            const auto &sampler = anim.samplers[channel.samplerIndex];
            if (sampler.inputAccessor >= asset.accessors.size() || sampler.outputAccessor >= asset.accessors.size()) continue;

            const auto &input_accessor = asset.accessors[sampler.inputAccessor];
            const auto &output_accessor = asset.accessors[sampler.outputAccessor];
            if (input_accessor.count == 0) continue;

            const auto interp = [&]() {
                switch (sampler.interpolation) {
                    case fastgltf::AnimationInterpolation::Step: return AnimationInterpolation::Step;
                    case fastgltf::AnimationInterpolation::Linear: return AnimationInterpolation::Linear;
                    case fastgltf::AnimationInterpolation::CubicSpline: return AnimationInterpolation::CubicSpline;
                }
                return AnimationInterpolation::Linear;
            }();

            std::vector<float> times(input_accessor.count);
            fastgltf::copyFromAccessor<float>(asset, input_accessor, times.data());

            std::vector<float> values;
            if (target_spec->Path == AnimationPath::Weights) {
                // Weights: output accessor has keyframe_count * target_count scalar values
                values.resize(output_accessor.count);
                fastgltf::copyFromAccessor<float>(asset, output_accessor, values.data());
            } else {
                values.resize(output_accessor.count * target_spec->ComponentCount);
                if (target_spec->ComponentCount == 4) fastgltf::copyFromAccessor<vec4>(asset, output_accessor, reinterpret_cast<vec4 *>(values.data()));
                else fastgltf::copyFromAccessor<vec3>(asset, output_accessor, reinterpret_cast<vec3 *>(values.data()));
            }

            if (!times.empty()) max_time = std::max(max_time, times.back());

            clip.Channels.emplace_back(AnimationChannelData{
                .TargetNodeIndex = uint32_t(*channel.nodeIndex),
                .Target = target_spec->Path,
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
