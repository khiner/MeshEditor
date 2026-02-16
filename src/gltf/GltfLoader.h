// Note: Additional skin influence sets (JOINTS_1/WEIGHTS_1, etc.) are imported and compressed
// to the top 4 influences per vertex (sorted by weight, renormalized). The spec permits
// supporting only a single set of 4 (see glTF 2.0 §3.7.3.1); keeping the top 4 is strictly
// better. Lossy for round-trip export but visually near-lossless.
//
// TODO (glTF 2.0 coverage, non-material-first):
// - Support compressed geometry extensions (`EXT_meshopt_compression`, `KHR_draco_mesh_compression`):
//   minimal decode path: meshopt decode-only sources (`indexcodec.cpp`, `vertexcodec.cpp`, `vertexfilter.cpp`, plus `allocator.cpp` if needed);
//   draco decoder-only build (disable encoder/tools/tests, keep mesh decode path).
// - Preserve `EXT_mesh_gpu_instancing` custom instance attributes (e.g. `_ID`), not only TRS.
// - Add `KHR_animation_pointer` support (currently only core TRS/weights animation paths are handled).
// - After adding materials/textures: tangents import + morph target tangent deltas.

#pragma once

#include "Armature.h"
#include "CameraData.h"
#include "Transform.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
struct SceneMeshData {
    // Merged triangle/line/point primitives (Triangles/TriangleStrip/TriangleFan, Lines/LineStrip/LineLoop, Points)
    std::optional<MeshData> Triangles{}, Lines{}, Points{};
    std::optional<ArmatureDeformData> DeformData{};
    std::optional<MorphTargetData> MorphData{};
    std::string Name;
};

struct SceneCameraData {
    CameraData Camera;
    std::string Name;
};

struct SceneNodeData {
    uint32_t NodeIndex{};
    std::optional<uint32_t> ParentNodeIndex{};
    std::vector<uint32_t> ChildrenNodeIndices;
    Transform LocalTransform{};
    mat4 WorldTransform{1.f};
    bool InScene{false};
    bool IsJoint{false};
    std::optional<uint32_t> MeshIndex{}, SkinIndex{}, CameraIndex{};
    std::string Name;
};

struct SceneObjectData {
    enum class Type : uint8_t {
        Empty,
        Mesh,
        Camera,
    };

    Type ObjectType{Type::Empty};
    uint32_t NodeIndex{};
    std::optional<uint32_t> ParentNodeIndex{};
    mat4 WorldTransform{1.f};
    std::optional<uint32_t> MeshIndex{}, SkinIndex{}, CameraIndex{};
    std::optional<std::vector<float>> NodeWeights{}; // Per-node morph weight overrides (glTF node.weights)
    std::string Name;
};

struct SkinJointData {
    uint32_t JointNodeIndex{};
    std::optional<uint32_t> ParentJointNodeIndex{};
    Transform RestLocal{};
    std::string Name;
};

struct SceneSkinData {
    uint32_t SkinIndex{};
    std::string Name;
    std::optional<uint32_t> SkeletonNodeIndex{}, AnchorNodeIndex{}, ParentObjectNodeIndex{};
    std::vector<SkinJointData> Joints{}; // Parent-before-child order
    std::vector<mat4> InverseBindMatrices{}; // Order matches `Joints`
};

struct AnimationChannelData {
    uint32_t TargetNodeIndex;
    AnimationPath Target;
    AnimationInterpolation Interp{AnimationInterpolation::Linear};
    std::vector<float> TimesSeconds;
    std::vector<float> Values; // Packed: vec3 for T/S, vec4(xyzw) for R
};

struct AnimationClipData {
    std::string Name;
    float DurationSeconds{0};
    std::vector<AnimationChannelData> Channels;
};

struct SceneData {
    std::vector<SceneMeshData> Meshes;
    std::vector<SceneNodeData> Nodes;
    std::vector<SceneObjectData> Objects;
    std::vector<SceneSkinData> Skins;
    std::vector<AnimationClipData> Animations;
    std::vector<SceneCameraData> Cameras;
};

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path);
} // namespace gltf
