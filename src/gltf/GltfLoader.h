// TODO:
//  - Sparse accessors
//  - Draco mesh compression

// TODO: After adding materials/textures: Tangents import + morph target tangent deltas

#pragma once

#include "Armature.h"
#include "Transform.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
struct SceneMeshData {
    std::optional<MeshData> Triangles{}; // Merged triangle primitives (Triangles/TriangleStrip/TriangleFan)
    std::optional<ArmatureDeformData> DeformData{};
    std::optional<MorphTargetData> MorphData{};
    std::optional<MeshData> Lines{}; // Merged line primitives (Lines/LineStrip/LineLoop)
    std::optional<MeshData> Points{}; // Merged point primitives
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
    std::optional<uint32_t> MeshIndex{};
    std::optional<uint32_t> SkinIndex{};
    std::string Name;
};

struct SceneObjectData {
    enum class Type : uint8_t {
        Empty,
        Mesh,
    };

    Type ObjectType{Type::Empty};
    uint32_t NodeIndex{};
    std::optional<uint32_t> ParentNodeIndex{};
    mat4 WorldTransform{1.f};
    std::optional<uint32_t> MeshIndex{};
    std::optional<uint32_t> SkinIndex{};
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
    std::optional<uint32_t> SkeletonNodeIndex{};
    std::optional<uint32_t> AnchorNodeIndex{};
    std::optional<uint32_t> ParentObjectNodeIndex{};
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
};

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path);
} // namespace gltf
