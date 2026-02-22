// Note: Additional skin influence sets (JOINTS_1/WEIGHTS_1, etc.) are imported and compressed
// to the top 4 influences per vertex (sorted by weight, renormalized).
// The spec permits supporting only a single set of 4 (see glTF 2.0 §3.7.3.1).
// Lossy for round-trip export but visually near-lossless.
//
// TODO (glTF 2.0 coverage):
// - After adding materials/textures: tangents import + morph target tangent deltas.
// - Add `KHR_animation_pointer` support (requires materials/lights; currently only core TRS/weights animation paths are handled).
// - Support compressed geometry extensions (`EXT_meshopt_compression`, `KHR_draco_mesh_compression`):
//   minimal decode path: meshopt decode-only sources (`indexcodec.cpp`, `vertexcodec.cpp`, `vertexfilter.cpp`, plus `allocator.cpp` if needed);
//   draco decoder-only build (disable encoder/tools/tests, keep mesh decode path).

#pragma once

#include "Armature.h"
#include "CameraData.h"
#include "Transform.h"
#include "gpu/PunctualLight.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <fastgltf/types.hpp>

#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
struct TextureTransformData {
    float Rotation;
    vec2 UvOffset, UvScale;
    std::optional<uint32_t> TexCoordIndex;
};

struct TextureInfoData {
    uint32_t TextureIndex;
    uint32_t TexCoordIndex;
    std::optional<TextureTransformData> Transform;
};

struct PBRData {
    vec4 BaseColorFactor;
    float MetallicFactor, RoughnessFactor;
    vec3 EmissiveFactor;
    float NormalScale;
    float OcclusionStrength;
    std::optional<TextureInfoData> BaseColorTexture{}, MetallicRoughnessTexture{};
    std::optional<TextureInfoData> NormalTexture{}, OcclusionTexture{}, EmissiveTexture{};
};

struct SceneMaterialData {
    PBRData PbrData;
    fastgltf::AlphaMode AlphaMode;
    bool DoubleSided;
    float AlphaCutoff;
    std::string Name;
};

struct SceneTextureData {
    std::optional<uint32_t> SamplerIndex, ImageIndex, BasisuImageIndex, DdsImageIndex, WebpImageIndex;
    std::string Name;
};

struct SceneImageData {
    std::vector<std::byte> Bytes;
    fastgltf::MimeType MimeType;
    std::string Name;
};

struct SceneSamplerData {
    std::optional<fastgltf::Filter> MagFilter, MinFilter;
    fastgltf::Wrap WrapS, WrapT;
    std::string Name;
};

struct SceneMeshData {
    // Merged triangle/line/point primitives (Triangles/TriangleStrip/TriangleFan, Lines/LineStrip/LineLoop, Points)
    std::optional<MeshData> Triangles, Lines, Points;
    std::optional<ArmatureDeformData> DeformData;
    std::optional<MorphTargetData> MorphData;
    std::string Name;
};

struct SceneCameraData {
    CameraData Camera;
    std::string Name;
};

struct SceneLightData {
    PunctualLight Light;
    std::string Name;
};

struct SceneNodeData {
    uint32_t NodeIndex;
    std::optional<uint32_t> ParentNodeIndex;
    std::vector<uint32_t> ChildrenNodeIndices;
    Transform LocalTransform;
    Transform WorldTransform;
    bool InScene;
    bool IsJoint;
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::string Name;
};

struct SceneObjectData {
    enum class Type : uint8_t {
        Empty,
        Mesh,
        Camera,
        Light,
    };

    Type ObjectType;
    uint32_t NodeIndex;
    std::optional<uint32_t> ParentNodeIndex;
    Transform WorldTransform;
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::optional<std::vector<float>> NodeWeights; // Per-node morph weight overrides (glTF node.weights)
    std::string Name;
};

struct SkinJointData {
    uint32_t JointNodeIndex;
    std::optional<uint32_t> ParentJointNodeIndex;
    Transform RestLocal;
    std::string Name;
};

struct SceneSkinData {
    uint32_t SkinIndex;
    std::string Name;
    std::optional<uint32_t> SkeletonNodeIndex, AnchorNodeIndex, ParentObjectNodeIndex{};
    std::vector<SkinJointData> Joints; // Parent-before-child order
    std::vector<mat4> InverseBindMatrices; // Order matches `Joints`
};

struct AnimationChannelData {
    uint32_t TargetNodeIndex;
    AnimationPath Target;
    AnimationInterpolation Interp;
    std::vector<float> TimesSeconds;
    std::vector<float> Values; // Packed: vec3 for T/S, vec4(xyzw) for R
};

struct AnimationClipData {
    std::string Name;
    float DurationSeconds;
    std::vector<AnimationChannelData> Channels;
};

struct SceneData {
    std::vector<SceneMeshData> Meshes;
    std::vector<SceneMaterialData> Materials;
    std::vector<SceneTextureData> Textures;
    std::vector<SceneImageData> Images;
    std::vector<SceneSamplerData> Samplers;
    std::vector<SceneNodeData> Nodes;
    std::vector<SceneObjectData> Objects;
    std::vector<SceneSkinData> Skins;
    std::vector<AnimationClipData> Animations;
    std::vector<SceneCameraData> Cameras;
    std::vector<SceneLightData> Lights;
};

std::expected<SceneData, std::string> LoadSceneData(const std::filesystem::path &path);
} // namespace gltf
