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
#include "Camera.h"
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

#include <array>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
struct TextureTransform {
    float Rotation;
    vec2 UvOffset, UvScale;
    std::optional<uint32_t> TexCoordIndex;
};

struct TextureInfo {
    uint32_t TextureIndex;
    uint32_t TexCoordIndex;
    std::optional<TextureTransform> Transform;
};

struct Sheen {
    vec3 ColorFactor{0.f, 0.f, 0.f};
    float RoughnessFactor{0.f};
    std::optional<TextureInfo> ColorTexture{}, RoughnessTexture{};
};
struct Specular {
    float Factor{1.f};
    vec3 ColorFactor{1.f, 1.f, 1.f};
    std::optional<TextureInfo> Texture{}, ColorTexture{};
};
struct Transmission {
    float Factor{0.f};
    std::optional<TextureInfo> Texture{};
};
struct Volume {
    float ThicknessFactor{0.f};
    vec3 AttenuationColor{1.f, 1.f, 1.f};
    float AttenuationDistance{0.f};
    std::optional<TextureInfo> ThicknessTexture{};
};
struct Clearcoat {
    float Factor{0.f};
    float RoughnessFactor{0.f};
    float NormalScale{1.f};
    std::optional<TextureInfo> Texture{}, RoughnessTexture{}, NormalTexture{};
};
struct Anisotropy {
    float Strength{0.f};
    float Rotation{0.f};
    std::optional<TextureInfo> Texture{};
};
struct Iridescence {
    float Factor{0.f};
    float Ior{1.3f};
    float ThicknessMinimum{100.f};
    float ThicknessMaximum{400.f};
    std::optional<TextureInfo> Texture{}, ThicknessTexture{};
};

struct PBRMaterial {
    vec4 BaseColorFactor;
    float MetallicFactor, RoughnessFactor;
    vec3 EmissiveFactor;
    float NormalScale;
    float OcclusionStrength;
    float Ior{1.5f};
    std::optional<TextureInfo> BaseColorTexture{}, MetallicRoughnessTexture{};
    std::optional<TextureInfo> NormalTexture{}, OcclusionTexture{}, EmissiveTexture{};
    Sheen Sheen;
    Specular Specular;
    Transmission Transmission;
    Volume Volume;
    Clearcoat Clearcoat;
    Anisotropy Anisotropy;
    Iridescence Iridescence;
    fastgltf::AlphaMode AlphaMode;
    bool DoubleSided, Unlit;
    float AlphaCutoff;
    std::string Name;
};

struct Texture {
    std::optional<uint32_t> SamplerIndex, ImageIndex, BasisuImageIndex, DdsImageIndex, WebpImageIndex;
    std::string Name;
};

struct Image {
    std::vector<std::byte> Bytes;
    fastgltf::MimeType MimeType;
    std::string Name;
};

struct Sampler {
    std::optional<fastgltf::Filter> MagFilter, MinFilter;
    fastgltf::Wrap WrapS, WrapT;
    std::string Name;
};

struct MeshData {
    // Merged triangle/line/point primitives (Triangles/TriangleStrip/TriangleFan, Lines/LineStrip/LineLoop, Points)
    std::optional<::MeshData> Triangles, Lines, Points;
    std::optional<ArmatureDeformData> DeformData;
    std::optional<MorphTargetData> MorphData;
    std::string Name;
};

struct Camera {
    ::Camera Camera;
    std::string Name;
};

struct Light {
    PunctualLight Light;
    std::string Name;
};

struct ImageBasedLight {
    std::vector<std::array<uint32_t, 6>> SpecularImageIndicesByMip; // Mip-major; face order: +X, -X, +Y, -Y, +Z, -Z.
    std::optional<std::array<vec3, 9>> IrradianceCoefficients; // L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22.
    float Intensity{1.f};
    std::string Name;
};

struct Node {
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

struct Object {
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

struct SkinJoint {
    uint32_t JointNodeIndex;
    std::optional<uint32_t> ParentJointNodeIndex;
    Transform RestLocal;
    std::string Name;
};

struct Skin {
    uint32_t SkinIndex;
    std::string Name;
    std::optional<uint32_t> SkeletonNodeIndex, AnchorNodeIndex, ParentObjectNodeIndex{};
    std::vector<SkinJoint> Joints; // Parent-before-child order
    std::vector<mat4> InverseBindMatrices; // Order matches `Joints`
};

struct AnimationChannel {
    uint32_t TargetNodeIndex;
    AnimationPath Target;
    AnimationInterpolation Interp;
    std::vector<float> TimesSeconds;
    std::vector<float> Values; // Packed: vec3 for T/S, vec4(xyzw) for R
};

struct AnimationClip {
    std::string Name;
    float DurationSeconds;
    std::vector<AnimationChannel> Channels;
};

struct Scene {
    std::vector<MeshData> Meshes;
    std::vector<PBRMaterial> Materials;
    std::vector<Texture> Textures;
    std::vector<Image> Images;
    std::vector<Sampler> Samplers;
    std::vector<Node> Nodes;
    std::vector<Object> Objects;
    std::vector<Skin> Skins;
    std::vector<AnimationClip> Animations;
    std::vector<Camera> Cameras;
    std::vector<Light> Lights;
    std::optional<ImageBasedLight> ImageBasedLight;
};

std::expected<Scene, std::string> LoadScene(const std::filesystem::path &);
} // namespace gltf
