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
#include "Image.h"
#include "Transform.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"
#include "numeric/vec3.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace gltf {
enum class Filter : uint16_t {
    Nearest,
    Linear,
    NearestMipMapNearest,
    LinearMipMapNearest,
    NearestMipMapLinear,
    LinearMipMapLinear,
};
enum class Wrap : uint16_t {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
};

using Sheen = ::Sheen;
using Specular = ::Specular;
using Transmission = ::Transmission;
using Volume = ::Volume;
using Clearcoat = ::Clearcoat;
using Anisotropy = ::Anisotropy;
using Iridescence = ::Iridescence;
using PBRMaterial = ::PBRMaterial;

struct NamedMaterial {
    PBRMaterial Value{};
    std::string Name;
};

struct Texture {
    std::optional<uint32_t> SamplerIndex; // Index into `Scene::Samplers`
    std::optional<uint32_t> ImageIndex, WebpImageIndex, BasisuImageIndex, DdsImageIndex; // Indexes into `Scene::Images` in resolution order.
    std::string Name;
};

struct Sampler {
    std::optional<Filter> MagFilter, MinFilter;
    Wrap WrapS, WrapT;
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

struct Node {
    uint32_t NodeIndex;
    std::optional<uint32_t> ParentNodeIndex;
    std::vector<uint32_t> ChildrenNodeIndices;
    Transform LocalTransform, WorldTransform;
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
    std::vector<NamedMaterial> Materials;
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
