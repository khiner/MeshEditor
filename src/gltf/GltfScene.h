// Round-trip gaps (see tests/RoundtripTest.cpp for the per-path exception list):
//
// Lossy:
// - Additional skin influence sets (JOINTS_1+, WEIGHTS_1+) are compressed at import to the top 4 weights per vertex (sorted, renormalized).
//  (The spec does permit this single set of 4 - see glTF 2.0 §3.7.3.1.)
// - KHR_mesh_quantization: quantized attributes decode to FLOAT at import, save always emits FLOAT
// - EXT_mesh_gpu_instancing: per-instance TRS round-trips, but custom instancing attributes (`_FOO`) beyond TRANSLATION/ROTATION/SCALE aren't retained
//
// Unsupported (neither imported nor re-emitted):
// - KHR_draco_mesh_compression, EXT_meshopt_compression: files relying on these to carry geometry
//   will load with empty/missing vertex data, or fail entirely if the extension is listed as required.
// - KHR_animation_pointer: animation channels targeting extension pointer paths are silently dropped at import (along with their samplers).
//
// Parsed for round-trip but not consumed by the application:
//   Fields and features carried on gltf::Scene only for LoadScene -> SaveScene to preserve source fidelity.
//   Runtime (SceneGltf.cpp -> ECS) ignores them.
//   When we implement ECS -> Save round-trip, these are what needs sidecar storage.
// - Scene-level metadata:
//   - Scene::Copyright, Generator, MinVersion: asset.* metadata
//   - Scene::AssetExtras, AssetExtensions: raw JSON for asset.extras / asset.extensions
//   - Scene::DefaultSceneName, DefaultSceneRoots: default scene name and root-node source order
//   - Scene::ExtensionsRequired: verbatim pass-through
//   - Scene::MaterialVariants + MeshPrimitives::VariantMappings: KHR_materials_variants data
//     Full feature foundation (just wire a UI + render-path lookup on top)
//   - Scene::ExtrasByEntity: per-entity `extras` raw JSON, keyed by (fastgltf::Category, index)
// - Per-entity encoding hints:
//   - Image::Uri, SourceDataUri, SourceHadMimeType: runtime uses Bytes + MimeType;
//     these let save re-emit in the source form (external URI, data URI, bufferView, with/without mimeType)
//   - Node::SourceMatrix: save-side preservation for matrix-form transforms; runtime uses LocalTransform
//   - MaterialData::{BaseColor,MetallicRoughness,Normal,Occlusion,Emissive}Meta: KHR_texture_transform
//     override state (empty-extension vs missing, texCoord override vs parent texCoord)
//   - MeshVertexAttributes::Colors0ComponentCount: source VEC3 vs VEC4 for COLOR_0.
//   - MeshPrimitives::AttributeFlags: per-source-primitive attribute presence bitmask;
//     emits each channel only on primitives that carried it in source (vs our zero-backfilled merged form)
//   - MeshPrimitives::HasSourceIndices: per-primitive "source had an indices accessor" flag;
//     lets non-indexed source round-trip as non-indexed
//   - MorphTargetData::TangentDeltas: imported and re-emitted but not applied in the shader (the
//     GPU bandwidth/ALU cost isn't justified yet)

#pragma once

#include "AnimationData.h"
#include "Camera.h"
#include "Image.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/Transform.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "physics/PhysicsTypes.h"

#include <expected>
#include <filesystem>
#include <unordered_map>

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

// TextureInfo.TexCoord is the effective value (override if present, else base).
struct TextureTransformMeta {
    bool SourceHadExtension{};
    uint32_t SourceBaseTexCoord{};
    std::optional<uint32_t> SourceTexCoordOverride{};
};

// CPU-side material view. Each KHR_materials_* block is std::optional, set iff the source
// carried the extension. Downstream code gets the flat GPU-bindable PBRMaterial via gltf::ToGpu().
struct MaterialData {
    vec4 BaseColorFactor{1};
    vec3 EmissiveFactor{0};
    float MetallicFactor{1}, RoughnessFactor{1}, NormalScale{1}, OcclusionStrength{1};
    MaterialAlphaMode AlphaMode{MaterialAlphaMode::Opaque};
    float AlphaCutoff{0.5};
    uint32_t DoubleSided{}, Unlit{};
    TextureInfo BaseColorTexture{}, MetallicRoughnessTexture{}, NormalTexture{}, OcclusionTexture{}, EmissiveTexture{};
    // Nested extension textures (Sheen.ColorTexture etc.) don't round-trip overrides today.
    TextureTransformMeta BaseColorMeta{}, MetallicRoughnessMeta{}, NormalMeta{}, OcclusionMeta{}, EmissiveMeta{};

    std::optional<float> Ior{}, Dispersion{}, EmissiveStrength{};

    std::optional<::Sheen> Sheen{};
    std::optional<::Specular> Specular{};
    std::optional<::Transmission> Transmission{};
    std::optional<::DiffuseTransmission> DiffuseTransmission{};
    std::optional<::Volume> Volume{};
    std::optional<::Clearcoat> Clearcoat{};
    std::optional<::Anisotropy> Anisotropy{};
    std::optional<::Iridescence> Iridescence{};
};

// An absent extension and an all-default extension render identically, so it's safe
// to flatten optionals to default-initialized fields when crossing into the GPU type.
PBRMaterial ToGpu(const MaterialData &);

struct NamedMaterial {
    MaterialData Value{};
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
    ::MeshVertexAttributes TriangleAttrs, LineAttrs, PointAttrs;
    ::MeshPrimitives TrianglePrimitives;
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
    bool InScene, IsJoint;
    std::optional<mat4> SourceMatrix{};
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::string Name;

    // KHR_physics_rigid_bodies per-node data (empty if no physics on this node).
    // These use the ECS component types directly where possible.
    // For fields that reference other glTF nodes (joint ConnectedNode, trigger Nodes),
    // we store node indices here and resolve to entities in SceneGltf.cpp.
    std::optional<PhysicsMotion> Motion{};
    std::optional<PhysicsVelocity> Velocity{};
    std::optional<ColliderShape> Collider{};
    // Loader-side index refs; SceneGltf.cpp resolves these to entity refs on ColliderMaterial.
    struct MaterialRefs {
        std::optional<uint32_t> PhysicsMaterialIndex{}, CollisionFilterIndex{};
    };
    std::optional<MaterialRefs> Material{};
    std::optional<uint32_t> ColliderGeometryMeshIndex{}; // glTF mesh providing geometry for collider shape

    struct TriggerData {
        std::optional<PhysicsShape> Shape{};
        std::optional<uint32_t> GeometryMeshIndex{}; // glTF mesh for ConvexHull/TriangleMesh trigger shapes
        std::vector<uint32_t> NodeIndices{}; // glTF node indices (resolved to entities in SceneGltf)
        std::optional<uint32_t> CollisionFilterIndex{};
    };
    std::optional<TriggerData> Trigger{};

    struct JointData {
        uint32_t ConnectedNodeIndex{}; // glTF node index (resolved to entity in SceneGltf)
        uint32_t JointDefIndex{};
        bool EnableCollision{false};
    };
    std::optional<JointData> Joint{};
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

// Loader-side representation of a collision filter; system names are raw strings
// (no entities exist at load time). SceneGltf.cpp converts these to CollisionSystem
// entities and CollisionFilter{CollideMode, entity refs} during import.
struct CollisionFilterData {
    std::vector<std::string> CollisionSystems{}, CollideWithSystems{}, NotCollideWithSystems{};
    std::string Name{};
};

struct Scene {
    std::string Copyright, Generator, MinVersion;
    std::string AssetExtras, AssetExtensions; // raw minified JSON
    std::string DefaultSceneName;
    std::vector<uint32_t> DefaultSceneRoots;
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

    // KHR_physics_rigid_bodies document-level resources
    std::vector<PhysicsMaterial> PhysicsMaterials;
    std::vector<CollisionFilterData> CollisionFilters;
    std::vector<PhysicsJointDef> PhysicsJointDefs;

    std::vector<std::string> ExtensionsRequired;
    std::vector<std::string> MaterialVariants;
    std::unordered_map<uint64_t, std::string> ExtrasByEntity; // Key: (fastgltf::Category << 32) | entity_index
};

std::expected<Scene, std::string> LoadScene(const std::filesystem::path &);
std::expected<void, std::string> SaveScene(const Scene &, const std::filesystem::path &);
} // namespace gltf
