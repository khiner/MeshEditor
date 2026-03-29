// Note: Additional skin influence sets (JOINTS_1/WEIGHTS_1, etc.) are imported and compressed
// to the top 4 influences per vertex (sorted by weight, renormalized).
// The spec permits supporting only a single set of 4 (see glTF 2.0 §3.7.3.1).
// Lossy for round-trip export but visually near-lossless.
// - Known gap: morph target tangent deltas (targets[*].TANGENT) are not imported/applied yet.
//   Static tangents (vertex TANGENT) are imported and used for shading, but
//   morphing is deferred for now due to added GPU morph bandwidth/ALU cost.
// - Also skipping `KHR_animation_pointer` for now - too much complexity for an extension that isn't widely used yet
// - Probably should suppor `EXT_meshopt_compression`, with a minimal decode path:
//   - meshopt decode-only sources (`indexcodec.cpp`, `vertexcodec.cpp`, `vertexfilter.cpp`, plus `allocator.cpp` if needed);

#pragma once

#include "AnimationData.h"
#include "Camera.h"
#include "Image.h"
#include "Transform.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "mesh/ArmatureDeformData.h"
#include "mesh/MeshAttributes.h"
#include "mesh/MeshData.h"
#include "mesh/MorphTargetData.h"
#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "physics/PhysicsTypes.h"

#include <expected>
#include <filesystem>

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
    ::MeshVertexAttributes TriangleAttrs;
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
    bool InScene;
    bool IsJoint;
    std::optional<uint32_t> MeshIndex, SkinIndex, CameraIndex, LightIndex;
    std::string Name;

    // KHR_physics_rigid_bodies per-node data (empty if no physics on this node).
    // These use the ECS component types directly where possible. For fields that reference
    // other glTF nodes (joint ConnectedNode, trigger Nodes), we store node indices here
    // and resolve to entities in SceneGltf.cpp.
    std::optional<PhysicsMotion> Motion{};
    std::optional<PhysicsCollider> Collider{};
    std::optional<uint32_t> ColliderGeometryNodeIndex{}; // glTF node providing mesh geometry for collider shape

    struct TriggerData {
        std::optional<PhysicsShape> Shape{};
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

    // KHR_physics_rigid_bodies document-level resources.
    std::vector<PhysicsMaterial> PhysicsMaterials;
    std::vector<CollisionFilter> CollisionFilters;
    std::vector<PhysicsJointDef> PhysicsJointDefs;
};

std::expected<Scene, std::string> LoadScene(const std::filesystem::path &);
} // namespace gltf
