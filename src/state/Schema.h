#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace state {
using TypeId = uint32_t;
template<typename T> consteval std::string_view TypeName() {
    std::string_view text = __PRETTY_FUNCTION__;
    const auto begin = text.find("T = ") + 4;
    return text.substr(begin, text.rfind(']') - begin);
}
// Closed schema: adding a component or service assigns a compile-time slot here.
// Names keep concrete definitions in their domain TUs, including private service types.
inline constexpr std::string_view SchemaNames[] = {
    "fastfem::MassProperties",
    "std::variant<RotationQuat, RotationEuler, RotationAxisAngle>",
    "std::variant<primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>",
    "AcousticMaterial",
    "Active",
    "ActiveSamplerAnisotropy",
    "ActiveScene",
    "AdditiveBoxSelectBaseline",
    "AnimationClips",
    "AnimationTimelineView",
    "Animations",
    "Armature",
    "ArmatureModifier",
    "ArmatureObject",
    "ArmaturePoseState",
    "AudioDeviceResource",
    "AudioOutputConfig",
    "AudioOutputMix",
    "AudioSamples",
    "AudioTrackers",
    "AuthoredCornerNormals",
    "BodyPoseCache",
    "BoneActive",
    "BoneAdjacencyIndices",
    "BoneAttachment",
    "BoneConstraints",
    "BoneDisplayScale",
    "BoneIndex",
    "BoneDelta",
    "BoneJoint",
    "BoneJointEntities",
    "BoneSelection",
    "BoneSubPartOf",
    "BoxSelectState",
    "ColliderMaterial",
    "ColliderPolicy",
    "ColliderShape",
    "CollisionFilter",
    "CollisionSystem",
    "ContactDynamics",
    "ContactSurface",
    "EditMode",
    "EnabledInteractionModes",
    "EntityDestroyTracker",
    "EntityNameCounts",
    "EnvironmentStore",
    "ExciteSelectionBaseline",
    "FrameState",
    "GizmoInteraction",
    "GltfNode",
    "GpuBuffers",
    "GpuSceneState",
    "Hidden",
    "ImageLight",
    "Instance",
    "Interaction",
    "LastEvaluatedFrame",
    "LightIndex",
    "LookingThrough",
    "MasterCapture",
    "MaterialPreviewLighting",
    "MaterialStore",
    "MaterialVariants",
    "MaterializedTextures",
    "MeshActiveElement",
    "MeshBuffers",
    "MeshBvh",
    "MeshElementSelection",
    "MeshGeometryDirty",
    "MeshHandle",
    "MeshMaterialAssignment",
    "MeshMaterialSlotSelection",
    "MeshPipelines",
    "MeshPositionsChanged",
    "MeshShadingSummary",
    "MeshSourceLayout",
    "MeshStore",
    "ModalAudio",
    "ModalEigenSummary",
    "ModalGain",
    "ModalModes",
    "ModalSolveJobs",
    "ModalSolveSettings",
    "ModalSoundControls",
    "ModalTuning",
    "ModalWarmStart",
    "ModelsBuffer",
    "MonitorLimiter",
    "MorphWeightRange",
    "Name",
    "Orthographic",
    "ObjectExtrasTag",
    "ObjectKind",
    "OrbitToActive",
    "Path",
    "PbrMeshFeatures",
    "PendingBoxSelect",
    "PendingEditElementClick",
    "PendingHide",
    "PendingImportMesh",
    "PendingPick",
    "PendingRenderRequest",
    "PendingSetEditMode",
    "PendingTransform",
    "Perspective",
    "PhysicsBodyHandle",
    "PhysicsConstraintHandle",
    "PhysicsContactImpacts",
    "PhysicsJoint",
    "PhysicsJointDef",
    "PhysicsMaterial",
    "PhysicsMotion",
    "PhysicsSimulationSettings",
    "PhysicsState",
    "PhysicsSustainedContacts",
    "PhysicsVelocity",
    "Pipelines",
    "PlaybackFrame",
    "PosedLocal",
    "PunctualLight",
    "RealImpactActiveMicrophone",
    "RealImpactMicrophone",
    "RealImpactVertices",
    "Recording",
    "RenderInstance",
    "RenderSamplerSlots",
    "RenderTargets",
    "RenderedLighting",
    "ReportContacts",
    "RotationUiDriving",
    "SamplePlayback",
    "ScaleLocked",
    "Scene",
    "SceneMembership",
    "SceneNode",
    "Selected",
    "SelectionSlots",
    "ShadeSmoothAngle",
    "SoundVertices",
    "SoundVerticesModel",
    "SourceIndex",
    "StartBoneLength",
    "StartScreenTransform",
    "StartTransform",
    "Striker",
    "StudioEnvironment",
    "SubElementOf",
    "SurfaceFinishKey",
    "SurfaceRelief",
    "SurfaceSoundControls",
    "TetBuffers",
    "TextureStore",
    "TimelineNavigation",
    "TimelinePlayback",
    "TimelineRange",
    "Transform",
    "TransformGizmoState",
    "TriggerNodes",
    "TriggerTag",
    "VertexForce",
    "VertexSamples",
    "VertexStoreId",
    "VideoRecording",
    "ViewCamera",
    "ViewportConsumerFence",
    "ViewportDisplay",
    "ViewportExtent",
    "ViewportIcons",
    "ViewportRenderResources",
    "ViewportTheme",
    "Visibility",
    "WindowsState",
    "WorkspaceLights",
    "WorldTransform",
    "action::DragFieldStart",
    "action::Errors",
    "gltf::SourceAssets",
    "mtl::BindlessSet",
    "mtl::Context",
    "mtl::LibraryCache",
    "project::Assets",
    "project::Project *",
    "state::Entity",
    "std::unique_ptr<ValidationSession>",
};
inline constexpr size_t SchemaSize = sizeof(SchemaNames) / sizeof(*SchemaNames);
constexpr bool SameName(std::string_view a, std::string_view b) {
    constexpr std::string_view anonymous = "(anonymous namespace)::";
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (a.substr(i).starts_with(anonymous)) {
            i += anonymous.size();
            continue;
        }
        if (b.substr(j).starts_with(anonymous)) {
            j += anonymous.size();
            continue;
        }
        if (i < a.size() && a[i] == ' ') {
            ++i;
            continue;
        }
        if (j < b.size() && b[j] == ' ') {
            ++j;
            continue;
        }
        if (i == a.size() || j == b.size() || a[i++] != b[j++]) return false;
    }
    return true;
}
template<typename T> inline constexpr TypeId TypeIndex = [] {
    constexpr auto name = TypeName<T>();
    for (TypeId i = 0; i < SchemaSize; ++i)
        if (SameName(name, SchemaNames[i])) return i;
    return TypeId(SchemaSize);
}();
template<typename T> consteval TypeId Type() {
    constexpr auto index = TypeIndex<std::remove_cv_t<T>>;
    static_assert(index < SchemaSize, "Type is missing from the MeshEditor state schema");
    return index;
}

// Serialized records identify a type by a hash of its schema name, so slot numbers can change without invalidating them.
enum class TypeKey : uint32_t {};
constexpr uint32_t HashName(std::string_view name) {
    uint32_t hash = 2166136261u;
    for (const char c : name) hash = (hash ^ uint8_t(c)) * 16777619u;
    return hash;
}
template<typename T> consteval TypeKey Key() { return TypeKey{HashName(SchemaNames[Type<T>()])}; }
inline constexpr auto KeyTable = [] {
    std::array<std::pair<uint32_t, TypeId>, SchemaSize> table{};
    for (TypeId i = 0; i < SchemaSize; ++i) table[i] = {HashName(SchemaNames[i]), i};
    std::ranges::sort(table);
    return table;
}();
static_assert(std::ranges::adjacent_find(KeyTable, {}, [](const auto &entry) { return entry.first; }) == KeyTable.end(), "Schema name hashes collide");
// The slot for a serialized key, or SchemaSize for a key absent from the schema.
constexpr TypeId Slot(TypeKey key) {
    const auto it = std::ranges::lower_bound(KeyTable, uint32_t(key), {}, [](const auto &entry) { return entry.first; });
    return it != KeyTable.end() && it->first == uint32_t(key) ? it->second : TypeId(SchemaSize);
}
} // namespace state
