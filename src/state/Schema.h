#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace state {
using TypeId = uint32_t;
template<typename T> consteval std::string_view TypeName() {
    const std::string_view text = __PRETTY_FUNCTION__;
    const auto begin = text.find("T = ") + 4;
    return text.substr(begin, text.rfind(']') - begin);
}
// A type's own name, without its namespace or template arguments.
constexpr std::string_view LeafName(std::string_view name) {
    const auto base = name.substr(0, name.find('<'));
    const auto scope = base.rfind("::");
    return scope == std::string_view::npos ? base : base.substr(scope + 2);
}
template<typename T> consteval std::string_view LeafName() { return LeafName(TypeName<T>()); }
// Closed schema: adding a component or service assigns a compile-time slot here.
// Names keep concrete definitions in their domain TUs, including private service types.
inline constexpr std::string_view SchemaNames[] = {
    "fastfem::MassProperties",
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
    "BoneAttachment",
    "BoneConstraints",
    "BoneDisplayScale",
    "BoneIndex",
    "BoneDelta",
    "BoneJoint",
    "BoneJointEntities",
    "BoneSelection",
    "BoneSubPartOf",
    "BoundsRun",
    "BoxSelectState",
    "ClusterLodDemand",
    "ClusterLodDirty",
    "ColliderListedUnder",
    "ColliderMaterial",
    "ColliderPolicy",
    "ColliderShape",
    "CollisionFilter",
    "CollisionSystem",
    "ConstraintTargets",
    "ContactDynamics",
    "ContactSurface",
    "CookCount",
    "EditMode",
    "EditPrimaries",
    "EnabledInteractionModes",
    "EntityNameCounts",
    "EnvironmentStore",
    "EvaluatedMesh",
    "ExciteSelectionBaseline",
    "FrameState",
    "GizmoInteraction",
    "GltfNode",
    "GpuBuffers",
    "GpuSceneState",
    "HasVariantMappings",
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
    "MeshBoundsDirty",
    "MeshColliderList",
    "MeshDisplayDirty",
    "MeshEditWork",
    "MeshElementSelection",
    "MeshEntities",
    "MeshGeometryDirty",
    "MeshHandle",
    "MeshLayoutDirty",
    "MeshMaterialAssignment",
    "MeshMaterialSlotSelection",
    "MeshPipelines",
    "MeshPositionsChanged",
    "MeshSourceLayout",
    "MeshStore",
    "MeshVertexOverlays",
    "MeshletPositionsDirty",
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
    "OutlinerRows",
    "Path",
    "PbrMeshFeatures",
    "PendingBoxSelect",
    "PendingEditElementClick",
    "PendingImportMesh",
    "PendingObjectRemovals",
    "PendingPick",
    "PendingRenderRequest",
    "PendingSetEditMode",
    "PendingSlotRemovals",
    "PendingTransform",
    "Perspective",
    "PhysicsBodyHandle",
    "PhysicsConstraintHandle",
    "PhysicsDefinitionUses",
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
    "PosedLayout",
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
    "ResyncState",
    "SamplePlayback",
    "SavedViewCamera",
    "ScaleLocked",
    "Scene",
    "SceneChildren",
    "SceneMembership",
    "SceneParent",
    "Selected",
    "SelectedKeyframes",
    "SelectionFlags",
    "SelectionSlots",
    "ShadeSmoothAngle",
    "SoundVertices",
    "SoundVerticesModel",
    "SourceIndex",
    "StartBoneLength",
    "StartPivot",
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
    "TransformRoots",
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
    "action::mesh::Bevel",
    "action::mesh::Bisect",
    "action::mesh::ConvexHull",
    "action::mesh::Decimate",
    "action::mesh::Delete",
    "action::mesh::Dissolve",
    "action::mesh::Duplicate",
    "action::mesh::EdgeSplit",
    "action::mesh::Extrude",
    "action::mesh::ExtrudeRepeat",
    "action::mesh::FillHoles",
    "action::mesh::FlipNormals",
    "action::mesh::Inset",
    "action::mesh::Merge",
    "action::mesh::Poke",
    "action::mesh::Randomize",
    "action::mesh::RecalculateNormals",
    "action::mesh::Rip",
    "action::mesh::ShrinkFatten",
    "action::mesh::SmoothVertices",
    "action::mesh::Solidify",
    "action::mesh::Spin",
    "action::mesh::Split",
    "action::mesh::Subdivide",
    "action::mesh::Symmetrize",
    "action::mesh::Triangulate",
    "action::mesh::TrisToQuads",
    "action::mesh::Wireframe",
    "gltf::SourceAssets",
    "mtl::BindlessSet",
    "mtl::Context",
    "mtl::LibraryCache",
    "procedure::Boolean",
    "procedure::BooleanMath",
    "procedure::BooleanValue",
    "procedure::Compare",
    "procedure::CookPlan",
    "procedure::FloatMath",
    "procedure::FloatValue",
    "procedure::GeometryProcedure",
    "procedure::Index",
    "procedure::InlineSelection",
    "procedure::InstanceOnPoints",
    "procedure::IntegerValue",
    "procedure::Join",
    "procedure::MeshToPoints",
    "procedure::NamedAttribute",
    "procedure::NodeError",
    "procedure::NodeLayout",
    "procedure::NodeSelected",
    "procedure::NodeTheme",
    "procedure::Normal",
    "procedure::ObjectInfo",
    "procedure::Output",
    "procedure::Position",
    "procedure::ProcedureInstances",
    "procedure::ProcedureOwnedMesh",
    "procedure::PublishSelection",
    "procedure::RealizeInstances",
    "procedure::RotationValue",
    "procedure::SetPosition",
    "procedure::Source",
    "procedure::StoreNamedAttribute",
    "procedure::TransformElements",
    "procedure::TransformGeometry",
    "procedure::VectorMath",
    "procedure::VectorValue",
    "primitive::Circle",
    "primitive::Cone",
    "primitive::Cuboid",
    "primitive::Cylinder",
    "primitive::IcoSphere",
    "primitive::Plane",
    "primitive::Torus",
    "primitive::UVSphere",
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
    for (TypeId i = 0; i < SchemaSize; ++i) {
        if (SameName(name, SchemaNames[i])) return i;
    }
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
    uint32_t hash{2166136261u};
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
    const auto *const it = std::ranges::lower_bound(KeyTable, uint32_t(key), {}, [](const auto &entry) { return entry.first; });
    return it != KeyTable.end() && it->first == uint32_t(key) ? it->second : TypeId(SchemaSize);
}
} // namespace state
