#include "CameraTypes.h"
#include "Path.h"
#include "PathSerialize.h"
#include "gpu/PunctualLight.h"
#include "gpu/Transform.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

#include "action/ActionIndex.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "armature/ArmatureSerialize.h"
#include "armature/BoneConstraint.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "audio/SoundVertices.h"
#ifdef SURFACE_AUDIO
#include "audio/surface/SurfaceAudio.h"
#endif
#include "gizmo/GizmoInteraction.h"
#include "gltf/GltfScene.h"
#include "mesh/Mesh.h"
#include "mesh/MeshBvh.h"
#include "mesh/MeshComponents.h"
#include "mesh/PrimitiveType.h"
#include "mesh/TetBuffers.h"
#include "physics/PhysicsContact.h"
#include "physics/PhysicsTypes.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/MeshBuffers.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/RotationUi.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/BoneSelection.h"
#include "selection/SelectionComponents.h"
#include "snapshot/SnapshotRoles.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"

#include <entt/entity/registry.hpp>

#include <array>
#include <ranges>
#include <set>
#include <unordered_map>
#include <unordered_set>

// Every registry component must appear in exactly one of Persistent or Derived.
namespace {
// ViewCamera and LookingThrough require constructor seeds before deserialization.
void EmplaceViewCamera(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    ViewCamera v{vec3{0, 0, 1}, vec3{0}, Camera{}};
    if (zpp::bits::failure(zpp::bits::in{bytes}(v))) return;
    r.emplace_or_replace<ViewCamera>(e, v);
}
void EmplaceLookingThrough(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    LookingThrough l{ViewCamera{vec3{0, 0, 1}, vec3{0}, Camera{}}};
    if (zpp::bits::failure(zpp::bits::in{bytes}(l))) return;
    r.emplace_or_replace<LookingThrough>(e, std::move(l));
}

// Non-default-constructible serialized types specialize this emplacer.
template<typename C>
inline constexpr void (*CustomEmplace)(entt::registry &, entt::entity, std::span<const std::byte>) = nullptr;
template<> inline constexpr auto CustomEmplace<ViewCamera> = &EmplaceViewCamera;
template<> inline constexpr auto CustomEmplace<LookingThrough> = &EmplaceLookingThrough;

// Bone transforms are derived from RestLocal and ArmaturePose.
template<typename C>
inline constexpr bool (*SkipEntityFor)(const entt::registry &, entt::entity) = nullptr;
template<> inline constexpr auto SkipEntityFor<Transform> = [](const entt::registry &r, entt::entity e) { return r.all_of<BoneIndex>(e); };

// Aligned storage supports non-default-constructible implicit-lifetime types.
template<typename C>
void EmplaceTrivial(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    if constexpr (std::is_empty_v<C>) {
        r.emplace_or_replace<C>(e);
    } else {
        alignas(C) std::byte storage[sizeof(C)];
        std::memcpy(storage, bytes.data(), sizeof(C));
        r.emplace_or_replace<C>(e, *std::launder(reinterpret_cast<const C *>(storage)));
    }
}

template<typename C>
void SerializeThunk(const void *component, std::vector<std::byte> &out) {
    thread_local std::vector<std::byte> buffer;
    buffer.clear();
    zpp::bits::out archive{buffer};
    // zpp aggregate reflection mis-encodes large const aggregates, while the output archive only reads this reference.
    if (zpp::bits::failure(archive(const_cast<C &>(*static_cast<const C *>(component))))) return;
    out.insert(out.end(), buffer.begin(), buffer.begin() + archive.position());
}

template<typename C>
void EmplaceSerialized(entt::registry &r, entt::entity e, std::span<const std::byte> bytes) {
    C value;
    if (zpp::bits::failure(zpp::bits::in{bytes}(value))) return;
    r.emplace_or_replace<C>(e, std::move(value));
}

using entt::type_list;

// Canonical serialized state.
using Persistent = type_list<
    Transform, ViewportTheme, WorkspaceLights, PunctualLight,
    Name, Selected, Active, ObjectKind, MeshActiveElement, Scene, ActiveScene, SceneMembership, SubElementOf,
    ScaleLocked, Instance, Hidden, SceneNode, ParentInverse, MeshHandle, VertexStoreId, ObjectExtrasTag,
    MeshElementSelection, MeshMaterialAssignment, MeshMaterialSlotSelection, MaterialVariants, MaterializedTextures, PbrMeshFeatures,
    PrimitiveShape, Path, Camera, ViewCamera, LookingThrough, Interaction, EditMode, OrbitToActive, ShadeSmoothAngle, AudioOutputConfig, AudioOutputMix, Striker, ModalSoundControls, ContactSurface, SurfaceSoundControls,
    AcousticMaterial, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, ModalSolveSettings, MassProperties, TetBuffers, ModalEigenSummary,
    SelectionXRay, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment, TransformGizmoState, ActionIndex,
    TimelineRange, TimelinePlayback, AnimationTimelineView,
    PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion,
    ColliderShape, ColliderMaterial, ColliderPolicy, PhysicsVelocity, TriggerTag, TriggerNodes, PhysicsJoint,
    Armature, ArmatureObject, BoneJointEntities, BoneJoint, BoneSubPartOf, BoneActive, BoneSelection,
    BoneConstraints, ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, ArmatureAnimation, ArmaturePose,
    NodeTransformAnimation, MorphWeightAnimation, MorphWeightState,
    SourceNodeIndex, SourceParentNodeIndex, SourceSiblingIndex, SourceMeshIndex, SourceCameraIndex,
    SourceLightIndex, SourcePhysicsMaterialIndex, SourceCollisionFilterIndex, SourcePhysicsJointDefIndex,
    SourceSceneIndex, SourceMeshKind, GltfObject, CameraName, LightName, SourceObjectName, MeshName,
    SourceMatrixTransform, SourceEmptyName, MeshSourceLayout, gltf::SourceAssets>;

// Reconstructed from Persistent state.
using Derived = type_list<
    RenderInstance, WorldTransform, PosedLocal, MeshBuffers, MeshShadingSummary, BoneAdjacencyIndices, ModelsBuffer,
    MaterialDirty, LightIndex, EnabledInteractionModes, LastEvaluatedFrame,
    PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache, BoneInstanceStateDirty, ArmaturePoseState,
    MorphWeightGpuRange, MeshElementSelectionStats, AdditiveBoxSelectBaseline, ExciteSelectionBaseline, EditSelectionDirty, PendingEditElementClick,
    PendingBoxSelect, PendingBoxSelectFinalize, BoxSelectStatsDirty, PendingPick, PendingTextureUploads, BoxSelectState, PlaybackFrame,
    PhysicsCacheInvalid, RotationUiVariant, RotationUiDriving, GizmoInteraction, PendingTransform, StartScreenTransform,
#ifdef SURFACE_AUDIO
    SurfaceRelief, SurfaceFinishKey,
#endif
    SoundVertices, ContactDynamics, ReportContacts, MeshBvh>;

// Field-wise serialization excludes indeterminate variant, optional, and padding bytes from snapshots.
using ForceSerialize = type_list<
    Camera, PrimitiveShape, ColliderShape, PhysicsMotion, // variant / optional
    ViewportDisplay, MaterialPreviewLighting, RenderedLighting, TransformGizmoState, AudioOutputMix, // padding
    TimelinePlayback, PhysicsJoint, BoneSubPartOf, ModalSolveSettings>; // padding

// Derived types requiring field-wise comparison.
using ForceFieldwise = type_list<RotationUiVariant>;

// Selects field-wise comparison for heap-backed or indeterminate object representations.
template<typename C>
inline constexpr bool NeedsFieldwise =
    CustomEmplace<C> != nullptr ||
    entt::type_list_contains_v<ForceSerialize, C> ||
    entt::type_list_contains_v<ForceFieldwise, C> ||
    (entt::type_list_contains_v<Persistent, C> && !std::is_trivially_copyable_v<C>);

// Padding-only cases require explicit ForceSerialize entries because they cannot be detected statically.
template<typename> inline constexpr bool IsVariantOrOptional = false;
template<typename... Ts> inline constexpr bool IsVariantOrOptional<std::variant<Ts...>> = true;
template<typename T> inline constexpr bool IsVariantOrOptional<std::optional<T>> = true;

template<typename C>
consteval bool HoldsVariantOrOptional() {
    if constexpr (IsVariantOrOptional<C>) return true;
    else if constexpr (std::is_trivially_copyable_v<C> && std::is_aggregate_v<C>) // non-aggregates (e.g. ViewCamera) aren't reflectable and use CustomEmplace
        return zpp::bits::visit_members_types<C>([]<typename... Ms>() { return (IsVariantOrOptional<std::remove_cvref_t<Ms>> || ...); });
    else return false;
}
template<typename... Cs>
consteval bool VariantComponentsFieldwise(type_list<Cs...>) { return (... && (!HoldsVariantOrOptional<Cs>() || NeedsFieldwise<Cs>)); }
static_assert(VariantComponentsFieldwise(Persistent{}), "A trivially-copyable Persistent component holds a std::variant/std::optional but would be memcpy-serialized. Add it to ForceSerialize.");

template<typename C>
bool ValuesEqual(const void *a, const void *b) {
    if constexpr (std::is_empty_v<C>) {
        return true;
    } else if constexpr (NeedsFieldwise<C>) {
        std::vector<std::byte> ba, bb;
        SerializeThunk<C>(a, ba);
        SerializeThunk<C>(b, bb);
        return ba == bb;
    } else {
        return std::memcmp(a, b, sizeof(C)) == 0;
    }
}

// Returns nullptr for derived types without a valid comparator.
template<typename C>
constexpr bool (*MakeComparator())(const void *, const void *) {
    if constexpr (std::is_empty_v<C> || NeedsFieldwise<C> || std::is_trivially_copyable_v<C>) return &ValuesEqual<C>;
    else return nullptr;
}

// Selects Tag, Bytes, or Serialized encoding from the component traits and overrides.
template<typename C>
snapshot::SnapshotEntry MakeEntry() {
    using snapshot::Encoding;
    if constexpr (std::is_empty_v<C>) return {Encoding::Tag, 0, nullptr, &EmplaceTrivial<C>, SkipEntityFor<C>};
    else if constexpr (CustomEmplace<C> != nullptr) return {Encoding::Serialized, 0, &SerializeThunk<C>, CustomEmplace<C>, SkipEntityFor<C>};
    else if constexpr (entt::type_list_contains_v<ForceSerialize, C>) return {Encoding::Serialized, 0, &SerializeThunk<C>, &EmplaceSerialized<C>, SkipEntityFor<C>};
    else if constexpr (std::is_trivially_copyable_v<C>) return {Encoding::Bytes, sizeof(C), nullptr, &EmplaceTrivial<C>, SkipEntityFor<C>};
    else return {Encoding::Serialized, 0, &SerializeThunk<C>, &EmplaceSerialized<C>, SkipEntityFor<C>};
}

template<typename... Cs>
std::array<entt::id_type, sizeof...(Cs)> TypeHashes(type_list<Cs...>) {
    return {entt::type_hash<Cs>::value()...};
}
template<typename... Cs>
std::array<std::pair<entt::id_type, snapshot::SnapshotEntry>, sizeof...(Cs)> TypeEntries(type_list<Cs...>) {
    return {std::pair{entt::type_hash<Cs>::value(), MakeEntry<Cs>()}...};
}

const auto ClassifiedHashes = TypeHashes(entt::type_list_cat_t<Persistent, Derived>{}) | std::ranges::to<std::unordered_set<entt::id_type>>();
} // namespace

namespace snapshot {
const std::unordered_map<entt::id_type, SnapshotEntry> &SnapshotTable() {
    static const auto table = TypeEntries(Persistent{}) | std::ranges::to<std::unordered_map<entt::id_type, SnapshotEntry>>();
    return table;
}

void VerifyCoverage(const entt::registry &r) {
    std::set<std::string> unclassified; // Stable diagnostic ordering.
    for (auto [id, set] : r.storage()) {
        if (set.empty()) continue;
        const auto &info = set.info();
        if (!std::string_view{info.name()}.starts_with("entt::")) {
            if (!ClassifiedHashes.contains(info.hash())) unclassified.emplace(info.name());
        }
    }
    if (unclassified.empty()) return;

    std::string msg = "snapshot: component(s) in registry storage are classified neither Persistent nor Derived "
                      "(add to a list in SnapshotRoles.cpp):";
    for (const auto &name : unclassified) (msg += "\n  ") += name;
    throw std::runtime_error(msg);
}

std::optional<bool> ComponentValuesEqual(entt::id_type type_hash, const void *a, const void *b) {
    using Comparator = bool (*)(const void *, const void *);
    static const auto comparators = [] {
        std::unordered_map<entt::id_type, Comparator> m;
        const auto add = [&]<typename... Cs>(type_list<Cs...>) {
            (m.emplace(entt::type_hash<Cs>::value(), MakeComparator<Cs>()), ...);
        };
        add(entt::type_list_cat_t<Persistent, Derived>{});
        return m;
    }();
    const auto it = comparators.find(type_hash);
    if (it == comparators.end() || it->second == nullptr) return std::nullopt;
    return it->second(a, b);
}
} // namespace snapshot
