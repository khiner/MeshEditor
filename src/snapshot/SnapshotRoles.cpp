#include "CameraTypes.h"
#include "Path.h"
#include "PathSerialize.h"
#include "gpu/PunctualLight.h"
#include "gpu/Transform.h"
#include "gpu/ViewportTheme.h"
#include "gpu/WorkspaceLights.h"

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
#include "audio/ModalEigenSummary.h"
#include "audio/ModalModes.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/GltfScene.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/PrimitiveType.h"
#include "mesh/TetMeshData.h"
#include "object/ExtrasComponents.h"
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

// Single source of truth for the snapshot partition: every registry component is listed as exactly one of Persistent or Derived.
// Persistent components are serialized, and the table is folded from this list.
// Derived components are rebuilt by ProcessComponentEvents, not serialized but listed so VerifyCoverage knows it's intentional.
// Anything in neither list hard-fails at save.
namespace {
// ViewCamera/LookingThrough aren't default-constructible, so seed via constructor then let serialize-in overwrite it.
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

// A component that zpp can serialize but can't default-construct specializes this to a custom emplacer,
// which the table builder uses instead of EmplaceSerialized.
template<typename C>
inline constexpr void (*CustomEmplace)(entt::registry &, entt::entity, std::span<const std::byte>) = nullptr;
template<> inline constexpr auto CustomEmplace<ViewCamera> = &EmplaceViewCamera;
template<> inline constexpr auto CustomEmplace<LookingThrough> = &EmplaceLookingThrough;

// Specialized to skip entities whose value is derived (a bone's Transform comes from RestLocal + ArmaturePose).
template<typename C>
inline constexpr bool (*SkipEntityFor)(const entt::registry &, entt::entity) = nullptr;
template<> inline constexpr auto SkipEntityFor<Transform> = [](const entt::registry &r, entt::entity e) { return r.all_of<BoneIndex>(e); };

// Per-encoding (de)serialization thunks the table dispatches through.

// Trivially-copyable (or empty-tag) component, restored by memcpy.
// Aligned storage (not `C value;`) so non-default-constructible trivially-copyable types work, since memcpy creates the object (implicit-lifetime).
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
    // zpp's aggregate reflection mis-encodes a const aggregate above ~12 members, so serialize through a non-const ref (the out archive only reads it).
    // This matches the type EmplaceSerialized reads back into.
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

// Canonical input state, serialized (per-type encoding auto-selected by MakeEntry below).
using Persistent = type_list<
    Transform, ViewportTheme, WorkspaceLights, PunctualLight,
    Name, Selected, Active, ObjectKind, MeshActiveElement, Scene, ActiveScene, SceneMembership, SubElementOf,
    ScaleLocked, Instance, Hidden, SceneNode, ParentInverse, MeshHandle, VertexStoreId, ObjectExtrasTag,
    SmoothShading, MeshConnectivity,
    MeshMaterialAssignment, MeshMaterialSlotSelection, MaterialVariants, MaterializedTextures, PbrMeshFeatures,
    PrimitiveShape, Path, Camera, ViewCamera, LookingThrough, Interaction, EditMode, OrbitToActive, AudioOutputConfig, AudioOutputMix, Striker, ModalSoundControls,
    AcousticMaterial, SoundVerticesModel, ModalModes, ModalGain, ModalTuning, MassProperties, TetMeshData, ModalEigenSummary,
    SelectionXRay, ViewportDisplay, MaterialPreviewLighting, RenderedLighting, StudioEnvironment, TransformGizmoState,
    TimelineRange, TimelinePlayback, AnimationTimelineView,
    PhysicsSimulationSettings, PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef, PhysicsMotion,
    ColliderShape, ColliderMaterial, ColliderPolicy, PhysicsVelocity, TriggerTag, TriggerNodes, PhysicsJoint,
    Armature, ArmatureObject, BoneJointEntities, BoneJoint, BoneSubPartOf, BoneActive, BoneSelection,
    BoneConstraints, ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, ArmatureAnimation, ArmaturePose,
    NodeTransformAnimation, MorphWeightAnimation, MorphWeightState,
    SourceNodeIndex, SourceParentNodeIndex, SourceSiblingIndex, SourceMeshIndex, SourceCameraIndex,
    SourceLightIndex, SourcePhysicsMaterialIndex, SourceCollisionFilterIndex, SourcePhysicsJointDefIndex,
    SourceSceneIndex, SourceMeshKind, GltfObject, CameraName, LightName, SkinName, SourceObjectName, MeshName,
    SourceMatrixTransform, SourceEmptyName, MeshSourceLayout, gltf::SourceAssets>;

// Reconstructed from the Persistent set by ProcessComponentEvents, on_construct, and reactive handlers.
// Never serialized, listed only so VerifyCoverage treats them as intentionally excluded.
using Derived = type_list<
    RenderInstance, WorldTransform, PosedLocal, MeshBuffers, BoneAdjacencyIndices, ModelsBuffer, VertexClass, BBoxWireframe, DeformedBounds,
    TetWireframe, MaterialDirty, LightIndex, EnabledInteractionModes, LastEvaluatedFrame,
    PhysicsBodyHandle, PhysicsConstraintHandle, BodyPoseCache, ColliderWireframe, BoneInstanceStateDirty, ArmaturePoseState,
    MorphWeightGpuRange, AdditiveBoxSelectBaseline, SelectionBitsDirty, ElementStatesDirty, PendingEditElementClick, OverlayExtra, OverlayVertexStoreId,
    PendingBoxSelect, PendingPick, PendingTextureUploads, SelectionBitsetRef, BoxSelectState, SelectedInstanceCount, PlaybackFrame,
    PhysicsCacheInvalid, RotationUiVariant, RotationUiDriving, GizmoInteraction, PendingTransform, StartScreenTransform,
    SoundVertices, ContactDynamics>;

// Trivially copyable, but the raw object bytes are nondeterministic: a std::variant's inactive alternative or a
// std::optional's disengaged storage holds uninitialized memory, and struct padding is never written. memcpy would
// capture those bytes and make the snapshot image differ run-to-run, so serialize these field-wise via zpp instead.
// (ViewCamera/LookingThrough have the same problem but are already serialized through their CustomEmplace above.)
using ForceSerialize = type_list<
    Camera, PrimitiveShape, ColliderShape, PhysicsMotion, // variant / optional
    ViewportDisplay, MaterialPreviewLighting, RenderedLighting, TransformGizmoState, AudioOutputMix, // padding
    TimelinePlayback, PhysicsJoint, BoneSubPartOf>; // padding

// Derived components that are memcmp-unsafe (variant/optional/padding), compared field-wise instead of by memcmp.
using ForceFieldwise = type_list<RotationUiVariant>;

// True when memcmp would be wrong for C (heap-backed, or a variant/optional/padded type), so compare it field-wise.
template<typename C>
inline constexpr bool NeedsFieldwise =
    CustomEmplace<C> != nullptr ||
    entt::type_list_contains_v<ForceSerialize, C> ||
    entt::type_list_contains_v<ForceFieldwise, C> ||
    (entt::type_list_contains_v<Persistent, C> && !std::is_trivially_copyable_v<C>);

// Guard: a trivially-copyable component with a variant/optional has indeterminate bytes, so it must be field-wise (NeedsFieldwise), not memcpy'd.
// Padding-only cases aren't statically detectable and stay listed by hand in ForceSerialize.
template<typename> inline constexpr bool IsVariantOrOptional = false;
template<typename... Ts> inline constexpr bool IsVariantOrOptional<std::variant<Ts...>> = true;
template<typename T> inline constexpr bool IsVariantOrOptional<std::optional<T>> = true;

template<typename C>
consteval bool HoldsVariantOrOptional() {
    if constexpr (IsVariantOrOptional<C>) return true;
    else if constexpr (std::is_trivially_copyable_v<C> && std::is_aggregate_v<C>) // non-aggregates (e.g. ViewCamera) aren't reflectable; they use CustomEmplace
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

// nullptr => incomparable: a non-trivially-copyable derived component with no serializer, so it's skipped.
template<typename C>
constexpr bool (*MakeComparator())(const void *, const void *) {
    if constexpr (std::is_empty_v<C> || NeedsFieldwise<C> || std::is_trivially_copyable_v<C>) return &ValuesEqual<C>;
    else return nullptr;
}

// Encoding deduced from the type: empty -> Tag; CustomEmplace or ForceSerialize -> Serialized (zpp);
// trivially copyable -> Bytes (memcpy); else Serialized. CustomEmplace handles non-default-constructible types.
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
    std::set<std::string> unclassified; // sorted + deduped for a stable message
    for (auto [id, set] : r.storage()) {
        if (set.empty()) continue;
        const auto &info = set.info();
        if (!std::string_view{info.name()}.starts_with("entt::")) { // skip entt:: entity / reactive storages, not components
            if (!ClassifiedHashes.contains(info.hash())) unclassified.emplace(info.name());
        }
    }
    if (unclassified.empty()) return;

    std::string msg = "snapshot: component(s) in registry storage are classified neither Persistent nor Derived "
                      "(add to a list in SnapshotRoles.cpp):";
    for (const auto &name : unclassified) (msg += "\n  ") += name;
    throw std::runtime_error(msg);
}

bool SnapshotSkipsEntity(const entt::registry &r, entt::entity e) { return r.all_of<OverlayExtra>(e); }

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
