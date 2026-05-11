#pragma once

#include "Camera.h"
#include "SceneOps.h"
#include "Tets.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ModalModes.h"
#include "audio/RealImpactComponents.h"
#include "entt_fwd.h"
#include "gpu/Element.h"
#include "gpu/InteractionMode.h"
#include "gpu/PunctualLight.h"
#include "gpu/PunctualLightType.h"
#include "gpu/ViewportTheme.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"
#include "numeric/quat.h"
#include "numeric/vec4.h"
#include "physics/PhysicsTypes.h"
#include "scene_impl/SceneComponents.h"
#include "scene_impl/SceneInternalTypes.h"
#include "scene_impl/SceneTransformUtils.h"
#include <entt/core/type_info.hpp>

#include <bit>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace action {

namespace selection {
struct Select {
    entt::entity Entity;
};
struct ToggleSelected {
    entt::entity Entity;
};
struct SelectBone {
    entt::entity Entity;
};
struct ExtendActive {
    entt::entity Entity;
};
struct ExtendBoneActive {
    entt::entity Entity;
};
struct SetBoneSelectionPart {
    entt::entity Entity;
    std::optional<BoneSel> Part;
    bool Additive;
};
struct DeselectAll {};
struct SnapshotBoxSelectBaseline {};
struct ClearBoxSelectBaseline {};
struct BoneHit {
    entt::entity Entity;
    std::optional<BoneSel> Part;
};
struct ApplyBoxSelectObjectHits {
    std::vector<entt::entity> Hits;
    bool Additive;
};
struct ApplyBoxSelectBoneHits {
    std::vector<BoneHit> Hits;
    bool Additive;
};
struct ApplyEditElementClick {
    uvec2 MousePx;
    bool Toggle;
};
struct ApplyTreeSelection {
    enum class ClearKind : uint8_t { None,
                                     BonesOnly,
                                     All };
    std::vector<entt::entity> ToSelect;
    std::vector<entt::entity> ToDeselect;
    entt::entity NavToActive{null_entity};
    ClearKind Clear{ClearKind::None};
};
} // namespace selection

namespace object {
struct Delete {};
struct Duplicate {};
struct DuplicateLinked {};
struct ToggleHidden {};
struct SetSelectedVisible {
    bool Visible;
};
struct ParentToActive {};
struct ClearParent {};

struct AddEmpty {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddArmature {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddCamera {
    std::unique_ptr<ObjectCreateInfo> Info;
    std::optional<Camera> Props;
};
struct AddLight {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddMeshPrimitive {
    PrimitiveShape Shape;
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
struct ImportMesh {
    std::filesystem::path Path;
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
struct ReplaceMesh {
    std::unique_ptr<MeshData> Data;
};
} // namespace object

namespace project {
struct ClearMeshes {};

struct LoadGltf {
    std::filesystem::path Path;
};
struct SaveGltf {
    std::filesystem::path Path;
};
struct LoadRealImpact {
    std::filesystem::path Directory;
};
} // namespace project

namespace scene {
struct SetInteractionMode {
    InteractionMode Mode;
};
struct CycleInteractionMode {};
struct SetEditMode {
    Element Mode;
};
struct EnterLookThroughCamera {};
struct ExitLookThroughCamera {};
struct AnimateToCamera {};
struct Play {};
struct SetViewportShading {
    ViewportShadingMode Mode;
};
struct SelectAll {};
struct OrbitViewCamera {
    vec2 DeltaRad;
};
struct ZoomViewCamera {
    float Factor;
};
struct ApplyExciteImpact {
    entt::entity InstanceEntity;
    uint32_t VertexIndex;
};
struct ClearExciteImpacts {};
struct TickViewCamera {};
struct SetStudioEnvironment {
    uint32_t Index;
};
struct SetSourceIblIntensity {
    float Intensity;
};
struct ResetViewCamera {};
struct ResetViewportTheme {};
struct ResetPbrLighting {
    bool Rendered;
};
struct SetViewCameraTarget {
    vec3 Target;
};
struct SetViewCameraLens {
    ::Camera Data;
};
// `Mask=0` removes the component.
struct SetPbrMeshFeaturesMask {
    entt::entity Entity;
    uint32_t Mask;
};
struct SetRotationUiMode {
    entt::entity Entity;
    int Index;
};
// `R` must already be normalized.
struct SetTransformRotationFromUi {
    entt::entity Entity;
    quat R;
    RotationUiVariant UiVariant;
};
struct BeginGizmoDrag {
    std::vector<std::pair<entt::entity, StartTransform>> Starts;
    std::vector<std::pair<entt::entity, float>> StartBoneLengths;
};
struct UpdateGizmoDragLocals {
    std::vector<std::pair<entt::entity, Transform>> Locals;
    std::vector<std::pair<entt::entity, float>> BoneDisplayScales;
};
struct UpdateGizmoMeshEditPending {
    std::unique_ptr<PendingTransform> Value;
};
struct EndGizmoDrag {};
} // namespace scene

template<typename T>
struct Update {
    entt::entity Entity;
    entt::id_type ComponentType;
    uint16_t Offset;
    T Value;
};

namespace detail {
// Itanium ABI: a non-virtual data-member pointer's bit pattern is the byte offset.
template<typename P>
constexpr std::ptrdiff_t MemPtrOffset(P p) {
    static_assert(sizeof(P) == sizeof(std::ptrdiff_t));
    return std::bit_cast<std::ptrdiff_t>(p);
}
} // namespace detail

template<typename Component, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Field Component::*member, Field value) {
    static_assert(std::is_trivially_copyable_v<Field>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    return {e, entt::type_hash<Component>::value(), uint16_t(detail::MemPtrOffset(member)), std::move(value)};
}

template<typename Component, typename Outer, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Outer Component::*outer, Field Outer::*inner, Field value) {
    static_assert(std::is_trivially_copyable_v<Field>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    return {e, entt::type_hash<Component>::value(), uint16_t(detail::MemPtrOffset(outer) + detail::MemPtrOffset(inner)), std::move(value)};
}

template<typename Component, typename Outer, typename Middle, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Outer Component::*outer, Middle Outer::*middle, Field Middle::*inner, Field value) {
    static_assert(std::is_trivially_copyable_v<Field>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    return {e, entt::type_hash<Component>::value(), uint16_t(detail::MemPtrOffset(outer) + detail::MemPtrOffset(middle) + detail::MemPtrOffset(inner)), std::move(value)};
}

struct SetTag {
    entt::entity Entity;
    entt::id_type TagType;
    bool Present;
};

template<typename Tag>
constexpr SetTag SetTagOf(entt::entity e, bool present) {
    return {e, entt::type_hash<Tag>::value(), present};
}

template<typename T>
struct Replace {
    entt::entity Entity;
    T Value;
};

// Specialization keeps the variant alt small for heavy components.
template<>
struct Replace<PhysicsMotion> {
    entt::entity Entity;
    std::unique_ptr<PhysicsMotion> Value;
};

struct DestroyEntity {
    entt::entity Entity;
};

namespace physics {
struct SetName {
    entt::entity Entity;
    entt::id_type ComponentType;
    std::string Name;
};
template<typename T>
inline SetName SetNameOf(entt::entity e, std::string name) {
    return {e, entt::type_hash<T>::value(), std::move(name)};
}

struct SetMotionType {
    enum class Type : uint8_t { None,
                                Static,
                                Kinematic,
                                Dynamic };
    entt::entity Entity;
    Type Value;
};

// `LockKind=true` locks `ColliderPolicy.LockedKind`; set it when the variant alternative changed.
struct SetColliderShape {
    entt::entity Entity;
    PhysicsShape Shape;
    bool LockKind;
};

struct AddTrigger {
    entt::entity Entity;
};
struct RemoveTriggerNodes {
    entt::entity Entity;
};

// Create a new named entity carrying component `ComponentType`, named "<Prefix> <ordinal>".
struct CreateNamed {
    entt::id_type ComponentType;
    std::string_view Prefix;
};
template<typename T>
constexpr CreateNamed CreateNamedOf(std::string_view prefix) {
    return {entt::type_hash<T>::value(), prefix};
}

// `Add=true` appends iff not present; `Add=false` erases all occurrences.
struct ToggleFilterEntity {
    entt::entity FilterEntity;
    std::vector<entt::entity> CollisionFilter::*Field;
    entt::entity SystemEntity;
    bool Add;
};

template<typename T>
struct SetJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
    uint32_t Index;
    std::unique_ptr<T> Value;
};
template<typename T>
struct AddJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
};
template<typename T>
struct DeleteJointVecItem {
    entt::entity JointDefEntity;
    std::vector<T> PhysicsJointDef::*Field;
    uint32_t Index;
};
} // namespace physics

namespace audio {
struct SetModel {
    entt::entity Entity;
    SoundVerticesModel Model;
};
// `VertexIndex` indexes `SoundVertices::Vertices`; `MeshVertex` is the mesh handle stored there.
struct SetExciteVertex {
    entt::entity Entity;
    entt::entity MeshEntity;
    uint32_t VertexIndex;
    uint32_t MeshVertex;
};
struct SetActiveElementFromDsp {
    entt::entity MeshEntity;
    uint32_t Vertex;
};
struct StartExcite {
    entt::entity Entity;
    uint32_t Vertex;
};
struct StopExcite {
    entt::entity Entity;
};
struct DeleteSoundObject {
    entt::entity Entity;
};
struct StartRecording {
    entt::entity Entity;
    uint32_t FrameCount;
};
struct OpenModalForm {
    entt::entity Entity;
    std::unique_ptr<ModalModelCreateInfo> Info;
};
struct CancelModalForm {
    entt::entity Entity;
};
struct SubmitModalForm {
    entt::entity Entity;
    entt::entity MeshEntity;
};
struct AcceptModalGenerationResult {
    struct Data {
        ModalModes Modes;
        TetMeshData Tets;
    };
    entt::entity Entity;
    entt::entity MeshEntity;
    std::unique_ptr<Data> D;
};
struct AssignVertexSamples {
    struct Data {
        std::vector<uint32_t> MeshVertices;
        std::filesystem::path Path;
        std::vector<float> Frames;
    };
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::unique_ptr<Data> D;
};
struct SetVertexSamples {
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::vector<uint32_t> MeshVertices;
    std::vector<std::pair<std::filesystem::path, std::vector<float>>> Samples;
};
struct RemoveVertexSamples {
    entt::entity SceneEntity;
    entt::entity SoundEntity;
    std::vector<uint32_t> MeshVertices;
};
struct SetModalFormMaterial {
    entt::entity Entity;
    std::unique_ptr<AcousticMaterial> Material;
};
} // namespace audio

namespace timeline {
struct TogglePlay {};
struct SetFrame {
    int Frame;
};
struct SetStartFrame {
    int Frame;
};
struct SetEndFrame {
    int Frame;
};
struct JumpToStart {};
struct JumpToEnd {};
} // namespace timeline

namespace bone {
struct Add {};
struct Extrude {};
struct DuplicateSelected {};
struct DeleteSelected {};
struct ClearSelectedTransforms {
    bool Position{false}, Rotation{false}, Scale{false};
};
// BoneDisplayScale is written without firing the reactive.
struct SetEditHeadTailRoll {
    entt::entity Entity;
    vec3 LocalP;
    quat LocalR;
    float DisplayScale;
};
struct SetConstraintTarget {
    entt::entity Entity;
    uint32_t Index;
    entt::entity Target;
};
struct SetConstraintInfluence {
    entt::entity Entity;
    uint32_t Index;
    float Influence;
};
struct SetConstraintChildOfInverse {
    entt::entity Entity;
    uint32_t Index;
    std::unique_ptr<mat4> Inverse;
};
struct DeleteConstraint {
    entt::entity Entity;
    uint32_t Index;
};
enum class BoneConstraintKind : uint8_t { CopyTransforms,
                                          ChildOf };
struct AddConstraint {
    entt::entity Entity;
    BoneConstraintKind Kind;
};
} // namespace bone

using Action = std::variant<
    selection::Select,
    selection::ToggleSelected,
    selection::SelectBone,
    selection::ExtendActive,
    selection::ExtendBoneActive,
    selection::SetBoneSelectionPart,
    selection::DeselectAll,
    selection::SnapshotBoxSelectBaseline,
    selection::ClearBoxSelectBaseline,
    selection::ApplyBoxSelectObjectHits,
    selection::ApplyBoxSelectBoneHits,
    selection::ApplyEditElementClick,
    selection::ApplyTreeSelection,
    object::Delete,
    object::Duplicate,
    object::DuplicateLinked,
    object::ToggleHidden,
    object::SetSelectedVisible,
    object::ParentToActive,
    object::ClearParent,
    object::AddEmpty,
    object::AddArmature,
    object::AddCamera,
    object::AddLight,
    object::AddMeshPrimitive,
    object::ImportMesh,
    object::ReplaceMesh,
    project::ClearMeshes,
    scene::SetInteractionMode,
    scene::CycleInteractionMode,
    scene::SetEditMode,
    scene::EnterLookThroughCamera,
    scene::ExitLookThroughCamera,
    scene::AnimateToCamera,
    scene::Play,
    scene::SetViewportShading,
    scene::SelectAll,
    scene::OrbitViewCamera,
    scene::ZoomViewCamera,
    scene::ApplyExciteImpact,
    scene::ClearExciteImpacts,
    scene::TickViewCamera,
    scene::SetStudioEnvironment,
    scene::SetSourceIblIntensity,
    scene::ResetViewCamera,
    scene::ResetViewportTheme,
    scene::ResetPbrLighting,
    scene::SetViewCameraTarget,
    scene::SetViewCameraLens,
    scene::SetPbrMeshFeaturesMask,
    scene::SetRotationUiMode,
    scene::SetTransformRotationFromUi,
    scene::BeginGizmoDrag,
    scene::UpdateGizmoDragLocals,
    scene::UpdateGizmoMeshEditPending,
    scene::EndGizmoDrag,
    Update<bool>,
    Update<uint8_t>,
    Update<uint32_t>,
    Update<float>,
    Update<double>,
    Update<vec3>,
    Update<vec4>,
    Update<entt::entity>,
    Update<DebugChannel>,
    Update<CollideMode>,
    Update<PhysicsCombineMode>,
    Update<PhysicsDriveType>,
    Update<PhysicsDriveMode>,
    Update<vk::ClearColorValue>,
    Update<std::optional<uint32_t>>,
    SetTag,
    DestroyEntity,
    Replace<Camera>,
    Replace<MaterialDirty>,
    Replace<MeshMaterialAssignment>,
    Replace<MeshMaterialSlotSelection>,
    Replace<PhysicsMotion>,
    Replace<PunctualLight>,
    Replace<RealImpactActiveMicrophone>,
    physics::SetName,
    physics::SetMotionType,
    physics::SetColliderShape,
    physics::AddTrigger,
    physics::RemoveTriggerNodes,
    physics::CreateNamed,
    physics::ToggleFilterEntity,
    physics::SetJointVecItem<PhysicsJointLimit>,
    physics::AddJointVecItem<PhysicsJointLimit>,
    physics::DeleteJointVecItem<PhysicsJointLimit>,
    physics::SetJointVecItem<PhysicsJointDrive>,
    physics::AddJointVecItem<PhysicsJointDrive>,
    physics::DeleteJointVecItem<PhysicsJointDrive>,
    audio::SetModel,
    audio::SetExciteVertex,
    audio::SetActiveElementFromDsp,
    audio::StartExcite,
    audio::StopExcite,
    audio::DeleteSoundObject,
    audio::StartRecording,
    audio::OpenModalForm,
    audio::CancelModalForm,
    audio::SubmitModalForm,
    audio::AcceptModalGenerationResult,
    audio::AssignVertexSamples,
    audio::SetVertexSamples,
    audio::RemoveVertexSamples,
    audio::SetModalFormMaterial,
    timeline::TogglePlay,
    timeline::SetFrame,
    timeline::SetStartFrame,
    timeline::SetEndFrame,
    timeline::JumpToStart,
    timeline::JumpToEnd,
    bone::Add,
    bone::Extrude,
    bone::DuplicateSelected,
    bone::DeleteSelected,
    bone::ClearSelectedTransforms,
    bone::SetEditHeadTailRoll,
    bone::SetConstraintTarget,
    bone::SetConstraintInfluence,
    bone::SetConstraintChildOfInverse,
    bone::DeleteConstraint,
    bone::AddConstraint>;

using FallibleAction = std::variant<
    project::LoadGltf,
    project::SaveGltf,
    project::LoadRealImpact>;

} // namespace action
