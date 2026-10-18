#pragma once

#include "CameraTypes.h"
#include "object/ObjectCreateInfo.h"
#include "render/LightComponents.h"
#include "scene/Entity.h" // ObjectType

#include <expected>
#include <filesystem>

struct Armature;
struct MeshStore;

// These operations are idempotent.
void Show(state::Registry &, state::Entity);
void Hide(state::Registry &, state::Entity);
// Hides or shows the node and its descendants from their Visibility flags and the parent's Hidden state.
// Runs only when the node's own effective visibility changed.
void ApplyVisibility(state::Registry &, state::Entity);

void ApplySelectBehavior(state::Registry &, state::Entity, MeshInstanceCreateInfo::SelectBehavior);

// Callers apply SelectBehavior after entity creation.
std::pair<state::Entity, state::Entity> AddMesh(state::Registry &, uint32_t store_id, std::optional<MeshInstanceCreateInfo> = {});
state::Entity AddMeshInstance(state::Registry &, state::Entity mesh_entity, const MeshInstanceCreateInfo &);

state::Entity CreateExtrasObject(state::Registry &, ObjectType, const ObjectCreateInfo &, std::string_view default_name);

state::Entity AddEmpty(state::Registry &, MeshStore &, const ObjectCreateInfo & = {});
state::Entity AddCamera(state::Registry &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<CameraLens> = {});
state::Entity AddLight(state::Registry &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<PunctualLight> = {});

std::expected<std::pair<state::Entity, state::Entity>, std::string> ImportMesh(state::Registry &, state::Entity viewport, const std::filesystem::path &, MeshInstanceCreateInfo, bool deduplicate = false);

void RequestImportMesh(state::Registry &, state::Entity viewport, std::filesystem::path, MeshInstanceCreateInfo);

void Destroy(state::Registry &, state::Entity viewport, state::Entity);
void Destroy(state::Registry &, state::Entity viewport, std::span<const state::Entity>);
void ProcessObjectRemovals(state::Registry &, state::Entity viewport);
void ClearMeshes(state::Registry &, state::Entity viewport);
void DestroyArmatureData(state::Registry &, state::Entity arm_obj_entity);

state::Entity CreateBoneEntity(state::Registry &, state::Entity arm_obj_entity, const Armature &, uint32_t bone_index, state::Entity parent_entity, float display_scale);
void CreateBoneJoints(state::Registry &, state::Entity arm_obj_entity, state::Entity bone_entity, state::Entity joint_entity);
void CreateBoneInstances(state::Registry &, MeshStore &, state::Entity arm_obj_entity, state::Entity arm_data_entity);
