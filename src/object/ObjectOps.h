#pragma once

#include "CameraTypes.h"
#include "object/ObjectCreateInfo.h"
#include "render/LightComponents.h"
#include "scene/Entity.h" // ObjectType

#include <filesystem>
#include <span>

struct Armature;
struct MeshStore;

// These operations are idempotent.
void Show(state::Scene &, state::Entity);
void Hide(state::Scene &, state::Entity);
// Hides or shows the node and its descendants from their Visibility flags and the parent's Hidden state.
void ApplyVisibility(state::Scene &, state::Entity);

void ApplySelectBehavior(state::Scene &, state::Entity, MeshInstanceCreateInfo::SelectBehavior);

// Callers apply SelectBehavior after entity creation.
std::pair<state::Entity, state::Entity> AddMesh(state::Scene &, uint32_t store_id, std::optional<MeshInstanceCreateInfo> = {});
state::Entity AddMeshInstance(state::Scene &, state::Entity mesh_entity, const MeshInstanceCreateInfo &);

state::Entity CreateExtrasObject(state::Scene &, ObjectType, const ObjectCreateInfo &, std::string_view default_name);

state::Entity AddEmpty(state::Scene &, MeshStore &, const ObjectCreateInfo & = {});
state::Entity AddCamera(state::Scene &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<CameraLens> = {});
state::Entity AddLight(state::Scene &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<PunctualLight> = {});

std::pair<state::Entity, state::Entity> ImportMesh(state::Scene &, state::Entity viewport, const std::filesystem::path &, MeshInstanceCreateInfo, bool deduplicate = false);

void RequestImportMesh(state::Scene &, state::Entity viewport, std::filesystem::path, MeshInstanceCreateInfo);

void Destroy(state::Scene &, state::Entity viewport, state::Entity);
void ClearMeshes(state::Scene &, state::Entity viewport);
void DestroyArmatureData(state::Scene &, state::Entity arm_obj_entity);

state::Entity CreateBoneEntity(state::Scene &, state::Entity arm_obj_entity, const Armature &, uint32_t bone_index, state::Entity parent_entity);
void CreateBoneJoints(state::Scene &, state::Entity arm_obj_entity, state::Entity bone_entity, state::Entity joint_entity);
void CreateBoneInstances(state::Scene &, MeshStore &, state::Entity arm_obj_entity, state::Entity arm_data_entity);
