#pragma once

#include "CameraTypes.h"
#include "mesh/PrimitiveType.h"
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

// Creates a mesh entity drawing the store record, whose procedure's Source node takes the record's state as its history step ends.
state::Entity AddMeshData(state::Registry &, uint32_t store_id);
// Objects created without a parent join the frame's scene.
// Callers apply SelectBehavior after entity creation.
std::pair<state::Entity, state::Entity> AddMesh(state::Registry &, uint32_t store_id, const MeshInstanceCreateInfo &);
// Adds a mesh whose procedure roots at the primitive's generator node, named after the primitive when the info has no name, and returns the mesh entity and its instance.
std::pair<state::Entity, state::Entity> AddMeshPrimitive(state::Registry &, const PrimitiveParameters &, const MeshInstanceCreateInfo &);
state::Entity AddMeshInstance(state::Registry &, state::Entity mesh_entity, const MeshInstanceCreateInfo &);

state::Entity CreateExtrasObject(state::Registry &, ObjectType, const ObjectCreateInfo &, std::string_view default_name);

state::Entity AddEmpty(state::Registry &, MeshStore &, const ObjectCreateInfo &);
state::Entity AddCamera(state::Registry &, MeshStore &, const ObjectCreateInfo &, std::optional<CameraLens> = {});
state::Entity AddLight(state::Registry &, MeshStore &, const ObjectCreateInfo &, std::optional<PunctualLight> = {});

std::expected<std::pair<state::Entity, state::Entity>, std::string> ImportMesh(state::Registry &, const std::filesystem::path &, const MeshInstanceCreateInfo &, bool deduplicate = false);

void RequestImportMesh(state::Registry &, std::filesystem::path, MeshInstanceCreateInfo);

void Destroy(state::Registry &, state::Entity);
void Destroy(state::Registry &, std::span<const state::Entity>);
void ProcessObjectRemovals(state::Registry &);
void ClearMeshes(state::Registry &);
void DestroyArmatureData(state::Registry &, state::Entity arm_obj_entity);

state::Entity CreateBoneEntity(state::Registry &, state::Entity arm_obj_entity, const Armature &, uint32_t bone_index, state::Entity parent_entity, float display_scale);
void CreateBoneJoints(state::Registry &, state::Entity arm_obj_entity, state::Entity bone_entity, state::Entity joint_entity);
void CreateBoneInstances(state::Registry &, MeshStore &, state::Entity arm_obj_entity, state::Entity arm_data_entity);
