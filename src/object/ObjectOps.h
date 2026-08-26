#pragma once

#include "CameraTypes.h"
#include "gpu/PunctualLight.h"
#include "object/ObjectCreateInfo.h"
#include "scene/Entity.h" // ObjectType

#include <filesystem>
#include <span>

struct Armature;
struct MeshStore;

std::string CreateName(entt::registry &, std::string_view prefix);

// Idempotent visibility helpers that emplace or remove RenderInstance and leave the rest to the reactive handlers.
void Show(entt::registry &, entt::entity);
void Hide(entt::registry &, entt::entity);

void ApplySelectBehavior(entt::registry &, entt::entity, MeshInstanceCreateInfo::SelectBehavior);

// Entity creation. None apply SelectBehavior, so callers do that after.
std::pair<entt::entity, entt::entity> AddMesh(entt::registry &, uint32_t store_id, std::optional<MeshInstanceCreateInfo> = {});
entt::entity AddMeshInstance(entt::registry &, entt::entity mesh_entity, const MeshInstanceCreateInfo &);

// Creates a vertex-only buffer entity for derived overlay geometry (collider/AABB/tet wireframes).
entt::entity CreateExtrasObject(entt::registry &, ObjectType, const ObjectCreateInfo &, std::string_view default_name);

entt::entity AddEmpty(entt::registry &, MeshStore &, const ObjectCreateInfo & = {});
entt::entity AddCamera(entt::registry &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<Camera> = {});
entt::entity AddLight(entt::registry &, MeshStore &, const ObjectCreateInfo & = {}, std::optional<PunctualLight> = {});

// Loads a mesh file (with its materials/textures) and creates the mesh + instance entities.
std::pair<entt::entity, entt::entity> ImportMesh(entt::registry &, const std::filesystem::path &, MeshInstanceCreateInfo, bool deduplicate = false);

// Schedules a deferred mesh import.
void RequestImportMesh(entt::registry &, entt::entity viewport, std::filesystem::path, MeshInstanceCreateInfo);

void Destroy(entt::registry &, entt::entity viewport, entt::entity);
void ClearMeshes(entt::registry &, entt::entity viewport);
void DestroyArmatureData(entt::registry &, entt::entity arm_obj_entity);

entt::entity CreateBoneEntity(entt::registry &, entt::entity arm_obj_entity, const Armature &, uint32_t bone_index, entt::entity parent_entity);
void CreateBoneJoints(entt::registry &, entt::entity arm_obj_entity, entt::entity bone_entity, entt::entity joint_entity);
void CreateBoneInstances(entt::registry &, MeshStore &, entt::entity arm_obj_entity, entt::entity arm_data_entity);
