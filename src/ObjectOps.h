#pragma once

#include "Camera.h"
#include "Entity.h" // ObjectType
#include "ObjectCreateInfo.h"
#include "gpu/PunctualLight.h"

#include <filesystem>
#include <span>

struct Armature;
struct Mesh;
struct MeshStore;
struct GpuBuffers;

std::string CreateName(entt::registry &, std::string_view prefix);
void OnDestroyName(entt::registry &, entt::entity);

// Reactive handlers for RenderInstance lifecycle.
void AssignRenderInstanceObjectId(entt::registry &, entt::entity);
void EmitPendingHideOnRenderInstanceDestroy(entt::registry &, entt::entity);

// Idempotent visibility helpers; emplace/remove RenderInstance, the reactive handlers do the rest.
void Show(entt::registry &, entt::entity);
void Hide(entt::registry &, entt::entity);

void Select(entt::registry &, entt::entity);
void ToggleSelected(entt::registry &, entt::entity);
void ApplySelectBehavior(entt::registry &, entt::entity, MeshInstanceCreateInfo::SelectBehavior);

// Entity creation. None apply SelectBehavior — callers do that after.
std::pair<entt::entity, entt::entity> AddMesh(entt::registry &, MeshStore &, Mesh &&, std::optional<MeshInstanceCreateInfo> = {});
entt::entity AddMeshInstance(entt::registry &, entt::entity mesh_entity, MeshInstanceCreateInfo);

entt::entity CreateExtrasBufferEntity(entt::registry &, MeshStore &, GpuBuffers &, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes = {}, std::span<const uint32_t> edge_indices = {});
entt::entity CreateExtrasObject(entt::registry &, MeshStore &, GpuBuffers &, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices, ObjectType, ObjectCreateInfo, std::string_view default_name);

entt::entity AddEmpty(entt::registry &, MeshStore &, GpuBuffers &, ObjectCreateInfo = {});
entt::entity AddCamera(entt::registry &, MeshStore &, GpuBuffers &, ObjectCreateInfo = {}, std::optional<Camera> = {});
entt::entity AddLight(entt::registry &, MeshStore &, GpuBuffers &, ObjectCreateInfo = {}, std::optional<PunctualLight> = {});
entt::entity AddArmature(entt::registry &, MeshStore &, ObjectCreateInfo = {});

// Loads a mesh file (with its materials/textures) and creates the mesh + instance entities.
std::pair<entt::entity, entt::entity> ImportMesh(entt::registry &, const std::filesystem::path &, MeshInstanceCreateInfo);

// Object teardown (counterpart to the Add* creators above).
void Destroy(entt::registry &, entt::entity viewport, entt::entity);
void ClearMeshes(entt::registry &, entt::entity viewport);
void DestroyArmatureData(entt::registry &, entt::entity arm_obj_entity);

entt::entity CreateBoneEntity(entt::registry &, entt::entity arm_obj_entity, const Armature &, uint32_t bone_index, entt::entity parent_entity);
void CreateBoneJoints(entt::registry &, entt::entity arm_obj_entity, entt::entity bone_entity, entt::entity joint_entity);
void CreateBoneInstances(entt::registry &, MeshStore &, entt::entity arm_obj_entity, entt::entity arm_data_entity);
