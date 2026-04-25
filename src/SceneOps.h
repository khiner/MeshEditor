#pragma once

#include "Entity.h" // ObjectType
#include "entt_fwd.h"
#include "gpu/PunctualLight.h"
#include "gpu/Transform.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>

struct Armature;
struct Mesh;
struct MeshStore;
struct SceneBuffers;

struct MeshInstanceCreateInfo {
    std::string Name{};
    Transform Transform{};

    enum class SelectBehavior {
        Exclusive,
        Additive,
        None,
    };
    SelectBehavior Select{SelectBehavior::Exclusive};
    bool Visible{true};
};

struct ObjectCreateInfo {
    std::string Name{};
    Transform Transform{};
    MeshInstanceCreateInfo::SelectBehavior Select{MeshInstanceCreateInfo::SelectBehavior::Exclusive};
};

// Singleton components on Scene::SceneEntity.
struct NameRegistry {
    std::unordered_set<std::string> Names;
};
struct ObjectIdCounter {
    uint32_t Next{1};
};

std::string CreateName(entt::registry &, entt::entity scene_entity, std::string_view prefix);
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

// Entity creation. None apply SelectBehavior — Scene wrappers do that after.
std::pair<entt::entity, entt::entity> AddMesh(entt::registry &, MeshStore &, entt::entity scene_entity, Mesh &&, std::optional<MeshInstanceCreateInfo> = {});
entt::entity AddMeshInstance(entt::registry &, entt::entity scene_entity, entt::entity mesh_entity, MeshInstanceCreateInfo);

entt::entity CreateExtrasBufferEntity(entt::registry &, MeshStore &, SceneBuffers &, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes = {}, std::span<const uint32_t> edge_indices = {});
entt::entity CreateExtrasObject(entt::registry &, MeshStore &, SceneBuffers &, entt::entity scene_entity, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices, ObjectType, ObjectCreateInfo, std::string_view default_name);

entt::entity AddEmpty(entt::registry &, MeshStore &, SceneBuffers &, entt::entity scene_entity, ObjectCreateInfo = {});
entt::entity AddCamera(entt::registry &, MeshStore &, SceneBuffers &, entt::entity scene_entity, ObjectCreateInfo = {});
entt::entity AddLight(entt::registry &, MeshStore &, SceneBuffers &, entt::entity scene_entity, ObjectCreateInfo = {}, std::optional<PunctualLight> = {});

entt::entity CreateBoneEntity(entt::registry &, entt::entity scene_entity, entt::entity arm_obj_entity, const Armature &, uint32_t bone_index, entt::entity parent_entity);
void CreateBoneJoints(entt::registry &, entt::entity arm_obj_entity, entt::entity bone_entity, entt::entity joint_entity);
void CreateBoneInstances(entt::registry &, MeshStore &, entt::entity scene_entity, entt::entity arm_obj_entity, entt::entity arm_data_entity);
