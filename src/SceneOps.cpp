#include "SceneOps.h"

#include <cassert>
#include <format>
#include <limits>
#include <vector>

#include <entt/entity/registry.hpp>

#include "Armature.h"
#include "Camera.h"
#include "Entity.h"
#include "ExtrasMesh.h"
#include "Instance.h"
#include "MeshComponents.h"
#include "SceneDefaults.h"
#include "SceneTree.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "numeric/mat4.h"
#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneInternalTypes.h"

namespace {
template<typename T>
T *FindSingleton(entt::registry &r) {
    auto view = r.view<T>();
    auto it = view.begin();
    return it == view.end() ? nullptr : &r.get<T>(*it);
}
} // namespace

std::string CreateName(entt::registry &r, entt::entity scene_entity, std::string_view prefix) {
    auto &registry = r.get<NameRegistry>(scene_entity);
    const std::string prefix_str{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        if (auto name = i == 0 ? prefix_str : std::format("{}_{}", prefix, i); !registry.Names.contains(name)) {
            registry.Names.insert(name);
            return name;
        }
    }
    assert(false);
    return prefix_str;
}

void OnDestroyName(entt::registry &r, entt::entity e) {
    if (auto *registry = FindSingleton<NameRegistry>(r)) {
        registry->Names.erase(r.get<const Name>(e).Value);
    }
}

void AssignRenderInstanceObjectId(entt::registry &r, entt::entity e) {
    auto &ri = r.get<RenderInstance>(e);
    if (ri.ObjectId != 0) return;
    if (auto *counter = FindSingleton<ObjectIdCounter>(r)) ri.ObjectId = counter->Next++;
}

void EmitPendingHideOnRenderInstanceDestroy(entt::registry &r, entt::entity e) {
    const auto &ri = r.get<const RenderInstance>(e);
    if (ri.BufferIndex == UINT32_MAX) return; // Same-frame show+hide — never synced to GPU.
    r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
}

void Show(entt::registry &r, entt::entity e) {
    if (r.all_of<RenderInstance>(e)) return;
    r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX, 0u); // ObjectId 0 → on_construct fills it.
}

void Hide(entt::registry &r, entt::entity e) {
    if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
}

void Select(entt::registry &r, entt::entity e) {
    r.clear<Selected>();
    if (e != entt::null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void ToggleSelected(entt::registry &r, entt::entity e) {
    if (e == entt::null) return;
    if (r.all_of<Selected>(e)) r.remove<Selected>(e);
    else r.emplace_or_replace<Selected>(e);
}

void ApplySelectBehavior(entt::registry &r, entt::entity e, MeshInstanceCreateInfo::SelectBehavior behavior) {
    switch (behavior) {
        case MeshInstanceCreateInfo::SelectBehavior::Exclusive:
            Select(r, e);
            break;
        case MeshInstanceCreateInfo::SelectBehavior::Additive:
            r.emplace<Selected>(e);
            // Fallthrough
        case MeshInstanceCreateInfo::SelectBehavior::None:
            if (r.storage<Active>().empty()) r.emplace<Active>(e);
            break;
    }
}

entt::entity AddMeshInstance(entt::registry &r, entt::entity scene_entity, entt::entity mesh_entity, MeshInstanceCreateInfo info) {
    const auto e = r.create();
    r.emplace<Instance>(e, mesh_entity);
    r.emplace<ObjectKind>(e, ObjectType::Mesh);
    r.emplace<Transform>(e, info.Transform);
    r.emplace<WorldTransform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, scene_entity, info.Name));
    Show(r, e);
    if (!info.Visible) Hide(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

std::pair<entt::entity, entt::entity> AddMesh(entt::registry &r, MeshStore &meshes, entt::entity scene_entity, Mesh &&mesh, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = r.create();
    r.emplace<MeshBuffers>(mesh_entity, meshes.GetVerticesRange(mesh.GetStoreId()), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<Mesh>(mesh_entity, std::move(mesh));
    return {mesh_entity, info ? AddMeshInstance(r, scene_entity, mesh_entity, *info) : entt::null};
}

entt::entity CreateExtrasBufferEntity(entt::registry &r, MeshStore &meshes, SceneBuffers &buffers, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices) {
    const auto buffer_entity = r.create();
    const auto store_id = meshes.AllocateVertexBuffer(positions, {}).first;
    r.emplace<MeshBuffers>(buffer_entity, meshes.GetVerticesRange(store_id), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<ObjectExtrasTag>(buffer_entity);
    r.emplace<VertexStoreId>(buffer_entity, store_id);
    if (!vertex_classes.empty()) {
        r.emplace<VertexClass>(buffer_entity, buffers.VertexClassBuffer.Allocate(vertex_classes).Offset);
    }
    if (!edge_indices.empty()) {
        r.emplace<PendingEdgeIndices>(buffer_entity, std::vector<uint32_t>(edge_indices.begin(), edge_indices.end()));
    }
    return buffer_entity;
}

entt::entity CreateExtrasObject(entt::registry &r, MeshStore &meshes, SceneBuffers &buffers, entt::entity scene_entity, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices, ObjectType type, ObjectCreateInfo info, std::string_view default_name) {
    const auto buffer_entity = CreateExtrasBufferEntity(r, meshes, buffers, positions, vertex_classes, edge_indices);
    const auto e = r.create();
    r.emplace<ObjectKind>(e, type);
    r.emplace<Instance>(e, buffer_entity);
    r.emplace<Transform>(e, info.Transform);
    r.emplace<WorldTransform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, scene_entity, info.Name.empty() ? default_name : info.Name));
    Show(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

entt::entity AddEmpty(entt::registry &r, MeshStore &meshes, SceneBuffers &buffers, entt::entity scene_entity, ObjectCreateInfo info) {
    static constexpr vec3 positions[] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, -1}};
    static constexpr uint32_t edges[] = {0, 1, 2, 3, 4, 5};
    return CreateExtrasObject(r, meshes, buffers, scene_entity, positions, {}, edges, ObjectType::Empty, std::move(info), "Empty");
}

entt::entity AddCamera(entt::registry &r, MeshStore &meshes, SceneBuffers &buffers, entt::entity scene_entity, ObjectCreateInfo info) {
    Camera camera{DefaultPerspectiveCamera()};
    auto mesh = BuildCameraFrustumMesh(camera);
    const auto entity = CreateExtrasObject(r, meshes, buffers, scene_entity, mesh.Positions, {}, mesh.CreateEdgeIndices(), ObjectType::Camera, std::move(info), "Camera");
    r.emplace<Camera>(entity, camera);
    return entity;
}

entt::entity CreateBoneEntity(entt::registry &r, entt::entity scene_entity, entt::entity arm_obj_entity, const Armature &armature, uint32_t bone_index, entt::entity parent_entity) {
    const auto &bone = armature.Bones[bone_index];
    const auto bone_entity = r.create();
    r.emplace<BoneIndex>(bone_entity, bone_index);
    r.emplace<SubElementOf>(bone_entity, arm_obj_entity);
    r.emplace<Instance>(bone_entity, arm_obj_entity);
    r.emplace<Name>(bone_entity, CreateName(r, scene_entity, bone.Name));
    r.emplace<BoneDisplayScale>(bone_entity, ComputeBoneDisplayScale(armature, bone_index));
    const Transform bone_transform{bone.RestLocal.P, bone.RestLocal.R, vec3{1}};
    r.emplace<Transform>(bone_entity, bone_transform);
    r.emplace<WorldTransform>(bone_entity, bone_transform);
    SetParent(r, bone_entity, parent_entity);
    r.emplace_or_replace<ParentInverse>(bone_entity, I4);
    Show(r, bone_entity);
    return bone_entity;
}

void CreateBoneJoints(entt::registry &r, entt::entity arm_obj_entity, entt::entity bone_entity, entt::entity joint_entity) {
    auto make = [&](bool is_tail) {
        const auto e = r.create();
        r.emplace<SubElementOf>(e, arm_obj_entity);
        r.emplace<Instance>(e, joint_entity);
        r.emplace<BoneSubPartOf>(e, bone_entity, is_tail);
        Show(r, e);
        return e;
    };
    r.emplace<BoneJointEntities>(bone_entity, make(false), make(true));
}

void CreateBoneInstances(entt::registry &r, MeshStore &meshes, entt::entity scene_entity, entt::entity arm_obj_entity, entt::entity arm_data_entity) {
    const auto &armature = r.get<const Armature>(arm_data_entity);
    const uint32_t n = armature.Bones.size();
    if (n == 0) return;

    const auto bone_data = primitive::BoneOctahedron(1.f);
    const auto bone_store_id = meshes.AllocateVertexBuffer(bone_data.Mesh.Positions, bone_data.Attrs).first;
    r.emplace<MeshBuffers>(arm_obj_entity, meshes.GetVerticesRange(bone_store_id), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<VertexStoreId>(arm_obj_entity, bone_store_id);

    std::vector<entt::entity> bone_entities(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto parent_index = armature.Bones[i].ParentIndex;
        const auto parent = parent_index == InvalidBoneIndex ? arm_obj_entity : bone_entities[parent_index];
        bone_entities[i] = CreateBoneEntity(r, scene_entity, arm_obj_entity, armature, i, parent);
    }
    auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
    arm_obj.BoneEntities = std::move(bone_entities);

    auto sphere_data = primitive::BoneSphereDisc();
    const auto sphere_store_id = meshes.AllocateVertexBuffer(sphere_data.Mesh.Positions, {}).first;
    const auto joint_entity = r.create();
    r.emplace<BoneJoint>(joint_entity);
    r.emplace<MeshBuffers>(joint_entity, meshes.GetVerticesRange(sphere_store_id), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<VertexStoreId>(joint_entity, sphere_store_id);

    for (const auto bone_entity : arm_obj.BoneEntities) CreateBoneJoints(r, arm_obj_entity, bone_entity, joint_entity);
    arm_obj.JointEntity = joint_entity;
}

entt::entity AddLight(entt::registry &r, MeshStore &meshes, SceneBuffers &buffers, entt::entity scene_entity, ObjectCreateInfo info, std::optional<PunctualLight> props) {
    auto light = props.value_or(SceneDefaults::MakePunctualLight(PunctualLightType::Point));
    auto wireframe = BuildLightMesh(light);
    const auto entity = CreateExtrasObject(r, meshes, buffers, scene_entity, wireframe.Data.Positions, wireframe.VertexClasses, {}, ObjectType::Light, std::move(info), "Light");
    r.emplace<LightIndex>(entity, r.storage<LightIndex>().size());
    r.emplace<SubmitDirty>(entity);
    r.emplace<LightWireframeDirty>(entity);
    // Defer SetLight: TransformSlotOffset needs InstanceArena slot and RenderInstance.BufferIndex,
    // which aren't available until SyncModelsBuffers runs. Store light data temporarily;
    // ProcessComponentEvents syncs the slot after SyncModelsBuffers.
    r.emplace<PunctualLight>(entity, light);
    return entity;
}
