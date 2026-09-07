#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "render/Instance.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionOps.h"
#include <entt/entity/registry.hpp>
namespace {
// RenderInstance is derived from Instance + !Hidden.
// ObjectId 0 means on_construct<RenderInstance> fills it. BufferIndex UINT32_MAX means SyncModelsBuffers assigns it.
void EnsureRenderInstance(entt::registry &r, entt::entity e) {
    if (!r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX, 0u);
}
} // namespace

void Show(entt::registry &r, entt::entity e) {
    r.remove<Hidden>(e);
    if (r.all_of<Instance>(e)) EnsureRenderInstance(r, e); // re-show after a prior Hide
}

void Hide(entt::registry &r, entt::entity e) {
    r.emplace_or_replace<Hidden>(e); // OnConstructHidden removes the RenderInstance
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

entt::entity AddMeshInstance(entt::registry &r, entt::entity mesh_entity, const MeshInstanceCreateInfo &info) {
    const auto e = r.create();
    r.emplace<Instance>(e, mesh_entity);
    r.emplace<ObjectKind>(e, ObjectType::Mesh);
    r.emplace<Transform>(e, info.Transform);
    EmplaceUniqueName(r, e, info.Name);
    Show(r, e);
    if (!info.Visible) Hide(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

std::pair<entt::entity, entt::entity> AddMesh(entt::registry &r, uint32_t store_id, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = r.create();
    r.emplace<MeshHandle>(mesh_entity, MeshHandle{store_id});
    return {mesh_entity, info ? AddMeshInstance(r, mesh_entity, *info) : entt::null};
}

entt::entity CreateExtrasObject(entt::registry &r, ObjectType type, const ObjectCreateInfo &info, std::string_view default_name) {
    // The buffer starts empty and its wireframe is built later from the object's params.
    const auto buffer_entity = r.create();
    r.emplace<ObjectExtrasTag>(buffer_entity);
    const auto e = r.create();
    r.emplace<ObjectKind>(e, type);
    r.emplace<Instance>(e, buffer_entity);
    r.emplace<Transform>(e, info.Transform);
    EmplaceUniqueName(r, e, info.Name.empty() ? default_name : info.Name);
    Show(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

entt::entity AddEmpty(entt::registry &r, MeshStore &, const ObjectCreateInfo &info) {
    return CreateExtrasObject(r, ObjectType::Empty, info, "Empty");
}

entt::entity AddCamera(entt::registry &r, MeshStore &, const ObjectCreateInfo &info, std::optional<Camera> props) {
    const auto entity = CreateExtrasObject(r, ObjectType::Camera, info, "Camera");
    r.emplace<Camera>(entity, props.value_or(Camera{Defaults::PerspectiveCamera}));
    return entity;
}

entt::entity CreateBoneEntity(entt::registry &r, entt::entity arm_obj_entity, const Armature &armature, uint32_t bone_index, entt::entity parent_entity) {
    const auto &bone = armature.Bones[bone_index];
    const auto bone_entity = r.create();
    r.emplace<BoneIndex>(bone_entity, bone_index);
    r.emplace<SubElementOf>(bone_entity, arm_obj_entity);
    r.emplace<Instance>(bone_entity, arm_obj_entity);
    EmplaceUniqueName(r, bone_entity, bone.Name);
    r.emplace<BoneDisplayScale>(bone_entity, ComputeBoneDisplayScale(armature, bone_index));
    const Transform bone_transform{bone.RestLocal.P, bone.RestLocal.R, vec3{1}};
    r.emplace<Transform>(bone_entity, bone_transform);
    SetParent(r, bone_entity, parent_entity);
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

void CreateBoneInstances(entt::registry &r, MeshStore &meshes, entt::entity arm_obj_entity, entt::entity arm_data_entity) {
    const auto &armature = r.get<const Armature>(arm_data_entity);
    const uint32_t n = armature.Bones.size();
    if (n == 0) return;

    const auto bone_data = primitive::BoneOctahedron(1.f);
    const auto bone_store_id = meshes.AllocateVertexBuffer(bone_data.Mesh.Positions, bone_data.Attrs);
    r.emplace<VertexStoreId>(arm_obj_entity, bone_store_id);

    std::vector<entt::entity> bone_entities(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto parent_index = armature.Bones[i].ParentIndex;
        const auto parent = parent_index == InvalidBoneIndex ? arm_obj_entity : bone_entities[parent_index];
        bone_entities[i] = CreateBoneEntity(r, arm_obj_entity, armature, i, parent);
    }
    auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
    arm_obj.BoneEntities = std::move(bone_entities);

    auto sphere_data = primitive::BoneSphereDisc();
    const auto sphere_store_id = meshes.AllocateVertexBuffer(sphere_data.Mesh.Positions, {});
    const auto joint_entity = r.create();
    r.emplace<BoneJoint>(joint_entity);
    r.emplace<VertexStoreId>(joint_entity, sphere_store_id);

    for (const auto bone_entity : arm_obj.BoneEntities) CreateBoneJoints(r, arm_obj_entity, bone_entity, joint_entity);
    arm_obj.JointEntity = joint_entity;
}

entt::entity AddLight(entt::registry &r, MeshStore &, const ObjectCreateInfo &info, std::optional<PunctualLight> props) {
    const auto entity = CreateExtrasObject(r, ObjectType::Light, info, "Light");
    // PunctualLight is the canonical per-light data, the GPU Lights buffer is registered from it later.
    r.emplace<PunctualLight>(entity, props.value_or(Defaults::MakePunctualLight(PunctualLightType::Point)));
    return entity;
}
