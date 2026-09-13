#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "project/Registry.h"
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
    if (!r.all_of<RenderInstance>(e)) project::Emplace<RenderInstance>(r, e, r.get<Instance>(e).Entity, UINT32_MAX, 0u);
}
} // namespace

void Show(entt::registry &r, entt::entity e) {
    project::Remove<Hidden>(r, e);
    if (r.all_of<Instance>(e)) EnsureRenderInstance(r, e); // re-show after a prior Hide
}

void Hide(entt::registry &r, entt::entity e) {
    project::EmplaceOrReplace<Hidden>(r, e);
}

void ApplySelectBehavior(entt::registry &r, entt::entity e, MeshInstanceCreateInfo::SelectBehavior behavior) {
    switch (behavior) {
        case MeshInstanceCreateInfo::SelectBehavior::Exclusive:
            Select(r, e);
            break;
        case MeshInstanceCreateInfo::SelectBehavior::Additive:
            project::Emplace<Selected>(r, e);
            // Fallthrough
        case MeshInstanceCreateInfo::SelectBehavior::None:
            if (r.storage<Active>().empty()) project::Emplace<Active>(r, e);
            break;
    }
}

entt::entity AddMeshInstance(entt::registry &r, entt::entity mesh_entity, const MeshInstanceCreateInfo &info) {
    const auto e = project::Create(r);
    project::Emplace<Instance>(r, e, mesh_entity);
    project::Emplace<ObjectKind>(r, e, ObjectType::Mesh);
    project::Emplace<Transform>(r, e, info.Transform);
    EmplaceUniqueName(r, e, info.Name);
    Show(r, e);
    if (!info.Visible) Hide(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

std::pair<entt::entity, entt::entity> AddMesh(entt::registry &r, uint32_t store_id, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = project::Create(r);
    project::Emplace<MeshHandle>(r, mesh_entity, MeshHandle{store_id});
    return {mesh_entity, info ? AddMeshInstance(r, mesh_entity, *info) : entt::null};
}

entt::entity CreateExtrasObject(entt::registry &r, ObjectType type, const ObjectCreateInfo &info, std::string_view default_name) {
    // The buffer starts empty and its wireframe is built later from the object's params.
    const auto buffer_entity = project::Create(r);
    project::Emplace<ObjectExtrasTag>(r, buffer_entity);
    const auto e = project::Create(r);
    project::Emplace<ObjectKind>(r, e, type);
    project::Emplace<Instance>(r, e, buffer_entity);
    project::Emplace<Transform>(r, e, info.Transform);
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
    project::Emplace<Camera>(r, entity, props.value_or(Camera{Defaults::PerspectiveCamera}));
    return entity;
}

entt::entity CreateBoneEntity(entt::registry &r, entt::entity arm_obj_entity, const Armature &armature, uint32_t bone_index, entt::entity parent_entity) {
    const auto &bone = armature.Bones[bone_index];
    const auto bone_entity = project::Create(r);
    project::Emplace<BoneIndex>(r, bone_entity, bone_index);
    project::Emplace<SubElementOf>(r, bone_entity, arm_obj_entity);
    project::Emplace<Instance>(r, bone_entity, arm_obj_entity);
    EmplaceUniqueName(r, bone_entity, bone.Name);
    project::Emplace<BoneDisplayScale>(r, bone_entity, ComputeBoneDisplayScale(armature, bone_index));
    const Transform bone_transform{bone.RestLocal.P, bone.RestLocal.R, vec3{1}};
    project::Emplace<Transform>(r, bone_entity, bone_transform);
    SetParent(r, bone_entity, parent_entity);
    Show(r, bone_entity);
    return bone_entity;
}

void CreateBoneJoints(entt::registry &r, entt::entity arm_obj_entity, entt::entity bone_entity, entt::entity joint_entity) {
    auto make = [&](bool is_tail) {
        const auto e = project::Create(r);
        project::Emplace<SubElementOf>(r, e, arm_obj_entity);
        project::Emplace<Instance>(r, e, joint_entity);
        project::Emplace<BoneSubPartOf>(r, e, bone_entity, is_tail);
        Show(r, e);
        return e;
    };
    project::Emplace<BoneJointEntities>(r, bone_entity, make(false), make(true));
}

void CreateBoneInstances(entt::registry &r, MeshStore &meshes, entt::entity arm_obj_entity, entt::entity arm_data_entity) {
    const auto &armature = r.get<const Armature>(arm_data_entity);
    const uint32_t n = armature.Bones.size();
    if (n == 0) return;

    const auto bone_data = primitive::BoneOctahedron(1.f);
    const auto bone_store_id = meshes.AllocateVertexBuffer(bone_data.Mesh.Positions, bone_data.Attrs);
    project::Emplace<VertexStoreId>(r, arm_obj_entity, bone_store_id);

    std::vector<entt::entity> bone_entities(n);
    for (uint32_t i = 0; i < n; ++i) {
        const auto parent_index = armature.Bones[i].ParentIndex;
        const auto parent = parent_index == InvalidBoneIndex ? arm_obj_entity : bone_entities[parent_index];
        bone_entities[i] = CreateBoneEntity(r, arm_obj_entity, armature, i, parent);
    }
    auto &arm_obj = project::Mutable<ArmatureObject>(r, arm_obj_entity);
    arm_obj.BoneEntities = std::move(bone_entities);

    auto sphere_data = primitive::BoneSphereDisc();
    const auto sphere_store_id = meshes.AllocateVertexBuffer(sphere_data.Mesh.Positions, {});
    const auto joint_entity = project::Create(r);
    project::Emplace<BoneJoint>(r, joint_entity);
    project::Emplace<VertexStoreId>(r, joint_entity, sphere_store_id);

    for (const auto bone_entity : arm_obj.BoneEntities) CreateBoneJoints(r, arm_obj_entity, bone_entity, joint_entity);
    arm_obj.JointEntity = joint_entity;
}

entt::entity AddLight(entt::registry &r, MeshStore &, const ObjectCreateInfo &info, std::optional<PunctualLight> props) {
    const auto entity = CreateExtrasObject(r, ObjectType::Light, info, "Light");
    // PunctualLight is the canonical per-light data, the GPU Lights buffer is registered from it later.
    project::Emplace<PunctualLight>(r, entity, props.value_or(Defaults::MakePunctualLight(PunctualLightType::Point)));
    return entity;
}
