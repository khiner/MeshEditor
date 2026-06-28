#include "object/ObjectOps.h"

#include "CameraTypes.h"
#include "Path.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ExtrasComponents.h"
#include "object/ExtrasMesh.h"
#include "object/ObjectComponents.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialImport.h"
#include "render/MeshBuffers.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include "viewport/ViewCameraOps.h"

#include <entt/entity/registry.hpp>

#include <format>

using std::ranges::any_of, std::ranges::find, std::ranges::to;

std::string CreateName(entt::registry &r, std::string_view prefix) {
    auto &registry = r.ctx().get<NameRegistry>();
    std::string prefix_str{prefix};
    for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); ++i) {
        if (auto name = i == 0 ? prefix_str : std::format("{}_{}", prefix, i); !registry.Names.contains(name)) {
            registry.Names.insert(name);
            return name;
        }
    }
    assert(false);
    return prefix_str;
}

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

entt::entity AddMeshInstance(entt::registry &r, entt::entity mesh_entity, MeshInstanceCreateInfo info) {
    const auto e = r.create();
    r.emplace<Instance>(e, mesh_entity);
    r.emplace<ObjectKind>(e, ObjectType::Mesh);
    r.emplace<Transform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, info.Name));
    Show(r, e);
    if (!info.Visible) Hide(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

std::pair<entt::entity, entt::entity> AddMesh(entt::registry &r, MeshStore &, CreatedMesh &&mesh, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = r.create();
    r.emplace<MeshConnectivity>(mesh_entity, std::move(mesh.Connectivity));
    r.emplace<MeshHandle>(mesh_entity, MeshHandle{mesh.StoreId});
    return {mesh_entity, info ? AddMeshInstance(r, mesh_entity, *info) : entt::null};
}

entt::entity CreateExtrasBufferEntity(entt::registry &r, MeshStore &meshes, std::span<const vec3> positions, std::span<const uint32_t> edge_indices) {
    const auto buffer_entity = r.create();
    r.emplace<ObjectExtrasTag>(buffer_entity);
    r.emplace<OverlayVertexStoreId>(buffer_entity, meshes.AllocateOverlayVertexBuffer(positions).first);
    if (!edge_indices.empty()) {
        r.emplace<PendingEdgeIndices>(buffer_entity, std::vector<uint32_t>(edge_indices.begin(), edge_indices.end()));
    }
    return buffer_entity;
}

entt::entity CreateExtrasObject(entt::registry &r, ObjectType type, ObjectCreateInfo info, std::string_view default_name) {
    // The buffer starts empty; its wireframe is built later from the object's params.
    const auto buffer_entity = r.create();
    r.emplace<ObjectExtrasTag>(buffer_entity);
    const auto e = r.create();
    r.emplace<ObjectKind>(e, type);
    r.emplace<Instance>(e, buffer_entity);
    r.emplace<Transform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, info.Name.empty() ? default_name : info.Name));
    Show(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

entt::entity AddEmpty(entt::registry &r, MeshStore &, ObjectCreateInfo info) {
    return CreateExtrasObject(r, ObjectType::Empty, std::move(info), "Empty");
}

entt::entity AddCamera(entt::registry &r, MeshStore &, ObjectCreateInfo info, std::optional<Camera> props) {
    const auto entity = CreateExtrasObject(r, ObjectType::Camera, std::move(info), "Camera");
    r.emplace<Camera>(entity, props.value_or(Camera{Defaults::PerspectiveCamera}));
    return entity;
}

entt::entity CreateBoneEntity(entt::registry &r, entt::entity arm_obj_entity, const Armature &armature, uint32_t bone_index, entt::entity parent_entity) {
    const auto &bone = armature.Bones[bone_index];
    const auto bone_entity = r.create();
    r.emplace<BoneIndex>(bone_entity, bone_index);
    r.emplace<SubElementOf>(bone_entity, arm_obj_entity);
    r.emplace<Instance>(bone_entity, arm_obj_entity);
    r.emplace<Name>(bone_entity, CreateName(r, bone.Name));
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
    const auto bone_store_id = meshes.AllocateVertexBuffer(bone_data.Mesh.Positions, bone_data.Attrs).first;
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
    const auto sphere_store_id = meshes.AllocateVertexBuffer(sphere_data.Mesh.Positions, {}).first;
    const auto joint_entity = r.create();
    r.emplace<BoneJoint>(joint_entity);
    r.emplace<VertexStoreId>(joint_entity, sphere_store_id);

    for (const auto bone_entity : arm_obj.BoneEntities) CreateBoneJoints(r, arm_obj_entity, bone_entity, joint_entity);
    arm_obj.JointEntity = joint_entity;
}

entt::entity AddLight(entt::registry &r, MeshStore &, ObjectCreateInfo info, std::optional<PunctualLight> props) {
    const auto entity = CreateExtrasObject(r, ObjectType::Light, std::move(info), "Light");
    // PunctualLight is the canonical per-light data, the GPU Lights buffer is registered from it later.
    r.emplace<PunctualLight>(entity, props.value_or(Defaults::MakePunctualLight(PunctualLightType::Point)));
    return entity;
}

namespace {
// True if any component of type C has `C.*field == target`.
template<typename C, typename F>
bool AnyComponentRefersTo(entt::registry &r, F C::*field, entt::entity target) {
    return any_of(r.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}
} // namespace

void DestroyArmatureData(entt::registry &r, entt::entity arm_obj_entity) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &arm = r.get<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != entt::null) {
        if (auto *mb = r.try_get<MeshBuffers>(arm.JointEntity)) ReleaseMeshBuffers(r, *mb);
        if (auto *ref = r.try_get<VertexStoreId>(arm.JointEntity)) meshes.Release(ref->StoreId);
        if (auto *models = r.try_get<ModelsBuffer>(arm.JointEntity)) FreeInstanceRange(r, models->InstanceRange);
        r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, PendingHide>(arm.JointEntity);
        r.destroy(arm.JointEntity);
        arm.JointEntity = entt::null;
    }
    if (auto *mb = r.try_get<MeshBuffers>(arm_obj_entity)) ReleaseMeshBuffers(r, *mb);
    if (auto *adj = r.try_get<BoneAdjacencyIndices>(arm_obj_entity)) ReleaseEdgeIndices(r, adj->Indices);
    if (auto *ref = r.try_get<VertexStoreId>(arm_obj_entity)) meshes.Release(ref->StoreId);
    if (auto *models = r.try_get<ModelsBuffer>(arm_obj_entity)) FreeInstanceRange(r, models->InstanceRange);
    r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, BoneAdjacencyIndices, PendingHide>(arm_obj_entity);
}

void Destroy(entt::registry &r, entt::entity viewport, entt::entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    if (r.all_of<LookingThrough>(e)) ClearLookThrough(r, viewport);
    { // Clear relationships
        ClearParent(r, e);
        std::vector<entt::entity> children;
        for (auto child : Children{&r, e}) children.emplace_back(child);
        for (const auto child : children) ClearParent(r, child);
    }

    // Decrement SelectedInstanceCount before entity destruction (while all components are intact).
    // Cannot rely on on_destroy<Selected> — EnTT's pool removal order during r.destroy() is non-deterministic,
    // so Instance may already be gone when on_destroy<Selected> fires.
    if (r.all_of<Selected, Instance>(e)) {
        if (auto *count = r.try_get<SelectedInstanceCount>(r.get<Instance>(e).Entity))
            if (count->Value > 0) --count->Value;
    }

    entt::entity buffer_entity = entt::null;
    if (const auto *instance = r.try_get<Instance>(e)) {
        if (HasMesh(r, instance->Entity) || r.all_of<ObjectExtrasTag>(instance->Entity)) buffer_entity = instance->Entity;
        Hide(r, e);
    }
    std::vector<entt::entity> armature_data_entities;
    auto try_add_armature_data = [&](entt::entity data_entity) {
        if (r.valid(data_entity) && find(armature_data_entities, data_entity) == armature_data_entities.end()) {
            armature_data_entities.emplace_back(data_entity);
        }
    };
    if (const auto *armature = r.try_get<ArmatureObject>(e)) try_add_armature_data(armature->Entity);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) try_add_armature_data(armature_modifier->ArmatureEntity);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) try_add_armature_data(bone_attachment->ArmatureEntity);
    if (const auto *cw = r.try_get<ColliderWireframe>(e)) {
        for (uint8_t i = 0; i < cw->Count; ++i) {
            if (r.valid(cw->Instances[i])) {
                Hide(r, cw->Instances[i]);
                r.destroy(cw->Instances[i]);
            }
        }
    }
    if (const auto *bw = r.try_get<BBoxWireframe>(e); bw && r.valid(bw->Instance)) {
        Hide(r, bw->Instance);
        r.destroy(bw->Instance);
    }
    if (const auto *tw = r.try_get<TetWireframe>(e); tw && r.valid(tw->Instance)) {
        Hide(r, tw->Instance);
        r.destroy(tw->Instance);
    }

    if (const auto *light_index = r.try_get<LightIndex>(e)) {
        r.get_or_emplace<PendingLightRemovals>(viewport).Indices.emplace_back(light_index->Value);
    }

    if (r.all_of<ArmatureObject>(e)) {
        auto &arm = r.get<ArmatureObject>(e);
        auto destroy_visible = [&](entt::entity entity) {
            Hide(r, entity);
            r.destroy(entity);
        };
        for (const auto bone_entity : arm.BoneEntities) {
            if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                if (joints->Head != entt::null) destroy_visible(joints->Head);
                if (joints->Tail != entt::null) destroy_visible(joints->Tail);
            }
            r.remove<BoneJointEntities>(bone_entity);
        }

        // Destroy children before parents (reverse of topological order) so ClearParent
        // can access the parent's SceneNode to unlink the child.
        for (auto it = arm.BoneEntities.rbegin(); it != arm.BoneEntities.rend(); ++it) {
            ClearParent(r, *it);
            destroy_visible(*it);
        }
        DestroyArmatureData(r, e);
    }

    r.destroy(e);

    // If this was the last instance, destroy the buffer entity
    if (r.valid(buffer_entity)) {
        if (!AnyComponentRefersTo(r, &Instance::Entity, buffer_entity)) {
            if (auto *mesh_buffers = r.try_get<MeshBuffers>(buffer_entity)) {
                if (const auto *vcr = r.try_get<VertexClass>(buffer_entity)) {
                    ReleaseVertexClasses(r, vcr->Offset, mesh_buffers->Vertices.Count);
                }
                ReleaseMeshBuffers(r, *mesh_buffers);
            }
            if (const auto *vs = r.try_get<VertexStoreId>(buffer_entity)) meshes.Release(vs->StoreId);
            if (const auto *ov = r.try_get<OverlayVertexStoreId>(buffer_entity)) meshes.ReleaseOverlay(ov->StoreId);
            if (const auto *models = r.try_get<ModelsBuffer>(buffer_entity)) FreeInstanceRange(r, models->InstanceRange);
            r.destroy(buffer_entity);
        }
    }
    for (const auto armature_data_entity : armature_data_entities) {
        if (r.valid(armature_data_entity)) {
            const bool is_used = AnyComponentRefersTo(r, &ArmatureObject::Entity, armature_data_entity) ||
                AnyComponentRefersTo(r, &ArmatureModifier::ArmatureEntity, armature_data_entity) ||
                AnyComponentRefersTo(r, &BoneAttachment::ArmatureEntity, armature_data_entity);
            if (!is_used) r.destroy(armature_data_entity);
        }
    }

    // If no instances remain, release all imported textures and reset to the default material. The texture
    // manifest (Persistent restore input) tracks the released imported textures, so clear it in lockstep.
    if (r.view<Instance>().empty()) {
        ResetImportedTexturesAndMaterials(r);
        r.remove<MaterializedTextures>(viewport);
    }
}

void ClearMeshes(entt::registry &r, entt::entity viewport) {
    for (const auto e : r.view<Instance>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
}

std::pair<entt::entity, entt::entity> ImportMesh(entt::registry &r, const std::filesystem::path &path, MeshInstanceCreateInfo info, bool deduplicate) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto result = meshes.LoadMesh(path, deduplicate);
    if (!result) throw std::runtime_error(result.error());

    if (!result->Materials.empty()) ImportObjPlyMaterials(r, result->Materials, path, result->Mesh.StoreId);

    const auto entities = ::AddMesh(r, meshes, std::move(result->Mesh), std::move(info));
    r.emplace<Path>(entities.first, path);
    r.emplace<SmoothShading>(entities.first);
    return entities;
}
