#include "object/ObjectOps.h"
#include "assets/MeshImport.h"
#include "project/Assets.h"
#include "project/Registry.h"

#include "CameraTypes.h"
#include "Path.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
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
#include "viewport/ViewportEvents.h"

#include <entt/entity/registry.hpp>

#include <format>

using std::ranges::any_of, std::ranges::find, std::ranges::to;

namespace {
// True if any component of type C has `C.*field == target`.
template<typename C, typename F>
bool AnyComponentRefersTo(entt::registry &r, F C::*field, entt::entity target) {
    return any_of(r.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}
} // namespace

void DestroyArmatureData(entt::registry &r, entt::entity arm_obj_entity) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &arm = project::Mutable<ArmatureObject>(r, arm_obj_entity);
    if (arm.JointEntity != entt::null) {
        if (auto *mb = r.try_get<MeshBuffers>(arm.JointEntity)) ReleaseMeshBuffers(r, *mb);
        if (auto *ref = r.try_get<VertexStoreId>(arm.JointEntity)) meshes.Release(ref->StoreId);
        if (auto *models = r.try_get<ModelsBuffer>(arm.JointEntity)) FreeInstanceRange(r, models->InstanceRange);
        project::Remove<MeshBuffers, VertexStoreId, ModelsBuffer, PendingHide>(r, arm.JointEntity);
        project::Destroy(r, arm.JointEntity);
        arm.JointEntity = entt::null;
    }
    if (auto *mb = r.try_get<MeshBuffers>(arm_obj_entity)) ReleaseMeshBuffers(r, *mb);
    if (auto *adj = r.try_get<BoneAdjacencyIndices>(arm_obj_entity)) ReleaseEdgeIndices(r, adj->Indices);
    if (auto *ref = r.try_get<VertexStoreId>(arm_obj_entity)) meshes.Release(ref->StoreId);
    if (auto *models = r.try_get<ModelsBuffer>(arm_obj_entity)) FreeInstanceRange(r, models->InstanceRange);
    project::Remove<MeshBuffers, VertexStoreId, ModelsBuffer, BoneAdjacencyIndices, PendingHide>(r, arm_obj_entity);
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

    if (const auto *light_index = r.try_get<LightIndex>(e)) {
        project::GetOrEmplace<PendingLightRemovals>(r, viewport).Indices.emplace_back(light_index->Value);
    }

    if (r.all_of<ArmatureObject>(e)) {
        auto &arm = r.get<ArmatureObject>(e);
        auto destroy_visible = [&](entt::entity entity) {
            Hide(r, entity);
            project::Destroy(r, entity);
        };
        for (const auto bone_entity : arm.BoneEntities) {
            if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                if (joints->Head != entt::null) destroy_visible(joints->Head);
                if (joints->Tail != entt::null) destroy_visible(joints->Tail);
            }
            project::Remove<BoneJointEntities>(r, bone_entity);
        }

        // Destroy children before parents (reverse of topological order) so ClearParent can access the parent's SceneNode to unlink the child.
        for (auto it = arm.BoneEntities.rbegin(); it != arm.BoneEntities.rend(); ++it) {
            ClearParent(r, *it);
            destroy_visible(*it);
        }
        DestroyArmatureData(r, e);
    }

    project::Destroy(r, e);

    if (r.valid(buffer_entity)) {
        if (!AnyComponentRefersTo(r, &Instance::Entity, buffer_entity)) {
            if (auto *mesh_buffers = r.try_get<MeshBuffers>(buffer_entity)) ReleaseMeshBuffers(r, *mesh_buffers);
            if (const auto *vs = r.try_get<VertexStoreId>(buffer_entity)) meshes.Release(vs->StoreId);
            if (const auto *models = r.try_get<ModelsBuffer>(buffer_entity)) FreeInstanceRange(r, models->InstanceRange);
            project::Destroy(r, buffer_entity);
        }
    }
    for (const auto armature_data_entity : armature_data_entities) {
        if (r.valid(armature_data_entity)) {
            const bool is_used = AnyComponentRefersTo(r, &ArmatureObject::Entity, armature_data_entity) ||
                AnyComponentRefersTo(r, &ArmatureModifier::ArmatureEntity, armature_data_entity) ||
                AnyComponentRefersTo(r, &BoneAttachment::ArmatureEntity, armature_data_entity);
            if (!is_used) project::Destroy(r, armature_data_entity);
        }
    }

    // Release imported textures and reset the material when the final instance is removed.
    // Clear the persistent texture manifest at the same time.
    if (r.view<Instance>().empty()) {
        ResetImportedTexturesAndMaterials(r);
        project::Remove<MaterializedTextures>(r, viewport);
    }
}

void ClearMeshes(entt::registry &r, entt::entity viewport) {
    for (const auto e : r.view<Instance>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
}

std::pair<entt::entity, entt::entity> ImportMesh(entt::registry &r, entt::entity viewport, const std::filesystem::path &path, MeshInstanceCreateInfo info, bool deduplicate) {
    const auto stored_path = project::ResolveAsset(r, path);
    auto result = ReadMeshFile(stored_path);
    if (!result) throw std::runtime_error(result.error());

    // `deduplicate` merges vertices identical in every vertex-domain channel, keeping per-corner UVs and normals.
    const auto created = CreateMesh(r, {.Data = std::move(result->Mesh), .Attrs = std::move(result->Attrs), .Primitives = std::move(result->Primitives), .Weld = deduplicate});
    if (!result->Materials.empty()) ImportObjPlyMaterials(r, viewport, result->Materials, stored_path, created.StoreId);

    const auto entities = ::AddMesh(r, created.StoreId, std::move(info));
    project::Emplace<Path>(r, entities.first, path);
    return entities;
}

void RequestImportMesh(entt::registry &r, entt::entity viewport, std::filesystem::path path, MeshInstanceCreateInfo info) {
    project::EmplaceOrReplace<PendingImportMesh>(r, viewport, std::move(path), std::move(info));
}
