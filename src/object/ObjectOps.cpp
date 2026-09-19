#include "object/ObjectOps.h"
#include "assets/MeshImport.h"
#include "project/Assets.h"
#include "state/Scene.h"

#include "CameraTypes.h"
#include "Path.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "assets/MaterialImport.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MeshBuffers.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportEvents.h"

#include <format>

using std::ranges::any_of, std::ranges::find, std::ranges::to;

namespace {
// True if any component of type C has `C.*field == target`.
template<typename C, typename F>
bool AnyComponentRefersTo(state::Scene &r, F C::*field, state::Entity target) {
    return any_of(r.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}
} // namespace

void DestroyArmatureData(state::Scene &r, state::Entity arm_obj_entity) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &arm = r.edit<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != state::Null) {
        if (auto *mb = r.try_edit<MeshBuffers>(arm.JointEntity)) ReleaseMeshBuffers(r, *mb);
        if (auto *ref = r.try_get<VertexStoreId>(arm.JointEntity)) meshes.Release(ref->StoreId);
        if (auto *models = r.try_get<ModelsBuffer>(arm.JointEntity)) FreeInstanceRange(r, models->InstanceRange);
        r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, PendingHide>(arm.JointEntity);
        r.destroy(arm.JointEntity);
        arm.JointEntity = state::Null;
    }
    if (auto *mb = r.try_edit<MeshBuffers>(arm_obj_entity)) ReleaseMeshBuffers(r, *mb);
    if (auto *adj = r.try_get<BoneAdjacencyIndices>(arm_obj_entity)) ReleaseEdgeIndices(r, adj->Indices);
    if (auto *ref = r.try_get<VertexStoreId>(arm_obj_entity)) meshes.Release(ref->StoreId);
    if (auto *models = r.try_get<ModelsBuffer>(arm_obj_entity)) FreeInstanceRange(r, models->InstanceRange);
    r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, BoneAdjacencyIndices, PendingHide>(arm_obj_entity);
}

void Destroy(state::Scene &r, state::Entity viewport, state::Entity e) {
    auto &meshes = r.Context.get<MeshStore>();
    if (r.all_of<LookingThrough>(e)) ClearLookThrough(r, viewport);
    { // Clear relationships
        ClearParent(r, e);
        std::vector<state::Entity> children;
        for (auto child : Children{&r, e}) children.emplace_back(child);
        for (const auto child : children) ClearParent(r, child);
    }

    state::Entity buffer_entity = state::Null;
    if (const auto *instance = r.try_get<Instance>(e)) {
        if (HasMesh(r, instance->Entity) || r.all_of<ObjectExtrasTag>(instance->Entity)) buffer_entity = instance->Entity;
        Hide(r, e);
    }
    std::vector<state::Entity> armature_data_entities;
    auto try_add_armature_data = [&](state::Entity data_entity) {
        if (r.valid(data_entity) && find(armature_data_entities, data_entity) == armature_data_entities.end()) {
            armature_data_entities.emplace_back(data_entity);
        }
    };
    if (const auto *armature = r.try_get<ArmatureObject>(e)) try_add_armature_data(armature->Entity);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) try_add_armature_data(armature_modifier->ArmatureEntity);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) try_add_armature_data(bone_attachment->ArmatureEntity);

    if (const auto *light_index = r.try_get<LightIndex>(e)) {
        r.Context.get<GpuBuffers>().PendingLightRemovals.emplace_back(light_index->Value);
    }

    if (r.all_of<ArmatureObject>(e)) {
        auto &arm = r.get<ArmatureObject>(e);
        auto destroy_visible = [&](state::Entity entity) {
            Hide(r, entity);
            r.destroy(entity);
        };
        for (const auto bone_entity : arm.BoneEntities) {
            if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                if (joints->Head != state::Null) destroy_visible(joints->Head);
                if (joints->Tail != state::Null) destroy_visible(joints->Tail);
            }
            r.remove<BoneJointEntities>(bone_entity);
        }

        // Destroy children before parents (reverse of topological order) so ClearParent can access the parent's SceneNode to unlink the child.
        for (auto it = arm.BoneEntities.rbegin(); it != arm.BoneEntities.rend(); ++it) {
            ClearParent(r, *it);
            destroy_visible(*it);
        }
        DestroyArmatureData(r, e);
    }

    r.destroy(e);

    if (r.valid(buffer_entity)) {
        if (!AnyComponentRefersTo(r, &Instance::Entity, buffer_entity)) {
            if (auto *mesh_buffers = r.try_edit<MeshBuffers>(buffer_entity)) ReleaseMeshBuffers(r, *mesh_buffers);
            if (const auto *vs = r.try_get<VertexStoreId>(buffer_entity)) meshes.Release(vs->StoreId);
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

    // Release imported textures and reset the material when the final instance is removed.
    // Clear the persistent texture manifest at the same time.
    if (r.view<Instance>().empty()) {
        ResetImportedTexturesAndMaterials(r);
        r.remove<MaterializedTextures>(viewport);
    }
}

void ClearMeshes(state::Scene &r, state::Entity viewport) {
    for (const auto e : r.view<Instance>(state::Exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
}

std::pair<state::Entity, state::Entity> ImportMesh(state::Scene &r, state::Entity viewport, const std::filesystem::path &path, MeshInstanceCreateInfo info, bool deduplicate) {
    const auto stored_path = project::ResolveAsset(r, path);
    auto result = ReadMeshFile(stored_path);
    if (!result) throw std::runtime_error(result.error());

    // `deduplicate` merges vertices identical in every vertex-domain channel, keeping per-corner UVs and normals.
    auto created = CreateMesh(r, {.Data = std::move(result->Mesh), .Attrs = std::move(result->Attrs), .Primitives = std::move(result->Primitives), .Weld = deduplicate});
    if (!result->Materials.empty()) ImportObjPlyMaterials(r, viewport, result->Materials, stored_path, created.StoreId);

    const auto entities = ::AddMesh(r, created.StoreId, std::move(info));
    if (!created.AuthoredCornerNormals.empty()) r.emplace<AuthoredCornerNormals>(entities.first, std::move(created.AuthoredCornerNormals));
    r.emplace<Path>(entities.first, path);
    return entities;
}

void RequestImportMesh(state::Scene &r, state::Entity viewport, std::filesystem::path path, MeshInstanceCreateInfo info) {
    r.emplace_or_replace<PendingImportMesh>(viewport, std::move(path), std::move(info));
}
