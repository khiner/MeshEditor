#include "object/ObjectOps.h"
#include "Profile.h"
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

namespace {
// Unlink each affected sibling list once, including surviving children of deleted parents.
void ClearRelationships(state::Scene &r, std::span<const state::Entity> entities) {
    std::vector<state::Entity> detached;
    const auto add = [&](state::Entity e) {
        if (r.all_of<SceneParent>(e)) detached.push_back(e);
        detached.append_range(Children{&r, e});
    };
    for (const auto e : entities) {
        add(e);
        if (const auto *arm = r.try_get<ArmatureObject>(e)) {
            for (const auto bone : arm->BoneEntities) add(bone);
        }
    }
    ClearParents(r, detached);
}
} // namespace

void DestroyArmatureData(state::Scene &r, state::Entity arm_obj_entity) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &arm = r.edit<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != state::Null) {
        if (auto *ref = r.try_get<VertexStoreId>(arm.JointEntity)) meshes.Release(ref->StoreId);
        if (auto *models = r.try_get<ModelsBuffer>(arm.JointEntity)) FreeInstanceRange(r, models->InstanceRange);
        r.remove<VertexStoreId, ModelsBuffer>(arm.JointEntity);
        r.destroy(arm.JointEntity);
        arm.JointEntity = state::Null;
    }
    if (auto *ref = r.try_get<VertexStoreId>(arm_obj_entity)) meshes.Release(ref->StoreId);
    if (auto *models = r.try_get<ModelsBuffer>(arm_obj_entity)) FreeInstanceRange(r, models->InstanceRange);
    r.remove<VertexStoreId, ModelsBuffer>(arm_obj_entity);
}

void Destroy(state::Scene &r, state::Entity viewport, state::Entity e) {
    Destroy(r, viewport, std::span{&e, 1u});
}

void Destroy(state::Scene &r, state::Entity viewport, std::span<const state::Entity> entities) {
    if (entities.empty()) return;
    const profile::CpuScope scope{"DestroyObjects"};
    auto &pending = r.Context.emplace<PendingObjectRemovals>();
    ClearRelationships(r, entities);
    std::vector<state::Entity> destroyed;
    destroyed.reserve(entities.size());
    for (const auto e : entities) {
        if (!r.valid(e)) continue;
        if (r.all_of<LookingThrough>(e)) ClearLookThrough(r, viewport);

        if (const auto *instance = r.try_get<Instance>(e)) {
            if (HasMesh(r, instance->Entity) || r.all_of<ObjectExtrasTag>(instance->Entity)) pending.Buffers.emplace(instance->Entity);
        }
        auto try_add_armature_data = [&](state::Entity data_entity) {
            if (r.valid(data_entity)) pending.Armatures.emplace(data_entity);
        };
        if (const auto *armature = r.try_get<ArmatureObject>(e)) try_add_armature_data(armature->Entity);
        if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) try_add_armature_data(armature_modifier->ArmatureEntity);
        if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) try_add_armature_data(bone_attachment->ArmatureEntity);

        if (const auto *light_index = r.try_get<LightIndex>(e)) {
            r.Context.get<GpuBuffers>().PendingLightRemovals.emplace_back(light_index->Value);
        }

        if (r.all_of<ArmatureObject>(e)) {
            const auto &arm = r.get<ArmatureObject>(e);
            const auto retire_buffers = [&](state::Entity owner) {
                if (const auto *ref = r.try_get<VertexStoreId>(owner)) pending.StoreIds.push_back(ref->StoreId);
                if (const auto *models = r.try_get<ModelsBuffer>(owner)) pending.InstanceRanges.push_back(models->InstanceRange);
            };
            retire_buffers(e);
            if (arm.JointEntity != state::Null) {
                retire_buffers(arm.JointEntity);
                destroyed.push_back(arm.JointEntity);
            }
            for (const auto bone_entity : arm.BoneEntities) {
                if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                    if (joints->Head != state::Null) destroyed.push_back(joints->Head);
                    if (joints->Tail != state::Null) destroyed.push_back(joints->Tail);
                }
            }

            destroyed.append_range(arm.BoneEntities);
        }

        destroyed.push_back(e);
    }
    const profile::CpuScope component_scope{"DestroyObjectComponents"};
    r.destroy(destroyed);
}

void ProcessObjectRemovals(state::Scene &r, state::Entity viewport) {
    auto *pending = r.Context.find<PendingObjectRemovals>();
    if (!pending) return;
    const profile::CpuScope scope{"ProcessObjectRemovals"};
    auto &buffer_entities = pending->Buffers, &armature_data_entities = pending->Armatures;
    const bool removed_objects = !buffer_entities.empty() || !armature_data_entities.empty();
    if (removed_objects) {
        // Shared data survives while an instance uses it: a placed instance in its slot range, or one created since the last settle.
        if (!buffer_entities.empty()) {
            const auto object_ids = r.Context.get<const GpuBuffers>().Instances.ObjectIdBuffer.GetSpan<uint32_t>();
            std::vector<state::Entity> survivors;
            for (const auto data_entity : buffer_entities) {
                const auto *models = r.try_get<const ModelsBuffer>(data_entity);
                const auto slots = models ? object_ids.subspan(models->InstanceRange.Offset, models->InstanceCount) : std::span<const uint32_t>{};
                if (std::ranges::any_of(slots, [&](uint32_t id) { const auto *instance = r.try_get<const Instance>(r.EntityAt(ObjectIndex(id))); return instance && instance->Entity == data_entity; })) survivors.push_back(data_entity);
            }
            for (const auto e : reactive(r, state::Change::InstanceVisibility))
                if (const auto *instance = r.try_get<const Instance>(e)) survivors.push_back(instance->Entity);
            for (const auto data_entity : survivors) buffer_entities.remove(data_entity);
        }
        if (!armature_data_entities.empty()) {
            for (const auto [_, arm] : r.view<const ArmatureObject>().each()) armature_data_entities.remove(arm.Entity);
            for (const auto [_, modifier] : r.view<const ArmatureModifier>().each()) armature_data_entities.remove(modifier.ArmatureEntity);
            for (const auto [_, attachment] : r.view<const BoneAttachment>().each()) armature_data_entities.remove(attachment.ArmatureEntity);
        }
        auto buffers_to_destroy = SortedEntities(buffer_entities);
        for (const auto entity : buffers_to_destroy) {
            if (const auto *ref = r.try_get<VertexStoreId>(entity)) pending->StoreIds.push_back(ref->StoreId);
            if (const auto *models = r.try_get<ModelsBuffer>(entity)) pending->InstanceRanges.push_back(models->InstanceRange);
        }
        buffers_to_destroy.append_range(armature_data_entities);
        std::erase_if(buffers_to_destroy, [&](state::Entity e) { return !r.valid(e); });
        // Destroyed mesh handles and pose states queue their releases below.
        const profile::CpuScope scope{"DestroyDataComponents"};
        r.destroy(buffers_to_destroy);
    }
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    meshes.Release(pending->StoreIds);
    buffers.Instances.Free(std::move(pending->InstanceRanges));
    buffers.ArmatureDeformBuffer.Release(std::move(pending->DeformRanges));
    buffers.MorphWeightBuffer.Release(std::move(pending->MorphRanges));
    meshes.ReleaseSoundVertices(std::move(pending->SoundVertexRanges));
    r.Context.erase<PendingObjectRemovals>();

    // Release imported textures and reset the material when the final instance is removed.
    // Clear the persistent texture manifest at the same time.
    if (removed_objects && r.view<Instance>().empty()) {
        ResetImportedTexturesAndMaterials(r);
        r.remove<MaterializedTextures>(viewport);
    }
}

void ClearMeshes(state::Scene &r, state::Entity viewport) {
    Destroy(r, viewport, SortedEntities(r.view<Instance>(state::Exclude<SubElementOf>)));
}

std::expected<std::pair<state::Entity, state::Entity>, std::string> ImportMesh(state::Scene &r, state::Entity viewport, const std::filesystem::path &path, MeshInstanceCreateInfo info, bool deduplicate) {
    const auto stored_path = project::ResolveAsset(r, path);
    auto result = ReadMeshFile(stored_path);
    if (!result) return std::unexpected{result.error()};

    if (!result->Materials.empty()) {
        const auto imported = ImportObjPlyMaterials(r, viewport, result->Materials, stored_path);
        if (!imported) return std::unexpected{imported.error()};
        for (auto &material : result->Primitives.MaterialIndices) material = (*imported)[material];
    }

    // `deduplicate` merges vertices identical in every vertex-domain channel, keeping per-corner UVs and normals.
    auto created = CreateMesh(r, {.Data = std::move(result->Mesh), .Attrs = std::move(result->Attrs), .Primitives = std::move(result->Primitives), .Weld = deduplicate});
    const auto entities = ::AddMesh(r, created.StoreId, std::move(info));
    if (!created.AuthoredCornerNormals.empty()) r.emplace<AuthoredCornerNormals>(entities.first, std::move(created.AuthoredCornerNormals));
    r.emplace<Path>(entities.first, path);
    return entities;
}

void RequestImportMesh(state::Scene &r, state::Entity viewport, std::filesystem::path path, MeshInstanceCreateInfo info) {
    r.emplace_or_replace<PendingImportMesh>(viewport, std::move(path), std::move(info));
}
