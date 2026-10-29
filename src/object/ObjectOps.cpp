#include "object/ObjectOps.h"
#include "Profile.h"
#include "assets/MeshImport.h"
#include "project/Assets.h"
#include "scene/HierarchyInstance.h"

#include "Path.h"
#include "SortUnique.h"
#include "armature/ArmatureComponents.h"
#include "assets/MaterialImport.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Network.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/Textures.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "viewport/ViewportEvents.h"

namespace {
// Unlink each affected sibling list once, including surviving children of deleted parents.
void ClearRelationships(state::Registry &r, std::span<const state::Entity> entities) {
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

void DestroyArmatureData(state::Registry &r, state::Entity arm_obj_entity) {
    auto &arm = r.edit<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != state::Null) {
        r.destroy(arm.JointEntity);
        arm.JointEntity = state::Null;
    }
    r.remove<VertexStoreId, ModelsBuffer>(arm_obj_entity);
}

void Destroy(state::Registry &r, state::Entity e) {
    Destroy(r, std::span{&e, 1u});
}

void Destroy(state::Registry &r, std::span<const state::Entity> entities) {
    if (entities.empty()) return;
    const auto expanded = PrepareHierarchyRemoval(r, entities);
    entities = expanded;
    if (entities.empty()) return;
    const profile::CpuScope scope{"DestroyObjects"};
    auto &pending = r.Resources.emplace<PendingObjectRemovals>();
    ClearRelationships(r, entities);
    std::vector<state::Entity> destroyed;
    destroyed.reserve(entities.size());
    for (const auto e : entities) {
        if (!r.valid(e)) continue;
        if (const auto *instance = r.try_get<Instance>(e)) {
            if (HasMesh(r, instance->Entity) || DrawsMesh(r, instance->Entity) || r.all_of<ObjectExtrasTag>(instance->Entity)) pending.Buffers.emplace(instance->Entity);
        }
        if (const auto *collider = r.try_get<const ColliderShape>(e); collider && r.valid(collider->MeshEntity)) pending.Buffers.emplace(collider->MeshEntity);
        const auto try_add_armature_data = [&](state::Entity data_entity) {
            if (r.valid(data_entity)) pending.Armatures.emplace(data_entity);
        };
        if (const auto *armature = r.try_get<ArmatureObject>(e)) try_add_armature_data(armature->Entity);
        if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) try_add_armature_data(armature_modifier->ArmatureEntity);
        if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) try_add_armature_data(bone_attachment->ArmatureEntity);

        if (r.all_of<ArmatureObject>(e)) {
            const auto &arm = r.get<ArmatureObject>(e);
            if (arm.JointEntity != state::Null) destroyed.push_back(arm.JointEntity);
            for (const auto bone_entity : arm.BoneEntities) {
                if (const auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                    if (joints->Head != state::Null) destroyed.push_back(joints->Head);
                    if (joints->Tail != state::Null) destroyed.push_back(joints->Tail);
                }
            }

            destroyed.append_range(arm.BoneEntities);
        }

        procedure::RemoveNetworkSubnets(r, e);
        destroyed.push_back(e);
    }
    SortUnique(destroyed);
    const profile::CpuScope component_scope{"DestroyObjectComponents"};
    r.destroy(destroyed);
}

void ProcessObjectRemovals(state::Registry &r) {
    auto *pending = r.Resources.find<PendingObjectRemovals>();
    if (!pending) return;
    const profile::CpuScope scope{"ProcessObjectRemovals"};
    bool removed_objects = false;
    while (!pending->Objects.empty() || !pending->Buffers.empty() || !pending->Armatures.empty() || !pending->Dependents.empty()) {
        auto objects = std::exchange(pending->Objects, {});
        Destroy(r, objects);
        auto &geometry = pending->Buffers, &armatures = pending->Armatures;
        removed_objects |= !geometry.empty() || !armatures.empty();
        for (const auto [_, instance] : r.view<const Instance>().each()) geometry.remove(instance.Entity);
        for (const auto [_, collider] : r.view<const ColliderShape>().each()) geometry.remove(collider.MeshEntity);
        for (const auto [_, arm] : r.view<const ArmatureObject>().each()) armatures.remove(arm.Entity);
        for (const auto [_, modifier] : r.view<const ArmatureModifier>().each()) armatures.remove(modifier.ArmatureEntity);
        for (const auto [_, attachment] : r.view<const BoneAttachment>().each()) armatures.remove(attachment.ArmatureEntity);
        auto removed = std::exchange(pending->Dependents, {});
        removed.append_range(geometry);
        removed.append_range(armatures);
        geometry.clear();
        armatures.clear();
        SortUnique(removed);
        std::erase_if(removed, [&](state::Entity e) { return !r.valid(e); });
        r.destroy(removed);
    }
    auto &meshes = r.Resources.get<MeshStore>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    meshes.Release(pending->StoreIds);
    buffers.Instances.Free(std::move(pending->InstanceRanges));
    buffers.ArmatureDeformBuffer.Release(std::move(pending->DeformRanges));
    buffers.MorphWeightBuffer.Release(std::move(pending->MorphRanges));
    buffers.BoundsReduceEntries.Release(std::move(pending->BoundsEntryRanges));
    buffers.NormalDeriveEntries.Release(std::move(pending->DeriveEntryRanges));
    buffers.PreludeTiles.Release(std::move(pending->PreludeTileRanges));
    meshes.ReleaseSoundVertices(std::move(pending->SoundVertexRanges));
    r.Resources.erase<PendingObjectRemovals>();

    // Release imported textures and reset the material when the final instance is removed.
    // Clear the persistent texture manifest at the same time.
    if (removed_objects && r.view<Instance>().empty()) {
        ResetImportedTexturesAndMaterials(r);
        r.remove<MaterializedTextures>(state::Document);
    }
}

void ClearMeshes(state::Registry &r) {
    Destroy(r, SortedEntities(r.view<Instance>(state::Exclude<SubElementOf>)));
}

std::expected<std::pair<state::Entity, state::Entity>, std::string> ImportMesh(state::Registry &r, const std::filesystem::path &path, const MeshInstanceCreateInfo &info, bool deduplicate) {
    const auto stored_path = project::ResolveAsset(r, path);
    auto result = ReadMeshFile(stored_path);
    if (!result) return std::unexpected{result.error()};

    if (!result->Materials.empty()) {
        const auto imported = ImportObjPlyMaterials(r, result->Materials, stored_path);
        if (!imported) return std::unexpected{imported.error()};
        for (auto &material : result->Primitives.MaterialIndices) material = (*imported)[material];
    }

    // `deduplicate` merges vertices identical in every vertex-domain channel, keeping per-corner UVs and normals.
    auto created = CreateMesh(r, {.Data = std::move(result->Mesh), .Attrs = std::move(result->Attrs), .Primitives = std::move(result->Primitives), .Weld = deduplicate});
    const auto entities = ::AddMesh(r, created.StoreId, info);
    if (!created.AuthoredCornerNormals.empty()) r.emplace<AuthoredCornerNormals>(entities.first, std::move(created.AuthoredCornerNormals));
    r.emplace<Path>(entities.first, path);
    return entities;
}

void RequestImportMesh(state::Registry &r, std::filesystem::path path, MeshInstanceCreateInfo info) {
    r.emplace_or_replace<PendingImportMesh>(state::Document, std::move(path), std::move(info));
}
