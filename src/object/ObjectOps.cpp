#include "object/ObjectOps.h"

#include "CameraTypes.h"
#include "File.h"
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
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/OneShotGpu.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

#include <entt/entity/registry.hpp>

#include <iostream>

using std::ranges::any_of, std::ranges::find, std::ranges::to;

std::string CreateName(entt::registry &r, std::string_view prefix) {
    auto &registry = r.ctx().get<NameRegistry>();
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
    if (auto *registry = r.ctx().find<NameRegistry>()) registry->Names.erase(r.get<const Name>(e).Value);
}

void AssignRenderInstanceObjectId(entt::registry &r, entt::entity e) {
    auto &ri = r.get<RenderInstance>(e);
    if (ri.ObjectId != 0) return;
    if (auto *counter = r.ctx().find<ObjectIdCounter>()) ri.ObjectId = counter->Next++;
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
    r.emplace<WorldTransform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, info.Name));
    Show(r, e);
    if (!info.Visible) Hide(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

std::pair<entt::entity, entt::entity> AddMesh(entt::registry &r, MeshStore &meshes, Mesh &&mesh, std::optional<MeshInstanceCreateInfo> info) {
    const auto mesh_entity = r.create();
    r.emplace<MeshBuffers>(mesh_entity, meshes.GetVerticesRange(mesh.GetStoreId()), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<Mesh>(mesh_entity, std::move(mesh));
    return {mesh_entity, info ? AddMeshInstance(r, mesh_entity, *info) : entt::null};
}

entt::entity CreateExtrasBufferEntity(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices) {
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

entt::entity CreateExtrasObject(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, std::span<const vec3> positions, std::span<const uint8_t> vertex_classes, std::span<const uint32_t> edge_indices, ObjectType type, ObjectCreateInfo info, std::string_view default_name) {
    const auto buffer_entity = CreateExtrasBufferEntity(r, meshes, buffers, positions, vertex_classes, edge_indices);
    const auto e = r.create();
    r.emplace<ObjectKind>(e, type);
    r.emplace<Instance>(e, buffer_entity);
    r.emplace<Transform>(e, info.Transform);
    r.emplace<WorldTransform>(e, info.Transform);
    r.emplace<Name>(e, CreateName(r, info.Name.empty() ? default_name : info.Name));
    Show(r, e);
    ApplySelectBehavior(r, e, info.Select);
    return e;
}

entt::entity AddEmpty(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, ObjectCreateInfo info) {
    static constexpr vec3 positions[] = {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, -1}};
    static constexpr uint32_t edges[] = {0, 1, 2, 3, 4, 5};
    return CreateExtrasObject(r, meshes, buffers, positions, {}, edges, ObjectType::Empty, std::move(info), "Empty");
}

entt::entity AddCamera(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, ObjectCreateInfo info, std::optional<Camera> props) {
    const Camera camera = props.value_or(Camera{Defaults::PerspectiveCamera});
    auto mesh = BuildCameraFrustumMesh(camera);
    const auto entity = CreateExtrasObject(r, meshes, buffers, mesh.Positions, {}, mesh.CreateEdgeIndices(), ObjectType::Camera, std::move(info), "Camera");
    r.emplace<Camera>(entity, camera);
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
    r.emplace<WorldTransform>(bone_entity, bone_transform);
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
    r.emplace<MeshBuffers>(arm_obj_entity, meshes.GetVerticesRange(bone_store_id), SlottedRange{}, SlottedRange{}, SlottedRange{});
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
    r.emplace<MeshBuffers>(joint_entity, meshes.GetVerticesRange(sphere_store_id), SlottedRange{}, SlottedRange{}, SlottedRange{});
    r.emplace<VertexStoreId>(joint_entity, sphere_store_id);

    for (const auto bone_entity : arm_obj.BoneEntities) CreateBoneJoints(r, arm_obj_entity, bone_entity, joint_entity);
    arm_obj.JointEntity = joint_entity;
}

entt::entity AddLight(entt::registry &r, MeshStore &meshes, GpuBuffers &buffers, ObjectCreateInfo info, std::optional<PunctualLight> props) {
    auto light = props.value_or(Defaults::MakePunctualLight(PunctualLightType::Point));
    auto wireframe = BuildLightMesh(light);
    const auto entity = CreateExtrasObject(r, meshes, buffers, wireframe.Data.Positions, wireframe.VertexClasses, {}, ObjectType::Light, std::move(info), "Light");
    r.emplace<LightIndex>(entity, r.storage<LightIndex>().size());
    r.emplace<SubmitDirty>(entity);
    r.emplace<LightWireframeDirty>(entity);
    // Defer SetLight: TransformSlotOffset needs InstanceArena slot and RenderInstance.BufferIndex,
    // which aren't available until SyncModelsBuffers runs. Store light data temporarily;
    // ProcessComponentEvents syncs the slot after SyncModelsBuffers.
    r.emplace<PunctualLight>(entity, light);
    return entity;
}

namespace {
// True if any component of type C has `C.*field == target`.
template<class C, class F>
bool AnyComponentRefersTo(entt::registry &r, F C::*field, entt::entity target) {
    return any_of(r.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}
} // namespace

void DestroyArmatureData(entt::registry &r, entt::entity arm_obj_entity) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &arm = r.get<ArmatureObject>(arm_obj_entity);
    if (arm.JointEntity != entt::null) {
        if (auto *mb = r.try_get<MeshBuffers>(arm.JointEntity)) buffers.Release(*mb);
        if (auto *ref = r.try_get<VertexStoreId>(arm.JointEntity)) meshes.Release(ref->StoreId);
        if (auto *models = r.try_get<ModelsBuffer>(arm.JointEntity)) buffers.Instances.Free(models->InstanceRange);
        r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, PendingHide>(arm.JointEntity);
        r.destroy(arm.JointEntity);
        arm.JointEntity = entt::null;
    }
    if (auto *mb = r.try_get<MeshBuffers>(arm_obj_entity)) buffers.Release(*mb);
    if (auto *adj = r.try_get<BoneAdjacencyIndices>(arm_obj_entity)) buffers.EdgeIndexBuffer.Release(adj->Indices);
    if (auto *ref = r.try_get<VertexStoreId>(arm_obj_entity)) meshes.Release(ref->StoreId);
    if (auto *models = r.try_get<ModelsBuffer>(arm_obj_entity)) buffers.Instances.Free(models->InstanceRange);
    r.remove<MeshBuffers, VertexStoreId, ModelsBuffer, BoneAdjacencyIndices, PendingHide>(arm_obj_entity);
}

void Destroy(entt::registry &r, entt::entity viewport, entt::entity e) {
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    if (r.all_of<LookingThrough>(e)) {
        r.replace<ViewCamera>(viewport, r.get<LookingThrough>(e).SavedViewCamera);
        r.remove<LookingThrough>(e);
    }
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
        if (r.all_of<Mesh>(instance->Entity) || r.all_of<ObjectExtrasTag>(instance->Entity)) buffer_entity = instance->Entity;
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
        r.get_or_emplace<PendingLightRemovals>(viewport).Indices.push_back(light_index->Value);
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
                    buffers.VertexClassBuffer.Release({vcr->Offset, mesh_buffers->Vertices.Count});
                }
                buffers.Release(*mesh_buffers);
            }
            if (const auto *vs = r.try_get<VertexStoreId>(buffer_entity)) meshes.Release(vs->StoreId);
            if (const auto *models = r.try_get<ModelsBuffer>(buffer_entity)) buffers.Instances.Free(models->InstanceRange);
            r.destroy(buffer_entity);
        }
    }
    for (const auto armature_data_entity : armature_data_entities) {
        if (!r.valid(armature_data_entity)) continue;
        const bool is_used = AnyComponentRefersTo(r, &ArmatureObject::Entity, armature_data_entity) ||
            AnyComponentRefersTo(r, &ArmatureModifier::ArmatureEntity, armature_data_entity) ||
            AnyComponentRefersTo(r, &BoneAttachment::ArmatureEntity, armature_data_entity);
        if (!is_used) r.destroy(armature_data_entity);
    }

    // If no instances remain, release all imported textures and reset to the default material.
    if (r.view<Instance>().empty()) {
        // Index 0 is the default white texture (permanent); imported textures start at index 1.
        if (textures.Textures.size() > 1) {
            ReleaseSamplerSlots(slots, CollectSamplerSlots(std::span<const TextureEntry>{textures.Textures}.subspan(1)));
            textures.Textures.erase(textures.Textures.begin() + 1, textures.Textures.end());
        }
        textures.WhiteTextureSlot = textures.Textures.empty() ? InvalidSlot : textures.Textures.front().SamplerSlot;

        if (buffers.Materials.Count() > 1) buffers.Materials.SetCount(1u);
        if (auto &ms = r.ctx().get<MaterialStore>(); ms.Names.size() > 1) ms.Names.erase(ms.Names.begin() + 1, ms.Names.end());
    }
}

void ClearMeshes(entt::registry &r, entt::entity viewport) {
    for (const auto e : r.view<Instance>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
}

std::pair<entt::entity, entt::entity> ImportMesh(
    entt::registry &r,
    const std::filesystem::path &path, MeshInstanceCreateInfo info
) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    auto result = meshes.LoadMesh(path);
    if (!result) throw std::runtime_error(result.error());

    if (!result->Materials.empty()) {
        auto obj_batch = BeginTextureUploadBatch(vk.Device, *one_shot.Pool, buffers.Ctx);
        std::unordered_map<std::string, uint32_t> texture_slot_cache;
        const auto resolve_texture_slot =
            [&](
                const std::optional<std::filesystem::path> &source_texture_path,
                TextureColorSpace color_space,
                std::string_view material_name, std::string_view texture_label
            ) -> uint32_t {
            if (!source_texture_path) return InvalidSlot;
            auto texture_path = *source_texture_path;
            if (texture_path.is_relative()) texture_path = path.parent_path() / texture_path;
            texture_path = texture_path.lexically_normal();

            const auto cache_key = std::format("{}|{}", texture_path.generic_string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear");
            if (const auto it = texture_slot_cache.find(cache_key); it != texture_slot_cache.end()) return it->second;

            std::string encoded;
            try {
                encoded = File::Read(texture_path);
            } catch (const std::exception &e) {
                std::cerr << std::format(
                    "Warning: Failed to read OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                    texture_path.string(), material_name, texture_label, path.string(), e.what()
                );
                return InvalidSlot;
            }

            auto texture = CreateTextureEntryFromEncoded(
                vk,
                obj_batch,
                slots,
                std::as_bytes(std::span{encoded}),
                texture_path.filename().string(),
                std::format("{} ({})", texture_path.filename().string(), color_space == TextureColorSpace::Srgb ? "sRGB" : "Linear"),
                color_space,
                vk::SamplerAddressMode::eRepeat,
                vk::SamplerAddressMode::eRepeat,
                SamplerConfig{}
            );
            if (!texture) {
                std::cerr << std::format(
                    "Warning: Failed to decode OBJ texture '{}' for material '{}' ({}) in '{}': {}\n",
                    texture_path.string(), material_name, texture_label, path.string(), texture.error()
                );
                return InvalidSlot;
            }

            const auto sampler_slot = texture->SamplerSlot;
            textures.Textures.emplace_back(std::move(*texture));
            texture_slot_cache.emplace(cache_key, sampler_slot);
            return sampler_slot;
        };

        std::vector<uint32_t> scene_material_indices(result->Materials.size(), 0u);
        std::vector<std::string> names;
        names.reserve(result->Materials.size());
        buffers.Materials.Reserve(buffers.Materials.Count() + result->Materials.size());
        for (uint32_t material_index = 0; material_index < result->Materials.size(); ++material_index) {
            const auto &source = result->Materials[material_index];
            const auto material_name = source.Name.empty() ? std::format("Material{}", material_index) : source.Name;
            const auto base_color_texture = resolve_texture_slot(source.BaseColorTexturePath, TextureColorSpace::Srgb, material_name, "baseColor");
            const auto normal_texture = resolve_texture_slot(source.NormalTexturePath, TextureColorSpace::Linear, material_name, "normal");
            scene_material_indices[material_index] = buffers.Materials.Append({
                .BaseColorFactor = source.BaseColorFactor,
                .MetallicFactor = std::clamp(source.MetallicFactor, 0.f, 1.f),
                .RoughnessFactor = std::clamp(source.RoughnessFactor, 0.f, 1.f),
                .AlphaMode = (source.BaseColorFactor.w < 1.f || source.HasAlphaTexture) ?
                    MaterialAlphaMode::Blend :
                    MaterialAlphaMode::Opaque,
                .BaseColorTexture = {.Slot = base_color_texture != InvalidSlot ? base_color_texture : textures.WhiteTextureSlot},
                .NormalTexture = {.Slot = normal_texture},
            });
            names.emplace_back(material_name);
        }
        SubmitTextureUploadBatch(obj_batch, vk.Queue, *one_shot.Fence, vk.Device);

        auto &material_store = r.ctx().get<MaterialStore>();
        material_store.Names.insert(material_store.Names.end(), std::make_move_iterator(names.begin()), std::make_move_iterator(names.end()));

        if (auto primitive_materials = meshes.GetPrimitiveMaterialIndices(result->Mesh.GetStoreId()); !primitive_materials.empty()) {
            const auto fallback = scene_material_indices.front();
            for (auto &primitive_material : primitive_materials) {
                primitive_material = primitive_material < scene_material_indices.size() ? scene_material_indices[primitive_material] : fallback;
            }
        }
    }

    const auto entities = ::AddMesh(r, meshes, std::move(result->Mesh), std::move(info));
    r.emplace<Path>(entities.first, path);
    r.emplace<SmoothShading>(entities.first);
    return entities;
}
