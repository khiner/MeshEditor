#include "Apply.h"
#include "AnimationTimeline.h"
#include "Armature.h"
#include "Defaults.h"
#include "File.h"
#include "GpuBuffers.h"
#include "Instance.h"
#include "InteractionComponents.h"
#include "MeshComponents.h"
#include "NodeTransformAnimation.h"
#include "Path.h"
#include "Pipelines.h"
#include "SceneGraph.h"
#include "Selection.h"
#include "SelectionComponents.h"
#include "SoundVertices.h"
#include "Textures.h"
#include "Timer.h"
#include "TransformMath.h"
#include "VkFenceWait.h"
#include "audio/AudioSystem.h"
#include "audio/RealImpact.h"
#include "gltf/GltfScene.h"
#include "gpu/UpdateSelectionStatePushConstants.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsWorld.h"

#include <iostream>

using std::ranges::any_of, std::ranges::find, std::ranges::find_if;
using std::ranges::to;

namespace {
// True if any component of type C has `C.*field == target`.
template<class C, class F>
bool AnyComponentRefersTo(entt::registry &r, F C::*field, entt::entity target) {
    return any_of(r.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}

void SetLookThrough(entt::registry &r, entt::entity viewport, entt::entity target) {
    const auto previous = LookThroughCameraEntity(r);
    if (previous == target) return;
    // Preserve the saved view across camera switches; only capture fresh on first entry.
    auto saved = previous != entt::null ? r.get<LookingThrough>(previous).SavedViewCamera : r.get<ViewCamera>(viewport);
    if (previous != entt::null) r.remove<LookingThrough>(previous);
    r.emplace<LookingThrough>(target, std::move(saved));
}

void ExitLookThrough(entt::registry &r, entt::entity viewport) {
    const auto camera = LookThroughCameraEntity(r);
    if (camera == entt::null) return;
    r.replace<ViewCamera>(viewport, r.get<LookingThrough>(camera).SavedViewCamera);
    r.remove<LookingThrough>(camera);
}

void RebuildBoneStructure(entt::registry &r, entt::entity viewport, entt::entity arm_data_entity) {
    RebuildArmatureStructure(r, arm_data_entity);
    r.get<LastEvaluatedFrame>(viewport).Value = -1;
}

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

entt::entity CreateSingleBoneInstance(entt::registry &r, entt::entity arm_obj_entity, BoneId bone_id) {
    auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
    const auto &armature = r.get<const Armature>(arm_obj.Entity);
    const auto new_index = *armature.FindBoneIndex(bone_id);
    const auto parent_index = armature.Bones[new_index].ParentIndex;
    const auto parent_entity = parent_index == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[parent_index];
    const auto bone_entity = ::CreateBoneEntity(r, arm_obj_entity, armature, new_index, parent_entity);
    if (arm_obj.JointEntity != null_entity && r.valid(arm_obj.JointEntity)) {
        ::CreateBoneJoints(r, arm_obj_entity, bone_entity, arm_obj.JointEntity);
    }
    arm_obj.BoneEntities.emplace_back(bone_entity);
    return bone_entity;
}
} // namespace

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

namespace {
void ClearMeshes(entt::registry &r, entt::entity viewport) {
    for (const auto e : r.view<Instance>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
}

void JumpToStartFrame(entt::registry &r, entt::entity viewport) {
    const auto frame = r.get<const TimelineRange>(viewport).StartFrame;
    r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
    r.get<PlaybackFrame>(viewport).Value = frame;
    r.emplace_or_replace<PhysicsCacheInvalid>(viewport);
}

void NewDefaultScene(entt::registry &r, entt::entity viewport) {
    ClearMeshes(r, viewport);

    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(default_shape), {}, {}), MeshInstanceCreateInfo{.Name = ToString(default_shape)});
    r.emplace<PrimitiveShape>(mesh_entity, default_shape);

    // startup.blend data, in Blender's frame (Z-up, -Y forward)
    constexpr vec3 LightLoc{4.07625, 1.00545, 5.90386}, CameraLoc{7.358891, -6.925791, 4.958309}, CameraEulerXYZ{1.109319, 0, 0.815801};
    constexpr float Lens{50}, SensorX{36}, RenderW{16}, RenderH{9};
    // Blender Z-up -> MeshEditor Y-up is a -90° rotation about +X: (x, y, z) -> (x, z, -y)
    const auto to_y_up_pos = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat to_y_up_rot = glm::angleAxis(-float(M_PI_2), vec3{1, 0, 0});
    // Matches Blender glTF exporter (cameras.py / yvof_blender_to_gltf): horizontal fit since render aspect > sensor aspect
    const float hfov = 2 * std::atan(SensorX / (2 * Lens));
    const float yfov = 2 * std::atan(std::tan(hfov * 0.5) * RenderH / RenderW);

    ::AddLight(r, meshes, buffers, ObjectCreateInfo{.Name = "Light", .Transform = {.P = to_y_up_pos(LightLoc)}, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
    ::AddCamera(r, meshes, buffers, ObjectCreateInfo{.Name = "Camera", .Transform = {.P = to_y_up_pos(CameraLoc), .R = to_y_up_rot * quat{CameraEulerXYZ}}, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, Perspective{.FieldOfViewRad = yfov, .FarClip = 1000, .NearClip = DefaultPerspectiveNearClip});
}

template<typename...> struct TypeList {};

template<typename... Ts, typename Fn>
bool DispatchByTypeHash(TypeList<Ts...>, entt::id_type hash, Fn &&fn) {
    return ((entt::type_hash<Ts>::value() == hash && (fn.template operator()<Ts>(), true)) || ...);
}

// Lists of components used by multilps templated actions.
using UpdateableComponents = TypeList<
    Transform, ViewportDisplay, MaterialVariants, ArmatureAnimation, MorphWeightAnimation,
    NodeTransformAnimation, MaterialPreviewLighting, RenderedLighting, ViewportTheme,
    ViewCamera, PhysicsMaterial, CollisionFilter, ColliderMaterial, ColliderPolicy,
    PhysicsMotion, PhysicsVelocity, PhysicsJoint, TriggerNodes, ModalModelCreateInfo,
    PhysicsSimulationSettings, TransformGizmoState>;
using TagComponents = TypeList<SmoothShading, SubmitDirty, LightWireframeDirty, TriggerTag>;
using NamedPhysicsComponents = TypeList<PhysicsMaterial, CollisionSystem, CollisionFilter, PhysicsJointDef>;

entt::entity DuplicateOne(entt::registry &r, entt::entity e, bool &was_mesh_duplicate) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const ObjectCreateInfo create_info{
        .Name = std::format("{}_copy", GetName(r, e)),
        // Duplicate is created at root, so its local must match source's world.
        .Transform = Transform{r.get<const WorldTransform>(e)},
        .Select = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None,
    };

    if (!r.all_of<Instance>(e)) {
        if (const auto object_type = r.all_of<ObjectKind>(e) ? r.get<const ObjectKind>(e).Value : ObjectType::Empty; object_type == ObjectType::Armature) {
            const auto data_entity = r.create();
            r.emplace<Armature>(data_entity);
            const auto copy_entity = r.create();
            r.emplace<ObjectKind>(copy_entity, ObjectType::Armature);
            r.emplace<ArmatureObject>(copy_entity, data_entity);
            r.emplace<Transform>(copy_entity, create_info.Transform);
            r.emplace<WorldTransform>(copy_entity, create_info.Transform);
            r.emplace<Name>(copy_entity, ::CreateName(r, create_info.Name.empty() ? "Armature" : create_info.Name));
            ::ApplySelectBehavior(r, copy_entity, create_info.Select);
            ::CreateBoneInstances(r, meshes, copy_entity, data_entity);
            if (const auto *src_armature = r.try_get<ArmatureObject>(e)) {
                auto &dst = r.get<Armature>(data_entity);
                dst = r.get<const Armature>(src_armature->Entity);
            }
            return copy_entity;
        }
        return ::AddEmpty(r, meshes, buffers, create_info);
    }

    // Bone sub-entities (head/tail joints, bone instances) are not independently duplicable.
    if (r.all_of<BoneSubPartOf>(e)) return entt::null;

    // Object extras (Camera, Empty, Light) have Instance but create their own wireframe mesh.
    if (r.all_of<ObjectExtrasTag>(r.get<Instance>(e).Entity)) {
        if (const auto *src_cd = r.try_get<Camera>(e)) return ::AddCamera(r, meshes, buffers, create_info, *src_cd);
        if (r.all_of<LightIndex>(e)) return ::AddLight(r, meshes, buffers, create_info, buffers.Lights.Get(r.get<const LightIndex>(e).Value));
        return ::AddEmpty(r, meshes, buffers, create_info);
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = ::AddMesh(
        r, meshes,
        meshes.CloneMesh(r.get<const Mesh>(mesh_entity)),
        MeshInstanceCreateInfo{.Name = create_info.Name, .Transform = create_info.Transform, .Select = create_info.Select, .Visible = r.all_of<RenderInstance>(e)}
    );
    if (auto prim_shape = r.try_get<PrimitiveShape>(mesh_entity)) r.emplace<PrimitiveShape>(e_new.first, *prim_shape);
    if (r.all_of<SmoothShading>(mesh_entity)) r.emplace<SmoothShading>(e_new.first);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new.second, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new.second, *bone_attachment);
    was_mesh_duplicate = true;
    return e_new.second;
}

entt::entity DuplicateLinkedOne(entt::registry &r, entt::entity e) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (r.all_of<BoneSubPartOf>(e)) return entt::null;
    if (!r.all_of<Instance>(e)) {
        const auto select_behavior = r.all_of<Selected>(e) ? MeshInstanceCreateInfo::SelectBehavior::Additive : MeshInstanceCreateInfo::SelectBehavior::None;

        if (const auto *armature = r.try_get<ArmatureObject>(e)) {
            const auto e_new = r.create();
            r.emplace<Name>(e_new, ::CreateName(r, std::format("{}_copy", GetName(r, e))));
            r.emplace<ObjectKind>(e_new, ObjectType::Armature);
            r.emplace<ArmatureObject>(e_new, armature->Entity);
            const Transform t{r.get<const WorldTransform>(e)};
            r.emplace_or_replace<Transform>(e_new, t);
            r.emplace<WorldTransform>(e_new, t);

            ::ApplySelectBehavior(r, e_new, select_behavior);
            ::CreateBoneInstances(r, meshes, e_new, armature->Entity);
            return e_new;
        }
        return ::AddEmpty(r, meshes, buffers, {.Name = std::format("{}_copy", GetName(r, e)), .Transform = Transform{r.get<const WorldTransform>(e)}, .Select = select_behavior});
    }

    const auto mesh_entity = r.get<Instance>(e).Entity;
    const auto e_new = r.create();
    {
        uint instance_count = 0; // Count instances for naming (first duplicated instance is _1, etc.)
        for (const auto [_, instance] : r.view<Instance>().each()) {
            if (instance.Entity == mesh_entity) ++instance_count;
        }
        r.emplace<Name>(e_new, ::CreateName(r, std::format("{}_{}", GetName(r, e), instance_count)));
    }
    r.emplace<Instance>(e_new, mesh_entity);
    r.emplace<ObjectKind>(e_new, ObjectType::Mesh);
    const Transform t_new{r.get<const WorldTransform>(e)};
    r.emplace_or_replace<Transform>(e_new, t_new);
    r.emplace<WorldTransform>(e_new, t_new);
    Show(r, e_new);
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(e)) r.emplace<ArmatureModifier>(e_new, *armature_modifier);
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(e)) r.emplace<BoneAttachment>(e_new, *bone_attachment);

    r.emplace<Selected>(e_new);

    return e_new;
}
} // namespace

entt::entity GetMeshEntity(const entt::registry &r, entt::entity e) {
    if (const auto *instance = r.try_get<Instance>(e); instance && r.all_of<Mesh>(instance->Entity)) return instance->Entity;
    return entt::null;
}
entt::entity GetActiveMeshEntity(const entt::registry &r) {
    const auto active = FindActiveEntity(r);
    return active != entt::null ? GetMeshEntity(r, active) : entt::null;
}

entt::entity LookThroughCameraEntity(const entt::registry &r) {
    auto view = r.view<LookingThrough>();
    return view.empty() ? entt::null : *view.begin();
}

entt::entity FindMeshEntity(const entt::registry &r, entt::entity entity) {
    if (const auto *instance = r.try_get<const Instance>(entity)) return instance->Entity;
    return entity;
}

bool IsBoneEditMode(const entt::registry &r, entt::entity viewport) {
    if (r.get<const Interaction>(viewport).Mode != InteractionMode::Edit) return false;
    return FindArmatureObject(r, FindActiveEntity(r)) != entt::null;
}

bool AllSelectedAreMeshes(const entt::registry &r) {
    for (const auto [e, ok] : r.view<const Selected, const ObjectKind>().each()) {
        if (ok.Value != ObjectType::Mesh) return false;
    }
    return true;
}

bool CanDuplicate(const entt::registry &r, entt::entity viewport) {
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose) return false;
    if (IsBoneEditMode(r, viewport)) return !r.view<BoneSelection>().empty();
    return !r.view<Selected>().empty();
}
bool CanDuplicateLinked(const entt::registry &r, entt::entity viewport) { return CanDuplicate(r, viewport) && !IsBoneEditMode(r, viewport); }
bool CanDelete(const entt::registry &r, entt::entity viewport) { return CanDuplicate(r, viewport); }

std::vector<ElementRange> GetBitsetRangesForSelected(const entt::registry &r) {
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
        if (const auto *br = r.try_get<const MeshSelectionBitsetRange>(mesh_entity); br && br->Count > 0) {
            ranges.emplace_back(mesh_entity, br->Offset, br->Count);
        }
    }
    return ranges;
}

bool SetInteractionMode(entt::registry &r, entt::entity viewport, InteractionMode mode) {
    if (r.get<const Interaction>(viewport).Mode == mode) return false;

    const auto active_entity = FindActiveEntity(r);
    const auto active_arm = active_entity != entt::null ? FindArmatureObject(r, active_entity) : entt::null;
    const bool active_is_armature = active_arm != entt::null;
    if (mode == InteractionMode::Edit && !AllSelectedAreMeshes(r) && !active_is_armature) return false;
    if (mode == InteractionMode::Pose && !active_is_armature) return false;

    r.clear<VertexForce>();

    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit) {
        // Keep bitset ranges + bits so element selections survive toggling Edit mode off and back on.
        for (const auto [mesh_entity, br, mesh] : r.view<const MeshSelectionBitsetRange, const Mesh>().each()) {
            if (br.Count > 0) meshes.UpdateElementStates(mesh, Element::None, {}, {}, {}, {}, std::nullopt);
        }
        r.emplace_or_replace<ElementStatesDirty>(viewport);
    }
    if (mode == InteractionMode::Edit && !active_is_armature) {
        // Only assign ranges for selected meshes missing one; existing ranges preserve remembered selection.
        if (const auto edit_element = r.get<const EditMode>(viewport).Value; edit_element != Element::None) {
            uint32_t next_offset = 0;
            for (const auto [_, br] : r.view<const MeshSelectionBitsetRange>().each()) {
                next_offset = std::max(next_offset, (br.Offset + br.Count + 31) / 32 * 32);
            }
            auto *bits = buffers.SelectionBitset.Data();
            for (const auto mesh_entity : selection::GetSelectedMeshEntities(r)) {
                if (r.all_of<MeshSelectionBitsetRange>(mesh_entity)) continue;
                const auto &mesh = r.get<const Mesh>(mesh_entity);
                const uint32_t count = selection::GetElementCount(mesh, edit_element);
                if (count == 0) continue;

                selection::SelectAll(bits, next_offset, count);
                r.emplace<MeshSelectionBitsetRange>(mesh_entity, next_offset, count);
                next_offset = (next_offset + count + 31) / 32 * 32;
            }
        }
    }
    r.patch<Interaction>(viewport, [mode](auto &s) { s.Mode = mode; });
    r.patch<ViewportTheme>(viewport, [](auto &) {});
    return true;
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

void SetStudioEnvironment(entt::registry &r, uint32_t index) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) {
        hdri.Prefiltered = CreateIblFromHdri(
            vk, slots,
            pipelines.IblPrefilter, hdri.Path, hdri.Name,
            *one_shot.Pool, *one_shot.Fence, buffers.Ctx
        );
    }
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

void DispatchUpdateSelectionStates(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &vk_res = r.ctx().get<const VulkanResources>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    const auto &sel_slots = r.get<const SelectionSlots>(viewport);
    auto &meshes = r.ctx().get<MeshStore>();

    auto cb = *one_shot.Cb;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    const vk::MemoryBarrier input_barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, input_barrier, {}, {});

    const auto &compute = pipelines.UpdateSelectionState;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});

    for (const auto &range : ranges) {
        const auto &mesh = r.get<const Mesh>(range.MeshEntity);
        const auto &mesh_buffers = r.get<const MeshBuffers>(range.MeshEntity);
        const auto *active_element = r.try_get<const MeshActiveElement>(range.MeshEntity);

        uint32_t state_slot, state_offset;
        if (element == Element::Vertex) {
            state_slot = meshes.GetVertexStateSlot();
            state_offset = mesh_buffers.Vertices.Offset;
        } else if (element == Element::Edge) {
            const auto edge_range = meshes.GetEdgeStateRange(mesh.GetStoreId());
            state_slot = edge_range.Slot;
            state_offset = edge_range.Offset;
        } else {
            const auto face_range = meshes.GetFaceStateRange(mesh.GetStoreId());
            state_slot = face_range.Slot;
            state_offset = face_range.Offset;
        }

        const UpdateSelectionStatePushConstants pc{
            .BitsetSlot = sel_slots.SelectionBitset,
            .BitsetOffset = range.Offset,
            .StateSlot = state_slot,
            .StateOffset = state_offset,
            .ElementCount = range.Count,
            .ActiveHandle = active_element ? active_element->Handle : InvalidOffset,
            .EdgeMode = element == Element::Edge ? 1u : 0u,
        };
        cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cb.dispatch((range.Count + 255) / 256, 1, 1);
    }

    const vk::MemoryBarrier output_barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, output_barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    vk_res.Queue.submit(submit, *one_shot.Fence);
    WaitFor(*one_shot.Fence, vk_res.Device);
}

void ApplySelectionStateUpdate(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element
) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    DispatchUpdateSelectionStates(r, viewport, ranges, element);
    if (element == Element::Vertex) {
        for (const auto &range : ranges) {
            const auto &mesh = r.get<const Mesh>(range.MeshEntity);
            meshes.UpdateEdgeStatesFromVertices(mesh);
            meshes.UpdateFaceStatesFromVertices(mesh);
        }
    } else if (element == Element::Face || element == Element::Edge) {
        const auto *bits = buffers.SelectionBitset.Data();
        for (const auto &range : ranges) {
            const auto &mesh = r.get<const Mesh>(range.MeshEntity);
            const auto selected_handles = selection::ScanBitsetRange(bits, range.Offset, range.Count);
            std::optional<uint32_t> active_handle;
            if (const auto *active = r.try_get<const MeshActiveElement>(range.MeshEntity); active && active->Handle < range.Count) {
                active_handle = active->Handle;
            }
            if (element == Element::Face) meshes.UpdateEdgeStatesFromFaces(mesh, selected_handles, active_handle);
            if (element == Element::Edge) meshes.UpdateFaceStatesFromEdges(mesh);
            meshes.UpdateVertexStatesFromElements(mesh, selected_handles, element, active_handle);
        }
    }
    r.emplace_or_replace<ElementStatesDirty>(viewport);
}

void Apply(entt::registry &r, entt::entity viewport, const action::Action &action) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto patch_camera_stopped = [&](auto &&fn) {
        r.patch<ViewCamera>(viewport, [&](auto &c) { fn(c); c.StopMoving(); });
    };
    auto poke_active_lighting = [&] {
        const auto mode = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (mode == ViewportShadingMode::MaterialPreview) r.patch<MaterialPreviewLighting>(viewport, [](auto &) {});
        else if (mode == ViewportShadingMode::Rendered) r.patch<RenderedLighting>(viewport, [](auto &) {});
    };
    auto apply_box_select = [&]<typename Tag>(bool additive, auto restore_baseline, auto apply_hits) {
        r.clear<Tag>();
        if (additive) {
            if (const auto *baseline = r.try_get<const AdditiveBoxSelectBaseline>(viewport)) restore_baseline(*baseline);
        }
        apply_hits();
    };
    auto merge_bone_sel = [&](entt::entity e, const std::optional<BoneSel> &part, bool additive) {
        const auto sel = part ? BoneSelection::From(*part) : BoneSelection{};
        const auto *cur = r.try_get<BoneSelection>(e);
        r.emplace_or_replace<BoneSelection>(e, additive && cur ? *cur | sel : sel);
    };

    // AdditiveBoxSelectBaseline is meaningful only during an active box-select drag;
    // a click-selection always ends one, so its handler owns the cleanup.
    auto end_box_select_interaction = [&] { r.remove<AdditiveBoxSelectBaseline>(viewport); };
    std::visit(
        overloaded{
            [&](action::selection::Select a) { end_box_select_interaction(); ::Select(r, a.Entity); },
            [&](action::selection::ToggleSelected a) { end_box_select_interaction(); ::ToggleSelected(r, a.Entity); },
            [&](action::selection::SelectBone a) {
                end_box_select_interaction();
                ::SelectBone(r, a.Entity);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](action::selection::ExtendActive a) {
                end_box_select_interaction();
                r.clear<Active>();
                r.emplace<Active>(a.Entity);
                r.emplace_or_replace<Selected>(a.Entity);
            },
            [&](action::selection::ExtendBoneActive a) {
                end_box_select_interaction();
                r.clear<BoneActive>();
                r.emplace<BoneActive>(a.Entity);
                if (!r.all_of<BoneSelection>(a.Entity)) r.emplace<BoneSelection>(a.Entity, false, false, false);
                if (a.Part) merge_bone_sel(a.Entity, a.Part, a.Additive);
            },
            [&](const action::selection::SetBoneSelectionPart &a) { merge_bone_sel(a.Entity, a.Part, a.Additive); },
            [&](action::selection::DeselectAll) {
                end_box_select_interaction();
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const bool bone_mode = interaction_mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
                if (bone_mode) r.clear<BoneSelection>();
                else r.clear<Selected>();
            },
            [&](action::selection::SnapshotBoxSelectBaseline) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const bool active_is_armature = FindArmatureObject(r, active_entity) != entt::null;
                const bool bone_mode = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && active_is_armature);
                AdditiveBoxSelectBaseline baseline;
                if (interaction_mode == InteractionMode::Edit && !active_is_armature) {
                    if (const auto ranges = GetBitsetRangesForSelected(r); !ranges.empty()) {
                        const auto element_count = std::ranges::fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &range) { return std::max(total, range.Offset + range.Count); });
                        const uint32_t bitset_words = (element_count + 31) / 32;
                        const auto bits = r.get<const SelectionBitsetRef>(viewport).Value;
                        baseline.ElementBitset.assign(bits.begin(), bits.begin() + bitset_words);
                    }
                } else if (bone_mode) {
                    for (const auto e : r.view<BoneSelection>()) baseline.BoneSelections.emplace_back(e, r.get<BoneSelection>(e));
                } else if (interaction_mode == InteractionMode::Object) {
                    for (const auto e : r.view<Selected>()) baseline.SelectedEntities.push_back(e);
                }
                r.emplace_or_replace<AdditiveBoxSelectBaseline>(viewport, std::move(baseline));
            },
            [&](action::selection::ClearBoxSelectBaseline) { r.remove<AdditiveBoxSelectBaseline>(viewport); },
            [&](const action::selection::ApplyBoxSelectObjectHits &a) {
                apply_box_select.template operator()<Selected>(
                    a.Additive,
                    [&](const AdditiveBoxSelectBaseline &b) {
                        for (const auto e : b.SelectedEntities) {
                            if (r.valid(e)) r.emplace_or_replace<Selected>(e);
                        }
                    },
                    [&] {
                        for (const auto e : a.Hits) r.emplace_or_replace<Selected>(e);
                    }
                );
            },
            [&](const action::selection::ApplyEditElementClick &a) {
                end_box_select_interaction();
                r.emplace_or_replace<PendingEditElementClick>(viewport, a.MousePx, a.Toggle);
            },
            [&](const action::selection::ApplyTreeSelection &a) {
                using Clear = action::selection::ApplyTreeSelection::ClearKind;
                if (a.Clear == Clear::BonesOnly) r.clear<BoneSelection>();
                else if (a.Clear == Clear::All) r.clear<Selected, BoneSelection>();
                for (const auto e : a.ToSelect) {
                    if (r.all_of<BoneIndex>(e)) r.emplace_or_replace<BoneSelection>(e);
                    else if (!r.all_of<Selected>(e)) r.emplace<Selected>(e);
                }
                for (const auto e : a.ToDeselect) {
                    if (r.all_of<BoneIndex>(e)) {
                        if (r.all_of<BoneSelection>(e)) r.remove<BoneSelection>(e);
                    } else if (r.all_of<Selected>(e)) r.remove<Selected>(e);
                }
                if (a.NavToActive != entt::null) {
                    const bool is_bone = r.all_of<BoneIndex>(a.NavToActive);
                    if (is_bone ? r.all_of<BoneSelection>(a.NavToActive) : r.all_of<Selected>(a.NavToActive)) {
                        if (is_bone) {
                            r.clear<BoneActive>();
                            r.emplace<BoneActive>(a.NavToActive);
                        } else {
                            r.clear<Active>();
                            r.emplace<Active>(a.NavToActive);
                        }
                    }
                }
            },
            [&](const action::selection::ApplyBoxSelectBoneHits &a) {
                apply_box_select.template operator()<BoneSelection>(
                    a.Additive,
                    [&](const AdditiveBoxSelectBaseline &b) {
                        for (const auto &[e, sel] : b.BoneSelections) {
                            if (r.valid(e)) r.emplace_or_replace<BoneSelection>(e, sel);
                        }
                    },
                    [&] {
                        for (const auto &[entity, part] : a.Hits) merge_bone_sel(entity, part, a.Additive);
                    }
                );
            },
            [&](action::object::Delete) {
                if (!CanDelete(r, viewport)) return;
                for (const auto e : r.view<Selected>(entt::exclude<SubElementOf>) | to<std::vector>()) Destroy(r, viewport, e);
            },
            [&](action::object::Duplicate) {
                if (!CanDuplicate(r, viewport)) return;
                const Timer timer{"Duplicate"};
                const auto entities = r.view<Selected>() | to<std::vector>();

                // Pre-reserve arenas to avoid per-CloneMesh buffer growth.
                for (const auto e : entities) {
                    if (!r.all_of<Instance>(e) || r.all_of<BoneSubPartOf>(e)) continue;
                    const auto mesh_entity = r.get<Instance>(e).Entity;
                    if (r.all_of<ObjectExtrasTag>(mesh_entity) || !r.all_of<Mesh>(mesh_entity)) continue;
                    meshes.PlanClone(r.get<const Mesh>(mesh_entity));
                }
                meshes.CommitReserves();

                bool any_mesh_duplicate = false;
                for (const auto e : entities) {
                    const auto new_e = DuplicateOne(r, e, any_mesh_duplicate);
                    if (r.all_of<Active>(e)) {
                        r.remove<Active>(e);
                        r.emplace<Active>(new_e);
                    }
                    r.remove<Selected>(e);
                }
                if (any_mesh_duplicate) r.emplace_or_replace<ProfileNextProcessComponentEvents>(viewport);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](action::object::DuplicateLinked) {
                if (!CanDuplicateLinked(r, viewport)) return;
                const Timer timer{"DuplicateLinked"};
                for (const auto e : r.view<Selected>() | to<std::vector>()) {
                    const auto new_e = DuplicateLinkedOne(r, e);
                    if (r.all_of<Active>(e)) {
                        r.remove<Active>(e);
                        r.emplace<Active>(new_e);
                    }
                    r.remove<Selected>(e);
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](action::object::ToggleHidden) {
                for (const auto e : r.view<Selected>()) {
                    if (r.all_of<RenderInstance>(e)) Hide(r, e);
                    else Show(r, e);
                }
            },
            [&](const action::object::SetSelectedVisible &a) {
                for (const auto e : r.view<const Selected, const Instance>()) {
                    if (r.all_of<SubElementOf>(e)) continue;
                    if (a.Visible) Show(r, e);
                    else Hide(r, e);
                }
            },
            [&](const action::object::SetSelectedSmoothShading &a) {
                for (const auto me : selection::GetSelectedMeshEntities(r)) {
                    if (r.get<const Mesh>(me).FaceCount() == 0) continue;
                    if (a.Smooth) r.emplace_or_replace<SmoothShading>(me);
                    else r.remove<SmoothShading>(me);
                }
            },
            [&](action::object::ParentToActive) {
                const auto active = FindActiveEntity(r);
                if (active == entt::null) return;
                for (const auto e : r.view<Selected>()) {
                    if (e != active) SetParentKeepWorld(r, e, active);
                }
            },
            [&](action::object::ClearParent) {
                for (const auto e : r.view<Selected>()) ::ClearParent(r, e);
            },
            [&](const action::object::AddEmpty &a) {
                ::AddEmpty(r, meshes, buffers, *a.Info);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::object::AddArmature &a) {
                ::AddArmature(r, meshes, *a.Info);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::object::AddCamera &a) {
                ::AddCamera(r, meshes, buffers, *a.Info, a.Props);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::object::AddLight &a) {
                ::AddLight(r, meshes, buffers, *a.Info);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::object::AddMeshPrimitive &a) {
                const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(a.Shape), {}, {}), *a.Info);
                r.emplace<PrimitiveShape>(mesh_entity, a.Shape);
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::object::ImportMesh &a) { r.emplace_or_replace<PendingImportMesh>(viewport, a.Path, *a.Info); },
            [&](const action::object::ReplaceMesh &a) {
                const auto e = GetActiveMeshEntity(r);
                if (e == entt::null || selection::HasScaleLockedInstance(r, e)) return;

                if (auto *mb = r.try_get<MeshBuffers>(e)) buffers.Release(*mb);
                r.erase<MeshBuffers>(e);
                r.erase<Mesh>(e);

                auto new_mesh = meshes.CreateMesh(MeshData{*a.Data}, {}, {});
                r.emplace<MeshBuffers>(e, meshes.GetVerticesRange(new_mesh.GetStoreId()), SlottedRange{}, SlottedRange{}, SlottedRange{});
                r.emplace<Mesh>(e, std::move(new_mesh));
                r.emplace_or_replace<MeshGeometryDirty>(e);
            },
            [&](action::project::NewDefaultScene) { NewDefaultScene(r, viewport); },
            [&](action::bone::Add) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == entt::null) return;

                auto &armature = r.get<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                const auto &arm_wt = r.get<WorldTransform>(arm_obj_entity);
                const auto new_id = armature.AddBone("Bone", {}, {.P = (glm::conjugate(glm::normalize(arm_wt.R)) * -arm_wt.P) / arm_wt.S});
                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);

                const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, new_id);
                SelectBone(r, bone_entity);
                r.emplace_or_replace<BoneSelection>(bone_entity, false, true, false);
            },
            [&](action::bone::Extrude) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
                auto &armature = r.get<Armature>(arm_obj.Entity);
                auto result = ExtrudeBones(r, armature, arm_obj_entity);
                if (result.NewBoneIds.empty()) return;

                RebuildBoneStructure(r, viewport, arm_obj.Entity);
                r.clear<BoneSelection, BoneActive>();

                for (const auto id : result.NewBoneIds) {
                    const auto bone_entity = CreateSingleBoneInstance(r, arm_obj_entity, id);
                    r.replace<BoneDisplayScale>(bone_entity, 0.f);
                    r.emplace<BoneSelection>(bone_entity, false, true, false);
                    r.emplace_or_replace<BoneActive>(bone_entity);
                }
                for (const auto idx : result.UpdatedParentIndices) {
                    r.replace<BoneDisplayScale>(arm_obj.BoneEntities[idx], ComputeBoneDisplayScale(armature, idx));
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](action::bone::DuplicateSelected) {
                const auto arm_obj_entity = FindArmatureObject(r, FindActiveEntity(r));
                if (arm_obj_entity == entt::null) return;

                auto &armature = r.get<Armature>(r.get<ArmatureObject>(arm_obj_entity).Entity);
                auto result = DuplicateBones(r, armature, arm_obj_entity);
                if (result.Duplicated.empty()) return;

                RebuildBoneStructure(r, viewport, r.get<ArmatureObject>(arm_obj_entity).Entity);
                r.clear<BoneSelection, BoneActive>();

                entt::entity last_bone{};
                for (const auto &[orig_entity, new_id] : result.Duplicated) {
                    last_bone = CreateSingleBoneInstance(r, arm_obj_entity, new_id);
                    r.replace<BoneDisplayScale>(last_bone, r.get<const BoneDisplayScale>(orig_entity).Value);
                    r.emplace<BoneSelection>(last_bone);
                }
                r.emplace<BoneActive>(last_bone);

                r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate);
            },
            [&](const action::bone::ClearSelectedTransforms &a) { ClearSelectedBoneTransforms(r, a.Position, a.Rotation, a.Scale); },
            [&](action::bone::DeleteSelected) {
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                if (arm_obj_entity == entt::null) return;

                auto &arm_obj = r.get<ArmatureObject>(arm_obj_entity);
                auto &armature = r.get<Armature>(arm_obj.Entity);
                const auto to_delete = CollectBonesForDeletion(r, arm_obj_entity);
                if (to_delete.empty()) return;

                for (const auto idx : to_delete) {
                    const auto bone_entity = arm_obj.BoneEntities[idx];
                    const auto &bone = armature.Bones[idx];
                    const auto grandparent = bone.ParentIndex == InvalidBoneIndex ? arm_obj_entity : arm_obj.BoneEntities[bone.ParentIndex];

                    if (auto *joints = r.try_get<BoneJointEntities>(bone_entity)) {
                        if (joints->Head != null_entity) {
                            Hide(r, joints->Head);
                            r.destroy(joints->Head);
                        }
                        if (joints->Tail != null_entity) {
                            Hide(r, joints->Tail);
                            r.destroy(joints->Tail);
                        }
                        r.remove<BoneJointEntities>(bone_entity);
                    }

                    std::vector<entt::entity> children;
                    for (const auto child : Children{&r, bone_entity}) children.emplace_back(child);
                    for (const auto child : children) {
                        const auto &ct = r.get<const Transform>(child);
                        const auto t = ComposeLocalTransforms(bone.RestLocal, ct);
                        r.emplace_or_replace<Transform>(child, Transform{t.P, t.R, r.all_of<ScaleLocked>(child) ? ct.S : t.S});
                        ClearParent(r, child);
                        SetParent(r, child, grandparent);
                    }

                    ClearParent(r, bone_entity);
                    Hide(r, bone_entity);
                    r.destroy(bone_entity);
                }

                for (const auto idx : to_delete) {
                    armature.RemoveBone(armature.Bones[idx].Id);
                    arm_obj.BoneEntities.erase(arm_obj.BoneEntities.begin() + idx);
                }

                RebuildBoneStructure(r, viewport, arm_obj.Entity);

                for (uint32_t i = 0; i < arm_obj.BoneEntities.size(); ++i) r.get<BoneIndex>(arm_obj.BoneEntities[i]).Index = i;

                if (arm_obj.BoneEntities.empty()) DestroyArmatureData(r, arm_obj_entity);
                ::Select(r, arm_obj_entity);
            },
            [&](const action::view::SetInteractionMode &a) { SetInteractionMode(r, viewport, a.Mode); },
            [&](action::view::CycleInteractionMode) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto &enabled_modes = r.get<const EnabledInteractionModes>(viewport).Value;
                auto it = find(enabled_modes, interaction_mode);
                for (size_t i = 0; i < enabled_modes.size(); ++i) {
                    if (++it == enabled_modes.end()) it = enabled_modes.begin();
                    if (SetInteractionMode(r, viewport, *it)) break;
                }
            },
            [&](const action::view::SetEditMode &a) { r.emplace_or_replace<PendingSetEditMode>(viewport, a.Mode); },
            [&](action::selection::SelectAll) {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto active_entity = FindActiveEntity(r);
                const auto arm_obj_entity = FindArmatureObject(r, active_entity);
                const bool bone_select = interaction_mode == InteractionMode::Pose || (interaction_mode == InteractionMode::Edit && arm_obj_entity != entt::null);
                if (bone_select) {
                    if (arm_obj_entity == entt::null) return;
                    const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
                    r.clear<BoneActive, BoneSelection>();
                    for (const auto bone_entity : arm_obj.BoneEntities) r.emplace<BoneSelection>(bone_entity);
                    if (!arm_obj.BoneEntities.empty()) r.emplace<BoneActive>(arm_obj.BoneEntities.back());
                } else if (interaction_mode == InteractionMode::Edit) {
                    const auto ranges = GetBitsetRangesForSelected(r);
                    auto *bits = r.get<SelectionBitsetRef>(viewport).Value.data();
                    for (const auto &range : ranges) selection::SelectAll(bits, range.Offset, range.Count);
                    if (!ranges.empty()) r.emplace_or_replace<SelectionBitsDirty>(viewport);
                } else if (interaction_mode == InteractionMode::Object) {
                    r.clear<Active, Selected>();
                    entt::entity last{entt::null};
                    for (const auto [e, _] : r.view<const ObjectKind>().each()) {
                        r.emplace<Selected>(e);
                        last = e;
                    }
                    if (last != entt::null) r.emplace<Active>(last);
                }
            },
            [&](action::view::EnterLookThroughCamera) {
                const auto e = FindActiveEntity(r);
                if (e == entt::null) return;
                SetLookThrough(r, viewport, e);
                const auto &wt = r.get<WorldTransform>(e);
                const vec3 fwd = CameraForward(wt), away = -fwd;
                r.patch<ViewCamera>(viewport, [&](auto &vc) { vc.AnimateTo(wt.P + fwd, {std::atan2(away.z, away.x), std::asin(away.y)}, 1.f); });
            },
            [&](action::view::ExitLookThroughCamera) { ExitLookThrough(r, viewport); },
            [&](const action::view::OrbitViewCamera &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.RotateBy(a.DeltaRad); });
            },
            [&](const action::view::ZoomViewCamera &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &camera) { camera.ZoomBy(a.Factor); });
            },
            [&](const action::audio::ApplyExciteImpact &a) {
                r.emplace_or_replace<MeshActiveElement>(r.get<Instance>(a.InstanceEntity).Entity, a.VertexIndex);
                r.emplace_or_replace<VertexForce>(a.InstanceEntity, a.VertexIndex, 1.f);
            },
            [&](action::audio::ClearExciteImpacts) { r.clear<VertexForce>(); },
            [&](const action::project::SetStudioEnvironment &a) { r.emplace_or_replace<PendingSetStudioEnvironment>(viewport, a.Index); poke_active_lighting(); },
            [&](const action::project::SetSourceIblIntensity &a) {
                r.patch<gltf::SourceAssets>(viewport, [&](auto &sa) { if (sa.ImageBasedLight) sa.ImageBasedLight->Intensity = a.Intensity; });
                poke_active_lighting();
            },
            [&](action::view::ResetViewCamera) { patch_camera_stopped([](auto &c) { c = Defaults::ViewCamera; }); },
            [&](action::view::ResetViewportTheme) { r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme); },
            [&](const action::view::ResetPbrLighting &a) {
                static constexpr PBRViewportLighting Defaults{false, false, 1.f, 0.f, 0.5f, 0.f, true};
                if (a.Rendered) r.emplace_or_replace<RenderedLighting>(viewport, RenderedLighting{Defaults});
                else r.emplace_or_replace<MaterialPreviewLighting>(viewport, MaterialPreviewLighting{Defaults});
            },
            [&](const action::view::SetViewCameraTarget &a) { patch_camera_stopped([&](auto &c) { c.Target = a.Target; }); },
            [&](const action::view::SetViewCameraLens &a) { patch_camera_stopped([&](auto &c) { c.Data = a.Data; }); },
            [&](const action::view::SetViewCameraTargetDirection &a) {
                ExitLookThrough(r, viewport);
                r.patch<ViewCamera>(viewport, [&](auto &c) { c.SetTargetDirection(a.Direction); });
            },
            [&](const action::object::SetPbrMeshFeaturesMask &a) {
                const auto e = GetActiveMeshEntity(r);
                if (a.Mask != 0u) r.emplace_or_replace<PbrMeshFeatures>(e, a.Mask);
                else r.remove<PbrMeshFeatures>(e);
            },
            [&](const action::view::SetRotationUiMode &a) {
                const auto e = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && FindActiveBone(r) != entt::null ? FindActiveBone(r) : FindActiveEntity(r);
                r.replace<RotationUiVariant>(e, CreateVariantByIndex<RotationUiVariant>(a.Index));
                r.patch<Transform>(e, [](auto &) {});
            },
            [&](const action::view::SetTransformRotationFromUi &a) {
                const auto e = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && FindActiveBone(r) != entt::null ? FindActiveBone(r) : FindActiveEntity(r);
                r.replace<RotationUiVariant>(e, a.UiVariant);
                r.emplace_or_replace<RotationUiDriving>(e);
                r.patch<Transform>(e, [&](auto &t) { t.R = a.R; });
            },
            [&](const action::view::DragGizmo &a) {
                for (const auto &[e, _] : a.Locals) {
                    if (!r.all_of<StartTransform>(e)) r.emplace<StartTransform>(e, r.get<WorldTransform>(e), ToTransform(GetParentDelta(r, e)));
                }
                for (const auto &[e, _] : a.BoneDisplayScales) {
                    if (!r.all_of<StartBoneLength>(e)) {
                        if (const auto *ds = r.try_get<BoneDisplayScale>(e)) r.emplace<StartBoneLength>(e, ds->Value);
                    }
                }
                for (const auto &[e, local] : a.Locals) r.patch<Transform>(e, [&](auto &t) { t = local; });
                for (const auto &[e, length] : a.BoneDisplayScales) r.get_or_emplace<BoneDisplayScale>(e).Value = length;
            },
            [&](const action::view::DragGizmoMeshEdit &a) {
                for (const auto &[_, instance_entity] : selection::ComputePrimaryEditInstances(r, false)) {
                    if (!r.all_of<StartTransform>(instance_entity)) {
                        r.emplace<StartTransform>(instance_entity, r.get<WorldTransform>(instance_entity), ToTransform(GetParentDelta(r, instance_entity)));
                    }
                }
                r.emplace_or_replace<PendingTransform>(viewport, *a.Value);
            },
            [&](action::view::EndGizmoDrag) {
                r.clear<StartTransform, StartBoneLength>();
                r.remove<StartScreenTransform>(viewport);
            },
            [&](const action::view::SetActiveTool &a) {
                using Tool = action::view::SetActiveTool::Tool;
                using TT = TransformGizmo::Type;
                const auto type = a.Value == Tool::SelectBox || a.Value == Tool::SelectClick ? TT::None :
                    a.Value == Tool::Translate                                               ? TT::Translate :
                    a.Value == Tool::Rotate                                                  ? TT::Rotate :
                    a.Value == Tool::Scale                                                   ? TT::Scale :
                                                                                               TT::Universal;
                r.patch<TransformGizmoState>(viewport, [&](auto &s) { s.Config.Type = type; });
                if (a.Value == Tool::SelectBox || a.Value == Tool::SelectClick) {
                    const auto g = a.Value == Tool::SelectBox ? SelectionGesture::Box : SelectionGesture::Click;
                    r.patch<BoxSelectState>(viewport, [&](auto &b) { b.Gesture = g; });
                }
            },
            [&](const action::view::SetStartScreenTransform &a) {
                if (!a.Value) {
                    r.remove<StartScreenTransform>(viewport);
                    return;
                }
                // Mid-drag switch is a cancel-restart: revert any in-progress drag to its start state.
                // StartTransform / StartBoneLength components stay so the next drag (under the new latched type) reuses them.
                r.remove<PendingTransform>(viewport);
                for (const auto [e, st] : r.view<const StartTransform>().each()) {
                    const auto &pd = st.ParentDelta;
                    r.patch<Transform>(e, [&](auto &t) {
                        t.P = glm::conjugate(pd.R) * ((st.T.P - pd.P) / pd.S);
                        t.R = glm::conjugate(pd.R) * st.T.R;
                        if (!r.all_of<ScaleLocked>(e)) t.S = st.T.S / pd.S;
                    });
                }
                for (const auto [e, sbl] : r.view<const StartBoneLength>().each()) {
                    r.get_or_emplace<BoneDisplayScale>(e).Value = sbl.Value;
                }
                r.emplace_or_replace<StartScreenTransform>(viewport, *a.Value);
            },
            [&](const action::bone::SetEditHeadTailRoll &a) {
                const auto e = FindActiveBone(r);
                r.patch<Transform>(e, [&](auto &t) { t.P = a.LocalP; t.R = a.LocalR; });
                r.get<BoneDisplayScale>(e).Value = a.DisplayScale;
            },
            [&](const action::bone::SetConstraintTarget &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].TargetEntity = a.Target; });
            },
            [&](const action::bone::SetConstraintInfluence &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack[a.Index].Influence = a.Influence; });
            },
            [&](const action::bone::SetConstraintChildOfInverse &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { std::get<ChildOfData>(cs.Stack[a.Index].Data).InverseMatrix = *a.Inverse; });
            },
            [&](const action::bone::DeleteConstraint &a) {
                r.patch<BoneConstraints>(FindActiveBone(r), [&](auto &cs) { cs.Stack.erase(cs.Stack.begin() + a.Index); });
            },
            [&](const action::bone::AddConstraint &a) {
                const auto e = FindActiveBone(r);
                if (!r.all_of<BoneConstraints>(e)) r.emplace<BoneConstraints>(e);
                r.patch<BoneConstraints>(e, [&](auto &cs) {
                    cs.Stack.push_back(a.Kind == action::bone::BoneConstraintKind::ChildOf ? BoneConstraint{.Data = ChildOfData{}} : BoneConstraint{.Data = CopyTransformsData{}});
                });
            },
            [&]<typename Field>(const action::Update<Field> &a) {
                const auto e = a.Entity != entt::null ? a.Entity : viewport;
                DispatchByTypeHash(UpdateableComponents{}, a.ComponentType, [&]<typename T> {
                    r.patch<T>(e, [&](T &t) { *reinterpret_cast<Field *>(reinterpret_cast<std::byte *>(&t) + a.Offset) = a.Value; });
                });
            },
            [&]<typename Field>(const action::UpdateActive<Field> &a) {
                const auto e = FindActiveEntity(r);
                DispatchByTypeHash(UpdateableComponents{}, a.ComponentType, [&]<typename T> {
                    r.patch<T>(e, [&](T &t) { *reinterpret_cast<Field *>(reinterpret_cast<std::byte *>(&t) + a.Offset) = a.Value; });
                });
            },
            [&](const action::SetTag &a) {
                DispatchByTypeHash(TagComponents{}, a.TagType, [&]<typename T> {
                    if (a.Present) r.emplace_or_replace<T>(a.Entity);
                    else r.remove<T>(a.Entity);
                });
            },
            [&](const action::SetActiveTag &a) {
                const auto e = FindActiveEntity(r);
                DispatchByTypeHash(TagComponents{}, a.TagType, [&]<typename T> {
                    if (a.Present) r.emplace_or_replace<T>(e);
                    else r.remove<T>(e);
                });
            },
            [&]<typename T>(const action::Replace<T> &a) {
                if constexpr (std::is_same_v<T, PhysicsMotion>) r.emplace_or_replace<T>(a.Entity, *a.Value);
                else r.emplace_or_replace<T>(a.Entity, a.Value);
            },
            [&]<typename T>(const action::ReplaceActive<T> &a) {
                const auto e = FindActiveEntity(r);
                if constexpr (std::is_same_v<T, PhysicsMotion>) r.emplace_or_replace<T>(e, *a.Value);
                else if constexpr (std::is_same_v<T, PunctualLight>) {
                    const auto *old = r.try_get<const PunctualLight>(e);
                    const auto &n = a.Value;
                    if (!old || old->Type != n.Type || old->Range != n.Range || old->OuterConeCos != n.OuterConeCos || old->InnerConeCos != n.InnerConeCos) {
                        r.emplace_or_replace<LightWireframeDirty>(e);
                    }
                    r.emplace_or_replace<T>(e, n);
                } else r.emplace_or_replace<T>(e, a.Value);
            },
            [&](const action::DestroyEntity &a) { r.destroy(a.Entity); },
            [&](const action::physics::CreateNamed &a) {
                DispatchByTypeHash(NamedPhysicsComponents{}, a.ComponentType, [&]<typename T> {
                    r.emplace<T>(r.create(), T{.Name = std::format("{} {}", a.Prefix, r.view<T>().size())});
                });
            },
            [&](const action::physics::SetName &a) {
                DispatchByTypeHash(NamedPhysicsComponents{}, a.ComponentType, [&]<typename T> {
                    r.patch<T>(a.Entity, [&](T &x) { x.Name = a.Name; });
                });
            },
            [&](const action::physics::SetMotionType &a) {
                using Type = action::physics::SetMotionType::Type;
                const auto e = FindActiveEntity(r);
                const bool want_motion = a.Value == Type::Kinematic || a.Value == Type::Dynamic;
                const bool want_collider = a.Value == Type::Static || want_motion;
                if (!want_motion) r.remove<PhysicsMotion>(e);
                if (!want_collider) r.remove<ColliderShape>(e);
                if (want_collider && !r.all_of<ColliderShape>(e)) {
                    r.emplace<ColliderShape>(e);
                    r.emplace<ColliderPolicy>(e);
                }
                if (want_motion) {
                    const bool is_kinematic = a.Value == Type::Kinematic;
                    if (!r.all_of<PhysicsMotion>(e)) r.emplace<PhysicsMotion>(e, PhysicsMotion{.IsKinematic = is_kinematic});
                    else r.patch<PhysicsMotion>(e, [is_kinematic](PhysicsMotion &m) { m.IsKinematic = is_kinematic; });
                }
            },
            [&](const action::physics::SetColliderShape &a) {
                const auto e = FindActiveEntity(r);
                const auto owner_mesh = FindMeshEntity(r, e);
                r.patch<ColliderShape>(e, [&](ColliderShape &cs) {
                    cs.Shape = a.Shape;
                    if (IsMeshBackedShape(a.Shape) && cs.MeshEntity == null_entity) cs.MeshEntity = owner_mesh;
                });
                if (a.LockKind) r.patch<ColliderPolicy>(e, [](ColliderPolicy &p) { p.LockedKind = true; });
            },
            [&](action::physics::AddTrigger) {
                const auto e = FindActiveEntity(r);
                r.emplace<ColliderShape>(e);
                r.emplace<ColliderPolicy>(e);
                r.emplace<TriggerTag>(e);
            },
            [&](action::physics::RemoveTriggerNodes) { r.remove<TriggerNodes>(FindActiveEntity(r)); },
            [&](const action::physics::ToggleFilterEntity &a) {
                r.patch<CollisionFilter>(a.FilterEntity, [&](CollisionFilter &f) {
                    auto &vec = f.*(a.Field);
                    if (a.Add) {
                        if (std::find(vec.begin(), vec.end(), a.SystemEntity) == vec.end()) vec.push_back(a.SystemEntity);
                    } else std::erase(vec, a.SystemEntity);
                });
            },
            [&]<typename T>(const action::physics::SetJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*(a.Field))[a.Index] = *a.Value; });
            },
            [&]<typename T>(const action::physics::AddJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) { (d.*(a.Field)).push_back({}); });
            },
            [&]<typename T>(const action::physics::DeleteJointVecItem<T> &a) {
                r.patch<PhysicsJointDef>(a.JointDefEntity, [&](PhysicsJointDef &d) {
                    auto &vec = d.*(a.Field);
                    vec.erase(vec.begin() + a.Index);
                });
            },
            [&](const action::audio::SetModel &a) { ::SetModel(r, viewport, FindActiveEntity(r), a.Model); },
            [&](const action::audio::SetExciteVertex &a) {
                const auto e = FindActiveEntity(r);
                r.remove<VertexForce>(e);
                r.emplace_or_replace<MeshActiveElement>(GetActiveMeshEntity(r), a.MeshVertex);
                ::SetVertex(r, viewport, e, a.VertexIndex);
            },
            [&](const action::audio::SetActiveElementFromDsp &a) { r.emplace_or_replace<MeshActiveElement>(GetActiveMeshEntity(r), a.Vertex); },
            [&](const action::audio::StartExcite &a) {
                const auto e = FindActiveEntity(r);
                r.remove<VertexForce>(e);
                r.emplace<VertexForce>(e, a.Vertex, 1.f);
            },
            [&](action::audio::StopExcite) { r.remove<VertexForce>(FindActiveEntity(r)); },
            [&](action::audio::DeleteSoundObject) { RemoveAudioComponents(r, FindActiveEntity(r)); },
            [&](const action::audio::StartRecording &a) { r.emplace_or_replace<Recording>(FindActiveEntity(r), a.FrameCount); },
            [&](const action::audio::OpenModalForm &a) { r.emplace_or_replace<ModalModelCreateInfo>(FindActiveEntity(r), *a.Info); },
            [&](action::audio::CancelModalForm) { r.remove<ModalModelCreateInfo>(FindActiveEntity(r)); },
            [&](action::audio::SubmitModalForm) {
                const auto e = FindActiveEntity(r);
                ::Stop(r, viewport, e);
                r.emplace_or_replace<AcousticMaterial>(GetActiveMeshEntity(r), r.get<const ModalModelCreateInfo>(e).Material);
                r.remove<ModalModelCreateInfo>(e);
            },
            [&](const action::audio::AcceptModalGenerationResult &a) {
                const auto e = FindActiveEntity(r);
                if (!r.all_of<ScaleLocked>(e)) r.emplace<ScaleLocked>(e);
                r.emplace_or_replace<ModalModes>(e, a.D->Modes);
                r.emplace_or_replace<TetMeshData>(GetActiveMeshEntity(r), a.D->Tets);
                ::SetModel(r, viewport, e, SoundVerticesModel::Modal);
            },
            [&](const action::audio::AssignVertexSamples &a) {
                ::AssignVertexSample(r, viewport, FindActiveEntity(r), a.D->MeshVertices, a.D->Path, std::vector<float>{a.D->Frames});
            },
            [&](action::audio::SetVertexSamples a) { ::SetVertexSamples(r, viewport, a.SoundEntity, a.MeshVertices, std::move(a.Samples)); },
            [&](const action::audio::ActivateRealImpactMicrophone &a) {
                const auto dir = r.get<const Path>(r.get<const Instance>(a.TargetSoundEntity).Entity).Value.parent_path();
                const auto &vertex_indices = r.get<const RealImpactVertices>(a.TargetSoundEntity).Vertices;
                const auto mic_index = r.get<const RealImpactMicrophone>(a.MicrophoneEntity).Index;
                ::SetVertexSamples(r, viewport, a.TargetSoundEntity, vertex_indices, RealImpact::LoadSamples(dir, mic_index) | to<std::vector>());
                r.emplace_or_replace<RealImpactActiveMicrophone>(a.TargetSoundEntity, a.MicrophoneEntity);
            },
            [&](const action::audio::RemoveVertexSamples &a) { ::RemoveVertexSamples(r, viewport, FindActiveEntity(r), a.MeshVertices); },
            [&](const action::audio::SetModalFormMaterial &a) { r.patch<ModalModelCreateInfo>(FindActiveEntity(r), [&](auto &info) { info.Material = *a.Material; }); },
            [&](action::timeline::Play) {
                r.patch<ViewportDisplay>(viewport, [](auto &s) { s.ViewportShading = s.FillMode = ViewportShadingMode::MaterialPreview; s.ShowOverlays = false; });
                r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = !p.Playing; });
            },
            [&](const action::view::SetViewportShading &a) {
                r.patch<ViewportDisplay>(viewport, [&](auto &s) {
                    s.ViewportShading = a.Mode;
                    if (a.Mode != ViewportShadingMode::Wireframe) s.FillMode = a.Mode;
                });
            },
            [&](action::timeline::TogglePlay) { r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = !p.Playing; }); },
            [&](const action::timeline::SetFrame &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const action::timeline::SetStartFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &r) { r.StartFrame = a.Frame; }); },
            [&](const action::timeline::SetEndFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &r) { r.EndFrame = a.Frame; }); },
            [&](action::timeline::JumpToStart) { JumpToStartFrame(r, viewport); },
            [&](action::timeline::JumpToEnd) {
                const auto frame = r.get<const TimelineRange>(viewport).EndFrame;
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
                r.get<PlaybackFrame>(viewport).Value = frame;
            },
            [&](const action::timeline::SetView &a) { r.replace<AnimationTimelineView>(viewport, AnimationTimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}

std::expected<void, std::string> Apply(entt::registry &r, entt::entity viewport, const action::FallibleAction &action) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &physics = r.ctx().get<PhysicsWorld>();
    return std::visit(
        overloaded{
            [&](const action::project::SaveGltf &a) -> std::expected<void, std::string> {
                return gltf::SaveGltf(a.Path, {r, viewport, buffers, meshes, textures, &vk, &buffers.Ctx});
            },
            [&](const action::project::LoadGltf &a) -> std::expected<void, std::string> {
                const Timer timer{"LoadGltf"};
                auto result = gltf::LoadGltf(a.Path, {r, viewport, slots, buffers, meshes, textures, environments});
                if (!result) return std::unexpected{std::move(result.error())};

                // TODO: drive reactively from track<changes::PhysicsShape>.
                if (!r.view<ColliderShape>().empty()) physics.RecomputeSceneScale(r);

                if (result->FirstCameraObject != entt::null) {
                    const auto camera_entity = result->FirstCameraObject;
                    SetLookThrough(r, viewport, camera_entity);
                    const auto &wt = r.get<WorldTransform>(camera_entity);
                    r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.P + CameraForward(wt), r.get<Camera>(camera_entity)});
                }
                if (result->ImportedAnimation) {
                    JumpToStartFrame(r, viewport);
                    r.get<LastEvaluatedFrame>(viewport).Value = -1;
                }
                r.emplace_or_replace<ProfileNextProcessComponentEvents>(viewport);
                return {};
            },
            [&](const action::project::LoadRealImpact &a) -> std::expected<void, std::string> {
                auto object_name = RealImpact::ValidateDirectory(a.Directory);
                if (!object_name) return std::unexpected(std::move(object_name.error()));

                ClearMeshes(r, viewport);
                const auto [mesh_entity, instance_entity] = ImportMesh(
                    r,
                    a.Directory / "transformed.obj",
                    MeshInstanceCreateInfo{
                        .Name = std::move(*object_name),
                        .Transform = {.R = RealImpact::ObjectRotationToYUp},
                    }
                );

                // Ignore the npy file's vertex indices: deduplication may have invalidated them. Look up by position instead.
                std::vector<uint32_t> vertex_indices(RealImpact::NumImpactVertices);
                {
                    const auto impact_positions = RealImpact::LoadPositions(a.Directory);
                    const auto &mesh = r.get<Mesh>(mesh_entity);
                    for (size_t i = 0; i < impact_positions.size(); ++i) {
                        vertex_indices[i] = *mesh.FindNearestVertex(impact_positions[i]);
                    }
                }

                const auto listener_points = RealImpact::LoadListenerPoints(a.Directory);
                const auto [listener_mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh({primitive::Cylinder{0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f}}), {}, {}));
                for (const auto &listener_point : listener_points) {
                    static const auto rot_z = glm::angleAxis(float(M_PI_2), vec3{0, 0, 1}); // Cylinder's center is along the Y axis.
                    const auto listener_instance_entity = ::AddMeshInstance(
                        r, listener_mesh_entity,
                        {
                            .Name = std::format("RealImpact Microphone: {}", listener_point.Index),
                            .Transform = {
                                .P = listener_point.GetPosition(Defaults::World.Up, true),
                                .R = glm::angleAxis(glm::radians(float(listener_point.AngleDeg)), Defaults::World.Up) * rot_z,
                            },
                            .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                        }
                    );
                    r.emplace<RealImpactMicrophone>(listener_instance_entity, listener_point.Index);

                    if (listener_point.Index == RealImpact::CenteredListenerIndex) {
                        r.emplace<RealImpactActiveMicrophone>(instance_entity, listener_instance_entity);

                        auto material_name = RealImpact::FindMaterialName(r.get<Name>(instance_entity).Value);
                        if (const auto real_impact_material = material_name ?
                                find_if(materials::acoustic::All, [name = *material_name](const AcousticMaterial &m) { return m.Name == name; }) :
                                std::ranges::end(materials::acoustic::All)) {
                            r.emplace<AcousticMaterial>(mesh_entity, *real_impact_material);
                        }
                        r.emplace<ScaleLocked>(instance_entity);
                        r.emplace<RealImpactVertices>(instance_entity, vertex_indices);
                        SetVertexSamples(r, viewport, instance_entity, vertex_indices, to<std::vector>(RealImpact::LoadSamples(a.Directory, listener_point.Index)));
                    }
                }
                return {};
            },
        },
        action
    );
}
