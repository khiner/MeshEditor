#include "viewport/ViewportRenderGpu.h"
#include "Timer.h"
#include "animation/MorphWeightState.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "mesh/MeshStore.h"
#include "render/Drawing.h"
#include "render/Instance.h"
#include "render/Pipelines.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"
#include "vulkan/VulkanResources.h"

#include <entt/entity/registry.hpp>

using std::ranges::any_of, std::ranges::to;

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
constexpr vk::ClearColorValue ToClearColor(vec4 c) { return {std::array{c.r, c.g, c.b, c.a}}; }
// Inverse of the shader's display transform (toneMapPBRNeutral then sRGB encode), ignoring the tone
// map's desaturation term (not invertible, and negligible near the compression knee). Maps the
// display-referred clear color back to scene-linear for the HDR transmission pre-pass target.
vec3 DisplayToSceneLinear(vec3 color) {
    color = glm::pow(color, vec3{2.2f});
    constexpr float StartCompression = 0.8f - 0.04f;
    constexpr float D = 1.f - StartCompression;
    // The forward compression asymptotes to 1, so clamp just below to keep the inverse finite.
    if (const float peak = std::min(std::max({color.r, color.g, color.b}), 0.999f); peak >= StartCompression) {
        const float original_peak = D * D / (1.f - peak) - D + StartCompression;
        color *= original_peak / peak;
    }
    // Undo the black-level offset. The forward maps min channel x to 6.25x² for x < 0.08, else x - 0.04.
    const float x = std::min({color.r, color.g, color.b});
    const float offset = x < 0.04f ? 0.4f * std::sqrt(x) - x : 0.04f;
    return color + offset;
}
const vk::ClearColorValue Transparent{0, 0, 0, 0};
const std::vector<vk::ClearValue> SilhouetteClearValues{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};
DrawData MakeDrawData(const RenderBuffers &rb, uint32_t vertex_slot, const InstanceArena &instances) {
    return MakeDrawData(vertex_slot, rb.Vertices, rb.Indices, instances.TransformBuffer.Slot);
}
} // namespace

void FlushDrawList(entt::registry &r, vk::Device device, const DrawListBuilder &draw_list, DrawBufferPair &pair) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (!draw_list.Draws.empty()) pair.DrawData.Update(as_bytes(draw_list.Draws));
    if (!draw_list.IndirectCommands.empty()) pair.Indirect.Update(as_bytes(draw_list.IndirectCommands));
    buffers.EnsureIdentityIndexBuffer(draw_list.MaxIndexCount);
    if (auto descriptor_updates = buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        device.updateDescriptorSets(std::move(descriptor_updates), {});
        buffers.Ctx.ClearDeferredDescriptorUpdates();
    }
}

#ifdef MVK_FORCE_STAGED_TRANSFERS
void RecordTransferCommandBuffer(entt::registry &r, entt::entity viewport, vk::CommandBuffer cb) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    const Timer timer{"RecordTransferCommandBuffer"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    buffers.Ctx.RecordDeferredCopies(cb);
    cb.end();
}
#endif

namespace {
struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes) {
    std::unordered_map<entt::entity, DeformSlots> result;
    for (const auto [_, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
        if (result.contains(instance.Entity)) continue;
        const auto &mesh = GetMesh(r, instance.Entity);
        const auto bone_deform = meshes.GetBoneDeformRange(mesh.GetStoreId());
        if (bone_deform.Count == 0) continue;
        if (const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity)) {
            result[instance.Entity] = {
                .BoneDeformOffset = bone_deform.Offset,
                .ArmatureDeformOffset = pose_state->GpuDeformRange.Offset,
                .MorphDeformOffset = InvalidOffset,
                .MorphTargetCount = 0,
                .MorphWeightsByBufferIndex = {},
            };
        }
    }
    // Add morph target slots for mesh instances with morph data (per-instance weights)
    for (const auto [instance_entity, instance, gpu_range, ri] : r.view<const Instance, const MorphWeightGpuRange, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &mesh = GetMesh(r, mesh_entity);
        const auto morph_range = meshes.GetMorphTargetRange(mesh.GetStoreId());
        if (morph_range.Count == 0) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = morph_range.Offset;
        slots.MorphTargetCount = meshes.GetMorphTargetCount(mesh.GetStoreId());
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = gpu_range.Weights.Offset;
    }
    return result;
}

void PatchMorphWeights(DrawListBuilder &dl, size_t draws_before, const DeformSlots &deform) {
    if (deform.MorphWeightsByBufferIndex.empty()) return;
    for (size_t i = draws_before; i < dl.Draws.size(); ++i) {
        if (auto it = deform.MorphWeightsByBufferIndex.find(dl.Draws[i].FirstInstance); it != deform.MorphWeightsByBufferIndex.end()) {
            dl.Draws[i].MorphWeightsOffset = it->second;
        }
    }
}
// AppendExtrasDraw is templated on the customize_draw callable.
void AppendExtrasDraw(entt::registry &r, const InstanceArena &instances, DrawListBuilder &dl, DrawBatchInfo &batch, auto &&customize_draw) {
    batch = dl.BeginBatch();
    for (auto [entity, mesh_buffers, models] : r.view<ObjectExtrasTag, const MeshBuffers, const ModelsBuffer>().each()) {
        if (mesh_buffers.EdgeIndices.Count == 0) continue;
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, instances);
        if (const auto *vcr = r.try_get<VertexClass>(entity)) draw.VertexClassOffset = vcr->Offset;
        customize_draw(draw, instances);
        AppendDraw(dl, batch, mesh_buffers.EdgeIndices, models, draw);
    }
}
} // namespace

void RecordRenderCommandBuffer(entt::registry &r, entt::entity viewport, vk::CommandBuffer cb, bool silhouette_only) {
    const Timer timer{silhouette_only ? "RecordRenderCommandBuffer (silhouette)" : "RecordRenderCommandBuffer"};
    if (!silhouette_only) r.ctx().get<DrawState>().SelectionStale = true;

    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &pipelines = r.ctx().get<Pipelines>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = !is_wireframe_mode, show_overlays = settings.ShowOverlays;
    const auto &active_lighting = GetActivePbrLighting(r, viewport, settings.ViewportShading);
    const bool real_transmission = show_rendered &&
        active_lighting.RealTransmission &&
        pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);

    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &draw = r.ctx().get<DrawState>();
    auto &draw_list = draw.List;

    // Build mesh_entity -> deform slots mapping for skinned meshes (edit mode shows rest pose)
    const auto mesh_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(r, meshes);
    static const DeformSlots no_deform{};
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> const DeformSlots & {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return no_deform;
    };

    const auto is_silhouette_eligible = [&](entt::entity e) {
        if (!r.all_of<Instance, RenderInstance>(e)) return false;
        const auto buffer_entity = r.get<const Instance>(e).Entity;
        if (!r.valid(buffer_entity) || r.all_of<ObjectExtrasTag>(buffer_entity)) return false;
        // Bones get outlines from BoneWire/BoneSphereWire, not the screen-space silhouette system.
        if (r.all_of<ArmatureObject>(buffer_entity) || r.all_of<BoneJoint>(buffer_entity)) return false;
        const auto *mesh_buffers = r.try_get<const MeshBuffers>(buffer_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };

    // In Edit mode, compute primary edit instances (for draw routing and silhouette filtering)
    // and edit transform context (for pending vertex transforms) in one pass.
    std::unordered_map<entt::entity, entt::entity> primary_edit_instances;
    EditTransformContext edit_transform_context;
    const bool has_pending_transform = is_edit_mode && r.all_of<PendingTransform>(viewport);
    if (is_edit_mode) {
        const auto active = FindActiveEntity(r);
        for (const auto [e, instance, ok, ri] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
            if (ok.Value != ObjectType::Mesh) continue;
            if (auto &primary = primary_edit_instances[instance.Entity]; primary == entt::entity{} || e == active) primary = e;
            if (has_pending_transform && !r.all_of<ScaleLocked>(e)) {
                auto &primary_uf = edit_transform_context.TransformInstances[instance.Entity];
                if (primary_uf == entt::entity{} || e == active) primary_uf = e;
            }
        }
    }
    const auto patch_edit_pending_local_transform = [&](size_t draws_before, entt::entity mesh_entity) {
        if (!has_pending_transform) return;
        const auto context_it = edit_transform_context.TransformInstances.find(mesh_entity);
        if (context_it == edit_transform_context.TransformInstances.end()) return;
        const auto *primary_ri = r.try_get<const RenderInstance>(context_it->second);
        if (!primary_ri) return;
        for (size_t i = draws_before; i < draw_list.Draws.size(); ++i) {
            draw_list.Draws[i].HasPendingVertexTransform = 1u;
            draw_list.Draws[i].PrimaryEditInstanceIndex = primary_ri->BufferIndex;
        }
    };

    if (!silhouette_only) {
        draw_list.Draws.clear();
        draw_list.IndirectCommands.clear();
        draw_list.MaxIndexCount = 0;

        std::unordered_set<entt::entity> excitable_mesh_entities;
        if (is_excite_mode) {
            for (const auto [e, instance, excitable] : r.view<const Instance, const SoundVertices>().each()) {
                excitable_mesh_entities.emplace(instance.Entity);
            }
        }

        std::vector<entt::entity> blend_mesh_order;
        if (show_rendered) {
            // Transparent pass ordering: sort mesh draws back-to-front by camera distance.
            // This is a mesh-level approximation; interpenetrating transparent geometry may still require
            // per-primitive sorting or OIT for fully correct compositing.
            const auto camera_position = r.get<const ViewCamera>(viewport).Position();
            std::unordered_map<entt::entity, float> farthest_distance2_by_mesh;
            farthest_distance2_by_mesh.reserve(r.storage<RenderInstance>().size());
            for (const auto [entity, _, wt] : r.view<const RenderInstance, const WorldTransform>().each()) {
                entt::entity mesh_entity = entity;
                if (const auto *instance = r.try_get<const Instance>(entity)) mesh_entity = instance->Entity;
                if (!r.valid(mesh_entity) || !HasMesh(r, mesh_entity)) continue;
                const auto delta = wt.P - camera_position;
                const auto distance2 = dot(delta, delta);
                if (const auto it = farthest_distance2_by_mesh.find(mesh_entity); it != farthest_distance2_by_mesh.end()) {
                    it->second = std::max(it->second, distance2);
                } else {
                    farthest_distance2_by_mesh.emplace(mesh_entity, distance2);
                }
            }
            blend_mesh_order.reserve(farthest_distance2_by_mesh.size());
            for (const auto &[mesh_entity, _] : farthest_distance2_by_mesh) blend_mesh_order.emplace_back(mesh_entity);
            std::ranges::sort(
                blend_mesh_order,
                [&](const auto a, const auto b) {
                    // Break equal-distance ties (common for instances sharing a transform) by entity
                    // id, so the blend order — and thus the composited image — is deterministic.
                    const auto da = farthest_distance2_by_mesh.at(a), db = farthest_distance2_by_mesh.at(b);
                    if (da != db) return da > db;
                    return a < b;
                }
            );
        }

        // Single-pass entity data collection: avoids redundant ECS view iterations across fill, edge, wire, point, and normal batches.
        struct MeshEntityData {
            entt::entity Entity;
            const MeshBuffers &Buf;
            const ModelsBuffer &Mod;
            std::optional<Mesh> MeshComp;
            const DeformSlots &Deform;
            std::optional<uint32_t> PrimaryEditBufferIndex;
            bool IsSoundVertices, IsBone, IsBoneJoint, IsExtras, Smooth;
        };

        // Draw order resolves coincident surfaces, and pool iteration order varies with scene-load
        // history, so draw in descending entity id order (a fresh registry's iteration order).
        auto mesh_entity_order = r.view<const MeshBuffers, const ModelsBuffer>() | to<std::vector>();
        std::ranges::sort(mesh_entity_order, std::ranges::greater{});

        std::vector<MeshEntityData> mesh_entities;
        mesh_entities.reserve(mesh_entity_order.size());
        for (const auto entity : mesh_entity_order) {
            const auto &mesh_buffers = r.get<const MeshBuffers>(entity);
            const auto &models = r.get<const ModelsBuffer>(entity);
            std::optional<uint32_t> primary_bi;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                primary_bi = r.get<RenderInstance>(it->second).BufferIndex;
            }
            const bool is_bone_joint = r.all_of<BoneJoint>(entity);
            mesh_entities.emplace_back(entity, mesh_buffers, models, TryGetMesh(r, entity), get_deform_slots(entity), primary_bi, excitable_mesh_entities.contains(entity), r.all_of<ArmatureObject>(entity) || is_bone_joint, is_bone_joint, r.all_of<ObjectExtrasTag>(entity), r.all_of<SmoothShading>(entity));
        }

        // Entity -> mesh_entities index map for blend_mesh_order lookup
        std::unordered_map<entt::entity, size_t> blend_entity_idx;
        if (show_rendered) {
            blend_entity_idx.reserve(mesh_entities.size());
            for (size_t i = 0; i < mesh_entities.size(); ++i) blend_entity_idx[mesh_entities[i].Entity] = i;
        }

        if (show_fill) {
            const auto append_fill_mesh = [&](DrawBatchInfo &batch, const MeshEntityData &e, std::optional<bool> blend_target) {
                if (!e.MeshComp || e.MeshComp->FaceCount() == 0) return;
                const auto &mesh_buffers = e.Buf;
                const auto &models = e.Mod;
                const auto &mesh = *e.MeshComp;
                const auto &deform = e.Deform;
                auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
                const auto face_id_buffer = meshes.GetFaceIdRange(mesh.GetStoreId());
                const auto face_state_buffer = meshes.GetFaceStateRange(mesh.GetStoreId());
                const auto face_primitive_buffer = meshes.GetFacePrimitiveRange(mesh.GetStoreId());
                const auto primitive_material_buffer = meshes.GetPrimitiveMaterialRange(mesh.GetStoreId());
                dd.ObjectIdSlot = face_id_buffer.Slot;
                dd.FaceIdOffset = face_id_buffer.Offset;
                dd.FaceFirstTriOffset = e.Smooth ? InvalidOffset : meshes.GetFaceFirstTriRange(mesh.GetStoreId()).Offset;
                dd.FacePrimitiveOffset = face_primitive_buffer.Count > 0 ? face_primitive_buffer.Offset : InvalidOffset;
                dd.PrimitiveMaterialOffset = primitive_material_buffer.Count > 0 ? primitive_material_buffer.Offset : InvalidOffset;
                const auto append_fill_draw = [&](const DrawData &dd, uint32_t index_count, std::optional<uint32_t> model_index) {
                    const auto db = draw_list.Draws.size();
                    AppendDraw(draw_list, batch, index_count, models, dd, model_index);
                    PatchMorphWeights(draw_list, db, deform);
                    patch_edit_pending_local_transform(db, e.Entity);
                };
                const auto append_fill_for_instances = [&](const DrawData &dd, uint32_t index_count) {
                    if (e.PrimaryEditBufferIndex) {
                        // Draw primary with element state first, then all without (depth LESS won't overwrite)
                        auto primary_draw = dd;
                        primary_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(primary_draw, index_count, e.PrimaryEditBufferIndex);
                        auto other_draw = dd;
                        other_draw.ElementStateSlotOffset = {};
                        append_fill_draw(other_draw, index_count, std::nullopt);
                    } else {
                        auto all_draw = dd;
                        all_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(all_draw, index_count, std::nullopt);
                    }
                };

                if (show_rendered) {
                    const auto primitive_materials = meshes.GetPrimitiveMaterialIndices(mesh.GetStoreId());
                    const auto primitive_ranges = meshes.GetPrimitiveTriangleRanges(mesh.GetStoreId());
                    if (!primitive_materials.empty() && !primitive_ranges.empty()) {
                        const auto material_count = buffers.Materials.Count();
                        // Merge adjacent primitives with the same blend mode into single draw calls.
                        struct BlendDrawRange {
                            bool Blend;
                            uint32_t FirstTriangle, TriangleCount;
                        };
                        std::vector<BlendDrawRange> blend_ranges;
                        blend_ranges.reserve(primitive_ranges.size());
                        for (const auto &pr : primitive_ranges) {
                            if (pr.TriangleCount == 0u) continue;
                            auto pi = pr.PrimitiveIndex;
                            if (pi >= primitive_materials.size()) pi = primitive_materials.size() - 1u;
                            const bool is_blend = primitive_materials[pi] < material_count && buffers.Materials.Get(primitive_materials[pi]).AlphaMode == MaterialAlphaMode::Blend;
                            if (!blend_ranges.empty() && blend_ranges.back().Blend == is_blend) blend_ranges.back().TriangleCount += pr.TriangleCount;
                            else blend_ranges.emplace_back(is_blend, pr.FirstTriangle, pr.TriangleCount);
                        }
                        for (const auto &range : blend_ranges) {
                            if (blend_target && range.Blend != *blend_target) continue;
                            auto range_draw = dd;
                            range_draw.IndexSlotOffset.Offset += range.FirstTriangle * 3u;
                            range_draw.FaceIdOffset += range.FirstTriangle;
                            append_fill_for_instances(range_draw, range.TriangleCount * 3u);
                        }
                        return;
                    }
                }

                if (!blend_target || !*blend_target) append_fill_for_instances(dd, mesh_buffers.FaceIndices.Count);
            };

            if (show_rendered) {
                draw.FillOpaque = draw_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (!e.IsBone) append_fill_mesh(draw.FillOpaque, e, false);
                }
                draw.FillBlend = draw_list.BeginBatch();
                for (const auto mesh_entity : blend_mesh_order) {
                    if (auto it = blend_entity_idx.find(mesh_entity); it != blend_entity_idx.end()) {
                        append_fill_mesh(draw.FillBlend, mesh_entities[it->second], true);
                    }
                }
            } else {
                draw.FillOpaque = draw_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (!e.IsBone) append_fill_mesh(draw.FillOpaque, e, std::nullopt);
                }
            }
        }

        // Build bone batches for X-ray rendering (drawn after a mid-pass depth clear so bones are never occluded by scene meshes)
        // Only draw bones for: active armature in Edit/Pose mode, selected armatures in Object mode.
        const auto should_draw_armature_bones = [&](entt::entity arm_obj_entity) {
            if (is_wireframe_mode) return true; // Wireframe mode: always show bone outlines
            const bool is_bone_mode = is_edit_mode || interaction_mode == InteractionMode::Pose;
            if (is_bone_mode) return r.all_of<Active>(arm_obj_entity);
            return r.all_of<Selected>(arm_obj_entity);
        };
        // Map BoneJoint entities back to their owning armature object entities.
        std::unordered_map<entt::entity, entt::entity> joint_to_owner;
        for (const auto [e, arm_obj] : r.view<const ArmatureObject>().each()) {
            if (arm_obj.JointEntity != entt::null) joint_to_owner[arm_obj.JointEntity] = e;
        }
        draw.BoneFill = {};
        draw.BoneWire = {};
        draw.BoneSphereFill = {};
        draw.BoneSphereWire = {};
        if (show_overlays && settings.ShowBones) {
            draw.BoneFill = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : r.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                fill_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneWire = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : r.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (!should_draw_armature_bones(entity)) continue;
                if (const auto *adj = r.try_get<const BoneAdjacencyIndices>(entity)) {
                    auto wire_draw = MakeDrawData(mesh_buffers.Vertices, adj->Indices, buffers.Instances);
                    wire_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                    AppendDraw(draw_list, draw.BoneWire, adj->Indices.Count / 2, models, wire_draw);
                }
            }

            // Joint sphere batches
            draw.BoneSphereFill = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                fill_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneSphereWire = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.EdgeIndices.Count == 0) continue;
                if (const auto it = joint_to_owner.find(entity); it != joint_to_owner.end() && !should_draw_armature_bones(it->second)) continue;
                auto wire_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, buffers.Instances);
                wire_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereWire, mesh_buffers.EdgeIndices, models, wire_draw);
            }
        }

        // Edge quad batch (edit/excite mode triangle quads with self-AA, matches Blender's overlay_edit_mesh_edge)
        draw.EdgeQuad = draw_list.BeginBatch();
        if (is_edit_mode || is_excite_mode) {
            for (const auto &e : mesh_entities) {
                // Line meshes use draw.WireLine
                if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0 || e.Buf.FaceIndices.Count == 0) continue;
                auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                dd.ElementStateSlotOffset = meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
                const auto db = draw_list.Draws.size();
                if (e.PrimaryEditBufferIndex) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd, e.PrimaryEditBufferIndex);
                else if (e.IsSoundVertices) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd);
                PatchMorphWeights(draw_list, db, e.Deform);
                patch_edit_pending_local_transform(db, e.Entity);
            }
        }
        // Wire line batch (wireframe mode + line meshes, matches Blender's wireframe overlay)
        draw.WireLine = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0) continue;
            if (e.Buf.FaceIndices.Count > 0 && !is_wireframe_mode) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
            const auto db = draw_list.Draws.size();
            AppendDraw(draw_list, draw.WireLine, e.Buf.EdgeIndices, e.Mod, dd);
            PatchMorphWeights(draw_list, db, e.Deform);
            patch_edit_pending_local_transform(db, e.Entity);
        }

        draw.ExtrasLine = {};
        if (show_overlays && settings.ShowExtras) {
            AppendExtrasDraw(r, buffers.Instances, draw_list, draw.ExtrasLine, [](auto &, const auto &) {});
        }

        draw.Point = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone) continue;
            const bool is_point_mesh = e.Buf.FaceIndices.Count == 0 && e.Buf.EdgeIndices.Count == 0;
            if (!is_point_mesh && !((is_edit_mode && edit_mode == Element::Vertex) || is_excite_mode)) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.VertexIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = {meshes.GetVertexStateSlot(), e.Buf.Vertices.Offset};
            const auto db = draw_list.Draws.size();
            if (is_point_mesh) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd);
            else if (e.PrimaryEditBufferIndex) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd, e.PrimaryEditBufferIndex);
            else if (e.IsSoundVertices) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd);
            PatchMorphWeights(draw_list, db, e.Deform);
            patch_edit_pending_local_transform(db, e.Entity);
        }

        { // Normal overlay + bbox batches
            const auto vertex_slot = buffers.VertexBuffer.Buffer.Slot;
            draw.OverlayFaceNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Face); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayFaceNormals, it->second.Indices, e.Mod, dd);
                }
            }
            draw.OverlayVertexNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Vertex); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayVertexNormals, it->second.Indices, e.Mod, dd);
                }
            }
        }

        { // Build selection draw list
            auto &sel_list = draw.SelectionList;
            sel_list = {};

            const auto run_sel_pass = [&](auto &&indices_of, auto &&skip) {
                auto batch = sel_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (e.IsExtras || e.IsBoneJoint || skip(e)) continue;
                    const auto &indices = indices_of(e);
                    auto dd = MakeDrawData(e.Buf.Vertices, indices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                    dd.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
                    const auto db = sel_list.Draws.size();
                    if (e.PrimaryEditBufferIndex) AppendDraw(sel_list, batch, indices, e.Mod, dd, e.PrimaryEditBufferIndex);
                    else AppendDraw(sel_list, batch, indices, e.Mod, dd);
                    PatchMorphWeights(sel_list, db, e.Deform);
                }
                return batch;
            };
            const auto sel_tri = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.FaceIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count == 0; }
            );
            const auto sel_line = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.EdgeIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count == 0; }
            );
            const auto sel_point = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.VertexIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count > 0; }
            );

            DrawBatchInfo sel_bone_sphere;
            if (show_overlays && settings.ShowBones) {
                sel_bone_sphere = sel_list.BeginBatch();
                for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                    if (mesh_buffers.FaceIndices.Count == 0) continue;
                    auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                    dd.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
                    AppendDraw(sel_list, sel_bone_sphere, mesh_buffers.FaceIndices, models, dd);
                }
            }

            DrawBatchInfo sel_extras;
            if (show_overlays && settings.ShowExtras) {
                AppendExtrasDraw(r, buffers.Instances, sel_list, sel_extras, [](auto &dd, const auto &instances) {
                    dd.ObjectIdSlot = instances.ObjectIdBuffer.Slot;
                });
            }

            draw.SelectionDraws = {
                {SPT::SelectionFragmentTriangles, sel_tri},
                {SPT::SelectionFragmentLines, sel_line},
                {SPT::SelectionFragmentPoints, sel_point},
                {SPT::SelectionFragmentBoneSphere, sel_bone_sphere},
                {SPT::SelectionObjectExtrasLines, sel_extras},
            };
        }

        // Cache the main draw list state (everything before silhouette).
        draw.MainDrawCount = draw_list.Draws.size();
        draw.MainIndirectCount = draw_list.IndirectCommands.size();
    } else {
        // Silhouette-only: truncate to cached main portion, batch infos retained from last full build.
        draw_list.Draws.resize(draw.MainDrawCount);
        draw_list.IndirectCommands.resize(draw.MainIndirectCount);
    }

    // Silhouette batch (appended after main batches in both paths).
    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, instance, ri] : r.view<const Instance, const Selected, const RenderInstance>().each()) {
            if (!is_silhouette_eligible(e)) continue;
            if (auto it = primary_edit_instances.find(instance.Entity); it == primary_edit_instances.end() || it->second != e) {
                silhouette_instances.insert(e);
            }
        }
    }

    const bool has_object_silhouette_selection =
        any_of(r.view<const Selected, const Instance, const RenderInstance>().each(), [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); });
    const bool render_silhouette = (show_overlays && settings.ShowOutlineSelected) && !is_excite_mode &&
        (is_edit_mode ? !silhouette_instances.empty() : has_object_silhouette_selection);

    draw.Silhouette = {};
    if (render_silhouette) {
        draw.Silhouette = draw_list.BeginBatch();
        auto append_silhouette = [&](entt::entity e) {
            if (!is_silhouette_eligible(e)) return;
            const auto mesh_entity = r.get<Instance>(e).Entity;
            const auto &mesh_buffers = r.get<MeshBuffers>(mesh_entity);
            const auto &models = r.get<ModelsBuffer>(mesh_entity);
            const auto &deform = get_deform_slots(mesh_entity);
            auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            dd.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
            const auto draws_before = draw_list.Draws.size();
            AppendDraw(draw_list, draw.Silhouette, mesh_buffers.FaceIndices, models, dd, r.get<RenderInstance>(e).BufferIndex);
            PatchMorphWeights(draw_list, draws_before, deform);
            patch_edit_pending_local_transform(draws_before, mesh_entity);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) append_silhouette(e);
        } else {
            for (const auto e : r.view<Selected, RenderInstance>()) append_silhouette(e);
        }
    }

    FlushDrawList(r, vk.Device, draw_list, buffers.RenderDraw);
    const auto render_extent_px = RenderExtentPx(r);
    const vk::Extent2D render_extent{render_extent_px.x, render_extent_px.y};

    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});
    const uint32_t transform_vertex_state_slot = is_edit_mode ? meshes.GetVertexStateSlot() : InvalidSlot;
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const MainDrawPushConstants pc{{batch.DrawDataSlotOffset, transform_vertex_state_slot}};
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*buffers.RenderDraw.Indirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    auto record_pbr_batch = [&](const DrawBatchInfo &batch, PbrCompiler::Variant variant) {
        if (batch.DrawCount == 0) return;
        const auto layout = pipelines.Main.Compiler.Bind(cb, variant);
        const MainDrawPushConstants pc{{batch.DrawDataSlotOffset, transform_vertex_state_slot}};
        cb.pushConstants(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*buffers.RenderDraw.Indirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    const auto make_shader_read_barrier = [](vk::AccessFlags src_access, vk::ImageLayout layout, vk::Image image, const vk::ImageSubresourceRange &range) {
        return vk::ImageMemoryBarrier{src_access, vk::AccessFlagBits::eShaderRead, layout, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range};
    };
    const auto sync_fragment_shader_reads = [&](vk::PipelineStageFlags src_stages, auto &&barriers) {
        cb.pipelineBarrier(src_stages, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barriers);
    };

    const bool has_silhouette = render_silhouette && draw.Silhouette.DrawCount > 0;
    if (has_silhouette) { // Silhouette depth/object pass
        const auto &silhouette = pipelines.Silhouette;
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(silhouette.Renderer, SPT::SilhouetteDepthObject, draw.Silhouette);
        cb.endRenderPass();

        // Silhouette pass offscreen color writes -> edge pass fragment sampling.
        const std::array silhouette_to_edge_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eColorAttachmentOutput, silhouette_to_edge_barriers);

        const auto &silhouette_edge = pipelines.SilhouetteEdge;
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        const SilhouetteEdgeDepthObjectPushConstants edge_pc{sel_slots.SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();

        // Edge pass depth/color writes -> main pass silhouette sampling.
        const std::array edge_to_main_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::ImageLayout::eDepthStencilReadOnlyOptimal, *silhouette_edge.Resources->DepthImage.Image, DepthSubresourceRange),
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette_edge.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eLateFragmentTests | vk::PipelineStageFlagBits::eColorAttachmentOutput, edge_to_main_barriers);
    }

    const auto &main = pipelines.Main;
    const vk::Rect2D main_rect{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)};
    const std::vector<vk::ClearValue> main_clear_values{
        {vk::ClearDepthStencilValue{1, 0}},
        {ToClearColor(settings.ClearColor)},
        {vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 0}}},
    };

    // Real-transmission pre-pass: render Background + FillOpaque into TransmissionImage mip 0 in
    // scene-linear HDR, sampled by the main pass at the refracted exit point. TRANSMISSION_PREPASS
    // variants skip the display transform and drop transmission materials (no self-sampling).
    if (real_transmission && main.Transmission) {
        // Clear with the display clear color mapped back to scene-linear and un-exposed, so the
        // refracted view of the empty backdrop displays as the same clear color after the main pass.
        const vec3 clear_linear = DisplayToSceneLinear(vec3{settings.ClearColor}) / std::exp2(active_lighting.ExposureEV);
        const std::array prepass_clear_values{
            vk::ClearValue{vk::ClearDepthStencilValue{1, 0}},
            vk::ClearValue{ToClearColor(vec4{clear_linear, settings.ClearColor.a})},
            vk::ClearValue{Transparent},
        };
        cb.beginRenderPass({*main.PrepassRenderPass, *main.Transmission->Framebuffer, main_rect, prepass_clear_values}, vk::SubpassContents::eInline);
        main.PrepassBackground.RenderQuad(cb);
        if (buffers.IdentityIndexCount > 0 && show_fill) {
            cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            record_pbr_batch(draw.FillOpaque, PbrCompiler::Variant::OpaquePrepass);
        }
        cb.endRenderPass();

        // Generate mip chain via linear blits. After the render pass, mip 0 is in eShaderReadOnlyOptimal
        // (per attachment finalLayout); we re-transition it for blits, then leave all mips in eShaderReadOnlyOptimal.
        const auto mip_count = main.Transmission->MipCount;
        const auto image = *main.Transmission->Image.Image;
        // mip 0: shaderRO -> transferSrc
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
            {{vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead,
              vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
              vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}}
        );
        // mips 1..N-1: undefined -> transferDst
        if (mip_count > 1) {
            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                {{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 1, mip_count - 1, 0, 1}}}
            );
        }
        int32_t mip_w = int32_t(main_rect.extent.width), mip_h = int32_t(main_rect.extent.height);
        for (uint32_t mip = 1; mip < mip_count; ++mip) {
            const int32_t next_w = std::max(1, mip_w / 2);
            const int32_t next_h = std::max(1, mip_h / 2);
            cb.blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                vk::ImageBlit{
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip - 1, 0, 1},
                    {vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_w, mip_h, 1}},
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, 1},
                    {vk::Offset3D{0, 0, 0}, vk::Offset3D{next_w, next_h, 1}},
                },
                vk::Filter::eLinear
            );
            // Promote this mip from transferDst to transferSrc for the next iteration.
            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                {{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
                  vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
                  vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1}}}
            );
            mip_w = next_w;
            mip_h = next_h;
        }
        // All mips currently in transferSrc — flip to shaderReadOnly for sampling in the main pass.
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
            {{vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
              vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
              vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, mip_count, 0, 1}}}
        );
    }

    // Main rendering pass
    cb.beginRenderPass({*main.Renderer.RenderPass, *main.Resources->Framebuffer, main_rect, main_clear_values}, vk::SubpassContents::eInline);

    // MoltenVK/Metal workaround: the grid's late depth test (it writes gl_FragDepth) misreads fast-cleared depth.
    // If no triangle draws will resolve the fast-clear before the grid, re-clear depth up front, before any depth writes.
    if (show_overlays && settings.ShowGrid && !has_silhouette && draw.FillOpaque.DrawCount == 0) {
        const vk::ClearAttachment grid_depth_resolve{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
        const vk::ClearRect grid_clear_rect{{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)}, 0, 1};
        cb.clearAttachments(grid_depth_resolve, grid_clear_rect);
    }

    // Background environment (PBR modes only; shader discards when WorldOpacity == 0 or no env slot)
    if (show_rendered) main.Renderer.ShaderPipelines.at(SPT::Background).RenderQuad(cb);

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (has_silhouette) {
        const auto &silhouette_depth = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth);
        const uint32_t depth_sampler_index = sel_slots.DepthSampler;
        cb.pushConstants(*silhouette_depth.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(depth_sampler_index), &depth_sampler_index);
        silhouette_depth.RenderQuad(cb);
    }

    if (buffers.IdentityIndexCount > 0) { // Meshes
        cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        // Solid faces
        if (show_fill) {
            if (show_rendered) {
                record_pbr_batch(draw.FillOpaque, PbrCompiler::Variant::Opaque);
                record_pbr_batch(draw.FillBlend, PbrCompiler::Variant::Blend);
            } else {
                record_draw_batch(main.Renderer, SPT::Fill, draw.FillOpaque);
            }
        }
        // Edit mode edges as triangle quads with self-AA
        record_draw_batch(main.Renderer, SPT::EdgeQuad, draw.EdgeQuad);
        // Wireframe/line mesh edges as GPU lines (LineAA composite handles AA)
        record_draw_batch(main.Renderer, SPT::Line, draw.WireLine);
        // Vertex points (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Point, draw.Point);
        // Object extras (cameras, lights, empties)
        record_draw_batch(main.Renderer, SPT::ObjectExtrasLine, draw.ExtrasLine);
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (has_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        // In mesh Edit mode, suppress active silhouette (element selection drives active state differently).
        // In armature Edit/Pose mode, the active bone gets the active-color silhouette.
        const auto active_entity = FindActiveEntity(r);
        const auto active_bone = FindActiveBone(r);
        const bool armature_mode = FindArmatureObject(r, active_entity) != entt::null;
        uint32_t active_object_id = 0;
        if (armature_mode && active_bone != entt::null) {
            if (r.all_of<RenderInstance>(active_bone)) {
                active_object_id = r.get<RenderInstance>(active_bone).ObjectId;
            }
        } else if (!is_edit_mode) {
            if (active_entity != entt::null && r.all_of<RenderInstance>(active_entity)) {
                active_object_id = r.get<RenderInstance>(active_entity).ObjectId;
            }
        }
        const SilhouetteEdgeColorPushConstants pc{
            TransformGizmo::IsUsing(r, viewport) && interaction_mode == InteractionMode::Object, sel_slots.ObjectIdSampler, active_object_id
        };
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        silhouette_edc.RenderQuad(cb);
    }

    if (buffers.IdentityIndexCount > 0) { // Selection overlays
        cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(main.Renderer, SPT::LineOverlayFaceNormals, draw.OverlayFaceNormals);
        record_draw_batch(main.Renderer, SPT::LineOverlayVertexNormals, draw.OverlayVertexNormals);
    }

    // Grid lines texture (drawn before bone depth clear so grid remains depth-tested against scene meshes)
    if (show_overlays && settings.ShowGrid) {
        main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);
    }

    { // Bone X-ray: clear depth so bones are never occluded by scene meshes (only mutually occlude each other)
        if (draw.BoneFill.DrawCount > 0 || draw.BoneSphereFill.DrawCount > 0) {
            cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            const vk::ClearAttachment depth_clear{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
            const auto extent = main.Resources->ColorImage.Extent;
            const vk::ClearRect clear_rect{{{0, 0}, ToExtent2D(extent)}, 0, 1};
            cb.clearAttachments(depth_clear, clear_rect);

            // In Object+wireframe mode, show only outlines (no fills).
            // In Edit/Pose+wireframe, fills are semitransparent and write far-plane depth (via shader) so wires are never occluded.
            const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
            if (!object_wireframe) {
                record_draw_batch(main.Renderer, SPT::BoneFill, draw.BoneFill);
                record_draw_batch(main.Renderer, SPT::BoneSphereFill, draw.BoneSphereFill);
            }
            // In non-wireframe Object mode, "Outline selected" off suppresses bone wire outlines.
            // In wireframe+Object mode, wires are the only bone visualization so always show them.
            const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                (!show_overlays || !settings.ShowOutlineSelected);
            if (!hide_bone_outlines) {
                record_draw_batch(main.Renderer, SPT::BoneWire, draw.BoneWire);
                record_draw_batch(main.Renderer, SPT::BoneSphereWire, draw.BoneSphereWire);
            }
        }
    }

    cb.endRenderPass();

    // The render pass ExternalFragReadDependency should cover this, but MoltenVK needs an explicit barrier
    // to flush the Metal render encoder's color writes before the next encoder samples them.
    {
        const std::array main_to_lineaa_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *main.Resources->ColorImage.Image, ColorSubresourceRange),
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *main.Resources->LineDataImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eColorAttachmentOutput, main_to_lineaa_barriers);
    }

    { // Line AA composite pass: blends anti-aliased lines from LineDataImage onto ColorImage → FinalColorImage
        const vk::ClearValue clear_value{vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 1}}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->FinalColorImage.Extent)};
        cb.beginRenderPass({*main.LineAARenderPass, *main.Resources->LineAAFramebuffer, rect, clear_value}, vk::SubpassContents::eInline);
        const struct {
            uint32_t ColorSamplerSlot, LineDataSamplerSlot;
        } line_aa_pc{sel_slots.ColorSampler, sel_slots.LineDataSampler};
        cb.pushConstants(*main.LineAAComposite.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(line_aa_pc), &line_aa_pc);
        main.LineAAComposite.RenderQuad(cb);
        cb.endRenderPass();
    }

    cb.end();
}
