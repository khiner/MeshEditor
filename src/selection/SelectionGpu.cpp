#include "selection/SelectionGpu.h"

#include "Profile.h"
#include "gpu/SoundPointPushConstants.h"

#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"
#include "gpu/VisibilitySelectionPushConstants.h"
#include "mesh/MeshStore.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "object/ObjectComponents.h"
#include "render/Drawing.h"
#include "render/Encoding.h"
#include "render/Instance.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionQueries.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"

#include <entt/entity/registry.hpp>

namespace {
void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// Selection queries complete synchronously for immediate readback.
void SubmitAndWait(MTL::CommandBuffer *command_buffer) {
    const profile::CpuScope scope{"SelectionSubmit"};
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

// Where a click landed, and how far from it the raster may resolve an element.
struct ElementPickTarget {
    uvec2 Px;
    uint32_t RadiusSq;
    bool ResolveId; // The second raster, which takes the lowest id reporting the winning key.
};

// A face resolves at the cursor pixel, and vertices and edges snap from within their radius.
uint32_t ElementPickRadiusSq(Element element) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    return radius * radius;
}

ElementSelectQuery MakeElementQuery(
    const SelectionSlots &sel_slots, uvec4 box, uint32_t box_result_slot, const std::optional<ElementPickTarget> &pick
) {
    return {
        box,
        box_result_slot,
        pick ? pick->Px : uvec2{},
        pick ? pick->RadiusSq : 0u,
        pick ? sel_slots.ElementPickKey : InvalidSlot,
        pick && pick->ResolveId ? sel_slots.ElementPickId : InvalidSlot,
    };
}

// No fragment writes an all-ones key or id, so both double as the empty state.
constexpr uint32_t EmptyElementPick{~uint32_t{0}};

void ResetElementPick(GpuBuffers &buffers) {
    *buffers.ElementPickKey.Data() = EmptyElementPick;
    *buffers.ElementPickId.Data() = EmptyElementPick;
}

// The element the two raster passes agreed on, or nothing when the cursor missed.
std::optional<uint32_t> ReadNearestPickedElement(const GpuBuffers &buffers, uint32_t max_element_id) {
    const uint32_t id = *buffers.ElementPickId.Data();
    if (id == EmptyElementPick || id == 0 || id > max_element_id) return {};
    return id - 1;
}

uint32_t MaxElementBound(auto &&ranges) {
    return std::ranges::fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &r) { return std::max(total, r.Offset + r.Count); });
}

void BindVisibilitySelectionTextures(MTL::RenderCommandEncoder *encoder, const Pipelines &pipelines) {
    encoder->setFragmentTexture(*pipelines.Main.Resources->VisibilityImage, 0u);
    encoder->setFragmentTexture(*pipelines.Main.Resources->DepthImage, 1u);
}

// Draw selection fragments into silhouette depth.
void RunSelectionPass(
    entt::registry &r, mtl::PassChain &chain, const DrawListBuilder &draw_list,
    bool render_depth, bool render_silhouette, bool draw_meshlets, uint32_t meshlet_flags, auto &&record_draws
) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    FlushDrawList(r, draw_list, buffers.SelectionDraw);
    buffers.SetSceneViewDrawSlots(buffers.SelectionDraw);

    if (render_depth) RecordSilhouetteDepthPass(chain, slots, pipelines, buffers, render_silhouette);
    if (draw_meshlets && buffers.MeshletInstanceCount > 0) {
        RecordMeshletCull(chain, slots, pipelines, buffers, {.RequiredInstanceFlags = meshlet_flags});
    }

    const auto extent = pipelines.Silhouette.Resources->DepthImage.Extent;
    const auto pass = mtl::MakePassDescriptor({}, mtl::LoadDepth(*pipelines.Silhouette.Resources->DepthImage, MTL::StoreActionDontCare));
    pass->setRenderTargetWidth(extent.Width);
    pass->setRenderTargetHeight(extent.Height);
    // The pick resolve reads the key an earlier raster wrote, and bindless buffers carry no tracked hazard.
    auto *encoder = encode::BeginScenePass(chain, pass, "SelectionDraws", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageBlit | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers);
    record_draws(encoder, extent);
}

void RenderElementSelectionPass(
    entt::registry &r, mtl::PassChain &chain, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max, std::optional<ElementPickTarget> pick
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    const auto primary_edit_instances = selection::ComputePrimaryEditInstances(r);
    const bool xray_selection = r.get<const SelectionXRay>(viewport).Value;
    const auto &selection = pipelines.SelectionFragment;
    // Faces raster through the visibility or meshlet paths, so only vertices and edges land here.
    const auto &element_raster = [&]() -> const mtl::MeshRenderPipeline & {
        if (element == Element::Vertex) {
            if (xray_selection) return write_bitset ? selection.ElementVertexXRayBitsetBox : selection.ElementVertexXRay;
            return write_bitset ? selection.ElementVertexBitsetBox : selection.ElementVertex;
        }
        if (xray_selection) return write_bitset ? selection.ElementEdgeXRayBitsetBox : selection.ElementEdgeXRay;
        return write_bitset ? selection.ElementEdgeBitsetBox : selection.ElementEdge;
    }();

    DrawListBuilder draw_list;
    const bool render_depth = !xray_selection;
    auto element_batch = draw_list.BeginBatch();
    auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({0, buffers.Instances.RecordBuffer.Count<InstanceRecord>()});
    for (auto &record : records) record.Flags &= ~uint32_t(MeshletInstanceFlag::ElementSelection);
    const bool face_point_fallback = element == Element::Face && write_bitset && xray_selection;
    for (const auto &range : ranges) {
        const auto &mesh_buffers = r.get<MeshBuffers>(range.MeshEntity);
        const auto &models = r.get<ModelsBuffer>(range.MeshEntity);
        const auto primary = primary_edit_instances.find(range.MeshEntity);
        const auto mark_instance = [&](uint32_t slot) {
            if (slot >= records.size()) return;
            records[slot].Flags |= uint32_t(MeshletInstanceFlag::ElementSelection);
            records[slot].ElementIdOffset = range.Offset;
        };
        if (primary != primary_edit_instances.end()) {
            mark_instance(r.get<RenderInstance>(primary->second).BufferIndex);
        } else {
            for (uint32_t i = 0; i < models.InstanceCount; ++i) mark_instance(models.InstanceRange.Offset + i);
        }
        if (element == Element::Face && !face_point_fallback) continue;

        const auto &mesh = GetMesh(r, range.MeshEntity);
        const auto &indices = element == Element::Vertex ? mesh_buffers.VertexIndices :
            element == Element::Edge                     ? mesh_buffers.EdgeIndices :
                                                           mesh_buffers.FaceIndices;
        auto draw = MakeDrawData(mesh_buffers.Vertices, indices, buffers.Instances);
        if (element == Element::Face) {
            const auto face_id_buffer = meshes.GetFaceIdRange(mesh.GetStoreId());
            draw.ObjectIdSlot = face_id_buffer.Slot;
            draw.FaceIdOffset = face_id_buffer.Offset;
        } else {
            draw.ObjectIdSlot = InvalidSlot;
            draw.FaceIdOffset = 0;
        }
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementIdOffset = range.Offset;
        if (primary != primary_edit_instances.end()) {
            AppendDraw(draw_list, element_batch, indices, models, draw, r.get<RenderInstance>(primary->second).BufferIndex);
        } else {
            AppendDraw(draw_list, element_batch, indices, models, draw);
        }
    }

    RunSelectionPass(
        r, chain, draw_list, render_depth, element != Element::Face, element == Element::Face && xray_selection,
        uint32_t(MeshletInstanceFlag::ElementSelection),
        [&](auto *encoder, mtl::Extent2D) {
            const SelectionElementPushConstants element_pc{MakeElementQuery(sel_slots, {box_min.x, box_min.y, box_max.x, box_max.y}, sel_slots.SelectionBitset, pick)};
            const auto raster = [&](const mtl::MeshRenderPipeline &pipeline, uint32_t indices_per_element, uint32_t elements_per_group) {
                pipeline.Bind(encoder);
                encoder->setFragmentBytes(&element_pc, sizeof(element_pc), BufferIndex_PushConstants);
                encode::DispatchMeshBatch(encoder, draw_list, element_batch, indices_per_element, elements_per_group, elements_per_group * indices_per_element);
            };
            const auto raster_lines = [&](const mtl::MeshRenderPipeline &pipeline) { raster(pipeline, 2, uint32_t(OverlayDispatch::LineGroupLines)); };
            const auto raster_points = [&](const mtl::MeshRenderPipeline &pipeline) { raster(pipeline, 1, uint32_t(OverlayDispatch::PointGroupPoints)); };
            if (element == Element::Face) {
                if (xray_selection) {
                    const auto &pipeline = write_bitset ? selection.MeshletFaceXRayBitsetBox : selection.MeshletFaceXRay;
                    pipeline.Bind(encoder);
                    encode::SetPushConstants(encoder, element_pc);
                    DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection));
                } else {
                    const auto &pipeline = write_bitset ? selection.VisibilityFaceBitsetBox : selection.VisibilityFace;
                    pipeline.Bind(encoder);
                    BindVisibilitySelectionTextures(encoder, pipelines);
                    encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{encode::VisibilityDecodePc(buffers), {}, element_pc.Query});
                    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
                }
            } else if (element_batch.DrawCount > 0) {
                if (element == Element::Vertex) raster_points(element_raster);
                else raster_lines(element_raster);
            }
            if (write_bitset && xray_selection) {
                // X-Ray face: the point pass catches edge-on faces, whose projected triangle has zero area.
                if (element == Element::Face) raster_points(selection.ElementFaceXRayPointsBitsetBox);
                // X-Ray edge: the point pass catches near-zero-length projected edges.
                if (element == Element::Edge) raster_points(selection.ElementEdgeXRayPointsBitsetBox);
            }
        }
    );

    r.ctx().get<DrawState>().SelectionStale = true;
}

} // namespace

std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px
) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const profile::CpuScope scope{"RunElementPick"};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    ResetElementPick(buffers);
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        const auto radius_sq = ElementPickRadiusSq(element);
        for (const bool resolve_id : {false, true}) {
            RenderElementSelectionPass(
                r, chain, viewport, ranges, element, false, {}, {},
                ElementPickTarget{mouse_px, radius_sq, resolve_id}
            );
        }
    }
    SubmitAndWait(command_buffer);
    if (const auto index = ReadNearestPickedElement(buffers, element_count)) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
    }
    return {};
}

// The mesh-shader pick emissions one selection pass draws, mirroring the overlay draws.
struct PickEmissions {
    std::vector<ExtrasLinePushConstants> ExtrasLines{};
    DrawBatchInfo BoneSpheres{};
    DrawBatchInfo Lines{}, Points{};
    std::optional<SoundPointPushConstants> SoundPoints{};
    std::optional<ElementPickTarget> Pick{};
    ObjectSelectQuery Object{};
};

void RenderSelectionPassWith(
    entt::registry &r, mtl::PassChain &chain, [[maybe_unused]] entt::entity viewport, bool render_depth,
    const SelectionBuildFn &build_fn, bool render_silhouette = true, bool decode_visibility_objects = false,
    const PickEmissions &picks = {}
) {
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    DrawListBuilder draw_list;
    build_fn(draw_list);

    const auto &selection = pipelines.SelectionFragment;
    // Visibility ids index the visible-meshlet list the scene pass produced, so this pass reads that
    // list and must not run a cull of its own over it.
    RunSelectionPass(r, chain, draw_list, render_depth, render_silhouette, /*draw_meshlets=*/false, 0, [&](auto *encoder, mtl::Extent2D) {
        const SelectionDrawPushConstants sel_pc{picks.Object};
        // Excite mode picks over the excitable vertices themselves, so only those can be struck.
        if (const auto &sound_points = picks.SoundPoints) {
            // Sound points raster through the element pick fragment, which reads the cursor from its own constants.
            const SelectionElementPushConstants point_pc{MakeElementQuery(sel_slots, {}, InvalidSlot, picks.Pick)};
            selection.SoundPoint.Bind(encoder);
            encoder->setFragmentBytes(&point_pc, sizeof(point_pc), BufferIndex_PushConstants);
            encode::SetMeshPushConstants(encoder, *sound_points);
            encoder->drawMeshThreadgroups(MTL::Size((sound_points->VertexCount + 159) / 160, 1, 1), MTL::Size(1, 1, 1), MTL::Size(160, 1, 1));
        }
        // Line and point meshes pick from the same emissions the overlay draws.
        const auto pick_batch = [&](const mtl::MeshRenderPipeline &pipeline, const DrawBatchInfo &batch, uint32_t indices_per_element, uint32_t elements_per_group, uint32_t threads_per_group) {
            if (batch.DrawCount == 0) return;
            pipeline.Bind(encoder);
            encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
            encode::DispatchMeshBatch(encoder, draw_list, batch, indices_per_element, elements_per_group, threads_per_group);
        };
        constexpr auto group_lines = uint32_t(OverlayDispatch::LineGroupLines);
        constexpr auto group_points = uint32_t(OverlayDispatch::PointGroupPoints);
        pick_batch(selection.Line, picks.Lines, 2, group_lines, group_lines * 2);
        pick_batch(selection.Point, picks.Points, 1, group_points, group_points);

        // Bone joints pick from the same emission the overlay draws, one threadgroup per joint.
        if (picks.BoneSpheres.DrawCount > 0) {
            selection.BoneSphere.Bind(encoder);
            encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
            encode::DispatchInstancedMeshBatch(encoder, draw_list, picks.BoneSpheres, uint32_t(OverlayDispatch::BoneSphereVertices));
        }
        // Extras lines (gizmos and collision shape wireframes) pick from the same emission the overlay draws.
        if (!picks.ExtrasLines.empty()) {
            selection.ExtrasLine.Bind(encoder);
            encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
            encode::DispatchExtrasLines(encoder, picks.ExtrasLines);
        }
        if (decode_visibility_objects && buffers.MeshletInstanceCount > 0) {
            selection.VisibilityObject.Bind(encoder);
            BindVisibilitySelectionTextures(encoder, pipelines);
            encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{encode::VisibilityDecodePc(buffers), picks.Object, {}});
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
        }
    });
}

void RenderSelectionPass(entt::registry &r, mtl::PassChain &chain, entt::entity viewport, const ObjectSelectQuery &query) {
    // Render depth so the selection-fragment pass has a valid depth attachment to load, even right after a resize recreated it.
    // Object-pick ignores depth, so its contents and the silhouette draws don't matter.
    const auto &draw = r.ctx().get<const DrawState>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    RenderSelectionPassWith(
        r, chain, viewport, /*render_depth=*/true,
        [&draw](DrawListBuilder &draw_list) { draw_list = draw.SelectionList; },
        /*render_silhouette=*/false,
        /*decode_visibility_objects=*/true,
        PickEmissions{
            .ExtrasLines = settings.ShowOverlays && settings.ShowExtras ?
                CollectExtrasLines(r, r.ctx().get<const GpuBuffers>().Instances) :
                std::vector<ExtrasLinePushConstants>{},
            .BoneSpheres = draw.SelectionBoneSpheres,
            .Lines = draw.SelectionLines,
            .Points = draw.SelectionPoints,
            .Object = query,
        }
    );
}

void RunBoxSelectElements(entt::registry &r, entt::entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    auto &buffers = r.ctx().get<GpuBuffers>();
    const profile::CpuScope scope{"RunBoxSelectElements"};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > GpuBuffers::SelectionBitsetWords) return;

    // Restore baseline bitset for additive mode, or clear for non-additive.
    auto *bits = buffers.SelectionBitset.Data();
    if (is_additive) {
        const auto *baseline = r.try_get<const AdditiveBoxSelectBaseline>(viewport);
        if (baseline && !baseline->ElementBitset.empty()) {
            const auto copy_words = std::min(bitset_words, uint32_t(baseline->ElementBitset.size()));
            memcpy(bits, baseline->ElementBitset.data(), copy_words * sizeof(uint32_t));
            if (copy_words < bitset_words) { // Zero any remaining words beyond the baseline
                memset(&bits[copy_words], 0, (bitset_words - copy_words) * sizeof(uint32_t));
            }
        }
    } else {
        memset(bits, 0, bitset_words * sizeof(uint32_t));
    }

    // Box-select writes element IDs directly from the selection fragment shader.
    auto *command_buffer = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        RenderElementSelectionPass(r, chain, viewport, ranges, element, true, box_min, box_max, {});
    }
    SubmitAndWait(command_buffer);
    ApplySelectionStateUpdate(r, viewport, ranges, element);
}

std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &r, entt::entity viewport, entt::entity instance_entity, uvec2 mouse_px) {
    if (!r.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = r.try_get<Instance>(instance_entity);
    if (!instance) return {};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();

    const profile::CpuScope scope{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = r.get<MeshBuffers>(mesh_entity);
    const auto &models = r.get<ModelsBuffer>(mesh_entity);
    const auto model_index = r.get<RenderInstance>(instance_entity).BufferIndex;
    ResetElementPick(buffers);
    auto *command_buffer = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        const auto sound_vertices = r.get<const SoundVertices>(instance_entity).Vertices;
        for (const bool resolve_id : {false, true}) {
            RenderSelectionPassWith(
                r, chain, viewport, true,
                [&](DrawListBuilder &draw_list) {
                    // The pick emission reads this one draw at index zero for the instance's transform and positions.
                    auto batch = draw_list.BeginBatch();
                    auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, buffers.Instances);
                    AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
                },
                /*render_silhouette=*/true, /*decode_visibility_objects=*/false,
                PickEmissions{
                    .SoundPoints = SoundPointPushConstants{
                        .DrawDataIndex = 0,
                        .VertexSlot = meshes.GetSoundVertexSlot(),
                        .VertexOffset = sound_vertices.Offset,
                        .VertexCount = sound_vertices.Count,
                    },
                    .Pick = ElementPickTarget{mouse_px, ElementPickRadiusSq(Element::Vertex), resolve_id},
                }
            );
        }
    }
    SubmitAndWait(command_buffer);
    r.ctx().get<DrawState>().SelectionStale = true;
    return ReadNearestPickedElement(buffers, vertex_count);
}

std::vector<entt::entity> RunObjectPick(entt::registry &r, entt::entity viewport, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const profile::CpuScope scope{"RunObjectPick"};
    // ObjectPickKeyBuffer is persistent across clicks: high 8 bits of each packed key store
    // a per-click epoch tag. We therefore avoid clearing all keys every click and only do a
    // full reset when the 8-bit epoch wraps; stale keys are filtered out by epoch on readback.
    if (object_pick_epoch_tag == 0) {
        ResetObjectPickKeys(buffers);
        object_pick_epoch_tag = 255;
    }
    const uint32_t epoch_inv = object_pick_epoch_tag--;

    std::fill_n(buffers.ObjectPickSeenBitset.Data(), (max_object_id + 31) / 32, 0u);
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        RenderSelectionPass(
            r, chain, viewport,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .TargetPx = mouse_px,
                .RadiusSq = radius_px * radius_px,
                .EpochInv = epoch_inv,
                .BestKeySlot = sel_slots.ObjectPickKey,
                .SeenBitsSlot = sel_slots.ObjectPickSeenBits,
                .BoxResultSlot = InvalidSlot,
            }
        );
    }
    SubmitAndWait(command_buffer);
    r.ctx().get<DrawState>().SelectionStale = false;

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : r.view<RenderInstance>().each()) {
        if (ri.ObjectId > 0 && ri.ObjectId <= max_object_id) object_id_to_entity[ri.ObjectId] = e;
    }

    struct SortedHit {
        uint32_t DistSq;
        uint32_t Layer; // 0 = bone (on top in main pass), 1 = other
        uint32_t Depth;
        entt::entity Entity;
        auto operator<=>(const SortedHit &) const = default;
    };

    const auto *bits = buffers.ObjectPickSeenBitset.Data();
    const auto *keys = buffers.ObjectPickKeys.Data();
    std::vector<SortedHit> hits;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t idx = object_id - 1;
        if ((bits[idx / 32] & (1u << (idx % 32))) == 0) continue;
        const auto it = object_id_to_entity.find(object_id);
        if (it == object_id_to_entity.end()) continue;
        const uint32_t packed_key = keys[idx];
        if ((packed_key >> 24) == epoch_inv) {
            const uint32_t layer = r.any_of<BoneIndex, BoneSubPartOf>(it->second) ? 0u : 1u;
            hits.emplace_back(SortedHit{(packed_key >> 16) & 0xffu, layer, packed_key & 0xffffu, it->second});
        }
    }
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    for (const auto &hit : hits) entities.emplace_back(hit.Entity);
    return entities;
}

std::vector<entt::entity> RunBoxSelect(entt::registry &r, entt::entity viewport, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    auto &buffers = r.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);

    const profile::CpuScope scope{"RunBoxSelect"};
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    memset(buffers.SelectionBitset.Data(), 0, ((max_object_id + 31) / 32) * sizeof(uint32_t));
    auto *command_buffer = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        RenderSelectionPass(
            r, chain, viewport,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .BestKeySlot = InvalidSlot,
                .Box = {box_min.x, box_min.y, box_max.x, box_max.y},
                .BoxResultSlot = sel_slots.SelectionBitset,
            }
        );
    }
    SubmitAndWait(command_buffer);
    r.ctx().get<DrawState>().SelectionStale = false;

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : r.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;

    const auto *bits = buffers.SelectionBitset.Data();
    std::vector<entt::entity> entities;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t bit_index = object_id - 1;
        const uint32_t mask = 1u << (bit_index % 32);
        if ((bits[bit_index / 32] & mask) != 0) {
            if (auto it = object_id_to_entity.find(object_id); it != object_id_to_entity.end()) {
                entities.emplace_back(it->second);
            }
        }
    }
    return entities;
}

void DispatchUpdateSelectionStates(
    entt::registry &r,
    std::span<const ElementRange> ranges, Element element
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &meshes = r.ctx().get<MeshStore>();

    const auto &buffers = r.ctx().get<const GpuBuffers>();
    // The state and bitset buffers reach the kernel through the argument buffer, so they must be resident before this one-shot runs.
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    encode::BindCompute(encoder, pipelines.UpdateSelectionState, slots, buffers);

    for (const auto &range : ranges) {
        const auto &mesh = GetMesh(r, range.MeshEntity);
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
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size((range.Count + 255) / 256, 1, 1), ThreadgroupSize::Linear256);
    }

    encoder->endEncoding();
    command_buffer->commit();
    // The host reads the element states straight after this returns.
    command_buffer->waitUntilCompleted();
}

void ApplySelectionStateUpdate(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element
) {
    auto &meshes = r.ctx().get<MeshStore>();
    DispatchUpdateSelectionStates(r, ranges, element);
    if (element == Element::Vertex) {
        for (const auto &range : ranges) {
            const auto &mesh = GetMesh(r, range.MeshEntity);
            meshes.UpdateEdgeStatesFromVertices(mesh);
            meshes.UpdateFaceStatesFromVertices(mesh);
        }
    } else if (element == Element::Face || element == Element::Edge) {
        for (const auto &range : ranges) {
            const auto &mesh = GetMesh(r, range.MeshEntity);
            std::optional<uint32_t> active_handle;
            if (const auto *active = r.try_get<const MeshActiveElement>(range.MeshEntity); active && active->Handle < range.Count) {
                active_handle = active->Handle;
            }
            if (element == Element::Face) {
                meshes.UpdateEdgeStatesFromFaces(mesh, active_handle);
                meshes.UpdateVertexStatesFromFaces(mesh, active_handle);
            } else {
                meshes.UpdateFaceStatesFromEdges(mesh);
                meshes.UpdateVertexStatesFromEdges(mesh, active_handle);
            }
        }
    }
    r.emplace_or_replace<ElementStatesDirty>(viewport);
}
