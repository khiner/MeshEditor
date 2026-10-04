#include "metal/AutoreleaseScope.h"
#include "numeric/VectorMath.h"
#include "numeric/uvec2.h"
#include "numeric/uvec4.h"

#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "ProcessEvents.h"
#include "state/Scene.h"

#include <Metal/MTLCommandQueue.hpp>

#include "Profile.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gpu/EditSharpnessPushConstants.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRoute.h"
#include "gpu/ObjectSelectionPushConstants.h"
#include "gpu/OverlayDispatch.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/VisibilitySelectionPushConstants.h"
#include "mesh/MeshComponents.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/Dispatch.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "render/Encoding.h"
#include "render/GpuBufferOps.h"
#include "render/GpuSceneState.h"
#include "render/MeshTopologyRepair.h"
#include "render/SceneUpdates.h"
#include "render/Instance.h"
#include "render/PickConstants.h"
#include "scene/Entity.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <bit>
#include <cmath>
#include <memory>

using state::Change;

namespace {
// One mesh's change to its selection in the transaction's source domain.
struct SelectionRequest {
    state::Entity MeshEntity;
    EditSelectionOperation Operation;
    std::span<const uint32_t> List{}; // Source elements relative to the domain's first block, for FillList.
};

// Writes the source-domain words in place, then derives the other domains and the aggregates on the GPU.
// `picked` is a canonical handle, and `query` merges the pending box query.
void ApplySelection(state::Scene &, std::span<const SelectionRequest>, Element, uint32_t picked = InvalidOffset, bool query = false);
std::vector<SelectionRequest> SelectionRequests(std::span<const ElementRange> ranges, EditSelectionOperation operation) {
    std::vector<SelectionRequest> requests;
    requests.reserve(ranges.size());
    for (const auto &range : ranges) requests.push_back({range.MeshEntity, operation});
    return requests;
}

void SubmitAndWait(const mtl::Context &ctx, MTL::CommandBuffer *command_buffer) {
    const profile::CpuScope scope{"SelectionSubmit"};
    // Selection culling may allocate bindless buffers while encoding.
    {
        const profile::CpuScope residency_scope{"SelectionResidency"};
        ctx.CommitResidency();
    }
    mtl::Commit(command_buffer);
    {
        const profile::CpuScope wait_scope{"SelectionWait"};
        command_buffer->waitUntilCompleted();
    }
    if (command_buffer->status() == MTL::CommandBufferStatusError) {
        const auto *error = command_buffer->error();
        throw std::runtime_error(std::string{"GPU selection command failed: "} +
            (error ? error->localizedDescription()->utf8String() : "unknown Metal error"));
    }
}

// Record selection passes into one command buffer and wait for them.
void SubmitSelectionPasses(state::Scene &r, auto &&record) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // End the final pass before submission.
        struct RecordingScope {
            RecordingScope() { profile::BeginRecording(); }
            ~RecordingScope() { profile::EndRecording(); }
        } recording;
        mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
        record(chain);
    }
    SubmitAndWait(ctx, command_buffer);
    profile::Resolve(command_buffer);
}

struct ElementPickTarget {
    uvec2 Px;
    uint32_t RadiusSq;
};

std::optional<PixelRect> ClampedRect(uvec2 lo, uvec2 hi, mtl::Extent2D target) {
    const auto limit = std::bit_cast<uvec2>(target);
    lo = Min(lo, limit);
    hi = Min(hi, limit);
    if (hi.x <= lo.x || hi.y <= lo.y) return {};
    return PixelRect{lo, hi - lo};
}

// The pixels an inclusive box covers within the target, or empty when it covers none.
std::optional<PixelRect> BoxRect(uvec4 box, mtl::Extent2D target) {
    return ClampedRect({box.x, box.y}, {uint32_t(std::min<uint64_t>(uint64_t{box.z} + 1u, target.Width)), uint32_t(std::min<uint64_t>(uint64_t{box.w} + 1u, target.Height))}, target);
}

// The pixels within the pick radius of `px`, or empty when none lie in the target.
std::optional<PixelRect> RadiusRect(uvec2 px, uint32_t radius_sq, mtl::Extent2D target) {
    const uint32_t radius = uint32_t(std::ceil(std::sqrt(float(radius_sq))));
    return ClampedRect(
        {px.x > radius ? px.x - radius : 0u, px.y > radius ? px.y - radius : 0u},
        {uint32_t(std::min<uint64_t>(uint64_t{px.x} + radius + 1u, target.Width)), uint32_t(std::min<uint64_t>(uint64_t{px.y} + radius + 1u, target.Height))}, target
    );
}

uint32_t ElementPickRadiusSq(Element element) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    return radius * radius;
}

ElementSelectQuery MakeElementQuery(
    const SelectionSlots &sel_slots, uvec4 box, SelectionQueryRef results, const std::optional<ElementPickTarget> &pick, bool resolve_id
) {
    return {
        box,
        results,
        pick ? pick->Px : uvec2{},
        pick ? pick->RadiusSq : 0u,
        pick ? sel_slots.ElementPickKey : InvalidSlot,
        pick && resolve_id ? sel_slots.ElementPickId : InvalidSlot,
    };
}

// No fragment writes an all-ones key or id, so both double as the empty state.
constexpr uint32_t EmptyElementPick{~uint32_t{0}};

void ResetElementPick(GpuBuffers &buffers) {
    buffers.ElementPickKey.GetMutableSpan<uint32_t>()[0] = EmptyElementPick;
    buffers.ElementPickId.GetMutableSpan<uint32_t>()[0] = EmptyElementPick;
}

std::optional<uint32_t> ReadNearestPickedElement(const GpuBuffers &buffers, uint32_t max_element_id) {
    const uint32_t id = buffers.ElementPickId.GetSpan<uint32_t>()[0];
    if (id == EmptyElementPick || id == 0 || id > max_element_id) return {};
    return id - 1;
}

// Rasterize ids and depth for the query rectangle with every surface opaque.
// Selection mode adds lines and points so object queries can hit them.
// Visibility mode leaves them out so elements are occluded by surfaces only, never by wire geometry.
void RecordSelectionVisibility(
    state::Scene &r, mtl::PassChain &chain, const PixelRect &rect, MeshletRouteMode mode,
    bool exact_edit_geometry = false
) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    RecordMeshletCull(chain, slots, pipelines, buffers, {.Mode = mode, .ExactEditGeometry = exact_edit_geometry});
    RecordMeshletVisibilityPass(chain, slots, pipelines, r.Context.get<const RenderTargets>(), buffers, false, 0u, rect);
}

// A depth rectangle rasterizes selection depth for the query so the draws test against it, with exact edit geometry when `exact_depth`.
// Picks raster twice; boxes raster once.
void RunSelectionPass(
    state::Scene &r, mtl::PassChain &chain, std::optional<PixelRect> depth_rect, bool exact_depth,
    std::optional<MeshletCullConfig> meshlet_cull, bool pick, auto &&record_draws
) {
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    auto &buffers = r.Context.get<GpuBuffers>();

    if (depth_rect) RecordSelectionVisibility(r, chain, *depth_rect, MeshletRouteMode::Visibility, exact_depth);
    if (meshlet_cull && buffers.MeshletInstanceCount > 0) {
        RecordMeshletCull(chain, slots, pipelines, buffers, *meshlet_cull);
    }

    const auto extent = r.Context.get<const RenderTargets>().Resources->ScratchDepth.Extent;
    const uint32_t raster_passes = pick ? 2u : 1u;
    for (uint32_t index = 0; index < raster_passes; ++index) {
        // Scene depth remains valid for shading and later picks; depth-free queries need no scratch contents.
        const auto depth = depth_rect ? mtl::LoadDepth(*r.Context.get<const RenderTargets>().Resources->VisibilityDepth) :
                                        mtl::DepthAttachment{*r.Context.get<const RenderTargets>().Resources->ScratchDepth, MTL::LoadActionDontCare, MTL::StoreActionDontCare};
        const auto pass = mtl::MakePassDescriptor({}, depth);
        pass->setRenderTargetWidth(extent.Width);
        pass->setRenderTargetHeight(extent.Height);
        // The pick resolve reads the key an earlier raster wrote, and bindless buffers carry no tracked hazard.
        auto *encoder = encode::BeginScenePass(chain, pass.get(), "SelectionPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageBlit | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers);
        record_draws(encoder, extent, index == 1u);
    }
}

void RenderElementSelectionPass(
    state::Scene &r, mtl::PassChain &chain, state::Entity viewport,
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max, std::optional<ElementPickTarget> pick
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &pipelines = GetPipelines(r);
    const auto &sel_slots = r.Context.get<const SelectionSlots>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();

    const bool xray_selection = XRayFlag(r.get<const ViewportDisplay>(viewport));
    const auto &selection = pipelines.SelectionFragment;
    const bool degenerate_point_pass = write_bitset && xray_selection && element != Element::Vertex;
    for (const auto &range : ranges) {
        [[maybe_unused]] const auto &mesh_buffers = RecordOf(r, range.MeshEntity);
        assert(meshes.MeshletCount(mesh_buffers) > 0u && "selectable mesh geometry must have persistent meshlets");
    }

    const auto target = r.Context.get<const RenderTargets>().Resources->ScratchDepth.Extent;
    const auto query_rect = write_bitset ? BoxRect({box_min.x, box_min.y, box_max.x, box_max.y}, target) : RadiusRect(pick->Px, pick->RadiusSq, target);
    if (!query_rect) return;
    // Vertices draw from the canonical vertex blocks of each mesh's primary edit instance.
    const auto &primaries = r.get<const EditPrimaries>(viewport).All;
    RunSelectionPass(
        r, chain, xray_selection ? std::nullopt : query_rect, true,
        element == Element::Vertex ? std::nullopt : std::optional{MeshletCullConfig{
            .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::ElementSelection),
            .RouteMask = 1u << uint32_t(MeshletRoute::OpaqueCullBack),
            .ExactEditGeometry = true,
        }},
        pick.has_value(),
        [&](auto *encoder, mtl::Extent2D, bool resolve_id) {
            const SelectionElementPushConstants element_pc{MakeElementQuery(sel_slots, {box_min.x, box_min.y, box_max.x, box_max.y}, write_bitset ? meshes.Arenas().Query.Ref() : SelectionQueryRef{}, pick, resolve_id)};
            if (write_bitset) encoder->setScissorRect({query_rect->Origin.x, query_rect->Origin.y, query_rect->Extent.x, query_rect->Extent.y});
            // Edges draw once per triangle corner.
            const auto draw_edges = [&] {
                for (uint32_t corner = 0u; corner < 3u; ++corner) {
                    DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 160u, corner);
                }
            };
            const auto &pipeline = selection.ElementRaster(element, write_bitset, xray_selection);
            pipeline.Bind(encoder);
            encoder->setFragmentBytes(&element_pc, sizeof(element_pc), BufferIndex_PushConstants);
            if (element == Element::Edge) draw_edges();
            else if (element == Element::Face) DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 160u);
            else for (const auto &range : ranges) {
                const auto primary = primaries.find(range.MeshEntity);
                if (primary != primaries.end()) DrawVertexBlocks(encoder, r, range.MeshEntity, r.get<const RenderInstance>(primary->second).BufferIndex);
            }
            if (degenerate_point_pass) {
                const auto &point_pipeline = element == Element::Face ?
                    selection.MeshletFaceXRayPointsBitsetBox :
                    selection.MeshletEdgeXRayPointsBitsetBox;
                point_pipeline.Bind(encoder);
                encoder->setFragmentBytes(&element_pc, sizeof(element_pc), BufferIndex_PushConstants);
                if (element == Element::Face) DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 64u);
                else draw_edges();
            }
        }
    );
}

} // namespace

std::optional<std::pair<state::Entity, uint32_t>> RunEditElementClick(
    state::Scene &r, state::Entity viewport,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px, bool toggle
) {
    if (ranges.empty() || element == Element::None) return {};

    const profile::CpuScope scope{"RunElementPick"};
    auto &buffers = r.Context.get<GpuBuffers>();
    ResetElementPick(buffers);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderElementSelectionPass(r, chain, viewport, ranges, element, false, {}, {}, ElementPickTarget{mouse_px, ElementPickRadiusSq(element)});
    });
    const auto picked = ReadNearestPickedElement(buffers, UINT32_MAX);
    const auto requests = SelectionRequests(ranges, toggle ? EditSelectionOperation::PickToggle : EditSelectionOperation::PickReplace);
    ApplySelection(r, requests, element, picked.value_or(InvalidOffset));
    if (!picked) return {};
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto &range : ranges) {
        const auto id = GetMesh(r, range.MeshEntity).GetStoreId();
        if (meshes.IsLiveElement(id, element, *picked)) return std::pair{range.MeshEntity, *picked - meshes.GetSelectionBitOffset(id, element)};
    }
    return {};
}

// The pixels a query covers, or empty when the box or radius picks nothing.
std::optional<PixelRect> ObjectQueryRect(const ObjectSelectQuery &query, mtl::Extent2D target) {
    if (query.BoxResultSlot != InvalidSlot) return BoxRect(query.Box, target);
    if (query.BestKeySlot != InvalidSlot) return RadiusRect(query.TargetPx, query.RadiusSq, target);
    return {};
}

void RecordVisibilityObjectSelection(
    state::Scene &r, mtl::PassChain &chain, const ObjectSelectQuery &query
) {
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    auto &buffers = r.Context.get<GpuBuffers>();

    const auto rect = ObjectQueryRect(query, r.Context.get<const RenderTargets>().Resources->VisibilityImage.Extent);
    if (!rect) return;
    RecordSelectionVisibility(r, chain, *rect, MeshletRouteMode::Selection);

    auto *encoder = chain.BeginCompute("VisibilityObjectSelection", MTL::StageFragment);
    encode::BindCompute(encoder, pipelines.VisibilityObjectSelection, slots, buffers);
    encoder->setTexture(*r.Context.get<const RenderTargets>().Resources->VisibilityImage, 0u);
    encoder->setTexture(*r.Context.get<const RenderTargets>().Resources->VisibilityDepth, 1u);
    encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{encode::VisibilityDecodePc(buffers), query, rect->Origin, rect->Extent});
    encoder->dispatchThreadgroups(
        MTL::Size((rect->Extent.x + 15u) / 16u, (rect->Extent.y + 15u) / 16u, 1u),
        ThreadgroupSize::Tile16
    );
}

// `through` rasters every covered surface for an object box instead of the visible surface alone.
// A sound instance picks among its mesh's selected vertices.
void RenderSelectionPickPass(state::Scene &r, mtl::PassChain &chain, std::optional<ObjectSelectQuery> object, bool through, state::Entity sound_instance = state::Null, std::optional<ElementPickTarget> pick = {}) {
    const auto &sel_slots = r.Context.get<const SelectionSlots>();
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &pipelines = GetPipelines(r);
    const auto &selection = pipelines.SelectionFragment;
    const bool raster_all = object && (object->BestKeySlot != InvalidSlot || through);
    if (object) {
        if (raster_all) {
            // Click cycling and X-ray boxes need every covered surface, including occluded objects.
            RecordMeshletCull(chain, r.Context.get<const mtl::BindlessSet>(), pipelines, buffers, {.Mode = MeshletRouteMode::Selection});
        } else RecordVisibilityObjectSelection(r, chain, *object);
        RecordOverlayJobCull(chain, r.Context.get<const mtl::BindlessSet>(), pipelines, buffers, true);
    }
    const bool sound = sound_instance != state::Null;
    const auto sound_rect = sound && pick ? RadiusRect(pick->Px, pick->RadiusSq, r.Context.get<const RenderTargets>().Resources->ScratchDepth.Extent) : std::nullopt;
    if (sound && !sound_rect) return;
    RunSelectionPass(r, chain, sound_rect, false, std::nullopt, pick.has_value(), [&](auto *encoder, mtl::Extent2D, bool resolve_id) {
        if (sound) {
            const SelectionElementPushConstants point_pc{MakeElementQuery(sel_slots, {}, {}, pick, resolve_id)};
            selection.ElementRaster(Element::Vertex, false, false).Bind(encoder);
            encoder->setFragmentBytes(&point_pc, sizeof(point_pc), BufferIndex_PushConstants);
            DrawVertexBlocks(encoder, r, r.get<const Instance>(sound_instance).Entity, r.get<const RenderInstance>(sound_instance).BufferIndex, true);
        }
        if (object) {
            const auto rect = ObjectQueryRect(*object, r.Context.get<const RenderTargets>().Resources->ScratchDepth.Extent);
            if (!rect) return;
            encoder->setScissorRect({rect->Origin.x, rect->Origin.y, rect->Extent.x, rect->Extent.y});
            if (raster_all) {
                selection.ObjectPick.Bind(encoder);
                encoder->setCullMode(MTL::CullModeNone); // Shared coverage handles sidedness and mirrored transforms.
                const VisibilitySelectionPushConstants pc{encode::MeshletDecodePc(buffers), *object};
                encoder->setFragmentBytes(&pc, sizeof(pc), BufferIndex_PushConstants);
                for (const auto route : {MeshletRoute::OpaqueCullBack, MeshletRoute::OpaqueCullFront, MeshletRoute::OpaqueDoubleSided, MeshletRoute::Coverage, MeshletRoute::Blend}) {
                    DrawMeshlets(encoder, buffers, uint32_t(route));
                }
            }
            const ObjectSelectionPushConstants sel_pc{*object};
            if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::Bone)).Meshlets > 0u) {
                selection.BoneSolid.Bind(encoder);
                encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
                DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Overlay), uint32_t(MeshletInstanceFlag::Bone), 24u);
            }
            if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::BoneJoint)).Meshlets > 0u) {
                selection.BoneSphere.Bind(encoder);
                encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
                DrawMeshlets(
                    encoder, buffers, uint32_t(MeshletRoute::Overlay),
                    uint32_t(MeshletInstanceFlag::BoneJoint), uint32_t(OverlayDispatch::BoneSphereVertices)
                );
            }
            selection.OverlayJobLines.Bind(encoder);
            encoder->setFragmentBytes(&sel_pc, sizeof(sel_pc), BufferIndex_PushConstants);
            DrawOverlayJobs(encoder, buffers, r.Context.get<const MeshStore>());
        }
    });
}

void RunBoxSelectElements(state::Scene &r, state::Entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    const profile::CpuScope scope{"RunBoxSelectElements"};

    // A query abandoned by a rejected transaction still holds its words.
    const auto &query = r.Context.get<const MeshStore>().Arenas().Query;
    if (const auto count = query.WordCount()) {
        auto masks = query.Masks.GetMutableSpan<uint32_t>();
        for (const auto word : query.Words.GetSpan<uint32_t>({0, count})) masks[word] = 0u;
        query.Count.GetMutableSpan<uint32_t>()[0] = 0u;
    }

    auto *baseline = is_additive ? r.try_edit<AdditiveBoxSelectBaseline>(viewport) : nullptr;
    const auto operation = !is_additive                 ? EditSelectionOperation::Clear :
        baseline && !baseline->ElementSelectionCaptured ? EditSelectionOperation::CaptureBaseline :
                                                          EditSelectionOperation::RestoreBaseline;
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderElementSelectionPass(r, chain, viewport, ranges, element, true, box_min, box_max, {});
    });
    ApplySelection(r, SelectionRequests(ranges, operation), element, InvalidOffset, true);
    if (baseline) baseline->ElementSelectionCaptured = true;
}

std::optional<uint32_t> RunSoundVerticesVertexPick(state::Scene &r, state::Entity instance_entity, uvec2 mouse_px) {
    if (!r.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = r.try_get<Instance>(instance_entity);
    if (!instance) return {};
    auto &buffers = r.Context.get<GpuBuffers>();

    const profile::CpuScope scope{"RunSoundVerticesVertexPick"};
    const auto &mesh = GetMesh(r, instance->Entity);
    if (mesh.VertexCount() == 0) return {};

    ResetElementPick(buffers);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderSelectionPickPass(r, chain, std::nullopt, false, instance_entity, ElementPickTarget{mouse_px, ElementPickRadiusSq(Element::Vertex)});
    });
    // Pick ids are handles relative to the mesh's first vertex, which on a sparse set reach past the vertex count.
    const auto &meshes = r.Context.get<const MeshStore>();
    return ReadNearestPickedElement(buffers, meshes.Arenas().Vertices.Capacity() - meshes.GetSelectionBitOffset(mesh.GetStoreId(), Element::Vertex));
}

namespace {
void ReserveObjectPicking(state::Scene &r, uint32_t count) {
    auto &buffers = r.Context.get<GpuBuffers>();
    if (count <= buffers.ObjectPickKeys.Count<uint32_t>()) return;
    buffers.ObjectPickKeys.SetCount<uint32_t>(count);
    buffers.ObjectPickSeenBitset.SetCount<uint32_t>((count + 31) / 32);
    buffers.ObjectBoxBitset.SetCount<uint32_t>((count + 31) / 32);
    buffers.ObjectPickEpochTag = 0;
    auto &slots = r.Context.get<mtl::BindlessSet>();
    const auto &selection = r.Context.get<const SelectionSlots>();
    slots.SetBuffer({SlotType::Buffer, selection.ObjectPickKey}, *buffers.ObjectPickKeys);
    slots.SetBuffer({SlotType::Buffer, selection.ObjectPickSeenBits}, *buffers.ObjectPickSeenBitset);
    slots.SetBuffer({SlotType::Buffer, selection.ObjectBoxBitset}, *buffers.ObjectBoxBitset);
}

// Sizes the object query buffers to the live entities and returns the highest object id a query may report, or zero without entities.
uint32_t PrepareObjectQuery(state::Scene &r) {
    const uint32_t next_object_id = r.EntityCapacity() + 1;
    if (next_object_id <= 1) return 0;
    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);
    ReserveObjectPicking(r, max_object_id);
    return max_object_id;
}

// Calls `fn(index, entity)` for each rendered entity whose bit is set, in object-id order.
void ForEachHitObject(const state::Scene &r, std::span<const uint32_t> bits, uint32_t max_object_id, auto &&fn) {
    for (uint32_t word = 0; word < (max_object_id + 31) / 32; ++word) {
        for (auto set = bits[word]; set; set &= set - 1u) {
            const uint32_t index = word * 32 + std::countr_zero(set);
            if (index >= max_object_id) break;
            const auto entity = r.EntityAt(index);
            if (r.all_of<RenderInstance>(entity)) fn(index, entity);
        }
    }
}
} // namespace

std::vector<state::Entity> RunObjectPick(state::Scene &r, uvec2 mouse_px, uint32_t radius_px) {
    const uint32_t max_object_id = PrepareObjectQuery(r);
    if (max_object_id == 0) return {};
    const auto &sel_slots = r.Context.get<const SelectionSlots>();
    auto &buffers = r.Context.get<GpuBuffers>();
    const profile::CpuScope scope{"RunObjectPick"};
    // The high byte rejects stale keys; clear on first use and whenever the 8-bit epoch wraps.
    if (buffers.ObjectPickEpochTag == 0) {
        std::ranges::fill(buffers.ObjectPickKeys.GetMutableSpan<uint32_t>(), std::numeric_limits<uint32_t>::max());
        buffers.ObjectPickEpochTag = 255;
    }
    const uint32_t epoch_inv = buffers.ObjectPickEpochTag--;

    std::ranges::fill(buffers.ObjectPickSeenBitset.GetMutableSpan<uint32_t>({0, (max_object_id + 31) / 32}), 0u);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderSelectionPickPass(
            r, chain,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .TargetPx = mouse_px,
                .RadiusSq = radius_px * radius_px,
                .EpochInv = epoch_inv,
                .BestKeySlot = sel_slots.ObjectPickKey,
                .SeenBitsSlot = sel_slots.ObjectPickSeenBits,
                .BoxResultSlot = InvalidSlot,
            },
            false
        );
    });
    struct SortedHit {
        uint32_t DistSq;
        uint32_t Layer;
        uint32_t Depth;
        state::Entity Entity;
        auto operator<=>(const SortedHit &) const = default;
    };

    const auto keys = buffers.ObjectPickKeys.GetSpan<uint32_t>();
    std::vector<SortedHit> hits;
    ForEachHitObject(r, buffers.ObjectPickSeenBitset.GetSpan<uint32_t>(), max_object_id, [&](uint32_t index, state::Entity entity) {
        const uint32_t packed_key = keys[index];
        if ((packed_key >> 24) != epoch_inv) return;
        const uint32_t layer = r.any_of<BoneIndex, BoneSubPartOf>(entity) ? 0u : 1u;
        hits.emplace_back(SortedHit{(packed_key >> 16) & 0xffu, layer, packed_key & 0xffffu, entity});
    });
    std::ranges::sort(hits);

    std::vector<state::Entity> entities;
    entities.reserve(hits.size());
    for (const auto &hit : hits) entities.emplace_back(hit.Entity);
    return entities;
}

std::vector<state::Entity> RunBoxSelect(state::Scene &r, state::Entity viewport, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    const uint32_t max_object_id = PrepareObjectQuery(r);
    if (max_object_id == 0) return {};
    auto &buffers = r.Context.get<GpuBuffers>();
    const profile::CpuScope scope{"RunBoxSelect"};
    const auto &sel_slots = r.Context.get<const SelectionSlots>();
    std::ranges::fill(buffers.ObjectBoxBitset.GetMutableSpan<uint32_t>({0, (max_object_id + 31) / 32}), 0u);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderSelectionPickPass(
            r, chain,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .BestKeySlot = InvalidSlot,
                .Box = {box_min.x, box_min.y, box_max.x, box_max.y},
                .BoxResultSlot = sel_slots.ObjectBoxBitset,
            },
            XRayActive(r.get<const ViewportDisplay>(viewport))
        );
    });
    std::vector<state::Entity> entities;
    ForEachHitObject(r, buffers.ObjectBoxBitset.GetSpan<uint32_t>(), max_object_id, [&](uint32_t, state::Entity entity) { entities.emplace_back(entity); });
    return entities;
}

namespace {
constexpr std::array SelectionElements{Element::Vertex, Element::Edge, Element::Face};
constexpr std::array SelectionDomains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};

void ApplySelection(state::Scene &r, std::span<const SelectionRequest> requests, Element element, uint32_t picked, bool query) {
    if (requests.empty() || element == Element::None) return;
    const profile::CpuScope scope{"ApplySelection"};
    auto &meshes = r.Context.get<MeshStore>();
    const auto &arenas = meshes.Arenas();
    const auto source = uint32_t(std::ranges::find(SelectionElements, element) - SelectionElements.begin());
    const std::array memberships{arenas.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        arenas.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(), arenas.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>()};
    // The query holds one canonical word per first hit, across every queried mesh.
    std::vector<std::pair<uint32_t, uint32_t>> query_words;
    if (query) {
        const auto &pending = arenas.Query;
        const auto count = pending.WordCount();
        auto masks = pending.Masks.GetMutableSpan<uint32_t>();
        for (const auto word : pending.Words.GetSpan<uint32_t>({0, count})) {
            if (masks[word]) query_words.emplace_back(word, masks[word]);
            masks[word] = 0u;
        }
        if (count) pending.Count.GetMutableSpan<uint32_t>()[0] = 0u;
    }
    std::vector<MeshStore::SelectionUpdate> updates;
    std::vector<uint32_t> ids;
    for (const auto &request : requests) ids.push_back(GetMesh(r, request.MeshEntity).GetStoreId());
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.EnsureSelectionState(r, chain, ids);
    for (uint32_t i = 0u; i < requests.size(); ++i) {
        const auto &request = requests[i];
        const auto id = ids[i];
        const auto operation = request.Operation;
        const auto origin = meshes.GetSelectionBitOffset(id, element);
        const auto &record = meshes.Get(id);
        const std::array owners{record.Vertices, record.EdgeData, record.FaceData};
        const auto summary = meshes.GetSelectionSummary(id);
        const bool rederive = operation == EditSelectionOperation::Derive || operation == EditSelectionOperation::ClearActive || summary.Mode != element;
        const bool replace = operation == EditSelectionOperation::Clear || operation == EditSelectionOperation::FillList ||
            operation == EditSelectionOperation::PickReplace || operation == EditSelectionOperation::RestoreBaseline;
        const auto picked_local = picked != InvalidOffset && meshes.IsLiveElement(id, element, picked) ? picked - origin : InvalidOffset;
        const auto bits = meshes.GetSelectedElements(id, element).Bits;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id, .Source = element});
        // Replacing or rederiving touches every block that holds a selected element in any domain.
        std::array<std::vector<uint32_t>, 3> selected;
        if (replace || rederive || operation == EditSelectionOperation::CaptureBaseline) {
            for (uint32_t d = 0u; d < 3u; ++d) {
                const auto words = meshes.GetSelectedElements(id, SelectionElements[d]).Bits;
                for (const auto block : meshes.GetBlockList(id, SelectionDomains[d]).Blocks)
                    if (std::ranges::any_of(words.subspan(block * MeshElementBlockWords, MeshElementBlockWords), [](uint32_t word) { return word != 0u; })) {
                        selected[d].push_back(block);
                    }
            }
        }
        if (operation == EditSelectionOperation::CaptureBaseline) {
            std::vector<std::pair<uint32_t, MeshArenas::SelectionBlock>> baseline;
            for (const auto block : selected[source]) {
                auto &entry = baseline.emplace_back(block, MeshArenas::SelectionBlock{});
                std::ranges::copy(bits.subspan(block * MeshElementBlockWords, MeshElementBlockWords), entry.second.begin());
            }
            meshes.SetSelectionBaseline(id, element, std::move(baseline), summary.ActiveHandle);
        }
        if (replace || rederive) update.Blocks = selected;
        // Words gain these bits after replacement clears the old source blocks.
        std::vector<std::pair<uint32_t, uint32_t>> additions;
        const auto add = [&](uint32_t handle) { additions.emplace_back(handle / 32u, 1u << (handle % 32u)); };
        for (const auto local : request.List)
            if (meshes.IsLiveElement(id, element, origin + local)) add(origin + local);
        if (operation == EditSelectionOperation::PickReplace && picked_local != InvalidOffset) add(picked);
        if (operation == EditSelectionOperation::RestoreBaseline) {
            for (const auto &[block, words] : meshes.GetDerived(id).SelectionBaseline)
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                    if (words[w]) additions.emplace_back(block * MeshElementBlockWords + w, words[w]);
        }
        for (const auto &[word, value] : query_words)
            if (owners[source] && memberships[source][word / MeshElementBlockWords].Owner == owners[source].Index) additions.emplace_back(word, value);
        const bool toggle = operation == EditSelectionOperation::PickToggle && picked_local != InvalidOffset;
        if (toggle) additions.emplace_back(picked / 32u, 0u);
        std::ranges::sort(additions);
        std::vector<uint32_t> blocks = replace ? selected[source] : std::vector<uint32_t>{};
        for (const auto &[word, value] : additions) blocks.push_back(word / MeshElementBlockWords);
        std::ranges::sort(blocks);
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        if (operation == EditSelectionOperation::Fill) {
            for (uint32_t d = 0u; d < 3u; ++d) {
                const auto all = meshes.GetBlockList(id, SelectionDomains[d]).Blocks;
                update.Blocks[d].assign(all.begin(), all.end());
            }
            blocks = update.Blocks[source];
            meshes.EditSelectionBlocks(element, blocks, [&](uint32_t block, auto &words) {
                std::ranges::copy(memberships[source][block].Live, words.begin());
            });
        } else {
            // Replacement dirties the old selection's blocks, so only added bits seed their neighbors.
            auto next = additions.begin();
            meshes.EditSelectionBlocks(element, blocks, [&](uint32_t block, auto &words) {
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                    const auto word = block * MeshElementBlockWords + w, before = words[w];
                    auto after = replace ? 0u : before;
                    for (; next != additions.end() && next->first == word; ++next) after |= next->second;
                    if (toggle && word == picked / 32u) {
                        const auto bit = 1u << (picked % 32u);
                        after = summary.ActiveHandle == picked_local ? after & ~bit : after | bit;
                    }
                    words[w] = after;
                    if (const auto seed = replace ? after & ~before : after ^ before) update.Seeds.push_back({source, word, seed});
                }
            });
        }
        if (rederive && operation != EditSelectionOperation::Fill) {
            for (const auto block : selected[source])
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w)
                    if (const auto word = bits[block * MeshElementBlockWords + w]) update.Seeds.push_back({source, block * MeshElementBlockWords + w, word});
        }
        auto &written = meshes.WriteSelectionSummary(id);
        written.Mode = element;
        if (operation == EditSelectionOperation::Clear || operation == EditSelectionOperation::FillList ||
            operation == EditSelectionOperation::ClearActive || operation == EditSelectionOperation::PickReplace) written.ActiveHandle = picked_local;
        else if (operation == EditSelectionOperation::RestoreBaseline) written.ActiveHandle = meshes.GetDerived(id).SelectionBaselineActive;
        else if (toggle) written.ActiveHandle = summary.ActiveHandle == picked_local ? InvalidOffset : picked_local;
        if (std::ranges::all_of(update.Blocks, [](const auto &domain) { return domain.empty(); }) && update.Seeds.empty()) updates.pop_back();
    }
    meshes.UpdateSelection(r, chain, updates);
    chain.Submit();
    for (const auto id : ids) meshes.PublishSelectionSummary(id);
    r.Context.get<GpuSceneState>().EditSelectionDirty = true;
}
} // namespace

void ApplyEditSelectionCommand(
    state::Scene &r, std::span<const ElementRange> ranges,
    Element element, EditSelectionOperation operation
) {
    ApplySelection(r, SelectionRequests(ranges, operation), element);
}

void ApplyEditSelectionLists(
    state::Scene &r,
    std::span<const std::pair<state::Entity, std::span<const uint32_t>>> lists, Element element
) {
    std::vector<SelectionRequest> requests;
    requests.reserve(lists.size());
    for (const auto &[mesh_entity, list] : lists)
        if (HasMesh(r, mesh_entity)) requests.push_back({mesh_entity, EditSelectionOperation::FillList, list});
    ApplySelection(r, requests, element);
}

void RefreshElementSelectionSummaries(state::Scene &r, std::span<const state::Entity> entities, std::optional<Element> mode) {
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto entity : entities) {
        const auto id = GetMesh(r, entity).GetStoreId();
        if (!meshes.Get(id).SelectionSummary.Count) continue;
        if (mode) {
            auto &summary = meshes.WriteSelectionSummary(id);
            summary.Mode = *mode;
            summary.ActiveHandle = InvalidOffset;
        }
        meshes.PublishSelectionSummary(id);
    }
}

void ConvertElementSelections(state::Scene &r, std::span<const state::Entity> mesh_entities, Element mode) {
    if (mesh_entities.empty()) return;
    auto &meshes = r.Context.get<MeshStore>();
    std::vector<uint32_t> ids;
    ids.reserve(mesh_entities.size());
    for (const auto mesh_entity : mesh_entities) ids.push_back(GetMesh(r, mesh_entity).GetStoreId());
    {
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.EnsureSelectionState(r, chain, ids);
    }
    std::vector<ElementRange> ranges;
    std::vector<state::Entity> all_selected;
    for (uint32_t i = 0u; i < mesh_entities.size(); ++i) {
        const auto mesh_entity = mesh_entities[i];
        const auto id = ids[i];
        r.remove<MeshActiveElement>(mesh_entity);
        if (mode == Element::None) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        // A mesh with every element selected keeps its bits in every domain.
        if (std::ranges::all_of(SelectionElements, [&](Element element) { return meshes.GetSelectedElements(id, element).Count() == mesh.ElementCount(element); })) all_selected.push_back(mesh_entity);
        else if (const auto count = mesh.ElementCount(mode); count > 0) ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(id, mode), count);
    }
    if (!ranges.empty()) ApplyEditSelectionCommand(r, ranges, mode, EditSelectionOperation::ClearActive);
    if (!all_selected.empty()) RefreshElementSelectionSummaries(r, all_selected, mode);
}

void ApplyEditSharpness(
    state::Scene &r, state::Entity, std::span<const state::Entity> mesh_entities,
    EditSharpnessOperation operation, bool value, float angle
) {
    if (mesh_entities.empty()) return;
    auto &meshes = r.Context.get<MeshStore>();
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    const auto source = operation == EditSharpnessOperation::SetSelectedFaces ? Element::Face :
        operation == EditSharpnessOperation::SetSelectedEdges ? Element::Edge :
        operation == EditSharpnessOperation::SetVertexEdges ? Element::Vertex : Element::None;
    std::vector<state::Entity> faced;
    std::vector<uint32_t> faced_ids;
    for (const auto mesh_entity : mesh_entities) {
        const auto *owner = HasMesh(r, mesh_entity) ? TryRecordOf(r, mesh_entity) : nullptr;
        if (!owner || owner->RenderTopology == InvalidOffset) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        if (mesh.FaceCount() == 0) continue;
        faced.push_back(mesh_entity);
        faced_ids.push_back(mesh.GetStoreId());
    }
    // Every mesh's selected handles, membership work and closures share one chain's scratch, and each phase submits once for every mesh.
    mtl::ComputeChain chain{meshes.BufferContext()};
    if (source != Element::None) meshes.EnsureSelectionState(r, chain, faced_ids);
    std::vector<EditSharpnessPushConstants> commands;
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<state::Entity> edited;
    const auto &arenas = meshes.Arenas();
    // Each mesh gathers its selected handles into its own range.
    chain.Concurrent([&] {
        for (uint32_t i = 0u; i < faced.size(); ++i) {
            const auto id = faced_ids[i];
            if (source != Element::None && !meshes.GetSelectedElements(id,source).Count()) continue;
            meshes.CaptureSharpnessWrite(id, operation);
            const auto mesh = GetMesh(r, faced[i]);
            const auto &record = meshes.Get(id);
            auto &pc = commands.emplace_back(EditSharpnessPushConstants{
                .VertexSelectionSlot = arenas.VertexSelection.Buffer.Slot,
                .CornersSlot = arenas.FaceCorners.Buffer.Slot,
                .FaceSharpnessSlot = arenas.FaceSharpness.Buffer.Slot,
                .EdgeSharpnessSlot = arenas.EdgeSharpness.Buffer.Slot,
                .FaceNormalsSlot = arenas.BaseFaceNormals.Buffer.Slot,
                .Connectivity = meshes.GetConnectivityRef(id),
                .EdgeCount = mesh.EdgeCount(),
                .FaceCount = mesh.FaceCount(),
                .Operation = operation,
                .Value = value ? 1u : 0u,
                .CosAngle = std::cos(angle),
            });
            if (source != Element::None) {
                const auto selected = meshes.GatherSelectedElements(r, chain, id, source, chain.Scratch);
                pc.Selected = {chain.Scratch.Buffer.Slot, selected.Offset};
                pc.SelectedCount = selected.Count;
            } else {
                pc.FaceWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch,arenas.FaceTriangles,record.FaceData)).Work;
                if (operation != EditSharpnessOperation::SetAllFaces) pc.EdgeWork = seeds.emplace_back(PrepareElementMembershipWork(chain.Scratch,arenas.EdgeHalfedges,record.EdgeData)).Work;
            }
            edited.push_back(faced[i]);
        }
    });
    if (commands.empty()) return;
    std::vector<ElementWork> seeded;
    for (const auto &seed : seeds) seeded.push_back(seed.Work);
    EncodeElementMembershipWork(r,chain,seeds);
    EncodeSortElementWork(r,chain,seeded);
    chain.Submit();
    for (const auto work : seeded) CheckElementWork(chain.Scratch,work);
    // The closures record after the sharpness writes, and the next submit commits both.
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
        for (const auto &pc : commands) {
            const uint32_t count = source != Element::None ? pc.SelectedCount :
                operation == EditSharpnessOperation::SetAllFaces ? pc.FaceCount : std::max(pc.EdgeCount, pc.FaceCount);
            if (!count) continue;
            encode::BindCompute(encoder, pipelines.EditSharpness, slots, buffers);
            encode::SetPushConstants(encoder, pc);
            encoder->dispatchThreadgroups(MTL::Size((count + 255u) / 256u, 1, 1), ThreadgroupSize::Linear256);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    });
    r.Context.get<GpuSceneState>().EditSelectionDirty = true;
    // One edited mesh's closures, changed triangles and corner classes.
    struct Edit {
        state::Entity Entity;
        uint32_t Id;
        ClosureSeed Vertices;
        MeshClosure Incident{}, Neighborhood{};
        FaceTriangles Triangles{};
        MeshStore::CornerClassUpdate Classes{};
    };
    std::vector<Edit> edits;
    for (const auto mesh_entity : edited) {
        const auto id = GetMesh(r,mesh_entity).GetStoreId();
        // The written elements' vertices: the selected faces' loop vertices, the selected vertices or edge endpoints, or all vertices.
        ClosureSeed vertices;
        if (source==Element::Face) vertices=EncodeFaceClosure(r,chain,id,EncodeSelectionSeed(r,chain,id,Element::Face,false)).Seed(Element::Vertex);
        else if (source==Element::Edge) vertices=EncodeEdgeVertices(r,chain,id,EncodeSelectionSeed(r,chain,id,Element::Edge,false));
        else vertices=EncodeSelectionSeed(r,chain,id,Element::Vertex,source==Element::None);
        if (vertices.Count) edits.push_back({.Entity=mesh_entity,.Id=id,.Vertices=std::move(vertices)});
    }
    if (operation==EditSharpnessOperation::SetVertexEdges) {
        // Each edge at a selected vertex changes the fans at both of its endpoints.
        for (auto &edit : edits) {
            edit.Incident=EncodeVertexClosure(r,chain,edit.Id,edit.Vertices);
            edit.Incident.EncodeIncidence(r,chain,edit.Id,Element::Edge);
        }
        chain.Submit();
        for (auto &edit : edits) {
            edit.Incident.Finish(chain);
            edit.Vertices=EncodeEdgeVertices(r,chain,edit.Id,edit.Incident.Seed(Element::Edge));
        }
    }
    for (auto &edit : edits) {
        edit.Neighborhood=EncodeVertexClosure(r,chain,edit.Id,edit.Vertices);
        edit.Neighborhood.EncodeIncidence(r,chain,edit.Id,Element::Face);
    }
    chain.Submit();
    for (auto &edit : edits) edit.Neighborhood.Finish(chain);
    std::erase_if(edits,[](const Edit &edit) { return !edit.Neighborhood.Counts[0]; });
    for (auto &edit : edits) {
        edit.Triangles=EncodeFaceTriangles(r,chain,edit.Id,edit.Neighborhood.Seed(Element::Face));
        // The neighborhood's corners include every corner at its vertices.
        edit.Classes=meshes.EncodeCornerClassification(r,chain,edit.Id,edit.Neighborhood.Elements[0],edit.Neighborhood.Counts[0],
            edit.Neighborhood.Counts[1],source==Element::None);
    }
    chain.Submit();
    std::vector<LocalNormalWork> normals;
    for (auto &edit : edits) {
        edit.Triangles.Finish(chain);
        meshes.PlanCornerClassification(r,chain,edit.Classes);
        const auto &neighborhood=edit.Neighborhood;
        normals.push_back({edit.Id,neighborhood.Elements[0],neighborhood.Counts[0],neighborhood.Elements[2],neighborhood.Counts[2]});
    }
    EncodeDeriveMeshNormals(r,chain,chain.Scratch,normals);
    chain.Submit();
    std::vector<MeshStore::SelectionUpdate> updates;
    std::vector<std::pair<state::Entity,FaceTriangles>> repairs;
    for (const auto &edit : edits) {
        meshes.FinishCornerClassification(chain,edit.Classes);
        // The neighborhood holds every written element and each vertex whose incident edges changed.
        if (meshes.Get(edit.Id).SelectionSummary.Count) {
            auto &update=updates.emplace_back(MeshStore::SelectionUpdate{.StoreId=edit.Id});
            for (const auto [d,domain]:{std::pair{0u,0u},std::pair{1u,3u},std::pair{2u,2u}})
                ForEachWorkBlock(chain.Scratch,edit.Neighborhood.Elements[domain],[&](uint32_t block,auto) { update.Blocks[d].push_back(block); });
        }
        const auto *owner=TryRecordOf(r,edit.Entity);
        if (owner && owner->PrimitiveRoot!=InvalidOffset) repairs.emplace_back(edit.Entity,edit.Triangles);
        else r.emplace_or_replace<MeshGeometryDirty>(edit.Entity,EditSelectionAfter::Keep,false);
    }
    meshes.UpdateSelection(r,chain,updates);
    chain.AfterSubmit([&meshes,updates=std::move(updates)] {
        for (const auto &update : updates) meshes.PublishSelectionSummary(update.StoreId);
    });
    RepairShadingRender(r,chain,repairs);
    chain.Submit();
    RequestRender(r,RenderRequest::Rebuild);
}

const EditSelectionSummary *GetElementSelectionSummary(const state::Scene &r, state::Entity mesh_entity, Element element) {
    if (element == Element::None || !r.all_of<MeshElementSelection, MeshHandle>(mesh_entity)) return nullptr;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto id = GetMesh(r, mesh_entity).GetStoreId();
    const auto &summary = meshes.GetSelectionSummary(id);
    return summary.Mode == element ? &summary : nullptr;
}
