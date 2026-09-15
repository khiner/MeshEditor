#include "selection/SelectionGpu.h"
#include "state/Scene.h"

#include <Metal/MTLCommandQueue.hpp>

#include "Profile.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gpu/EditSelectionPushConstants.h"
#include "gpu/EditSharpnessPushConstants.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRoute.h"
#include "gpu/ObjectSelectionPushConstants.h"
#include "gpu/OverlayDispatch.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/VisibilitySelectionPushConstants.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "render/Encoding.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <bit>
#include <cmath>

using state::Change;

namespace {
std::vector<EditSelectionPushConstants> BuildSelectionTransactions(
    state::Scene &, std::span<const ElementRange>, Element, EditSelectionOperation, uint32_t pick_id_slot = InvalidSlot
);
void RecordSelectionPrepare(state::Scene &, mtl::PassChain &, std::span<const EditSelectionPushConstants>);
void RecordSelectionDerive(state::Scene &, mtl::PassChain &, std::span<const EditSelectionPushConstants>);

void SubmitAndWait(const mtl::Context &ctx, MTL::CommandBuffer *command_buffer) {
    const profile::CpuScope scope{"SelectionSubmit"};
    // Selection culling may allocate bindless buffers while encoding.
    ctx.CommitResidency();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

// Record selection passes into one command buffer and wait for them.
void SubmitSelectionPasses(state::Scene &r, auto &&record) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // End the final pass before submission.
        mtl::PassChain chain{command_buffer};
        record(chain);
    }
    SubmitAndWait(ctx, command_buffer);
}

struct ElementPickTarget {
    uvec2 Px;
    uint32_t RadiusSq;
};

struct PixelRect {
    uvec2 Origin{}, Extent{};
};

std::optional<PixelRect> ClampedRect(uvec2 lo, uvec2 hi, mtl::Extent2D target) {
    const auto limit = std::bit_cast<uvec2>(target);
    lo = numeric::Min(lo, limit);
    hi = numeric::Min(hi, limit);
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
    const SelectionSlots &sel_slots, uvec4 box, uint32_t box_result_slot, const std::optional<ElementPickTarget> &pick, bool resolve_id
) {
    return {
        box,
        box_result_slot,
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

uint32_t MaxElementBound(auto &&ranges) {
    return std::ranges::fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &r) { return std::max(total, r.Offset + r.Count); });
}

void EnsureSelectionVisibility(state::Scene &r, mtl::PassChain &chain) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (buffers.Visibility == GpuBuffers::VisibilityState{buffers.MeshletVisibleGeneration, false}) return;
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    RecordMeshletCull(chain, slots, pipelines, buffers, {.Mode = MeshletRouteMode::Visibility});
    RecordMeshletVisibilityPass(chain, slots, pipelines, r.ctx().get<const RenderTargets>(), buffers);
}

// Preserve scene depth while selection culling rewrites the visible list used for ID decoding.
// Picks raster twice; boxes raster once.
void RunSelectionPass(
    state::Scene &r, mtl::PassChain &chain, bool test_depth,
    std::optional<MeshletCullConfig> meshlet_cull, bool pick, auto &&record_draws
) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    auto &buffers = r.ctx().get<GpuBuffers>();

    if (test_depth) EnsureSelectionVisibility(r, chain);
    if (meshlet_cull && buffers.MeshletInstanceCount > 0) {
        RecordMeshletCull(chain, slots, pipelines, buffers, *meshlet_cull);
    }

    const auto extent = r.ctx().get<const RenderTargets>().Resources->ScratchDepth.Extent;
    const uint32_t raster_passes = pick ? 2u : 1u;
    for (uint32_t index = 0; index < raster_passes; ++index) {
        // Scene depth remains valid for shading and later picks; depth-free queries need no scratch contents.
        const auto depth = test_depth ? mtl::LoadDepth(*r.ctx().get<const RenderTargets>().Resources->VisibilityDepth) :
                                        mtl::DepthAttachment{*r.ctx().get<const RenderTargets>().Resources->ScratchDepth, MTL::LoadActionDontCare, MTL::StoreActionDontCare};
        const auto pass = mtl::MakePassDescriptor({}, depth);
        pass->setRenderTargetWidth(extent.Width);
        pass->setRenderTargetHeight(extent.Height);
        // The pick resolve reads the key an earlier raster wrote, and bindless buffers carry no tracked hazard.
        auto *encoder = encode::BeginScenePass(chain, pass, "SelectionPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageBlit | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers);
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
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    const bool xray_selection = r.get<const SelectionXRay>(viewport).Value;
    const auto &selection = pipelines.SelectionFragment;
    const bool degenerate_point_pass = write_bitset && xray_selection && element != Element::Vertex;
    for (const auto &range : ranges) {
        [[maybe_unused]] const auto &mesh_buffers = r.edit<MeshBuffers>(range.MeshEntity);
        assert(mesh_buffers.Meshlets.Count > 0u && "selectable mesh geometry must have persistent meshlets");
    }

    RunSelectionPass(
        r, chain, !xray_selection,
        MeshletCullConfig{
            .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::ElementSelection),
            .RouteMask = 1u << uint32_t(MeshletRoute::OpaqueCullBack),
        },
        pick.has_value(),
        [&](auto *encoder, mtl::Extent2D, bool resolve_id) {
            const SelectionElementPushConstants element_pc{MakeElementQuery(sel_slots, {box_min.x, box_min.y, box_max.x, box_max.y}, meshes.Slots().SelectionBits, pick, resolve_id)};
            if (write_bitset) {
                const auto rect = BoxRect({box_min.x, box_min.y, box_max.x, box_max.y}, r.ctx().get<const RenderTargets>().Resources->ScratchDepth.Extent);
                if (!rect) return;
                encoder->setScissorRect({rect->Origin.x, rect->Origin.y, rect->Extent.x, rect->Extent.y});
            }
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
            else DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), element == Element::Vertex ? 64u : 160u);
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
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const profile::CpuScope scope{"RunElementPick"};
    auto &buffers = r.ctx().get<GpuBuffers>();
    ResetElementPick(buffers);
    const auto transactions = BuildSelectionTransactions(
        r, ranges, element,
        toggle ? EditSelectionOperation::PickToggle : EditSelectionOperation::PickReplace,
        r.ctx().get<const SelectionSlots>().ElementPickId
    );
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderElementSelectionPass(r, chain, viewport, ranges, element, false, {}, {}, ElementPickTarget{mouse_px, ElementPickRadiusSq(element)});
        RecordSelectionPrepare(r, chain, transactions);
        RecordSelectionDerive(r, chain, transactions);
    });
    r.ctx().get<GpuSceneState>().EditSelectionDirty = true;
    if (const auto index = ReadNearestPickedElement(buffers, element_count)) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
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
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    auto &buffers = r.ctx().get<GpuBuffers>();

    EnsureSelectionVisibility(r, chain);
    const auto rect = ObjectQueryRect(query, r.ctx().get<const RenderTargets>().Resources->VisibilityImage.Extent);
    if (!rect) return;

    auto *encoder = chain.BeginCompute("VisibilityObjectSelection", MTL::StageFragment);
    encode::BindCompute(encoder, pipelines.VisibilityObjectSelection, slots, buffers);
    encoder->setTexture(*r.ctx().get<const RenderTargets>().Resources->VisibilityImage, 0u);
    encoder->setTexture(*r.ctx().get<const RenderTargets>().Resources->VisibilityDepth, 1u);
    encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{encode::VisibilityDecodePc(buffers), query, rect->Origin, rect->Extent});
    encoder->dispatchThreadgroups(
        MTL::Size((rect->Extent.x + 15u) / 16u, (rect->Extent.y + 15u) / 16u, 1u),
        ThreadgroupSize::Tile16
    );
}

void RenderSelectionPickPass(state::Scene &r, mtl::PassChain &chain, std::optional<ObjectSelectQuery> object, std::optional<uint32_t> sound_instance = {}, std::optional<ElementPickTarget> pick = {}) {
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = GetPipelines(r);
    const auto &selection = pipelines.SelectionFragment;
    if (object) {
        if (object->BestKeySlot != InvalidSlot) {
            // Click cycling needs every covered surface, including occluded objects.
            RecordMeshletCull(chain, r.ctx().get<const mtl::BindlessSet>(), pipelines, buffers, {.Mode = MeshletRouteMode::Material});
        } else RecordVisibilityObjectSelection(r, chain, *object);
        RecordOverlayJobCull(chain, r.ctx().get<const mtl::BindlessSet>(), pipelines, buffers, true);
    }
    const auto sound_cull = sound_instance ?
        std::optional{MeshletCullConfig{
            .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::SoundPoint),
            .RouteMask = 1u << uint32_t(MeshletRoute::OpaqueCullBack),
        }} :
        std::nullopt;
    RunSelectionPass(r, chain, sound_instance.has_value(), sound_cull, pick.has_value(), [&](auto *encoder, mtl::Extent2D, bool resolve_id) {
        if (sound_instance) {
            const SelectionElementPushConstants point_pc{MakeElementQuery(sel_slots, {}, InvalidSlot, pick, resolve_id)};
            selection.ElementRaster(Element::Vertex, false, false).Bind(encoder);
            encoder->setFragmentBytes(&point_pc, sizeof(point_pc), BufferIndex_PushConstants);
            DrawMeshlets(
                encoder, buffers, uint32_t(MeshletRoute::OpaqueCullBack),
                uint32_t(MeshletInstanceFlag::SoundPoint), 64u, 0u, *sound_instance
            );
        }
        if (object) {
            const auto rect = ObjectQueryRect(*object, r.ctx().get<const RenderTargets>().Resources->ScratchDepth.Extent);
            if (!rect) return;
            encoder->setScissorRect({rect->Origin.x, rect->Origin.y, rect->Extent.x, rect->Extent.y});
            if (object->BestKeySlot != InvalidSlot) {
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
            DrawOverlayJobs(encoder, buffers, r.ctx().get<const MeshStore>());
        }
    });
}

void RunBoxSelectElements(state::Scene &r, state::Entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    const profile::CpuScope scope{"RunBoxSelectElements"};

    auto *baseline = is_additive ? r.try_edit<AdditiveBoxSelectBaseline>(viewport) : nullptr;
    const auto operation = !is_additive                 ? EditSelectionOperation::Clear :
        baseline && !baseline->ElementSelectionCaptured ? EditSelectionOperation::CaptureBaseline :
                                                          EditSelectionOperation::RestoreBaseline;
    const auto transactions = BuildSelectionTransactions(r, ranges, element, operation);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RecordSelectionPrepare(r, chain, transactions);
        RenderElementSelectionPass(r, chain, viewport, ranges, element, true, box_min, box_max, {});
        RecordSelectionDerive(r, chain, transactions);
    });
    if (baseline) baseline->ElementSelectionCaptured = true;
    r.ctx().get<GpuSceneState>().EditSelectionDirty = true;
}

std::optional<uint32_t> RunSoundVerticesVertexPick(state::Scene &r, state::Entity instance_entity, uvec2 mouse_px) {
    if (!r.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = r.try_get<Instance>(instance_entity);
    if (!instance) return {};
    auto &buffers = r.ctx().get<GpuBuffers>();

    const profile::CpuScope scope{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto model_index = r.get<RenderInstance>(instance_entity).BufferIndex;
    ResetElementPick(buffers);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderSelectionPickPass(r, chain, std::nullopt, model_index, ElementPickTarget{mouse_px, ElementPickRadiusSq(Element::Vertex)});
    });
    return ReadNearestPickedElement(buffers, vertex_count);
}

namespace {
void ReserveObjectPicking(state::Scene &r, uint32_t count) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    if (count <= buffers.ObjectPickKeys.Count<uint32_t>()) return;
    buffers.ObjectPickKeys.SetCount<uint32_t>(count);
    buffers.ObjectPickSeenBitset.SetCount<uint32_t>((count + 31) / 32);
    buffers.ObjectBoxBitset.SetCount<uint32_t>((count + 31) / 32);
    buffers.ObjectPickEpochTag = 0;
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    const auto &selection = r.ctx().get<const SelectionSlots>();
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
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t index = object_id - 1;
        if ((bits[index / 32] & (1u << (index % 32))) == 0) continue;
        const auto entity = r.EntityAt(index);
        if (r.all_of<RenderInstance>(entity)) fn(index, entity);
    }
}
} // namespace

std::vector<state::Entity> RunObjectPick(state::Scene &r, uvec2 mouse_px, uint32_t radius_px) {
    const uint32_t max_object_id = PrepareObjectQuery(r);
    if (max_object_id == 0) return {};
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
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
            }
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

std::vector<state::Entity> RunBoxSelect(state::Scene &r, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    const uint32_t max_object_id = PrepareObjectQuery(r);
    if (max_object_id == 0) return {};
    auto &buffers = r.ctx().get<GpuBuffers>();
    const profile::CpuScope scope{"RunBoxSelect"};
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    std::ranges::fill(buffers.ObjectBoxBitset.GetMutableSpan<uint32_t>({0, (max_object_id + 31) / 32}), 0u);
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RenderSelectionPickPass(
            r, chain,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .BestKeySlot = InvalidSlot,
                .Box = {box_min.x, box_min.y, box_max.x, box_max.y},
                .BoxResultSlot = sel_slots.ObjectBoxBitset,
            }
        );
    });
    std::vector<state::Entity> entities;
    ForEachHitObject(r, buffers.ObjectBoxBitset.GetSpan<uint32_t>(), max_object_id, [&](uint32_t, state::Entity entity) { entities.emplace_back(entity); });
    return entities;
}

namespace {
std::vector<EditSelectionPushConstants> BuildSelectionTransactions(
    state::Scene &r, std::span<const ElementRange> ranges, Element element,
    EditSelectionOperation operation, uint32_t pick_id_slot
) {
    std::vector<EditSelectionPushConstants> result;
    result.reserve(ranges.size());
    auto &meshes = r.ctx().get<MeshStore>();
    for (const auto &range : ranges) {
        const auto &mesh = GetMesh(r, range.MeshEntity);
        const auto &mesh_buffers = r.get<const MeshBuffers>(range.MeshEntity);
        const auto store_id = mesh.GetStoreId();
        meshes.EnsureSelectionBits(mesh);
        meshes.CaptureSelectionWrite(store_id);
        const auto &record = meshes.Get(store_id);
        const auto &derived = meshes.GetDerived(store_id);
        const auto &arenas = meshes.Arenas();
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        auto halfedge_to_edge = arenas.Connectivity.Slotted(record.ConnectivityHalfedgeToEdge);
        if (halfedge_to_edge.Count == 0) halfedge_to_edge.Offset = InvalidOffset;
        result.emplace_back(EditSelectionPushConstants{
            .Selection = meshes.GetEditSelectionStorage(store_id),
            .EdgeIndices = mesh_buffers.EdgeIndices,
            .Corners = corners,
            .Connectivity = arenas.Connectivity.Slotted(record.Connectivity),
            .HalfedgeToEdge = halfedge_to_edge,
            .EdgeHalfedges = arenas.Connectivity.Slotted(record.ConnectivityEdges),
            .Vertices = arenas.Vertices.Slotted(record.Vertices),
            .VertexFanAdjacencyOffset = OffsetOrInvalid(derived.VertexFanAdjacency),
            .VertexEdgeAdjacencyOffset = OffsetOrInvalid(derived.VertexEdgeAdjacency),
            .AdjacencySlot = meshes.Slots().Adjacency,
            .FaceSharpness = arenas.FaceSharpness.Slotted(record.FaceData),
            .EdgeSharpness = arenas.EdgeSharpness.Slotted(record.EdgeSharpness),
            .SelectionBaseline = meshes.GetSelectionBaselineRange(store_id),
            .VertexCount = mesh.VertexCount(),
            .EdgeCount = mesh.EdgeCount(),
            .FaceCount = mesh.FaceCount(),
            .HalfedgeCount = corners.Count,
            .Element = element,
            .ConnectivityFaceStarts = mesh.GetConnectivity().Faces.empty() ? 0u : 1u,
            .Operation = operation,
            .PickIdSlot = pick_id_slot,
        });
    }
    return result;
}

void RecordSelectionPrepare(
    state::Scene &r, mtl::PassChain &chain,
    std::span<const EditSelectionPushConstants> transactions
) {
    if (transactions.empty() || std::ranges::all_of(transactions, [](const auto &pc) { return pc.Operation == EditSelectionOperation::Derive; })) return;
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    auto *encoder = chain.BeginCompute("SelectionPrepare", MTL::StageFragment | MTL::StageDispatch);
    for (const auto &pc : transactions) {
        const uint32_t count = pc.Element == Element::Vertex ? pc.VertexCount :
            pc.Element == Element::Edge                      ? pc.EdgeCount :
                                                               pc.FaceCount;
        if (count == 0) continue;
        encode::BindCompute(encoder, pipelines.PrepareEditSelection, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size(((count + 31u) / 32u + 255u) / 256u, 1, 1), ThreadgroupSize::Linear256);
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        if (pc.Operation == EditSelectionOperation::FillList && pc.SelectionListCount > 0u) {
            encode::BindCompute(encoder, pipelines.FillEditSelectionList, slots, buffers);
            encode::SetPushConstants(encoder, pc);
            encoder->dispatchThreadgroups(MTL::Size((pc.SelectionListCount + 255u) / 256u, 1, 1), ThreadgroupSize::Linear256);
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        }
    }
}

void RecordSelectionDerive(
    state::Scene &r, mtl::PassChain &chain,
    std::span<const EditSelectionPushConstants> transactions
) {
    if (transactions.empty()) return;
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    uint32_t partial_count = 0;
    for (const auto &pc : transactions) partial_count = std::max(partial_count, (pc.VertexCount + 511u) / 512u);
    buffers.EditSelectionPositionSums.SetCount<vec3>(std::max(partial_count, 1u));
    auto *encoder = chain.BeginCompute("SelectionDerive", MTL::StageFragment | MTL::StageDispatch);
    for (auto pc : transactions) {
        pc.PositionSumsSlot = buffers.EditSelectionPositionSums.Slot;
        encode::BindCompute(encoder, pipelines.ResetEditSelectionSummary, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);

        const uint32_t word_count = std::max({
            (pc.VertexCount + 15u) / 16u,
            (pc.EdgeCount + 15u) / 16u,
            (pc.FaceCount + 15u) / 16u,
        });
        if (word_count == 0) continue;
        encode::BindCompute(encoder, pipelines.DeriveEditSelection, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size((word_count + 255) / 256, 1, 1), ThreadgroupSize::Linear256);
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        encode::BindCompute(encoder, pipelines.SumEditSelectionPosition, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(32, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
}

void ApplySelectionTransactions(state::Scene &r, std::span<const EditSelectionPushConstants> transactions) {
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        RecordSelectionPrepare(r, chain, transactions);
        RecordSelectionDerive(r, chain, transactions);
    });
    r.ctx().get<GpuSceneState>().EditSelectionDirty = true;
}
} // namespace

void ApplyEditSelectionCommand(
    state::Scene &r, std::span<const ElementRange> ranges,
    Element element, EditSelectionOperation operation
) {
    if (ranges.empty() || element == Element::None) return;
    const auto transactions = BuildSelectionTransactions(r, ranges, element, operation);
    ApplySelectionTransactions(r, transactions);
}

void ApplyEditSelectionLists(
    state::Scene &r,
    std::span<const std::pair<state::Entity, SlottedRange>> lists, Element element
) {
    if (lists.empty() || element == Element::None) return;
    std::vector<ElementRange> ranges;
    std::vector<SlottedRange> valid_lists;
    ranges.reserve(lists.size());
    valid_lists.reserve(lists.size());
    for (const auto &[mesh_entity, list] : lists) {
        if (!HasMesh(r, mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        ranges.emplace_back(mesh_entity, 0u, mesh.ElementCount(element));
        valid_lists.push_back(list);
    }
    auto transactions = BuildSelectionTransactions(r, ranges, element, EditSelectionOperation::FillList);
    if (transactions.empty()) return;
    for (uint32_t i = 0; i < transactions.size(); ++i) {
        transactions[i].SelectionList = valid_lists[i];
        transactions[i].SelectionListCount = valid_lists[i].Count;
    }
    ApplySelectionTransactions(r, transactions);
}

void ApplyEditSharpness(
    state::Scene &r, state::Entity viewport, std::span<const state::Entity> mesh_entities,
    EditSharpnessOperation operation, bool value, float angle
) {
    if (mesh_entities.empty()) return;
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<EditSharpnessPushConstants> commands;
    std::vector<state::Entity> edited;
    commands.reserve(mesh_entities.size());
    edited.reserve(mesh_entities.size());
    const bool uses_selection = operation == EditSharpnessOperation::SetSelectedFaces ||
        operation == EditSharpnessOperation::SetSelectedEdges ||
        operation == EditSharpnessOperation::SetVertexEdges;
    for (const auto mesh_entity : mesh_entities) {
        if (!HasMesh(r, mesh_entity) || !r.all_of<MeshBuffers>(mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        if (mesh.FaceCount() == 0) continue;
        const auto id = mesh.GetStoreId();
        if (uses_selection) meshes.EnsureSelectionBits(mesh);
        meshes.CaptureSharpnessWrite(id, operation);
        const auto &record = meshes.Get(id);
        const auto &arenas = meshes.Arenas();
        const auto corners = arenas.FaceCorners.Slotted(record.FaceCorners);
        commands.emplace_back(EditSharpnessPushConstants{
            .VertexSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Vertex) : SlottedRange{},
            .EdgeSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Edge) : SlottedRange{},
            .FaceSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Face) : SlottedRange{},
            .FaceSharpness = arenas.FaceSharpness.Slotted(record.FaceData),
            .EdgeSharpness = arenas.EdgeSharpness.Slotted(record.EdgeSharpness),
            .Connectivity = arenas.Connectivity.Slotted(record.Connectivity),
            .EdgeHalfedges = arenas.Connectivity.Slotted(record.ConnectivityEdges),
            .EdgeIndices = r.get<const MeshBuffers>(mesh_entity).EdgeIndices,
            .FaceNormals = arenas.BaseFaceNormals.Slotted(record.FaceData),
            .VertexCount = mesh.VertexCount(),
            .EdgeCount = mesh.EdgeCount(),
            .FaceCount = mesh.FaceCount(),
            .HalfedgeCount = corners.Count,
            .ConnectivityFaceStarts = mesh.GetConnectivity().Faces.empty() ? 0u : 1u,
            .Operation = operation,
            .Value = value ? 1u : 0u,
            .CosAngle = std::cos(angle),
        });
        edited.push_back(mesh_entity);
    }
    if (commands.empty()) return;
    std::vector<ElementRange> selection_ranges;
    const auto element = r.get<const EditMode>(viewport).Value;
    if (element != Element::None) {
        for (const auto mesh_entity : edited) {
            if (!r.all_of<MeshElementSelection>(mesh_entity)) continue;
            const auto mesh = GetMesh(r, mesh_entity);
            const auto count = mesh.ElementCount(element);
            if (count > 0) selection_ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), element), count);
        }
    }
    const auto selection_transactions = BuildSelectionTransactions(
        r, selection_ranges, element, EditSelectionOperation::Derive
    );
    SubmitSelectionPasses(r, [&](mtl::PassChain &chain) {
        auto *encoder = chain.BeginCompute("EditSharpness");
        const auto &slots = r.ctx().get<const mtl::BindlessSet>();
        const auto &pipelines = GetPipelines(r);
        const auto &buffers = r.ctx().get<const GpuBuffers>();
        for (const auto &pc : commands) {
            encode::BindCompute(encoder, pipelines.EditSharpness, slots, buffers);
            encode::SetPushConstants(encoder, pc);
            const uint32_t count = std::max(pc.EdgeCount, pc.FaceCount);
            encoder->dispatchThreadgroups(MTL::Size((count + 255u) / 256u, 1, 1), ThreadgroupSize::Linear256);
        }
        RecordSelectionDerive(r, chain, selection_transactions);
    });
    for (const auto mesh_entity : edited) reactive(r, Change::MeshShading).emplace(mesh_entity);
}

const EditSelectionSummary *GetElementSelectionSummary(const state::Scene &r, state::Entity mesh_entity, Element element) {
    if (element == Element::None || !r.all_of<MeshElementSelection, MeshHandle>(mesh_entity)) return nullptr;
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto id = r.get<const MeshHandle>(mesh_entity).StoreId;
    const auto &summary = meshes.GetSelectionSummary(id);
    return summary.Mode == element ? &summary : nullptr;
}
