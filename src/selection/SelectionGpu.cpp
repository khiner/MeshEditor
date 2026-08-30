#include "selection/SelectionGpu.h"

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
#include "object/ObjectComponents.h"
#include "render/Encoding.h"
#include "render/Instance.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionQueries.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <entt/entity/registry.hpp>

#include <bit>
#include <cmath>

namespace {
std::vector<EditSelectionPushConstants> BuildSelectionTransactions(
    entt::registry &, std::span<const ElementRange>, Element, EditSelectionOperation, uint32_t pick_id_slot = InvalidSlot
);
void RecordSelectionPrepare(entt::registry &, MTL::CommandBuffer *, std::span<const EditSelectionPushConstants>);
void RecordSelectionDerive(entt::registry &, MTL::CommandBuffer *, std::span<const EditSelectionPushConstants>);

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

void RunSelectionPass(
    entt::registry &r, mtl::PassChain &chain, bool render_depth, bool render_silhouette,
    std::optional<MeshletCullConfig> meshlet_cull, auto &&record_draws
) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    if (render_depth) RecordSilhouetteDepthPass(chain, slots, pipelines, buffers, render_silhouette);
    if (meshlet_cull && buffers.MeshletInstanceCount > 0) {
        RecordMeshletCull(chain, slots, pipelines, buffers, *meshlet_cull);
    }

    const auto extent = pipelines.Silhouette.Resources->DepthImage.Extent;
    const auto pass = mtl::MakePassDescriptor({}, mtl::LoadDepth(*pipelines.Silhouette.Resources->DepthImage, MTL::StoreActionDontCare));
    pass->setRenderTargetWidth(extent.Width);
    pass->setRenderTargetHeight(extent.Height);
    // The pick resolve reads the key an earlier raster wrote, and bindless buffers carry no tracked hazard.
    auto *encoder = encode::BeginScenePass(chain, pass, "SelectionPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageBlit | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers);
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

    const bool xray_selection = r.get<const SelectionXRay>(viewport).Value;
    const auto &selection = pipelines.SelectionFragment;
    const bool render_depth = !xray_selection;
    const bool degenerate_point_pass = write_bitset && xray_selection && element != Element::Vertex;
    for (const auto &range : ranges) {
        [[maybe_unused]] const auto &mesh_buffers = r.get<MeshBuffers>(range.MeshEntity);
        assert(mesh_buffers.Meshlets.Count > 0u && "selectable mesh geometry must have persistent meshlets");
    }

    RunSelectionPass(
        r, chain, render_depth, true,
        MeshletCullConfig{
            .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::ElementSelection),
            .RouteMask = 1u << uint32_t(MeshletRoute::OpaqueCullBack),
        },
        [&](auto *encoder, mtl::Extent2D) {
            const SelectionElementPushConstants element_pc{MakeElementQuery(sel_slots, {box_min.x, box_min.y, box_max.x, box_max.y}, meshes.GetSelectionBitsSlot(), pick)};
            if (write_bitset) {
                const auto extent = pipelines.Silhouette.Resources->DepthImage.Extent;
                const auto min_x = std::min(box_min.x, extent.Width);
                const auto min_y = std::min(box_min.y, extent.Height);
                const auto max_x = uint32_t(std::min<uint64_t>(uint64_t{box_max.x} + 1u, extent.Width));
                const auto max_y = uint32_t(std::min<uint64_t>(uint64_t{box_max.y} + 1u, extent.Height));
                if (max_x <= min_x || max_y <= min_y) return;
                encoder->setScissorRect({min_x, min_y, max_x - min_x, max_y - min_y});
            }
            const auto &pipeline = selection.ElementRaster(element, write_bitset, xray_selection);
            pipeline.Bind(encoder);
            encoder->setFragmentBytes(&element_pc, sizeof(element_pc), BufferIndex_PushConstants);
            if (element == Element::Edge) {
                for (uint32_t corner = 0u; corner < 3u; ++corner) {
                    DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 160u, corner);
                }
            } else {
                DrawMeshlets(
                    encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection),
                    element == Element::Vertex ? 64u : 160u
                );
            }
            if (degenerate_point_pass) {
                const auto &point_pipeline = element == Element::Face ?
                    selection.MeshletFaceXRayPointsBitsetBox :
                    selection.MeshletEdgeXRayPointsBitsetBox;
                point_pipeline.Bind(encoder);
                encoder->setFragmentBytes(&element_pc, sizeof(element_pc), BufferIndex_PushConstants);
                if (element == Element::Face) {
                    DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 64u);
                } else {
                    for (uint32_t corner = 0u; corner < 3u; ++corner) {
                        DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection), 160u, corner);
                    }
                }
            }
        }
    );
}

} // namespace

std::optional<std::pair<entt::entity, uint32_t>> RunEditElementClick(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px, bool toggle
) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const profile::CpuScope scope{"RunElementPick"};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    ResetElementPick(buffers);
    const auto transactions = BuildSelectionTransactions(
        r, ranges, element,
        toggle ? EditSelectionOperation::PickToggle : EditSelectionOperation::PickReplace,
        r.ctx().get<const SelectionSlots>().ElementPickId
    );
    ctx.CommitResidency();
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
    RecordSelectionPrepare(r, command_buffer, transactions);
    RecordSelectionDerive(r, command_buffer, transactions);
    SubmitAndWait(command_buffer);
    for (const auto &range : ranges) RefreshElementSelectionStats(r, range.MeshEntity);
    r.emplace_or_replace<EditSelectionDirty>(viewport);
    if (const auto index = ReadNearestPickedElement(buffers, element_count)) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
    }
    return {};
}

struct PixelRect {
    uvec2 Origin{}, Extent{};
};

std::optional<PixelRect> ObjectQueryRect(const ObjectSelectQuery &query, mtl::Extent2D target) {
    const auto limit = std::bit_cast<uvec2>(target);
    uvec2 lo{}, hi{}; // Exclusive high bound.
    if (query.BoxResultSlot != InvalidSlot) {
        lo = numeric::Min(uvec2{query.Box.x, query.Box.y}, limit);
        hi = numeric::Min(uvec2{
                              uint32_t(std::min<uint64_t>(uint64_t{query.Box.z} + 1u, target.Width)),
                              uint32_t(std::min<uint64_t>(uint64_t{query.Box.w} + 1u, target.Height)),
                          },
                          limit);
    } else if (query.BestKeySlot != InvalidSlot) {
        const uint32_t radius = uint32_t(std::ceil(std::sqrt(float(query.RadiusSq))));
        lo = numeric::Min(uvec2{
                              query.TargetPx.x > radius ? query.TargetPx.x - radius : 0u,
                              query.TargetPx.y > radius ? query.TargetPx.y - radius : 0u,
                          },
                          limit);
        hi = numeric::Min(uvec2{
                              uint32_t(std::min<uint64_t>(uint64_t{query.TargetPx.x} + radius + 1u, target.Width)),
                              uint32_t(std::min<uint64_t>(uint64_t{query.TargetPx.y} + radius + 1u, target.Height)),
                          },
                          limit);
    }
    if (hi.x <= lo.x || hi.y <= lo.y) return {};
    return PixelRect{lo, hi - lo};
}

void RecordVisibilityObjectSelection(
    entt::registry &r, mtl::PassChain &chain, const ObjectSelectQuery &query
) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    // A cull without a following visibility raster invalidates the id-to-visible-list mapping.
    // Rebuild that same shared representation on demand instead of maintaining selection geometry.
    if (buffers.VisibilityIdGeneration != buffers.MeshletVisibleGeneration) {
        RecordMeshletCull(chain, slots, pipelines, buffers, {.Mode = MeshletRouteMode::Visibility});
        RecordMeshletVisibilityPass(chain, slots, pipelines, buffers);
    }
    const auto rect = ObjectQueryRect(query, pipelines.Main.Resources->VisibilityImage.Extent);
    if (!rect) return;

    auto *encoder = chain.BeginCompute("VisibilityObjectSelection", MTL::StageFragment);
    encode::BindCompute(encoder, pipelines.VisibilityObjectSelection, slots, buffers);
    encoder->setTexture(*pipelines.Main.Resources->VisibilityImage, 0u);
    encoder->setTexture(*pipelines.Main.Resources->DepthImage, 1u);
    encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{encode::VisibilityDecodePc(buffers), query, rect->Origin, rect->Extent});
    encoder->dispatchThreadgroups(
        MTL::Size((rect->Extent.x + 15u) / 16u, (rect->Extent.y + 15u) / 16u, 1u),
        ThreadgroupSize::Tile16
    );
}

void RenderSelectionPickPass(entt::registry &r, mtl::PassChain &chain, std::optional<ObjectSelectQuery> object, std::optional<uint32_t> sound_instance = {}, std::optional<ElementPickTarget> pick = {}) {
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &selection = pipelines.SelectionFragment;
    if (object) {
        RecordVisibilityObjectSelection(r, chain, *object);
        RecordOverlayJobCull(chain, r.ctx().get<const mtl::BindlessSet>(), pipelines, buffers, true);
    }
    if (sound_instance) {
        RecordMeshletCull(
            chain, r.ctx().get<const mtl::BindlessSet>(), pipelines, buffers,
            {
                .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::SoundPoint),
                .RouteMask = 1u << uint32_t(MeshletRoute::OpaqueCullBack),
            }
        );
    }
    RunSelectionPass(r, chain, true, sound_instance.has_value(), std::nullopt, [&](auto *encoder, mtl::Extent2D) {
        if (sound_instance) {
            const SelectionElementPushConstants point_pc{MakeElementQuery(sel_slots, {}, InvalidSlot, pick)};
            selection.ElementRaster(Element::Vertex, false, false).Bind(encoder);
            encoder->setFragmentBytes(&point_pc, sizeof(point_pc), BufferIndex_PushConstants);
            DrawMeshlets(
                encoder, buffers, uint32_t(MeshletRoute::OpaqueCullBack),
                uint32_t(MeshletInstanceFlag::SoundPoint), 64u, 0u, *sound_instance
            );
        }
        if (object) {
            const ObjectSelectionPushConstants sel_pc{*object};
            // Bone joints pick from the same persistent route and emission as the overlay.
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

void RunBoxSelectElements(entt::registry &r, entt::entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    const profile::CpuScope scope{"RunBoxSelectElements"};

    auto *baseline = is_additive ? r.try_get<AdditiveBoxSelectBaseline>(viewport) : nullptr;
    const auto operation = !is_additive                 ? EditSelectionOperation::Clear :
        baseline && !baseline->ElementSelectionCaptured ? EditSelectionOperation::CaptureBaseline :
                                                          EditSelectionOperation::RestoreBaseline;
    const auto transactions = BuildSelectionTransactions(r, ranges, element, operation);
    const auto &ctx = r.ctx().get<const mtl::Context>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordSelectionPrepare(r, command_buffer, transactions);
    {
        mtl::PassChain chain{command_buffer};
        RenderElementSelectionPass(r, chain, viewport, ranges, element, true, box_min, box_max, {});
    }
    RecordSelectionDerive(r, command_buffer, transactions);
    command_buffer->commit();
    if (baseline) baseline->ElementSelectionCaptured = true;
    r.emplace_or_replace<EditSelectionDirty>(viewport);
    r.emplace_or_replace<BoxSelectStatsDirty>(viewport);
}

std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &r, entt::entity instance_entity, uvec2 mouse_px) {
    if (!r.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = r.try_get<Instance>(instance_entity);
    if (!instance) return {};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    const profile::CpuScope scope{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto model_index = r.get<RenderInstance>(instance_entity).BufferIndex;
    ResetElementPick(buffers);
    auto *command_buffer = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        for (const bool resolve_id : {false, true}) {
            RenderSelectionPickPass(r, chain, std::nullopt, model_index, ElementPickTarget{mouse_px, ElementPickRadiusSq(Element::Vertex), resolve_id});
        }
    }
    SubmitAndWait(command_buffer);
    return ReadNearestPickedElement(buffers, vertex_count);
}

std::vector<entt::entity> RunObjectPick(entt::registry &r, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const profile::CpuScope scope{"RunObjectPick"};
    // ObjectPickKeys persists across clicks, with the high 8 bits of each packed key holding a per-click epoch tag.
    // A full reset runs only when the 8-bit epoch wraps, and readback filters stale keys by epoch.
    if (object_pick_epoch_tag == 0) {
        ResetObjectPickKeys(buffers);
        object_pick_epoch_tag = 255;
    }
    const uint32_t epoch_inv = object_pick_epoch_tag--;

    std::fill_n(buffers.ObjectPickSeenBitset.Data(), (max_object_id + 31) / 32, 0u);
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
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
    }
    SubmitAndWait(command_buffer);
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

std::vector<entt::entity> RunBoxSelect(entt::registry &r, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    auto &buffers = r.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);

    const profile::CpuScope scope{"RunBoxSelect"};
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    memset(buffers.ObjectBoxBitset.Data(), 0, ((max_object_id + 31) / 32) * sizeof(uint32_t));
    auto *command_buffer = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        RenderSelectionPickPass(
            r, chain,
            ObjectSelectQuery{
                .MaxId = max_object_id,
                .BestKeySlot = InvalidSlot,
                .Box = {box_min.x, box_min.y, box_max.x, box_max.y},
                .BoxResultSlot = sel_slots.ObjectBoxBitset,
            }
        );
    }
    SubmitAndWait(command_buffer);
    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : r.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;

    const auto *bits = buffers.ObjectBoxBitset.Data();
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

namespace {
std::vector<EditSelectionPushConstants> BuildSelectionTransactions(
    entt::registry &r, std::span<const ElementRange> ranges, Element element,
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
        const auto corners = meshes.GetFaceCornerRange(store_id);
        auto halfedge_to_edge = meshes.GetConnectivityHalfedgeToEdgeRange(store_id);
        if (halfedge_to_edge.Count == 0) halfedge_to_edge.Offset = InvalidOffset;
        result.emplace_back(EditSelectionPushConstants{
            .Selection = meshes.GetEditSelectionStorage(store_id),
            .EdgeIndices = mesh_buffers.EdgeIndices,
            .Corners = corners,
            .Connectivity = meshes.GetConnectivityRange(store_id),
            .HalfedgeToEdge = halfedge_to_edge,
            .EdgeHalfedges = meshes.GetConnectivityEdgeRange(store_id),
            .Vertices = meshes.GetVerticesRange(store_id),
            .VertexFanAdjacencyOffset = OffsetOrInvalid(meshes.GetVertexFanAdjacencyRange(store_id)),
            .VertexEdgeAdjacencyOffset = OffsetOrInvalid(meshes.GetVertexEdgeAdjacencyRange(store_id)),
            .AdjacencySlot = meshes.GetAdjacencySlot(),
            .FaceSharpness = meshes.GetFaceSharpnessRange(store_id),
            .EdgeSharpness = meshes.GetEdgeSharpnessSlottedRange(store_id),
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
    entt::registry &r, MTL::CommandBuffer *command_buffer,
    std::span<const EditSelectionPushConstants> transactions
) {
    if (transactions.empty() || std::ranges::all_of(transactions, [](const auto &pc) { return pc.Operation == EditSelectionOperation::Derive; })) return;
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    auto *encoder = command_buffer->computeCommandEncoder();
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
    encoder->endEncoding();
}

void RecordSelectionDerive(
    entt::registry &r, MTL::CommandBuffer *command_buffer,
    std::span<const EditSelectionPushConstants> transactions
) {
    if (transactions.empty()) return;
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    auto *encoder = command_buffer->computeCommandEncoder();
    for (const auto &pc : transactions) {
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
    }
    encoder->endEncoding();
}
} // namespace

void ApplyEditSelectionCommand(
    entt::registry &r, entt::entity viewport, std::span<const ElementRange> ranges,
    Element element, EditSelectionOperation operation
) {
    if (ranges.empty() || element == Element::None) return;
    const auto transactions = BuildSelectionTransactions(r, ranges, element, operation);
    const auto &ctx = r.ctx().get<const mtl::Context>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordSelectionPrepare(r, command_buffer, transactions);
    RecordSelectionDerive(r, command_buffer, transactions);
    SubmitAndWait(command_buffer);
    for (const auto &range : ranges) RefreshElementSelectionStats(r, range.MeshEntity);
    r.emplace_or_replace<EditSelectionDirty>(viewport);
}

void ApplyEditSelectionLists(
    entt::registry &r, entt::entity viewport,
    std::span<const std::pair<entt::entity, SlottedRange>> lists, Element element
) {
    if (lists.empty() || element == Element::None) return;
    std::vector<ElementRange> ranges;
    std::vector<SlottedRange> valid_lists;
    ranges.reserve(lists.size());
    valid_lists.reserve(lists.size());
    for (const auto &[mesh_entity, list] : lists) {
        if (!HasMesh(r, mesh_entity)) continue;
        const auto mesh = GetMesh(r, mesh_entity);
        ranges.emplace_back(mesh_entity, 0u, selection::GetElementCount(mesh, element));
        valid_lists.push_back(list);
    }
    auto transactions = BuildSelectionTransactions(r, ranges, element, EditSelectionOperation::FillList);
    if (transactions.empty()) return;
    for (uint32_t i = 0; i < transactions.size(); ++i) {
        transactions[i].SelectionList = valid_lists[i];
        transactions[i].SelectionListCount = valid_lists[i].Count;
    }
    const auto &ctx = r.ctx().get<const mtl::Context>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordSelectionPrepare(r, command_buffer, transactions);
    RecordSelectionDerive(r, command_buffer, transactions);
    SubmitAndWait(command_buffer);
    r.emplace_or_replace<EditSelectionDirty>(viewport);
}

void ApplyEditSharpness(
    entt::registry &r, entt::entity viewport, std::span<const entt::entity> mesh_entities,
    EditSharpnessOperation operation, bool value, float angle
) {
    if (mesh_entities.empty()) return;
    auto &meshes = r.ctx().get<MeshStore>();
    std::vector<EditSharpnessPushConstants> commands;
    std::vector<entt::entity> edited;
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
        const auto corners = meshes.GetFaceCornerRange(id);
        commands.emplace_back(EditSharpnessPushConstants{
            .VertexSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Vertex) : SlottedRange{},
            .EdgeSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Edge) : SlottedRange{},
            .FaceSelectionBits = uses_selection ? meshes.GetSelectionBitsRange(id, Element::Face) : SlottedRange{},
            .FaceSharpness = meshes.GetFaceSharpnessRange(id),
            .EdgeSharpness = meshes.GetEdgeSharpnessSlottedRange(id),
            .Connectivity = meshes.GetConnectivityRange(id),
            .EdgeHalfedges = meshes.GetConnectivityEdgeRange(id),
            .EdgeIndices = r.get<const MeshBuffers>(mesh_entity).EdgeIndices,
            .FaceNormals = meshes.GetBaseFaceNormalRange(id),
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
            const auto count = selection::GetElementCount(mesh, element);
            if (count > 0) selection_ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(mesh.GetStoreId(), element), count);
        }
    }
    const auto selection_transactions = BuildSelectionTransactions(
        r, selection_ranges, element, EditSelectionOperation::Derive
    );
    const auto &ctx = r.ctx().get<const mtl::Context>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    for (const auto &pc : commands) {
        encode::BindCompute(encoder, pipelines.EditSharpness, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        const uint32_t count = std::max(pc.EdgeCount, pc.FaceCount);
        encoder->dispatchThreadgroups(MTL::Size((count + 255u) / 256u, 1, 1), ThreadgroupSize::Linear256);
    }
    encoder->endEncoding();
    RecordSelectionDerive(r, command_buffer, selection_transactions);
    SubmitAndWait(command_buffer);
    for (const auto mesh_entity : edited) r.emplace_or_replace<MeshShadingDirty>(mesh_entity);
}

void RefreshElementSelectionStats(entt::registry &r, entt::entity mesh_entity) {
    if (!r.all_of<MeshElementSelection, MeshHandle>(mesh_entity)) return;
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto mesh = GetMesh(r, mesh_entity);
    const auto id = mesh.GetStoreId();
    const auto &summary = meshes.GetSelectionSummary(id);
    MeshElementSelectionStats stats{
        .SelectedCount = summary.SelectedCount,
        .SelectedVertexCount = summary.SelectedVertexCount,
        .SelectedVertexPositionSum = summary.PositionSum,
        .AnySharp = (summary.SharpnessFlags & 1u) != 0u,
        .AnySmooth = (summary.SharpnessFlags & 2u) != 0u,
    };
    r.emplace_or_replace<MeshElementSelectionStats>(mesh_entity, stats);
}

void RefreshElementSelectionSharpness(entt::registry &r, entt::entity mesh_entity) {
    auto *stats = r.try_get<MeshElementSelectionStats>(mesh_entity);
    if (!stats || !r.all_of<MeshHandle>(mesh_entity)) return;
    const auto &summary = r.ctx().get<const MeshStore>().GetSelectionSummary(r.get<const MeshHandle>(mesh_entity).StoreId);
    stats->AnySharp = (summary.SharpnessFlags & 1u) != 0u;
    stats->AnySmooth = (summary.SharpnessFlags & 2u) != 0u;
}

void PublishBoxSelectElementStats(entt::registry &r, entt::entity viewport) {
    if (!r.all_of<BoxSelectStatsDirty>(viewport)) return;
    r.remove<BoxSelectStatsDirty>(viewport);
    const profile::CpuScope scope{"RefreshSelectionStatsCpu"};
    for (const auto &range : GetElementRangesForSelected(r, viewport)) RefreshElementSelectionStats(r, range.MeshEntity);
}

void FinalizeBoxSelectElements(entt::registry &r, entt::entity viewport) {
    if (!r.all_of<BoxSelectStatsDirty>(viewport)) return;
    auto *fence = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    SubmitAndWait(fence);
    PublishBoxSelectElementStats(r, viewport);
}
