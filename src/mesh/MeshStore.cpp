#include "MeshStore.h"
#include "gpu/InstanceJob.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshPipelines.h"
#include "mesh/PageFootprint.h"
#include "metal/Dispatch.h"

#include "Profile.h"
#include "SortUnique.h"
#include "project/store/History.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include <bitset>
#include <map>

namespace {
constexpr uint32_t UniformFaceMode{uint32_t(CornerClassMode::UniformFace)};

constexpr auto NoAllocation = [](auto &) -> Range * { return nullptr; };

using Domain = MeshStore::ElementDomain;

// The word at byte Offset of an arena's value holds a handle of the record's Target set, or InvalidOffset.
// A run field holds the first of consecutive handles, with their count at byte RunCount, and a zero count is an empty run.
// A payload run field holds an offset in the record's range of its Payload pool instead.
// A field with neither holds the offset of the record's primitive material table.
struct RenderField {
    uint32_t Offset;
    std::optional<RenderSet> Target{};
    uint32_t RunCount{InvalidOffset};
    std::optional<RenderPayload> Payload{};
};
constexpr std::array MeshletFields{
    RenderField{.Offset = offsetof(MeshletRecord, TriangleOffset), .RunCount = offsetof(MeshletRecord, TriangleCount), .Payload = RenderPayload::TriangleIds},
    RenderField{.Offset = offsetof(MeshletRecord, VertexOffset), .RunCount = offsetof(MeshletRecord, VertexCount), .Payload = RenderPayload::VertexCorners},
    RenderField{.Offset = offsetof(MeshletRecord, LocalTriangleOffset), .RunCount = offsetof(MeshletRecord, TriangleCount), .Payload = RenderPayload::LocalTriangles},
    RenderField{offsetof(MeshletRecord, Primitive), RenderSet::Primitives},
    RenderField{offsetof(MeshletRecord, GroupIndex), RenderSet::Groups},
    RenderField{offsetof(MeshletRecord, RefinedGroup), RenderSet::Groups},
};
constexpr std::array SpatialNodeFields{
    RenderField{offsetof(MeshletSpatialNode, Parent), RenderSet::Meshlets},
    RenderField{offsetof(MeshletSpatialNode, Left), RenderSet::Meshlets},
    RenderField{offsetof(MeshletSpatialNode, Right), RenderSet::Meshlets},
    RenderField{offsetof(MeshletSpatialNode, Meshlet), RenderSet::Meshlets},
};
constexpr std::array GroupLinkFields{
    RenderField{.Offset = offsetof(ClusterGroupLinks, MemberOffset), .RunCount = offsetof(ClusterGroupLinks, MemberCount), .Payload = RenderPayload::GroupClusters},
    RenderField{.Offset = offsetof(ClusterGroupLinks, ProxyOffset), .RunCount = offsetof(ClusterGroupLinks, ProxyCount), .Payload = RenderPayload::GroupClusters},
};
constexpr std::array LodNodeFields{
    RenderField{offsetof(LodNode, ChildOffset), RenderSet::Nodes, offsetof(LodNode, ChildCount)},
    RenderField{offsetof(LodNode, Members), RenderSet::Members, offsetof(LodNode, MeshletCount)},
};
constexpr std::array PrimitiveFields{
    RenderField{offsetof(PrimitiveRecord, LodRootNode), RenderSet::Nodes},
    RenderField{offsetof(PrimitiveRecord, LodFinestNode), RenderSet::Nodes},
    RenderField{offsetof(PrimitiveRecord, PrimitiveMaterialOffset)},
};
constexpr std::array MeshletHandle{RenderField{0u, RenderSet::Meshlets}};
constexpr std::array NodeHandle{RenderField{0u, RenderSet::Nodes}};
constexpr std::array PrimitiveHandle{RenderField{0u, RenderSet::Primitives}};

// A mirror arena shares its ranges with a master arena earlier in the roster and allocates nothing of its own.
// Every word of a referencing arena's values is a handle of its References domain relative to the value's record, or InvalidOffset.
// A span value's two words are the first handle and the end of a run of consecutive handles.
struct ArenaInfo {
    uint32_t Bits{};
    const char *Name{};
    bool Mirror{false};
    Domain Elements{Domain::None};
    bool BlockIndexed{false};
    Domain Layer{Domain::None}; // The domain whose handles index an element layer's record ranges
    std::optional<AttributeType> Named{}; // Named-attribute type, whose ranges the attribute table stores
    Domain References{Domain::None};
    bool ReferenceSpan{false};
    std::optional<RenderSet> Render{}; // The render collection keying the arena's values, or its blocks when BlockIndexed
    std::span<const RenderField> Fields{};
    bool RenderRange{false}; // A record range of render state, which only a clone of the render records copies
    bool Tracked() const { return Name != nullptr; }
};

struct LayerInfo {
    Domain Elements;
    uint32_t Bits;
    const char *Name;
    bool Render{false};
};
// Calls f(layer, info, entries, range) for every layer, where entries(record) is zero for an absent layer and range(record) borrows its storage.
void ForEachLayer(auto &b, auto &&f) {
    using R = MeshStore::Record;
    using enum MeshStore::ChangeBits;
    f(b.FacePrimitives, LayerInfo{Domain::Face, AttributesChanged, "FacePrimitive"}, [](const R &r) { return uint32_t(r.FacePrimitivesReady); }, [](auto &r) { return &r.FacePrimitives; });
    f(b.VertexPrimitives, LayerInfo{Domain::Vertex, AttributesChanged, "VertexPrimitive"}, [](const R &r) { return uint32_t(r.VertexPrimitivesReady); }, [](auto &r) { return &r.VertexPrimitives; });
    f(b.Skin, LayerInfo{Domain::Vertex, AttributesChanged, "Skin"}, [](const R &r) { return uint32_t(r.SkinBlocksReady); }, [](auto &r) { return &r.Skin; });
    f(b.CustomNormals, LayerInfo{Domain::Halfedge, AttributesChanged, "CustomNormal"}, [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Normal) != 0u); }, [](auto &r) { return &r.CustomNormals; });
    f(b.CornerTangents, LayerInfo{Domain::Halfedge, AttributesChanged, "CornerTangent"}, [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Tangent) != 0u); }, [](auto &r) { return &r.CornerTangents; });
    f(b.CornerColors, LayerInfo{Domain::Halfedge, AttributesChanged, "CornerColor"}, [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Color0) != 0u); }, [](auto &r) { return &r.CornerColors; });
    f(b.VertexColors, LayerInfo{Domain::Vertex, AttributesChanged, "VertexColor"}, [](const R &r) { return uint32_t((r.VertexAttributes & MeshAttributeBit_Color0) != 0u); }, [](auto &r) { return &r.VertexColors; });
    constexpr std::array UvNames{"CornerUv0", "CornerUv1", "CornerUv2", "CornerUv3"};
    for (uint32_t i = 0; i < 4; ++i) {
        f(b.CornerUvs[i], LayerInfo{Domain::Halfedge, AttributesChanged, UvNames[i]}, [i](const R &r) { return uint32_t((r.CornerAttributes & (MeshAttributeBit_TexCoord0 << i)) != 0u); }, [i](auto &r) { return &r.CornerUvs[i]; });
    }
    f(b.Morph, LayerInfo{Domain::Vertex, DeformChanged, "Morph"}, [](const R &r) { return r.MorphBlocksReady ? r.MorphTargetCount : 0u; }, [](auto &r) { return &r.Morph; });
    f(b.CornerSectors, LayerInfo{Domain::Halfedge, 0u, "CornerSector"}, [](const R &r) { return uint32_t(r.CornerSectors.Count != 0u); }, [](auto &r) { return &r.CornerSectors; });
    f(b.NormalSectors, LayerInfo{Domain::Halfedge, 0u, "NormalSector"}, [](const R &r) { return uint32_t(r.NormalSectors.Count != 0u); }, [](auto &r) { return &r.NormalSectors; });
    constexpr std::array OwnerNames{"ElementMeshlets0", "ElementMeshlets1", "ElementMeshlets2"};
    for (uint32_t i = 0u; i < 3u; ++i) {
        f(b.Render.ElementMeshlets[i], LayerInfo{MeshStore::RenderDomain(i), 0u, OwnerNames[i], true}, [i](const R &r) { return uint32_t(r.ElementMeshlets[i].Count != 0u); }, [i](auto &r) { return &r.ElementMeshlets[i]; });
    }
}

// The arena roster: calls f(arena, info, allocation) for every arena, where allocation(record) borrows its sole allocation, or returns null.
// Tracked arenas come first in their history order, then the derived arenas and the mirrors, each after its master.
void ForEachArena(auto &b, auto &&f) {
    using enum MeshStore::ChangeBits;
    f(b.Vertices, ArenaInfo{GeometryChanged, "Vertices", false, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.FaceTriangles, ArenaInfo{.Bits = TopologyChanged, .Name = "FaceTriangleStart", .Elements = Domain::Face, .References = Domain::Triangle}, [](auto &e) { return &e.FaceData; });
    ForEachLayer(b, [&](auto &layer, const LayerInfo &info, auto &&, auto &&range) {
        f(layer.Values, ArenaInfo{.Bits = info.Bits, .Name = info.Name, .Layer = info.Elements, .RenderRange = info.Render}, range);
    });
    f(b.PrimitiveMaterials, ArenaInfo{AttributesChanged, "PrimitiveMaterial"}, [](auto &e) { return &e.PrimitiveMaterials; });
    f(b.VertexSelection, ArenaInfo{SelectionChanged, "VertexSelection", true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeSelection, ArenaInfo{SelectionChanged, "EdgeSelection", true, Domain::Edge, true}, NoAllocation);
    f(b.FaceSelection, ArenaInfo{SelectionChanged, "FaceSelection", true, Domain::Face, true}, NoAllocation);
    f(b.VertexHidden, ArenaInfo{SelectionChanged, "VertexHidden", true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeHidden, ArenaInfo{SelectionChanged, "EdgeHidden", true, Domain::Edge, true}, NoAllocation);
    f(b.FaceHidden, ArenaInfo{SelectionChanged, "FaceHidden", true, Domain::Face, true}, NoAllocation);
    f(b.SelectionSummary, ArenaInfo{SelectionChanged, "SelectionSummary"}, [](auto &e) { return &e.SelectionSummary; });
    f(b.Triangles, ArenaInfo{.Bits = TopologyChanged, .Name = "Triangle", .Elements = Domain::Triangle, .References = Domain::Halfedge}, [](auto &e) { return &e.TriangleData; });
    f(b.FaceCorners, ArenaInfo{.Bits = TopologyChanged, .Name = "FaceCorner", .Elements = Domain::Halfedge, .References = Domain::Vertex}, [](auto &e) { return &e.FaceCorners; });
    f(b.EdgeHalfedges, ArenaInfo{.Bits = TopologyChanged, .Name = "EdgeHalfedge", .Elements = Domain::Edge, .References = Domain::Halfedge}, [](auto &e) { return &e.EdgeData; });
    f(b.OutgoingHalfedges, ArenaInfo{.Bits = TopologyChanged, .Name = "OutgoingHalfedge", .Mirror = true, .Elements = Domain::Vertex, .References = Domain::Halfedge}, [](auto &e) { return &e.Vertices; });
    f(b.VertexCorners, ArenaInfo{TopologyChanged, "VertexCorner", true, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.OppositeHalfedges, ArenaInfo{.Bits = TopologyChanged, .Name = "OppositeHalfedge", .Mirror = true, .Elements = Domain::Halfedge, .References = Domain::Halfedge}, [](auto &e) { return &e.FaceCorners; });
    f(b.HalfedgeEdges, ArenaInfo{.Bits = TopologyChanged, .Name = "HalfedgeEdge", .Mirror = true, .Elements = Domain::Halfedge, .References = Domain::Edge}, [](auto &e) { return &e.FaceCorners; });
    f(b.HalfedgeFaces, ArenaInfo{.Bits = TopologyChanged, .Name = "HalfedgeFace", .Mirror = true, .Elements = Domain::Halfedge, .References = Domain::Face}, [](auto &e) { return &e.FaceCorners; });
    f(b.FaceRanges, ArenaInfo{.Bits = TopologyChanged, .Name = "FaceRange", .Mirror = true, .Elements = Domain::Face, .References = Domain::Halfedge, .ReferenceSpan = true}, [](auto &e) { return &e.FaceData; });
    f(b.EdgeSharpness, ArenaInfo{ShadingChanged, "EdgeSharpness", true, Domain::Edge}, [](auto &e) { return &e.EdgeData; });
    f(b.PointNormals, ArenaInfo{ShadingChanged, "PointNormal"}, [](auto &e) { return &e.PointNormals; });
    f(b.TetPositions, ArenaInfo{0, "TetPosition"}, NoAllocation);
    f(b.TetEdgeIndices, ArenaInfo{0, "TetEdgeIndex"}, NoAllocation);
    f(b.FaceSharpness, ArenaInfo{ShadingChanged, "FaceSharpness", true, Domain::Face}, [](auto &e) { return &e.FaceData; });
    f(b.SoundVertices, ArenaInfo{}, NoAllocation);
    f(b.VertexAggregates, ArenaInfo{0, nullptr, true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeAggregates, ArenaInfo{0, nullptr, true, Domain::Edge, true}, NoAllocation);
    f(b.FaceAggregates, ArenaInfo{0, nullptr, true, Domain::Face, true}, NoAllocation);
    f(b.BaseVertexNormals, ArenaInfo{0, "BaseVertexNormal", true, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.BaseFaceNormals, ArenaInfo{0, "BaseFaceNormal", true, Domain::Face}, [](auto &e) { return &e.FaceData; });
    constexpr std::array ValueNames{"AttributeFloats", "AttributeVec2s", "AttributeVec3s", "AttributeVec4s", "AttributeInts", "AttributeBools"};
    for (uint32_t i = 0u; i < AttributeTypes.size(); ++i) {
        WithAttributeValues(b.Named, AttributeTypes[i], [&](auto &values) {
            f(values, ArenaInfo{.Bits = AttributesChanged, .Name = ValueNames[i], .Named = AttributeTypes[i]}, NoAllocation);
        });
    }
    // A mirror of a render collection holds bytes only, sized by its collection's blocks.
    auto &render = b.Render;
    f(render.ExtrasFaces, ArenaInfo{0, "ExtrasFaces"}, NoAllocation);
    f(render.ExtrasEdges, ArenaInfo{0, "ExtrasEdges"}, NoAllocation);
    f(render.Meshlets, ArenaInfo{.Name = "Meshlets", .Render = RenderSet::Meshlets, .Fields = MeshletFields}, NoAllocation);
    f(render.MeshletSpatialNodes, ArenaInfo{.Name = "MeshletSpatialNodes", .Mirror = true, .Render = RenderSet::Meshlets, .Fields = SpatialNodeFields}, NoAllocation);
    f(render.MeshletLodLeaves, ArenaInfo{.Name = "MeshletLodLeaves", .Mirror = true, .Render = RenderSet::Meshlets, .Fields = NodeHandle}, NoAllocation);
    f(render.PositionDirty, ArenaInfo{.Name = "PositionDirty", .Mirror = true, .BlockIndexed = true, .Render = RenderSet::Meshlets}, NoAllocation);
    f(render.MeshletTriangleIds, ArenaInfo{.Name = "MeshletTriangleIds", .RenderRange = true}, [](auto &e) { return &e.MeshletTriangles.Values; });
    f(render.MeshletVertexCorners, ArenaInfo{.Name = "MeshletVertexCorners", .RenderRange = true}, [](auto &e) { return &e.MeshletVertices.Values; });
    f(render.MeshletLocalTriangles, ArenaInfo{.Name = "MeshletLocalTriangles", .RenderRange = true}, [](auto &e) { return &e.MeshletLocalTriangles.Values; });
    f(render.ClusterGroups, ArenaInfo{.Name = "ClusterGroups", .Render = RenderSet::Groups}, NoAllocation);
    f(render.GroupLinks, ArenaInfo{.Name = "GroupLinks", .Mirror = true, .Render = RenderSet::Groups, .Fields = GroupLinkFields}, NoAllocation);
    f(render.DirtyGroups, ArenaInfo{.Name = "DirtyGroups", .Mirror = true, .BlockIndexed = true, .Render = RenderSet::Groups}, NoAllocation);
    f(render.GroupClusterIds, ArenaInfo{.Name = "GroupClusterIds", .RenderRange = true}, [](auto &e) { return &e.GroupClusters.Values; });
    f(render.LodNodes, ArenaInfo{.Name = "LodNodes", .Render = RenderSet::Nodes, .Fields = LodNodeFields}, NoAllocation);
    f(render.LodParents, ArenaInfo{.Name = "LodParents", .Mirror = true, .Render = RenderSet::Nodes, .Fields = NodeHandle}, NoAllocation);
    f(render.LodMembers, ArenaInfo{.Name = "LodMembers", .Render = RenderSet::Members, .Fields = MeshletHandle}, NoAllocation);
    f(render.Primitives, ArenaInfo{.Name = "Primitives", .Render = RenderSet::Primitives, .Fields = PrimitiveFields}, NoAllocation);
    f(render.PrimitiveRoutes, ArenaInfo{.Name = "PrimitiveRoutes", .Fields = PrimitiveHandle, .RenderRange = true}, [](auto &e) { return &e.PrimitiveRoutes; });
    // GPU mesh records contain runtime slots, which the restore rebuilds, so they have no history.
    f(render.MeshRecords, ArenaInfo{}, NoAllocation);
}

template<typename Arena> using ArenaValue = decltype(std::declval<const Arena &>().Get(Range{}))::value_type;

constexpr bool SelectableDomain(Domain domain) { return domain == Domain::Vertex || domain == Domain::Edge || domain == Domain::Face; }

template<typename Block> Block FilledBlock(AttributeValue value) {
    Block block;
    if constexpr (std::same_as<Block, SelectionBlock>) {
        block.fill(value[0] ? ~0u : 0u);
    } else {
        typename Block::value_type element;
        std::memcpy(&element, value.data(), sizeof(element));
        block.fill(element);
    }
    return block;
}

void CaptureCopy(CloneCopies &copies, mtl::Buffer &buffer, uint64_t stride, uint64_t from, uint64_t to, uint64_t count = 1u) {
    buffer.CaptureWrite(to * stride, count * stride);
    copies.Copy(buffer, {.Source = from * stride, .Destination = to * stride, .Bytes = count * stride});
}
void ForEachPairRun(std::span<const uvec2> pairs, auto &&fn) {
    for (size_t first = 0u; first < pairs.size();) {
        auto end = first + 1u;
        while (end < pairs.size() && pairs[end].x == pairs[end - 1u].x + 1u && pairs[end].y == pairs[end - 1u].y + 1u) ++end;
        fn(pairs[first].x, pairs[first].y, uint32_t(end - first));
        first = end;
    }
}

Domain SelectionDomain(Element element) {
    if (element == Element::Vertex) return Domain::Vertex;
    if (element == Element::Edge) return Domain::Edge;
    if (element == Element::Face) return Domain::Face;
    throw std::invalid_argument("Missing selection domain.");
}
auto &SelectionArena(auto &arenas, Domain domain) {
    if (domain == Domain::Vertex) return arenas.VertexSelection;
    if (domain == Domain::Edge) return arenas.EdgeSelection;
    if (domain == Domain::Face) return arenas.FaceSelection;
    throw std::invalid_argument("Missing selection domain.");
}

auto &HiddenArena(auto &arenas, Domain domain) {
    if (domain == Domain::Vertex) return arenas.VertexHidden;
    if (domain == Domain::Edge) return arenas.EdgeHidden;
    if (domain == Domain::Face) return arenas.FaceHidden;
    throw std::invalid_argument("Missing visibility domain.");
}

uint32_t SelectableIndex(Domain domain) {
    if (domain == Domain::Vertex) return 0u;
    if (domain == Domain::Edge) return 1u;
    if (domain == Domain::Face) return 2u;
    throw std::invalid_argument("Missing selection domain.");
}
auto &AggregateArena(auto &arenas, Domain domain) {
    if (domain == Domain::Vertex) return arenas.VertexAggregates;
    if (domain == Domain::Edge) return arenas.EdgeAggregates;
    if (domain == Domain::Face) return arenas.FaceAggregates;
    throw std::invalid_argument("Missing selection domain.");
}

void ClearVertexRoots(MeshArenas &a, ElementSetRef vertices) {
    a.Vertices.ForEachBlock(vertices, [&](uint32_t block, const auto &) {
        const Range range{block * MeshElementBlockSize, MeshElementBlockSize};
        a.VertexCorners.CaptureWrite(range);
        std::ranges::fill(a.VertexCorners.GetMutable(range), uvec2{InvalidOffset, 0u});
        a.OutgoingHalfedges.CaptureWrite(range);
        std::ranges::fill(a.OutgoingHalfedges.GetMutable(range), InvalidOffset);
    });
}

// Gives `to` the render records of `from`.
void CopyRender(MeshStore::Record &to, const MeshStore::Record &from) {
    for (const auto set : RenderSets) RenderSetOf(to, set) = RenderSetOf(from, set);
    to.Meshlets = from.Meshlets;
    to.PrimitiveRoutes = from.PrimitiveRoutes;
    to.MeshletTriangles = from.MeshletTriangles;
    to.MeshletVertices = from.MeshletVertices;
    to.MeshletLocalTriangles = from.MeshletLocalTriangles;
    to.GroupClusters = from.GroupClusters;
    to.SpatialRoot = from.SpatialRoot;
    to.MeshletRevision = from.MeshletRevision;
    to.Level0Count = from.Level0Count;
    to.RenderTopologies = from.RenderTopologies;
    to.LodDepth = from.LodDepth;
    to.ElementMeshlets = from.ElementMeshlets;
    to.ElementMeshletBlockCounts = from.ElementMeshletBlockCounts;
}
void ResetRender(MeshStore::Record &r) { CopyRender(r, {}); }

bool HasRender(const MeshStore::Record &r) {
    return std::ranges::any_of(RenderSets, [&](RenderSet set) { return bool(RenderSetOf(r, set)); });
}

bool MaskAny(const BufferArena<uint32_t> &masks, const auto &arena, ElementSetRef set) {
    bool any = false;
    arena.ForEachBlock(set, [&](uint32_t b, const auto &block) { any |= b < masks.Buffer.Count<uint32_t>() && (masks.Get({b, 1u})[0] & block.Live[0]) != 0u; });
    return any;
}
} // namespace

RenderArenas::RenderArenas(mtl::BufferContext &ctx)
    : ExtrasFaces{ctx, SlotType::IndexBuffer},
      ExtrasEdges{ctx, SlotType::IndexBuffer},
      Meshlets{ctx, SlotType::Buffer},
      MeshletSpatialNodes{ctx, SlotType::Buffer},
      MeshletLodLeaves{ctx, SlotType::Buffer},
      PositionDirty{ctx, SlotType::Buffer},
      MeshletTriangleIds{ctx, SlotType::Buffer},
      MeshletVertexCorners{ctx, SlotType::Buffer},
      MeshletLocalTriangles{ctx, SlotType::Buffer},
      ClusterGroups{ctx, SlotType::Buffer},
      GroupLinks{ctx, SlotType::Buffer},
      DirtyGroups{ctx, SlotType::Buffer},
      GroupClusterIds{ctx, SlotType::Buffer},
      LodNodes{ctx, SlotType::Buffer},
      LodParents{ctx, SlotType::Buffer},
      LodMembers{ctx, SlotType::Buffer},
      Primitives{ctx, SlotType::Buffer},
      PrimitiveRoutes{ctx, SlotType::Buffer},
      MeshRecords{ctx, SlotType::Buffer},
      ElementMeshlets{{{ctx, SlotType::Buffer, InvalidOffset}, {ctx, SlotType::Buffer, InvalidOffset}, {ctx, SlotType::Buffer, InvalidOffset}}} {}

void RenderArenas::SyncMirrors() {
    MeshletSpatialNodes.Mirror({0, Meshlets.Capacity()});
    MeshletLodLeaves.Mirror({0, Meshlets.Capacity()});
    PositionDirty.Mirror({0, Meshlets.Capacity() / RenderBlockSize});
    GroupLinks.Mirror({0, ClusterGroups.Capacity()});
    DirtyGroups.Mirror({0, ClusterGroups.Capacity() / RenderBlockSize});
    LodParents.Mirror({0, LodNodes.Capacity()});
}

bool MaskHas(const BufferArena<uint32_t> &masks, ElementAddress address) {
    const auto block = address.Value() / RenderBlockSize;
    return block < masks.Buffer.Count<uint32_t>() && (masks.Get({block, 1u})[0] >> address.Value() % RenderBlockSize & 1u);
}
void MaskSet(BufferArena<uint32_t> &masks, ElementAddress address) {
    if (!MaskHas(masks, address)) masks.GetMutable({address.Value() / RenderBlockSize, 1u})[0] |= 1u << address.Value() % RenderBlockSize;
}
void MaskClear(BufferArena<uint32_t> &masks, ElementAddress address) {
    if (MaskHas(masks, address)) masks.GetMutable({address.Value() / RenderBlockSize, 1u})[0] &= ~(1u << address.Value() % RenderBlockSize);
}
bool MeshStore::HasDirtyPositions(const Record &r) const { return MaskAny(Buffers.Render.PositionDirty, Buffers.Render.Meshlets, r.MeshletData); }
bool MeshStore::HasDirtyGroups(const Record &r) const { return MaskAny(Buffers.Render.DirtyGroups, Buffers.Render.ClusterGroups, r.GroupData); }

RenderRun RenderArenas::Insert(RenderSet which, ElementSetRef &set, uint32_t count) {
    if (!count) return {};
    const auto run = WithRenderSet(*this, which, [&](auto &arena) {
        const auto handles = arena.Insert(set, count, nullptr).Handles;
        return RenderRun{ElementHandle{handles.First}, arena.Address(set, ElementHandle{handles.First}), count};
    });
    SyncMirrors();
    if (which == RenderSet::Meshlets || which == RenderSet::Groups) {
        for (uint32_t i = 0u; i < count; ++i) MaskClear(which == RenderSet::Meshlets ? PositionDirty : DirtyGroups, run.Address + i);
    }
    return run;
}

std::vector<uint32_t> RenderArenas::ListMembers(ElementSetRef &set, std::span<const uint32_t> clusters, std::span<const uint32_t> counts) {
    std::vector<uint32_t> handles(counts.size(), InvalidOffset);
    // Each list is followed by half its count of free headroom.
    uint64_t slots = 0u;
    for (const auto count : counts) slots += count + count / 2u;
    const auto run = Insert(RenderSet::Members, set, uint32_t(std::min<uint64_t>(slots, InvalidOffset)));
    if (!run.Count) return handles;
    const auto members = LodMembers.GetMutable({run.Address.Value(), run.Count});
    std::ranges::fill(members, InvalidOffset);
    for (uint32_t i = 0u, slot = 0u, first = 0u; i < counts.size(); first += counts[i], slot += counts[i] + counts[i] / 2u, ++i) {
        if (!counts[i]) continue;
        handles[i] = run.Handle.Value + slot;
        std::ranges::copy(clusters.subspan(first, counts[i]), members.begin() + slot);
    }
    return handles;
}

MeshletPayloadSlots RenderArenas::PayloadSlots() const {
    return {MeshletTriangleIds.Buffer.Slot, MeshletVertexCorners.Buffer.Slot, MeshletLocalTriangles.Buffer.Slot};
}

AttributeArenas::AttributeArenas(mtl::BufferContext &ctx)
    : Floats{ctx, SlotType::Buffer}, Vec2s{ctx, SlotType::Buffer}, Vec3s{ctx, SlotType::Buffer}, Vec4s{ctx, SlotType::Buffer}, Ints{ctx, SlotType::Buffer}, Bools{ctx, SlotType::Buffer} {}

MeshArenas::MeshArenas(mtl::BufferContext &ctx)
    : Vertices{ctx, SlotType::VertexBuffer},
      FaceTriangles{ctx, SlotType::ObjectIdBuffer},
      FaceSharpness{ctx, SlotType::Buffer},
      FaceCorners{ctx, SlotType::IndexBuffer},
      Triangles{ctx, SlotType::Buffer},
      OutgoingHalfedges{ctx, SlotType::Buffer},
      VertexCorners{ctx, SlotType::Buffer},
      VertexFans{ctx},
      OppositeHalfedges{ctx, SlotType::Buffer},
      HalfedgeEdges{ctx, SlotType::Buffer},
      HalfedgeFaces{ctx, SlotType::Buffer},
      FaceRanges{ctx, SlotType::Buffer},
      EdgeHalfedges{ctx, SlotType::Buffer},
      VertexSelection{ctx, SlotType::Buffer},
      EdgeSelection{ctx, SlotType::Buffer},
      FaceSelection{ctx, SlotType::Buffer},
      VertexHidden{ctx, SlotType::Buffer},
      EdgeHidden{ctx, SlotType::Buffer},
      FaceHidden{ctx, SlotType::Buffer},
      VertexAggregates{ctx, SlotType::Buffer}, EdgeAggregates{ctx, SlotType::Buffer}, FaceAggregates{ctx, SlotType::Buffer},
      SelectionTree{ctx},
      Query{ctx},
      SelectionSummary{ctx, SlotType::Buffer},
      EdgeSharpness{ctx, SlotType::Buffer},
      CustomNormals{ctx, SlotType::Buffer},
      PointNormals{ctx, SlotType::Buffer},
      CornerTangents{ctx, SlotType::CornerTangentBuffer},
      CornerColors{ctx, SlotType::CornerColorBuffer, vec4{1.f}},
      VertexColors{ctx, SlotType::CornerColorBuffer, vec4{1.f}},
      CornerUvs{{{ctx, SlotType::CornerUvBuffer}, {ctx, SlotType::CornerUvBuffer}, {ctx, SlotType::CornerUvBuffer}, {ctx, SlotType::CornerUvBuffer}}},
      FacePrimitives{ctx, SlotType::ElementPrimitiveBuffer},
      VertexPrimitives{ctx, SlotType::ElementPrimitiveBuffer},
      PrimitiveMaterials{ctx, SlotType::PrimitiveMaterialBuffer},
      Skin{ctx, SlotType::BoneDeformBuffer},
      Morph{ctx, SlotType::MorphTargetBuffer},
      TetPositions{ctx, SlotType::Buffer},
      TetEdgeIndices{ctx, SlotType::Buffer},
      SoundVertices{ctx, SlotType::Buffer},
      CornerSectors{ctx, SlotType::Buffer, InvalidOffset},
      NormalSectors{ctx, SlotType::Buffer},
      BaseVertexNormals{ctx, SlotType::Buffer},
      BaseFaceNormals{ctx, SlotType::Buffer},
      Named{ctx},
      Render{ctx} {}

struct MeshStore::HistoryState {
    struct Extent {
        uint64_t End;
        uint32_t Id, Bits;
    };
    enum class StreamKind { Values,
                            Blocks,
                            Sets };
    struct DomainStream {
        mtl::Buffer *Buffer;
        Domain Elements;
        uint32_t Stride, Bits;
        StreamKind Kind;
    };
    struct AttributeStream {
        mtl::Buffer *Buffer;
        uint32_t Stride;
        AttributeType Type;
    };
    struct IndexedRecord {
        std::vector<std::pair<uint32_t, uint32_t>> Sets; // Domain and set index.
        std::vector<std::pair<mtl::Buffer *, uint64_t>> Extents; // Buffer and first byte.
    };
    store::Records Entries, Free, Attributes;
    // Each non-domain arena's record extents, keyed by their disjoint first bytes.
    std::unordered_map<mtl::Buffer *, std::map<uint64_t, Extent>> Ranges;
    std::array<std::unordered_map<uint32_t, uint32_t>, uint32_t(Domain::Triangle) + 1u> Owners;
    std::vector<DomainStream> Streams;
    std::vector<AttributeStream> AttributeStreams;
    std::vector<IndexedRecord> Indexed;
    std::vector<uint32_t> Dirty;
    bool AllDirty{true};

    HistoryState(MeshStore &mesh, store::History &history) : Entries(mesh.Records), Free(mesh.FreeIds), Attributes(mesh.NamedAttributes) {
        Entries.Trie.CollectChanged = true;
        history.Track(Entries, "mesh.entries", 0);
        history.Track(Free, "mesh.free", 0);
        Attributes.Trie.CollectChanged = true;
        history.Track(Attributes, "mesh.attributes", 0);
    }

    void Unindex(uint32_t id) {
        if (id >= Indexed.size()) return;
        auto &indexed = Indexed[id];
        for (const auto [domain, set] : indexed.Sets) {
            if (const auto it = Owners[domain].find(set); it != Owners[domain].end() && it->second == id) Owners[domain].erase(it);
        }
        for (const auto [buffer, begin] : indexed.Extents) {
            auto &ranges = Ranges[buffer];
            if (const auto it = ranges.find(begin); it != ranges.end() && it->second.Id == id) ranges.erase(it);
        }
        indexed = {};
    }
    void IndexRecord(MeshStore &mesh, uint32_t id) {
        Unindex(id);
        if (id >= mesh.Records.size() || !mesh.Records[id].Alive) return;
        if (Indexed.size() <= id) Indexed.resize(id + 1u);
        auto &indexed = Indexed[id];
        ForEachArena(mesh.Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
            if (!info.Tracked() || info.Mirror || info.RenderRange) return;
            constexpr auto Stride = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
            if (const auto *allocation = ranges(mesh.Records[id])) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(*allocation)>, ElementSetRef>) {
                    if (!*allocation) return;
                    Owners[uint32_t(info.Elements)][allocation->Index] = id;
                    indexed.Sets.emplace_back(uint32_t(info.Elements), allocation->Index);
                } else if (const auto range = *allocation; range.Count) {
                    const auto begin = uint64_t(range.Offset) * Stride;
                    Ranges[&arena.Buffer][begin] = {uint64_t(range.Offset + range.Count) * Stride, id, info.Bits};
                    indexed.Extents.emplace_back(&arena.Buffer, begin);
                }
            }
        });
    }
    void Index(MeshStore &mesh) {
        if (AllDirty) {
            for (auto &[buffer, ranges] : Ranges) ranges.clear();
            for (auto &owners : Owners) owners.clear();
            Indexed.clear();
            for (uint32_t id = 0; id < mesh.Records.size(); ++id) IndexRecord(mesh, id);
        } else {
            SortUnique(Dirty);
            for (const auto id : Dirty) IndexRecord(mesh, id);
        }
        Dirty.clear();
        AllDirty = false;
    }
};

MeshStore::MeshStore(mtl::BufferContext &ctx)
    : Buffers{ctx},
      SlotTable{
          .Vertices = Buffers.Vertices.Buffer.Slot,
          .FaceTriangleStart = Buffers.FaceTriangles.Buffer.Slot,
          .FaceSharpness = Buffers.FaceSharpness.Buffer.Slot,
          .EdgeSharpness = Buffers.EdgeSharpness.Buffer.Slot,
          .PrimitiveMaterial = Buffers.PrimitiveMaterials.Buffer.Slot,
          .TetPosition = Buffers.TetPositions.Buffer.Slot,
          .TetEdgeIndex = Buffers.TetEdgeIndices.Buffer.Slot,
          .SoundVertex = Buffers.SoundVertices.Buffer.Slot,
          .BaseVertexNormal = Buffers.BaseVertexNormals.Buffer.Slot,
          .BaseFaceNormal = Buffers.BaseFaceNormals.Buffer.Slot,
      },
      BlockLists{ctx, SlotType::Buffer} {}
MeshStore::~MeshStore() {
    for (auto &kept : Kept) Release(kept.Pinned);
}

MeshStore::HistoryScope::HistoryScope(MeshStore &meshes, store::History &history) : Meshes{meshes}, Previous{std::exchange(meshes.Hist, &history)} {}
MeshStore::HistoryScope::~HistoryScope() { Meshes.Hist = Previous; }

void MeshStore::Track(store::History &history) {
    const auto first = history.Tracks.size();
    Buffers.VertexFans.Track(history, "mesh.vertexFans");
    Buffers.VertexFans.Items.Buffer.History()->Trie.CollectChanged = false;
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        // A mirror owns no allocator, so only its bytes carry history.
        if (info.Mirror) arena.Buffer.Track(history, std::string{"mesh."} + info.Name, mtl::HistoryPageBytes);
        else arena.Track(history, std::string{"mesh."} + info.Name, mtl::HistoryPageBytes);
        // Changes map to records by byte extent rather than by page.
        arena.Buffer.History()->Trie.CollectChanged = false;
        if constexpr (requires { arena.Blocks; }) {
            arena.Blocks.Buffer.History()->Trie.CollectChanged = false;
            arena.Sets.Buffer.History()->Trie.CollectChanged = false;
            arena.Table.Buffer.History()->Trie.CollectChanged = false;
        }
    });
    Tracked = std::make_unique<HistoryState>(*this, history);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        if (info.Named) Tracked->AttributeStreams.push_back({&arena.Buffer, uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>)), *info.Named});
        else if (info.Elements == Domain::None) Tracked->Ranges.try_emplace(&arena.Buffer);
        else Tracked->Streams.push_back({&arena.Buffer, info.Elements, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), info.Bits, info.BlockIndexed ? HistoryState::StreamKind::Blocks : HistoryState::StreamKind::Values});
        if constexpr (requires { arena.Blocks; }) {
            if (info.Render) return;
            Tracked->Streams.push_back({&arena.Blocks.Buffer, info.Elements, sizeof(MeshElementBlock), TopologyChanged, HistoryState::StreamKind::Blocks});
            Tracked->Streams.push_back({&arena.Sets.Buffer, info.Elements, sizeof(MeshElementSet), TopologyChanged, HistoryState::StreamKind::Sets});
        }
    });
    Hist = &history;
    for (auto i = first; i < history.Tracks.size(); ++i) {
        const auto &track = history.Tracks[i];
        TrackOwner owner{};
        if (track.K == store::History::Kind::Pages) owner.Pages = history.PageTracks[track.Index];
        else if (track.K == store::History::Kind::Records) owner.Records = history.RecordTracks[track.Index];
        else continue;
        OwnerOf.emplace(owner.Pages ? static_cast<const void *>(owner.Pages) : static_cast<const void *>(owner.Records), Owners.size());
        TrackIndices.push_back(i);
        Owners.push_back(owner);
    }
}

namespace {
void WithOwner(auto &owner, auto &&fn) {
    if (owner.Pages) fn(*owner.Pages);
    else fn(*owner.Records);
}
std::vector<std::byte> PinnedSlot(auto &owner, const store::Version &v, uint64_t slot) {
    const auto &trie = owner.Trie;
    if (!v.Root || slot >= trie.SlotsFor(v.S.Length)) return {};
    const store::Node *n = v.Root;
    for (auto level = trie.Levels; n->Kind == store::NodeKind::Interior; --level) n = n->Children[(slot / store::SlotSpan(level - 1)) % store::Fanout];
    if (n->Kind == store::NodeKind::Owned) {
        const auto bytes = owner.Encode(n->Value);
        return {bytes.begin(), bytes.end()};
    }
    if (n->Kind == store::NodeKind::Absent || !owner.Present(slot)) return {};
    const auto bytes = owner.Read(slot);
    return {bytes.begin(), bytes.end()};
}
template<typename T> std::optional<T> PageRecord(std::span<const std::byte> page, uint32_t index) {
    if (page.empty()) return std::nullopt;
    const auto mask = store::RecordMask(page);
    if (!(mask >> index & 1u)) return std::nullopt;
    for (auto bits = mask;; bits &= bits - 1u) {
        const auto value = store::TakeRecord(page);
        if (uint32_t(std::countr_zero(bits)) != index) continue;
        T record{};
        zpp::bits::in{value}(record).or_throw();
        return record;
    }
}
} // namespace

void MeshStore::Release(Version &version) {
    for (size_t k = 0; k < version.Tracks.size(); ++k) WithOwner(Owners[k], [&](auto &track) { track.Trie.Release(version.Tracks[k]); });
    version.Tracks.clear();
}

store::Hash128 MeshStore::StateHash() {
    ++Step;
    std::vector<store::Stamp> stamps;
    stamps.reserve(Owners.size());
    for (auto &owner : Owners) {
        WithOwner(owner, [&](auto &track) {
            track.Settle();
            stamps.push_back(track.Trie.CurrentStamp(track.Length()));
        });
    }
    return store::StateHash(stamps);
}

MeshStore::Version MeshStore::Pin() {
    ++Step;
    Version version;
    version.Tracks.reserve(Owners.size());
    for (auto &owner : Owners) {
        WithOwner(owner, [&](auto &track) {
            track.Settle();
            version.Tracks.push_back(track.Trie.Pin(track.Length()));
        });
    }
    return version;
}

void MeshStore::Restore(const Version &version) {
    const profile::CpuScope scope{"MeshStoreRestore"};
    for (auto &owner : Owners) WithOwner(owner, [](auto &track) { track.Trie.ExternalWritesForbidden = true; });
    for (size_t k = 0; k < Owners.size(); ++k) WithOwner(Owners[k], [&](auto &track) { (void)track.Restore(version.Tracks[k]); });
    for (auto &owner : Owners) WithOwner(owner, [](auto &track) { track.Trie.ExternalWritesForbidden = false; });
    FinishRestore();
}

MeshStore::KeptState *MeshStore::UseKept(store::Hash128 state) {
    const auto found = std::ranges::find(Kept, state, &KeptState::State);
    if (found == Kept.end()) return nullptr;
    std::rotate(found, found + 1, Kept.end());
    return &Kept.back();
}

store::Hash128 MeshStore::KeepState() {
    const auto state = StateHash();
    if (UseKept(state)) return state;
    Kept.emplace_back(state, Pin(), false);
    TrimKept();
    return state;
}

void MeshStore::TrimKept() {
    for (auto kept = Kept.begin(); Kept.size() > MaxKeptStates && kept != Kept.end();) {
        if (!kept->Recorded) {
            ++kept;
            continue;
        }
        Release(kept->Pinned);
        kept = Kept.erase(kept);
    }
}

bool MeshStore::RestoreKeptState(store::Hash128 state) {
    const auto *kept = UseKept(state);
    if (kept) Restore(kept->Pinned);
    return kept;
}

bool MeshStore::RecordKeptStates(std::span<const store::Hash128> states) {
    ++Step;
    const auto listed = [&](store::Hash128 state) { return std::ranges::binary_search(states, std::pair{state.A, state.B}, {}, [](const store::Hash128 &s) { return std::pair{s.A, s.B}; }); };
    std::vector<store::History::KeptVersions> keeps;
    for (const auto &kept : Kept) {
        if (listed(kept.State)) keeps.push_back({kept.State, kept.Pinned.Tracks});
    }
    if (!Hist || !Hist->Keep(keeps, TrackIndices)) return false;
    for (auto kept = Kept.begin(); kept != Kept.end();) {
        if (kept->Recorded || listed(kept->State)) {
            kept->Recorded = true;
            ++kept;
            continue;
        }
        Release(kept->Pinned);
        kept = Kept.erase(kept);
    }
    TrimKept();
    return true;
}

namespace {
std::vector<uint64_t> DifferingSlots(auto &owner, const store::Version &v) {
    owner.Settle();
    const auto &trie = owner.Trie;
    const auto limit = trie.SlotsFor(v.S.Length), live_limit = trie.SlotsFor(owner.Length());
    std::vector<uint64_t> slots;
    const auto live_hash = [&](uint64_t slot) {
        return slot < live_limit && slot < trie.SlotHashes.size() && trie.SlotHashes[slot].State == store::LiveTrie::SlotState::Value ? trie.SlotHashes[slot].H : store::Hash128{};
    };
    // The live slots of the range past the version's length.
    const auto added = [&](uint64_t base, uint64_t end) {
        for (auto slot = std::max(base, limit); slot < std::min(end, live_limit); ++slot) {
            if (live_hash(slot) != store::Hash128{}) slots.push_back(slot);
        }
    };
    // `present` is the node the live trie holds at the same position, or null where the live trie holds none.
    // A node the live trie also holds there has only aliased leaves.
    const auto walk = [&](this auto &&self, const store::Node *n, const store::Node *present, uint32_t level, uint64_t base) -> void {
        const auto end = base + store::SlotSpan(level);
        if (base >= std::max(limit, live_limit)) return;
        if (n == present) return added(base, end);
        switch (n->Kind) {
            case store::NodeKind::Aliased: return added(base, end);
            case store::NodeKind::Owned:
                if (base < limit && (n->Default ? store::Hash128{} : n->Hash) != live_hash(base)) slots.push_back(base);
                return;
            case store::NodeKind::Absent:
                for (auto slot = base; slot < std::min(end, live_limit); ++slot) {
                    if (live_hash(slot) != store::Hash128{}) slots.push_back(slot);
                }
                return;
            case store::NodeKind::Interior: {
                const auto *present_children = present && present->Kind == store::NodeKind::Interior ? present->Children : nullptr;
                for (uint64_t digit = 0; digit < store::Fanout; ++digit) self(n->Children[digit], present_children ? present_children[digit] : nullptr, level - 1, base + digit * store::SlotSpan(level - 1));
                return;
            }
        }
        std::unreachable();
    };
    walk(v.Root, trie.Root, trie.Levels, 0);
    return slots;
}
void ForEachDifferingExtent(std::span<const uint64_t> slots, uint64_t page_bytes, uint64_t first, uint64_t end, auto &&fn) {
    for (auto it = std::ranges::lower_bound(slots, first / page_bytes); it != slots.end() && *it * page_bytes < end; ++it) {
        fn(std::max(first, *it * page_bytes), std::min(end, (*it + 1u) * page_bytes));
    }
}
} // namespace

// The stored bytes of mesh storage in one state, read from its own pin of the state's kept version or else the history's content logs.
struct MeshStore::StoredState {
    MeshStore &Meshes;
    std::optional<Version> Pinned{};
    std::vector<store::Hash128> Roots{};
    std::optional<store::StateReader> Reader{};
    std::map<std::pair<size_t, uint64_t>, std::vector<std::byte>> Slots{};
    std::vector<std::optional<std::vector<uint64_t>>> DifferingByTrack;
    bool Failed{};

    StoredState(MeshStore &meshes, store::Hash128 state) : Meshes(meshes), DifferingByTrack(meshes.Owners.size()) {
        if (const auto *kept = meshes.UseKept(state)) {
            Pinned = kept->Pinned;
            for (auto &track : Pinned->Tracks) ++track.Root->Refs;
            return;
        }
        if (!meshes.Hist) return;
        Roots = meshes.Hist->RootsOf(state, meshes.TrackIndices);
        if (!Roots.empty()) Reader.emplace(*meshes.Hist);
    }
    StoredState(const StoredState &) = delete;
    ~StoredState() {
        if (Pinned) Meshes.Release(*Pinned);
    }
    bool Held() const { return Pinned || (Reader && Reader->Open()); }

    size_t OwnerOf(const mtl::Buffer &buffer) const { return Meshes.OwnerOf.at(buffer.History()); }
    uint64_t PageBytes(size_t k) const { return Meshes.Owners[k].Pages->PageBytes; }
    // The stored bytes of track k's slot, empty for an absent or default slot.
    const std::vector<std::byte> &Slot(size_t k, uint64_t slot) {
        const auto [it, inserted] = Slots.try_emplace({k, slot});
        if (!inserted) return it->second;
        if (Pinned) {
            WithOwner(Meshes.Owners[k], [&](auto &track) { it->second = PinnedSlot(track, Pinned->Tracks[k], slot); });
        } else if (auto bytes = Reader->Read(Meshes.TrackIndices[k], Roots[k], slot)) {
            it->second = std::move(*bytes);
        } else {
            Failed = true;
        }
        return it->second;
    }
    std::span<const uint64_t> Differing(size_t k) {
        auto &differing = DifferingByTrack[k];
        if (!differing) {
            if (Pinned) WithOwner(Meshes.Owners[k], [&](auto &track) { differing = DifferingSlots(track, Pinned->Tracks[k]); });
            else if (auto slots = Reader->DifferingSlots(Meshes.TrackIndices[k], Roots[k])) differing = std::move(*slots);
            else {
                Failed = true;
                differing.emplace();
            }
        }
        return *differing;
    }
    // The stored bytes of a tracked buffer, zero where its pages are absent.
    void Read(const mtl::Buffer &buffer, uint64_t offset, std::span<std::byte> out) {
        const auto k = OwnerOf(buffer);
        const auto page_bytes = PageBytes(k);
        for (uint64_t at = offset, end = offset + out.size(); at < end;) {
            const auto page = at / page_bytes, begin = at % page_bytes, take = std::min(page_bytes - begin, end - at);
            const auto &bytes = Slot(k, page);
            if (bytes.size() >= begin + take) std::memcpy(out.data() + (at - offset), bytes.data() + begin, take);
            else std::memset(out.data() + (at - offset), 0, take);
            at += take;
        }
    }
    template<typename T> T ReadAs(const mtl::Buffer &buffer, uint64_t index) {
        T value;
        Read(buffer, index * sizeof(T), std::as_writable_bytes(std::span{&value, 1u}));
        return value;
    }
    void Copy(const mtl::Buffer &buffer, uint64_t from, uint64_t to, uint64_t size) {
        if (size) Read(buffer, from, buffer.GetMutableRange(to, size));
    }

    // The attribute entries of each record in the stored state, and the live ones in the pages where it differs, ascending.
    // Each record's index holds while only clones and restores of records it reads change the attribute table.
    std::unordered_map<uint32_t, std::vector<uint32_t>> StoredAttributes, DifferingAttributes;
    bool AttributesIndexed{};
    void IndexAttributes() {
        if (std::exchange(AttributesIndexed, true)) return;
        const auto k = Meshes.OwnerOf.at(&Meshes.Tracked->Attributes);
        for (uint64_t page = 0u;; ++page) {
            const auto &bytes = Slot(k, page);
            if (bytes.empty()) break;
            for (uint32_t index = 0u; index < store::RecordsPerPage; ++index) {
                if (const auto entry = PageRecord<NamedAttribute>(bytes, index); entry && entry->Record != InvalidOffset) StoredAttributes[entry->Record].push_back(uint32_t(page * store::RecordsPerPage + index));
            }
        }
        for (const auto page : Differing(k)) {
            const auto end = std::min<uint64_t>((page + 1u) * store::RecordsPerPage, Meshes.NamedAttributes.size());
            for (auto attribute = page * store::RecordsPerPage; attribute < end; ++attribute) {
                if (const auto record = Meshes.NamedAttributes[attribute].Record; record != InvalidOffset) DifferingAttributes[record].push_back(uint32_t(attribute));
            }
        }
    }
    // The record's stored attribute entries.
    std::span<const uint32_t> StoredAttributesOf(uint32_t id) {
        IndexAttributes();
        const auto found = StoredAttributes.find(id);
        return found != StoredAttributes.end() ? std::span<const uint32_t>{found->second} : std::span<const uint32_t>{};
    }
    // The record's live attribute entries, which match its stored ones outside the pages where the stored state differs.
    std::vector<uint32_t> LiveAttributesOf(uint32_t id) {
        IndexAttributes();
        const auto differing = Differing(Meshes.OwnerOf.at(&Meshes.Tracked->Attributes));
        std::vector<uint32_t> live;
        for (const auto attribute : StoredAttributesOf(id)) {
            if (!std::ranges::binary_search(differing, uint64_t(attribute / store::RecordsPerPage))) live.push_back(attribute);
        }
        if (const auto found = DifferingAttributes.find(id); found != DifferingAttributes.end()) live.append_range(found->second);
        std::ranges::sort(live);
        return live;
    }
    // The stored entry of an attribute the state holds.
    NamedAttribute StoredAttribute(uint32_t attribute) {
        return *PageRecord<NamedAttribute>(Slot(Meshes.OwnerOf.at(&Meshes.Tracked->Attributes), attribute / store::RecordsPerPage), attribute % store::RecordsPerPage);
    }
};

MeshStore::StateReaders::StateReaders() = default;
MeshStore::StateReaders::StateReaders(StateReaders &&) noexcept = default;
MeshStore::StateReaders &MeshStore::StateReaders::operator=(StateReaders &&) noexcept = default;
MeshStore::StateReaders::~StateReaders() = default;

namespace {
// The readers' reader of the state, with no failed read of an earlier record.
MeshStore::StoredState &ReaderOf(MeshStore &meshes, MeshStore::StateReaders &readers, store::Hash128 state) {
    auto &held = readers.Held[state];
    if (!held) held = std::make_unique<MeshStore::StoredState>(meshes, state);
    held->Failed = false;
    return *held;
}
// Clears the render set's block-indexed masks at the blocks, which a released or cleared set leaves free.
void ClearRenderMasks(MeshArenas &buffers, RenderSet set, std::span<const uint32_t> blocks) {
    ForEachArena(buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if constexpr (std::same_as<std::remove_cvref_t<decltype(arena)>, BufferArena<uint32_t>>) {
            if (info.Render != set || !info.BlockIndexed) return;
            const auto words = arena.Buffer.template GetSpan<uint32_t>();
            for (const auto block : blocks) {
                if (words[block]) arena.GetMutable({block, 1u})[0] = 0u;
            }
        }
    });
}
} // namespace

std::optional<uint32_t> MeshStore::CloneStored(store::Hash128 state, uint32_t source, StateReaders &readers) {
    const profile::CpuScope scope{"CloneStored"};
    auto &reader = ReaderOf(*this, readers, state);
    if (!reader.Held()) return std::nullopt;
    const auto entries_owner = OwnerOf.at(&Tracked->Entries);
    const auto stored = PageRecord<Record>(reader.Slot(entries_owner, source / store::RecordsPerPage), source % store::RecordsPerPage);
    if (reader.Failed || !stored || !stored->Alive) return std::nullopt;

    struct StoredSet {
        MeshElementSet Header{};
        std::vector<uint32_t> Entries;
    };
    std::array<StoredSet, 6> sets;
    for (const auto domain : ElementDomains) {
        WithDomain(Buffers, domain, [&](const auto &arena) {
            const auto set = DomainSet(*stored, domain);
            if (!set) return;
            auto &s = sets[uint32_t(domain)];
            s.Header = reader.ReadAs<MeshElementSet>(arena.Sets.Buffer, set.Index);
            s.Entries.resize(s.Header.BlockCount);
            reader.Read(arena.Table.Buffer, uint64_t(s.Header.First) * sizeof(uint32_t), std::as_writable_bytes(std::span{s.Entries}));
        });
    }
    Record record{*stored};
    ResetRender(record);
    record.SelectionSummary = {};
    const auto id = AcquireId(std::move(record));
    DerivedRecords[id].NormalRevision = ++NextNormalRevision;
    {
        auto &dst = Records[id];
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
            const auto *from = ranges(*stored);
            if (!from || info.Mirror || info.RenderRange || &arena.Buffer == &Buffers.SelectionSummary.Buffer) return;
            auto *to = ranges(dst); // NOLINT(misc-const-correctness)
            using Allocation = std::remove_cvref_t<decltype(*from)>;
            if constexpr (std::same_as<Allocation, ElementSetRef>) {
                if constexpr (requires { arena.Blocks; }) {
                    if (!*from) return;
                    const auto &s = sets[uint32_t(info.Elements)];
                    *to = arena.CloneMembership(s.Header, s.Entries, [&](uint32_t block) { return reader.ReadAs<MeshElementBlock>(arena.Blocks.Buffer, block); });
                }
            } else if constexpr (requires { { arena.Allocate(from->Count) } -> std::same_as<Allocation>; }) {
                *to = arena.Allocate(from->Count);
            }
        });
    }
    SyncMirrors();
    const auto &dst = Records[id];
    BlockPairs pairs;
    for (const auto domain : ElementDomains) {
        WithDomain(Buffers, domain, [&](const auto &arena) {
            const auto &from = sets[uint32_t(domain)].Entries;
            const auto to = arena.Entries(DomainSet(dst, domain));
            for (uint32_t local = 0u; local < std::min<size_t>(from.size(), to.size()); ++local) {
                if (from[local] != InvalidOffset && to[local] != InvalidOffset) pairs[uint32_t(domain)].push_back({from[local], to[local]});
            }
        });
    }
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (!info.Tracked()) return;
        using Allocation = std::remove_cvref_t<decltype(*ranges(*stored))>;
        constexpr auto Stride = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
        if constexpr (std::same_as<Allocation, ElementSetRef>) {
            if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
            constexpr auto Bytes = uint64_t(MeshElementBlockSize) * Stride;
            ForEachPairRun(pairs[uint32_t(info.Elements)], [&](uint32_t from, uint32_t to, uint32_t count) { reader.Copy(arena.Buffer, from * Bytes, to * Bytes, count * Bytes); });
        } else if constexpr (std::same_as<Allocation, Range>) {
            if (info.RenderRange || &arena.Buffer == &Buffers.SelectionSummary.Buffer) return;
            const auto *from = ranges(*stored);
            if (const auto *to = ranges(dst); from && to->Count) reader.Copy(arena.Buffer, uint64_t(from->Offset) * Stride, uint64_t(to->Offset) * Stride, uint64_t(from->Count) * Stride);
        }
    });
    for (const auto domain : SelectionDomains) {
        ForEachPairRun(pairs[uint32_t(domain)], [&](uint32_t from, uint32_t to, uint32_t count) {
            reader.Copy(HiddenArena(Buffers, domain).Buffer, uint64_t(from) * sizeof(SelectionBlock), uint64_t(to) * sizeof(SelectionBlock), uint64_t(count) * sizeof(SelectionBlock));
        });
    }
    if (const auto &vertices = pairs[uint32_t(Domain::Vertex)]; !vertices.empty()) {
        std::vector<std::pair<uint32_t, std::array<uvec2, MeshElementBlockSize>>> roots;
        uint64_t fan_count = 0u;
        for (const auto [from, to] : vertices) {
            auto &block = roots.emplace_back(to, std::array<uvec2, MeshElementBlockSize>{}).second;
            reader.Read(Buffers.VertexCorners.Buffer, uint64_t(from) * MeshElementBlockSize * sizeof(uvec2), std::as_writable_bytes(std::span{block}));
            const auto live = reader.ReadAs<MeshElementBlock>(Buffers.Vertices.Blocks.Buffer, from).Live;
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot) {
                if (!(live[slot / 32u] >> (slot % 32u) & 1u)) block[slot] = {InvalidOffset, 0u};
                fan_count += block[slot].y;
            }
        }
        if (fan_count >= InvalidOffset) throw std::length_error("Copied fan items exceed their address space.");
        const auto fans = Buffers.VertexFans.Items.Allocate(uint32_t(fan_count));
        uint32_t next = fans.Offset;
        for (auto &[to, block] : roots) {
            for (auto &root : block) {
                if (!root.y) continue;
                reader.Copy(Buffers.VertexFans.Items.Buffer, uint64_t(root.x) * sizeof(uvec2), uint64_t(next) * sizeof(uvec2), uint64_t(root.y) * sizeof(uvec2));
                root.x = next;
                next += root.y;
            }
            std::ranges::copy(block, Buffers.VertexCorners.GetMutable({to * MeshElementBlockSize, MeshElementBlockSize}).begin());
        }
    }
    for (const auto attribute : reader.StoredAttributesOf(source)) {
        const auto entry = reader.StoredAttribute(attribute);
        const auto values = WithAttributeValues(Buffers.Named, entry.Type, [&](auto &arena) {
            const auto range = arena.Allocate(entry.Values.Count);
            constexpr auto Stride = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
            if (range.Count) reader.Copy(arena.Buffer, uint64_t(entry.Values.Offset) * Stride, uint64_t(range.Offset) * Stride, uint64_t(range.Count) * Stride);
            return range;
        });
        AcquireAttribute({.Record = id, .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Values = values, .Default = entry.Default});
    }
    if (reader.Failed) {
        Release(id);
        return std::nullopt;
    }
    return id;
}

std::optional<MeshStore::Change> MeshStore::RestoreRecord(store::Hash128 state, uint32_t id, StateReaders &readers, bool refreshed) {
    const profile::CpuScope scope{"RestoreRecord"};
    if (!Tracked || id >= Records.size() || !Records[id].Alive) return std::nullopt;
    auto &reader = ReaderOf(*this, readers, state);
    if (!reader.Held()) return std::nullopt;
    const auto copy_differing = [&](const mtl::Buffer &buffer, uint64_t first, uint64_t end) {
        const auto k = reader.OwnerOf(buffer);
        bool any = false;
        ForEachDifferingExtent(reader.Differing(k), reader.PageBytes(k), first, end, [&](uint64_t lo, uint64_t hi) {
            reader.Copy(buffer, lo, lo, hi - lo);
            any = true;
        });
        return any;
    };
    const auto stored = PageRecord<Record>(reader.Slot(OwnerOf.at(&Tracked->Entries), id / store::RecordsPerPage), id % store::RecordsPerPage);
    if (reader.Failed || !stored || !stored->Alive) return std::nullopt;
    auto &live = WriteRecord(id);
    Change change{id, EntryChanged};
    bool placed = false;
    const auto changed_block = [&](Domain domain, uint32_t block) {
        const auto index = domain == Domain::Halfedge ? 3u : SelectableDomain(domain) ? SelectableIndex(domain) :
                                                                                        InvalidOffset;
        if (index != InvalidOffset) change.Blocks[index].push_back(block);
    };
    const auto stored_differs = [&](const mtl::Buffer &buffer, uint64_t first, uint64_t end) {
        const auto k = reader.OwnerOf(buffer);
        const auto page = reader.PageBytes(k);
        const auto &bytes = reader.Slot(k, first / page);
        const auto at = first % page, size = end - first;
        const auto live = buffer.GetSpan<std::byte>({uint32_t(first), uint32_t(size)});
        if (bytes.size() < at + size) return std::ranges::any_of(live, [](std::byte b) { return b != std::byte{}; });
        return std::memcmp(bytes.data() + at, live.data(), size) != 0;
    };
    const auto changed_vertices = [&](uint64_t first, uint64_t end, uint32_t block, uint32_t local) {
        constexpr auto Stride = uint64_t(sizeof(Vertex));
        const auto &vertices = Buffers.Vertices.Buffer;
        auto lo = first, hi = end;
        while (lo < hi && !stored_differs(vertices, lo, lo + Stride)) lo += Stride;
        if (lo >= hi) return false;
        while (hi > lo && !stored_differs(vertices, hi - Stride, hi)) hi -= Stride;
        const auto base = uint64_t(block) * MeshElementBlockSize * Stride;
        change.VertexRanges.push_back({uint32_t(local * MeshElementBlockSize + (lo - base) / Stride), uint32_t((hi - lo) / Stride)});
        return true;
    };
    const auto for_each_stream = [&](Domain domain, std::optional<RenderSet> render, auto &&fn) {
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
            if (!info.Tracked() || info.RenderRange || info.Named || &arena.Buffer == &Buffers.VertexCorners.Buffer) return;
            if (render ? info.Render != render : info.Elements != domain || info.Render) return;
            constexpr auto Stride = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
            const uint64_t block_bytes = info.BlockIndexed ? Stride : Stride * (render ? RenderBlockSize : MeshElementBlockSize);
            fn(arena.Buffer, info.Bits, block_bytes);
        });
    };
    // Fan item copies from the stored state, each run `{from, to, count}` extended while its source and target continue.
    const auto append_items = [](std::vector<uvec3> &runs, uint32_t from, uint32_t to, uint32_t count) {
        if (!runs.empty() && runs.back().x + runs.back().z == from && runs.back().y + runs.back().z == to) runs.back().z += count;
        else runs.push_back({from, to, count});
    };
    const auto copy_items = [&](std::span<const uvec3> runs) {
        for (const auto run : runs) reader.Copy(Buffers.VertexFans.Items.Buffer, uint64_t(run.x) * sizeof(uvec2), uint64_t(run.y) * sizeof(uvec2), uint64_t(run.z) * sizeof(uvec2));
    };
    const auto restore_set = [&](auto &arena, Domain domain, std::optional<RenderSet> render, ElementSetRef sp, ElementSetRef &sc) {
        const auto membership_of = [&](uint32_t block) { return reader.ReadAs<MeshElementBlock>(arena.Blocks.Buffer, block); };
        const auto clear_blocks = [&](std::span<const uint32_t> blocks) {
            if (domain == Domain::Vertex) {
                std::vector<uvec2> released;
                for (const auto b : blocks) {
                    const auto roots = Buffers.VertexCorners.GetMutable({b * MeshElementBlockSize, MeshElementBlockSize});
                    for (auto &root : roots) {
                        if (root.y) released.push_back(root);
                        root = {InvalidOffset, 0u};
                    }
                }
                Buffers.VertexFans.Release(released);
            }
            if (SelectableDomain(domain)) {
                EditSelectionBlocks(SelectionElements[SelectableIndex(domain)], blocks, [](uint32_t, auto &words) { words = {}; });
                EditHiddenBlocks(SelectionElements[SelectableIndex(domain)], blocks, [](uint32_t, auto &words) { words = {}; });
            }
            if (render) ClearRenderMasks(Buffers, *render, blocks);
        };
        if (!sp) {
            if (!sc) return;
            std::vector<uint32_t> blocks;
            arena.ForEachBlock(sc, [&](uint32_t b, const auto &) { blocks.push_back(b); });
            std::ranges::sort(blocks);
            for (const auto b : blocks) changed_block(domain, b);
            clear_blocks(blocks);
            arena.Release(sc);
            sc = {};
            change.Bits |= TopologyChanged;
            return;
        }
        const auto header = reader.ReadAs<MeshElementSet>(arena.Sets.Buffer, sp.Index);
        std::vector<uint32_t> stored_table(header.BlockCount);
        reader.Read(arena.Table.Buffer, uint64_t(header.First) * sizeof(uint32_t), std::as_writable_bytes(std::span{stored_table}));
        // Each moved stored block and the live block it restores into, and whether each local block receives one.
        std::vector<uvec2> moved;
        std::vector<bool> receives(stored_table.size());
        const auto move = [&](uint32_t local, uint32_t to) {
            moved.push_back({stored_table[local], to});
            receives[local] = true;
        };
        if (!sc) {
            sc = arena.CloneMembership(header, stored_table, membership_of);
            placed = true;
            const auto entries = arena.Entries(sc);
            for (uint32_t local = 0u; local < stored_table.size(); ++local) {
                if (stored_table[local] != InvalidOffset) move(local, entries[local]);
            }
        } else {
            const std::vector<uint32_t> live_table{arena.Entries(sc).begin(), arena.Entries(sc).end()};
            std::vector<uint32_t> cleared, locals;
            for (uint32_t local = 0u; local < std::max(stored_table.size(), live_table.size()); ++local) {
                const auto bp = local < stored_table.size() ? stored_table[local] : InvalidOffset;
                const auto bc = local < live_table.size() ? live_table[local] : InvalidOffset;
                if (bp == InvalidOffset && bc != InvalidOffset) cleared.push_back(bc);
                else if (bp != InvalidOffset && bc == InvalidOffset) locals.push_back(local);
            }
            if (!cleared.empty()) {
                std::ranges::sort(cleared);
                clear_blocks(cleared);
                for (const auto b : arena.Clear(sc, cleared)) changed_block(domain, b);
                change.Bits |= TopologyChanged;
            }
            const auto added = arena.Place(sc, locals);
            placed = placed || !added.empty();
            for (uint32_t i = 0u; i < added.size(); ++i) move(locals[i], added[i]);
            const auto entries = arena.Entries(sc);
            for (uint32_t local = 0u; local < stored_table.size(); ++local) {
                const auto bp = stored_table[local], bc = entries[local];
                if (bp != InvalidOffset && bc != bp && !receives[local]) move(local, bc);
            }
        }
        // Consecutive moved blocks copy each stream and their metadata as one run.
        ForEachPairRun(moved, [&](uint32_t from, uint32_t to, uint32_t count) {
            for_each_stream(domain, render, [&](const mtl::Buffer &buffer, uint32_t bits, uint64_t block_bytes) {
                reader.Copy(buffer, from * block_bytes, to * block_bytes, count * block_bytes);
                change.Bits |= bits;
            });
            const auto blocks = arena.Blocks.GetMutable({to, count});
            reader.Read(arena.Blocks.Buffer, uint64_t(from) * sizeof(MeshElementBlock), std::as_writable_bytes(blocks));
            for (auto &block : blocks) block.Owner = sc.Index;
            arena.Reindex({to, count});
        });
        if (!moved.empty()) change.Bits |= TopologyChanged;
        for (const auto [_, to] : moved) changed_block(domain, to);
        if (domain == Domain::Vertex && !moved.empty()) {
            // Block by block in move order, a moved block releases its live fans and takes its stored fans' items into one run.
            std::vector<uvec2> released;
            std::vector<uvec3> item_runs;
            for (const auto [from, to] : moved) {
                released.clear();
                for (const auto root : Buffers.VertexCorners.Get({to * MeshElementBlockSize, MeshElementBlockSize})) {
                    if (root.y) released.push_back(root);
                }
                Buffers.VertexFans.Release(released);
                std::array<uvec2, MeshElementBlockSize> roots;
                reader.Read(Buffers.VertexCorners.Buffer, uint64_t(from) * MeshElementBlockSize * sizeof(uvec2), std::as_writable_bytes(std::span{roots}));
                uint32_t items_count = 0u;
                for (const auto root : roots) items_count += root.y;
                auto next = items_count ? Buffers.VertexFans.Items.Allocate(items_count).Offset : 0u;
                for (auto &root : roots) {
                    if (!root.y) continue;
                    append_items(item_runs, root.x, next, root.y);
                    root.x = std::exchange(next, next + root.y);
                }
                std::ranges::copy(roots, Buffers.VertexCorners.GetMutable({to * MeshElementBlockSize, MeshElementBlockSize}).begin());
                if (header.Count) change.VertexRanges.push_back({arena.Blocks.Get({to, 1u})[0].Local * MeshElementBlockSize, MeshElementBlockSize});
            }
            copy_items(item_runs);
        }
        const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
        const auto kept_block = [&](uint32_t b) {
            return b < members.size() && members[b].Owner == sc.Index && members[b].Local < stored_table.size() && stored_table[members[b].Local] == b && !receives[members[b].Local];
        };
        // The record's kept blocks, ascending, which the restore compares page by page where the stored state differs.
        std::vector<uint32_t> kept;
        for (const auto b : arena.Entries(sc)) {
            if (b != InvalidOffset && kept_block(b)) kept.push_back(b);
        }
        std::ranges::sort(kept);
        // Calls `fn(lo, hi)` for each part of block `b`'s bytes that lies in a page where the stored state differs.
        const auto for_each_differing = [&](const mtl::Buffer &buffer, uint32_t b, uint64_t block_bytes, auto &&fn) {
            const auto k = reader.OwnerOf(buffer);
            ForEachDifferingExtent(reader.Differing(k), reader.PageBytes(k), b * block_bytes, (b + 1u) * block_bytes, fn);
        };
        for_each_stream(domain, render, [&](const mtl::Buffer &buffer, uint32_t bits, uint64_t block_bytes) {
            for (const auto b : kept) {
                for_each_differing(buffer, b, block_bytes, [&](uint64_t lo, uint64_t hi) {
                    if (&buffer == &Buffers.Vertices.Buffer ? !changed_vertices(lo, hi, b, members[b].Local) : !stored_differs(buffer, lo, hi)) return;
                    reader.Copy(buffer, lo, lo, hi - lo);
                    change.Bits |= bits;
                    changed_block(domain, b);
                });
            }
        });
        for (const auto b : kept) {
            bool differs = false;
            for_each_differing(arena.Blocks.Buffer, b, sizeof(MeshElementBlock), [&](uint64_t, uint64_t) { differs = true; });
            if (!differs) continue;
            auto block = membership_of(b);
            block.Owner = sc.Index;
            if (block.Local == members[b].Local && block.Count == members[b].Count && block.Live == members[b].Live) continue;
            arena.Blocks.GetMutable({b, 1u})[0] = block;
            arena.Reindex({b, 1u});
            change.Bits |= TopologyChanged;
            changed_block(domain, b);
        }
        auto live_header = arena.Set(sc);
        // The set stays dense only where the live blocks holding its elements are consecutive in table order.
        const auto entries = arena.Table.Get({live_header.First, header.BlockCount});
        const auto flags = std::ranges::adjacent_find(entries, [](uint32_t a, uint32_t b) { return b != a + 1u; }) == entries.end() ? header.Flags : 0u;
        if (live_header.Count != header.Count || live_header.BlockCount != header.BlockCount || live_header.Flags != flags || (change.Bits & TopologyChanged)) {
            live_header.Count = header.Count;
            live_header.BlockCount = header.BlockCount;
            live_header.Flags = flags;
            ++live_header.Revision;
            arena.Sets.GetMutable({sc.Index, 1u})[0] = live_header;
            change.Bits |= TopologyChanged;
        }
        if (domain != Domain::Vertex) return;
        const auto corners = reader.OwnerOf(Buffers.VertexCorners.Buffer), items = reader.OwnerOf(Buffers.VertexFans.Items.Buffer);
        if (reader.Differing(corners).empty() && reader.Differing(items).empty()) return;
        // Unchanged fans restore their items where the stored state differs, and each block's rewritten fans release together and take their stored items into one run.
        std::vector<Range> unchanged;
        std::vector<uvec2> released;
        std::vector<uvec3> item_runs;
        arena.ForEachBlock(sc, [&](uint32_t b, const auto &) {
            if (!kept_block(b)) return;
            std::array<uvec2, MeshElementBlockSize> stored_roots;
            reader.Read(Buffers.VertexCorners.Buffer, uint64_t(b) * MeshElementBlockSize * sizeof(uvec2), std::as_writable_bytes(std::span{stored_roots}));
            const auto live_roots = Buffers.VertexCorners.Get({b * MeshElementBlockSize, MeshElementBlockSize});
            std::bitset<MeshElementBlockSize> rewritten;
            released.clear();
            uint32_t items_count = 0u;
            for (uint32_t i = 0u; i < MeshElementBlockSize; ++i) {
                const auto root = stored_roots[i];
                if (root == live_roots[i]) {
                    AppendRange(unchanged, {root.x, root.y});
                    continue;
                }
                if (live_roots[i].y) released.push_back(live_roots[i]);
                items_count += root.y;
                rewritten.set(i);
            }
            if (rewritten.none()) return;
            Buffers.VertexFans.Release(released);
            auto next = items_count ? Buffers.VertexFans.Items.Allocate(items_count).Offset : 0u;
            for (uint32_t i = 0u; i < MeshElementBlockSize; ++i) {
                auto &root = stored_roots[i];
                if (!rewritten[i] || !root.y) continue;
                append_items(item_runs, root.x, next, root.y);
                root.x = std::exchange(next, next + root.y);
            }
            std::ranges::copy(stored_roots, Buffers.VertexCorners.GetMutable({b * MeshElementBlockSize, MeshElementBlockSize}).begin());
            change.Bits |= TopologyChanged;
            changed_block(domain, b);
        });
        for (const auto range : unchanged) copy_differing(Buffers.VertexFans.Items.Buffer, uint64_t(range.Offset) * sizeof(uvec2), uint64_t(range.Offset + range.Count) * sizeof(uvec2));
        copy_items(item_runs);
    };
    const auto restore_domain = [&](Domain domain) { WithDomain(Buffers, domain, [&](auto &arena) { restore_set(arena, domain, std::nullopt, DomainSet(*stored, domain), DomainSet(live, domain)); }); };
    for (const auto domain : ElementDomains) {
        if (domain != Domain::Triangle) restore_domain(domain);
    }
    // Whether the stored values of a primitive range differ from the live ones.
    const auto primitives_differ = [&](const auto &arena, Range from, Range to) {
        constexpr auto Stride = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
        if (from.Offset != to.Offset || from.Count != to.Count) return true;
        const auto k = reader.OwnerOf(arena.Buffer);
        bool differs = false;
        ForEachDifferingExtent(reader.Differing(k), reader.PageBytes(k), from.Offset * Stride, uint64_t(from.Offset + from.Count) * Stride, [&](uint64_t lo, uint64_t hi) { differs = differs || stored_differs(arena.Buffer, lo, hi); });
        return differs;
    };
    // The live triangles and render records stay where the stored record shares the live topology and everything its render records read but positions.
    const bool live_render = refreshed && live.RenderTopologies != 0u && !(change.Bits & TopologyChanged) && stored->Classification == live.Classification && stored->CornerAttributes == live.CornerAttributes &&
        !primitives_differ(Buffers.FacePrimitives.Values, stored->FacePrimitives, live.FacePrimitives) && !primitives_differ(Buffers.VertexPrimitives.Values, stored->VertexPrimitives, live.VertexPrimitives) &&
        !primitives_differ(Buffers.PrimitiveMaterials, stored->PrimitiveMaterials, live.PrimitiveMaterials);
    change.LiveRender = live_render;
    if (!live_render) {
        restore_domain(Domain::Triangle);
        for (const auto set : RenderSets) {
            WithRenderSet(Buffers.Render, set, [&](auto &arena) { restore_set(arena, Domain::None, set, RenderSetOf(*stored, set), RenderSetOf(live, set)); });
        }
    }
    const auto restore_range = [&](auto &arena, Range from, Range &to, uint32_t count) {
        constexpr auto Stride = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
        if (!from.Count) {
            if (!to.Count) return false;
            arena.Release(to);
            to = {};
            return true;
        }
        if (to.Count && to.Offset == from.Offset) {
            if (to.Count > from.Count) arena.Shrink(to, from.Count);
            else if (to.Count < from.Count && !arena.Extend(to, from.Count - to.Count)) {
                arena.Release(to);
                to = {};
            }
        } else if (to.Count) {
            arena.Release(to);
            to = {};
        }
        if (!to.Count) to = arena.Reserve(from) ? from : arena.Allocate(from.Count);
        if (to.Offset == from.Offset) return copy_differing(arena.Buffer, uint64_t(from.Offset) * Stride, uint64_t(from.Offset + count) * Stride);
        reader.Copy(arena.Buffer, uint64_t(from.Offset) * Stride, uint64_t(to.Offset) * Stride, uint64_t(count) * Stride);
        return true;
    };
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&, auto &&range) {
        if (live_render && info.Render) return;
        if (restore_range(layer.Values, *range(*stored), *range(live), range(*stored)->Count)) change.Bits |= info.Bits | AttributesChanged;
    });
    const auto record_range = [&](const ArenaInfo &info) { return info.Tracked() && info.Elements == Domain::None && !info.Render && !info.Named && info.Layer == Domain::None; };
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (!record_range(info) || (live_render && info.RenderRange)) return;
        if constexpr (std::same_as<std::remove_cvref_t<decltype(*ranges(*stored))>, Range> && requires { arena.Reserve(Range{}); }) {
            if (!ranges(*stored) || &arena.Buffer == &Buffers.Render.MeshletTriangleIds.Buffer || &arena.Buffer == &Buffers.Render.MeshletVertexCorners.Buffer || &arena.Buffer == &Buffers.Render.MeshletLocalTriangles.Buffer || &arena.Buffer == &Buffers.Render.GroupClusterIds.Buffer) return;
            if (restore_range(arena, *ranges(*stored), *ranges(live), ranges(*stored)->Count)) change.Bits |= info.Bits ? info.Bits : EntryChanged;
        }
    });
    for (const auto payload : live_render ? std::span<const RenderPayload>{} : std::span{RenderPayloads}) {
        WithPayload(Buffers.Render, *stored, payload, [&](auto &pool, const PayloadRange &from) {
            WithPayload(Buffers.Render, live, payload, [&](auto &, PayloadRange &to) {
                Range capacity{to.Values.Offset, to.Capacity};
                restore_range(pool, {from.Values.Offset, from.Capacity}, capacity, from.Values.Count);
                to = {{capacity.Offset, from.Values.Count}, capacity.Count};
            });
        });
    }
    std::vector<std::pair<uint32_t, NamedAttribute>> live_attributes;
    for (const auto attribute : reader.LiveAttributesOf(id)) live_attributes.emplace_back(attribute, NamedAttributes[attribute]);
    std::vector<bool> matched(live_attributes.size());
    for (const auto attribute : reader.StoredAttributesOf(id)) {
        const auto entry = reader.StoredAttribute(attribute);
        const auto found = std::ranges::find_if(live_attributes, [&](const auto &candidate) { return candidate.second.NameView() == entry.NameView(); });
        const bool reuse = found != live_attributes.end() && found->second.Domain == entry.Domain && found->second.Type == entry.Type;
        if (found != live_attributes.end()) matched[size_t(found - live_attributes.begin())] = true;
        if (found != live_attributes.end() && !reuse) {
            ReleaseAttribute(found->first);
            found->second = {};
        }
        Range values = reuse ? found->second.Values : Range{};
        WithAttributeValues(Buffers.Named, entry.Type, [&](auto &arena) { restore_range(arena, entry.Values, values, entry.Values.Count); });
        NamedAttribute restored{.Record = id, .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Values = values, .Default = entry.Default};
        if (reuse) {
            if (found->second.Values.Offset != values.Offset || found->second.Values.Count != values.Count || found->second.Default != entry.Default) WriteAttribute(found->first) = restored;
        } else {
            AcquireAttribute(std::move(restored));
        }
        change.Bits |= AttributesChanged;
    }
    for (uint32_t i = 0u; i < live_attributes.size(); ++i) {
        if (!matched[i]) {
            ReleaseAttribute(live_attributes[i].first);
            change.Bits |= AttributesChanged;
        }
    }
    if (reader.Failed) throw std::runtime_error("The recorded mesh state lost pages its record needs.");
    {
        Record merged{*stored};
        merged.StoreId = id;
        for (const auto domain : ElementDomains) DomainSet(merged, domain) = DomainSet(live, domain);
        for (const auto set : RenderSets) RenderSetOf(merged, set) = RenderSetOf(live, set);
        ForEachLayer(Buffers, [&](auto &, const LayerInfo &, auto &&, auto &&range) { *range(merged) = *range(live); });
        ForEachArena(Buffers, [&](auto &, const ArenaInfo &info, auto &&ranges) {
            if (!record_range(info)) return;
            if constexpr (std::same_as<std::remove_cvref_t<decltype(*ranges(*stored))>, Range>) {
                if (ranges(*stored)) *ranges(merged) = *ranges(live);
            }
        });
        for (const auto payload : RenderPayloads) WithPayload(Buffers.Render, merged, payload, [&](auto &, PayloadRange &range) { WithPayload(Buffers.Render, live, payload, [&](auto &, const PayloadRange &from) { range = from; }); });
        merged.ExtrasFaces = live.ExtrasFaces;
        merged.ExtrasEdges = live.ExtrasEdges;
        if (live_render) CopyRender(merged, live);
        live = merged;
    }
    if (placed) SyncMirrors();
    DerivedRecords[id] = {.NormalRevision = ++NextNormalRevision, .CreatedStep = DerivedRecords[id].CreatedStep};
    ReleaseBlockLists(id);
    for (auto &blocks : change.Blocks) SortUnique(blocks);
    std::ranges::sort(change.VertexRanges, {}, &Range::Offset);
    return change;
}

namespace {
void CaptureRange(const auto &arena, Range range) {
    using Value = ArenaValue<std::remove_cvref_t<decltype(arena)>>;
    arena.Buffer.CaptureWrite(uint64_t(range.Offset) * sizeof(Value), uint64_t(range.Count) * sizeof(Value));
}
void ForEachBlockRun(const auto &elements, ElementSetRef set, auto &&fn) {
    constexpr auto Size = std::remove_cvref_t<decltype(elements)>::BlockSize;
    Range run{};
    elements.ForEachBlock(set, [&](uint32_t block, const auto &) {
        if (run.Count && block * Size != run.Offset + run.Count) {
            fn(run);
            run = {};
        }
        if (!run.Count) run.Offset = block * Size;
        run.Count += Size;
    });
    if (run.Count) fn(run);
}
void CaptureElementSet(const auto &values, const auto &elements, ElementSetRef set) {
    ForEachBlockRun(elements, set, [&](Range run) { CaptureRange(values, run); });
}
void CaptureRange(const auto &arena, ElementSetRef set) { CaptureElementSet(arena, arena, set); }
// Writes zero bytes over the range's values.
void ZeroRange(auto &arena, Range range) {
    if (!range.Count) return;
    const auto values = arena.GetMutable(range);
    std::memset(static_cast<void *>(values.data()), 0, values.size_bytes());
}
// Writes zero over the values of those of the ascending blocks that are free, in every tracked arena of the domain.
void ZeroFreeBlocks(auto &buffers, MeshStore::ElementDomain domain, std::span<const uint32_t> blocks) {
    const auto free = WithDomain(buffers, domain, [&](const auto &arena) {
        return blocks | std::views::filter([&](uint32_t b) { return arena.Blocks.Get({b, 1u})[0].Owner == InvalidOffset; }) | std::ranges::to<std::vector>();
    });
    ForEachArena(buffers, [&](auto &values, const ArenaInfo &info, auto &&) {
        if (info.Elements != domain || info.BlockIndexed || !info.Tracked()) return;
        ForEachIndexRun(free, [&](size_t first, size_t count) { ZeroRange(values, {free[first] * MeshElementBlockSize, uint32_t(count) * MeshElementBlockSize}); });
    });
}
} // namespace

void MeshStore::CaptureSelectionSummary(uint32_t id) {
    if (Tracked) CaptureRange(Buffers.SelectionSummary, Records.at(id).SelectionSummary);
}

void MeshStore::CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation operation, const GeometrySelection &selection, std::span<const uint32_t> vertex_edges) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    const auto addresses = [&](Domain domain, auto &&handles) {
        std::vector<uint32_t> result;
        result.reserve(std::ranges::size(handles));
        for (const auto handle : handles) result.push_back(HandleAddress(id, domain, ElementHandle{handle}).Value());
        return result;
    };
    switch (operation) {
        case EditSharpnessOperation::SetSelectedFaces:
            Buffers.FaceSharpness.Buffer.CaptureWriteElements(addresses(Domain::Face, selection.Faces), sizeof(uint8_t));
            break;
        case EditSharpnessOperation::SetSelectedEdges:
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(addresses(Domain::Edge, selection.Edges), sizeof(uint8_t));
            break;
        case EditSharpnessOperation::SetVertexEdges:
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(addresses(Domain::Edge, vertex_edges), sizeof(uint8_t));
            break;
        case EditSharpnessOperation::SetAllFaces:
            CaptureElementSet(Buffers.FaceSharpness, Buffers.FaceTriangles, record.FaceData);
            break;
        case EditSharpnessOperation::SmoothAll:
        case EditSharpnessOperation::SmoothByAngle:
            CaptureElementSet(Buffers.FaceSharpness, Buffers.FaceTriangles, record.FaceData);
            CaptureElementSet(Buffers.EdgeSharpness, Buffers.EdgeHalfedges, record.EdgeData);
            break;
    }
}

void MeshStore::CaptureConnectivityWrite(uint32_t id) {
    const auto &record = Records.at(id);
    CaptureRange(Buffers.OutgoingHalfedges, Buffers.Vertices.Dense(record.Vertices));
    CaptureRange(Buffers.VertexCorners, Buffers.Vertices.Dense(record.Vertices));
    CaptureRange(Buffers.OppositeHalfedges, Buffers.FaceCorners.Dense(record.FaceCorners));
    CaptureRange(Buffers.HalfedgeEdges, Buffers.FaceCorners.Dense(record.FaceCorners));
    CaptureRange(Buffers.HalfedgeFaces, Buffers.FaceCorners.Dense(record.FaceCorners));
    CaptureRange(Buffers.FaceRanges, Buffers.FaceTriangles.Dense(record.FaceData));
    CaptureRange(Buffers.FaceTriangles, record.FaceData);
    CaptureRange(Buffers.Triangles, record.TriangleData);
    CaptureRange(Buffers.EdgeHalfedges, record.EdgeData);
}

void MeshStore::CaptureWeldWrite(uint32_t id) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    CaptureRange(Buffers.Vertices, record.Vertices);
    CaptureRange(Buffers.FaceCorners, record.FaceCorners);
    const Range vertices{0u, Buffers.Vertices.Count(record.Vertices)};
    Buffers.Skin.Capture(record.Skin, vertices);
    Buffers.Morph.Capture(record.Morph, vertices, record.MorphTargetCount);
}

MeshStore::Record &MeshStore::WriteRecord(uint32_t id) {
    if (Tracked) {
        Tracked->Entries.Write(id, 1);
        Tracked->Dirty.push_back(id);
    }
    return Records.at(id);
}

void MeshStore::IndexHistory() {
    if (!Tracked || (!Tracked->AllDirty && Tracked->Dirty.empty())) return;
    const profile::CpuScope scope{"MeshStoreIndexHistory"};
    Tracked->Index(*this);
}

std::vector<MeshStore::Change> MeshStore::TakeChanges() {
    if (!Tracked) return {};
    // Fan item changes map to their mesh through its changed canonical roots, so the item track's extents only drain.
    Buffers.VertexFans.Items.Buffer.History()->TakeChangedExtents();
    // A table entry changes only with its set's header, which maps the change to its record.
    for (const auto domain : ElementDomains) WithDomain(Buffers, domain, [](auto &arena) { arena.Table.Buffer.History()->TakeChangedExtents(); });
    for (const auto set : RenderSets) {
        WithRenderSet(Buffers.Render, set, [](auto &arena) {
            arena.Table.Buffer.History()->TakeChangedExtents();
            arena.Sets.Buffer.History()->TakeChangedExtents();
            for (const auto [begin, end] : arena.Blocks.Buffer.History()->TakeChangedExtents()) {
                arena.Reindex({uint32_t(begin / sizeof(MeshElementBlock)), uint32_t((end + sizeof(MeshElementBlock) - 1u) / sizeof(MeshElementBlock) - begin / sizeof(MeshElementBlock))});
            }
        });
    }
    IndexHistory();
    struct ChangedRange {
        uint32_t Id, Bits;
        Range Vertices{};
        uint32_t Domain{InvalidOffset}, Block{}; // A changed block in the order of Change::Blocks.
    };
    std::vector<ChangedRange> changed;
    for (const auto id : Tracked->Entries.TakeChanged()) changed.push_back({id, EntryChanged});
    for (const auto &[buffer, ranges] : Tracked->Ranges) {
        // Byte extents keep a record whose neighbor shares a page out of the changes.
        for (const auto [begin, end] : buffer->History()->TakeChangedExtents()) {
            auto it = ranges.upper_bound(begin);
            if (it != ranges.begin() && std::prev(it)->second.End > begin) --it;
            for (; it != ranges.end() && it->first < end; ++it) changed.push_back({it->second.Id, it->second.Bits, {}});
        }
    }
    for (const auto &stream : Tracked->Streams) {
        const auto &owners = Tracked->Owners[uint32_t(stream.Elements)];
        for (const auto [begin, end] : stream.Buffer->History()->TakeChangedExtents()) {
            const uint64_t first = begin / stream.Stride, last = (end + stream.Stride - 1u) / stream.Stride;
            WithDomain(Buffers, stream.Elements, [&](auto &arena) {
                const auto blocks = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
                const auto values = stream.Kind == HistoryState::StreamKind::Values;
                const uint64_t first_block = values ? first / MeshElementBlockSize : first;
                const uint64_t last_block = values ? (last + MeshElementBlockSize - 1u) / MeshElementBlockSize : last;
                if (stream.Buffer == &arena.Blocks.Buffer) arena.Reindex({uint32_t(first_block), uint32_t(last_block - first_block)});
                for (uint64_t block = first_block; block < last_block; ++block) {
                    const auto root = stream.Kind == HistoryState::StreamKind::Sets ? uint32_t(block) : block < blocks.size() ? blocks[block].Owner :
                                                                                                                                InvalidOffset;
                    const auto it = owners.find(root);
                    if (stream.Kind != HistoryState::StreamKind::Sets && SelectableDomain(stream.Elements)) {
                        const auto domain = SelectableIndex(stream.Elements);
                        const auto previous = Buffers.SelectionTree.Owner(domain, uint32_t(block));
                        if (previous != InvalidOffset && (it == owners.end() || previous / 3u != it->second)) {
                            changed.push_back({previous / 3u, stream.Bits, {}, domain, uint32_t(block)});
                        }
                    }
                    if (it == owners.end()) continue;
                    Range vertices{};
                    if (stream.Bits == GeometryChanged) {
                        const auto start = std::max(first, block * MeshElementBlockSize);
                        const auto stop = std::min(last, (block + 1u) * MeshElementBlockSize);
                        vertices = {blocks[block].Local * MeshElementBlockSize + uint32_t(start % MeshElementBlockSize), uint32_t(stop - start)};
                    }
                    const auto domain = stream.Kind == HistoryState::StreamKind::Sets ? InvalidOffset :
                        stream.Elements == Domain::Halfedge                           ? 3u :
                        SelectableDomain(stream.Elements)                             ? SelectableIndex(stream.Elements) :
                                                                                        InvalidOffset;
                    changed.push_back({it->second, stream.Bits, vertices, domain, uint32_t(block)});
                }
            });
        }
    }
    for (const auto &stream : Tracked->AttributeStreams) {
        for (const auto [begin, end] : stream.Buffer->History()->TakeChangedExtents()) {
            const auto first = begin / stream.Stride, last = (end + stream.Stride - 1u) / stream.Stride;
            for (const auto &entry : NamedAttributes) {
                if (entry.Record != InvalidOffset && entry.Type == stream.Type && entry.Values.Offset < last && first < entry.Values.Offset + entry.Values.Count) changed.push_back({entry.Record, AttributesChanged});
            }
        }
    }
    std::ranges::sort(changed, [](const auto &a, const auto &b) { return std::pair{a.Id, a.Vertices.Offset} < std::pair{b.Id, b.Vertices.Offset}; });
    std::vector<Change> changes;
    for (const auto &range : changed) {
        if (changes.empty() || changes.back().StoreId != range.Id) changes.push_back({range.Id, 0});
        auto &change = changes.back();
        change.Bits |= range.Bits;
        if (range.Domain != InvalidOffset) change.Blocks[range.Domain].push_back(range.Block);
        change.VertexRanges.push_back(range.Vertices);
    }
    for (auto &change : changes) {
        for (auto &blocks : change.Blocks) SortUnique(blocks);
        CoalesceRanges(change.VertexRanges);
    }
    // A restore can return a set to an earlier revision with different membership, so each changed record's lists refill on their next read.
    std::vector<uint32_t> ids;
    ids.reserve(changes.size());
    for (const auto &change : changes) ids.push_back(change.StoreId);
    ReleaseBlockLists(ids);
    return changes;
}

void MeshStore::SyncMirrors() {
    const profile::CpuScope scope{"MeshStoreSyncMirrors"};
    Buffers.Query.Reserve(std::max({Buffers.Vertices.Capacity(), Buffers.EdgeHalfedges.Capacity(), Buffers.FaceTriangles.Capacity()}));
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Mirror || info.Elements == Domain::None) return;
        WithDomain(Buffers, info.Elements, [&](const auto &owner) {
            arena.Mirror({0, info.BlockIndexed ? owner.Capacity() / MeshElementBlockSize : owner.Capacity()});
        });
    });
    Buffers.Render.MeshRecords.Mirror({0, uint32_t(Records.size())});
    Buffers.Render.SyncMirrors();
}

void MeshStore::FinishRestore() {
    ++Step;
    const auto changed = Tracked->Entries.Changed();
    Tracked->Dirty.append_range(changed);
    DerivedRecords.resize(Records.size());
    for (const auto id : changed) {
        if (id < Records.size()) {
            DerivedRecords[id] = {};
            if (Records[id].Alive) DerivedRecords[id].NormalRevision = ++NextNormalRevision;
        }
    }
    if (!changed.empty()) SyncMirrors();
    for (const auto attribute : Tracked->Attributes.TakeChanged()) IndexAttribute(attribute);
    IndexedAttributes.resize(NamedAttributes.size());
}

void MeshStore::FillBaseVertexNormalMirror(ElementSetRef vertices, Range point_normals) {
    if (point_normals.Count > 0) {
        std::ranges::copy(Buffers.PointNormals.Get(point_normals), Buffers.BaseVertexNormals.GetMutable(Buffers.Vertices.Dense(vertices)).begin());
    } else {
        ForEachBlockRun(Buffers.Vertices, vertices, [&](Range run) { std::ranges::fill(Buffers.BaseVertexNormals.GetMutable(run), vec3{0}); });
    }
}

ElementHandleRange MeshStore::InsertElements(uint32_t id, ElementDomain domain, uint32_t count, BufferArena<uint32_t> *list) {
    if (!Records.at(id).Alive || domain == Domain::None) throw std::invalid_argument("Invalid element insertion request.");
    if (!count) return {};
    const auto set = DomainSet(Records[id], domain);
    const bool first_face = domain == Domain::Face && !Buffers.FaceTriangles.Count(set);
    auto &record = !set || first_face || domain == Domain::Triangle ? WriteRecord(id) : Records[id];
    auto inserted = WithDomain(Buffers, domain, [&](auto &arena) {
        return arena.Insert(DomainSet(record, domain), count, list);
    });
    auto &gains = inserted.Blocks;
    std::ranges::sort(gains, {}, &ElementBlockGain::Block);
    std::vector<uint32_t> blocks;
    blocks.reserve(gains.size());
    for (const auto &gain : gains) blocks.push_back(gain.Block);
    SyncMirrors();
    if (first_face) {
        record.FacePrimitivesReady = true;
        record.ConnectivityFaceStarts = true;
    }
    std::vector<uint32_t> fresh;
    const auto bound = WithDomain(Buffers, domain, [&](const auto &arena) {
        for (const auto &gain : gains) {
            if (const auto &block = arena.Blocks.Get({gain.Block, 1u})[0]; block.Live == gain.Added) fresh.push_back(block.Local);
        }
        return arena.Set(DomainSet(record, domain)).BlockCount;
    });
    std::ranges::sort(fresh);
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
        const auto per = entries(record);
        if (info.Elements != domain || !per) return;
        auto covered = *range(record);
        if (layer.Cover(covered, bound, per)) *range(WriteRecord(id)) = covered;
        ForEachIndexRun(fresh, [&](size_t first, size_t n) { layer.Clear(covered, {fresh[first], uint32_t(n)}, per); });
    });
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        if (ElementDomainOf(entry.Domain) != domain) return;
        WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) {
            auto covered = entry.Values;
            if (CoverRange(values, covered, bound)) WriteAttribute(attribute).Values = covered;
            const auto filled = FilledBlock<ArenaValue<std::remove_cvref_t<decltype(values)>>>(entry.Default);
            ForEachIndexRun(fresh, [&](size_t first, size_t n) { FillRange(values, covered, {fresh[first], uint32_t(n)}, filled); });
        });
    });
    // The emitter writes the gained blocks and their mirrors, including holes sharing pages with surviving elements.
    ForEachArena(Buffers, [&](auto &values, const ArenaInfo &info, auto &&) {
        if (info.Elements != domain || (info.BlockIndexed && !info.Mirror)) return;
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            const Range run{blocks[first], uint32_t(count)};
            CaptureRange(values, info.BlockIndexed ? run : Range{run.Offset * MeshElementBlockSize, run.Count * MeshElementBlockSize});
        });
    });
    if (SelectableDomain(domain)) {
        auto &hidden = HiddenArena(Buffers, domain);
        for (const auto &gain : gains) {
            auto &words = hidden.GetMutable({gain.Block, 1u})[0];
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[w] &= ~gain.Added[w];
        }
    }
    if (domain == Domain::Triangle) record.TriangleCount = Buffers.Triangles.Count(record.TriangleData);
    return inserted.Handles;
}

std::vector<uint32_t> MeshStore::EraseElements(uint32_t id, ElementDomain domain, const BufferArena<uint32_t> &storage, ElementWork work) {
    if (!Records.at(id).Alive) throw std::invalid_argument("Retired mesh.");
    if (domain == Domain::Triangle) WriteRecord(id);
    const auto set = DomainSet(Records[id], domain);
    auto blocks = WithDomain(Buffers, domain, [&](auto &arena) { return arena.Erase(set, storage.Buffer, work); });
    if (SelectableDomain(domain) && !blocks.empty()) {
        WithDomain(Buffers, domain, [&](const auto &arena) {
            EditSelectionBlocks(SelectionElements[SelectableIndex(domain)], blocks, [&](uint32_t block, auto &words) {
                const auto &live = arena.Blocks.Get({block, 1u})[0].Live;
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[w] &= live[w];
            });
        });
    }
    FinishEraseElements(id, domain, blocks);
    return blocks;
}

void MeshStore::TrimInsertedElements(uint32_t id, ElementDomain domain, ElementHandleRange &inserted, const BufferArena<uint32_t> &list, uint32_t used) {
    if (used > inserted.Count) throw std::logic_error("An insertion's elements are fewer than the ones used.");
    if (used == inserted.Count) return;
    auto &record = WriteRecord(id);
    const auto blocks = WithDomain(Buffers, domain, [&](auto &arena) { return arena.Shrink(DomainSet(record, domain), inserted, list.Buffer.GetSpan<uint32_t>(), used); });
    FinishEraseElements(id, domain, blocks);
}

void MeshStore::FinishEraseElements(uint32_t id, ElementDomain domain, std::span<const uint32_t> blocks) {
    if (blocks.empty()) return;
    const auto set = DomainSet(Records[id], domain);
    std::vector<uint32_t> left;
    WithDomain(Buffers, domain, [&](const auto &arena) {
        for (const auto b : blocks) {
            if (!set || arena.Blocks.Get({b, 1})[0].Owner != set.Index) left.push_back(b);
        }
        if (!SelectableDomain(domain)) return;
        const auto element = SelectionElements[SelectableIndex(domain)];
        EditHiddenBlocks(element, blocks, [&](uint32_t block, auto &words) {
            const auto &member = arena.Blocks.Get({block, 1u})[0];
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[w] &= member.Owner == set.Index ? member.Live[w] : 0u;
        });
        if (!Records[id].SelectionSummary.Count) return;
        const auto &summary = GetSelectionSummary(id);
        if (summary.Mode == element && summary.ActiveHandle != InvalidOffset && !arena.Contains(set, ElementHandle{summary.ActiveHandle})) WriteSelectionSummary(id).ActiveHandle = InvalidOffset;
    });
    // Storage that leaves a record the step created is in no state, so it returns to zero.
    const bool created = CreatedInStep(id);
    if (domain == Domain::Triangle) Records[id].TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
    // Posed sector normals key their blocks by corner address, so corner blocks leaving a record with sectors renew its pose layout.
    if (domain == Domain::Halfedge && !left.empty() && Records[id].CornerSectors.Count) DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
    if (domain == Domain::Vertex) {
        std::vector<uvec2> released;
        for (const auto b : blocks) {
            const auto &block = Buffers.Vertices.Blocks.Get({b, 1u})[0];
            const auto live = block.Owner == set.Index ? block.Live : decltype(block.Live){};
            const auto roots = Buffers.VertexCorners.GetMutable({b * 256u, 256u});
            for (uint32_t i = 0u; i < 256u; ++i) {
                if (live[i / 32u] & (1u << (i % 32u))) continue;
                if (roots[i].y) released.push_back(roots[i]);
                roots[i] = {};
            }
        }
        if (created) {
            for (const auto root : released) ZeroRange(Buffers.VertexFans.Items, {root.x, root.y});
        }
        Buffers.VertexFans.Release(released);
    }
    if (created) ZeroFreeBlocks(Buffers, domain, left);
}

uint32_t MeshStore::GetCornerClassMode(uint32_t id) const {
    return Records.at(id).Classification;
}

std::span<uint32_t> MeshStore::EditPrimitiveMaterials(uint32_t id) { return Buffers.PrimitiveMaterials.GetMutable(Records.at(id).PrimitiveMaterials); }
std::span<uint8_t> MeshStore::EditFaceSharpness(uint32_t id) { return Buffers.FaceSharpness.GetMutable(Buffers.FaceTriangles.Dense(Records.at(id).FaceData)); }
std::span<uint8_t> MeshStore::EditEdgeSharpness(uint32_t id) { return Buffers.EdgeSharpness.GetMutable(Buffers.EdgeHalfedges.Dense(Records.at(id).EdgeData)); }

void MeshStore::SetCustomCornerNormals(uint32_t id, std::span<const CustomNormal> offsets) {
    auto &record = WriteRecord(id);
    const auto corners = Buffers.FaceCorners.Dense(record.FaceCorners);
    if (offsets.size() != corners.Count) throw std::invalid_argument("Custom normal count differs from polygon corners.");
    if (std::ranges::any_of(offsets, [](CustomNormal value) { return value.Offset.x >= 0.f; })) {
        Buffers.CustomNormals.Initialize(record.CustomNormals, corners.Count, offsets);
        record.CornerAttributes |= MeshAttributeBit_Normal;
    } else {
        Buffers.CustomNormals.Release(record.CustomNormals);
        record.CornerAttributes &= ~MeshAttributeBit_Normal;
    }
}

ElementLayers MeshStore::Layers(uint32_t id) const {
    const auto &r = Records.at(id);
    const auto &b = Buffers;
    return {
        .FacePrimitives = b.FacePrimitives.Ref(r.FacePrimitives),
        .VertexPrimitives = b.VertexPrimitives.Ref(r.VertexPrimitives),
        .Skin = b.Skin.Ref(r.Skin),
        .Morph = b.Morph.Ref(r.Morph),
        .CornerTangent = b.CornerTangents.Ref(r.CornerTangents),
        .CornerColor = b.CornerColors.Ref(r.CornerColors),
        .VertexColor = b.VertexColors.Ref(r.VertexColors),
        .CornerUvs = {b.CornerUvs[0].Ref(r.CornerUvs[0]), b.CornerUvs[1].Ref(r.CornerUvs[1]), b.CornerUvs[2].Ref(r.CornerUvs[2]), b.CornerUvs[3].Ref(r.CornerUvs[3])},
        .CustomNormals = b.CustomNormals.Ref(r.CustomNormals),
        .CornerSectors = b.CornerSectors.Ref(r.CornerSectors),
        .NormalSectors = b.NormalSectors.Ref(r.NormalSectors),
    };
}

void MeshStore::SetMorphShadingAuthored(uint32_t id, bool authored) { WriteRecord(id).MorphShadingAuthored = authored; }

NamedAttribute &MeshStore::WriteAttribute(uint32_t attribute) {
    if (Tracked) Tracked->Attributes.Write(attribute, 1);
    return NamedAttributes.at(attribute);
}

void MeshStore::IndexAttribute(uint32_t attribute) {
    if (attribute < IndexedAttributes.size()) {
        if (const auto record = IndexedAttributes[attribute]; record == InvalidOffset) FreeAttributes.erase(attribute);
        else std::erase(RecordAttributes[record], attribute);
    }
    if (attribute >= NamedAttributes.size()) return;
    if (attribute >= IndexedAttributes.size()) IndexedAttributes.resize(attribute + 1u, InvalidOffset);
    const auto record = IndexedAttributes[attribute] = NamedAttributes[attribute].Record;
    if (record == InvalidOffset) {
        FreeAttributes.insert(attribute);
        return;
    }
    if (record >= RecordAttributes.size()) RecordAttributes.resize(record + 1u);
    auto &own = RecordAttributes[record];
    own.insert(std::ranges::upper_bound(own, attribute), attribute);
}

uint32_t MeshStore::AcquireAttribute(NamedAttribute &&entry) {
    const auto attribute = FreeAttributes.empty() ? uint32_t(NamedAttributes.size()) : *FreeAttributes.begin();
    if (attribute < NamedAttributes.size()) {
        WriteAttribute(attribute) = std::move(entry);
    } else {
        if (Tracked) Tracked->Attributes.Write(attribute, 1);
        NamedAttributes.emplace_back(std::move(entry));
    }
    IndexAttribute(attribute);
    return attribute;
}

std::expected<uint32_t, MeshStore::AttributeError> MeshStore::AddAttribute(uint32_t id, AttributeDomain domain, AttributeType type, std::string_view name, AttributeValue fallback) {
    if (!Records.at(id).Alive) throw std::invalid_argument("Retired mesh.");
    if (name.empty() || name.size() >= AttributeName{}.size()) return std::unexpected{AttributeError::InvalidName};
    if (FindAttribute(id, name) != InvalidOffset) return std::unexpected{AttributeError::Duplicate};
    NamedAttribute entry{.Record = id, .Domain = domain, .Type = type, .Default = fallback};
    std::ranges::copy(name, entry.Name.begin());
    const auto elements = ElementDomainOf(domain);
    const auto blocks = WithDomain(Buffers, elements, [&](const auto &arena) { return arena.HandleBound(DomainSet(Records[id], elements)) / MeshElementBlockSize; });
    WithAttributeValues(Buffers.Named, type, [&](auto &values) {
        CoverRange(values, entry.Values, blocks);
        FillRange(values, entry.Values, {0u, blocks}, FilledBlock<ArenaValue<std::remove_cvref_t<decltype(values)>>>(fallback));
    });
    return AcquireAttribute(std::move(entry));
}

uint32_t MeshStore::FindAttribute(uint32_t id, std::string_view name) const {
    if (id >= RecordAttributes.size()) return InvalidOffset;
    const auto found = std::ranges::find(RecordAttributes[id], name, [&](uint32_t attribute) { return NamedAttributes[attribute].NameView(); });
    return found == RecordAttributes[id].end() ? InvalidOffset : *found;
}

void MeshStore::ReleaseAttribute(uint32_t attribute) {
    const auto &entry = NamedAttributes.at(attribute);
    if (entry.Record == InvalidOffset) throw std::invalid_argument("Released attribute.");
    if (entry.Values.Count) WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) { values.Release(entry.Values); });
    WriteAttribute(attribute) = {};
    IndexAttribute(attribute);
}

std::expected<void, MeshStore::AttributeError> MeshStore::RenameAttribute(uint32_t attribute, std::string_view name) {
    if (name.empty() || name.size() >= AttributeName{}.size()) return std::unexpected{AttributeError::InvalidName};
    if (FindAttribute(NamedAttributes.at(attribute).Record, name) != InvalidOffset) return std::unexpected{AttributeError::Duplicate};
    auto &entry = WriteAttribute(attribute);
    entry.Name = {};
    std::ranges::copy(name, entry.Name.begin());
    return {};
}

std::expected<uint32_t, MeshStore::AttributeError> MeshStore::CopyAttribute(uint32_t attribute, std::string_view name) {
    const auto entry = NamedAttributes.at(attribute);
    const auto copy = AddAttribute(entry.Record, entry.Domain, entry.Type, name, entry.Default);
    if (!copy) return copy;
    WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) {
        const auto to = NamedAttributes[*copy].Values;
        std::ranges::copy(values.Get({entry.Values.Offset, to.Count}), values.GetMutable(to).begin());
    });
    return copy;
}

NamedAttributeRef MeshStore::AttributeRef(uint32_t attribute) const {
    const auto &entry = NamedAttributes.at(attribute);
    return {WithAttributeValues(Buffers.Named, entry.Type, [](const auto &values) { return values.Buffer.Slot; }), entry.Values.Offset * MeshElementBlockSize};
}

void MeshStore::CaptureAttributeWrites(uint32_t id, AttributeDomain domain, std::span<const uint32_t> blocks) const {
    const auto members = WithDomain(Buffers, ElementDomainOf(domain), [](const auto &arena) { return arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(); });
    ForEachNamedAttribute(id, [&](uint32_t, const NamedAttribute &entry) {
        if (entry.Domain != domain) return;
        std::vector<uint32_t> values;
        for (const auto block : blocks) values.push_back(entry.Values.Offset + members[block].Local);
        WithAttributeValues(Buffers.Named, entry.Type, [&](const auto &arena) { arena.Buffer.CaptureWriteElements(values, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>)); });
    });
}

TetBuffers MeshStore::AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices) {
    return {Buffers.TetPositions.Allocate(positions), Buffers.TetEdgeIndices.Allocate(edge_indices)};
}

void MeshStore::ReleaseTets(TetBuffers tets) {
    Buffers.TetPositions.Release(tets.Positions);
    Buffers.TetEdgeIndices.Release(tets.EdgeIndices);
}

Range MeshStore::AllocateSoundVertices(std::span<const uint32_t> vertices) { return Buffers.SoundVertices.Allocate(vertices); }
void MeshStore::ReleaseSoundVertices(std::vector<Range> ranges) { Buffers.SoundVertices.Release(std::move(ranges)); }
void MeshStore::EnsureSelectionState(state::Registry &r, mtl::ComputeChain &chain, std::span<const uint32_t> requested) {
    std::vector<uint32_t> ids{requested.begin(), requested.end()};
    SortUnique(ids);
    std::vector<SelectionUpdate> updates;
    for (const auto id : ids) {
        if (!Records.at(id).SelectionSummary.Count) updates.push_back({.StoreId = id});
    }
    if (updates.empty()) return;
    SyncMirrors();
    for (auto &update : updates) {
        WriteRecord(update.StoreId).SelectionSummary = Buffers.SelectionSummary.Allocate(1);
        WriteSelectionSummary(update.StoreId) = {};
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto list = GetBlockList(update.StoreId, SelectionDomains[d]);
            update.Blocks[d].assign(list.Blocks.begin(), list.Blocks.end());
        }
    }
    UpdateSelection(r, chain, updates);
    chain.AfterSubmit([this, ids = updates | std::views::transform(&SelectionUpdate::StoreId) | std::ranges::to<std::vector>()] {
        for (const auto id : ids) PublishSelectionSummary(id);
    });
}

SelectionView MeshStore::GetSelectedElements(uint32_t id, Element element) const {
    const auto domain = SelectionDomain(element);
    const auto &root = GetSelectionRoot(id, element);
    const auto leaves = AggregateArena(Buffers, domain).Buffer.GetSpan<SelectionAggregate>();
    return WithDomain(Buffers, domain, [&](const auto &arena) -> SelectionView {
        return {SelectionArena(Buffers, domain).Buffer.GetSpan<uint32_t>(), arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(), arena.Entries(DomainSet(Records.at(id), domain)), Buffers.SelectionTree.Read(3u * id + SelectableIndex(domain), leaves), root.Selected};
    });
}

SelectionView MeshStore::GetHiddenElements(uint32_t id, Element element) const {
    auto view = GetSelectedElements(id, element);
    view.Bits = HiddenArena(Buffers, SelectionDomain(element)).Buffer.GetSpan<uint32_t>();
    view.Selected = GetSelectionRoot(id, element).Hidden;
    view.Kind = SelectionIndexMask::Hidden;
    return view;
}
uint32_t MeshStore::GetHiddenSlot(Element element) const { return HiddenArena(Buffers, SelectionDomain(element)).Buffer.Slot; }

const SelectionAggregate &MeshStore::GetSelectionRoot(uint32_t id, Element element) const {
    return Buffers.SelectionTree.Get(3u * id + SelectableIndex(SelectionDomain(element)));
}
SlotOffset MeshStore::GetVertexSelectionRoot(uint32_t id) const { return Buffers.SelectionTree.Ref(3u * id); }

EditSelectionSummary &MeshStore::WriteSelectionSummary(uint32_t id) {
    CaptureSelectionSummary(id);
    return Buffers.SelectionSummary.GetMutable(Records.at(id).SelectionSummary)[0];
}

void MeshStore::PublishSelectionSummary(uint32_t id) {
    auto &summary = WriteSelectionSummary(id);
    const auto &vertices = GetSelectionRoot(id, Element::Vertex);
    const auto *selected = summary.Mode == Element::None ? nullptr : &GetSelectionRoot(id, summary.Mode);
    summary.PositionSum = vertices.PositionSum;
    summary.SelectedCount = selected ? selected->Selected : 0u;
    summary.SelectedVertexCount = vertices.Selected;
    summary.SharpnessFlags = selected ? selected->Flags & (SelectionSelectedSharp | SelectionSelectedSmooth) : 0u;
}

void MeshStore::SetSelectionBaseline(uint32_t id, Element element, std::vector<std::pair<uint32_t, SelectionBlock>> blocks, uint32_t active) {
    auto &derived = DerivedRecords.at(id);
    derived.SelectionBaseline = std::move(blocks);
    derived.SelectionBaselineActive = active;
    derived.SelectionBaselineElement = element;
}

bool MeshStore::IsLiveElement(uint32_t id, Element element, ElementHandle handle) const {
    const auto domain = SelectionDomain(element);
    return WithDomain(Buffers, domain, [&](const auto &arena) { return arena.Contains(DomainSet(Records.at(id), domain), handle); });
}

std::vector<Range> MeshStore::LiveVertexRanges(uint32_t id, std::span<const Range> ranges) const {
    const auto entries = Buffers.Vertices.Entries(Records.at(id).Vertices);
    const auto blocks = Buffers.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>();
    std::vector<Range> live;
    for (const auto range : ranges) {
        const auto end = std::min(uint64_t(range.Offset) + range.Count, uint64_t(entries.size()) * MeshElementBlockSize);
        // Each step covers the handles of one live word from h to the range's end.
        for (uint64_t h = range.Offset; h < end; h = (h / 32u + 1u) * 32u) {
            const auto entry = entries[h / MeshElementBlockSize];
            const auto count = uint32_t(std::min(end - h, 32u - h % 32u));
            auto bits = entry == InvalidOffset ? 0u : blocks[entry].Live[h % MeshElementBlockSize / 32u] >> (h % 32u) & uint32_t((uint64_t{1} << count) - 1u);
            for (auto first = uint32_t(h); bits;) {
                const auto dead = uint32_t(std::countr_zero(bits)), run = uint32_t(std::countr_one(bits >> dead));
                AppendRange(live, {first + dead, run});
                first += dead + run;
                bits = dead + run < 32u ? bits >> (dead + run) : 0u;
            }
        }
    }
    return live;
}
std::optional<ElementHandle> MeshStore::LiveHandleAt(uint32_t id, Element element, ElementAddress address) const {
    const auto domain = SelectionDomain(element);
    return WithDomain(Buffers, domain, [&](const auto &arena) -> std::optional<ElementHandle> {
        const auto set = DomainSet(Records.at(id), domain);
        if (address.Value() >= arena.Capacity() || arena.Blocks.Get({address.Value() / MeshElementBlockSize, 1u})[0].Owner != set.Index) return {};
        const auto handle = arena.Handle(address);
        return arena.Contains(set, handle) ? std::optional{handle} : std::nullopt;
    });
}
ElementAddress MeshStore::HandleAddress(uint32_t id, ElementDomain domain, ElementHandle handle) const {
    return WithDomain(Buffers, domain, [&](const auto &arena) { return arena.Address(DomainSet(Records.at(id), domain), handle); });
}
std::vector<uint32_t> MeshStore::ArenaBlocks(uint32_t id, ElementDomain domain, const BufferArena<uint32_t> &storage, ElementWork work) const {
    return WithDomain(Buffers, domain, [&](const auto &arena) { return WorkBlocks(storage, work, 0u, arena.Entries(DomainSet(Records.at(id), domain))); });
}
uint32_t MeshStore::GetSelectionSlot(Element element) const { return SelectionArena(Buffers, SelectionDomain(element)).Buffer.Slot; }
EditSelectionStorage MeshStore::GetEditSelectionStorage(uint32_t id) const {
    const auto summary = Records.at(id).SelectionSummary;
    const auto binding = [&](Element element) -> SlotOffset { return {GetSelectionSlot(element), 0u}; };
    return {
        binding(Element::Vertex), binding(Element::Edge), binding(Element::Face),
        summary.Count > 0 ? Buffers.SelectionSummary.Slotted(summary) : SlottedRange{},
        GetHiddenSlot(Element::Vertex), GetHiddenSlot(Element::Edge), GetHiddenSlot(Element::Face)
    };
}
const EditSelectionSummary &MeshStore::GetSelectionSummary(uint32_t id) const { return Buffers.SelectionSummary.Get(Records.at(id).SelectionSummary)[0]; }

vec3 ComposeCornerNormal(const MeshStore &meshes, uint32_t id, CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &sources) {
    const auto face = meshes.HandleAddress(id, MeshStore::ElementDomain::Face, ElementHandle{face_ids[ci / 3u]}).Value();
    if (mode == UniformFaceMode || (mode == uint32_t(CornerClassMode::Mixed) && sharpness[face])) {
        const auto posed = sources.PosedFaceNormals.Find(face);
        return posed == InvalidOffset ? sources.FaceNormals[face] : sources.PosedFaceNormals.Values[posed];
    }
    const auto root = mode == uint32_t(CornerClassMode::Mixed) ? sectors.GetOr(ci, InvalidOffset) : InvalidOffset;
    if (root == InvalidOffset) {
        const auto v = meshes.HandleAddress(id, MeshStore::ElementDomain::Vertex, ElementHandle{vertices[ci]}).Value();
        const auto posed = sources.PosedVertexNormals.Find(v);
        return posed == InvalidOffset ? sources.VertexNormals[v] : sources.PosedVertexNormals.Values[posed];
    }
    const auto posed = sources.PosedSectorNormals.Find(meshes.HandleAddress(id, MeshStore::ElementDomain::Halfedge, ElementHandle{root}).Value());
    return posed == InvalidOffset ? normal_sectors[root].Normal : sources.PosedSectorNormals.Values[posed];
}

CornerNormalView MeshStore::GetCornerNormalView(uint32_t id) const {
    const auto &record = Records.at(id);
    return {
        .ClassMode = record.Classification,
        .CornerVertices = Buffers.FaceCorners.Buffer.GetSpan<uint32_t>(),
        .Vertices = Buffers.Vertices.Buffer.GetSpan<Vertex>(),
        .VertexNormals = Buffers.BaseVertexNormals.Buffer.GetSpan<vec3>(),
        .FaceNormals = Buffers.BaseFaceNormals.Buffer.GetSpan<vec3>(),
        .Connectivity = GetConnectivity(id),
        .FaceSharpness = Buffers.FaceSharpness.Buffer.GetSpan<uint8_t>(),
        .CornerSectors = Buffers.CornerSectors.View(record.CornerSectors),
        .NormalSectors = Buffers.NormalSectors.View(record.NormalSectors),
        .CustomNormals = Buffers.CustomNormals.View(record.CustomNormals),
    };
}

std::span<const vec3> MeshStore::GetCornerNormals(const Mesh &mesh) const {
    const auto id = mesh.GetStoreId();
    const auto &record = Records.at(id);
    static thread_local std::vector<vec3> corners;
    corners.resize(size_t{record.TriangleCount} * 3);
    const auto normals = GetCornerNormalView(id);
    const auto handles = GetTriangleCorners(id);
    for (uint32_t ci = 0; ci < corners.size(); ++ci) corners[ci] = normals[handles[ci]];
    return corners;
}

namespace {
SharpnessSummary LiveSharpness(uint32_t flags) {
    return {(flags & SelectionLiveSharp) != 0u, (flags & SelectionLiveSharp) != 0u && !(flags & SelectionLiveSmooth)};
}
} // namespace

SharpnessSummary MeshStore::GetEdgeSharpnessSummary(uint32_t id) const { return LiveSharpness(GetSelectionRoot(id, Element::Edge).Flags); }

namespace {
void WriteVertices(std::span<Vertex> dst, std::span<const vec3> positions) {
    for (uint32_t i = 0; i < positions.size(); ++i) dst[i] = {.Position = positions[i]};
}
} // namespace

void MeshStore::AllocateConnectivity(uint32_t id, uint32_t halfedge_count, uint32_t face_count, bool face_starts, std::span<const uint32_t> face_offsets, std::span<const std::array<uint32_t, 2>> wire_edges) {
    auto &record = WriteRecord(id);
    record.ConnectivityFaceStarts = face_starts;
    if (!record.FaceCorners) record.FaceCorners = Buffers.FaceCorners.Allocate(halfedge_count);
    if (!record.FaceData) record.FaceData = Buffers.FaceTriangles.Allocate(face_count);
    if (!record.EdgeData) record.EdgeData = Buffers.EdgeHalfedges.Allocate(halfedge_count);
    SyncMirrors();
    if (!wire_edges.empty()) {
        if (uint64_t(wire_edges.size()) * 2u + 3ull * face_count > halfedge_count) throw std::invalid_argument("Wire corners exceed connectivity allocation.");
        const auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        const auto run = Buffers.FaceCorners.Dense(record.FaceCorners);
        const auto owners = Buffers.HalfedgeFaces.GetMutable(run), opposites = Buffers.OppositeHalfedges.GetMutable(run);
        const uint32_t wire_start = halfedge_count - 2u * uint32_t(wire_edges.size());
        for (uint32_t e = 0; e < wire_edges.size(); ++e) {
            const uint32_t h = wire_start + 2u * e;
            corners[h] = wire_edges[e][1];
            corners[h + 1u] = wire_edges[e][0];
            owners[h] = owners[h + 1u] = InvalidOffset;
            opposites[h] = h + 1u;
            opposites[h + 1u] = h;
        }
    }
    if (!face_starts || face_offsets.empty()) return;
    const auto faces = Buffers.FaceRanges.GetMutable(Buffers.FaceTriangles.Dense(record.FaceData));
    const uint32_t face_corners = halfedge_count - 2u * uint32_t(wire_edges.size());
    for (uint32_t f = 0; f < face_count; ++f) faces[f] = {face_offsets[f], f + 1 < face_count ? face_offsets[f + 1] : face_corners};
}

void MeshStore::FinishConnectivity(uint32_t id, uint32_t edge_count, uint32_t open_edges) {
    auto &record = WriteRecord(id);
    auto &edges = Buffers.EdgeHalfedges;
    // An edge's corners are its unpaired ones plus the two its first halfedge pairs unless it is open.
    record.UnpairedCorners = Buffers.FaceCorners.Count(record.FaceCorners) + 2u * open_edges - 2u * edge_count;
    ElementHandleRange allocated{.First = 0u, .Count = edges.Count(record.EdgeData)};
    const auto blocks = edges.Shrink(record.EdgeData, allocated, {}, edge_count);
    if (CreatedInStep(id)) ZeroFreeBlocks(Buffers, Domain::Edge, blocks);
}

MeshConnectivity MeshStore::GetConnectivity(uint32_t id) const {
    const auto &r = Records.at(id);
    return {
        .VertexTable = Buffers.Vertices.Entries(r.Vertices),
        .HalfedgeTable = Buffers.FaceCorners.Entries(r.FaceCorners),
        .EdgeTable = Buffers.EdgeHalfedges.Entries(r.EdgeData),
        .FaceTable = Buffers.FaceTriangles.Entries(r.FaceData),
        .VertexBlocks = Buffers.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        .HalfedgeBlocks = Buffers.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        .EdgeBlocks = Buffers.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        .FaceBlocks = Buffers.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        .VertexCount = Buffers.Vertices.Count(r.Vertices),
        .HalfedgeCount = Buffers.FaceCorners.Count(r.FaceCorners),
        .EdgeCount = Buffers.EdgeHalfedges.Count(r.EdgeData),
        .FaceCount = Buffers.FaceTriangles.Count(r.FaceData),
        .UnpairedCornerCount = r.UnpairedCorners,
        .OutgoingValues = Buffers.OutgoingHalfedges.Buffer.GetSpan<uint32_t>(),
        .OppositeValues = Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>(),
        .HalfedgeEdgeValues = Buffers.HalfedgeEdges.Buffer.GetSpan<uint32_t>(),
        .HalfedgeFaceValues = Buffers.HalfedgeFaces.Buffer.GetSpan<uint32_t>(),
        .EdgeHalfedgeValues = Buffers.EdgeHalfedges.Buffer.GetSpan<uint32_t>(),
        .FaceRangeValues = Buffers.FaceRanges.Buffer.GetSpan<uvec2>(),
        .VertexCornerValues = Buffers.VertexCorners.Buffer.GetSpan<uvec2>(),
        .FanItems = Buffers.VertexFans.Items.Buffer.GetSpan<uvec2>(),
    };
}

void MeshStore::ReleaseBlockLists(uint32_t id) {
    ReleaseBlockLists(std::span{&id, 1u});
}

void MeshStore::ReleaseBlockLists(std::span<const uint32_t> ids) {
    const std::scoped_lock lock{BlockListLock};
    std::vector<Range> words;
    for (const auto id : ids) {
        if (id >= BlockListEntries.size()) continue;
        for (auto &entry : BlockListEntries[id]) {
            if (entry.Words.Count) words.push_back(entry.Words);
            entry = {};
        }
    }
    if (FrameReadsBlockLists) RetiredBlockLists.append_range(words);
    else BlockLists.Release(std::move(words));
}

void MeshStore::RetireBlockListWords(Range words) const {
    if (!words.Count) return;
    if (FrameReadsBlockLists) RetiredBlockLists.push_back(words);
    else BlockLists.Release(words);
}

void MeshStore::FrameSubmitted() {
    const std::scoped_lock lock{BlockListLock};
    FrameReadsBlockLists = true;
}

void MeshStore::FrameCompleted() {
    const std::scoped_lock lock{BlockListLock};
    FrameReadsBlockLists = false;
    BlockLists.Release(std::exchange(RetiredBlockLists, {}));
}

MeshStore::BlockList MeshStore::GetBlockList(uint32_t id, ElementDomain domain) const {
    const std::scoped_lock lock{BlockListLock};
    const auto set = DomainSet(Records.at(id), domain);
    if (BlockListEntries.size() < Records.size()) BlockListEntries.resize(Records.size());
    auto &entry = BlockListEntries[id][uint32_t(domain) - 1u];
    WithDomain(Buffers, domain, [&](const auto &arena) {
        const auto revision = set ? arena.Set(set).Revision : 0u;
        if (entry.Set == set && entry.Revision == revision) return;
        const auto membership = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
        std::vector<uint32_t> blocks;
        for (const auto b : arena.Entries(set)) {
            if (b != InvalidOffset) blocks.push_back(b);
        }
        const auto count = uint32_t(blocks.size());
        RetireBlockListWords(entry.Words);
        entry.Words = BlockLists.Allocate(2u * count);
        const auto words = BlockLists.GetMutable(entry.Words);
        uint32_t live = 0u;
        for (uint32_t i = 0u; i < count; ++i) {
            words[i] = blocks[i];
            words[count + i] = live += membership[blocks[i]].Count;
        }
        entry.Set = set;
        entry.Revision = revision;
    });
    const auto count = entry.Words.Count / 2u;
    return {BlockLists.Get({entry.Words.Offset, count}), BlockLists.Get({entry.Words.Offset + count, count}), {BlockLists.Buffer.Slot, entry.Words.Offset}};
}

DomainBlocks BlocksOf(const MeshStore &meshes, uint32_t id, MeshStore::ElementDomain domain) {
    const auto list = meshes.GetBlockList(id, domain);
    return {list.Gpu, uint32_t(list.Blocks.size()), WithDomain(meshes.Arenas(), domain, [](const auto &arena) { return arena.Blocks.Buffer.Slot; })};
}

namespace {
template<typename T>
ElementView<T> LiveView(const MeshStore &store, const ElementArena<T> &arena, ElementSetRef set, uint32_t id, MeshStore::ElementDomain domain) {
    if (!set) return {};
    const auto &header = arena.Set(set);
    const auto values = arena.Buffer.template GetSpan<T>();
    if (header.Flags & 1u) return {values, arena.Dense(set).Offset, header.Count};
    const auto list = store.GetBlockList(id, domain);
    return {values, arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(), list.Blocks, list.LivePrefix, header.Count};
}
} // namespace

ElementHandle MeshStore::LiveElementAt(uint32_t id, ElementDomain domain, uint32_t ordinal) const {
    const auto set = DomainSet(Records.at(id), domain);
    return WithDomain(Buffers, domain, [&](const auto &arena) {
        const auto view = LiveView(*this, arena, set, id, domain);
        if (ordinal >= view.size()) throw std::out_of_range("Live element ordinal.");
        return ElementHandle{view.Handle(ordinal)};
    });
}

uint32_t MeshStore::LiveElementOrdinal(uint32_t id, ElementDomain domain, ElementHandle handle) const {
    const auto set = DomainSet(Records.at(id), domain);
    return WithDomain(Buffers, domain, [&](const auto &arena) -> uint32_t {
        if (!arena.Contains(set, handle)) return InvalidOffset;
        if (arena.Set(set).Flags & 1u) return handle.Value;
        const auto entries = arena.Entries(set);
        const auto local = handle.Value / MeshElementBlockSize, offset = handle.Value % MeshElementBlockSize;
        const auto &live = arena.Blocks.Get({entries[local], 1u})[0].Live;
        const auto list = GetBlockList(id, domain);
        const auto membership = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
        const auto at = uint32_t(std::ranges::lower_bound(list.Blocks, local, {}, [&](uint32_t block) { return membership[block].Local; }) - list.Blocks.begin());
        auto rank = at ? list.LivePrefix[at - 1u] : 0u;
        for (uint32_t word = 0u; word < offset / 32u; ++word) rank += std::popcount(live[word]);
        return rank + std::popcount(live[offset / 32u] & ((1u << (offset % 32u)) - 1u));
    });
}

ElementTables MeshStore::GetElementTables(uint32_t id) const {
    const auto &r = Records.at(id);
    return {
        .Vertices = Buffers.Vertices.TableRef(r.Vertices),
        .Halfedges = Buffers.FaceCorners.TableRef(r.FaceCorners),
        .Edges = Buffers.EdgeHalfedges.TableRef(r.EdgeData),
        .Faces = Buffers.FaceTriangles.TableRef(r.FaceData),
        .Triangles = Buffers.Triangles.TableRef(r.TriangleData),
    };
}

ConnectivityRef MeshStore::GetConnectivityRef(uint32_t id) const {
    return {
        .Tables = GetElementTables(id),
        .CornersSlot = Buffers.FaceCorners.Buffer.Slot,
        .OutgoingSlot = Buffers.OutgoingHalfedges.Buffer.Slot,
        .OppositesSlot = Buffers.OppositeHalfedges.Buffer.Slot,
        .HalfedgeEdgesSlot = Buffers.HalfedgeEdges.Buffer.Slot,
        .HalfedgeFacesSlot = Buffers.HalfedgeFaces.Buffer.Slot,
        .FaceRangesSlot = Buffers.FaceRanges.Buffer.Slot,
        .EdgesSlot = Buffers.EdgeHalfedges.Buffer.Slot,
        .VertexCornersSlot = Buffers.VertexCorners.Buffer.Slot,
        .FanItemsSlot = Buffers.VertexFans.Items.Buffer.Slot,
    };
}

TriangleCorners MeshStore::GetTriangleCorners(uint32_t id) const {
    const auto &record = Records.at(id);
    return {TriangleView(id), {}, Buffers.Triangles.Entries(record.TriangleData), Buffers.FaceCorners.Entries(record.FaceCorners)};
}

ElementView<uvec3> MeshStore::TriangleView(uint32_t id) const { return LiveView(*this, Buffers.Triangles, Records.at(id).TriangleData, id, ElementDomain::Triangle); }

uint32_t MeshStore::CreateMeshSource(const MeshData &data) {
    const auto vertices = Buffers.Vertices.Allocate(data.Positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), data.Positions);
    const auto id = AcquireId({.Vertices = vertices, .Alive = true});
    SyncMirrors();
    ClearVertexRoots(Buffers, vertices);
    // Both face corners and loose-edge endpoints must exist before GPU welding remaps them.
    AllocateConnectivity(id, data.HalfedgeCount(), data.FaceCount(), !data.FaceOffsets.empty(), data.FaceOffsets, data.Edges);
    if (data.FaceCount() > 0) {
        std::ranges::copy(data.FaceCorners, Buffers.FaceCorners.GetMutable(Get(id).FaceCorners).begin());
        auto &record = WriteRecord(id);
        record.TriangleCount = uint32_t(data.FaceCorners.size()) - 2u * data.FaceCount();
        record.TriangleData = Buffers.Triangles.Allocate(record.TriangleCount);
    }
    return id;
}

void MeshStore::CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &deform, const std::optional<MorphTargetData> &morph) {
    auto &record = WriteRecord(id);
    const auto vertex_count = Buffers.Vertices.Count(record.Vertices);
    if (vertex_count == 0) return;
    if (deform) {
        std::vector<BoneDeformVertex> skin(vertex_count);
        for (uint32_t i = 0; i < vertex_count; ++i) skin[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        Buffers.Skin.Initialize(record.Skin, vertex_count, skin);
        record.SkinBlocksReady = true;
    }
    if (morph && morph->TargetCount > 0) {
        record.MorphTargetCount = morph->TargetCount;
        const bool has_normal_deltas = !morph->NormalDeltas.empty();
        std::vector<MorphTargetVertex> targets(uint64_t(vertex_count) * record.MorphTargetCount);
        for (size_t i = 0; i < targets.size(); ++i) {
            targets[i] = {.PositionDelta = morph->PositionDeltas[i], .NormalDelta = has_normal_deltas ? morph->NormalDeltas[i] : vec3{0}};
        }
        Buffers.Morph.Initialize(record.Morph, vertex_count, targets, record.MorphTargetCount);
        record.MorphBlocksReady = true;
        record.DefaultMorphWeights = morph->DefaultWeights;
        record.DefaultMorphWeights.resize(record.MorphTargetCount, 0.f);
    }
}

void MeshStore::ShrinkMeshSource(uint32_t id, uint32_t welded_vertices) {
    auto &record = WriteRecord(id);
    const auto vertices = Buffers.Vertices.Dense(record.Vertices);
    const auto kept_blocks = std::max(1u, ElementArena<Vertex>::BlockCount(welded_vertices));
    Buffers.Skin.Values.Shrink(record.Skin, kept_blocks);
    Buffers.Morph.Values.Shrink(record.Morph, kept_blocks * record.MorphTargetCount);
    ElementHandleRange allocated{.First = 0u, .Count = vertices.Count};
    const auto blocks = Buffers.Vertices.Shrink(record.Vertices, allocated, {}, welded_vertices);
    if (CreatedInStep(id)) ZeroFreeBlocks(Buffers, Domain::Vertex, blocks);
}

uint32_t MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertices = Buffers.Vertices.Allocate(positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), positions);
    const auto point_normals = attrs.Normals ? Buffers.PointNormals.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    const auto id = AcquireId({.Vertices = vertices, .PointNormals = point_normals, .HasAuthoredNormals = attrs.Normals.has_value(), .Alive = true});
    SyncMirrors();
    ClearVertexRoots(Buffers, vertices);
    FillBaseVertexNormalMirror(vertices, point_normals);
    return id;
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count, const MeshVertexAttributes &attrs, bool corner_uvs0) {
    const uint32_t vertices = data.Positions.size();
    const uint32_t faces = data.FaceCount();
    const uint32_t halfedges = data.HalfedgeCount();
    const uint32_t triangles = uint32_t(data.FaceCorners.size()) - 2u * faces;
    Buffers.Vertices.PlanAdditional(vertices);
    Buffers.BaseVertexNormals.PlanAdditional(vertices);
    Buffers.FaceTriangles.PlanAdditional(faces);
    Buffers.FaceSharpness.PlanAdditional(faces);
    Buffers.BaseFaceNormals.PlanAdditional(faces);
    Buffers.Triangles.PlanAdditional(triangles);
    Buffers.FaceCorners.PlanAdditional(halfedges);
    Buffers.EdgeHalfedges.PlanAdditional(halfedges);
    Buffers.PrimitiveMaterials.PlanAdditional(primitives.MaterialIndices.size());
    const auto corner_blocks = ElementArena<uint32_t>::BlockCount(halfedges), vertex_blocks = ElementArena<uint32_t>::BlockCount(vertices);
    if (faces) {
        if (attrs.Tangents) Buffers.CornerTangents.Values.PlanAdditional(corner_blocks);
        if (attrs.Colors0) Buffers.CornerColors.Values.PlanAdditional(corner_blocks);
        const std::array uvs{&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3};
        for (uint32_t set = 0; set < uvs.size(); ++set) {
            if (*uvs[set] || (set == 0u && corner_uvs0)) Buffers.CornerUvs[set].Values.PlanAdditional(corner_blocks);
        }
        Buffers.FacePrimitives.Values.PlanAdditional(ElementArena<uint32_t>::BlockCount(faces));
    } else if (attrs.Colors0) {
        Buffers.VertexColors.Values.PlanAdditional(vertex_blocks);
    }
    if (has_deform) Buffers.Skin.Values.PlanAdditional(vertex_blocks);
    if (morph_target_count > 0) Buffers.Morph.Values.PlanAdditional(vertex_blocks * morph_target_count);
}

void MeshStore::PlanClone(const Mesh &mesh) {
    const auto &record = Records.at(mesh.GetStoreId());
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        const auto *allocation = ranges(record);
        if (!allocation || info.Mirror) return;
        if constexpr (std::same_as<std::remove_cvref_t<decltype(*allocation)>, Range>) arena.PlanAdditional(allocation->Count);
        else if (*allocation) {
            WithDomain(Buffers, info.Elements, [&](const auto &owner) {
                arena.PlanAdditional(owner.Set(*allocation).BlockCount * MeshElementBlockSize);
            });
        }
    });
}

uint64_t MeshStore::RecordWords(TopologyCounts counts) const {
    uint64_t words = 0u;
    ForEachArena(Buffers, [&](const auto &arena, const ArenaInfo &info, auto &&) {
        const uint32_t elements = info.Elements == Domain::Vertex ? counts.Vertices :
            info.Elements == Domain::Face                         ? counts.Faces :
            info.Elements == Domain::None                         ? 0u :
                                                                    counts.Halfedges;
        const uint64_t blocks = ElementArena<uint32_t>::BlockCount(elements);
        words += (info.BlockIndexed ? blocks : blocks * MeshElementBlockSize) * sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>) / sizeof(uint32_t);
    });
    return words;
}

void MeshStore::CommitReserves() {
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.CommitPlanned(); });
}

void MeshStore::CreateMesh(uint32_t id, const MeshData &data, const MeshVertexAttributes &attrs, const MeshPrimitives &primitives, const CornerLayers &layers, bool has_authored_normals) {
    const profile::CpuScope scope{"CreateMesh"};
    Unshaded.push_back(id);
    const uint32_t face_count = data.FaceCount();

    // Source creation and welding completed the vertex-domain arena ranges.
    auto &record = WriteRecord(id);
    record.HasAuthoredNormals = has_authored_normals || attrs.Normals.has_value();
    record.PointNormals = attrs.Normals ? Buffers.PointNormals.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    FillBaseVertexNormalMirror(record.Vertices, record.PointNormals);

    const auto write_primitive_tables = [&](uint32_t primitive_count) {
        record.FacePrimitivesReady = face_count != 0u;
        record.VertexPrimitivesReady = face_count == 0u;
        auto &attributes = face_count ? Buffers.FacePrimitives : Buffers.VertexPrimitives;
        attributes.Initialize(face_count ? record.FacePrimitives : record.VertexPrimitives, face_count ? face_count : Buffers.Vertices.Count(record.Vertices), primitives.ElementPrimitiveIndices);

        record.PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(primitive_count);
        const auto pm_span = Buffers.PrimitiveMaterials.GetMutable(record.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        else std::ranges::fill(pm_span, 0u);
    };

    if (face_count > 0) {
        const auto corners = Buffers.FaceCorners.Count(record.FaceCorners);
        if (!layers.Tangents.empty()) {
            record.CornerAttributes |= MeshAttributeBit_Tangent;
            Buffers.CornerTangents.Initialize(record.CornerTangents, corners, layers.Tangents);
        }
        if (!layers.Colors.empty()) {
            record.CornerAttributes |= MeshAttributeBit_Color0;
            Buffers.CornerColors.Initialize(record.CornerColors, corners, layers.Colors);
        }
        for (uint32_t set = 0; set < layers.Uvs.size(); ++set) {
            if (layers.Uvs[set].empty()) continue;
            record.CornerAttributes |= MeshAttributeBit_TexCoord0 << set;
            Buffers.CornerUvs[set].Initialize(record.CornerUvs[set], corners, layers.Uvs[set]);
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.ElementPrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.ElementPrimitiveIndices) + 1u) :
            1u;
        write_primitive_tables(primitive_count);

    } else if (!primitives.ElementPrimitiveIndices.empty()) {
        // Primitive indices are source-wide, so the material table spans every primitive of the source mesh.
        const auto primitive_count = primitives.MaterialIndices.empty() ? 1u : uint32_t(primitives.MaterialIndices.size());
        write_primitive_tables(primitive_count);
    }

    if (face_count == 0 && attrs.Colors0) {
        record.VertexAttributes |= MeshAttributeBit_Color0;
        Buffers.VertexColors.Initialize(record.VertexColors, Buffers.Vertices.Count(record.Vertices), *attrs.Colors0);
    }

    std::ranges::fill(Buffers.FaceSharpness.GetMutable(Buffers.FaceTriangles.Dense(record.FaceData)), uint8_t{0});
    std::ranges::fill(Buffers.EdgeSharpness.GetMutable(Buffers.EdgeHalfedges.Dense(record.EdgeData)), uint8_t{0});
}

uint32_t MeshStore::BeginTopologyOutput(uint32_t source, std::span<const uint32_t> materials) {
    const auto &src = Records.at(source);
    Record record{
        .CornerAttributes = src.CornerAttributes,
        .VertexAttributes = src.VertexAttributes,
        .PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(uint32_t(materials.size())),
        .FacePrimitivesReady = true,
        .SelectionSummary = Buffers.SelectionSummary.Allocate(1),
        .ConnectivityFaceStarts = true,
        .SkinBlocksReady = src.SkinBlocksReady,
        .MorphBlocksReady = src.MorphBlocksReady,
        .MorphTargetCount = src.MorphTargetCount,
        .HasAuthoredNormals = src.HasAuthoredNormals,
        .DefaultMorphWeights = src.DefaultMorphWeights,
        .Alive = true,
    };
    std::ranges::copy(materials, Buffers.PrimitiveMaterials.GetMutable(record.PrimitiveMaterials).begin());
    std::vector<NamedAttribute> attributes;
    const auto add = [&](uint32_t, const NamedAttribute &entry) { attributes.push_back({.Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default}); };
    ForEachNamedAttribute(source, add);
    const auto id = AcquireId(std::move(record));
    WriteSelectionSummary(id) = {};
    for (auto &attribute : attributes) {
        attribute.Record = id;
        AcquireAttribute(std::move(attribute));
    }
    return id;
}

MeshStore::BlockPairs MeshStore::PairBlocks(uint32_t source, uint32_t destination) const {
    BlockPairs pairs;
    for (const auto domain : ElementDomains) {
        WithDomain(Buffers, domain, [&](const auto &arena) {
            const auto from = arena.Entries(DomainSet(Records[source], domain)), to = arena.Entries(DomainSet(Records[destination], domain));
            for (uint32_t local = 0u; local < std::min(from.size(), to.size()); ++local) {
                if (from[local] != InvalidOffset && to[local] != InvalidOffset) pairs[uint32_t(domain)].push_back({from[local], to[local]});
            }
        });
    }
    return pairs;
}

void MeshStore::CopyBlocks(CloneCopies &copies, uint32_t destination, uint32_t source, const BlockPairs &pairs) {
    const auto &src = Records[source];
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        using Allocation = std::remove_cvref_t<decltype(*ranges(src))>;
        if constexpr (std::same_as<Allocation, ElementSetRef>) {
            if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
            constexpr auto Bytes = uint64_t(MeshElementBlockSize) * sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
            ForEachPairRun(pairs[uint32_t(info.Elements)], [&](uint32_t from, uint32_t to, uint32_t count) { CaptureCopy(copies, arena.Buffer, Bytes, from, to, count); });
        }
    });
    for (const auto domain : SelectionDomains) {
        ForEachPairRun(pairs[uint32_t(domain)], [&](uint32_t from, uint32_t to, uint32_t count) {
            CaptureCopy(copies, SelectionArena(Buffers, domain).Buffer, sizeof(SelectionBlock), from, to, count);
            CaptureCopy(copies, HiddenArena(Buffers, domain).Buffer, sizeof(SelectionBlock), from, to, count);
            CaptureCopy(copies, AggregateArena(Buffers, domain).Buffer, sizeof(SelectionAggregate), from, to, count);
        });
    }
    if (const auto &vertices = pairs[uint32_t(Domain::Vertex)]; !vertices.empty()) {
        const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
        std::vector<uvec2> released;
        uint64_t fan_count = 0u;
        for (const auto [from, to] : vertices) {
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot) {
                if (const auto root = roots[to * MeshElementBlockSize + slot]; root.y) released.push_back(root);
            }
            Buffers.Vertices.ForEachSlot(from, [&](uint32_t slot) { fan_count += roots[from * MeshElementBlockSize + slot].y; });
        }
        Buffers.VertexFans.Release(released);
        if (fan_count >= InvalidOffset) throw std::length_error("Copied fan items exceed their address space.");
        const auto fans = Buffers.VertexFans.Items.Allocate(uint32_t(fan_count));
        Buffers.VertexFans.Items.CaptureWrite(fans);
        uint32_t next = fans.Offset;
        Range run{InvalidOffset, 0u};
        const auto flush = [&] {
            copies.Copy(Buffers.VertexFans.Items.Buffer, {.Source = uint64_t(run.Offset) * sizeof(uvec2), .Destination = uint64_t(next - run.Count) * sizeof(uvec2), .Bytes = uint64_t(run.Count) * sizeof(uvec2)});
            run = {InvalidOffset, 0u};
        };
        ForEachPairRun(vertices, [&](uint32_t from, uint32_t to, uint32_t count) {
            auto written = Buffers.VertexCorners.GetMutable({to * MeshElementBlockSize, count * MeshElementBlockSize});
            std::ranges::fill(written, uvec2{InvalidOffset, 0u});
            for (uint32_t b = 0u; b < count; ++b) {
                Buffers.Vertices.ForEachSlot(from + b, [&](uint32_t slot) {
                    const auto root = roots[(from + b) * MeshElementBlockSize + slot];
                    if (!root.y) return;
                    written[b * MeshElementBlockSize + slot] = {next, root.y};
                    if (root.x != run.Offset + run.Count) {
                        flush();
                        run.Offset = root.x;
                    }
                    run.Count += root.y;
                    next += root.y;
                });
            }
        });
        flush();
    }
    DerivedRecords[destination].NormalRevision = ++NextNormalRevision;
}

std::vector<uint32_t> MeshStore::CloneMeshes(CloneCopies &copies, std::span<const uint32_t> source_ids, bool render) {
    const profile::CpuScope scope{"CloneMeshes"};
    std::vector<uint32_t> ids;
    std::vector<BlockPairs> pairs;
    ids.reserve(source_ids.size());
    pairs.reserve(source_ids.size());
    // The whole batch allocates before any copy encodes, since arena growth can move either side.
    for (const auto src_id : source_ids) {
        Record copy{Records.at(src_id)};
        ResetRender(copy);
        const auto id = AcquireId(std::move(copy));
        DerivedRecords[id] = DerivedRecords.at(src_id);
        DerivedRecords[id].NormalRevision = ++NextNormalRevision;
        DerivedRecords[id].CreatedStep = Step;
        const auto &src = Records[src_id];
        auto &dst = Records[id];
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
            const auto *from = ranges(src);
            if (!from || info.Mirror || (!render && info.RenderRange)) return;
            auto *to = ranges(dst); // NOLINT(misc-const-correctness)
            using Allocation = std::remove_cvref_t<decltype(*from)>;
            if constexpr (requires { { arena.CloneMembership(*from) } -> std::same_as<Allocation>; }) *to = arena.CloneMembership(*from);
            else if constexpr (requires { { arena.Allocate(from->Count) } -> std::same_as<Allocation>; }) *to = arena.Allocate(from->Count);
        });
        ids.push_back(id);
    }
    SyncMirrors();
    for (uint32_t i = 0u; i < ids.size(); ++i) {
        const auto &src = Records[source_ids[i]];
        pairs.push_back(PairBlocks(source_ids[i], ids[i]));
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &, auto &&ranges) {
            const auto *from = ranges(src);
            if constexpr (std::same_as<std::remove_cvref_t<decltype(*from)>, Range>) {
                if (const auto *to = ranges(Records[ids[i]]); from && to->Count) CaptureCopy(copies, arena.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), from->Offset, to->Offset, from->Count);
            }
        });
        std::vector<uint32_t> attributes;
        ForEachNamedAttribute(source_ids[i], [&](uint32_t attribute, const NamedAttribute &) { attributes.push_back(attribute); });
        for (const auto attribute : attributes) {
            const auto entry = NamedAttributes[attribute];
            const auto values = WithAttributeValues(Buffers.Named, entry.Type, [&](auto &arena) {
                const auto range = arena.Allocate(entry.Values.Count);
                if (range.Count) CaptureCopy(copies, arena.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), entry.Values.Offset, range.Offset, range.Count);
                return range;
            });
            AcquireAttribute({.Record = ids[i], .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Values = values, .Default = entry.Default});
        }
        CopyBlocks(copies, ids[i], source_ids[i], pairs[i]);
        auto &derived = DerivedRecords[ids[i]];
        if (derived.SelectionBaselineElement != Element::None) {
            const auto domain = SelectionDomain(derived.SelectionBaselineElement);
            WithDomain(Buffers, domain, [&](const auto &arena) {
                const auto owner = DomainSet(src, domain).Index;
                const auto to = arena.Entries(DomainSet(Records[ids[i]], domain));
                std::erase_if(derived.SelectionBaseline, [&](const auto &entry) { return arena.Blocks.Get({entry.first, 1u})[0].Owner != owner; });
                for (auto &[block, words] : derived.SelectionBaseline) block = to[arena.Blocks.Get({block, 1u})[0].Local];
            });
            std::ranges::sort(derived.SelectionBaseline, {}, &std::pair<uint32_t, SelectionBlock>::first);
        }
    }
    if (render) CloneRenderRecords(copies, source_ids, ids);
    return ids;
}

void MeshStore::ResetEditState(state::Registry &r, mtl::ComputeChain &chain, uint32_t id, uint32_t source) {
    SelectionUpdate update{.StoreId = id};
    auto pairs = PairBlocks(source, id);
    for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
        auto &paired = pairs[uint32_t(SelectionDomains[s])];
        std::ranges::sort(paired, {}, &uvec2::y);
        const auto selection = SelectionArena(Buffers, SelectionDomains[s]).Buffer.GetSpan<SelectionBlock>();
        const auto hidden = HiddenArena(Buffers, SelectionDomains[s]).Buffer.GetSpan<SelectionBlock>();
        std::vector<uint32_t> selected, shown;
        for (const auto [from, to] : paired) {
            if (selection[to] != SelectionBlock{}) selected.push_back(to);
            if (hidden[to] != hidden[from]) shown.push_back(to);
        }
        EditSelectionBlocks(SelectionElements[s], selected, [](uint32_t, auto &words) { words = {}; });
        EditHiddenBlocks(SelectionElements[s], shown, [&](uint32_t block, auto &words) {
            words = hidden[std::ranges::lower_bound(paired, block, {}, &uvec2::y)->x];
        });
        std::ranges::set_union(selected, shown, std::back_inserter(update.Blocks[s]));
    }
    if (Records.at(id).SelectionSummary.Count) UpdateSelection(r, chain, std::array{update});
}

void MeshStore::SelectMasks(state::Registry &r, mtl::ComputeChain &chain, uint32_t id, const GeometryMasks &masks) {
    SelectionUpdate update{.StoreId = id};
    for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
        const auto from = masks.Get(SelectionElements[s]);
        const auto selection = SelectionArena(Buffers, SelectionDomains[s]).Buffer.GetSpan<SelectionBlock>();
        const auto members = WithDomain(Buffers, SelectionDomains[s], [](const auto &arena) { return arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(); });
        // A selection marks live elements only.
        const auto mask = [&](uint32_t block) {
            auto words = from.Arena->Get({from.First + block, 1u})[0];
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[w] &= members[block].Live[w];
            return words;
        };
        for (const auto block : GetBlockList(id, SelectionDomains[s]).Blocks) {
            if (selection[block] != mask(block)) update.Blocks[s].push_back(block);
        }
        EditSelectionBlocks(SelectionElements[s], update.Blocks[s], [&](uint32_t block, auto &words) { words = mask(block); });
    }
    if (Records.at(id).SelectionSummary.Count && std::ranges::any_of(update.Blocks, [](const auto &blocks) { return !blocks.empty(); })) UpdateSelection(r, chain, std::array{update});
}

namespace {
// ORs the first `bits` bits of `from` into `to` from bit `at`.
void OrBits(std::span<uint32_t> to, uint64_t at, std::span<const uint32_t> from, uint32_t bits) {
    const auto shift = uint32_t(at % 32u);
    for (uint64_t i = 0u, word = at / 32u; i * 32u < bits; ++i, ++word) {
        const auto valid = std::min<uint64_t>(32u, bits - i * 32u);
        const auto value = valid == 32u ? from[i] : from[i] & ((1u << valid) - 1u);
        to[word] |= value << shift;
        if (shift && value >> (32u - shift)) to[word + 1u] |= value >> (32u - shift);
    }
}
// The mask words `words_of(local, block)` of a set's local blocks below the bound, with zeros at a local block without a block.
std::vector<uint32_t> SetBits(std::span<const uint32_t> entries, uint32_t bound, auto &&words_of) {
    std::vector<uint32_t> bits(uint64_t(ElementArena<uint32_t>::BlockCount(bound)) * MeshElementBlockWords, 0u);
    for (uint32_t local = 0u; local * MeshElementBlockSize < bound; ++local) {
        if (entries[local] != InvalidOffset) std::ranges::copy(words_of(local, entries[local]), bits.begin() + local * MeshElementBlockWords);
    }
    return bits;
}
// A mask's words over the blocks of a dense set, as one bit per handle.
std::span<uint32_t> BlockBits(auto &masks, Range blocks) {
    const auto words = masks.GetMutable(blocks);
    return {words.empty() ? nullptr : words[0].data(), words.size() * MeshElementBlockWords};
}
} // namespace

std::vector<uint32_t> MeshStore::Concatenate(state::Registry &r, mtl::ComputeChain &chain, std::span<const std::vector<ConcatPart>> concatenations) {
    const profile::CpuScope scope{"Concatenate"};
    // A part's live handle bound and the target handle of its first copy, by domain, with the primitive offset and step of its copies past the first part's first.
    struct Placed {
        std::array<uint32_t, 6> Bound{}, First{};
        uint32_t PrimitiveBase{}, PrimitiveStep{};
        bool Tabled{};
    };
    struct Output {
        uint32_t Id;
        std::vector<Placed> Parts;
        std::array<uint32_t, 6> Total{};
        std::vector<uint32_t> Entries; // Each layer's entries per handle, in ForEachLayer order
    };
    std::vector<Output> outputs;
    std::vector<uint32_t> reclassified;
    for (const auto &parts : concatenations) {
        const auto &first = Records.at(parts.front().StoreId);
        Record record{first};
        ResetRender(record);
        bool reclassify = false;
        for (const auto &part : parts | std::views::drop(1)) {
            const auto &src = Records.at(part.StoreId);
            record.CornerAttributes |= src.CornerAttributes;
            record.VertexAttributes |= src.VertexAttributes;
            record.FacePrimitivesReady |= src.FacePrimitivesReady;
            record.VertexPrimitivesReady |= src.VertexPrimitivesReady;
            record.ConnectivityFaceStarts |= src.ConnectivityFaceStarts;
            record.HasAuthoredNormals |= src.HasAuthoredNormals;
            reclassify = reclassify || first.Classification != src.Classification || bool(first.CornerSectors.Count) != bool(src.CornerSectors.Count);
        }
        Output output;
        std::array<bool, 6> present{};
        // Each copy keeps its part's corner pairs.
        record.UnpairedCorners = 0u;
        for (const auto &part : parts) {
            record.UnpairedCorners += Records[part.StoreId].UnpairedCorners * part.Copies;
            auto &placed = output.Parts.emplace_back();
            for (const auto domain : ElementDomains) {
                WithDomain(Buffers, domain, [&](const auto &arena) {
                    const auto d = uint32_t(domain);
                    const auto set = DomainSet(Records[part.StoreId], domain);
                    present[d] = present[d] || bool(set);
                    placed.Bound[d] = arena.LiveBound(set);
                    placed.First[d] = output.Total[d];
                    output.Total[d] += placed.Bound[d] * part.Copies;
                });
            }
        }
        ForEachLayer(Buffers, [&](auto &, const LayerInfo &info, auto &&entries, auto &&) { output.Entries.push_back(info.Render ? 0u : entries(record)); });
        // The primitive table joins each copy's table, a copy without one taking one primitive when the table exists.
        std::vector<uint32_t> table{std::from_range, Buffers.PrimitiveMaterials.Get(first.PrimitiveMaterials)};
        for (uint32_t p = 0u; p < parts.size(); ++p) {
            const auto copies = parts[p].Copies - uint32_t(p == 0u);
            const auto theirs = Buffers.PrimitiveMaterials.Get(Records[parts[p].StoreId].PrimitiveMaterials);
            auto &placed = output.Parts[p];
            placed.Tabled = copies && (!table.empty() || !theirs.empty());
            if (!placed.Tabled) continue;
            if (table.empty()) table.push_back(0u);
            placed.PrimitiveBase = uint32_t(table.size());
            placed.PrimitiveStep = std::max(uint32_t(theirs.size()), 1u);
            for (uint32_t c = 0u; c < copies; ++c) {
                if (theirs.empty()) table.push_back(0u);
                else table.append_range(theirs);
            }
        }
        std::vector<NamedAttribute> attributes;
        for (const auto &part : parts) {
            ForEachNamedAttribute(part.StoreId, [&](uint32_t, const NamedAttribute &entry) {
                if (std::ranges::none_of(attributes, [&](const NamedAttribute &a) { return a.NameView() == entry.NameView(); })) attributes.push_back({.Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default});
            });
        }
        ForEachArena(Buffers, [&](auto &, const ArenaInfo &, auto &&ranges) {
            if (auto *allocation = ranges(record)) *allocation = {};
        });
        record.ExtrasFaces = record.ExtrasEdges = {};
        const auto summary = first.SelectionSummary.Count, normals = first.PointNormals.Count;
        output.Id = AcquireId(std::move(record));
        const auto id = output.Id;
        DerivedRecords[id].NormalRevision = ++NextNormalRevision;
        for (const auto domain : ElementDomains) {
            const auto d = uint32_t(domain);
            if (!present[d]) continue;
            WithDomain(Buffers, domain, [&](auto &arena) {
                std::vector<uint32_t> live(uint64_t(std::max(1u, arena.BlockCount(output.Total[d]))) * MeshElementBlockWords, 0u);
                const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
                for (uint32_t p = 0u; p < parts.size(); ++p) {
                    const auto &placed = output.Parts[p];
                    const auto bits = SetBits(arena.Entries(DomainSet(Records[parts[p].StoreId], domain)), placed.Bound[d], [&](uint32_t, uint32_t block) { return members[block].Live; });
                    for (uint32_t c = 0u; c < parts[p].Copies; ++c) OrBits(live, placed.First[d] + uint64_t(c) * placed.Bound[d], bits, placed.Bound[d]);
                }
                DomainSet(WriteRecord(id), domain) = arena.CreateLive(live, output.Total[d]);
            });
        }
        const auto blocks_of = [&](Domain domain) { return WithDomain(Buffers, domain, [&](const auto &arena) { return arena.Entries(DomainSet(Records[id], domain)).size(); }); };
        ForEachLayer(Buffers, [&, layer = 0u](auto &target, const LayerInfo &info, auto &&, auto &&range) mutable {
            if (const auto per = output.Entries[layer++]) *range(WriteRecord(id)) = target.Values.Allocate(uint32_t(blocks_of(info.Elements)) * per);
        });
        for (auto &attribute : attributes) {
            attribute.Record = id;
            attribute.Values = WithAttributeValues(Buffers.Named, attribute.Type, [&](auto &values) { return values.Allocate(uint32_t(blocks_of(ElementDomainOf(attribute.Domain)))); });
            AcquireAttribute(std::move(attribute));
        }
        auto &created = WriteRecord(id);
        if (!table.empty()) created.PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(table);
        created.SelectionSummary = Buffers.SelectionSummary.Allocate(summary);
        created.PointNormals = Buffers.PointNormals.Allocate(normals);
        if (reclassify) reclassified.push_back(id);
        outputs.push_back(std::move(output));
    }
    SyncMirrors();
    CloneCopies copies;
    std::vector<PlaceCopiesJob> placements;
    std::vector<SelectionUpdate> updates;
    for (uint32_t o = 0u; o < outputs.size(); ++o) {
        const auto &parts = concatenations[o];
        const auto &output = outputs[o];
        const auto id = output.Id;
        const auto &first = Records[parts.front().StoreId];
        CaptureCopy(copies, Buffers.SelectionSummary.Buffer, sizeof(EditSelectionSummary), first.SelectionSummary.Offset, Records[id].SelectionSummary.Offset, Records[id].SelectionSummary.Count);
        CaptureCopy(copies, Buffers.PointNormals.Buffer, sizeof(vec3), first.PointNormals.Offset, Records[id].PointNormals.Offset, Records[id].PointNormals.Count);
        // The first address of the record's handles in each domain, whose blocks are consecutive.
        std::array<uint32_t, 6> base{}, block_counts{};
        for (const auto domain : ElementDomains) {
            WithDomain(Buffers, domain, [&](const auto &arena) {
                const auto entries = arena.Entries(DomainSet(Records[id], domain));
                base[uint32_t(domain)] = entries.empty() ? 0u : entries[0] * MeshElementBlockSize;
                block_counts[uint32_t(domain)] = uint32_t(entries.size());
            });
        }
        std::vector<std::span<uint32_t>> masks;
        for (const auto domain : SelectionDomains) {
            const Range blocks{base[uint32_t(domain)] / MeshElementBlockSize, block_counts[uint32_t(domain)]};
            for (auto *arena : {&SelectionArena(Buffers, domain), &HiddenArena(Buffers, domain)}) std::ranges::fill(masks.emplace_back(BlockBits(*arena, blocks)), 0u);
        }
        std::vector<std::pair<uint32_t, std::span<uint32_t>>> bools;
        ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
            if (entry.Type == AttributeType::Bool) std::ranges::fill(bools.emplace_back(attribute, BlockBits(Buffers.Named.Bools, entry.Values)).second, 0u);
        });
        for (uint32_t p = 0u; p < parts.size(); ++p) {
            const auto [source, n, transforms] = parts[p];
            const auto &src = Records[source];
            const auto &placed = output.Parts[p];
            const auto &bound = placed.Bound;
            ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
                using Allocation = std::remove_cvref_t<decltype(*ranges(src))>;
                if constexpr (std::same_as<Allocation, ElementSetRef>) {
                    if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
                    const auto d = uint32_t(info.Elements);
                    if (!bound[d]) return;
                    constexpr auto Size = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
                    const auto to = base[d] + placed.First[d];
                    arena.Buffer.CaptureWrite(to * Size, uint64_t(bound[d]) * n * Size);
                    const auto entries = WithDomain(Buffers, info.Elements, [&](const auto &elements) { return elements.Entries(DomainSet(src, info.Elements)); });
                    for (uint32_t local = 0u; local * MeshElementBlockSize < bound[d]; ++local) {
                        if (entries[local] == InvalidOffset) continue;
                        const auto slots = std::min(MeshElementBlockSize, bound[d] - local * MeshElementBlockSize);
                        copies.Copy(arena.Buffer, {.Source = uint64_t(entries[local]) * MeshElementBlockSize * Size, .Destination = (uint64_t(to) + local * MeshElementBlockSize) * Size, .Bytes = slots * Size, .Step = bound[d] * Size, .Copies = n});
                    }
                    if (info.References == Domain::None) return;
                    const auto reference = uint32_t(info.References);
                    copies.Offset(arena.Buffer, {.ByteOffset = to * Size, .Count = uint32_t(bound[d] * (Size / sizeof(uint32_t))), .Shift = placed.First[reference], .Copies = n, .Step = bound[d] * Size, .ShiftStep = bound[reference]});
                }
            });
            for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
                const auto domain = SelectionDomains[s];
                const auto d = uint32_t(domain);
                const auto entries = WithDomain(Buffers, domain, [&](const auto &elements) { return elements.Entries(DomainSet(src, domain)); });
                for (uint32_t m = 0u; m < 2u; ++m) {
                    const auto &arena = m ? HiddenArena(Buffers, domain) : SelectionArena(Buffers, domain);
                    const auto bits = SetBits(entries, bound[d], [&](uint32_t, uint32_t block) { return arena.Get({block, 1u})[0]; });
                    for (uint32_t c = 0u; c < n; ++c) OrBits(masks[2u * s + m], placed.First[d] + uint64_t(c) * bound[d], bits, bound[d]);
                }
            }
            {
                ForEachLayer(Buffers, [&, layer = 0u](auto &target, const LayerInfo &info, auto &&entries, auto &&range) mutable {
                    const auto per = output.Entries[layer++];
                    const auto d = uint32_t(info.Elements);
                    if (!per || !bound[d]) return;
                    using T = typename std::remove_cvref_t<decltype(target)>::Block::value_type;
                    const auto to = *range(Records[id]), from = *range(src);
                    const auto held = from.Count ? entries(src) : 0u;
                    target.Capture(to, {placed.First[d], bound[d] * n}, per);
                    if (per == 1u && held == 1u) {
                        copies.Copy(target.Values.Buffer, {.Source = uint64_t(from.Offset) * MeshElementBlockSize * sizeof(T), .Destination = (uint64_t(to.Offset) * MeshElementBlockSize + placed.First[d]) * sizeof(T), .Bytes = uint64_t(bound[d]) * sizeof(T), .Step = uint64_t(bound[d]) * sizeof(T), .Copies = n});
                    } else {
                        for (uint32_t c = 0u; c < n; ++c) {
                            for (uint32_t h = 0u; h < bound[d];) {
                                const auto at = placed.First[d] + c * bound[d] + h;
                                const auto count = std::min({MeshElementBlockSize - h % MeshElementBlockSize, MeshElementBlockSize - at % MeshElementBlockSize, bound[d] - h});
                                for (uint32_t e = 0u; e < per; ++e) {
                                    const auto destination = target.Index(to, at, e, per);
                                    if (e < held) copies.Copy(target.Values.Buffer, {.Source = uint64_t(target.Index(from, h, e, held)) * sizeof(T), .Destination = uint64_t(destination) * sizeof(T), .Bytes = uint64_t(count) * sizeof(T)});
                                    else std::ranges::fill(target.Values.Buffer.template GetMutableSpan<T>({destination, count}), target.Absent);
                                }
                                h += count;
                            }
                        }
                    }
                    if constexpr (std::same_as<T, uint32_t>) {
                        const auto values = uint64_t(to.Offset) * MeshElementBlockSize + placed.First[d];
                        // Sector roots are halfedge handles, and primitive indices read past the primitives of the copies before them.
                        if (&target == &Buffers.CornerSectors) copies.Offset(target.Values.Buffer, {.ByteOffset = values * 4u, .Count = bound[d], .Shift = placed.First[d], .Copies = n, .Step = bound[d] * 4u, .ShiftStep = bound[d]});
                        const bool primitives = &target == &Buffers.FacePrimitives || &target == &Buffers.VertexPrimitives;
                        if (primitives && placed.Tabled) {
                            const auto skipped = uint32_t(p == 0u);
                            copies.Offset(target.Values.Buffer, {.ByteOffset = (values + uint64_t(skipped) * bound[d]) * 4u, .Count = bound[d], .Shift = placed.PrimitiveBase, .Copies = n - skipped, .Step = bound[d] * 4u, .ShiftStep = placed.PrimitiveStep});
                        }
                    }
                });
            }
            ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
                const auto d = uint32_t(ElementDomainOf(entry.Domain));
                if (!bound[d]) return;
                const auto found = FindAttribute(source, entry.NameView());
                const auto from = found != InvalidOffset && NamedAttributes[found].Type == entry.Type && NamedAttributes[found].Domain == entry.Domain ? found : InvalidOffset;
                const auto at = uint64_t(entry.Values.Offset) * MeshElementBlockSize + placed.First[d];
                WithAttributeScalar(entry.Type, [&]<AttributeScalar T>() {
                    auto &values = ValuesOf<T>(Buffers.Named);
                    if constexpr (std::same_as<T, bool>) {
                        const auto bits = from != InvalidOffset ?
                            SetBits(WithDomain(Buffers, Domain(d), [&](const auto &elements) { return elements.Entries(DomainSet(src, Domain(d))); }), bound[d], [&](uint32_t local, uint32_t) { return values.Get({NamedAttributes[from].Values.Offset + local, 1u})[0]; }) :
                            std::vector<uint32_t>(ElementArena<uint32_t>::BlockCount(bound[d]) * MeshElementBlockWords, entry.Default[0] ? ~0u : 0u);
                        auto &mask = std::ranges::find(bools, attribute, &std::pair<uint32_t, std::span<uint32_t>>::first)->second;
                        for (uint32_t c = 0u; c < n; ++c) OrBits(mask, placed.First[d] + uint64_t(c) * bound[d], bits, bound[d]);
                    } else if (from != InvalidOffset) {
                        values.Buffer.CaptureWrite(at * sizeof(T), uint64_t(bound[d]) * n * sizeof(T));
                        copies.Copy(values.Buffer, {.Source = uint64_t(NamedAttributes[from].Values.Offset) * MeshElementBlockSize * sizeof(T), .Destination = at * sizeof(T), .Bytes = uint64_t(bound[d]) * sizeof(T), .Step = uint64_t(bound[d]) * sizeof(T), .Copies = n});
                    } else {
                        T value;
                        std::memcpy(&value, entry.Default.data(), sizeof(T));
                        std::ranges::fill(values.Buffer.template GetMutableSpan<T>({uint32_t(at), bound[d] * n}), value);
                    }
                });
            });
            // Each copy's fan items follow those of the copy before in vertex order, and its roots are the first copy's shifted by the items before it.
            constexpr auto V = uint32_t(Domain::Vertex), H = uint32_t(Domain::Halfedge), F = uint32_t(Domain::Face);
            if (bound[V]) {
                const auto at = base[V] + placed.First[V];
                Buffers.VertexCorners.CaptureWrite({at, bound[V] * n});
                const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
                const auto entries = Buffers.Vertices.Entries(src.Vertices);
                std::vector<uvec2> runs, written(bound[V], uvec2{InvalidOffset, 0u});
                uint32_t items = 0u;
                for (uint32_t local = 0u; local * MeshElementBlockSize < bound[V]; ++local) {
                    if (entries[local] == InvalidOffset) continue;
                    Buffers.Vertices.ForEachSlot(entries[local], [&](uint32_t slot) {
                        const auto root = roots[entries[local] * MeshElementBlockSize + slot];
                        if (!root.y) return;
                        written[local * MeshElementBlockSize + slot] = {items, root.y};
                        if (runs.empty() || runs.back().x + runs.back().y != root.x) runs.push_back({root.x, 0u});
                        runs.back().y += root.y;
                        items += root.y;
                    });
                }
                const auto fans = Buffers.VertexFans.Items.Allocate(items * n);
                Buffers.VertexFans.Items.CaptureWrite(fans);
                for (auto &root : written) root.x = root.y ? root.x + fans.Offset : InvalidOffset;
                std::ranges::copy(written, Buffers.VertexCorners.GetMutable({at, bound[V]}).begin());
                for (uint32_t next = fans.Offset; const auto run : runs) {
                    copies.Copy(Buffers.VertexFans.Items.Buffer, {.Source = uint64_t(run.x) * sizeof(uvec2), .Destination = uint64_t(next) * sizeof(uvec2), .Bytes = uint64_t(run.y) * sizeof(uvec2), .Step = uint64_t(items) * sizeof(uvec2), .Copies = n});
                    next += run.y;
                }
                for (const auto [word, domain] : {std::pair{0u, H}, std::pair{1u, F}}) {
                    copies.Offset(Buffers.VertexFans.Items.Buffer, {.ByteOffset = uint64_t(fans.Offset) * sizeof(uvec2) + word * 4u, .Count = items, .Stride = 2u, .Shift = placed.First[domain], .Copies = n, .Step = uint64_t(items) * sizeof(uvec2), .ShiftStep = bound[domain]});
                }
                copies.Copy(Buffers.VertexCorners.Buffer, {.Source = uint64_t(at) * sizeof(uvec2), .Destination = uint64_t(at + bound[V]) * sizeof(uvec2), .Bytes = uint64_t(bound[V]) * sizeof(uvec2), .Step = uint64_t(bound[V]) * sizeof(uvec2), .Copies = n - 1u});
                copies.Offset(Buffers.VertexCorners.Buffer, {.ByteOffset = uint64_t(at + bound[V]) * sizeof(uvec2), .Count = bound[V], .Stride = 2u, .Shift = items, .Copies = n - 1u, .Step = uint64_t(bound[V]) * sizeof(uvec2), .ShiftStep = items});
                if (transforms.Slot != InvalidSlot) placements.push_back({.Transforms = transforms, .First = at, .Bound = bound[V], .Count = bound[V] * n, .VertexSlot = Buffers.Vertices.Buffer.Slot, .MembershipSlot = Buffers.Vertices.Blocks.Buffer.Slot});
            }
        }
        if (const auto total = output.Total[uint32_t(Domain::Vertex)], end = block_counts[uint32_t(Domain::Vertex)] * MeshElementBlockSize; total < end) {
            const Range tail{base[uint32_t(Domain::Vertex)] + total, end - total};
            Buffers.VertexCorners.CaptureWrite(tail);
            std::ranges::fill(Buffers.VertexCorners.GetMutable(tail), uvec2{InvalidOffset, 0u});
        }
        WriteRecord(id).TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
        auto &update = updates.emplace_back(SelectionUpdate{.StoreId = id});
        for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
            const auto d = uint32_t(SelectionDomains[s]);
            for (uint32_t b = 0u; b < block_counts[d]; ++b) update.Blocks[s].push_back(base[d] / MeshElementBlockSize + b);
        }
    }
    const auto &pipelines = GetMeshPipelines(r);
    copies.Encode(chain, pipelines);
    if (!placements.empty()) mtl::GatherPass(chain, pipelines[MeshPass::PlaceCopies], std::span{placements}, &PlaceCopiesJob::Count);
    if (!reclassified.empty()) UpdateCornerClassification(r, chain, reclassified);
    UpdateSelection(r, chain, updates);
    return outputs | std::views::transform(&Output::Id) | std::ranges::to<std::vector>();
}


void MeshStore::CloneRenderRecords(CloneCopies &copies, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids) {
    auto &render = Buffers.Render;
    std::vector<uint32_t> sources, clones;
    for (uint32_t i = 0u; i < source_ids.size(); ++i) {
        const auto &source = Records[source_ids[i]];
        if (!HasRender(source)) continue;
        auto &target = Records[clone_ids[i]];
        target.RenderTopologies = source.RenderTopologies;
        target.Level0Count = source.Level0Count;
        target.MeshletRevision = source.MeshletRevision;
        target.LodDepth = source.LodDepth;
        target.SpatialRoot = source.SpatialRoot;
        target.Meshlets = source.Meshlets;
        target.ElementMeshletBlockCounts = source.ElementMeshletBlockCounts;
        for (const auto set : RenderSets) WithRenderSet(render, set, [&](auto &arena) { RenderSetOf(target, set) = arena.CloneMembership(RenderSetOf(source, set)); });
        for (const auto payload : RenderPayloads) WithPayload(render, target, payload, [](auto &, PayloadRange &range) { range.Capacity = range.Values.Count; });
        sources.push_back(source_ids[i]);
        clones.push_back(clone_ids[i]);
    }
    if (sources.empty()) return;
    render.SyncMirrors();
    for (uint32_t c = 0u; c < sources.size(); ++c) {
        const auto &source = Records[sources[c]];
        auto &target = Records[clones[c]];
        std::array<std::vector<uvec2>, RenderSets.size()> pairs;
        for (const auto set : RenderSets) {
            WithRenderSet(render, set, [&](const auto &arena) {
                const auto from = arena.Entries(RenderSetOf(source, set)), to = arena.Entries(RenderSetOf(target, set));
                for (uint32_t local = 0u; local < from.size(); ++local) {
                    if (from[local] != InvalidOffset) pairs[uint32_t(set)].push_back({from[local], to[local]});
                }
            });
        }
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
            if (!info.Render) return;
            constexpr auto Size = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
            const auto block_size = WithRenderSet(render, *info.Render, [](const auto &keys) { return std::remove_cvref_t<decltype(keys)>::BlockSize; });
            const auto stride = info.BlockIndexed ? Size : Size * block_size;
            ForEachPairRun(pairs[uint32_t(*info.Render)], [&](uint32_t from, uint32_t to, uint32_t count) {
                CaptureCopy(copies, arena.Buffer, stride, from, to, count);
                for (const auto &field : info.Fields) {
                    if (!field.Target && !field.Payload) copies.Offset(arena.Buffer, {.ByteOffset = uint64_t(to) * stride + field.Offset, .Count = count * block_size, .Stride = uint32_t(Size / sizeof(uint32_t)), .Shift = target.PrimitiveMaterials.Offset - source.PrimitiveMaterials.Offset});
                }
            });
        });
    }
}

namespace {
template<uint32_t BlockSize> struct SetTable {
    std::span<const uint32_t> Entries;
    std::span<const MeshElementBlock> Blocks;
    ElementAddress Address(ElementHandle handle) const { return Resolve<BlockSize>(Entries, handle); }
    bool IsLive(uint32_t handle) const {
        const auto local = handle / BlockSize, slot = handle % BlockSize;
        return local < Entries.size() && Entries[local] != InvalidOffset && (Blocks[Entries[local]].Live[slot / 32u] >> (slot % 32u) & 1u);
    }
    void ForEachLive(auto &&fn) const {
        for (uint32_t local = 0u; local < Entries.size(); ++local) {
            if (Entries[local] == InvalidOffset) continue;
            for (uint32_t w = 0u; w * 32u < BlockSize; ++w) {
                for (auto bits = Blocks[Entries[local]].Live[w]; bits; bits &= bits - 1u) {
                    const auto slot = w * 32u + uint32_t(std::countr_zero(bits));
                    fn(local * BlockSize + slot, Entries[local] * BlockSize + slot);
                }
            }
        }
    }
};
template<typename Arena> SetTable<Arena::BlockSize> TableOf(const Arena &arena, ElementSetRef set) { return {arena.Entries(set), arena.Blocks.Buffer.template GetSpan<MeshElementBlock>()}; }
using ElementTable = SetTable<MeshElementBlockSize>;
using RenderTable = SetTable<RenderBlockSize>;

struct RecordTables {
    std::array<ElementTable, 6> Sets;
    RecordTables(const MeshArenas &b, const MeshStore::Record &r) {
        for (const auto domain : MeshStore::ElementDomains) WithDomain(b, domain, [&](const auto &arena) { Sets[uint32_t(domain)] = TableOf(arena, DomainSet(r, domain)); });
    }
    ElementAddress Address(MeshStore::ElementDomain domain, ElementHandle handle) const { return Sets[uint32_t(domain)].Address(handle); }
    bool IsLive(MeshStore::ElementDomain domain, uint32_t handle) const { return Sets[uint32_t(domain)].IsLive(handle); }
    void ForEachLive(MeshStore::ElementDomain domain, auto &&fn) const { Sets[uint32_t(domain)].ForEachLive(fn); }
};
struct RenderTables {
    std::array<RenderTable, RenderSets.size()> Sets;
    RenderTables(const RenderArenas &render, const MeshStore::Record &r) {
        for (const auto set : RenderSets) WithRenderSet(render, set, [&](const auto &arena) { Sets[uint32_t(set)] = TableOf(arena, RenderSetOf(r, set)); });
    }
    const RenderTable &operator[](RenderSet set) const { return Sets[uint32_t(set)]; }
};

// Spatial nodes visit at the addresses of the finest triangle clusters.
void ForEachRenderValue(const MeshArenas &b, const MeshStore::Record &r, const RenderTables &tables, auto &&fn) {
    const auto &render = b.Render;
    ForEachArena(b, [&](const auto &arena, const ArenaInfo &info, auto &&ranges) {
        using Value = ArenaValue<std::remove_cvref_t<decltype(arena)>>;
        const auto bytes = std::as_bytes(arena.Buffer.template GetSpan<Value>());
        const auto visit = [&](uint32_t index) { fn(info, bytes.subspan(uint64_t(index) * sizeof(Value), sizeof(Value))); };
        if constexpr (std::same_as<std::remove_cvref_t<decltype(*ranges(r))>, Range>) {
            if (const auto *range = ranges(r); range && info.RenderRange && info.Layer == Domain::None) {
                for (uint32_t i = 0u; i < range->Count; ++i) visit(range->Offset + i);
            }
        }
        if (!info.Render || info.BlockIndexed) return;
        if (&arena.Buffer != &render.MeshletSpatialNodes.Buffer) return tables[*info.Render].ForEachLive([&](uint32_t, uint32_t address) { visit(address); });
        if (r.SpatialRoot == InvalidOffset) return;
        tables[RenderSet::Meshlets].ForEachLive([&](uint32_t, uint32_t address) {
            const auto &meshlet = render.Meshlets.Get({address, 1u})[0];
            if (meshlet.RefinedGroup == InvalidOffset && meshlet.Topology == 0u) visit(address);
        });
    });
}
uint32_t WordAt(std::span<const std::byte> value, uint32_t offset) {
    uint32_t word;
    std::memcpy(&word, value.data() + offset, sizeof(word));
    return word;
}

bool TableWellFormed(const auto &arena, ElementSetRef set) {
    if (set.Index >= arena.Sets.Buffer.template Count<MeshElementSet>()) return false;
    const auto &header = arena.Sets.Get({set.Index, 1u})[0];
    if (header.First == InvalidOffset || !header.BlockCount || header.BlockCount > header.Capacity || uint64_t(header.First) + header.Capacity > arena.Table.Buffer.template Count<uint32_t>()) return false;
    const auto entries = arena.Table.Get({header.First, header.BlockCount});
    if (entries.front() == InvalidOffset || entries.back() == InvalidOffset) return false;
    const auto blocks = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
    uint32_t count = 0u;
    for (uint32_t local = 0u; local < entries.size(); ++local) {
        const auto b = entries[local];
        if (b == InvalidOffset) continue;
        if (b >= blocks.size()) return false;
        const auto &block = blocks[b];
        uint32_t live = 0u;
        for (const auto word : block.Live) live += uint32_t(std::popcount(word));
        if (block.Owner != set.Index || block.Local != local || block.Count != live) return false;
        constexpr auto Size = std::remove_cvref_t<decltype(arena)>::BlockSize;
        if ((header.Flags & 1u) && (b != entries[0] + local || live != std::min(Size, header.Count > local * Size ? header.Count - local * Size : 0u))) return false;
        count += live;
    }
    return count == header.Count;
}
bool MaskWithinLive(const BufferArena<uint32_t> &masks, const RenderTable &table) {
    const auto words = masks.Buffer.GetSpan<uint32_t>();
    for (const auto block : table.Entries) {
        if (block != InvalidOffset && (block >= words.size() || words[block] & ~table.Blocks[block].Live[0])) return false;
    }
    return true;
}
Domain ActiveDomain(const EditSelectionSummary &summary) {
    const bool selectable = summary.Mode == Element::Vertex || summary.Mode == Element::Edge || summary.Mode == Element::Face;
    return selectable ? SelectionDomain(summary.Mode) : Domain::None;
}
constexpr std::array DomainNames{"none", "vertex", "halfedge", "edge", "face", "triangle"};
constexpr std::array RenderSetNames{"meshlet", "primitive", "node", "group", "member"};
} // namespace

bool MeshStore::Audit(std::string &why) const {
    const auto &render = Buffers.Render;
    std::array<std::vector<uint32_t>, 6> set_records;
    std::array<std::vector<uint32_t>, RenderSets.size()> render_set_records;
    for (uint32_t id = 0u; id < Records.size(); ++id) {
        const auto &r = Records[id];
        if (!r.Alive) continue;
        const auto fail = [&](const std::string &what) {
            why = "Mesh record " + std::to_string(id) + " " + what + ".";
            return false;
        };
        const auto claim = [&](std::vector<uint32_t> &records, ElementSetRef set) {
            if (set.Index >= records.size()) records.resize(set.Index + 1u, InvalidOffset);
            return std::exchange(records[set.Index], id) == InvalidOffset;
        };
        for (const auto domain : ElementDomains) {
            const auto set = DomainSet(r, domain);
            if (!set) continue;
            if (!claim(set_records[uint32_t(domain)], set)) return fail(std::string{"shares its "} + DomainNames[uint32_t(domain)] + " set");
            if (!WithDomain(Buffers, domain, [&](const auto &arena) { return TableWellFormed(arena, set); })) return fail(std::string{"has a malformed "} + DomainNames[uint32_t(domain)] + " table");
        }
        for (const auto set : RenderSets) {
            const auto ref = RenderSetOf(r, set);
            if (!ref) continue;
            if (!claim(render_set_records[uint32_t(set)], ref)) return fail(std::string{"shares its "} + RenderSetNames[uint32_t(set)] + " set");
            if (!WithRenderSet(render, set, [&](const auto &arena) { return TableWellFormed(arena, ref); })) return fail(std::string{"has a malformed "} + RenderSetNames[uint32_t(set)] + " table");
        }
        const RecordTables tables{Buffers, r};
        const auto names = [&](Domain domain, uint32_t handle) { return handle == InvalidOffset || tables.IsLive(domain, handle); };
        std::string invalid;
        ForEachArena(Buffers, [&](const auto &arena, const ArenaInfo &info, auto &&) {
            if (!invalid.empty() || info.References == Domain::None) return;
            constexpr auto Words = uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>) / sizeof(uint32_t));
            const auto words = arena.Buffer.template GetSpan<uint32_t>();
            tables.ForEachLive(info.Elements, [&](uint32_t handle, uint32_t address) {
                const auto value = words.subspan(uint64_t(address) * Words, Words);
                bool valid = true;
                if (info.ReferenceSpan) {
                    valid = value[0] < value[1];
                    for (auto h = value[0]; valid && h < value[1]; ++h) valid = tables.IsLive(info.References, h);
                } else {
                    for (const auto h : value) valid &= names(info.References, h);
                }
                if (!valid && invalid.empty()) invalid = std::string{info.Name} + " at " + std::to_string(handle);
            });
        });
        if (!invalid.empty()) return fail("names a foreign or dead element in " + invalid);
        const auto items = Buffers.VertexFans.Items.Buffer.GetSpan<uvec2>();
        const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
        tables.ForEachLive(Domain::Vertex, [&](uint32_t v, uint32_t address) {
            const auto root = roots[address];
            if (!root.y || !invalid.empty()) return;
            if (uint64_t(root.x) + root.y > items.size()) invalid = "fan of vertex " + std::to_string(v);
            for (uint32_t i = 0u; invalid.empty() && i < root.y; ++i) {
                const auto item = items[root.x + i];
                if (!tables.IsLive(Domain::Halfedge, item.x) || !names(Domain::Face, item.y)) invalid = "fan item of vertex " + std::to_string(v);
            }
        });
        if (!invalid.empty()) return fail("names a foreign or dead element in its " + invalid);
        uint32_t unpaired = 0u;
        const auto owners = Buffers.HalfedgeFaces.Buffer.GetSpan<uint32_t>(), opposites = Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>();
        tables.ForEachLive(Domain::Halfedge, [&](uint32_t, uint32_t address) { unpaired += owners[address] != InvalidOffset && opposites[address] == InvalidOffset; });
        if (unpaired != r.UnpairedCorners) return fail("counts " + std::to_string(r.UnpairedCorners) + " unpaired face corners of " + std::to_string(unpaired));
        if (const auto sectors = Buffers.CornerSectors.View(r.CornerSectors); !sectors.empty()) {
            tables.ForEachLive(Domain::Halfedge, [&](uint32_t h, uint32_t) {
                if (invalid.empty() && !names(Domain::Halfedge, sectors[h])) invalid = "sector root of corner " + std::to_string(h);
            });
        }
        if (!invalid.empty()) return fail("names a foreign or dead element in its " + invalid);
        const auto covers = [&](Range range, uint32_t blocks, uint32_t high_water) { return range.Count >= blocks && range.Offset + range.Count <= high_water; };
        ForEachLayer(Buffers, [&](const auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
            const auto per = entries(r), blocks = uint32_t(tables.Sets[uint32_t(info.Elements)].Entries.size());
            if (invalid.empty() && !(per ? covers(*range(r), blocks * per, layer.Values.HighWaterMark()) : !range(r)->Count)) invalid = info.Name;
        });
        ForEachNamedAttribute(id, [&](uint32_t, const NamedAttribute &entry) {
            const auto blocks = uint32_t(tables.Sets[uint32_t(ElementDomainOf(entry.Domain))].Entries.size());
            if (invalid.empty() && !WithAttributeValues(Buffers.Named, entry.Type, [&](const auto &values) { return covers(entry.Values, blocks, values.HighWaterMark()); })) invalid = entry.NameView();
        });
        if (!invalid.empty()) return fail("has a short " + invalid + " range");
        if (r.SelectionSummary.Count) {
            const auto &summary = Buffers.SelectionSummary.Get(r.SelectionSummary)[0];
            const auto domain = ActiveDomain(summary);
            if (summary.ActiveHandle != InvalidOffset && (domain == Domain::None || !tables.IsLive(domain, summary.ActiveHandle))) return fail("has a foreign or dead active element");
        }
        if (!HasRender(r)) continue;
        const RenderTables members{render, r};
        const auto used = [&](RenderPayload payload) { return WithPayload(render, r, payload, [](const auto &, const PayloadRange &range) { return range.Values.Count; }); };
        if (std::ranges::any_of(RenderPayloads, [&](RenderPayload payload) { return WithPayload(render, r, payload, [](const auto &, const PayloadRange &range) { return range.Values.Count > range.Capacity; }); })) return fail("uses a payload range past its capacity");
        if (r.SpatialRoot != InvalidOffset && !members[RenderSet::Meshlets].IsLive(r.SpatialRoot)) return fail("has a foreign or dead spatial root");
        if (!MaskWithinLive(render.PositionDirty, members[RenderSet::Meshlets]) || !MaskWithinLive(render.DirtyGroups, members[RenderSet::Groups])) return fail("marks a dead render member dirty");
        ForEachRenderValue(Buffers, r, members, [&](const ArenaInfo &info, std::span<const std::byte> value) {
            for (const auto &field : info.Fields) {
                const auto handle = WordAt(value, field.Offset);
                const auto count = field.RunCount == InvalidOffset ? 1u : WordAt(value, field.RunCount);
                bool valid = true;
                if (field.Payload) valid = handle == InvalidOffset || uint64_t(handle) + count <= used(*field.Payload);
                else if (!field.Target) valid = handle == OffsetOrInvalid(r.PrimitiveMaterials);
                else {
                    for (uint32_t i = 0u; handle != InvalidOffset && i < count; ++i) valid &= members[*field.Target].IsLive(handle + i);
                }
                if (!valid && invalid.empty()) invalid = std::string{info.Name} + " field at byte " + std::to_string(field.Offset);
            }
        });
        for (uint32_t topology = 0u; topology < 3u; ++topology) {
            const auto owners = render.ElementMeshlets[topology].View(r.ElementMeshlets[topology]);
            if (!owners.empty()) {
                tables.ForEachLive(RenderDomain(topology), [&](uint32_t handle, uint32_t) {
                    if (owners[handle] != InvalidOffset && !members[RenderSet::Meshlets].IsLive(owners[handle]) && invalid.empty()) invalid = "element owners";
                });
            }
        }
        const auto group_ids = render.GroupClusterIds.Get(r.GroupClusters.Values);
        members[RenderSet::Groups].ForEachLive([&](uint32_t g, uint32_t address) {
            const auto &links = render.GroupLinks.Get({address, 1u})[0];
            for (const auto [first, count] : {std::pair{links.MemberOffset, links.MemberCount}, std::pair{links.ProxyOffset, links.ProxyCount}}) {
                for (uint32_t i = 0u; invalid.empty() && i < count; ++i) {
                    if (!members[RenderSet::Meshlets].IsLive(group_ids[first + i])) invalid = "group " + std::to_string(g);
                }
            }
        });
        if (!invalid.empty()) return fail("names a foreign render member in its " + invalid);
        members[RenderSet::Nodes].ForEachLive([&](uint32_t handle, uint32_t address) {
            const auto &node = render.LodNodes.Get({address, 1u})[0];
            if (invalid.empty() && (node.ChildCount ? node.Members != InvalidOffset : node.MeshletCount && node.Members == InvalidOffset)) invalid = "node " + std::to_string(handle);
        });
        if (!invalid.empty()) return fail("has a malformed traversal " + invalid);
        // An extras record's clusters draw the triangles of its extras index run by index and its corners by position in that run.
        const auto extras = r.ExtrasFaces;
        const auto triangle_ids = render.MeshletTriangleIds.Get(r.MeshletTriangles.Values), vertex_corners = render.MeshletVertexCorners.Get(r.MeshletVertices.Values);
        members[RenderSet::Meshlets].ForEachLive([&](uint32_t m, uint32_t address) {
            const auto &meshlet = render.Meshlets.Get({address, 1u})[0];
            if (!invalid.empty() || meshlet.Topology >= 3u) return;
            const auto domain = RenderDomain(meshlet.Topology);
            const auto owners = render.ElementMeshlets[meshlet.Topology].View(r.ElementMeshlets[meshlet.Topology]);
            if (meshlet.RefinedGroup == InvalidOffset) {
                for (uint32_t i = 0u; invalid.empty() && i < meshlet.TriangleCount; ++i) {
                    const auto handle = triangle_ids[meshlet.TriangleOffset + i];
                    const bool owned = extras.Count ? uint64_t(handle) * 3u + 3u <= extras.Count : tables.IsLive(domain, handle) && owners.GetOr(handle, InvalidOffset) == m;
                    if (!owned) invalid = "cluster " + std::to_string(m) + " element " + std::to_string(handle);
                }
            }
            // A coarse cluster of a stale group keeps the corners it was built from until the group rebuilds.
            if (meshlet.RefinedGroup != InvalidOffset && MaskHas(render.DirtyGroups, members[RenderSet::Groups].Address(ElementHandle{meshlet.RefinedGroup}))) return;
            const auto corner_domain = meshlet.Topology == 0u ? Domain::Halfedge : Domain::Vertex;
            for (uint32_t i = 0u; invalid.empty() && i < meshlet.VertexCount; ++i) {
                const auto corner = vertex_corners[meshlet.VertexOffset + i];
                const bool live = extras.Count ? corner >= extras.Offset && corner < extras.Offset + extras.Count : tables.IsLive(corner_domain, corner);
                if (!live) invalid = "cluster " + std::to_string(m) + " vertex " + std::to_string(corner);
            }
        });
        if (!invalid.empty()) return fail("draws a foreign, dead or unowned element in " + invalid);
    }
    return true;
}

store::Hash128 MeshStore::CanonicalHash(uint32_t id) const {
    const auto &r = Records.at(id);
    const auto &render = Buffers.Render;
    std::vector<std::byte> bytes;
    const auto put = [&](const auto &value) {
        const auto *first = reinterpret_cast<const std::byte *>(&value);
        bytes.insert(bytes.end(), first, first + sizeof(value));
    };
    const auto put_all = [&](auto &&values) {
        const auto view = std::as_bytes(std::span{values});
        bytes.insert(bytes.end(), view.begin(), view.end());
    };
    put(r.CornerAttributes), put(r.VertexAttributes), put(r.FacePrimitivesReady), put(r.VertexPrimitivesReady), put(r.ConnectivityFaceStarts);
    put(r.SkinBlocksReady), put(r.MorphBlocksReady), put(r.MorphTargetCount), put(r.TriangleCount), put(r.UnpairedCorners), put(r.HasAuthoredNormals), put(r.MorphShadingAuthored);
    put_all(r.DefaultMorphWeights);
    const RecordTables tables{Buffers, r};
    const auto put_shape = [&](const auto &table) {
        put(uint32_t(table.Entries.size()));
        for (const auto block : table.Entries) put(block == InvalidOffset ? decltype(MeshElementBlock::Live){} : table.Blocks[block].Live);
    };
    for (const auto domain : ElementDomains) put_shape(tables.Sets[uint32_t(domain)]);
    ForEachArena(Buffers, [&](const auto &arena, const ArenaInfo &info, auto &&ranges) {
        using Value = ArenaValue<std::remove_cvref_t<decltype(arena)>>;
        using Allocation = std::remove_cvref_t<decltype(*ranges(r))>;
        if (!info.Bits) return;
        const auto values = arena.Buffer.template GetSpan<Value>();
        if constexpr (std::same_as<Allocation, ElementSetRef>) {
            if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
            tables.ForEachLive(info.Elements, [&](uint32_t, uint32_t address) { put(values[address]); });
        } else if (const auto *range = ranges(r); range && range->Count) {
            put_all(values.subspan(range->Offset, range->Count));
        }
    });
    const auto items = Buffers.VertexFans.Items.Buffer.GetSpan<uvec2>();
    tables.ForEachLive(Domain::Vertex, [&](uint32_t, uint32_t address) {
        const auto root = Buffers.VertexCorners.Get({address, 1u})[0];
        put(root.y);
        put_all(items.subspan(root.y ? root.x : 0u, root.y));
    });
    for (const auto domain : SelectionDomains) {
        const auto selection = SelectionArena(Buffers, domain).Buffer.GetSpan<SelectionBlock>(), hidden = HiddenArena(Buffers, domain).Buffer.GetSpan<SelectionBlock>();
        const auto &table = tables.Sets[uint32_t(domain)];
        for (const auto block : table.Entries) {
            if (block != InvalidOffset) {
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) put(uvec2{selection[block][w] & table.Blocks[block].Live[w], hidden[block][w] & table.Blocks[block].Live[w]});
            }
        }
    }
    // Each layer's entries at the record's live handles, so holes and capacity slack hash nothing, with the render records' layers among them.
    const auto put_layers = [&](bool render) {
        ForEachLayer(Buffers, [&](const auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
            if (info.Render != render) return;
            const auto per = entries(r);
            put(per);
            tables.ForEachLive(info.Elements, [&](uint32_t handle, uint32_t) {
                for (uint32_t e = 0u; e < per; ++e) put(layer.Get(*range(r), handle, e, per));
            });
        });
    };
    put_layers(false);
    std::vector<std::pair<std::string_view, uint32_t>> named;
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) { named.emplace_back(entry.NameView(), attribute); });
    std::ranges::sort(named);
    for (const auto [name, attribute] : named) {
        const auto &entry = NamedAttributes[attribute];
        put_all(name), put(entry.Domain), put(entry.Type), put(entry.Default);
        WithAttributeScalar(entry.Type, [&]<AttributeScalar T>() {
            const auto view = AttributeView<T>(attribute);
            tables.ForEachLive(ElementDomainOf(entry.Domain), [&](uint32_t handle, uint32_t) { put(view[handle]); });
        });
    }
    if (!HasRender(r)) return store::HashBytes(bytes);
    const RenderTables members{render, r};
    put(r.RenderTopologies), put(r.Level0Count), put(r.MeshletRevision), put(r.LodDepth), put(r.SpatialRoot), put(r.Meshlets), put(r.ElementMeshletBlockCounts);
    for (const auto set : RenderSets) put_shape(members[set]);
    ForEachRenderValue(Buffers, r, members, [&](const ArenaInfo &info, std::span<const std::byte> value) {
        const auto first = bytes.size();
        bytes.insert(bytes.end(), value.begin(), value.end());
        for (const auto &field : info.Fields) {
            if (field.Target || field.Payload) continue;
            const uint32_t named = WordAt(value, field.Offset) == OffsetOrInvalid(r.PrimitiveMaterials);
            std::memcpy(bytes.data() + first + field.Offset, &named, sizeof(named));
        }
    });
    put_layers(true);
    for (const auto [masks, set] : {std::pair{&render.PositionDirty, RenderSet::Meshlets}, std::pair{&render.DirtyGroups, RenderSet::Groups}}) {
        const auto &table = members[set];
        for (const auto block : table.Entries) {
            if (block != InvalidOffset) put(masks->Get({block, 1u})[0] & table.Blocks[block].Live[0]);
        }
    }
    return store::HashBytes(bytes);
}

void MeshStore::Release(uint32_t id) {
    if (id < Records.size() && Records[id].Alive) Release(std::span{&id, 1u});
}

void MeshStore::Release(std::span<const uint32_t> requested) {
    std::vector<uint32_t> ids{requested.begin(), requested.end()};
    SortUnique(ids);
    std::erase_if(ids, [&](uint32_t id) { return id >= Records.size() || !Records[id].Alive; });
    if (ids.empty()) return;
    const profile::CpuScope scope{"ReleaseMeshStorage"};
    struct DomainRetirement {
        std::vector<ElementSetRef> Sets;
        std::vector<uint32_t> Blocks;
        std::span<const MeshElementBlock> Membership;
        uint32_t LiveCount{};
    };
    std::array<DomainRetirement, 6> closure;
    std::vector<Range> materials, summaries, normals;
    for (const auto id : ids) {
        const auto &record = Records[id];
        for (const auto domain : ElementDomains) {
            WithDomain(Buffers, domain, [&](const auto &arena) {
                const auto set = DomainSet(record, domain);
                if (!set) return;
                auto &c = closure[uint32_t(domain)];
                c.Membership = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
                c.Sets.push_back(set);
                c.LiveCount += arena.Set(set).Count;
                arena.ForEachBlock(set, [&](uint32_t block, const auto &) { c.Blocks.push_back(block); });
            });
        }
        if (record.PrimitiveMaterials.Count) materials.push_back(record.PrimitiveMaterials);
        if (record.SelectionSummary.Count) summaries.push_back(record.SelectionSummary);
        if (record.PointNormals.Count) normals.push_back(record.PointNormals);
    }
    for (auto &c : closure) {
        if (!std::ranges::is_sorted(c.Blocks)) std::ranges::sort(c.Blocks);
    }
    std::vector<uint32_t> temporary;
    for (const auto id : ids) {
        if (CreatedInStep(id)) temporary.push_back(id);
    }
    // Each record created in this step leaves zeros in its storage, written arena by arena over the records' coalesced ranges.
    const auto zero_ranges = [](auto &arena, std::vector<Range> ranges) {
        CoalesceRanges(ranges);
        for (const auto range : ranges) ZeroRange(arena, range);
    };
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&allocation) {
        if (temporary.empty() || !info.Tracked() || info.RenderRange || &arena.Buffer == &Buffers.VertexCorners.Buffer) return;
        std::vector<Range> ranges;
        for (const auto id : temporary) {
            const auto *allocated = allocation(Records[id]);
            if (!allocated) continue;
            if constexpr (std::same_as<std::remove_cvref_t<decltype(*allocated)>, ElementSetRef>) {
                WithDomain(Buffers, info.Elements, [&](const auto &elements) { ForEachBlockRun(elements, *allocated, [&](Range run) { ranges.push_back(run); }); });
            } else {
                ranges.push_back(*allocated);
            }
        }
        zero_ranges(arena, std::move(ranges));
    });
    const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
    std::vector<Range> items, extras_faces_zero, extras_edges_zero;
    for (const auto id : temporary) {
        const auto &record = Records[id];
        Buffers.Vertices.ForEach(record.Vertices, [&](ElementHandle, ElementAddress address) { AppendRange(items, {roots[address.Value()].x, roots[address.Value()].y}); });
        extras_faces_zero.push_back(record.ExtrasFaces);
        extras_edges_zero.push_back(record.ExtrasEdges);
    }
    zero_ranges(Buffers.VertexFans.Items, std::move(items));
    zero_ranges(Buffers.Render.ExtrasFaces, std::move(extras_faces_zero));
    zero_ranges(Buffers.Render.ExtrasEdges, std::move(extras_edges_zero));
    // Render records release while their element domains still exist, since owner blocks are found through them.
    if (Tracked) ForEachIndexRun(ids, [&](size_t first, size_t count) { Tracked->Entries.Write(ids[first], count); });
    std::vector<Record *> records;
    std::vector<Range> extras_faces, extras_edges;
    for (const auto id : ids) {
        records.push_back(&Records[id]);
        AppendRange(extras_faces, Records[id].ExtrasFaces);
        AppendRange(extras_edges, Records[id].ExtrasEdges);
    }
    ReleaseRender(records);
    Buffers.Render.ExtrasFaces.Release(std::move(extras_faces));
    Buffers.Render.ExtrasEdges.Release(std::move(extras_edges));
    const auto &vertices = closure[uint32_t(Domain::Vertex)];
    Buffers.VertexFans.Release(ElementView<uvec2>{Buffers.VertexCorners.Buffer.GetSpan<uvec2>(), vertices.Membership, vertices.Blocks, {}, vertices.LiveCount});
    ForEachIndexRun(vertices.Blocks, [&](size_t first, size_t count) {
        const Range range{vertices.Blocks[first] * MeshElementBlockSize, uint32_t(count) * MeshElementBlockSize};
        std::ranges::fill(Buffers.VertexCorners.GetMutable(range), uvec2{});
    });
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto masks = SelectionArena(Buffers, SelectionDomains[d]).Buffer.template GetSpan<SelectionBlock>();
        auto selected = closure[uint32_t(SelectionDomains[d])].Blocks;
        std::erase_if(selected, [&](uint32_t block) {
            return block >= masks.size() || std::ranges::none_of(masks[block], [](uint32_t word) { return word != 0u; });
        });
        EditSelectionBlocks(SelectionElements[d], selected, [](uint32_t, auto &words) { words = {}; });
        const auto hidden = HiddenArena(Buffers, SelectionDomains[d]).Buffer.GetSpan<SelectionBlock>();
        auto hidden_blocks = closure[uint32_t(SelectionDomains[d])].Blocks;
        std::erase_if(hidden_blocks, [&](uint32_t block) { return block >= hidden.size() || std::ranges::none_of(hidden[block], [](uint32_t word) { return word != 0u; }); });
        EditHiddenBlocks(SelectionElements[d], hidden_blocks, [](uint32_t, auto &words) { words = {}; });
    }
    ReleaseMeshes(Buffers.SelectionTree, ids);
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&, auto &&range) {
        if (info.Render) return;
        std::vector<Range> ranges;
        for (const auto id : ids) AppendRange(ranges, *range(Records[id]));
        layer.Values.Release(std::move(ranges));
    });
    std::vector<uint32_t> attributes;
    for (const auto id : ids) {
        if (id < RecordAttributes.size()) attributes.append_range(RecordAttributes[id]);
    }
    std::ranges::sort(attributes);
    for (const auto attribute : attributes) {
        if (const auto &entry = NamedAttributes[attribute]; CreatedInStep(entry.Record)) WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) { ZeroRange(values, entry.Values); });
        ReleaseAttribute(attribute);
    }
    for (const auto domain : ElementDomains) {
        WithDomain(Buffers, domain, [&](auto &arena) {
            const auto &c = closure[uint32_t(domain)];
            arena.Destroy(c.Sets, c.Blocks);
        });
    }
    Buffers.PrimitiveMaterials.Release(std::move(materials));
    Buffers.SelectionSummary.Release(std::move(summaries));
    Buffers.PointNormals.Release(std::move(normals));
    if (Tracked) {
        Tracked->Free.Write(FreeIds.size(), ids.size());
        Tracked->Dirty.append_range(ids);
    }
    ReleaseBlockLists(ids);
    for (const auto id : ids) {
        Records[id] = {};
        DerivedRecords[id] = {};
    }
    Released.append_range(ids);
    FreeIds.append_range(ids);
}

uint32_t MeshStore::AppendPayload(Record &record, RenderPayload payload, uint32_t count) {
    return WithPayload(Buffers.Render, WriteRecord(record.StoreId), payload, [&](auto &pool, PayloadRange &range) {
        const auto used = range.Values.Count;
        if (uint64_t(used) + count >= InvalidOffset) throw std::length_error("Payload range exceeds its pool's address space.");
        if (used + count > range.Capacity) {
            Range capacity{range.Values.Offset, range.Capacity};
            if (range.Capacity && pool.Extend(capacity, used + count - range.Capacity)) {
                range.Capacity = capacity.Count;
            } else {
                // Coalescing places the replacement at or before the released range, making forward copies safe.
                if (range.Capacity) pool.Release(capacity);
                const auto grown = pool.Allocate(std::max(2u * range.Capacity, used + count));
                std::ranges::copy(pool.Get(range.Values), pool.GetMutable({grown.Offset, used}).begin());
                range = {{grown.Offset, used}, grown.Count};
                Buffers.Render.MeshRecords.GetMutable({record.StoreId, 1u})[0].Display.Payloads = PayloadBasesOf(record);
            }
        }
        range.Values.Count += count;
        return used;
    });
}

void MeshStore::ReleasePayloadRuns(Record &record, RenderPayload payload, std::vector<Range> runs) {
    std::ranges::sort(runs, std::ranges::greater{}, &Range::Offset);
    WithPayload(Buffers.Render, WriteRecord(record.StoreId), payload, [&](auto &, PayloadRange &range) {
        for (const auto run : runs) {
            if (run.Offset + run.Count == range.Values.Count) range.Values.Count = run.Offset;
        }
    });
}

void MeshStore::ReleaseMeshletStorage(Record &owner, std::span<const uint32_t> handles) {
    auto &render = Buffers.Render;
    std::array<std::vector<Range>, RenderPayloads.size()> runs;
    const auto entries = render.Meshlets.Entries(owner.MeshletData);
    const auto words = render.Meshlets.Buffer.GetSpan<uint32_t>();
    for (const auto handle : handles) {
        const auto address = Resolve<RenderBlockSize>(entries, ElementHandle{handle});
        const auto value = words.subspan(uint64_t(address.Value()) * (sizeof(MeshletRecord) / sizeof(uint32_t)));
        for (const auto &field : MeshletFields) {
            if (field.Payload && value[field.Offset / 4u] != InvalidOffset) runs[uint32_t(*field.Payload)].push_back({value[field.Offset / 4u], value[field.RunCount / 4u]});
        }
        MaskClear(render.PositionDirty, address);
    }
    for (const auto payload : RenderPayloads) ReleasePayloadRuns(owner, payload, std::move(runs[uint32_t(payload)]));
    render.Meshlets.Erase(owner.MeshletData, handles);
}

void MeshStore::ReleaseRender(std::span<Record *const> records) {
    const profile::CpuScope scope{"ReleaseRenderStorage"};
    auto &render = Buffers.Render;
    std::array<std::vector<ElementSetRef>, RenderSets.size()> sets;
    std::array<std::vector<uint32_t>, RenderSets.size()> blocks;
    std::vector<Range> routes;
    std::array<std::vector<Range>, RenderPayloads.size()> payloads;
    std::array<std::vector<Range>, 3> owners;
    for (auto *record : records) {
        auto &b = *record;
        if (!b.MeshletData && b.Meshlets.Count) {
            std::vector<uint32_t> handles(b.Meshlets.Count);
            std::ranges::iota(handles, b.Meshlets.Offset);
            ReleaseMeshletStorage(WriteRecord(b.StoreId), handles);
        }
        for (const auto payload : RenderPayloads) WithPayload(render, b, payload, [&](const auto &, const PayloadRange &range) { AppendRange(payloads[uint32_t(payload)], {range.Values.Offset, range.Capacity}); });
        for (const auto set : RenderSets) {
            WithRenderSet(render, set, [&](const auto &arena) {
                const auto ref = RenderSetOf(b, set);
                if (!ref) return;
                sets[uint32_t(set)].push_back(ref);
                arena.ForEachBlock(ref, [&](uint32_t block, const auto &) { blocks[uint32_t(set)].push_back(block); });
            });
        }
        for (uint32_t topology = 0u; topology < 3u; ++topology) AppendRange(owners[topology], b.ElementMeshlets[topology]);
        AppendRange(routes, b.PrimitiveRoutes);
    }
    for (auto &list : blocks) std::ranges::sort(list);
    for (const auto set : RenderSets) ClearRenderMasks(Buffers, set, blocks[uint32_t(set)]);
    for (const auto set : RenderSets) WithRenderSet(render, set, [&](auto &arena) { arena.Destroy(sets[uint32_t(set)], blocks[uint32_t(set)]); });
    for (auto *record : records) ResetRender(*record);
    for (uint32_t i = 0u; i < owners.size(); ++i) render.ElementMeshlets[i].Values.Release(std::move(owners[i]));
    render.PrimitiveRoutes.Release(std::move(routes));
    render.MeshletTriangleIds.Release(std::move(payloads[0]));
    render.MeshletVertexCorners.Release(std::move(payloads[1]));
    render.MeshletLocalTriangles.Release(std::move(payloads[2]));
    render.GroupClusterIds.Release(std::move(payloads[3]));
}

void MeshStore::ReservePrimitiveRoutes(Record &record, uint32_t count) {
    if (uint64_t(count) * 3u > UINT32_MAX) throw std::length_error("Primitive routes exceed arena address space.");
    count *= 3u;
    if (count <= record.PrimitiveRoutes.Count) return;
    std::vector<uint32_t> routes(count, InvalidOffset);
    std::ranges::copy(Buffers.Render.PrimitiveRoutes.Get(record.PrimitiveRoutes), routes.begin());
    Buffers.Render.PrimitiveRoutes.Update(record.PrimitiveRoutes, routes);
}

void MeshStore::SetExtrasIndices(uint32_t id, std::span<const uint32_t> faces, std::span<const uint32_t> edges) {
    auto &record = WriteRecord(id);
    record.ExtrasFaces = Buffers.Render.ExtrasFaces.Allocate(faces);
    record.ExtrasEdges = Buffers.Render.ExtrasEdges.Allocate(edges);
}

void Reset(MeshStore &m, state::Registry &) {
    for (auto &kept : m.Kept) m.Release(kept.Pinned);
    m.Kept.clear();
    if (m.Tracked) {
        m.Tracked->Entries.Write(0, m.Records.size());
        m.Tracked->Free.Write(0, m.FreeIds.size());
        m.Tracked->Attributes.Write(0, m.NamedAttributes.size());
        m.Tracked->AllDirty = true;
    }
    m.Buffers.VertexFans.Reset();
    m.Buffers.SelectionTree.Reset();
    ForEachArena(m.Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.Reset(); });
    m.Records.clear();
    m.DerivedRecords.clear();
    m.NamedAttributes.clear();
    m.RecordAttributes.clear();
    m.FreeAttributes.clear();
    m.IndexedAttributes.clear();
    {
        const std::scoped_lock lock{m.BlockListLock};
        m.BlockLists.Reset();
        m.BlockListEntries.clear();
        m.RetiredBlockLists.clear();
    }
    m.FreeIds.clear();
    m.Released.clear();
    m.Unshaded.clear();
}

uint32_t MeshStore::AcquireId(Record &&record) {
    if (!FreeIds.empty()) {
        const auto reused = FreeIds.back();
        if (Tracked) Tracked->Free.Write(FreeIds.size() - 1, 1);
        FreeIds.pop_back();
        record.StoreId = reused;
        WriteRecord(reused) = std::move(record);
        DerivedRecords[reused] = {.CreatedStep = Step};
        ReleaseMeshes(Buffers.SelectionTree, std::span{&reused, 1u});
        return reused;
    }
    if (Tracked) {
        Tracked->Entries.Write(Records.size(), 1);
        Tracked->Dirty.push_back(uint32_t(Records.size()));
    }
    Records.emplace_back(std::move(record));
    DerivedRecords.push_back({.CreatedStep = Step});
    const auto id = uint32_t(Records.size() - 1);
    Records[id].StoreId = id;
    Buffers.Render.MeshRecords.Mirror({0, uint32_t(Records.size())});
    ReleaseMeshes(Buffers.SelectionTree, std::span{&id, 1u});
    return id;
}
