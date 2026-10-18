#include "MeshStore.h"
#include "gpu/PolygonTriangulation.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshPipelines.h"
#include "mesh/PageFootprint.h"
#include "metal/Dispatch.h"

#include "Profile.h"
#include "SortUnique.h"
#include "project/store/Pages.h"

#include <map>

namespace {
constexpr uint32_t UniformFaceMode{uint32_t(CornerClassMode::UniformFace)};

constexpr auto NoAllocation = [](auto &) -> Range * { return nullptr; };

// The change bits a tracked arena reports and its history name, with a derived arena left unnamed.
// A mirror arena shares its ranges with a master arena earlier in the roster and allocates nothing of its own.
// Every word of a referencing arena's values is a handle of its References domain relative to the value's record, or InvalidOffset.
// A span value's two words are the first handle and the end of a run of consecutive handles.
using Domain = MeshStore::ElementDomain;

// A render field of an arena's values: the word at byte Offset is a handle of the record's Target set, or InvalidOffset.
// A run field references consecutive handles with a count stored at byte RunCount.
// A zero count denotes an empty run.
// A payload run field names an offset in the record's range of its Payload pool instead.
// A field with neither contains the offset of the record's primitive material table.
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

struct ArenaInfo {
    uint32_t Bits{};
    const char *Name{};
    bool Mirror{false};
    Domain Elements{Domain::None};
    bool BlockIndexed{false};
    Domain Layer{Domain::None}; // The domain whose handles index an element layer's record ranges
    std::optional<AttributeType> Named{}; // Named-attribute type; the attribute table stores its ranges
    Domain References{Domain::None};
    bool ReferenceSpan{false};
    std::optional<RenderSet> Render{}; // The render collection keying the arena's values, or its blocks when BlockIndexed
    std::span<const RenderField> Fields{}; // The render fields of each value
    bool RenderRange{false}; // A record range of render state, which only a clone of the render records copies
    bool Tracked() const { return Name != nullptr; }
};

struct LayerInfo {
    Domain Elements;
    uint32_t Bits;
    const char *Name;
    bool Render{false};
};
// Visit layers as f(layer, info, entries, range).
// entries(record) is zero for absent layers; range(record) borrows storage.
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
    // Sector roots and their normals are present while the corner classification needs them.
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
    // Named attribute values map their changes to records through the attribute table's ranges.
    constexpr std::array ValueNames{"AttributeFloats", "AttributeVec2s", "AttributeVec3s", "AttributeVec4s", "AttributeInts", "AttributeBools"};
    for (uint32_t i = 0u; i < AttributeTypes.size(); ++i) {
        WithAttributeValues(b.Named, AttributeTypes[i], [&](auto &values) {
            f(values, ArenaInfo{.Bits = AttributesChanged, .Name = ValueNames[i], .Named = AttributeTypes[i]}, NoAllocation);
        });
    }
    // Render arenas, whose collections the records' render sets key.
    // A mirror of a render collection contains bytes only, sized by its collection's blocks.
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

// Captures the `count` values of `stride` bytes at `to` and queues their copy from `from`.
void CaptureCopy(CloneCopies &copies, mtl::Buffer &buffer, uint64_t stride, uint64_t from, uint64_t to, uint64_t count = 1u) {
    buffer.CaptureWrite(to * stride, count * stride);
    copies.Copy(buffer, from * stride, to * stride, count * stride);
}
// Calls fn(from, to, count) for each maximal run of pairs consecutive on both sides.
void ForEachPairRun(std::span<const uvec2> pairs, auto &&fn) {
    for (size_t first = 0u; first < pairs.size();) {
        auto end = first + 1u;
        while (end < pairs.size() && pairs[end].x == pairs[end - 1u].x + 1u && pairs[end].y == pairs[end - 1u].y + 1u) ++end;
        fn(pairs[first].x, pairs[first].y, uint32_t(end - first));
        first = end;
    }
}

constexpr std::array SelectionDomains{Domain::Vertex, Domain::Edge, Domain::Face};
constexpr std::array SelectionElements{Element::Vertex, Element::Edge, Element::Face};
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

// A fresh vertex has no corner fan and no outgoing halfedge.
void ClearVertexRoots(MeshArenas &a, ElementSetRef vertices) {
    a.Vertices.ForEachBlock(vertices, [&](uint32_t block, const auto &) {
        const Range range{block * MeshElementBlockSize, MeshElementBlockSize};
        a.VertexCorners.CaptureWrite(range);
        std::ranges::fill(a.VertexCorners.GetMutable(range), uvec2{InvalidOffset, 0u});
        a.OutgoingHalfedges.CaptureWrite(range);
        std::ranges::fill(a.OutgoingHalfedges.GetMutable(range), InvalidOffset);
    });
}

// Clears the render sets and fields, keeping the record's canonical fields, extras index data and id.
void ResetRender(MeshStore::Record &r) {
    for (const auto set : RenderSets) RenderSetOf(r, set) = {};
    r.Meshlets = r.PrimitiveRoutes = {};
    r.MeshletTriangles = r.MeshletVertices = r.MeshletLocalTriangles = r.GroupClusters = {};
    r.SpatialRoot = InvalidOffset;
    r.MeshletRevision = 0u;
    r.Level0Count = r.RenderTopologies = r.LodDepth = 0u;
    r.ElementMeshlets = {};
    r.ElementMeshletBlockCounts = {};
}

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
    // Clear dirty bits when reusing a member slot.
    if (which == RenderSet::Meshlets || which == RenderSet::Groups) {
        for (uint32_t i = 0u; i < count; ++i) MaskClear(which == RenderSet::Meshlets ? PositionDirty : DirtyGroups, run.Address + i);
    }
    return run;
}

uint32_t RenderArenas::ListMembers(ElementSetRef &set, std::span<const uint32_t> clusters) {
    if (clusters.empty()) return InvalidOffset;
    const auto count = uint32_t(clusters.size());
    const auto run = Insert(RenderSet::Members, set, count + count / 2u);
    const auto members = LodMembers.GetMutable({run.Address.Value(), run.Count});
    std::ranges::fill(std::ranges::copy(clusters, members.begin()).out, members.end(), InvalidOffset);
    return run.Handle.Value;
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
    // The entries a record added to the index, so a changed record replaces only its own.
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
    // Each element layer's byte extents written since the last TakeWrites, which Ranges maps to their records.
    std::unordered_map<mtl::Buffer *, std::vector<std::pair<uint64_t, uint64_t>>> LayerWrites;
    // Records whose ranges or sets changed since the last index, and whether every record did.
    std::vector<uint32_t> Dirty;
    bool AllDirty{true};

    HistoryState(MeshStore &mesh, store::History &history) : Entries(mesh.Records), Free(mesh.FreeIds), Attributes(mesh.NamedAttributes) {
        Entries.Trie.CollectChanged = true;
        history.Track(Entries, "mesh.entries", 0);
        history.Track(Free, "mesh.free", 0);
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
        // A render range restores without a record change, as the render collections do.
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
      BlockLists{ctx, SlotType::Buffer} {
    // Element value arenas, their mirrors and their membership log their writes, and selection state derives from the blocks they name.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (info.Bits == SelectionChanged || !info.Tracked() || info.RenderRange || info.Elements == Domain::None) return;
        const auto stride = uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
        WriteLogs.push_back({&arena.Buffer, info.Elements, info.BlockIndexed ? WriteIndex::Blocks : WriteIndex::Elements, stride});
        if constexpr (requires { arena.Blocks; }) WriteLogs.push_back({&arena.Blocks.Buffer, info.Elements, WriteIndex::Blocks, uint32_t(sizeof(MeshElementBlock))});
    });
    for (auto &log : WriteLogs) log.Buffer->Writes = &log.Extents;
}
MeshStore::~MeshStore() = default;

MeshStore::Writes MeshStore::TakeWrites() {
    Writes writes;
    for (auto &log : WriteLogs) {
        auto &blocks = writes.Blocks[uint32_t(log.Domain)];
        for (const auto [begin, end] : log.Extents) {
            const auto first = begin / log.Stride, last = (end + log.Stride - 1u) / log.Stride;
            if (log.Index == WriteIndex::Elements) {
                for (auto block = first / MeshElementBlockSize; block <= (last - 1u) / MeshElementBlockSize; ++block) blocks.push_back(uint32_t(block));
            } else {
                for (auto block = first; block < last; ++block) blocks.push_back(uint32_t(block));
            }
        }
        log.Extents.clear();
    }
    // A tracked store logs its layers' writes, and each written block names a local block of the record whose range contains it.
    // Use the history index for unchanged records and current ranges for changed records.
    if (Tracked) {
        auto &dirty = Tracked->Dirty;
        SortUnique(dirty);
        const bool indexed = !Tracked->AllDirty;
        ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
            if (info.Render) return;
            auto &log = Tracked->LayerWrites.at(&layer.Values.Buffer);
            if (log.empty()) return;
            constexpr auto Stride = sizeof(typename std::remove_cvref_t<decltype(layer)>::Block);
            // Disjoint unindexed ranges in offset order.
            std::vector<std::pair<Range, uint32_t>> held;
            const auto hold = [&](uint32_t id) {
                if (id < Records.size() && Records[id].Alive && range(Records[id])->Count) held.emplace_back(*range(Records[id]), id);
            };
            if (indexed) {
                for (const auto id : dirty) hold(id);
            } else {
                for (uint32_t id = 0u; id < Records.size(); ++id) hold(id);
            }
            std::ranges::sort(held, {}, [](const auto &entry) { return entry.first.Offset; });
            auto &blocks = writes.Blocks[uint32_t(info.Elements)];
            const auto &ranges = Tracked->Ranges.at(&layer.Values.Buffer);
            for (const auto [begin, end] : log) {
                const uint64_t first = begin / Stride, last = (end + Stride - 1u) / Stride;
                const auto take = [&](uint32_t id, uint64_t values_first, uint64_t values_end) {
                    const auto &record = Records[id];
                    const auto table = WithDomain(Buffers, info.Elements, [&](const auto &arena) { return arena.Entries(DomainSet(record, info.Elements)); });
                    const auto per = std::max(entries(record), 1u);
                    for (auto block = std::max(first, values_first); block < std::min(last, values_end); ++block) {
                        if (const auto local = (block - values_first) / per; local < table.size() && table[local] != InvalidOffset) blocks.push_back(table[local]);
                    }
                };
                if (indexed) {
                    auto it = ranges.upper_bound(begin);
                    if (it != ranges.begin() && std::prev(it)->second.End > begin) --it;
                    for (; it != ranges.end() && it->first < end; ++it) {
                        if (!std::ranges::binary_search(dirty, it->second.Id)) take(it->second.Id, it->first / Stride, it->second.End / Stride);
                    }
                }
                for (auto it = std::ranges::lower_bound(held, first, {}, [](const auto &entry) { return uint64_t(entry.first.Offset) + entry.first.Count - 1u; }); it != held.end() && it->first.Offset < last; ++it) {
                    take(it->second, it->first.Offset, uint64_t(it->first.Offset) + it->first.Count);
                }
            }
            log.clear();
        });
    }
    for (auto &blocks : writes.Blocks) SortUnique(blocks);
    writes.Attributes = std::exchange(WrittenAttributes, {});
    SortUnique(writes.Attributes);
    return writes;
}

void MeshStore::Track(store::History &history) {
    Buffers.VertexFans.Track(history, "mesh.vertexFans");
    Buffers.VertexFans.Items.Buffer.History()->Trie.CollectChanged = false;
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        // A mirror owns no allocator, so only its bytes carry history.
        if (info.Mirror) arena.Buffer.Track(history, std::string{"mesh."} + info.Name);
        else arena.Track(history, std::string{"mesh."} + info.Name);
        // Changes map to records by byte extent rather than by page.
        arena.Buffer.History()->Trie.CollectChanged = false;
        if constexpr (requires { arena.Blocks; }) {
            arena.Blocks.Buffer.History()->Trie.CollectChanged = false;
            arena.Sets.Buffer.History()->Trie.CollectChanged = false;
            arena.Table.Buffer.History()->Trie.CollectChanged = false;
        }
    });
    Tracked = std::make_unique<HistoryState>(*this, history);
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&, auto &&) {
        if (!info.Render) layer.Values.Buffer.Writes = &Tracked->LayerWrites[&layer.Values.Buffer];
    });
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        if (info.Named) Tracked->AttributeStreams.push_back({&arena.Buffer, uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>)), *info.Named});
        else if (info.Elements == Domain::None) Tracked->Ranges.try_emplace(&arena.Buffer);
        else Tracked->Streams.push_back({&arena.Buffer, info.Elements, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), info.Bits, info.BlockIndexed ? HistoryState::StreamKind::Blocks : HistoryState::StreamKind::Values});
        if constexpr (requires { arena.Blocks; }) {
            // A render collection's membership restores without a record change, and its blocks reindex as they restore.
            if (info.Render) return;
            Tracked->Streams.push_back({&arena.Blocks.Buffer, info.Elements, sizeof(MeshElementBlock), TopologyChanged, HistoryState::StreamKind::Blocks});
            Tracked->Streams.push_back({&arena.Sets.Buffer, info.Elements, sizeof(MeshElementSet), TopologyChanged, HistoryState::StreamKind::Sets});
        }
    });
}

namespace {
void CaptureRange(const auto &arena, Range range) {
    using Value = ArenaValue<std::remove_cvref_t<decltype(arena)>>;
    arena.Buffer.CaptureWrite(uint64_t(range.Offset) * sizeof(Value), uint64_t(range.Count) * sizeof(Value));
}
// Visits each run of consecutive blocks a set owns, in its block order, as a handle range.
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

// Visits selected addresses in ascending order.
void CaptureSelected(const mtl::Buffer &buffer, uint32_t stride, SelectionView selection) {
    const auto *history = buffer.History();
    if (!history && !buffer.Writes) return;
    const uint64_t page_size = history ? history->PageBytes : mtl::HistoryPageBytes;
    std::vector<uint32_t> pages;
    selection.ForEachAddress([&](ElementAddress address) {
        const uint64_t offset = uint64_t(address.Value()) * stride;
        buffer.LogWrite(offset, offset + stride);
        const auto first = offset / page_size, end = (offset + stride + page_size - 1) / page_size;
        for (auto page = first; page < end; ++page) {
            if (pages.empty() || pages.back() < page) pages.push_back(uint32_t(page));
        }
    });
    buffer.CaptureWritePages(pages);
}
} // namespace

void MeshStore::CaptureVertexEdit(uint32_t id) {
    if (!Tracked) return;
    CaptureSelected(Buffers.Vertices.Buffer, sizeof(Vertex), GetSelectedElements(id, Element::Vertex));
}

void MeshStore::CaptureSelectionSummary(uint32_t id) {
    if (Tracked) CaptureRange(Buffers.SelectionSummary, Records.at(id).SelectionSummary);
}

void MeshStore::CaptureSelectionBlocks(Element element, std::span<const uint32_t> blocks) {
    if (Tracked) SelectionArena(Buffers, SelectionDomain(element)).Buffer.CaptureWriteElements(blocks, sizeof(SelectionBlock));
}

void MeshStore::CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation operation, const GeometrySelection &selection) {
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
        case EditSharpnessOperation::SetVertexEdges: {
            const auto edges = GetVertexEdgeIncidence(id);
            std::vector<uint32_t> touched;
            for (const auto vertex : selection.Vertices) {
                for (const auto edge : edges.Incident(vertex)) touched.push_back(edge);
            }
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(addresses(Domain::Edge, touched), sizeof(uint8_t));
            break;
        }
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
    // Fan replacements always change canonical roots, which map changes to the owning mesh.
    // The item track still needs its extent queue drained.
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
    // Canonical streams resolve only dirty blocks through the current ownership metadata.
    // No per-mesh range index is built for these streams.
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
    // A named attribute's value block maps to the record of the attribute whose range contains it.
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
        if (!range.Vertices.Count) continue;
        auto &vertices = change.VertexRanges;
        if (!vertices.empty() && range.Vertices.Offset <= vertices.back().Offset + vertices.back().Count) {
            vertices.back().Count = std::max(vertices.back().Offset + vertices.back().Count, range.Vertices.Offset + range.Vertices.Count) - vertices.back().Offset;
        } else {
            vertices.push_back(range.Vertices);
        }
    }
    for (auto &change : changes) {
        for (auto &blocks : change.Blocks) SortUnique(blocks);
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
    // Each layer and named attribute of the domain covers the grown table, and the local blocks of blocks new to the set take its absent value.
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
    // Collect listed blocks outside the set, including every listed block when the set is retired.
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
    if (domain == Domain::Triangle) Records[id].TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
    // Posed sector normals key their blocks by corner address, so corner blocks leaving a record with sectors renew its pose layout.
    if (domain == Domain::Halfedge && !left.empty() && Records[id].CornerSectors.Count) DerivedRecords.at(id).NormalRevision = ++NextNormalRevision;
    if (domain == Domain::Vertex) {
        // Dead slots, including every slot of a retired block, free their fans and clear their roots.
        std::vector<uvec2> released;
        for (const auto b : blocks) {
            const auto &block = Buffers.Vertices.Blocks.Get({b, 1u})[0];
            const auto live = block.Owner == set.Index ? block.Live : decltype(block.Live){};
            const auto roots = Buffers.VertexCorners.GetMutable({b * 256u, 256u});
            for (uint32_t i = 0u; i < 256u; ++i) {
                if (live[i / 32u] & (1u << (i % 32u))) continue;
                if (roots[i].y) released.push_back(roots[i]);
                roots[i] = {InvalidOffset, 0u};
            }
        }
        Buffers.VertexFans.Release(released);
    }
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

uint32_t MeshStore::AcquireAttribute(NamedAttribute &&entry) {
    const auto attribute = uint32_t(std::ranges::find(NamedAttributes, InvalidOffset, &NamedAttribute::Record) - NamedAttributes.begin());
    if (attribute < NamedAttributes.size()) {
        WriteAttribute(attribute) = std::move(entry);
    } else {
        if (Tracked) Tracked->Attributes.Write(attribute, 1);
        NamedAttributes.emplace_back(std::move(entry));
    }
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
    for (uint32_t attribute = 0u; attribute < NamedAttributes.size(); ++attribute) {
        if (NamedAttributes[attribute].Record == id && NamedAttributes[attribute].NameView() == name) return attribute;
    }
    return InvalidOffset;
}

void MeshStore::ReleaseAttribute(uint32_t attribute) {
    const auto &entry = NamedAttributes.at(attribute);
    if (entry.Record == InvalidOffset) throw std::invalid_argument("Released attribute.");
    if (entry.Values.Count) WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) { values.Release(entry.Values); });
    WriteAttribute(attribute) = {};
    WrittenAttributes.push_back(attribute);
}

std::expected<void, MeshStore::AttributeError> MeshStore::RenameAttribute(uint32_t attribute, std::string_view name) {
    if (name.empty() || name.size() >= AttributeName{}.size()) return std::unexpected{AttributeError::InvalidName};
    if (FindAttribute(NamedAttributes.at(attribute).Record, name) != InvalidOffset) return std::unexpected{AttributeError::Duplicate};
    auto &entry = WriteAttribute(attribute);
    entry.Name = {};
    std::ranges::copy(name, entry.Name.begin());
    WrittenAttributes.push_back(attribute);
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
    // Dead slots contain no selection, so the new state's masks are clear and only the aggregates need deriving.
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

BoundaryEdgeView MeshStore::GetBoundaryEdges(uint32_t id) const {
    const auto &record = Records.at(id);
    if (!record.FaceData || !record.EdgeData) return {};
    return {
        Buffers.SelectionTree.Read(3u * id + 1u, Buffers.EdgeAggregates.Buffer.GetSpan<SelectionAggregate>()),
        Buffers.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(),
        Buffers.EdgeHalfedges.Entries(record.EdgeData),
        Buffers.FaceCorners.Entries(record.FaceCorners),
        Buffers.EdgeHalfedges.Buffer.GetSpan<uint32_t>(),
        Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>(),
    };
}

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

vec3 ComposeCornerNormal(const MeshConnectivity &connectivity, CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &sources) {
    const auto face = connectivity.FaceAddress(face_ids[ci / 3u]).Value();
    if (mode == UniformFaceMode || (mode == uint32_t(CornerClassMode::Mixed) && sharpness[face])) {
        const auto posed = sources.PosedFaceNormals.Find(face);
        return posed == InvalidOffset ? sources.FaceNormals[face] : sources.PosedFaceNormals.Values[posed];
    }
    const auto root = mode == uint32_t(CornerClassMode::Mixed) ? sectors.GetOr(ci, InvalidOffset) : InvalidOffset;
    if (root == InvalidOffset) {
        const auto v = connectivity.VertexAddress(vertices[ci]).Value();
        const auto posed = sources.PosedVertexNormals.Find(v);
        return posed == InvalidOffset ? sources.VertexNormals[v] : sources.PosedVertexNormals.Values[posed];
    }
    const auto posed = sources.PosedSectorNormals.Find(connectivity.HalfedgeAddress(root).Value());
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

SharpnessSummary MeshStore::GetFaceSharpnessSummary(uint32_t id) const { return LiveSharpness(GetSelectionRoot(id, Element::Face).Flags); }
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

void MeshStore::FinishConnectivity(uint32_t id, uint32_t edge_count) {
    auto &record = WriteRecord(id);
    auto &edges = Buffers.EdgeHalfedges;
    ElementHandleRange allocated{.First = 0u, .Count = edges.Count(record.EdgeData)};
    edges.Shrink(record.EdgeData, allocated, {}, edge_count);
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

namespace {
// A dense set resolves ordinals by offset, and a sparse set through its cached block list.
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
        // The block list skips the table's empty entries before the block.
        const auto at = local - uint32_t(std::count(entries.begin(), entries.begin() + local, InvalidOffset));
        const auto list = GetBlockList(id, domain);
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
ElementView<Vertex> MeshStore::VertexView(uint32_t id) const { return LiveView(*this, Buffers.Vertices, Records.at(id).Vertices, id, ElementDomain::Vertex); }

uint32_t MeshStore::CreateMeshSource(const MeshData &data) {
    const auto vertices = Buffers.Vertices.Allocate(data.Positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), data.Positions);
    const auto id = AcquireId({.Vertices = vertices, .Alive = true});
    SyncMirrors();
    ClearVertexRoots(Buffers, vertices);
    // Both face corners and loose-edge endpoints must exist before GPU welding remaps them.
    AllocateConnectivity(id, data.HalfedgeCount(), data.FaceCount(), !data.FaceOffsets.empty(), data.FaceOffsets, data.Edges);
    if (data.FaceCount() > 0) std::ranges::copy(data.FaceCorners, Buffers.FaceCorners.GetMutable(Get(id).FaceCorners).begin());
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
    Buffers.Vertices.Shrink(record.Vertices, allocated, {}, welded_vertices);
}

uint32_t MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertices = Buffers.Vertices.Allocate(positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), positions);
    // Imported normals initialize the canonical vertex normals of face-less meshes.
    const auto point_normals = attrs.Normals ? Buffers.PointNormals.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    const auto id = AcquireId({.Vertices = vertices, .PointNormals = point_normals, .HasAuthoredNormals = attrs.Normals.has_value(), .Alive = true});
    SyncMirrors();
    ClearVertexRoots(Buffers, vertices);
    FillBaseVertexNormalMirror(vertices, point_normals);
    return id;
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count, const MeshVertexAttributes &attrs) {
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
            if (*uvs[set]) Buffers.CornerUvs[set].Values.PlanAdditional(corner_blocks);
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
    // Imported normals initialize the canonical vertex normals of face-less meshes.
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
        SyncMirrors();
        const auto first_tri_span = Buffers.FaceTriangles.GetMutable(record.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.FaceSize(fi) - 2u;
        }
        record.TriangleCount = tri_offset;

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

        record.TriangleData = Buffers.Triangles.Allocate(tri_offset);
        auto triangle_span = Buffers.Triangles.GetMutable(record.TriangleData);
        // A new record's sets are dense, so its handles index its arena runs.
        const auto corner_vertices = Buffers.FaceCorners.Get(record.FaceCorners);
        const auto vertices = Buffers.Vertices.Get(record.Vertices);
        std::vector<vec2> projected;
        std::vector<uint32_t> next, previous;
        uint32_t ti = 0u;
        for (uint32_t fi = 0u; fi < face_count; ++fi) {
            const uint32_t n = data.FaceSize(fi), first = data.FaceStart(fi);
            if (n > 3u) {
                projected.resize(n);
                next.resize(n);
                previous.resize(n);
            }
            TriangulatePolygon(n, [&](uint32_t i) { return vertices[corner_vertices[first + i]].Position; }, projected.data(), next.data(), previous.data(), [&](uvec3 triangle, uint32_t t) { triangle_span[ti + t] = {first + triangle.x, first + triangle.y, first + triangle.z}; });
            ti += n - 2u;
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.ElementPrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.ElementPrimitiveIndices) + 1u) :
            1u;
        write_primitive_tables(primitive_count);

    } else if (!primitives.ElementPrimitiveIndices.empty()) {
        // Point and line meshes carry one color and one primitive index per vertex.
        // Primitive indices are source-wide, so the material table spans every primitive of the source mesh.
        const auto primitive_count = primitives.MaterialIndices.empty() ? 1u : uint32_t(primitives.MaterialIndices.size());
        write_primitive_tables(primitive_count);
    }

    if (face_count == 0 && attrs.Colors0) {
        record.VertexAttributes |= MeshAttributeBit_Color0;
        Buffers.VertexColors.Initialize(record.VertexColors, Buffers.Vertices.Count(record.Vertices), *attrs.Colors0);
    }

    const Mesh mesh{*this, id};
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
    Buffers.SelectionSummary.GetMutable(record.SelectionSummary)[0] = {};
    std::vector<NamedAttribute> attributes;
    const auto add = [&](uint32_t, const NamedAttribute &entry) { attributes.push_back({.Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default}); };
    ForEachNamedAttribute(source, add);
    const auto id = AcquireId(std::move(record));
    CaptureSelectionSummary(id);
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
    // Element values and their mirrors, apart from the vertex fan roots, which take new fan runs below.
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
    // Each copied vertex takes a new fan run, and the destination's runs at the copied blocks release.
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
            copies.Copy(Buffers.VertexFans.Items.Buffer, uint64_t(run.Offset) * sizeof(uvec2), uint64_t(next - run.Count) * sizeof(uvec2), uint64_t(run.Count) * sizeof(uvec2));
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

namespace {
// A domain's index among a Change's blocks, or InvalidOffset for a domain it does not list.
constexpr uint32_t ChangeIndex(Domain domain) {
    return domain == Domain::Halfedge ? 3u : SelectableDomain(domain) ? SelectableIndex(domain) :
                                                                        InvalidOffset;
}
// Queues the copy of each run of ascending local blocks from one record range to another, each local block `entries` blocks of `bytes`.
void CopyLocalRuns(CloneCopies &copies, mtl::Buffer &buffer, uint64_t bytes, Range from, Range to, std::span<const uint32_t> locals, uint32_t entries) {
    ForEachIndexRun(locals, [&](size_t first, size_t count) { CaptureCopy(copies, buffer, bytes, from.Offset + locals[first] * entries, to.Offset + locals[first] * entries, uint32_t(count) * entries); });
}
// The ascending local blocks of a domain's restored pairs.
std::vector<uint32_t> RestoredLocals(const MeshArenas &b, Domain domain, const std::array<std::vector<uvec2>, 6> &restored) {
    std::vector<uint32_t> locals;
    WithDomain(b, domain, [&](const auto &arena) {
        for (const auto pair : restored[uint32_t(domain)]) locals.push_back(arena.Blocks.Get({pair.y, 1u})[0].Local);
    });
    std::ranges::sort(locals);
    return locals;
}
} // namespace

std::vector<uint32_t> MeshStore::CloneMeshes(CloneCopies &copies, std::span<const uint32_t> source_ids, bool render) {
    const profile::CpuScope scope{"CloneMeshes"};
    std::vector<uint32_t> ids;
    std::vector<BlockPairs> pairs;
    ids.reserve(source_ids.size());
    pairs.reserve(source_ids.size());
    // Allocate the whole batch before encoding copies: arena growth may move either side.
    for (const auto src_id : source_ids) {
        Record copy{Records.at(src_id)};
        ResetRender(copy);
        const auto id = AcquireId(std::move(copy));
        DerivedRecords[id] = DerivedRecords.at(src_id);
        DerivedRecords[id].NormalRevision = ++NextNormalRevision;
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
        // The baseline's mask blocks move onto the clone's blocks, and its active handle names the same element in the clone.
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

void MeshStore::ReleaseDomain(uint32_t id, ElementDomain domain) {
    WithDomain(Buffers, domain, [&](auto &arena) {
        auto &set = DomainSet(WriteRecord(id), domain);
        std::vector<uint32_t> blocks;
        arena.ForEachBlock(set, [&](uint32_t block, const auto &) { blocks.push_back(block); });
        std::ranges::sort(blocks);
        arena.Destroy(std::span{&set, 1u}, blocks);
        set = {};
    });
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&, auto &&range) {
        if (info.Elements == domain && range(Records[id])->Count) layer.Release(*range(WriteRecord(id)));
    });
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        if (ElementDomainOf(entry.Domain) != domain || !entry.Values.Count) return;
        WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) { values.Release(entry.Values); });
        WriteAttribute(attribute).Values = {};
    });
}

void MeshStore::DetachBlocks(uint32_t id, ElementDomain domain, std::span<const uint32_t> blocks) {
    if (blocks.empty()) return;
    const auto cleared = WithDomain(Buffers, domain, [&](auto &arena) { return arena.Clear(DomainSet(Records[id], domain), blocks); });
    if (SelectableDomain(domain)) EditSelectionBlocks(SelectionElements[SelectableIndex(domain)], cleared, [](uint32_t, auto &words) { words = {}; });
    FinishEraseElements(id, domain, cleared);
}

MeshStore::Change MeshStore::Resync(CloneCopies &copies, uint32_t id, uint32_t source, const DomainBlocks &written, std::span<const uint32_t> written_attributes, std::vector<uint32_t> &attributes) {
    const profile::CpuScope scope{"ResyncMesh"};
    ReleaseRender(std::array{&WriteRecord(id)});
    Change change{.StoreId = id, .Bits = GeometryChanged | TopologyChanged};
    // Restore written blocks and blocks present in only one record; match the source table layout.
    BlockPairs restored;
    for (const auto domain : ElementDomains) {
        const auto d = uint32_t(domain);
        std::vector<uint32_t> removed;
        WithDomain(Buffers, domain, [&](auto &arena) {
            const auto from_set = DomainSet(Records[source], domain);
            if (!from_set) {
                arena.ForEachBlock(DomainSet(Records[id], domain), [&](uint32_t block, const auto &) { removed.push_back(block); });
                if (!removed.empty()) {
                    std::ranges::sort(removed);
                    if (SelectableDomain(domain)) EditSelectionBlocks(SelectionElements[SelectableIndex(domain)], removed, [](uint32_t, auto &words) { words = {}; });
                    ReleaseDomain(id, domain);
                }
                return;
            }
            const std::vector<uint32_t> from{arena.Entries(from_set).begin(), arena.Entries(from_set).end()};
            std::vector<uint32_t> locals;
            if (!DomainSet(Records[id], domain)) {
                DomainSet(WriteRecord(id), domain) = arena.Create();
                locals.resize(from.size());
                std::ranges::iota(locals, 0u);
            } else {
                const auto to = arena.Entries(DomainSet(Records[id], domain));
                const auto end = uint32_t(std::max(from.size(), to.size()));
                for (const auto local : written[d]) {
                    if (local < end) locals.push_back(local);
                }
                for (uint32_t local = 0u; local < end; ++local) {
                    if ((local < from.size() && from[local] != InvalidOffset) != (local < to.size() && to[local] != InvalidOffset)) locals.push_back(local);
                }
                SortUnique(locals);
            }
            std::vector<uint32_t> placed;
            {
                const auto to = arena.Entries(DomainSet(Records[id], domain));
                for (const auto local : locals) {
                    const bool held = local < from.size() && from[local] != InvalidOffset, holds = local < to.size() && to[local] != InvalidOffset;
                    if (held && !holds) placed.push_back(local);
                    else if (!held && holds) removed.push_back(to[local]);
                }
            }
            std::ranges::sort(removed);
            DetachBlocks(id, domain, removed);
            arena.Place(DomainSet(Records[id], domain), placed);
            const auto to = arena.Entries(DomainSet(Records[id], domain));
            auto &live = restored[d];
            for (const auto local : locals) {
                if (local < from.size() && from[local] != InvalidOffset) live.push_back({from[local], to[local]});
            }
            // The copy is dense when the dense source's blocks pair with consecutive blocks.
            const auto flags = arena.Set(from_set).Flags;
            const bool dense = (flags & 1u) && to.size() == from.size() && std::ranges::all_of(std::views::iota(size_t{0}, to.size()), [&](size_t k) { return to[k] == to[0] + k; });
            arena.AssignLive(DomainSet(Records[id], domain), live, dense ? flags : 0u);
        });
        if (const auto index = ChangeIndex(domain); index != InvalidOffset) {
            auto &blocks = change.Blocks[index];
            for (const auto pair : restored[d]) blocks.push_back(pair.y);
            blocks.append_range(removed);
            SortUnique(blocks);
        }
    }
    SyncMirrors();
    {
        const auto &src = Records[source];
        auto &record = WriteRecord(id);
        const auto own = record;
        record = src;
        record.StoreId = id;
        record.Vertices = own.Vertices;
        record.FaceData = own.FaceData;
        record.EdgeData = own.EdgeData;
        record.TriangleData = own.TriangleData;
        record.FaceCorners = own.FaceCorners;
        record.PrimitiveMaterials = own.PrimitiveMaterials;
        record.SelectionSummary = own.SelectionSummary;
        record.PointNormals = own.PointNormals;
        record.ExtrasFaces = own.ExtrasFaces;
        record.ExtrasEdges = own.ExtrasEdges;
        ResetRender(record);
        record.TriangleCount = Buffers.Triangles.Count(record.TriangleData);
        const auto copy_range = [&](auto &arena, Range &to, Range from) {
            if (to.Count != from.Count) {
                if (to.Count) arena.Release(to);
                to = from.Count ? arena.Allocate(from.Count) : Range{};
            }
            if (from.Count) CaptureCopy(copies, arena.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), from.Offset, to.Offset, from.Count);
        };
        copy_range(Buffers.PrimitiveMaterials, record.PrimitiveMaterials, src.PrimitiveMaterials);
        copy_range(Buffers.SelectionSummary, record.SelectionSummary, src.SelectionSummary);
        copy_range(Buffers.PointNormals, record.PointNormals, src.PointNormals);
        ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
            if (info.Render) return;
            auto &to = *range(record);
            to = *range(own);
            const auto from = *range(src);
            const auto per = entries(src);
            if (!per || !from.Count) return layer.Release(to);
            if (entries(own) != per) layer.Release(to);
            const bool whole = !to.Count;
            const auto bound = WithDomain(Buffers, info.Elements, [&](const auto &arena) { return arena.HandleBound(DomainSet(record, info.Elements)) / MeshElementBlockSize; });
            layer.Cover(to, bound, per);
            constexpr auto Bytes = sizeof(typename std::remove_cvref_t<decltype(layer)>::Block);
            if (whole) CaptureCopy(copies, layer.Values.Buffer, Bytes, from.Offset, to.Offset, bound * per);
            else CopyLocalRuns(copies, layer.Values.Buffer, Bytes, from, to, RestoredLocals(Buffers, info.Elements, restored), per);
        });
        DerivedRecords[id] = {};
    }
    // Reuse payloads for attributes still associated with their source copies; allocate fresh copies for the others.
    std::vector<uvec2> kept, whole;
    std::vector<uint32_t> released;
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        const bool intact = std::ranges::binary_search(attributes, attribute) && !std::ranges::binary_search(written_attributes, attribute);
        const auto from = intact ? FindAttribute(source, entry.NameView()) : InvalidOffset;
        if (from != InvalidOffset && NamedAttributes[from].Type == entry.Type && NamedAttributes[from].Domain == entry.Domain) kept.emplace_back(from, attribute);
        else released.push_back(attribute);
    });
    for (const auto attribute : released) ReleaseAttribute(attribute);
    ForEachNamedAttribute(source, [&](uint32_t attribute, const NamedAttribute &) {
        if (!std::ranges::contains(kept, attribute, &uvec2::x)) whole.emplace_back(attribute, 0u);
    });
    // Copy restored local blocks into retained attributes and the full source range into new attributes.
    for (const auto [from, to] : kept) {
        const auto entry = NamedAttributes[from];
        const auto bound = WithDomain(Buffers, ElementDomainOf(entry.Domain), [&](const auto &arena) { return arena.HandleBound(DomainSet(Records[id], ElementDomainOf(entry.Domain))) / MeshElementBlockSize; });
        WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) {
            auto range = NamedAttributes[to].Values;
            if (CoverRange(values, range, bound)) WriteAttribute(to).Values = range;
            CopyLocalRuns(copies, values.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(values)>>), entry.Values, range, RestoredLocals(Buffers, ElementDomainOf(entry.Domain), restored), 1u);
        });
    }
    for (auto &pair : whole) {
        const auto entry = NamedAttributes[pair.x];
        const auto values = WithAttributeValues(Buffers.Named, entry.Type, [&](auto &arena) {
            const auto range = arena.Allocate(entry.Values.Count);
            if (range.Count) CaptureCopy(copies, arena.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), entry.Values.Offset, range.Offset, range.Count);
            return range;
        });
        pair.y = AcquireAttribute({.Record = id, .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Values = values, .Default = entry.Default});
    }
    attributes.clear();
    for (const auto pair : kept) attributes.push_back(pair.y);
    for (const auto pair : whole) attributes.push_back(pair.y);
    std::ranges::sort(attributes);
    CopyBlocks(copies, id, source, restored);
    return change;
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
        const auto mask = [&](uint32_t block) { return from.Arena->Get({from.First + block, 1u})[0]; };
        for (const auto block : GetBlockList(id, SelectionDomains[s]).Blocks) {
            if (selection[block] != mask(block)) update.Blocks[s].push_back(block);
        }
        EditSelectionBlocks(SelectionElements[s], update.Blocks[s], [&](uint32_t block, auto &words) { words = mask(block); });
    }
    if (Records.at(id).SelectionSummary.Count) UpdateSelection(r, chain, std::array{update});
}

void MeshStore::Join(state::Registry &r, uint32_t id, uint32_t source) {
    const profile::CpuScope scope{"JoinMeshes"};
    ReleaseRender(std::array{&WriteRecord(source)});
    // Layers either record carries cover the join, and deform channels follow the destination.
    {
        auto &record = WriteRecord(id);
        const auto &src = Records.at(source);
        record.CornerAttributes |= src.CornerAttributes;
        record.VertexAttributes |= src.VertexAttributes;
        record.FacePrimitivesReady |= src.FacePrimitivesReady;
        record.VertexPrimitivesReady |= src.VertexPrimitivesReady;
        record.ConnectivityFaceStarts |= src.ConnectivityFaceStarts;
        record.HasAuthoredNormals |= src.HasAuthoredNormals;
    }
    // A sector root pairing that differs leaves the destination's classification to derive anew.
    const bool reclassify = Records[id].Classification != Records[source].Classification || bool(Records[id].CornerSectors.Count) != bool(Records[source].CornerSectors.Count);
    // The source's blocks follow the destination's in its tables, so their handles and references shift by the destination's block counts.
    std::array<uint32_t, 6> shifts{}, counts{};
    DomainBlocks moved;
    for (const auto domain : ElementDomains) {
        WithDomain(Buffers, domain, [&](auto &arena) {
            const auto d = uint32_t(domain);
            auto &from = DomainSet(WriteRecord(source), domain);
            arena.ForEachBlock(from, [&](uint32_t block, const auto &) { moved[d].push_back(block); });
            counts[d] = arena.HandleBound(from) / MeshElementBlockSize;
            shifts[d] = arena.Adopt(DomainSet(WriteRecord(id), domain), from);
            std::ranges::sort(moved[d]);
        });
    }
    CloneCopies copies;
    // Append source layer values to the destination, using defaults for absent source layers.
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&entries, auto &&range) {
        const auto d = uint32_t(info.Elements);
        const auto per = entries(Records[id]);
        if (!per) return;
        auto to = *range(Records[id]);
        const auto from = *range(Records[source]);
        const bool fresh = !to.Count;
        if (layer.Cover(to, shifts[d] + counts[d], per)) *range(WriteRecord(id)) = to;
        if (fresh) layer.Clear(to, {0u, shifts[d]}, per);
        if (!counts[d]) return;
        const auto held = from.Count ? entries(Records[source]) : 0u;
        constexpr auto Bytes = sizeof(typename std::remove_cvref_t<decltype(layer)>::Block);
        if (held == per) return CaptureCopy(copies, layer.Values.Buffer, Bytes, from.Offset, to.Offset + shifts[d] * per, counts[d] * per);
        layer.Clear(to, {shifts[d], counts[d]}, per);
        for (uint32_t local = 0u; local < counts[d]; ++local) {
            for (uint32_t e = 0u; e < std::min(held, per); ++e) CaptureCopy(copies, layer.Values.Buffer, Bytes, from.Offset + local * held + e, to.Offset + (shifts[d] + local) * per + e);
        }
    });
    // Join attributes with matching name, type, and domain; use defaults for missing values.
    const auto join_attribute = [&](uint32_t attribute, uint32_t from) {
        const auto entry = NamedAttributes[attribute];
        const auto d = uint32_t(ElementDomainOf(entry.Domain));
        WithAttributeValues(Buffers.Named, entry.Type, [&](auto &values) {
            using Block = ArenaValue<std::remove_cvref_t<decltype(values)>>;
            auto to = entry.Values;
            const bool fresh = !to.Count;
            if (CoverRange(values, to, shifts[d] + counts[d])) WriteAttribute(attribute).Values = to;
            if (fresh) FillRange(values, to, {0u, shifts[d]}, FilledBlock<Block>(entry.Default));
            if (!counts[d]) return;
            if (from == InvalidOffset) FillRange(values, to, {shifts[d], counts[d]}, FilledBlock<Block>(entry.Default));
            else CaptureCopy(copies, values.Buffer, sizeof(Block), NamedAttributes[from].Values.Offset, to.Offset + shifts[d], counts[d]);
        });
    };
    std::vector<uint32_t> own;
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &) { own.push_back(attribute); });
    for (const auto attribute : own) {
        const auto &entry = NamedAttributes[attribute];
        const auto from = FindAttribute(source, entry.NameView());
        join_attribute(attribute, from != InvalidOffset && NamedAttributes[from].Type == entry.Type && NamedAttributes[from].Domain == entry.Domain ? from : InvalidOffset);
    }
    std::vector<uint32_t> theirs;
    ForEachNamedAttribute(source, [&](uint32_t attribute, const NamedAttribute &) { theirs.push_back(attribute); });
    for (const auto attribute : theirs) {
        const auto entry = NamedAttributes[attribute];
        if (FindAttribute(id, entry.NameView()) == InvalidOffset) join_attribute(AddAttribute(id, entry.Domain, entry.Type, entry.NameView(), entry.Default).value(), attribute);
    }
    // The source's primitives follow the destination's in its material table.
    // Each side takes one primitive at least, so a side without a primitive table reads primitive 0 of its own run.
    const auto own_table = Records[id].PrimitiveMaterials, their_table = Records[source].PrimitiveMaterials;
    const bool tabled = own_table.Count || their_table.Count;
    const auto primitives = std::max(own_table.Count, 1u);
    if (tabled) {
        std::vector<uint32_t> merged(primitives + std::max(their_table.Count, 1u), 0u);
        std::ranges::copy(Buffers.PrimitiveMaterials.Get(own_table), merged.begin());
        std::ranges::copy(Buffers.PrimitiveMaterials.Get(their_table), merged.begin() + primitives);
        auto &record = WriteRecord(id);
        if (own_table.Count) Buffers.PrimitiveMaterials.Release(own_table);
        record.PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(std::span<const uint32_t>{merged});
    }
    // Every moved reference adds its domain's shift, through the roster's referencing arenas, sector roots and vertex fans.
    const auto shift = [&](mtl::Buffer &buffer, uint64_t first, uint64_t words, Domain domain, uint32_t stride = 1u) {
        if (!shifts[uint32_t(domain)] || !words) return;
        buffer.CaptureWrite(first * 4u, words * 4u);
        copies.Offset(buffer, first * 4u, uint32_t((words + stride - 1u) / stride), shifts[uint32_t(domain)] * MeshElementBlockSize, stride);
    };
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (info.References == Domain::None) return;
        constexpr auto Words = uint64_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>) / sizeof(uint32_t));
        const auto &blocks = moved[uint32_t(info.Elements)];
        ForEachIndexRun(blocks, [&](size_t first, size_t count) { shift(arena.Buffer, uint64_t(blocks[first]) * MeshElementBlockSize * Words, uint64_t(count) * MeshElementBlockSize * Words, info.References); });
    });
    {
        constexpr auto H = uint32_t(Domain::Halfedge);
        if (const auto sectors = Records[id].CornerSectors; sectors.Count) shift(Buffers.CornerSectors.Values.Buffer, uint64_t(sectors.Offset + shifts[H]) * MeshElementBlockSize, uint64_t(counts[H]) * MeshElementBlockSize, Domain::Halfedge);
    }
    {
        const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
        for (const auto [domain, word] : {std::pair{Domain::Halfedge, 0u}, std::pair{Domain::Face, 1u}}) {
            for (const auto block : moved[uint32_t(Domain::Vertex)]) {
                Buffers.Vertices.ForEachSlot(block, [&](uint32_t slot) {
                    const auto root = roots[block * MeshElementBlockSize + slot];
                    if (root.y) shift(Buffers.VertexFans.Items.Buffer, uint64_t(root.x) * 2u + word, 2u * uint64_t(root.y) - 1u, domain, 2u);
                });
            }
        }
    }
    WriteRecord(id).TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
    mtl::ComputeChain chain{BufferContext()};
    copies.Encode(chain, GetMeshPipelines(r));
    chain.Submit();
    if (tabled) {
        for (const auto [layer, values, domain] : {std::tuple{&Buffers.FacePrimitives, Records[id].FacePrimitives, Domain::Face}, std::tuple{&Buffers.VertexPrimitives, Records[id].VertexPrimitives, Domain::Vertex}}) {
            if (values.Count) {
                for (auto &primitive : layer->Edit(values, {shifts[uint32_t(domain)] * MeshElementBlockSize, counts[uint32_t(domain)] * MeshElementBlockSize})) primitive += primitives;
            }
        }
    }
    if (reclassify) UpdateCornerClassification(r, chain, std::array{id});
    UpdateSelection(r, chain, std::array{SelectionUpdate{.StoreId = id, .Blocks = {moved[uint32_t(Domain::Vertex)], moved[uint32_t(Domain::Edge)], moved[uint32_t(Domain::Face)]}}});
    chain.Submit();
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
        // The copied payload ranges include only used values.
        for (const auto payload : RenderPayloads) WithPayload(render, target, payload, [](auto &, PayloadRange &range) { range.Capacity = range.Values.Count; });
        sources.push_back(source_ids[i]);
        clones.push_back(clone_ids[i]);
    }
    if (sources.empty()) return;
    render.SyncMirrors();
    for (uint32_t c = 0u; c < sources.size(); ++c) {
        const auto &source = Records[sources[c]];
        auto &target = Records[clones[c]];
        // Pair collection blocks by local block index and copy their indexed values byte for byte.
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
                // A material field names the clone's own table.
                for (const auto &field : info.Fields) {
                    if (!field.Target && !field.Payload) copies.Offset(arena.Buffer, uint64_t(to) * stride + field.Offset, count * block_size, target.PrimitiveMaterials.Offset - source.PrimitiveMaterials.Offset, uint32_t(Size / sizeof(uint32_t)));
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
    // Visits the live members in ascending handle order with their addresses.
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

// Call fn(info, value) for live render-arena members in table order and for render-range values except element owners.
// Visit spatial nodes indexed by the finest triangle clusters.
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

// Require each table entry to reference its owned block and the header to match the first and last nonempty entries.
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
// Whether a per-block mask contains bits only at the set's live members.
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
        // No two records share a set, and each set's table is well formed.
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
        // Vertex fan items name the record's corners and faces.
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
        if (const auto sectors = Buffers.CornerSectors.View(r.CornerSectors); !sectors.empty()) {
            tables.ForEachLive(Domain::Halfedge, [&](uint32_t h, uint32_t) {
                if (invalid.empty() && !names(Domain::Halfedge, sectors[h])) invalid = "sector root of corner " + std::to_string(h);
            });
        }
        if (!invalid.empty()) return fail("names a foreign or dead element in its " + invalid);
        // Require layer and attribute ranges to cover their domain handle bounds; absent layers have empty ranges.
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
        // Require render references to identify live set members, used payload values, or material-table entries.
        // Finest clusters must own their rendered elements and reference live corners.
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
        // A traversal leaf lists its clusters and an internal node lists none.
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
    put(r.SkinBlocksReady), put(r.MorphBlocksReady), put(r.MorphTargetCount), put(r.TriangleCount), put(r.HasAuthoredNormals), put(r.MorphShadingAuthored);
    put_all(r.DefaultMorphWeights);
    const RecordTables tables{Buffers, r};
    // Each table's shape, with an empty mask at an entry holding no block.
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
    // Render values at live members in table order, with a material field hashing whether it names the record's table.
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
        std::ranges::fill(Buffers.VertexCorners.GetMutable(range), uvec2{InvalidOffset, 0u});
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
    for (const auto id : ids) {
        for (uint32_t d = 0u; d < 3u; ++d) Buffers.SelectionTree.Release(3u * id + d);
    }
    ForEachLayer(Buffers, [&](auto &layer, const LayerInfo &info, auto &&, auto &&range) {
        if (info.Render) return;
        std::vector<Range> ranges;
        for (const auto id : ids) AppendRange(ranges, *range(Records[id]));
        layer.Values.Release(std::move(ranges));
    });
    for (uint32_t attribute = 0u; attribute < NamedAttributes.size(); ++attribute) {
        if (std::ranges::binary_search(ids, NamedAttributes[attribute].Record)) ReleaseAttribute(attribute);
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
        // Release meshlets and their payload ranges from the fragment render owner.
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
    // Clear released block masks before reuse.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if constexpr (std::same_as<std::remove_cvref_t<decltype(arena)>, BufferArena<uint32_t>>) {
            if (!info.Render || !info.BlockIndexed) return;
            const auto words = arena.Buffer.template GetSpan<uint32_t>();
            for (const auto block : blocks[uint32_t(*info.Render)]) {
                if (words[block]) arena.GetMutable({block, 1u})[0] = 0u;
            }
        }
    });
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
    {
        const std::scoped_lock lock{m.BlockListLock};
        m.BlockLists.Reset();
        m.BlockListEntries.clear();
        m.RetiredBlockLists.clear();
    }
    m.FreeIds.clear();
    m.Released.clear();
    m.Unshaded.clear();
    for (auto &log : m.WriteLogs) log.Extents.clear();
    m.WrittenAttributes.clear();
}

VertexEdgeIncidence MeshStore::GetVertexEdgeIncidence(uint32_t id) const {
    return {GetConnectivity(id)};
}

uint32_t MeshStore::AcquireId(Record &&record) {
    if (!FreeIds.empty()) {
        const auto reused = FreeIds.back();
        if (Tracked) Tracked->Free.Write(FreeIds.size() - 1, 1);
        FreeIds.pop_back();
        record.StoreId = reused;
        WriteRecord(reused) = std::move(record);
        for (uint32_t d = 0u; d < 3u; ++d) Buffers.SelectionTree.Release(3u * reused + d);
        return reused;
    }
    if (Tracked) {
        Tracked->Entries.Write(Records.size(), 1);
        Tracked->Dirty.push_back(uint32_t(Records.size()));
    }
    Records.emplace_back(std::move(record));
    DerivedRecords.emplace_back();
    const auto id = uint32_t(Records.size() - 1);
    Records[id].StoreId = id;
    Buffers.Render.MeshRecords.Mirror({0, uint32_t(Records.size())});
    for (uint32_t d = 0u; d < 3u; ++d) Buffers.SelectionTree.Release(3u * id + d);
    return id;
}
