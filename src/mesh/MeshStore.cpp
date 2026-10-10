#include "MeshStore.h"
#include "gpu/PolygonTriangulation.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"

#include "CornerNormalOffset.h"
#include "Profile.h"
#include "SortUnique.h"
#include "project/store/Pages.h"

#include <map>

namespace {
constexpr uint32_t UniformFaceMode{uint32_t(CornerClassMode::UniformFace)};

constexpr auto NoAllocation = [](auto &) -> Range * { return nullptr; };

// The change bits a tracked arena reports and its history name, with a derived arena left unnamed.
// A mirror arena shares its ranges with a master arena earlier in the roster and allocates nothing of its own.
using Domain = MeshStore::ElementDomain;

struct ArenaInfo {
    uint32_t Bits{};
    const char *Name{};
    bool Mirror{false};
    Domain Elements{Domain::None};
    bool BlockIndexed{false};
    const BufferArena<uint32_t> *ValueOwners{};
    const BufferArena<uint32_t> *AttributeOwners{}; // A named attribute payload arena's owners
    bool Tracked() const { return Name != nullptr; }
};

// One roster owns optional attribute lifecycle, allocation and history routing.
// entries(record) is the payload blocks each of the record's element blocks owns, and zero when the record lacks the attribute.
void ForEachAttribute(auto &b, auto &&f) {
    using R = MeshStore::Record;
    using enum MeshStore::ChangeBits;
    f(b.FacePrimitives, Domain::Face, AttributesChanged, "FacePrimitive", "FacePrimitiveBlocks", "FacePrimitiveOwners", [](const R &r) { return uint32_t(r.FacePrimitivesReady); });
    f(b.VertexPrimitives, Domain::Vertex, AttributesChanged, "VertexPrimitive", "VertexPrimitiveBlocks", "VertexPrimitiveOwners", [](const R &r) { return uint32_t(r.VertexPrimitivesReady); });
    f(b.Skin, Domain::Vertex, AttributesChanged, "Skin", "SkinBlocks", "SkinOwners", [](const R &r) { return uint32_t(r.SkinBlocksReady); });
    f(b.CustomNormals, Domain::Halfedge, AttributesChanged, "CustomNormal", "CustomNormalBlocks", "CustomNormalOwners", [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Normal) != 0u); });
    f(b.CornerTangents, Domain::Halfedge, AttributesChanged, "CornerTangent", "CornerTangentBlocks", "CornerTangentOwners", [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Tangent) != 0u); });
    f(b.CornerColors, Domain::Halfedge, AttributesChanged, "CornerColor", "CornerColorBlocks", "CornerColorOwners", [](const R &r) { return uint32_t((r.CornerAttributes & MeshAttributeBit_Color0) != 0u); });
    f(b.VertexColors, Domain::Vertex, AttributesChanged, "VertexColor", "VertexColorBlocks", "VertexColorOwners", [](const R &r) { return uint32_t((r.VertexAttributes & MeshAttributeBit_Color0) != 0u); });
    constexpr std::array Names{"CornerUv0", "CornerUv1", "CornerUv2", "CornerUv3"};
    constexpr std::array Blocks{"CornerUv0Blocks", "CornerUv1Blocks", "CornerUv2Blocks", "CornerUv3Blocks"};
    constexpr std::array Owners{"CornerUv0Owners", "CornerUv1Owners", "CornerUv2Owners", "CornerUv3Owners"};
    for (uint32_t i = 0; i < 4; ++i)
        f(b.CornerUvs[i], Domain::Halfedge, AttributesChanged, Names[i], Blocks[i], Owners[i], [i](const R &r) { return uint32_t((r.CornerAttributes & (MeshAttributeBit_TexCoord0 << i)) != 0u); });
    f(b.Morph, Domain::Vertex, DeformChanged, "MorphBlockValues", "MorphBlockBindings", "MorphBlockOwners", [](const R &r) { return r.MorphBlocksReady ? r.MorphTargetCount : 0u; });
}

// The arena roster: calls f(arena, info, allocation) for every arena, where allocation(record) borrows its sole allocation, or returns null.
// Tracked arenas come first in their history order, then the derived arenas and the mirrors, each after its master.
void ForEachArena(MeshArenas &b, auto &&f) {
    using enum MeshStore::ChangeBits;
    f(b.Vertices, ArenaInfo{GeometryChanged, "Vertices", false, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.FaceTriangles, ArenaInfo{TopologyChanged, "FaceTriangleStart", false, Domain::Face}, [](auto &e) { return &e.FaceData; });
    ForEachAttribute(b, [&](auto &a, Domain domain, uint32_t bits, const char *values, const char *blocks, const char *owners, auto &&) {
        f(a.Values, ArenaInfo{bits, values, false, domain, false, &a.Owners}, NoAllocation);
        f(a.Blocks, ArenaInfo{bits, blocks, false, domain, true}, NoAllocation);
        f(a.Owners, ArenaInfo{0, owners}, NoAllocation);
    });
    f(b.PrimitiveMaterials, ArenaInfo{AttributesChanged, "PrimitiveMaterial"}, [](auto &e) { return &e.PrimitiveMaterials; });
    f(b.VertexSelection, ArenaInfo{SelectionChanged, "VertexSelection", true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeSelection, ArenaInfo{SelectionChanged, "EdgeSelection", true, Domain::Edge, true}, NoAllocation);
    f(b.FaceSelection, ArenaInfo{SelectionChanged, "FaceSelection", true, Domain::Face, true}, NoAllocation);
    f(b.VertexHidden, ArenaInfo{SelectionChanged, "VertexHidden", true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeHidden, ArenaInfo{SelectionChanged, "EdgeHidden", true, Domain::Edge, true}, NoAllocation);
    f(b.FaceHidden, ArenaInfo{SelectionChanged, "FaceHidden", true, Domain::Face, true}, NoAllocation);
    f(b.SelectionSummary, ArenaInfo{SelectionChanged, "SelectionSummary"}, [](auto &e) { return &e.SelectionSummary; });
    f(b.Triangles, ArenaInfo{TopologyChanged, "Triangle", false, Domain::Triangle}, [](auto &e) { return &e.TriangleData; });
    f(b.FaceCorners, ArenaInfo{TopologyChanged, "FaceCorner", false, Domain::Halfedge}, [](auto &e) { return &e.FaceCorners; });
    f(b.EdgeHalfedges, ArenaInfo{TopologyChanged, "EdgeHalfedge", false, Domain::Edge}, [](auto &e) { return &e.EdgeData; });
    f(b.OutgoingHalfedges, ArenaInfo{TopologyChanged, "OutgoingHalfedge", true, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.VertexCorners, ArenaInfo{TopologyChanged, "VertexCorner", true, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.OppositeHalfedges, ArenaInfo{TopologyChanged, "OppositeHalfedge", true, Domain::Halfedge}, [](auto &e) { return &e.FaceCorners; });
    f(b.HalfedgeEdges, ArenaInfo{TopologyChanged, "HalfedgeEdge", true, Domain::Halfedge}, [](auto &e) { return &e.FaceCorners; });
    f(b.HalfedgeFaces, ArenaInfo{TopologyChanged, "HalfedgeFace", true, Domain::Halfedge}, [](auto &e) { return &e.FaceCorners; });
    f(b.FaceRanges, ArenaInfo{TopologyChanged, "FaceRange", true, Domain::Face}, [](auto &e) { return &e.FaceData; });
    f(b.EdgeSharpness, ArenaInfo{ShadingChanged, "EdgeSharpness", true, Domain::Edge}, [](auto &e) { return &e.EdgeData; });
    f(b.PointNormals, ArenaInfo{ShadingChanged, "PointNormal"}, [](auto &e) { return &e.PointNormals; });
    f(b.TetPositions, ArenaInfo{0, "TetPosition"}, NoAllocation);
    f(b.TetEdgeIndices, ArenaInfo{0, "TetEdgeIndex"}, NoAllocation);
    f(b.FaceSharpness, ArenaInfo{ShadingChanged, "FaceSharpness", true, Domain::Face}, [](auto &e) { return &e.FaceData; });
    f(b.SoundVertices, ArenaInfo{}, NoAllocation);
    f(b.VertexAggregates, ArenaInfo{0, nullptr, true, Domain::Vertex, true}, NoAllocation);
    f(b.EdgeAggregates, ArenaInfo{0, nullptr, true, Domain::Edge, true}, NoAllocation);
    f(b.FaceAggregates, ArenaInfo{0, nullptr, true, Domain::Face, true}, NoAllocation);
    f(b.CornerSectors.Values, ArenaInfo{0, "CornerSectorsValues"}, NoAllocation);
    f(b.CornerSectors.Blocks, ArenaInfo{0, "CornerSectorsBlocks"}, NoAllocation);
    f(b.CornerSectors.Owners, ArenaInfo{0, "CornerSectorsOwners"}, NoAllocation);
    f(b.NormalSectors.Values, ArenaInfo{0, "NormalSectorsValues"}, NoAllocation);
    f(b.NormalSectors.Blocks, ArenaInfo{0, "NormalSectorsBlocks"}, NoAllocation);
    f(b.NormalSectors.Owners, ArenaInfo{0, "NormalSectorsOwners"}, NoAllocation);
    f(b.BaseVertexNormals, ArenaInfo{0, "BaseVertexNormal", true, Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.BaseFaceNormals, ArenaInfo{0, "BaseFaceNormal", true, Domain::Face}, [](auto &e) { return &e.FaceData; });
    // Named attribute payloads map their changes to records through their owners.
    constexpr std::array PayloadNames{"AttributeFloats", "AttributeVec2s", "AttributeVec3s", "AttributeVec4s", "AttributeInts", "AttributeBools"};
    constexpr std::array PayloadOwnerNames{"AttributeFloatOwners", "AttributeVec2Owners", "AttributeVec3Owners", "AttributeVec4Owners", "AttributeIntOwners", "AttributeBoolOwners"};
    for (uint32_t i = 0u; i < AttributeTypes.size(); ++i) WithPayloads(b.Named, AttributeTypes[i], [&](auto &payloads) {
        f(payloads.Values, ArenaInfo{.Bits = AttributesChanged, .Name = PayloadNames[i], .AttributeOwners = &payloads.Owners}, NoAllocation);
        f(payloads.Owners, ArenaInfo{0, PayloadOwnerNames[i], true}, NoAllocation);
    });
    f(b.Named.Tables, ArenaInfo{0, "AttributeTables"}, NoAllocation);
    // Render arenas, whose records the roots and construction ranges of the records own.
    // A mirror of a render arena holds bytes only, sized by its master's allocations.
    auto &render = b.Render;
    f(render.ExtrasFaces, ArenaInfo{0, "ExtrasFaces"}, NoAllocation);
    f(render.ExtrasEdges, ArenaInfo{0, "ExtrasEdges"}, NoAllocation);
    f(render.Meshlets, ArenaInfo{0, "Meshlets"}, NoAllocation);
    f(render.MeshletSpatialNodes, ArenaInfo{0, "MeshletSpatialNodes", true}, NoAllocation);
    f(render.ActiveMeshlets.Nodes, ArenaInfo{0, "MeshletIndexNodes"}, NoAllocation);
    f(render.ActiveMeshlets.Leaves, ArenaInfo{0, "MeshletIndexLeaves"}, NoAllocation);
    f(render.MeshletTriangleIds, ArenaInfo{0, "MeshletTriangleIds"}, NoAllocation);
    f(render.MeshletVertexCorners, ArenaInfo{0, "MeshletVertexCorners"}, NoAllocation);
    f(render.MeshletLocalTriangles, ArenaInfo{0, "MeshletLocalTriangles"}, NoAllocation);
    f(render.ClusterGroups, ArenaInfo{0, "ClusterGroups"}, NoAllocation);
    f(render.LodNodes, ArenaInfo{0, "LodNodes"}, NoAllocation);
    f(render.MeshletLodLeaves, ArenaInfo{0, "MeshletLodLeaves", true}, NoAllocation);
    f(render.LodParents, ArenaInfo{0, "LodParents", true}, NoAllocation);
    f(render.GroupLinks, ArenaInfo{0, "GroupLinks", true}, NoAllocation);
    f(render.GroupClusterIds, ArenaInfo{0, "GroupClusterIds"}, NoAllocation);
    f(render.Primitives, ArenaInfo{0, "Primitives"}, NoAllocation);
    f(render.PrimitiveRoutes, ArenaInfo{0, "PrimitiveRoutes"}, NoAllocation);
    // GPU mesh records hold runtime slots, which the restore rebuilds, so they have no history.
    f(render.MeshRecords, ArenaInfo{}, NoAllocation);
    constexpr std::array OwnerNames{"ElementMeshlets0", "ElementMeshlets1", "ElementMeshlets2"};
    constexpr std::array OwnerBlockNames{"ElementMeshlets0Blocks", "ElementMeshlets1Blocks", "ElementMeshlets2Blocks"};
    constexpr std::array OwnerOwnerNames{"ElementMeshlets0Owners", "ElementMeshlets1Owners", "ElementMeshlets2Owners"};
    for (uint32_t i = 0u; i < 3u; ++i) {
        f(render.ElementMeshlets[i].Values, ArenaInfo{0, OwnerNames[i]}, NoAllocation);
        f(render.ElementMeshlets[i].Blocks, ArenaInfo{0, OwnerBlockNames[i]}, NoAllocation);
        f(render.ElementMeshlets[i].Owners, ArenaInfo{0, OwnerOwnerNames[i]}, NoAllocation);
    }
}

template<typename Arena> using ArenaValue = typename decltype(std::declval<const Arena &>().Get(Range{}))::value_type;

constexpr bool SelectableDomain(Domain domain) { return domain == Domain::Vertex || domain == Domain::Edge || domain == Domain::Face; }

// A payload block holding `value` at every element.
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

void ClearVertexRoots(MeshArenas &a, ElementSetRef vertices) {
    a.Vertices.ForEachBlock(vertices, [&](uint32_t block, const auto &) {
        const Range range{block * MeshElementBlockSize, MeshElementBlockSize};
        a.VertexCorners.Buffer.CaptureWrite(uint64_t(range.Offset) * sizeof(uvec2), uint64_t(range.Count) * sizeof(uvec2));
        std::ranges::fill(a.VertexCorners.GetMutable(range), uvec2{InvalidOffset, 0u});
    });
}

// Clears the cluster, hierarchy and owner fields, keeping the record's canonical fields, extras index data and id.
void ResetRender(MeshStore::Record &r) {
    const MeshStore::Record fresh{};
    r.Primitives = fresh.Primitives;
    r.Meshlets = fresh.Meshlets;
    r.MeshletTriangles = fresh.MeshletTriangles;
    r.MeshletVertices = fresh.MeshletVertices;
    r.MeshletLocalTriangles = fresh.MeshletLocalTriangles;
    r.MeshletRoot = r.PrimitiveRoot = r.SpatialRoot = InvalidOffset;
    r.PrimitiveRoutes = fresh.PrimitiveRoutes;
    r.MeshletRevision = 0u;
    r.Level0Count = 0u;
    r.RenderTopologies = 0u;
    r.ElementMeshletOrigins = fresh.ElementMeshletOrigins;
    r.ElementMeshletBlockCounts = fresh.ElementMeshletBlockCounts;
    r.ClusterGroups = r.LodNodes = r.CoarseVertices = r.CoarseLocalTriangles = fresh.ClusterGroups;
    r.GroupRoot = r.NodeRoot = InvalidOffset;
    r.LodDepth = 0u;
    r.PositionDirtyRoot = r.DirtyGroupRoot = InvalidOffset;
}

} // namespace

RenderArenas::RenderArenas(mtl::BufferContext &ctx)
    : ExtrasFaces{ctx, SlotType::IndexBuffer},
      ExtrasEdges{ctx, SlotType::IndexBuffer},
      Meshlets{ctx, SlotType::Buffer},
      MeshletSpatialNodes{ctx, SlotType::Buffer},
      ActiveMeshlets{ctx},
      MeshletTriangleIds{ctx, SlotType::Buffer},
      MeshletVertexCorners{ctx, SlotType::Buffer},
      MeshletLocalTriangles{ctx, SlotType::Buffer},
      ClusterGroups{ctx, SlotType::Buffer},
      LodNodes{ctx, SlotType::Buffer},
      MeshletLodLeaves{ctx, SlotType::Buffer},
      LodParents{ctx, SlotType::Buffer},
      GroupLinks{ctx, SlotType::Buffer},
      GroupClusterIds{ctx, SlotType::Buffer},
      Primitives{ctx, SlotType::Buffer},
      PrimitiveRoutes{ctx, SlotType::Buffer},
      MeshRecords{ctx, SlotType::Buffer},
      ElementMeshlets{{{ctx, SlotType::Buffer}, {ctx, SlotType::Buffer}, {ctx, SlotType::Buffer}}} {}

Range RenderArenas::AllocateMeshlets(uint32_t count) {
    const auto range = Meshlets.Allocate(count);
    MeshletLodLeaves.Mirror(range);
    MeshletSpatialNodes.Mirror(range);
    return range;
}

void RenderArenas::ReleaseMeshletStorage(std::span<const uint32_t> handles) {
    std::array<std::vector<Range>, 4> ranges;
    for (const auto handle : handles) {
        const auto &record = Meshlets.Get({handle, 1u})[0];
        if (record.RefinedGroup == InvalidOffset) ranges[0].push_back({record.TriangleOffset, record.TriangleCount});
        ranges[1].push_back({record.VertexOffset, record.VertexCount});
        if (record.Topology == 0u) ranges[2].push_back({record.LocalTriangleOffset, record.TriangleCount * 3u});
        ranges[3].push_back({handle, 1u});
    }
    MeshletTriangleIds.Release(std::move(ranges[0]));
    MeshletVertexCorners.Release(std::move(ranges[1]));
    MeshletLocalTriangles.Release(std::move(ranges[2]));
    Meshlets.Release(std::move(ranges[3]));
}

AttributeArenas::AttributeArenas(mtl::BufferContext &ctx)
    : Floats{ctx}, Vec2s{ctx}, Vec3s{ctx}, Vec4s{ctx}, Ints{ctx}, Bools{ctx}, Tables{ctx, SlotType::Buffer} {}

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
      CornerSectors{ctx, SlotType::Buffer},
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
        const BufferArena<uint32_t> *ValueOwners{};
    };
    struct AttributeStream {
        mtl::Buffer *Buffer;
        uint32_t Stride;
        const BufferArena<uint32_t> *Owners;
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
        for (const auto [domain, set] : indexed.Sets)
            if (const auto it = Owners[domain].find(set); it != Owners[domain].end() && it->second == id) Owners[domain].erase(it);
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
            if (!info.Tracked() || info.Mirror) return;
            constexpr auto Stride = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
            if (const auto *allocation = ranges(mesh.Records[id])) {
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*allocation)>, ElementSetRef>) {
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
          .Skin = Buffers.Skin.Ref(),
          .Morph = Buffers.Morph.Ref(),
          .TetPosition = Buffers.TetPositions.Buffer.Slot,
          .TetEdgeIndex = Buffers.TetEdgeIndices.Buffer.Slot,
          .SoundVertex = Buffers.SoundVertices.Buffer.Slot,
          .CornerSector = Buffers.CornerSectors.Ref(),
          .NormalSector = Buffers.NormalSectors.Ref(),
          .BaseVertexNormal = Buffers.BaseVertexNormals.Buffer.Slot,
          .BaseFaceNormal = Buffers.BaseFaceNormals.Buffer.Slot,
      },
      BlockLists{ctx, SlotType::Buffer} {
    // Element value arenas, their mirrors and layers, and their membership log their writes, and selection state derives from the blocks they name.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (info.Elements == Domain::None || info.Bits == SelectionChanged || !info.Tracked()) return;
        const auto stride = uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>));
        if (info.ValueOwners) WriteLogs.push_back({&arena.Buffer, info.Elements, WriteIndex::Payloads, stride, info.ValueOwners});
        else WriteLogs.push_back({&arena.Buffer, info.Elements, info.BlockIndexed ? WriteIndex::Blocks : WriteIndex::Elements, stride});
        if constexpr (requires { arena.Blocks; }) WriteLogs.push_back({&arena.Blocks.Buffer, info.Elements, WriteIndex::Blocks, uint32_t(sizeof(MeshElementBlock))});
    });
    WriteLogs.push_back({&Buffers.CornerSectors.Values.Buffer, Domain::Halfedge, WriteIndex::Payloads, uint32_t(sizeof(decltype(Buffers.CornerSectors)::Block)), &Buffers.CornerSectors.Owners});
    WriteLogs.push_back({&Buffers.NormalSectors.Values.Buffer, Domain::Halfedge, WriteIndex::Payloads, uint32_t(sizeof(decltype(Buffers.NormalSectors)::Block)), &Buffers.NormalSectors.Owners});
    for (auto &log : WriteLogs) log.Buffer->Writes = &log.Extents;
}
MeshStore::~MeshStore() = default;

MeshStore::Writes MeshStore::TakeWrites() {
    Writes writes;
    for (auto &log : WriteLogs) {
        auto &blocks = writes.Blocks[uint32_t(log.Domain)];
        const auto owners = log.Owners ? log.Owners->Buffer.GetSpan<uint32_t>() : std::span<const uint32_t>{};
        for (const auto [begin, end] : log.Extents) {
            const auto first = begin / log.Stride, last = (end + log.Stride - 1u) / log.Stride;
            switch (log.Index) {
                case WriteIndex::Elements:
                    for (auto block = first / MeshElementBlockSize; block <= (last - 1u) / MeshElementBlockSize; ++block) blocks.push_back(uint32_t(block));
                    break;
                case WriteIndex::Blocks:
                    for (auto block = first; block < last; ++block) blocks.push_back(uint32_t(block));
                    break;
                case WriteIndex::Payloads:
                    for (auto payload = first; payload < last; ++payload)
                        if (payload < owners.size() && owners[payload]) blocks.push_back(owners[payload] - 1u);
                    break;
            }
        }
        log.Extents.clear();
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
        }
    });
    Tracked = std::make_unique<HistoryState>(*this, history);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        if (info.AttributeOwners) Tracked->AttributeStreams.push_back({&arena.Buffer, uint32_t(sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>)), info.AttributeOwners});
        else if (info.Elements == Domain::None) Tracked->Ranges.try_emplace(&arena.Buffer);
        else Tracked->Streams.push_back({&arena.Buffer, info.Elements, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), info.Bits, info.BlockIndexed || info.ValueOwners ? HistoryState::StreamKind::Blocks : HistoryState::StreamKind::Values, info.ValueOwners});
        if constexpr (requires { arena.Blocks; }) {
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
    Range run{};
    elements.ForEachBlock(set, [&](uint32_t block, const auto &) {
        if (run.Count && block * MeshElementBlockSize != run.Offset + run.Count) {
            fn(run);
            run = {};
        }
        if (!run.Count) run.Offset = block * MeshElementBlockSize;
        run.Count += MeshElementBlockSize;
    });
    if (run.Count) fn(run);
}
void CaptureElementSet(const auto &values, const auto &elements, ElementSetRef set) {
    ForEachBlockRun(elements, set, [&](Range run) { CaptureRange(values, run); });
}
void CaptureRange(const auto &arena, ElementSetRef set) { CaptureElementSet(arena, arena, set); }

// Requires selected indices in ascending order.
void CaptureSelected(const mtl::Buffer &buffer, uint32_t stride, SelectionView selection) {
    const auto *history = buffer.History();
    if (!history && !buffer.Writes) return;
    const uint64_t page_size = history ? history->PageBytes : mtl::HistoryPageBytes;
    std::vector<uint32_t> pages;
    selection.ForEach([&](uint32_t handle) {
        const uint64_t offset = uint64_t(handle) * stride;
        buffer.LogWrite(offset, offset + stride);
        const auto first = offset / page_size, end = (offset + stride + page_size - 1) / page_size;
        for (auto page = first; page < end; ++page)
            if (pages.empty() || pages.back() < page) pages.push_back(uint32_t(page));
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
    switch (operation) {
        case EditSharpnessOperation::SetSelectedFaces:
            Buffers.FaceSharpness.Buffer.CaptureWriteElements(selection.Faces, sizeof(uint8_t));
            break;
        case EditSharpnessOperation::SetSelectedEdges:
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(selection.Edges, sizeof(uint8_t));
            break;
        case EditSharpnessOperation::SetVertexEdges: {
            const auto edges = GetVertexEdgeIncidence(id);
            std::vector<uint32_t> touched;
            for (const auto vertex : selection.Vertices)
                for (const auto edge : edges.Incident(vertex)) touched.push_back(edge);
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(touched, sizeof(uint8_t));
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
    const auto vertices = Buffers.Vertices.Dense(record.Vertices);
    if (record.SkinBlocksReady) Buffers.Skin.CaptureHandles(vertices);
    if (record.MorphBlocksReady) Buffers.Morph.CaptureHandles(vertices, record.MorphTargetCount);
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
    // Fan replacements always change canonical roots, which map changes to the
    // owning mesh. The item track still needs its extent queue drained.
    Buffers.VertexFans.Items.Buffer.History()->TakeChangedExtents();
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
    // Canonical streams resolve only dirty blocks through the current ownership
    // metadata. No per-mesh range index is built for these streams.
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
                for (uint64_t b = first_block; b < last_block; ++b) {
                    auto block = b;
                    if (stream.ValueOwners) {
                        const auto reverse = stream.ValueOwners->Buffer.GetSpan<uint32_t>();
                        if (b >= reverse.size() || reverse[b] == 0u) continue;
                        block = reverse[b] - 1u;
                    }
                    const auto root = stream.Kind == HistoryState::StreamKind::Sets ? uint32_t(block) : block < blocks.size() ? blocks[block].Owner :
                                                                                                                                InvalidOffset;
                    const auto it = owners.find(root);
                    if (stream.Kind != HistoryState::StreamKind::Sets && SelectableDomain(stream.Elements)) {
                        const auto domain = SelectableIndex(stream.Elements);
                        const auto previous = Buffers.SelectionTree.Owner(domain, uint32_t(block));
                        if (previous != InvalidOffset && (it == owners.end() || previous / 3u != it->second))
                            changed.push_back({previous / 3u, stream.Bits, {}, domain, uint32_t(block)});
                    }
                    if (it == owners.end()) continue;
                    Range vertices{};
                    if (stream.Bits == GeometryChanged) {
                        const auto start = std::max(first, b * MeshElementBlockSize);
                        const auto stop = std::min(last, (b + 1u) * MeshElementBlockSize);
                        vertices = {uint32_t(start), uint32_t(stop - start)};
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
    // A named attribute's payload block maps to the record of the attribute that owns it.
    for (const auto &stream : Tracked->AttributeStreams) {
        const auto owners = stream.Owners->Buffer.GetSpan<uint32_t>();
        for (const auto [begin, end] : stream.Buffer->History()->TakeChangedExtents())
            for (auto payload = begin / stream.Stride; payload < (end + stream.Stride - 1u) / stream.Stride; ++payload)
                if (payload < owners.size() && owners[payload]) changed.push_back({NamedAttributes[owners[payload] - 1u].Record, AttributesChanged});
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
        } else vertices.push_back(range.Vertices);
    }
    for (auto &change : changes)
        for (auto &blocks : change.Blocks) {
            SortUnique(blocks);
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
    Buffers.CornerSectors.ReserveBlocks(Buffers.FaceCorners.Capacity() / MeshElementBlockSize);
    Buffers.NormalSectors.ReserveBlocks(Buffers.FaceCorners.Capacity() / MeshElementBlockSize);
    Buffers.Query.Reserve(std::max({Buffers.Vertices.Capacity(), Buffers.EdgeHalfedges.Capacity(), Buffers.FaceTriangles.Capacity()}));
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Mirror || info.Elements == Domain::None) return;
        WithDomain(Buffers, info.Elements, [&](const auto &owner) {
            arena.Mirror({0, info.BlockIndexed ? owner.Capacity() / MeshElementBlockSize : owner.Capacity()});
        });
    });
    Buffers.Render.MeshRecords.Mirror({0, uint32_t(Records.size())});
    ForEachAttribute(Buffers, [&](auto &a, Domain domain, uint32_t, const char *, const char *, const char *, auto &&) {
        WithDomain(Buffers, domain, [&](const auto &owner) { a.ReserveBlocks(owner.Capacity() / MeshElementBlockSize); });
    });
}

void MeshStore::FinishRestore() {
    const auto changed = Tracked->Entries.Changed();
    Tracked->Dirty.append_range(changed);
    DerivedRecords.resize(Records.size());
    for (const auto id : changed)
        if (id < Records.size()) {
            DerivedRecords[id] = {};
            if (Records[id].Alive) DerivedRecords[id].NormalRevision = ++NextNormalRevision;
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
    // A new set, the triangle count and the first face's flags change the record.
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
    ForEachAttribute(Buffers, [&](auto &attributes, Domain owner, uint32_t, const char *, const char *, const char *, auto &&entries) {
        if (owner == domain && (entries(record) || first_face)) attributes.Attach(blocks, attributes.Absent, std::max(entries(record), 1u));
    });
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        if (ElementDomainOf(entry.Domain) == domain) AttachAttributeBlocks(attribute, blocks);
    });
    // The emitter writes the gained blocks and their mirrors, including holes sharing pages with surviving elements.
    ForEachArena(Buffers, [&](auto &values, const ArenaInfo &info, auto &&) {
        if (info.Elements != domain || info.ValueOwners || (info.BlockIndexed && !info.Mirror)) return;
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
    if (first_face) {
        record.FacePrimitivesReady = true;
        record.ConnectivityFaceStarts = true;
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
    if (SelectableDomain(domain)) WithDomain(Buffers, domain, [&](const auto &arena) {
        const auto set = DomainSet(Records[id], domain);
        EditHiddenBlocks(SelectionElements[SelectableIndex(domain)], blocks, [&](uint32_t block, auto &words) {
            const auto &member = arena.Blocks.Get({block, 1u})[0];
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[w] &= member.Owner == set.Index ? member.Live[w] : 0u;
        });
    });
    if (domain == Domain::Triangle) Records[id].TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
    if (domain == Domain::Halfedge) {
        auto &derived = DerivedRecords.at(id);
        const auto owner = Records[id].FaceCorners.Index;
        for (const auto block : blocks) {
            if (Buffers.FaceCorners.Blocks.Get({block, 1})[0].Owner == owner || !Buffers.CornerSectors.PayloadBlock(block)) continue;
            Buffers.CornerSectors.Release(block);
            Buffers.NormalSectors.Release(block);
            derived.NormalRevision = ++NextNormalRevision;
            --WriteRecord(id).SectorBlockCount;
        }
    }
    if (domain == Domain::Vertex) {
        // Dead slots, including every slot of a retired block, free their fans and clear their roots.
        const auto owner = Records[id].Vertices;
        std::vector<uvec2> released;
        for (const auto b : blocks) {
            const auto &block = Buffers.Vertices.Blocks.Get({b, 1u})[0];
            const auto live = block.Owner == owner.Index ? block.Live : decltype(block.Live){};
            auto roots = Buffers.VertexCorners.GetMutable({b * 256u, 256u});
            for (uint32_t i = 0u; i < 256u; ++i) {
                if (live[i / 32u] & (1u << (i % 32u))) continue;
                if (roots[i].y) released.push_back(roots[i]);
                roots[i] = {InvalidOffset, 0u};
            }
        }
        Buffers.VertexFans.Release(released);
    }
    const auto set = DomainSet(Records[id], domain);
    ForEachAttribute(Buffers, [&](auto &attributes, Domain owner, uint32_t, const char *, const char *, const char *, auto &&entries) {
        if (owner != domain || !entries(Records[id])) return;
        WithDomain(Buffers, domain, [&](const auto &arena) {
            for (const auto b : blocks)
                if (arena.Blocks.Get({b, 1})[0].Owner != set.Index) attributes.Release(b);
        });
    });
    // A retired set leaves every listed block.
    std::vector<uint32_t> left;
    WithDomain(Buffers, domain, [&](const auto &arena) {
        for (const auto b : blocks)
            if (!set || arena.Blocks.Get({b, 1})[0].Owner != set.Index) left.push_back(b);
    });
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        if (ElementDomainOf(entry.Domain) == domain) ReleaseAttributeBlocks(attribute, left);
    });
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
    bool any = false;
    for (uint32_t i = 0; i < corners.Count; i += MeshElementBlockSize) {
        const auto values = offsets.subspan(i, std::min(MeshElementBlockSize, corners.Count - i));
        const uint32_t h = corners.Offset + i;
        if (std::ranges::any_of(values, [](CustomNormal value) { return value.Offset.x >= 0.f; })) {
            Buffers.CustomNormals.Initialize({h, uint32_t(values.size())}, values);
            any = true;
        } else Buffers.CustomNormals.Release(h / MeshElementBlockSize);
    }
    if (any) record.CornerAttributes |= MeshAttributeBit_Normal;
    else record.CornerAttributes &= ~MeshAttributeBit_Normal;
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
    const auto attribute = AcquireAttribute(std::move(entry));
    const auto elements = ElementDomainOf(domain);
    std::vector<uint32_t> blocks;
    WithDomain(Buffers, elements, [&](const auto &arena) {
        arena.ForEachBlock(DomainSet(Records[id], elements), [&](uint32_t block, const auto &) { blocks.push_back(block); });
    });
    std::ranges::sort(blocks);
    AttachAttributeBlocks(attribute, blocks);
    return attribute;
}

uint32_t MeshStore::FindAttribute(uint32_t id, std::string_view name) const {
    for (uint32_t attribute = 0u; attribute < NamedAttributes.size(); ++attribute)
        if (NamedAttributes[attribute].Record == id && NamedAttributes[attribute].NameView() == name) return attribute;
    return InvalidOffset;
}

void MeshStore::ReleaseAttribute(uint32_t attribute) {
    const auto &entry = NamedAttributes.at(attribute);
    if (entry.Record == InvalidOffset) throw std::invalid_argument("Released attribute.");
    std::vector<uint32_t> blocks(entry.Table.Count);
    std::iota(blocks.begin(), blocks.end(), entry.FirstBlock);
    ReleaseAttributeBlocks(attribute, blocks);
    if (entry.Table.Count) Buffers.Named.Tables.Release(entry.Table);
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
    // Both attributes hold a payload at each element block of their record's domain.
    std::vector<uint32_t> blocks(entry.Table.Count);
    std::iota(blocks.begin(), blocks.end(), entry.FirstBlock);
    const auto from = AttributePayloadBlocks(attribute, blocks), to = AttributePayloadBlocks(*copy, blocks);
    WithPayloads(Buffers.Named, entry.Type, [&](auto &payloads) {
        using Block = typename std::remove_cvref_t<decltype(payloads)>::Block;
        payloads.Values.Buffer.CaptureWriteElements(to, sizeof(Block));
        auto *values = reinterpret_cast<Block *>(payloads.Values.Buffer.Contents().data());
        for (size_t i = 0u; i < from.size(); ++i) values[to[i]] = values[from[i]];
    });
    return copy;
}

void MeshStore::AttachAttributeBlocks(uint32_t attribute, std::span<const uint32_t> blocks) {
    if (blocks.empty()) return;
    auto &tables = Buffers.Named.Tables;
    const auto &entry = NamedAttributes.at(attribute);
    // A table run spans its blocks from its first, so a block outside it moves the attribute to a run spanning both.
    const auto first = entry.Table.Count ? std::min(entry.FirstBlock, blocks.front()) : blocks.front();
    const auto end = entry.Table.Count ? std::max(entry.FirstBlock + entry.Table.Count, blocks.back() + 1u) : blocks.back() + 1u;
    if (first != entry.FirstBlock || end - first != entry.Table.Count) {
        const auto run = tables.Allocate(end - first);
        auto words = tables.GetMutable(run);
        std::ranges::fill(words, 0u);
        if (entry.Table.Count) {
            std::ranges::copy(tables.Get(entry.Table), words.begin() + (entry.FirstBlock - first));
            tables.Release(entry.Table);
        }
        auto &moved = WriteAttribute(attribute);
        moved.Table = run;
        moved.FirstBlock = first;
    }
    WithPayloads(Buffers.Named, entry.Type, [&](auto &payloads) {
        using Block = typename std::remove_cvref_t<decltype(payloads)>::Block;
        const auto table = tables.Get(entry.Table);
        std::vector<uint32_t> unbound;
        for (const auto block : blocks)
            if (!table[block - entry.FirstBlock]) unbound.push_back(block);
        if (unbound.empty()) return;
        // Each payload is its own first-fit allocation, so released payloads are reused.
        std::vector<Range> runs;
        runs.reserve(unbound.size());
        auto allocation = payloads.Values.BeginAllocation();
        payloads.Values.ReserveAdditional(unbound.size());
        for (size_t i = 0u; i < unbound.size(); ++i) runs.push_back(payloads.Values.Allocate(1u));
        allocation.Commit();
        payloads.Owners.Mirror({0u, payloads.Values.HighWaterMark()});
        std::vector<uint32_t> words;
        words.reserve(unbound.size());
        for (const auto block : unbound) words.push_back(entry.Table.Offset + block - entry.FirstBlock);
        payloads.Values.Buffer.CaptureWriteRanges(runs, sizeof(Block));
        payloads.Owners.Buffer.CaptureWriteRanges(runs, sizeof(uint32_t));
        tables.Buffer.CaptureWriteElements(words, sizeof(uint32_t));
        auto *values = reinterpret_cast<Block *>(payloads.Values.Buffer.Contents().data());
        auto *owners = reinterpret_cast<uint32_t *>(payloads.Owners.Buffer.Contents().data());
        auto *table_words = reinterpret_cast<uint32_t *>(tables.Buffer.Contents().data());
        const auto filled = FilledBlock<Block>(entry.Default);
        for (size_t i = 0u; i < unbound.size(); ++i) {
            values[runs[i].Offset] = filled;
            owners[runs[i].Offset] = attribute + 1u;
            table_words[words[i]] = runs[i].Offset + 1u;
        }
    });
}

void MeshStore::ReleaseAttributeBlocks(uint32_t attribute, std::span<const uint32_t> blocks) {
    const auto &entry = NamedAttributes.at(attribute);
    auto &tables = Buffers.Named.Tables;
    const auto table = tables.Get(entry.Table);
    std::vector<uint32_t> words;
    std::vector<Range> payloads;
    for (const auto block : blocks) {
        const auto word = block - entry.FirstBlock;
        if (word >= table.size() || !table[word]) continue;
        words.push_back(entry.Table.Offset + word);
        payloads.push_back({table[word] - 1u, 1u});
    }
    if (words.empty()) return;
    tables.Buffer.CaptureWriteElements(words, sizeof(uint32_t));
    auto *table_words = reinterpret_cast<uint32_t *>(tables.Buffer.Contents().data());
    for (const auto word : words) table_words[word] = 0u;
    std::ranges::sort(payloads, {}, &Range::Offset);
    CoalesceRanges(payloads);
    WithPayloads(Buffers.Named, entry.Type, [&](auto &arenas) {
        for (const auto run : payloads) {
            std::ranges::fill(arenas.Owners.GetMutable(run), 0u);
            arenas.Values.Release(run);
        }
    });
}

NamedAttributeRef MeshStore::AttributeRef(uint32_t attribute) const {
    const auto &entry = NamedAttributes.at(attribute);
    const auto values = WithPayloads(Buffers.Named, entry.Type, [](const auto &payloads) { return payloads.Values.Buffer.Slot; });
    return {Buffers.Named.Tables.Buffer.Slot, values, entry.Table.Offset, entry.FirstBlock};
}

std::vector<uint32_t> MeshStore::AttributePayloadBlocks(uint32_t attribute, std::span<const uint32_t> blocks) const {
    const auto &entry = NamedAttributes.at(attribute);
    const auto table = Buffers.Named.Tables.Get(entry.Table);
    std::vector<uint32_t> payloads;
    for (const auto block : blocks)
        if (const auto word = block - entry.FirstBlock; word < table.size() && table[word]) payloads.push_back(table[word] - 1u);
    return payloads;
}

void MeshStore::CaptureAttributeWrites(uint32_t id, AttributeDomain domain, std::span<const uint32_t> blocks) const {
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        if (entry.Domain != domain) return;
        WithPayloads(Buffers.Named, entry.Type, [&](const auto &payloads) {
            payloads.Values.Buffer.CaptureWriteElements(AttributePayloadBlocks(attribute, blocks), sizeof(typename std::remove_cvref_t<decltype(payloads)>::Block));
        });
    });
}

void MeshStore::CopyAttributeBlocks(CloneCopies &copies, uint32_t from, uint32_t to, Range map, std::span<const uint32_t> sources) {
    if (sources.empty()) return;
    std::vector<uvec2> pairs;
    pairs.reserve(sources.size());
    for (const auto block : sources) pairs.push_back({block, copies.MapHandle(map, block * MeshElementBlockSize) / MeshElementBlockSize});
    std::ranges::sort(pairs, {}, &uvec2::y);
    std::vector<uint32_t> targets;
    targets.reserve(pairs.size());
    for (const auto pair : pairs) targets.push_back(pair.y);
    AttachAttributeBlocks(to, targets);
    const auto &source = NamedAttributes[from], &target = NamedAttributes[to];
    const auto tables = Buffers.Named.Tables.Buffer.GetSpan<uint32_t>();
    const auto payload = [&](const NamedAttribute &entry, uint32_t block) {
        const auto word = block - entry.FirstBlock;
        return word < entry.Table.Count ? tables[entry.Table.Offset + word] : 0u;
    };
    WithPayloads(Buffers.Named, source.Type, [&](auto &payloads) {
        constexpr auto Bytes = sizeof(typename std::remove_cvref_t<decltype(payloads)>::Block);
        for (const auto pair : pairs) {
            const auto read = payload(source, pair.x), write = payload(target, pair.y);
            if (!read || !write) continue;
            CaptureCopy(copies, payloads.Values.Buffer, Bytes, read - 1u, write - 1u);
        }
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
void MeshStore::EnsureSelectionState(state::Scene &r, mtl::ComputeChain &chain, std::span<const uint32_t> requested) {
    std::vector<uint32_t> ids{requested.begin(), requested.end()};
    SortUnique(ids);
    std::vector<SelectionUpdate> updates;
    for (const auto id : ids)
        if (!Records.at(id).SelectionSummary.Count) updates.push_back({.StoreId = id});
    if (updates.empty()) return;
    SyncMirrors();
    // Dead slots hold no selection, so the new state's masks are clear and only the aggregates need deriving.
    for (auto &update : updates) {
        WriteRecord(update.StoreId).SelectionSummary = Buffers.SelectionSummary.Allocate(1);
        WriteSelectionSummary(update.StoreId) = {};
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto list = GetBlockList(update.StoreId, SelectionDomains[d]);
            update.Blocks[d].assign(list.Blocks.begin(), list.Blocks.end());
        }
    }
    UpdateSelection(r, chain, updates);
    chain.Submit();
    for (const auto &update : updates) PublishSelectionSummary(update.StoreId);
}

SelectionView MeshStore::GetSelectedElements(uint32_t id, Element element) const {
    const auto domain = SelectionDomain(element);
    const auto &root = GetSelectionRoot(id, element);
    const auto leaves = AggregateArena(Buffers, domain).Buffer.GetSpan<SelectionAggregate>();
    return {SelectionArena(Buffers, domain).Buffer.GetSpan<uint32_t>(), Buffers.SelectionTree.Read(3u * id + SelectableIndex(domain), leaves), root.Selected};
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
        Buffers.EdgeHalfedges.Buffer.GetSpan<uint32_t>(),
        Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>(),
        record.EdgeData.Index,
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

bool MeshStore::IsLiveElement(uint32_t id, Element element, uint32_t handle) const {
    const auto domain = SelectionDomain(element);
    return WithDomain(Buffers, domain, [&](const auto &owner) {
        const auto set = DomainSet(Records.at(id), domain);
        if (!set || handle >= owner.Capacity()) return false;
        const auto &block = owner.Blocks.Get({handle / MeshElementBlockSize, 1u})[0];
        return block.Owner == set.Index && (block.Live[(handle & 255u) / 32u] & (1u << (handle & 31u))) != 0u;
    });
}
uint32_t MeshStore::GetSelectionBitOffset(uint32_t id, Element element) const {
    const auto domain = SelectionDomain(element);
    return WithDomain(Buffers, domain, [&](const auto &owner) { return owner.First(DomainSet(Records.at(id), domain)); });
}
uint32_t MeshStore::GetSelectionSlot(Element element) const { return SelectionArena(Buffers, SelectionDomain(element)).Buffer.Slot; }
EditSelectionStorage MeshStore::GetEditSelectionStorage(uint32_t id) const {
    const auto summary = Records.at(id).SelectionSummary;
    const auto binding = [&](Element element) -> SlotOffset {
        return {GetSelectionSlot(element), GetSelectionBitOffset(id, element) / 32u};
    };
    return {
        binding(Element::Vertex), binding(Element::Edge), binding(Element::Face),
        summary.Count > 0 ? Buffers.SelectionSummary.Slotted(summary) : SlottedRange{},
        GetHiddenSlot(Element::Vertex), GetHiddenSlot(Element::Edge), GetHiddenSlot(Element::Face)
    };
}
const EditSelectionSummary &MeshStore::GetSelectionSummary(uint32_t id) const { return Buffers.SelectionSummary.Get(Records.at(id).SelectionSummary)[0]; }

vec3 ComposeCornerNormal(CornerAttributeView<uint32_t> sectors, ElementAttributeView<NormalSector> normal_sectors, std::span<const uint8_t> sharpness, uint32_t mode, uint32_t ci, TriangleVertexView vertices, TriangleFaceView face_ids, const CornerNormalSources &sources) {
    const auto face = face_ids[ci / 3u];
    if (mode == UniformFaceMode || (mode == uint32_t(CornerClassMode::Mixed) && sharpness[face])) {
        const auto posed = sources.PosedFaceNormals.Find(face);
        return posed == InvalidOffset ? sources.FaceNormals[face] : sources.PosedFaceNormals.Values[posed];
    }
    const auto root = mode == uint32_t(CornerClassMode::Mixed) ? sectors.GetOr(ci, InvalidOffset) : InvalidOffset;
    if (root == InvalidOffset) {
        const auto v = vertices[ci];
        const auto posed = sources.PosedVertexNormals.Find(v);
        return posed == InvalidOffset ? sources.VertexNormals[v] : sources.PosedVertexNormals.Values[posed];
    }
    const auto posed = sources.PosedSectorNormals.Find(normal_sectors.Index(root));
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
        .CornerSectors = Buffers.CornerSectors.View(record.SectorBlockCount != 0u),
        .NormalSectors = Buffers.NormalSectors.View(),
        .CustomNormals = Buffers.CustomNormals.View(record.CornerAttributes & MeshAttributeBit_Normal),
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

namespace {
auto Handles(auto values, auto *tag) {
    return std::span{reinterpret_cast<decltype(tag)>(values.data()), values.size()};
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
        auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        const uint32_t first = Buffers.Vertices.First(record.Vertices);
        const auto run = Buffers.FaceCorners.Dense(record.FaceCorners);
        auto owners = Buffers.HalfedgeFaces.GetMutable(run), opposites = Buffers.OppositeHalfedges.GetMutable(run);
        const uint32_t wire_start = halfedge_count - 2u * uint32_t(wire_edges.size());
        for (uint32_t e = 0; e < wire_edges.size(); ++e) {
            const uint32_t h = wire_start + 2u * e;
            corners[h] = first + wire_edges[e][1];
            corners[h + 1u] = first + wire_edges[e][0];
            owners[h] = owners[h + 1u] = InvalidOffset;
            opposites[h] = run.Offset + h + 1u;
            opposites[h + 1u] = run.Offset + h;
        }
    }
    if (!face_starts || face_offsets.empty()) return;
    const auto faces = Buffers.FaceRanges.GetMutable(Buffers.FaceTriangles.Dense(record.FaceData));
    const uint32_t first_corner = Buffers.FaceCorners.First(record.FaceCorners);
    const uint32_t face_corners = halfedge_count - 2u * uint32_t(wire_edges.size());
    for (uint32_t f = 0; f < face_count; ++f) faces[f] = {first_corner + face_offsets[f], first_corner + (f + 1 < face_count ? face_offsets[f + 1] : face_corners)};
}

void MeshStore::FinishConnectivity(uint32_t id, uint32_t edge_count) {
    auto &record = WriteRecord(id);
    auto &edges = Buffers.EdgeHalfedges;
    ElementHandleRange allocated{.First = edges.First(record.EdgeData), .Count = edges.Count(record.EdgeData)};
    edges.Shrink(record.EdgeData, allocated, {}, edge_count);
}

MeshConnectivity MeshStore::GetConnectivity(uint32_t id) const {
    const auto &r = Records.at(id);
    return {
        .VertexBlocks = Buffers.Vertices.Membership(r.Vertices),
        .HalfedgeBlocks = Buffers.FaceCorners.Membership(r.FaceCorners),
        .EdgeBlocks = Buffers.EdgeHalfedges.Membership(r.EdgeData),
        .FaceBlocks = Buffers.FaceTriangles.Membership(r.FaceData),
        .VertexFirst = Buffers.Vertices.First(r.Vertices),
        .VertexCount = Buffers.Vertices.Count(r.Vertices),
        .HalfedgeFirst = Buffers.FaceCorners.First(r.FaceCorners),
        .HalfedgeCount = Buffers.FaceCorners.Count(r.FaceCorners),
        .EdgeFirst = Buffers.EdgeHalfedges.First(r.EdgeData),
        .FaceFirst = Buffers.FaceTriangles.First(r.FaceData),
        .OutgoingHalfedges = Handles(Buffers.OutgoingHalfedges.Buffer.GetSpan<uint32_t>(), (const he::HH *)nullptr),
        .Opposites = Handles(Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>(), (const he::HH *)nullptr),
        .HalfedgeToEdge = Handles(Buffers.HalfedgeEdges.Buffer.GetSpan<uint32_t>(), (const he::EH *)nullptr),
        .HalfedgeToFace = Handles(Buffers.HalfedgeFaces.Buffer.GetSpan<uint32_t>(), (const he::FH *)nullptr),
        .EdgeCount = Buffers.EdgeHalfedges.Count(r.EdgeData),
        .Edges = Handles(Buffers.EdgeHalfedges.Buffer.GetSpan<uint32_t>(), (const he::HH *)nullptr),
        .FaceCount = Buffers.FaceTriangles.Count(r.FaceData),
        .Faces = Handles(Buffers.FaceRanges.Buffer.GetSpan<uvec2>(), (const MeshConnectivity::Face *)nullptr),
        .VertexCorners = Buffers.VertexCorners.Buffer.GetSpan<uvec2>(),
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
        if (set && (arena.Set(set).Flags & 1u)) {
            blocks.resize(arena.Set(set).BlockCount);
            std::iota(blocks.begin(), blocks.end(), arena.Set(set).First);
        } else {
            for (auto b = set ? arena.Set(set).First : InvalidOffset; b != InvalidOffset; b = membership[b].Next) blocks.push_back(b);
            if (!std::ranges::is_sorted(blocks)) std::ranges::sort(blocks);
        }
        const auto count = uint32_t(blocks.size());
        RetireBlockListWords(entry.Words);
        entry.Words = BlockLists.Allocate(2u * count);
        auto words = BlockLists.GetMutable(entry.Words);
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
    if (header.Flags & 1u) return {values, header.First * MeshElementBlockSize, header.Count};
    const auto list = store.GetBlockList(id, domain);
    return {values, arena.Blocks.Buffer.template GetSpan<MeshElementBlock>(), list.Blocks, list.LivePrefix, header.Count};
}
} // namespace

uint32_t MeshStore::LiveElementAt(uint32_t id, ElementDomain domain, uint32_t ordinal) const {
    const auto set = DomainSet(Records.at(id), domain);
    return WithDomain(Buffers, domain, [&](const auto &arena) {
        const auto view = LiveView(*this, arena, set, id, domain);
        if (ordinal >= view.size()) throw std::out_of_range("Live element ordinal.");
        return view.Handle(ordinal);
    });
}

uint32_t MeshStore::LiveElementOrdinal(uint32_t id, ElementDomain domain, uint32_t handle) const {
    if (handle == InvalidOffset) return InvalidOffset;
    const auto set = DomainSet(Records.at(id), domain);
    return WithDomain(Buffers, domain, [&](const auto &arena) -> uint32_t {
        const auto &header = arena.Set(set);
        if (header.Flags & 1u) {
            const auto first = header.First * MeshElementBlockSize;
            return handle >= first && handle - first < header.Count ? handle - first : InvalidOffset;
        }
        const auto list = GetBlockList(id, domain);
        const auto block = handle / MeshElementBlockSize, offset = handle % MeshElementBlockSize;
        const auto it = std::ranges::lower_bound(list.Blocks, block);
        if (it == list.Blocks.end() || *it != block) return InvalidOffset;
        const auto &live = arena.Blocks.Get({block, 1u})[0].Live;
        if (!(live[offset / 32u] & (1u << (offset % 32u)))) return InvalidOffset;
        const auto at = uint32_t(it - list.Blocks.begin());
        auto rank = at ? list.LivePrefix[at - 1u] : 0u;
        for (uint32_t word = 0u; word < offset / 32u; ++word) rank += std::popcount(live[word]);
        return rank + std::popcount(live[offset / 32u] & ((1u << (offset % 32u)) - 1u));
    });
}

ConnectivityRef MeshStore::GetConnectivityRef(uint32_t id) const {
    const auto &r = Records.at(id);
    return {
        {Buffers.OutgoingHalfedges.Buffer.Slot, Buffers.Vertices.First(r.Vertices)},
        {Buffers.OppositeHalfedges.Buffer.Slot, Buffers.FaceCorners.First(r.FaceCorners)},
        {Buffers.HalfedgeEdges.Buffer.Slot, Buffers.FaceCorners.First(r.FaceCorners)},
        {Buffers.HalfedgeFaces.Buffer.Slot, Buffers.FaceCorners.First(r.FaceCorners)},
        {Buffers.FaceRanges.Buffer.Slot, Buffers.FaceTriangles.First(r.FaceData)},
        {Buffers.EdgeHalfedges.Buffer.Slot, Buffers.EdgeHalfedges.First(r.EdgeData)},
        {Buffers.VertexCorners.Buffer.Slot, Buffers.Vertices.First(r.Vertices)},
        Buffers.VertexFans.Items.Buffer.Slot,
    };
}

TriangleCorners MeshStore::GetTriangleCorners(uint32_t id) const { return {TriangleView(id)}; }

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
    if (data.FaceCount() > 0) {
        const auto &record = Get(id);
        const auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        const uint32_t first_vertex = Buffers.Vertices.First(record.Vertices);
        for (size_t h = 0; h < data.FaceCorners.size(); ++h) corners[h] = first_vertex + data.FaceCorners[h];
    }
    return id;
}

void MeshStore::CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &deform, const std::optional<MorphTargetData> &morph) {
    auto &record = WriteRecord(id);
    const auto vertices = Buffers.Vertices.Dense(record.Vertices);
    const uint32_t vertex_count = vertices.Count;
    if (vertex_count == 0) return;
    if (deform) {
        std::vector<BoneDeformVertex> skin(vertex_count);
        for (uint32_t i = 0; i < vertex_count; ++i) skin[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        Buffers.Skin.Initialize(vertices, skin);
        record.SkinBlocksReady = true;
    }
    if (morph && morph->TargetCount > 0) {
        record.MorphTargetCount = morph->TargetCount;
        const bool has_normal_deltas = !morph->NormalDeltas.empty();
        std::vector<MorphTargetVertex> targets(uint64_t(vertex_count) * record.MorphTargetCount);
        for (size_t i = 0; i < targets.size(); ++i)
            targets[i] = {.PositionDelta = morph->PositionDeltas[i], .NormalDelta = has_normal_deltas ? morph->NormalDeltas[i] : vec3{0}};
        Buffers.Morph.Initialize(vertices, targets, record.MorphTargetCount);
        record.MorphBlocksReady = true;
        record.DefaultMorphWeights = morph->DefaultWeights;
        record.DefaultMorphWeights.resize(record.MorphTargetCount, 0.f);
    }
}

void MeshStore::ShrinkMeshSource(uint32_t id, uint32_t welded_vertices) {
    auto &record = WriteRecord(id);
    const auto vertices = Buffers.Vertices.Dense(record.Vertices);
    const uint32_t first_block = vertices.Offset / MeshElementBlockSize;
    const uint32_t old_blocks = (vertices.Count + MeshElementBlockSize - 1u) / MeshElementBlockSize;
    const uint32_t kept_blocks = (welded_vertices + MeshElementBlockSize - 1u) / MeshElementBlockSize;
    for (uint32_t b = first_block + kept_blocks; b < first_block + old_blocks; ++b) {
        if (record.SkinBlocksReady) Buffers.Skin.Release(b);
        if (record.MorphBlocksReady) Buffers.Morph.Release(b);
    }
    ElementHandleRange allocated{.First = vertices.Offset, .Count = vertices.Count};
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
    const auto corner_blocks = ElementArena<uint32_t>::BlockCount(halfedges);
    if (faces) {
        if (attrs.Tangents) Buffers.CornerTangents.Values.PlanAdditional(corner_blocks);
        if (attrs.Colors0) Buffers.CornerColors.Values.PlanAdditional(corner_blocks);
        const std::array uvs{&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3};
        for (uint32_t set = 0; set < uvs.size(); ++set)
            if (*uvs[set]) Buffers.CornerUvs[set].Values.PlanAdditional(corner_blocks);
    } else if (attrs.Colors0) Buffers.VertexColors.Values.PlanAdditional(ElementArena<uint32_t>::BlockCount(vertices));
    const auto vertex_blocks = ElementArena<uint32_t>::BlockCount(vertices);
    if (has_deform) {
        Buffers.Skin.Values.PlanAdditional(vertex_blocks);
        Buffers.Skin.Owners.PlanAdditional(vertex_blocks);
    }
    if (morph_target_count > 0) {
        Buffers.Morph.Values.PlanAdditional(vertex_blocks * morph_target_count);
        Buffers.Morph.Owners.PlanAdditional(vertex_blocks * morph_target_count);
    }
}

void MeshStore::PlanClone(const Mesh &mesh) {
    const auto &record = Records.at(mesh.GetStoreId());
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        const auto *allocation = ranges(record);
        if (!allocation || info.Mirror) return;
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*allocation)>, Range>) arena.PlanAdditional(allocation->Count);
        else if (*allocation) WithDomain(Buffers, info.Elements, [&](const auto &owner) {
            arena.PlanAdditional(owner.Set(*allocation).BlockCount * MeshElementBlockSize);
        });
    });
}

void MeshStore::CommitReserves() {
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.CommitPlanned(); });
}

void MeshStore::CreateMesh(uint32_t id, const MeshData &data, const MeshVertexAttributes &attrs, const MeshPrimitives &primitives, const CornerLayers &layers, bool has_authored_normals) {
    const profile::CpuScope scope{"CreateMesh"};
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
        const auto range = face_count ? Buffers.FaceTriangles.Dense(record.FaceData) : Buffers.Vertices.Dense(record.Vertices);
        attributes.Initialize(range, primitives.ElementPrimitiveIndices);

        record.PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(primitive_count);
        auto pm_span = Buffers.PrimitiveMaterials.GetMutable(record.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        else std::ranges::fill(pm_span, 0u);
    };

    if (face_count > 0) {
        SyncMirrors();
        auto first_tri_span = Buffers.FaceTriangles.GetMutable(record.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.FaceSize(fi) - 2u;
        }
        record.TriangleCount = tri_offset;

        const auto corners = Buffers.FaceCorners.Dense(record.FaceCorners);
        if (!layers.Tangents.empty()) {
            record.CornerAttributes |= MeshAttributeBit_Tangent;
            Buffers.CornerTangents.Initialize(corners, layers.Tangents);
        }
        if (!layers.Colors.empty()) {
            record.CornerAttributes |= MeshAttributeBit_Color0;
            Buffers.CornerColors.Initialize(corners, layers.Colors);
        }
        for (uint32_t set = 0; set < layers.Uvs.size(); ++set) {
            if (layers.Uvs[set].empty()) continue;
            record.CornerAttributes |= MeshAttributeBit_TexCoord0 << set;
            Buffers.CornerUvs[set].Initialize(corners, layers.Uvs[set]);
        }

        record.TriangleData = Buffers.Triangles.Allocate(tri_offset);
        const uint32_t first_triangle = Buffers.Triangles.First(record.TriangleData);
        for (auto &first : first_tri_span) first += first_triangle;
        auto triangle_span = Buffers.Triangles.GetMutable(record.TriangleData);
        const auto corner_vertices = Buffers.FaceCorners.Buffer.GetSpan<uint32_t>();
        const auto vertices = Buffers.Vertices.Buffer.GetSpan<Vertex>();
        std::vector<vec2> projected;
        std::vector<uint32_t> next, previous;
        uint32_t ti = 0u;
        for (uint32_t fi = 0u; fi < face_count; ++fi) {
            const uint32_t n = data.FaceSize(fi), first = corners.Offset + data.FaceStart(fi);
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
        Buffers.VertexColors.Initialize(Buffers.Vertices.Dense(record.Vertices), *attrs.Colors0);
    }

    // The sharpness stores start smooth.
    const Mesh mesh{*this, id};
    std::ranges::fill(Buffers.FaceSharpness.GetMutable(Buffers.FaceTriangles.Dense(record.FaceData)), uint8_t{0});
    std::ranges::fill(Buffers.EdgeSharpness.GetMutable(Buffers.EdgeHalfedges.Dense(record.EdgeData)), uint8_t{0});
}

uint32_t MeshStore::BeginTopologyOutput(uint32_t source, uint32_t second, std::span<const uint32_t> materials) {
    const auto &src = Records.at(source);
    const auto *other = second != InvalidOffset ? &Records.at(second) : nullptr;
    Record record{
        .CornerAttributes = src.CornerAttributes | (other ? other->CornerAttributes : 0u),
        .VertexAttributes = src.VertexAttributes | (other ? other->VertexAttributes : 0u),
        .PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(uint32_t(materials.size())),
        .FacePrimitivesReady = true,
        .SelectionSummary = Buffers.SelectionSummary.Allocate(1),
        .ConnectivityFaceStarts = true,
        .SkinBlocksReady = src.SkinBlocksReady,
        .MorphBlocksReady = src.MorphBlocksReady,
        .MorphTargetCount = src.MorphTargetCount,
        .HasAuthoredNormals = src.HasAuthoredNormals || (other && other->HasAuthoredNormals),
        .DefaultMorphWeights = src.DefaultMorphWeights,
        .Alive = true,
    };
    std::ranges::copy(materials, Buffers.PrimitiveMaterials.GetMutable(record.PrimitiveMaterials).begin());
    Buffers.SelectionSummary.GetMutable(record.SelectionSummary)[0] = {};
    std::vector<NamedAttribute> attributes;
    const auto add = [&](uint32_t, const NamedAttribute &entry) { attributes.push_back({.Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default}); };
    ForEachNamedAttribute(source, add);
    if (other) ForEachNamedAttribute(second, [&](uint32_t attribute, const NamedAttribute &entry) { if (FindAttribute(source, entry.NameView()) == InvalidOffset) add(attribute, entry); });
    const auto id = AcquireId(std::move(record));
    CaptureSelectionSummary(id);
    // The output's attributes take payloads as its elements insert.
    for (auto &attribute : attributes) {
        attribute.Record = id;
        AcquireAttribute(std::move(attribute));
    }
    return id;
}

void MeshStore::CopyBlocks(CloneCopies &copies, uint32_t destination, uint32_t source, const std::array<Range, 6> &maps, const DomainBlocks &sources, std::span<const uvec2> attributes) {
    const auto &src = Records[source];
    const auto &dst = Records[destination];
    const auto map = [&](Domain domain) { return maps[uint32_t(domain)]; };
    const auto target = [&](Domain domain, uint32_t block) { return copies.MapHandle(map(domain), block * MeshElementBlockSize) / MeshElementBlockSize; };
    const auto targets = [&](Domain domain) {
        std::vector<uint32_t> blocks;
        blocks.reserve(sources[uint32_t(domain)].size());
        for (const auto block : sources[uint32_t(domain)]) blocks.push_back(target(domain, block));
        return blocks;
    };
    // Element values and their mirrors, apart from the vertex fan roots, which take new fan runs below.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        using Allocation = std::remove_cvref_t<decltype(*ranges(src))>;
        if constexpr (std::is_same_v<Allocation, ElementSetRef>) {
            if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
            constexpr auto Bytes = uint64_t(MeshElementBlockSize) * sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
            for (const auto block : sources[uint32_t(info.Elements)]) CaptureCopy(copies, arena.Buffer, Bytes, block, target(info.Elements, block));
        }
    });
    for (const auto domain : SelectionDomains)
        for (const auto block : sources[uint32_t(domain)]) {
            const auto to = target(domain, block);
            CaptureCopy(copies, SelectionArena(Buffers, domain).Buffer, sizeof(SelectionBlock), block, to);
            CaptureCopy(copies, HiddenArena(Buffers, domain).Buffer, sizeof(SelectionBlock), block, to);
            CaptureCopy(copies, AggregateArena(Buffers, domain).Buffer, sizeof(SelectionAggregate), block, to);
        }
    // A layer the destination carries holds a payload at each of its blocks.
    ForEachAttribute(Buffers, [&](auto &attribute, Domain domain, uint32_t, const char *, const char *, const char *, auto &&entries) {
        const auto &from = sources[uint32_t(domain)];
        if (from.empty()) return;
        const auto to = targets(domain);
        const auto count = entries(dst);
        if (!count) return attribute.Release(to);
        attribute.Attach(to, attribute.Absent, count);
        constexpr auto Bytes = uint64_t(MeshElementBlockSize) * sizeof(typename std::remove_cvref_t<decltype(attribute)>::Block::value_type);
        for (uint32_t i = 0u; i < from.size(); ++i)
            if (attribute.PayloadBlock(from[i]))
                for (uint32_t e = 0u; e < std::min(count, entries(src)); ++e) CaptureCopy(copies, attribute.Values.Buffer, Bytes, attribute.PayloadBlock(from[i]) - 1u + e, attribute.PayloadBlock(to[i]) - 1u + e);
    });
    // Sector roots are sparse corner payloads.
    std::vector<uint32_t> sectors;
    if (const auto &from = sources[uint32_t(Domain::Halfedge)]; !from.empty()) {
        const auto to = targets(Domain::Halfedge);
        std::vector<uint32_t> attach, release;
        for (uint32_t i = 0u; i < from.size(); ++i) {
            const bool has = Buffers.CornerSectors.PayloadBlock(from[i]), had = Buffers.CornerSectors.PayloadBlock(to[i]);
            if (has && !had) attach.push_back(to[i]);
            else if (!has && had) release.push_back(to[i]);
        }
        Buffers.CornerSectors.Attach(attach);
        Buffers.NormalSectors.Attach(attach);
        Buffers.CornerSectors.Release(release);
        Buffers.NormalSectors.Release(release);
        if (!attach.empty() || !release.empty()) WriteRecord(destination).SectorBlockCount += uint32_t(attach.size()) - uint32_t(release.size());
        constexpr auto CornerBytes = sizeof(decltype(Buffers.CornerSectors)::Block), NormalBytes = sizeof(decltype(Buffers.NormalSectors)::Block);
        for (uint32_t i = 0u; i < from.size(); ++i) {
            if (!Buffers.CornerSectors.PayloadBlock(from[i])) continue;
            CaptureCopy(copies, Buffers.CornerSectors.Values.Buffer, CornerBytes, Buffers.CornerSectors.PayloadBlock(from[i]) - 1u, Buffers.CornerSectors.PayloadBlock(to[i]) - 1u);
            CaptureCopy(copies, Buffers.NormalSectors.Values.Buffer, NormalBytes, Buffers.NormalSectors.PayloadBlock(from[i]) - 1u, Buffers.NormalSectors.PayloadBlock(to[i]) - 1u);
            sectors.push_back(to[i]);
        }
    }
    for (const auto [from, to] : attributes) CopyAttributeBlocks(copies, from, to, map(ElementDomainOf(NamedAttributes[from].Domain)), sources[uint32_t(ElementDomainOf(NamedAttributes[from].Domain))]);
    // Each copied vertex takes a new fan run, and the destination's runs at the copied blocks release.
    if (const auto &from = sources[uint32_t(Domain::Vertex)]; !from.empty()) {
        const auto to = targets(Domain::Vertex);
        const auto roots = Buffers.VertexCorners.Buffer.GetSpan<uvec2>();
        std::vector<uvec2> released;
        uint64_t fan_count = 0u;
        for (uint32_t i = 0u; i < from.size(); ++i) {
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot)
                if (const auto root = roots[to[i] * MeshElementBlockSize + slot]; root.y) released.push_back(root);
            Buffers.Vertices.ForEachLive(from[i], [&](uint32_t v) { fan_count += roots[v].y; });
        }
        Buffers.VertexFans.Release(released);
        if (fan_count >= InvalidOffset) throw std::length_error("Copied fan items exceed their address space.");
        const auto fans = Buffers.VertexFans.Items.Allocate(uint32_t(fan_count));
        Buffers.VertexFans.Items.CaptureWrite(fans);
        uint32_t next = fans.Offset;
        for (uint32_t i = 0u; i < from.size(); ++i) {
            auto written = Buffers.VertexCorners.GetMutable({to[i] * MeshElementBlockSize, MeshElementBlockSize});
            std::ranges::fill(written, uvec2{InvalidOffset, 0u});
            Buffers.Vertices.ForEachLive(from[i], [&](uint32_t v) {
                const auto root = roots[v];
                if (!root.y) return;
                written[v % MeshElementBlockSize] = {next, root.y};
                copies.Copy(Buffers.VertexFans.Items.Buffer, uint64_t(root.x) * sizeof(uvec2), uint64_t(next) * sizeof(uvec2), uint64_t(root.y) * sizeof(uvec2));
                next += root.y;
            });
        }
        for (const auto [domain, word] : {std::pair{Domain::Halfedge, 0u}, std::pair{Domain::Face, 1u}})
            copies.RebaseByBlock(Buffers.VertexFans.Items.Buffer, uint64_t(fans.Offset) * sizeof(uvec2) + word * 4u, fans.Count, map(domain), 2u);
    }
    const auto rebase = [&](auto &arena, Domain values, Domain references, uint32_t words = 1u, bool span = false) {
        for (const auto block : targets(values)) {
            const auto offset = uint64_t(block) * MeshElementBlockSize * words * 4u;
            copies.RebaseByBlock(arena.Buffer, offset, MeshElementBlockSize * (span ? 1u : words), map(references), span ? words : 1u, 0u, 0u, span);
        }
    };
    rebase(Buffers.FaceCorners, Domain::Halfedge, Domain::Vertex);
    rebase(Buffers.OutgoingHalfedges, Domain::Vertex, Domain::Halfedge);
    rebase(Buffers.OppositeHalfedges, Domain::Halfedge, Domain::Halfedge);
    rebase(Buffers.HalfedgeEdges, Domain::Halfedge, Domain::Edge);
    rebase(Buffers.HalfedgeFaces, Domain::Halfedge, Domain::Face);
    rebase(Buffers.FaceTriangles, Domain::Face, Domain::Triangle);
    rebase(Buffers.Triangles, Domain::Triangle, Domain::Halfedge, 3u);
    rebase(Buffers.EdgeHalfedges, Domain::Edge, Domain::Halfedge);
    rebase(Buffers.FaceRanges, Domain::Face, Domain::Halfedge, 2u, true);
    for (const auto block : sectors) {
        const auto payload = Buffers.CornerSectors.Payload(block * MeshElementBlockSize, MeshElementBlockSize);
        copies.RebaseByBlock(Buffers.CornerSectors.Values.Buffer, uint64_t(payload.Offset) * 4u, payload.Count, map(Domain::Halfedge));
    }
    DerivedRecords[destination].NormalRevision = ++NextNormalRevision;
}

namespace {
// A domain's index among a Change's blocks, or InvalidOffset for a domain it does not list.
constexpr uint32_t ChangeIndex(Domain domain) {
    return domain == Domain::Halfedge ? 3u : SelectableDomain(domain) ? SelectableIndex(domain) :
                                                                        InvalidOffset;
}
// Maps the active handle of the copy's selection summary, copied from the source's, onto the copy's handles.
void RebaseActiveHandle(CloneCopies &copies, MeshArenas &arenas, const MeshStore::Record &source, const MeshStore::Record &copy, const std::array<Range, 6> &maps) {
    if (!source.SelectionSummary.Count) return;
    const auto &summary = arenas.SelectionSummary.Get(source.SelectionSummary)[0];
    if (summary.ActiveHandle == InvalidOffset) return;
    const auto domain = SelectionDomain(summary.Mode);
    const auto [from, to] = WithDomain(arenas, domain, [&](const auto &arena) { return std::pair{arena.First(DomainSet(source, domain)), arena.First(DomainSet(copy, domain))}; });
    copies.RebaseByBlock(arenas.SelectionSummary.Buffer, uint64_t(copy.SelectionSummary.Offset) * sizeof(EditSelectionSummary) + offsetof(EditSelectionSummary, ActiveHandle), 1u, maps[uint32_t(domain)], 1u, from, to);
}
} // namespace

std::vector<uint32_t> MeshStore::CloneMeshes(CloneCopies &copies, std::span<const uint32_t> source_ids, bool render, std::span<BlockPairs> pairs) {
    const profile::CpuScope scope{"CloneMeshes"};
    std::vector<uint32_t> ids;
    std::vector<std::array<Range, 6>> maps(source_ids.size());
    ids.reserve(source_ids.size());
    // Allocate the whole batch before encoding copies: arena growth may move either side.
    for (const auto src_id : source_ids) {
        Record copy{Records.at(src_id)};
        ResetRender(copy);
        const auto id = AcquireId(std::move(copy));
        DerivedRecords[id] = DerivedRecords.at(src_id);
        DerivedRecords[id].NormalRevision = ++NextNormalRevision;
        const auto &src = Records[src_id];
        auto &dst = Records[id];
        dst.SectorBlockCount = 0u;
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
            const auto *from = ranges(src);
            if (!from || info.Mirror) return;
            auto *to = ranges(dst);
            if constexpr (requires { arena.CloneMembership(*from); }) *to = arena.CloneMembership(*from);
            else if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*from)>, Range>) *to = arena.Allocate(from->Count);
        });
        ids.push_back(id);
    }
    SyncMirrors();
    for (uint32_t i = 0u; i < ids.size(); ++i) {
        const auto &src = Records[source_ids[i]];
        DomainBlocks sources;
        for (const auto domain : ElementDomains) {
            std::vector<uvec2> blocks;
            WithDomain(Buffers, domain, [&](const auto &arena) {
                auto next = arena.First(DomainSet(Records[ids[i]], domain)) / MeshElementBlockSize;
                arena.ForEachBlock(DomainSet(src, domain), [&](uint32_t block, const auto &) { blocks.push_back({block, next++}); });
            });
            maps[i][uint32_t(domain)] = copies.MapBlocks(blocks);
            for (const auto block : blocks) sources[uint32_t(domain)].push_back(block.x);
            if (!pairs.empty()) pairs[i][uint32_t(domain)] = std::move(blocks);
        }
        // The record's own ranges copy whole.
        ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &, auto &&ranges) {
            const auto *from = ranges(src);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*from)>, Range>) {
                if (from && from->Count) CaptureCopy(copies, arena.Buffer, sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>), from->Offset, ranges(Records[ids[i]])->Offset, from->Count);
            }
        });
        std::vector<uvec2> attributes;
        ForEachNamedAttribute(source_ids[i], [&](uint32_t attribute, const NamedAttribute &) { attributes.push_back({attribute, 0u}); });
        for (auto &attribute : attributes) {
            const auto entry = NamedAttributes[attribute.x];
            attribute.y = AcquireAttribute({.Record = ids[i], .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default});
        }
        CopyBlocks(copies, ids[i], source_ids[i], maps[i], sources, attributes);
        const auto &dst = Records[ids[i]];
        auto &derived = DerivedRecords[ids[i]];
        const auto mapped = [&](Domain domain, uint32_t h) { return copies.MapHandle(maps[i][uint32_t(domain)], h); };
        for (auto &[block, words] : derived.SelectionBaseline) block = mapped(SelectionDomain(derived.SelectionBaselineElement), block * MeshElementBlockSize) / MeshElementBlockSize;
        std::ranges::sort(derived.SelectionBaseline, {}, &std::pair<uint32_t, SelectionBlock>::first);
        if (derived.SelectionBaselineActive != InvalidOffset) {
            const auto domain = SelectionDomain(derived.SelectionBaselineElement);
            const auto [from, to] = WithDomain(Buffers, domain, [&](const auto &owner) { return std::pair{owner.First(DomainSet(src, domain)), owner.First(DomainSet(dst, domain))}; });
            derived.SelectionBaselineActive = mapped(domain, from + derived.SelectionBaselineActive) - to;
        }
        RebaseActiveHandle(copies, Buffers, src, dst, maps[i]);
    }
    if (render) CloneRenderRecords(copies, source_ids, ids, maps);
    return ids;
}

void MeshStore::ReleaseDomain(uint32_t id, ElementDomain domain) {
    WithDomain(Buffers, domain, [&](auto &arena) {
        auto &set = DomainSet(WriteRecord(id), domain);
        std::vector<uint32_t> blocks;
        arena.ForEachBlock(set, [&](uint32_t block, const auto &) { blocks.push_back(block); });
        std::ranges::sort(blocks);
        if (domain == Domain::Halfedge) {
            Buffers.CornerSectors.Release(blocks);
            Buffers.NormalSectors.Release(blocks);
            WriteRecord(id).SectorBlockCount = 0u;
        }
        ForEachAttribute(Buffers, [&](auto &attributes, Domain owner, uint32_t, const char *, const char *, const char *, auto &&) {
            if (owner == domain) attributes.Release(blocks);
        });
        arena.Destroy(std::span{&set, 1u}, blocks);
        set = {};
    });
}

void MeshStore::DetachBlocks(uint32_t id, ElementDomain domain, std::span<const uint32_t> blocks) {
    if (blocks.empty()) return;
    const auto cleared = WithDomain(Buffers, domain, [&](auto &arena) { return arena.Clear(DomainSet(Records[id], domain), blocks); });
    if (SelectableDomain(domain)) EditSelectionBlocks(SelectionElements[SelectableIndex(domain)], cleared, [](uint32_t, auto &words) { words = {}; });
    FinishEraseElements(id, domain, cleared);
}

std::optional<MeshStore::Change> MeshStore::Resync(CloneCopies &copies, uint32_t id, uint32_t source, BlockPairs &pairs, const DomainBlocks &written, std::span<const uint32_t> written_attributes, std::vector<uint32_t> &attributes) {
    const profile::CpuScope scope{"ResyncMesh"};
    std::array<Range, 6> maps{};
    // The source keeps every paired block, and each block of its own it wrote is paired.
    for (const auto domain : ElementDomains) {
        const auto d = uint32_t(domain);
        const auto from = DomainSet(Records.at(source), domain), to = DomainSet(Records.at(id), domain);
        const bool shaped = WithDomain(Buffers, domain, [&](const auto &arena) {
            if (bool(from) != bool(to) || (from ? arena.Set(from).BlockCount : 0u) != pairs[d].size()) return false;
            maps[d] = copies.MapBlocks(pairs[d]);
            const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
            return std::ranges::all_of(written[d], [&](uint32_t block) {
                return !from || block >= members.size() || members[block].Owner != from.Index || copies.MapHandle(maps[d], block * MeshElementBlockSize) != InvalidOffset;
            });
        });
        if (!shaped) return {};
    }
    ReleaseRender(std::array{&WriteRecord(id)});
    Change change{.StoreId = id, .Bits = GeometryChanged | TopologyChanged};
    // Each written pair restores, each written block only the copy holds leaves it, and a paired block the copy lost takes a new block.
    struct Plan {
        std::vector<uint32_t> Touched, Detached, Lost;
    };
    std::array<Plan, 6> plans;
    for (const auto domain : ElementDomains) {
        const auto d = uint32_t(domain);
        const auto &paired = pairs[d];
        if (paired.empty()) continue;
        auto &plan = plans[d];
        const auto from = DomainSet(Records[source], domain), to = DomainSet(Records[id], domain);
        const auto pair_of = [&](uint32_t block) {
            const auto it = std::ranges::lower_bound(paired, block, {}, &uvec2::y);
            return it != paired.end() && it->y == block ? uint32_t(it - paired.begin()) : InvalidOffset;
        };
        WithDomain(Buffers, domain, [&](const auto &arena) {
            const auto owner = [&](uint32_t block) { return block < arena.Blocks.Buffer.template Count<MeshElementBlock>() ? arena.Blocks.Get({block, 1u})[0].Owner : InvalidOffset; };
            for (const auto block : written[d]) {
                if (owner(block) == from.Index) plan.Touched.push_back(pair_of(copies.MapHandle(maps[d], block * MeshElementBlockSize) / MeshElementBlockSize));
                else if (const auto i = pair_of(block); i != InvalidOffset) plan.Touched.push_back(i);
                else if (owner(block) == to.Index) plan.Detached.push_back(block);
            }
            SortUnique(plan.Touched);
            for (const auto i : plan.Touched)
                if (owner(paired[i].y) != to.Index) plan.Lost.push_back(i);
        });
    }
    // A face runs through adjacent corner and triangle blocks, so one of those domains that lost a block takes fresh blocks laid out as the source's, and every pair then restores to the moved handles.
    std::array<bool, 6> relaid{};
    for (const auto domain : {Domain::Halfedge, Domain::Triangle}) relaid[uint32_t(domain)] = !plans[uint32_t(domain)].Lost.empty();
    const bool moved = std::ranges::any_of(relaid, std::identity{});
    DomainBlocks restored;
    for (const auto domain : ElementDomains) {
        const auto d = uint32_t(domain);
        auto &paired = pairs[d];
        if (paired.empty()) continue;
        auto &[touched, detached, lost] = plans[d];
        const auto from = DomainSet(Records[source], domain);
        WithDomain(Buffers, domain, [&](auto &arena) {
            if (relaid[d]) {
                ReleaseDomain(id, domain);
                std::ranges::sort(paired, {}, &uvec2::x);
                const auto fresh = DomainSet(WriteRecord(id), domain) = arena.Create(uint32_t(paired.size()) * MeshElementBlockSize);
                for (uint32_t k = 0u; k < paired.size(); ++k) paired[k].y = arena.Set(fresh).First + k;
            } else {
                DetachBlocks(id, domain, detached);
                const auto added = arena.Extend(DomainSet(Records[id], domain), uint32_t(lost.size()));
                for (uint32_t k = 0u; k < lost.size(); ++k) paired[lost[k]].y = added[k];
            }
            if (moved) {
                touched.resize(paired.size());
                std::iota(touched.begin(), touched.end(), 0u);
            }
            if (!lost.empty()) {
                std::ranges::sort(paired, {}, &uvec2::y);
                maps[d] = copies.MapBlocks(paired);
            }
            if (touched.empty() && detached.empty()) return;
            std::vector<uvec2> live;
            live.reserve(touched.size());
            for (const auto i : touched) live.push_back(paired[i]);
            const auto to = DomainSet(Records[id], domain);
            // The copy is dense again when its blocks pair with the dense source's in order from its first block.
            const auto &header = arena.Set(from);
            const bool dense = (header.Flags & 1u) && paired.front().y == arena.Set(to).First &&
                std::ranges::all_of(std::views::iota(size_t{0}, paired.size()), [&](size_t k) { return paired[k].x == paired[0].x + k && paired[k].y == paired[0].y + k; });
            arena.AssignLive(to, live, dense ? header.Flags : 0u);
            for (const auto pair : live) restored[d].push_back(pair.x);
            if (const auto index = ChangeIndex(domain); index != InvalidOffset) {
                auto &blocks = change.Blocks[index];
                for (const auto pair : live) blocks.push_back(pair.y);
                blocks.append_range(detached);
                SortUnique(blocks);
            }
        });
        std::ranges::sort(restored[d]);
    }
    SyncMirrors();
    // The record takes the source's fields over its own sets, ranges and sector count.
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
        record.SectorBlockCount = own.SectorBlockCount;
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
        RebaseActiveHandle(copies, Buffers, src, record, maps);
        DerivedRecords[id] = {};
    }
    // An attribute keeps its payloads when it still holds its source attribute's copy, and the others copy anew.
    std::vector<uvec2> kept, whole;
    std::vector<uint32_t> released;
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &entry) {
        const bool intact = std::ranges::binary_search(attributes, attribute) && !std::ranges::binary_search(written_attributes, attribute) && !relaid[uint32_t(ElementDomainOf(entry.Domain))];
        const auto from = intact ? FindAttribute(source, entry.NameView()) : InvalidOffset;
        if (from != InvalidOffset && NamedAttributes[from].Type == entry.Type && NamedAttributes[from].Domain == entry.Domain) kept.push_back({from, attribute});
        else released.push_back(attribute);
    });
    for (const auto attribute : released) ReleaseAttribute(attribute);
    ForEachNamedAttribute(source, [&](uint32_t attribute, const NamedAttribute &) {
        if (std::ranges::find(kept, attribute, &uvec2::x) == kept.end()) whole.push_back({attribute, 0u});
    });
    for (auto &pair : whole) {
        const auto entry = NamedAttributes[pair.x];
        pair.y = AcquireAttribute({.Record = id, .Domain = entry.Domain, .Type = entry.Type, .Name = entry.Name, .Default = entry.Default});
    }
    attributes.clear();
    for (const auto pair : kept) attributes.push_back(pair.y);
    for (const auto pair : whole) attributes.push_back(pair.y);
    std::ranges::sort(attributes);
    CopyBlocks(copies, id, source, maps, restored, kept);
    for (const auto [from, to] : whole) {
        const auto domain = ElementDomainOf(NamedAttributes[from].Domain);
        std::vector<uint32_t> blocks;
        for (const auto pair : pairs[uint32_t(domain)]) blocks.push_back(pair.x);
        CopyAttributeBlocks(copies, from, to, maps[uint32_t(domain)], blocks);
    }
    return change;
}

void MeshStore::ResetEditState(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const BlockPairs &pairs) {
    SelectionUpdate update{.StoreId = id};
    for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
        const auto &paired = pairs[uint32_t(SelectionDomains[s])];
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

void MeshStore::SelectMasks(state::Scene &r, mtl::ComputeChain &chain, uint32_t id, const GeometryMasks &masks) {
    SelectionUpdate update{.StoreId = id};
    for (uint32_t s = 0u; s < SelectionDomains.size(); ++s) {
        const auto from = masks.Get(SelectionElements[s]);
        const auto selection = SelectionArena(Buffers, SelectionDomains[s]).Buffer.GetSpan<SelectionBlock>();
        const auto mask = [&](uint32_t block) { return from.Arena->Get({from.First + block, 1u})[0]; };
        for (const auto block : GetBlockList(id, SelectionDomains[s]).Blocks)
            if (selection[block] != mask(block)) update.Blocks[s].push_back(block);
        EditSelectionBlocks(SelectionElements[s], update.Blocks[s], [&](uint32_t block, auto &words) { words = mask(block); });
    }
    if (Records.at(id).SelectionSummary.Count) UpdateSelection(r, chain, std::array{update});
}

void MeshStore::Join(state::Scene &r, uint32_t id, uint32_t source) {
    const profile::CpuScope scope{"JoinMeshes"};
    // Layers either record carries cover the join, and the destination's blocks take defaults for the source's.
    {
        std::vector<uint32_t> before;
        ForEachAttribute(Buffers, [&](auto &, Domain, uint32_t, const char *, const char *, const char *, auto &&entries) { before.push_back(entries(Records[id])); });
        auto &record = WriteRecord(id);
        const auto &src = Records.at(source);
        record.CornerAttributes |= src.CornerAttributes;
        record.VertexAttributes |= src.VertexAttributes;
        record.FacePrimitivesReady |= src.FacePrimitivesReady;
        record.VertexPrimitivesReady |= src.VertexPrimitivesReady;
        record.ConnectivityFaceStarts |= src.ConnectivityFaceStarts;
        record.HasAuthoredNormals |= src.HasAuthoredNormals;
        uint32_t i = 0u;
        ForEachAttribute(Buffers, [&](auto &attribute, Domain domain, uint32_t, const char *, const char *, const char *, auto &&entries) {
            if (before[i++] || !entries(record)) return;
            std::vector<uint32_t> blocks;
            WithDomain(Buffers, domain, [&](const auto &arena) { arena.ForEachBlock(DomainSet(record, domain), [&](uint32_t block, const auto &) { blocks.push_back(block); }); });
            attribute.Attach(blocks, attribute.Absent, entries(record));
        });
    }
    CloneCopies copies;
    std::array<Range, 6> maps{};
    DomainBlocks sources, targets;
    for (const auto domain : ElementDomains) WithDomain(Buffers, domain, [&](auto &arena) {
        const auto d = uint32_t(domain);
        arena.ForEachBlock(DomainSet(Records[source], domain), [&](uint32_t block, const auto &) { sources[d].push_back(block); });
        if (sources[d].empty()) return;
        auto &set = DomainSet(WriteRecord(id), domain);
        targets[d] = arena.Extend(set, uint32_t(sources[d].size()));
        std::vector<uvec2> pairs;
        pairs.reserve(sources[d].size());
        for (uint32_t i = 0u; i < sources[d].size(); ++i) pairs.push_back({sources[d][i], targets[d][i]});
        arena.AssignLive(set, pairs, 0u);
        maps[d] = copies.MapBlocks(pairs);
        std::ranges::sort(targets[d]);
    });
    SyncMirrors();
    // Attributes of one name, type and domain join, and an attribute of either side holds its default on the other's elements.
    std::vector<uvec2> joined;
    std::vector<uint32_t> own;
    ForEachNamedAttribute(id, [&](uint32_t attribute, const NamedAttribute &) { own.push_back(attribute); });
    for (const auto attribute : own) {
        const auto &entry = NamedAttributes[attribute];
        const auto from = FindAttribute(source, entry.NameView());
        if (from != InvalidOffset && NamedAttributes[from].Type == entry.Type && NamedAttributes[from].Domain == entry.Domain) joined.push_back({from, attribute});
        else AttachAttributeBlocks(attribute, targets[uint32_t(ElementDomainOf(entry.Domain))]);
    }
    std::vector<uint32_t> theirs;
    ForEachNamedAttribute(source, [&](uint32_t attribute, const NamedAttribute &) { theirs.push_back(attribute); });
    for (const auto attribute : theirs) {
        const auto entry = NamedAttributes[attribute];
        if (FindAttribute(id, entry.NameView()) != InvalidOffset) continue;
        joined.push_back({attribute, AddAttribute(id, entry.Domain, entry.Type, entry.NameView(), entry.Default).value()});
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
    CopyBlocks(copies, id, source, maps, sources, joined);
    WriteRecord(id).TriangleCount = Buffers.Triangles.Count(Records[id].TriangleData);
    mtl::ComputeChain chain{BufferContext()};
    copies.Encode(chain, GetMeshPipelines(r));
    chain.Submit();
    if (tabled)
        for (const auto [layer, domain] : {std::pair{&Buffers.FacePrimitives, Domain::Face}, std::pair{&Buffers.VertexPrimitives, Domain::Vertex}})
            for (const auto block : targets[uint32_t(domain)])
                if (layer->PayloadBlock(block))
                    for (auto &primitive : layer->Edit(block * MeshElementBlockSize, MeshElementBlockSize)) primitive += primitives;
    if (Records[id].Classification != Records[source].Classification) UpdateCornerClassification(r, chain, std::array{id});
    UpdateSelection(r, chain, std::array{SelectionUpdate{.StoreId = id, .Blocks = {targets[uint32_t(Domain::Vertex)], targets[uint32_t(Domain::Edge)], targets[uint32_t(Domain::Face)]}}});
    chain.Submit();
}

namespace {
// A source's handles of one kind in ascending order, and the clone's handle of each at its rank.
// A render record references only its owner's handles of each kind, so every handle it maps is one of Sources.
struct HandleMap {
    std::vector<uint32_t> Sources;
    uint32_t First{InvalidOffset};
    uint32_t operator()(uint32_t handle) const {
        return handle == InvalidOffset ? InvalidOffset : First + uint32_t(std::ranges::lower_bound(Sources, handle) - Sources.begin());
    }
};
} // namespace

void MeshStore::CloneRenderRecords(CloneCopies &copies, std::span<const uint32_t> source_ids, std::span<const uint32_t> clone_ids, std::span<const std::array<Range, 6>> maps) {
    auto &render = Buffers.Render;
    auto &index = render.ActiveMeshlets;
    const auto members = [&](uint32_t root) {
        HandleMap map;
        index.ForEach(root, [&](uint32_t id) { map.Sources.push_back(id); });
        return map;
    };
    struct Clone {
        uint32_t SourceId, CloneId, MapIndex;
        HandleMap Meshlets, Primitives, Nodes, Groups;
    };
    // A membership root the batch's index update creates for a clone: a leaf node's members, or a dirty root's.
    struct MemberRoot {
        enum class Kind : uint8_t { Leaf,
                                    PositionDirty,
                                    DirtyGroups };
        uint32_t Clone;
        Kind Type;
        uint32_t Node{InvalidOffset};
        std::vector<uint32_t> Members;
    };
    std::vector<Clone> clones;
    uint64_t meshlet_count = 0u, primitive_count = 0u, node_count = 0u, group_count = 0u;
    for (uint32_t i = 0u; i < source_ids.size(); ++i) {
        const auto &source = Records[source_ids[i]];
        if (source.MeshletRoot == InvalidOffset) continue;
        auto &clone = clones.emplace_back(Clone{.SourceId = source_ids[i], .CloneId = clone_ids[i], .MapIndex = i});
        clone.Meshlets = members(source.MeshletRoot);
        clone.Primitives = members(source.PrimitiveRoot);
        clone.Nodes = members(source.NodeRoot);
        clone.Groups = members(source.GroupRoot);
        meshlet_count += clone.Meshlets.Sources.size();
        primitive_count += clone.Primitives.Sources.size();
        node_count += clone.Nodes.Sources.size();
        group_count += clone.Groups.Sources.size();
    }
    if (clones.empty()) return;
    // Every arena grows once for the batch.
    render.Meshlets.ReserveAdditional(meshlet_count);
    render.MeshletLodLeaves.ReserveAdditional(meshlet_count);
    render.MeshletSpatialNodes.ReserveAdditional(meshlet_count);
    render.Primitives.ReserveAdditional(primitive_count);
    render.LodNodes.ReserveAdditional(node_count);
    render.LodParents.ReserveAdditional(node_count);
    render.ClusterGroups.ReserveAdditional(group_count);
    render.GroupLinks.ReserveAdditional(group_count);
    const auto copy = [&](auto &arena, uint64_t from, uint64_t to, uint64_t count) {
        constexpr uint64_t Size = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
        copies.Copy(arena.Buffer, from * Size, to * Size, count * Size);
    };
    std::vector<MeshletIndexEdit> ownership;
    std::vector<MemberRoot> roots;
    for (uint32_t c = 0u; c < clones.size(); ++c) {
        auto &clone = clones[c];
        const auto &source = Records[clone.SourceId];
        auto &target = Records[clone.CloneId];
        target.RenderTopologies = source.RenderTopologies;
        target.Level0Count = source.Level0Count;
        target.MeshletRevision = source.MeshletRevision;
        target.LodDepth = source.LodDepth;
        target.Meshlets = render.AllocateMeshlets(uint32_t(clone.Meshlets.Sources.size()));
        clone.Meshlets.First = target.Meshlets.Offset;
        target.Primitives = render.Primitives.Allocate(uint32_t(clone.Primitives.Sources.size()));
        clone.Primitives.First = target.Primitives.Offset;
        target.LodNodes = render.LodNodes.Allocate(uint32_t(clone.Nodes.Sources.size()));
        render.LodParents.Mirror(target.LodNodes);
        clone.Nodes.First = target.LodNodes.Offset;
        target.ClusterGroups = render.ClusterGroups.Allocate(uint32_t(clone.Groups.Sources.size()));
        render.GroupLinks.Mirror(target.ClusterGroups);
        clone.Groups.First = target.ClusterGroups.Offset;
        // The chain gathers the source's records by rank and rebases their references to the clone's handle at the same rank.
        const auto clusters = index.Ref(source.MeshletRoot), nodes_index = index.Ref(source.NodeRoot);
        // Cluster payloads pack in cluster order, so a source built in one pass keeps its construction layout and copies in a few runs.
        {
            const auto source_records = render.Meshlets.Buffer.GetSpan<MeshletRecord>();
            uint64_t triangle_ids = 0u, vertex_corners = 0u, local_triangles = 0u;
            for (const auto id : clone.Meshlets.Sources) {
                const auto &record = source_records[id];
                if (record.RefinedGroup == InvalidOffset) triangle_ids += record.TriangleCount;
                vertex_corners += record.VertexCount;
                if (record.Topology == 0u) local_triangles += uint64_t(record.TriangleCount) * 3u;
            }
            target.MeshletTriangles = render.MeshletTriangleIds.Allocate(uint32_t(triangle_ids));
            target.MeshletVertices = render.MeshletVertexCorners.Allocate(uint32_t(vertex_corners));
            target.MeshletLocalTriangles = render.MeshletLocalTriangles.Allocate(uint32_t(local_triangles));
            render.MeshletTriangleIds.CaptureWrite(target.MeshletTriangles);
            render.MeshletVertexCorners.CaptureWrite(target.MeshletVertices);
            render.MeshletLocalTriangles.CaptureWrite(target.MeshletLocalTriangles);
            auto records = render.Meshlets.GetMutable(target.Meshlets);
            std::array<std::vector<Range>, 3> element_runs, vertex_runs;
            const auto append_run = [](std::vector<Range> &runs, Range range) {
                if (!range.Count) return;
                if (!runs.empty() && runs.back().Offset + runs.back().Count == range.Offset) runs.back().Count += range.Count;
                else runs.push_back(range);
            };
            uint32_t next_triangle = target.MeshletTriangles.Offset, next_vertex = target.MeshletVertices.Offset, next_local = target.MeshletLocalTriangles.Offset;
            for (uint32_t m = 0u; m < clone.Meshlets.Sources.size(); ++m) {
                const auto id = clone.Meshlets.Sources[m];
                auto record = source_records[id];
                if (record.RefinedGroup == InvalidOffset) {
                    copy(render.MeshletTriangleIds, record.TriangleOffset, next_triangle, record.TriangleCount);
                    record.TriangleOffset = std::exchange(next_triangle, next_triangle + record.TriangleCount);
                    append_run(element_runs.at(record.Topology), {record.TriangleOffset, record.TriangleCount});
                }
                copy(render.MeshletVertexCorners, record.VertexOffset, next_vertex, record.VertexCount);
                record.VertexOffset = std::exchange(next_vertex, next_vertex + record.VertexCount);
                append_run(vertex_runs.at(record.Topology), {record.VertexOffset, record.VertexCount});
                if (record.Topology == 0u) {
                    copy(render.MeshletLocalTriangles, record.LocalTriangleOffset, next_local, uint64_t(record.TriangleCount) * 3u);
                    record.LocalTriangleOffset = std::exchange(next_local, next_local + record.TriangleCount * 3u);
                }
                records[m] = record;
            }
            constexpr auto RecordWords = uint32_t(sizeof(MeshletRecord) / sizeof(uint32_t)), SpatialWords = uint32_t(sizeof(MeshletSpatialNode) / sizeof(uint32_t));
            const auto record_bytes = uint64_t(target.Meshlets.Offset) * sizeof(MeshletRecord);
            copies.RebaseByRank(render.Meshlets.Buffer, record_bytes + offsetof(MeshletRecord, Primitive), target.Meshlets.Count, index.Ref(source.PrimitiveRoot), target.Primitives.Offset, RecordWords);
            for (const auto field : {offsetof(MeshletRecord, GroupIndex), offsetof(MeshletRecord, RefinedGroup)})
                copies.RebaseByRank(render.Meshlets.Buffer, record_bytes + field, target.Meshlets.Count, index.Ref(source.GroupRoot), target.ClusterGroups.Offset, RecordWords);
            render.MeshletLodLeaves.CaptureWrite(target.Meshlets);
            copies.GatherByRank(render.MeshletLodLeaves.Buffer, target.Meshlets.Offset, target.Meshlets.Count, sizeof(uint32_t), clusters);
            copies.RebaseByRank(render.MeshletLodLeaves.Buffer, uint64_t(target.Meshlets.Offset) * sizeof(uint32_t), target.Meshlets.Count, nodes_index, target.LodNodes.Offset);
            render.MeshletSpatialNodes.CaptureWrite(target.Meshlets);
            copies.GatherByRank(render.MeshletSpatialNodes.Buffer, target.Meshlets.Offset, target.Meshlets.Count, sizeof(MeshletSpatialNode), clusters);
            const auto spatial_bytes = uint64_t(target.Meshlets.Offset) * sizeof(MeshletSpatialNode);
            for (const auto field : {offsetof(MeshletSpatialNode, Parent), offsetof(MeshletSpatialNode, Left), offsetof(MeshletSpatialNode, Right), offsetof(MeshletSpatialNode, Meshlet)})
                copies.RebaseByRank(render.MeshletSpatialNodes.Buffer, spatial_bytes + field, target.Meshlets.Count, clusters, target.Meshlets.Offset, SpatialWords);
            target.SpatialRoot = clone.Meshlets(source.SpatialRoot);
            const auto &map = maps[clone.MapIndex];
            for (uint32_t topology = 0u; topology < 3u; ++topology) {
                const bool triangles = topology == 0u;
                const auto domain = triangles ? Domain::Triangle : topology == 1u ? Domain::Edge :
                                                                                    Domain::Vertex;
                const auto source_origin = triangles ? 0u : RenderDomainFirst(source, topology);
                const auto target_origin = triangles ? 0u : RenderDomainFirst(target, topology);
                for (const auto run : element_runs[topology])
                    copies.RebaseByBlock(render.MeshletTriangleIds.Buffer, uint64_t(run.Offset) * 4u, run.Count, map[uint32_t(domain)], 1u, source_origin, target_origin);
                for (const auto run : vertex_runs[topology])
                    copies.RebaseByBlock(render.MeshletVertexCorners.Buffer, uint64_t(run.Offset) * 4u, run.Count, map[uint32_t(triangles ? Domain::Halfedge : Domain::Vertex)], 1u, triangles ? 0u : Buffers.Vertices.First(source.Vertices), triangles ? 0u : Buffers.Vertices.First(target.Vertices));
            }
        }
        // Primitives and their routes. A primitive's construction slice keeps its place in the clone's packed triangle IDs.
        {
            const auto materials = OffsetOrInvalid(target.PrimitiveMaterials);
            auto primitives = render.Primitives.GetMutable(target.Primitives);
            for (uint32_t p = 0u; p < clone.Primitives.Sources.size(); ++p) {
                auto primitive = render.Primitives.Get({clone.Primitives.Sources[p], 1u})[0];
                primitive.PrimitiveMaterialOffset = materials;
                primitive.TriangleOffset = primitive.TriangleOffset - source.MeshletTriangles.Offset + target.MeshletTriangles.Offset;
                primitive.LodRootNode = clone.Nodes(primitive.LodRootNode);
                primitive.LodFinestNode = clone.Nodes(primitive.LodFinestNode);
                primitives[p] = primitive;
            }
            if (source.PrimitiveRoutes.Count) {
                std::vector<uint32_t> routes;
                for (const auto route : render.PrimitiveRoutes.Get(source.PrimitiveRoutes)) routes.push_back(clone.Primitives(route));
                target.PrimitiveRoutes = render.PrimitiveRoutes.Allocate(routes);
            }
        }
        // LOD nodes and parents, with each leaf's members listed for a membership root of its own.
        {
            auto nodes = render.LodNodes.GetMutable(target.LodNodes);
            render.LodParents.CaptureWrite(target.LodNodes);
            copies.GatherByRank(render.LodParents.Buffer, target.LodNodes.Offset, target.LodNodes.Count, sizeof(uint32_t), nodes_index);
            copies.RebaseByRank(render.LodParents.Buffer, uint64_t(target.LodNodes.Offset) * sizeof(uint32_t), target.LodNodes.Count, nodes_index, target.LodNodes.Offset);
            for (uint32_t n = 0u; n < clone.Nodes.Sources.size(); ++n) {
                const auto id = clone.Nodes.Sources[n];
                auto node = render.LodNodes.Get({id, 1u})[0];
                if (node.ChildCount) node.ChildOffset = clone.Nodes(node.ChildOffset);
                if (node.MeshletRoot != InvalidOffset) {
                    auto &leaf = roots.emplace_back(MemberRoot{.Clone = c, .Type = MemberRoot::Kind::Leaf, .Node = target.LodNodes.Offset + n});
                    index.ForEach(node.MeshletRoot, [&](uint32_t member) { leaf.Members.push_back(clone.Meshlets(member)); });
                }
                node.MeshletRoot = InvalidOffset;
                nodes[n] = node;
            }
        }
        // Cluster groups, with each group's member run then proxy run filling one allocation.
        {
            uint64_t group_ids = 0u;
            for (const auto group : clone.Groups.Sources) {
                const auto &links = render.GroupLinks.Get({group, 1u})[0];
                group_ids += links.MemberCount + links.ProxyCount;
            }
            const auto runs = render.GroupClusterIds.Allocate(uint32_t(group_ids));
            render.GroupClusterIds.CaptureWrite(runs);
            auto groups = render.ClusterGroups.GetMutable(target.ClusterGroups);
            auto links = render.GroupLinks.GetMutable(target.ClusterGroups);
            uint32_t next = 0u;
            const auto copy_run = [&](uint32_t offset, uint32_t count) {
                const auto first = runs.Offset + next;
                copy(render.GroupClusterIds, offset, first, count);
                copies.RebaseByRank(render.GroupClusterIds.Buffer, uint64_t(first) * sizeof(uint32_t), count, clusters, target.Meshlets.Offset);
                next += count;
                return first;
            };
            for (uint32_t g = 0u; g < clone.Groups.Sources.size(); ++g) {
                const auto group = clone.Groups.Sources[g];
                groups[g] = render.ClusterGroups.Get({group, 1u})[0];
                const auto source_links = render.GroupLinks.Get({group, 1u})[0];
                const auto member_offset = copy_run(source_links.MemberOffset, source_links.MemberCount);
                links[g] = {.MemberOffset = member_offset, .MemberCount = source_links.MemberCount, .ProxyOffset = copy_run(source_links.ProxyOffset, source_links.ProxyCount), .ProxyCount = source_links.ProxyCount};
            }
        }
        // Element owners of the source's element blocks move onto the clone's blocks, naming the clone's clusters.
        // Live slots retain their offset within their remapped block.
        for (uint32_t topology = 0u; topology < 3u; ++topology)
            if (source.ElementMeshletBlockCounts[topology]) {
                auto &owners = render.ElementMeshlets[topology];
                const auto source_blocks = MeshletOwnerBlocks(source, topology);
                const auto clone_first = RenderDomainFirst(target, topology);
                const auto domain = topology == 0u ? Domain::Triangle : topology == 1u ? Domain::Edge :
                                                                                         Domain::Vertex;
                const auto map = maps[clone.MapIndex][uint32_t(domain)];
                std::vector<uint32_t> target_blocks;
                for (const auto block : source_blocks) target_blocks.push_back(copies.MapHandle(map, block * MeshElementBlockSize) / MeshElementBlockSize);
                owners.Attach(target_blocks, InvalidOffset);
                for (uint32_t b = 0u; b < source_blocks.size(); ++b) {
                    const auto from = owners.Payload(source_blocks[b] * MeshElementBlockSize, MeshElementBlockSize);
                    const auto to = owners.Payload(target_blocks[b] * MeshElementBlockSize, MeshElementBlockSize);
                    owners.Values.Buffer.CaptureWrite(uint64_t(to.Offset) * sizeof(uint32_t), uint64_t(to.Count) * sizeof(uint32_t));
                    copies.Copy(owners.Values.Buffer, uint64_t(from.Offset) * sizeof(uint32_t), uint64_t(to.Offset) * sizeof(uint32_t), uint64_t(to.Count) * sizeof(uint32_t));
                    copies.RebaseByRank(owners.Values.Buffer, uint64_t(to.Offset) * sizeof(uint32_t), to.Count, clusters, target.Meshlets.Offset);
                }
                target.ElementMeshletBlockCounts[topology] = uint32_t(source_blocks.size());
                // Triangle owners name absolute triangle handles, and point and line owners name elements from the domain's first handle.
                target.ElementMeshletOrigins[topology] = topology == 0u ? 0u : clone_first;
            }
        for (const auto range : {target.Meshlets, target.Primitives, target.LodNodes, target.ClusterGroups}) ownership.push_back({.Insert = range});
        for (const auto [root, kind] : {std::pair{source.PositionDirtyRoot, MemberRoot::Kind::PositionDirty}, std::pair{source.DirtyGroupRoot, MemberRoot::Kind::DirtyGroups}}) {
            if (root == InvalidOffset) continue;
            auto &dirty = roots.emplace_back(MemberRoot{.Clone = c, .Type = kind});
            const auto &map = kind == MemberRoot::Kind::PositionDirty ? clone.Meshlets : clone.Groups;
            index.ForEach(root, [&](uint32_t member) { dirty.Members.push_back(map(member)); });
        }
    }
    const auto first_root = uint32_t(ownership.size());
    for (const auto &root : roots) ownership.push_back({.Added = root.Members});
    index.Update(ownership);
    for (uint32_t i = 0u; i < roots.size(); ++i) {
        const auto &[c, kind, node, _] = roots[i];
        const auto root = ownership[first_root + i].Root;
        auto &target = Records[clones[c].CloneId];
        switch (kind) {
            case MemberRoot::Kind::Leaf: render.LodNodes.GetMutable({node, 1u})[0].MeshletRoot = root; break;
            case MemberRoot::Kind::PositionDirty: target.PositionDirtyRoot = root; break;
            case MemberRoot::Kind::DirtyGroups: target.DirtyGroupRoot = root; break;
        }
    }
    for (uint32_t c = 0u; c < clones.size(); ++c) {
        auto &target = Records[clones[c].CloneId];
        const auto *edits = &ownership[4u * c];
        target.MeshletRoot = edits[0].Root;
        target.PrimitiveRoot = edits[1].Root;
        target.NodeRoot = edits[2].Root;
        target.GroupRoot = edits[3].Root;
    }
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
        std::span<const MeshElementSet> Headers;
        std::span<const MeshElementBlock> Membership;
    };
    std::array<DomainRetirement, 6> closure;
    constexpr std::array domains{Domain::Vertex, Domain::Halfedge, Domain::Edge, Domain::Face, Domain::Triangle};
    for (const auto domain : domains) WithDomain(Buffers, domain, [&](const auto &arena) {
        auto &c = closure[uint32_t(domain)];
        c.Headers = arena.Sets.Buffer.template GetSpan<MeshElementSet>();
        c.Membership = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
    });
    std::vector<Range> materials, summaries, normals;
    for (const auto id : ids) {
        const auto &record = Records[id];
        for (const auto domain : domains) {
            const auto set = DomainSet(record, domain);
            if (!set) continue;
            auto &c = closure[uint32_t(domain)];
            if (set.Index >= c.Headers.size() || c.Headers[set.Index].First == InvalidOffset) throw std::invalid_argument("Invalid element set in mesh retirement.");
            c.Sets.push_back(set);
            for (auto block = c.Headers[set.Index].First; block != InvalidOffset; block = c.Membership[block].Next) c.Blocks.push_back(block);
        }
        if (record.PrimitiveMaterials.Count) materials.push_back(record.PrimitiveMaterials);
        if (record.SelectionSummary.Count) summaries.push_back(record.SelectionSummary);
        if (record.PointNormals.Count) normals.push_back(record.PointNormals);
    }
    for (auto &c : closure)
        if (!std::ranges::is_sorted(c.Blocks)) std::ranges::sort(c.Blocks);
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
    uint32_t live_vertices = 0u;
    for (const auto set : vertices.Sets) live_vertices += vertices.Headers[set.Index].Count;
    Buffers.VertexFans.Release(ElementView<uvec2>{Buffers.VertexCorners.Buffer.GetSpan<uvec2>(), vertices.Membership, vertices.Blocks, {}, live_vertices});
    ForEachIndexRun(vertices.Blocks, [&](size_t first, size_t count) {
        const Range range{vertices.Blocks[first] * MeshElementBlockSize, uint32_t(count) * MeshElementBlockSize};
        std::ranges::fill(Buffers.VertexCorners.GetMutable(range), uvec2{InvalidOffset, 0u});
    });
    const auto &corners = closure[uint32_t(Domain::Halfedge)].Blocks;
    Buffers.CornerSectors.Release(corners);
    Buffers.NormalSectors.Release(corners);
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
    for (const auto id : ids)
        for (uint32_t d = 0u; d < 3u; ++d) Buffers.SelectionTree.Release(3u * id + d);
    ForEachAttribute(Buffers, [&](auto &attributes, Domain domain, uint32_t, const char *, const char *, const char *, auto &&) {
        attributes.Release(closure[uint32_t(domain)].Blocks);
    });
    for (uint32_t attribute = 0u; attribute < NamedAttributes.size(); ++attribute)
        if (std::ranges::binary_search(ids, NamedAttributes[attribute].Record)) ReleaseAttribute(attribute);
    for (const auto domain : domains) WithDomain(Buffers, domain, [&](auto &arena) {
        const auto &c = closure[uint32_t(domain)];
        arena.Destroy(c.Sets, c.Blocks);
    });
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

void MeshStore::ReleaseRender(std::span<Record *const> records) {
    const profile::CpuScope scope{"ReleaseRenderStorage"};
    auto &render = Buffers.Render;
    std::vector<Range> groups, group_ids, nodes, primitives, routes;
    std::vector<Range> membership_nodes, membership_leaves;
    std::vector<uint32_t> meshlets;
    std::array<std::vector<uint32_t>, 3> owner_blocks;
    const auto membership = render.ActiveMeshlets.Read();
    const auto lod_nodes = render.LodNodes.Buffer.GetSpan<LodNode>();
    const auto links = render.GroupLinks.Get({0u, render.GroupLinks.Buffer.Count<ClusterGroupLinks>()});
    enum class Kind { Group,
                      Node,
                      Meshlet,
                      Primitive };
    std::vector<uint32_t> roots, additional_roots;
    std::vector<Kind> jobs;
    const auto collect = [&](uint32_t root, Kind kind) {
        if (root == InvalidOffset) return;
        roots.push_back(root);
        jobs.push_back(kind);
    };
    const auto release_group = [&](uint32_t id) {
        const auto &group = links[id];
        AppendRange(group_ids, {group.MemberOffset, group.MemberCount});
        AppendRange(group_ids, {group.ProxyOffset, group.ProxyCount});
        AppendRange(groups, {id, 1u});
    };
    const auto release_node = [&](uint32_t id) {
        if (lod_nodes[id].MeshletRoot != InvalidOffset) additional_roots.push_back(lod_nodes[id].MeshletRoot);
        AppendRange(nodes, {id, 1u});
    };
    for (const auto *record : records) {
        const auto &b = *record;
        for (const auto root : {b.PositionDirtyRoot, b.DirtyGroupRoot})
            if (root != InvalidOffset) additional_roots.push_back(root);
        if (b.GroupRoot == InvalidOffset) {
            for (uint32_t i = 0u; i < b.ClusterGroups.Count; ++i) release_group(b.ClusterGroups.Offset + i);
        } else collect(b.GroupRoot, Kind::Group);
        if (b.NodeRoot == InvalidOffset) {
            for (uint32_t i = 0u; i < b.LodNodes.Count; ++i) release_node(b.LodNodes.Offset + i);
        } else collect(b.NodeRoot, Kind::Node);
        if (b.MeshletRoot == InvalidOffset) {
            for (uint32_t i = 0u; i < b.Meshlets.Count; ++i) meshlets.push_back(b.Meshlets.Offset + i);
        } else {
            collect(b.MeshletRoot, Kind::Meshlet);
            for (uint32_t topology = 0u; topology < 3u; ++topology)
                if (b.ElementMeshletBlockCounts[topology]) owner_blocks[topology].append_range(MeshletOwnerBlocks(b, topology));
        }
        if (b.PrimitiveRoot == InvalidOffset) AppendRange(primitives, b.Primitives);
        else collect(b.PrimitiveRoot, Kind::Primitive);
        AppendRange(routes, b.PrimitiveRoutes);
    }
    membership.CollectOwned(roots, membership_nodes, membership_leaves, [&](uint32_t root, const MeshletIndexLeaf &leaf) {
        MeshletIndex::ForEach(leaf, [&](uint32_t handle) {
            switch (jobs[root]) {
                case Kind::Group: release_group(handle); break;
                case Kind::Node: release_node(handle); break;
                case Kind::Meshlet: meshlets.push_back(handle); break;
                case Kind::Primitive: AppendRange(primitives, {handle, 1u}); break;
            }
        });
    });
    membership.CollectOwned(additional_roots, membership_nodes, membership_leaves, [](uint32_t, const auto &) {});
    for (auto *record : records) ResetRender(*record);
    for (uint32_t i = 0u; i < owner_blocks.size(); ++i) {
        SortUnique(owner_blocks[i]);
        render.ElementMeshlets[i].Release(owner_blocks[i]);
    }
    render.ActiveMeshlets.Leaves.Release(std::move(membership_leaves));
    render.ActiveMeshlets.Nodes.Release(std::move(membership_nodes));
    render.ReleaseMeshletStorage(meshlets);
    render.GroupClusterIds.Release(std::move(group_ids));
    render.ClusterGroups.Release(std::move(groups));
    render.LodNodes.Release(std::move(nodes));
    render.Primitives.Release(std::move(primitives));
    render.PrimitiveRoutes.Release(std::move(routes));
}

void MeshStore::ReservePrimitiveRoutes(Record &record, uint32_t count) {
    if (uint64_t(count) * 3u > UINT32_MAX) throw std::length_error("Primitive routes exceed arena address space.");
    count *= 3u;
    if (count <= record.PrimitiveRoutes.Count) return;
    std::vector<uint32_t> routes(count, InvalidOffset);
    std::ranges::copy(Buffers.Render.PrimitiveRoutes.Get(record.PrimitiveRoutes), routes.begin());
    Buffers.Render.PrimitiveRoutes.Update(record.PrimitiveRoutes, routes);
}

std::vector<uint32_t> MeshStore::MeshletOwnerBlocks(const Record &record, uint32_t topology) const {
    // Owners attach to whole blocks of the record's element domain, which no other record shares.
    const auto &owners = Buffers.Render.ElementMeshlets[topology];
    std::vector<uint32_t> blocks;
    WithRenderDomain(record, topology, [&](const auto &arena, ElementSetRef set) {
        arena.ForEachBlock(set, [&](uint32_t block, const auto &) { if (owners.PayloadBlock(block)) blocks.push_back(block); });
    });
    std::ranges::sort(blocks);
    return blocks;
}

uint32_t MeshStore::RenderDomainFirst(const Record &record, uint32_t topology) const {
    return WithRenderDomain(record, topology, [](const auto &arena, ElementSetRef set) { return arena.First(set); });
}

void MeshStore::SetExtrasIndices(uint32_t id, std::span<const uint32_t> faces, std::span<const uint32_t> edges) {
    auto &record = WriteRecord(id);
    const auto first = Buffers.Vertices.First(record.Vertices);
    const auto offset = [&](auto &arena, std::span<const uint32_t> indices) {
        const auto range = arena.Allocate(uint32_t(indices.size()));
        std::ranges::transform(indices, arena.GetMutable(range).begin(), [first](uint32_t v) { return first + v; });
        return range;
    };
    record.ExtrasFaces = offset(Buffers.Render.ExtrasFaces, faces);
    record.ExtrasEdges = offset(Buffers.Render.ExtrasEdges, edges);
}

void MeshStore::Clear() {
    if (Tracked) {
        Tracked->Entries.Write(0, Records.size());
        Tracked->Free.Write(0, FreeIds.size());
        Tracked->Attributes.Write(0, NamedAttributes.size());
        Tracked->AllDirty = true;
    }
    Buffers.VertexFans.Reset();
    Buffers.SelectionTree.Reset();
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.Reset(); });
    Records.clear();
    DerivedRecords.clear();
    NamedAttributes.clear();
    {
        const std::scoped_lock lock{BlockListLock};
        BlockLists.Reset();
        BlockListEntries.clear();
        RetiredBlockLists.clear();
    }
    FreeIds.clear();
    Released.clear();
    for (auto &log : WriteLogs) log.Extents.clear();
    WrittenAttributes.clear();
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
