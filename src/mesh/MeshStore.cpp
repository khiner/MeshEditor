
#include "MeshStore.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"

#include "CornerNormalOffset.h"
#include "Profile.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/MeshPipelines.h"
#include "project/store/Pages.h"

#include <map>

namespace {
constexpr uint32_t UniformFaceMode{uint32_t(CornerClassMode::UniformFace)};

// A run of copied uint32 reference pairs.
struct ReferencePairCopy { uint32_t Source, Destination, Count; };

// Adds delta to copied uint32 arena references, preserving the null sentinel.
// Range.Offset and stride are in uint32 words, and Range.Count counts references.
// byte_base locates a record stream without narrowing its byte address.
void EncodeRebaseIndices(MTL::CommandBuffer *command, const mtl::ComputePipeline &pipeline, MTL::Buffer *buffer, Range range, uint32_t delta, uint32_t stride = 1, uint64_t byte_base = 0) {
    if (range.Count == 0 || delta == 0) return;
    auto *encoder = command->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.State());
    encoder->setBuffer(buffer, byte_base + uint64_t(range.Offset) * sizeof(uint32_t), 0);
    const uint32_t pc[]{range.Count, delta, stride, 0u};
    encoder->setBytes(pc, sizeof(pc), 1);
    encoder->dispatchThreads(MTL::Size(range.Count, 1, 1), MTL::Size(256, 1, 1));
    encoder->endEncoding();
}

// Copies each run of reference pairs, adding first_delta and second_delta to the non-null pair members.
void EncodeCopyReferencePairs(MTL::CommandBuffer *command, const mtl::ComputePipeline &pipeline, MTL::Buffer *buffer,
    std::span<const ReferencePairCopy> jobs, uint32_t first_delta, uint32_t second_delta) {
    if (jobs.empty()) return;
    static_assert(sizeof(ReferencePairCopy) == 12u);
    std::vector<std::array<uint32_t, 2>> tiles;
    for (uint32_t i = 0u; i < jobs.size(); ++i)
        for (uint64_t first = 0u; first < jobs[i].Count; first += 32u) tiles.push_back({i, uint32_t(first)});
    auto *device = buffer->device();
    auto input = NS::TransferPtr(device->newBuffer(jobs.data(), jobs.size_bytes(), MTL::ResourceStorageModeShared));
    auto work = NS::TransferPtr(device->newBuffer(tiles.data(), tiles.size() * sizeof(tiles[0]), MTL::ResourceStorageModeShared));
    if (!input || !work) throw std::runtime_error("Reference copy descriptors failed.");
    auto *encoder = command->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.State());
    encoder->setBuffer(buffer, 0, 0);
    encoder->setBuffer(input.get(), 0, 1);
    encoder->setBuffer(work.get(), 0, 2);
    const uint32_t delta[]{first_delta, second_delta};
    encoder->setBytes(delta, sizeof(delta), 3);
    encoder->dispatchThreadgroups(MTL::Size(tiles.size(), 1, 1), MTL::Size(32, 1, 1));
    encoder->endEncoding();
}

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
    f(b.SelectionRoots, ArenaInfo{}, NoAllocation);
    f(b.CornerSectors.Values, ArenaInfo{0,"CornerSectorsValues"}, NoAllocation);
    f(b.CornerSectors.Blocks, ArenaInfo{0,"CornerSectorsBlocks"}, NoAllocation);
    f(b.CornerSectors.Owners, ArenaInfo{0,"CornerSectorsOwners"}, NoAllocation);
    f(b.NormalSectors.Values, ArenaInfo{0,"NormalSectorsValues"}, NoAllocation);
    f(b.NormalSectors.Blocks, ArenaInfo{0,"NormalSectorsBlocks"}, NoAllocation);
    f(b.NormalSectors.Owners, ArenaInfo{0,"NormalSectorsOwners"}, NoAllocation);
    f(b.BaseVertexNormals, ArenaInfo{0,"BaseVertexNormal",true,Domain::Vertex}, [](auto &e) { return &e.Vertices; });
    f(b.BaseFaceNormals, ArenaInfo{0,"BaseFaceNormal",true,Domain::Face}, [](auto &e) { return &e.FaceData; });
}

template<typename Arena> using ArenaValue = typename decltype(std::declval<const Arena &>().Get(Range{}))::value_type;

decltype(auto) WithDomain(auto &b, Domain domain, auto &&fn) {
    switch (domain) {
        case Domain::Vertex: return fn(b.Vertices);
        case Domain::Halfedge: return fn(b.FaceCorners);
        case Domain::Edge: return fn(b.EdgeHalfedges);
        case Domain::Face: return fn(b.FaceTriangles);
        case Domain::Triangle: return fn(b.Triangles);
        case Domain::None: throw std::logic_error("Missing element domain.");
    }
}

auto &DomainSet(auto &record, Domain domain) {
    switch (domain) {
        case Domain::Vertex: return record.Vertices;
        case Domain::Halfedge: return record.FaceCorners;
        case Domain::Edge: return record.EdgeData;
        case Domain::Face: return record.FaceData;
        case Domain::Triangle: return record.TriangleData;
        case Domain::None: throw std::invalid_argument("Missing element domain.");
    }
}

constexpr bool SelectableDomain(Domain domain) { return domain == Domain::Vertex || domain == Domain::Edge || domain == Domain::Face; }

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
    a.Vertices.ForEachBlock(vertices,[&](uint32_t block, const auto &) {
        const Range range{block*MeshElementBlockSize,MeshElementBlockSize};
        a.VertexCorners.Buffer.CaptureWrite(uint64_t(range.Offset)*sizeof(uvec2),uint64_t(range.Count)*sizeof(uvec2));
        std::ranges::fill(a.VertexCorners.GetMutable(range),uvec2{InvalidOffset,0u});
    });
}

Range DenseRange(const MeshArenas &, const ArenaInfo &, Range range) { return range; }
Range DenseRange(const MeshArenas &b, const ArenaInfo &info, ElementSetRef set) {
    return WithDomain(b, info.Elements, [&](const auto &owner) { return owner.Dense(set); });
}

} // namespace

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
      VertexAggregates{ctx, SlotType::Buffer}, EdgeAggregates{ctx, SlotType::Buffer}, FaceAggregates{ctx, SlotType::Buffer},
      SelectionRoots{ctx, SlotType::Buffer},
      Query{ctx},
      SelectionSummary{ctx, SlotType::Buffer},
      EdgeSharpness{ctx, SlotType::Buffer},
      CustomNormals{ctx, SlotType::Buffer},
      PointNormals{ctx, SlotType::Buffer},
      CornerTangents{ctx, SlotType::CornerTangentBuffer},
      CornerColors{ctx, SlotType::CornerColorBuffer},
      VertexColors{ctx, SlotType::CornerColorBuffer},
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
      BaseFaceNormals{ctx, SlotType::Buffer} {}

struct MeshStore::HistoryState {
    struct Extent {
        uint64_t End;
        uint32_t Id, Bits;
    };
    enum class StreamKind { Values, Blocks, Sets };
    struct DomainStream {
        mtl::Buffer *Buffer;
        Domain Elements;
        uint32_t Stride, Bits;
        StreamKind Kind;
        const BufferArena<uint32_t> *ValueOwners{};
    };
    // The entries a record added to the index, so a changed record replaces only its own.
    struct IndexedRecord {
        std::vector<std::pair<uint32_t, uint32_t>> Sets; // Domain and set index.
        std::vector<std::pair<mtl::Buffer *, uint64_t>> Extents; // Buffer and first byte.
    };
    store::Records Entries, Free;
    // Each non-domain arena's record extents, keyed by their disjoint first bytes.
    std::unordered_map<mtl::Buffer *, std::map<uint64_t, Extent>> Ranges;
    std::array<std::unordered_map<uint32_t, uint32_t>, uint32_t(Domain::Triangle) + 1u> Owners;
    std::vector<DomainStream> Streams;
    std::vector<IndexedRecord> Indexed;
    // Records whose ranges or sets changed since the last index, and whether every record did.
    std::vector<uint32_t> Dirty;
    bool AllDirty{true};

    HistoryState(MeshStore &mesh, store::History &history) : Entries(mesh.Records), Free(mesh.FreeIds) {
        Entries.Trie.CollectChanged = true;
        history.Track(Entries, "mesh.entries", 0);
        history.Track(Free, "mesh.free", 0);
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
            std::ranges::sort(Dirty);
            Dirty.erase(std::unique(Dirty.begin(), Dirty.end()), Dirty.end());
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
      BlockLists{ctx, SlotType::Buffer},
      SelectionWork{ctx, SlotType::Buffer, mtl::BufferLifetime::Workspace},
      SelectionDirty{ctx, SlotType::Buffer, mtl::BufferLifetime::Workspace} {}
MeshStore::~MeshStore() = default;

void MeshStore::Track(store::History &history) {
    Buffers.VertexFans.Track(history,"mesh.vertexFans");
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
        if (info.Elements == Domain::None) Tracked->Ranges.try_emplace(&arena.Buffer);
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
    if (!history) return;
    const uint64_t page_size = history->PageBytes;
    std::vector<uint32_t> pages;
    selection.ForEach([&](uint32_t handle) {
        const uint64_t offset = uint64_t(handle) * stride;
        const auto first = offset / page_size, end = (offset + stride + page_size - 1) / page_size;
        for (auto page=first;page<end;++page)
            if (pages.empty() || pages.back()<page) pages.push_back(uint32_t(page));
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
    if (Tracked) SelectionArena(Buffers, SelectionDomain(element)).Buffer.CaptureWriteElements(blocks, sizeof(MeshArenas::SelectionBlock));
}

void MeshStore::CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation operation) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    switch (operation) {
        case EditSharpnessOperation::SetSelectedFaces:
            CaptureSelected(Buffers.FaceSharpness.Buffer, 1, GetSelectedElements(id, Element::Face));
            break;
        case EditSharpnessOperation::SetSelectedEdges:
            CaptureSelected(Buffers.EdgeSharpness.Buffer, 1, GetSelectedElements(id, Element::Edge));
            break;
        case EditSharpnessOperation::SetVertexEdges: {
            const auto edges = GetVertexEdgeIncidence(id);
            std::vector<uint32_t> touched;
            GetSelectedElements(id, Element::Vertex).ForEach([&](uint32_t vertex) {
                for (const auto edge : edges.Incident(vertex)) touched.push_back(edge);
            });
            Buffers.EdgeSharpness.Buffer.CaptureWriteElements(touched, sizeof(uint8_t));
            break;
        }
        case EditSharpnessOperation::SetAllFaces:
            CaptureElementSet(Buffers.FaceSharpness,Buffers.FaceTriangles,record.FaceData);
            break;
        case EditSharpnessOperation::SmoothAll:
        case EditSharpnessOperation::SmoothByAngle:
            CaptureElementSet(Buffers.FaceSharpness,Buffers.FaceTriangles,record.FaceData);
            CaptureElementSet(Buffers.EdgeSharpness,Buffers.EdgeHalfedges,record.EdgeData);
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
    ReleaseBlockLists(id);
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
    for (const auto id : Tracked->Entries.Trie.TakeChanged()) changed.push_back({uint32_t(id), EntryChanged});
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
                    const auto root = stream.Kind == HistoryState::StreamKind::Sets ? uint32_t(block) : block < blocks.size() ? blocks[block].Owner : InvalidOffset;
                    const auto it = owners.find(root);
                    if (it == owners.end()) continue;
                    Range vertices{};
                    if (stream.Bits == GeometryChanged) {
                        const auto start = std::max(first, b * MeshElementBlockSize);
                        const auto stop = std::min(last, (b + 1u) * MeshElementBlockSize);
                        vertices = {uint32_t(start), uint32_t(stop - start)};
                    }
                    const auto domain = stream.Kind == HistoryState::StreamKind::Sets ? InvalidOffset :
                        stream.Elements == Domain::Halfedge ? 3u : SelectableDomain(stream.Elements) ? SelectableIndex(stream.Elements) : InvalidOffset;
                    changed.push_back({it->second, stream.Bits, vertices, domain, uint32_t(block)});
                }
            });
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
        } else vertices.push_back(range.Vertices);
    }
    for (auto &change : changes)
        for (auto &blocks : change.Blocks) {
            std::ranges::sort(blocks);
            blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        }
    return changes;
}

void MeshStore::SyncMirrors() {
    const profile::CpuScope scope{"MeshStoreSyncMirrors"};
    Buffers.CornerSectors.ReserveBlocks(Buffers.FaceCorners.Capacity() / MeshElementBlockSize);
    Buffers.NormalSectors.ReserveBlocks(Buffers.FaceCorners.Capacity() / MeshElementBlockSize);
    Buffers.Query.Reserve(std::max({Buffers.Vertices.Capacity(), Buffers.EdgeHalfedges.Capacity(), Buffers.FaceTriangles.Capacity()}));
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Mirror) return;
        WithDomain(Buffers, info.Elements, [&](const auto &owner) {
            arena.Mirror({0, info.BlockIndexed ? owner.Capacity() / MeshElementBlockSize : owner.Capacity()});
        });
    });
    Buffers.SelectionRoots.Mirror({0, 3u * uint32_t(Records.size())});
    ForEachAttribute(Buffers, [&](auto &a, Domain domain, uint32_t, const char *, const char *, const char *, auto &&) {
        WithDomain(Buffers, domain, [&](const auto &owner) { a.ReserveBlocks(owner.Capacity() / MeshElementBlockSize); });
    });
}

void MeshStore::FinishRestore() {
    RenderStale.clear();
    for (const auto id : Tracked->Entries.Trie.ChangedSlots) Tracked->Dirty.push_back(uint32_t(id));
    for (const auto id : Tracked->Entries.Trie.ChangedSlots)
        if (id < DerivedRecords.size()) DerivedRecords[id] = {};
    DerivedRecords.resize(Records.size());
    for (const auto id : Tracked->Entries.Trie.ChangedSlots)
        if (id < Records.size() && Records[id].Alive) DerivedRecords[id].NormalRevision=++NextNormalRevision;
    // A restore can return a set to an earlier revision with different membership, so every list refills on its next read.
    {
        const std::scoped_lock lock{BlockListLock};
        ++BlockListEpoch;
    }
    SyncMirrors();
}

void MeshStore::FillBaseVertexNormalMirror(ElementSetRef vertices, Range point_normals) {
    if (point_normals.Count > 0) {
        std::ranges::copy(Buffers.PointNormals.Get(point_normals), Buffers.BaseVertexNormals.GetMutable(Buffers.Vertices.Dense(vertices)).begin());
    } else {
        ForEachBlockRun(Buffers.Vertices, vertices, [&](Range run) { std::ranges::fill(Buffers.BaseVertexNormals.GetMutable(run), vec3{0}); });
    }
}

ElementHandleRange MeshStore::InsertElements(uint32_t id, ElementDomain domain, uint32_t count, mtl::Buffer *list) {
    if (!Records.at(id).Alive || domain == Domain::None) throw std::invalid_argument("Invalid element insertion request.");
    if (!count) {
        if (list) list->SetUsedSize(0);
        return {};
    }
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
        if (owner == domain && (entries(record) || first_face)) attributes.Attach(blocks, {}, std::max(entries(record), 1u));
    });
    // The emitter writes the gained blocks and their mirrors, including holes sharing pages with surviving elements.
    ForEachArena(Buffers, [&](auto &values, const ArenaInfo &info, auto &&) {
        if (info.Elements != domain || info.ValueOwners || (info.BlockIndexed && !info.Mirror)) return;
        ForEachIndexRun(blocks, [&](size_t first, size_t count) {
            const Range run{blocks[first], uint32_t(count)};
            CaptureRange(values, info.BlockIndexed ? run : Range{run.Offset * MeshElementBlockSize, run.Count * MeshElementBlockSize});
        });
    });
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

void MeshStore::TrimInsertedElements(uint32_t id, ElementDomain domain, ElementHandleRange &inserted, const mtl::Buffer &list, uint32_t used) {
    if (used > inserted.Count) throw std::logic_error("An insertion's elements are fewer than the ones used.");
    if (used == inserted.Count) return;
    auto &record = WriteRecord(id);
    const auto blocks = WithDomain(Buffers, domain, [&](auto &arena) { return arena.Shrink(DomainSet(record, domain), inserted, list.GetSpan<uint32_t>(), used); });
    FinishEraseElements(id, domain, blocks);
}

void MeshStore::FinishEraseElements(uint32_t id, ElementDomain domain, std::span<const uint32_t> blocks) {
    if (blocks.empty()) return;
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
        const auto owner=Records[id].Vertices;
        std::vector<uvec2> released;
        for (const auto b : blocks) {
            const auto &block=Buffers.Vertices.Blocks.Get({b,1u})[0];
            const auto live=block.Owner==owner.Index ? block.Live : decltype(block.Live){};
            auto roots=Buffers.VertexCorners.GetMutable({b*256u,256u});
            for (uint32_t i=0u; i<256u; ++i) {
                if (live[i/32u] & (1u << (i%32u))) continue;
                if (roots[i].y) released.push_back(roots[i]);
                roots[i]={InvalidOffset,0u};
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

TetBuffers MeshStore::AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices) {
    return {Buffers.TetPositions.Allocate(positions), Buffers.TetEdgeIndices.Allocate(edge_indices)};
}

void MeshStore::ReleaseTets(TetBuffers tets) {
    Buffers.TetPositions.Release(tets.Positions);
    Buffers.TetEdgeIndices.Release(tets.EdgeIndices);
}

Range MeshStore::AllocateSoundVertices(std::span<const uint32_t> vertices) { return Buffers.SoundVertices.Allocate(vertices); }
void MeshStore::ReleaseSoundVertices(Range range) { Buffers.SoundVertices.Release(range); }
void MeshStore::EnsureSelectionState(state::Scene &r, std::span<const uint32_t> ids) {
    std::vector<SelectionUpdate> updates;
    for (const auto id : ids) if (!Records.at(id).SelectionSummary.Count) updates.push_back({.StoreId = id});
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
    UpdateSelection(r, updates);
    for (const auto &update : updates) PublishSelectionSummary(update.StoreId);
}

SelectionView MeshStore::GetSelectedElements(uint32_t id, Element element) const {
    const auto domain = SelectionDomain(element);
    const auto &root = GetSelectionRoot(id, element);
    return {SelectionArena(Buffers, domain).Buffer.GetSpan<uint32_t>(), GetBlockList(id, domain).Blocks,
            AggregateArena(Buffers, domain).Buffer.GetSpan<SelectionAggregate>(), root.Selected};
}

BoundaryEdgeView MeshStore::GetBoundaryEdges(uint32_t id) const {
    const auto &record = Records.at(id);
    if (!record.FaceData || !record.EdgeData) return {};
    return {
        GetBlockList(id, Domain::Edge).Blocks, Buffers.EdgeAggregates.Buffer.GetSpan<SelectionAggregate>(),
        Buffers.EdgeHalfedges.Blocks.Buffer.GetSpan<MeshElementBlock>(), Buffers.EdgeHalfedges.Buffer.GetSpan<uint32_t>(),
        Buffers.OppositeHalfedges.Buffer.GetSpan<uint32_t>(), record.EdgeData.Index,
    };
}

const SelectionAggregate &MeshStore::GetSelectionRoot(uint32_t id, Element element) const {
    return Buffers.SelectionRoots.Get({3u * id + SelectableIndex(SelectionDomain(element)), 1u})[0];
}
SlotOffset MeshStore::GetSelectionRoots(uint32_t id) const { return {Buffers.SelectionRoots.Buffer.Slot, 3u * id}; }

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

void MeshStore::SetSelectionBaseline(uint32_t id, Element element, std::vector<std::pair<uint32_t, MeshArenas::SelectionBlock>> blocks, uint32_t active) {
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
        summary.Count > 0 ? Buffers.SelectionSummary.Slotted(summary) : SlottedRange{}
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

void MeshStore::AllocateConnectivity(uint32_t id, uint32_t halfedge_count, uint32_t face_count, bool face_starts,
                                     std::span<const uint32_t> face_offsets,
                                     std::span<const std::array<uint32_t,2>> wire_edges) {
    auto &record = WriteRecord(id);
    record.ConnectivityFaceStarts = face_starts;
    if (!record.FaceCorners) record.FaceCorners = Buffers.FaceCorners.Allocate(halfedge_count);
    if (!record.FaceData) record.FaceData = Buffers.FaceTriangles.Allocate(face_count);
    if (!record.EdgeData) record.EdgeData = Buffers.EdgeHalfedges.Allocate(halfedge_count);
    SyncMirrors();
    if (!wire_edges.empty()) {
        if (face_count || uint64_t(wire_edges.size()) * 2u != halfedge_count) throw std::invalid_argument("Wire corner count differs from connectivity allocation.");
        auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        const uint32_t first = Buffers.Vertices.First(record.Vertices);
        for (uint32_t e = 0; e < wire_edges.size(); ++e) {
            corners[2u * e] = first + wire_edges[e][1];
            corners[2u * e + 1u] = first + wire_edges[e][0];
        }
    }
    if (!face_starts || face_offsets.empty()) return;
    const auto faces = Buffers.FaceRanges.GetMutable(Buffers.FaceTriangles.Dense(record.FaceData));
    const uint32_t first_corner = Buffers.FaceCorners.First(record.FaceCorners);
    for (uint32_t f = 0; f < face_count; ++f) faces[f] = {first_corner + face_offsets[f], first_corner + (f + 1 < face_count ? face_offsets[f + 1] : halfedge_count)};
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
        .HalfedgeFirst = Buffers.FaceCorners.First(r.FaceCorners), .HalfedgeCount = Buffers.FaceCorners.Count(r.FaceCorners),
        .EdgeFirst = Buffers.EdgeHalfedges.First(r.EdgeData), .FaceFirst = Buffers.FaceTriangles.First(r.FaceData),
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
    const std::scoped_lock lock{BlockListLock};
    if (id >= BlockListEntries.size()) return;
    for (auto &entry : BlockListEntries[id]) {
        RetireBlockListWords(entry.Words);
        entry = {};
    }
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
    for (const auto words : RetiredBlockLists) BlockLists.Release(words);
    RetiredBlockLists.clear();
}

MeshStore::BlockList MeshStore::GetBlockList(uint32_t id, ElementDomain domain) const {
    const std::scoped_lock lock{BlockListLock};
    const auto set = DomainSet(Records.at(id), domain);
    if (BlockListEntries.size() < Records.size()) BlockListEntries.resize(Records.size());
    auto &entry = BlockListEntries[id][uint32_t(domain) - 1u];
    WithDomain(Buffers, domain, [&](const auto &arena) {
        const auto revision = set ? arena.Set(set).Revision : 0u;
        if (entry.Set == set && entry.Revision == revision && entry.Epoch == BlockListEpoch) return;
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
        entry.Epoch = BlockListEpoch;
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
    ClearVertexRoots(Buffers,vertices);
    // A face mesh's corners are its face loops.
    // An edge mesh has no weld and receives its corners in CreateMesh.
    if (data.FaceCount() > 0) {
        auto &record = WriteRecord(id);
        record.FaceCorners = Buffers.FaceCorners.Allocate(uint32_t(data.FaceCorners.size()));
        const auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        const uint32_t first_vertex = Buffers.Vertices.First(record.Vertices);
        for (size_t h = 0; h < corners.size(); ++h) corners[h] = first_vertex + data.FaceCorners[h];
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
    ClearVertexRoots(Buffers,vertices);
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
        if (const auto *allocation = ranges(record)) arena.PlanAdditional(DenseRange(Buffers, info, *allocation).Count);
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
        record.FacePrimitivesReady = face_count!=0u;
        record.VertexPrimitivesReady = face_count==0u;
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
        uint32_t ti = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto n_tris = data.FaceSize(fi) - 2u;
            const auto first = corners.Offset + data.FaceStart(fi);
            for (uint32_t t = 0; t < n_tris; ++t) triangle_span[ti++] = {first, first + t + 1u, first + t + 2u};
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

void MeshStore::RetireLineConnectivity(state::Scene &r, uint32_t id) {
    const auto &record=Records.at(id);
    if (Mesh{*this,id}.FaceCount() || !Buffers.EdgeHalfedges.Count(record.EdgeData)) throw std::logic_error("Line retirement requires a face-less line mesh.");
    mtl::ComputeChain chain{BufferContext()};
    auto &storage=chain.Scratch;
    const std::array seeds{
        PrepareElementMembershipWork(storage,Buffers.EdgeHalfedges,record.EdgeData),
        PrepareElementMembershipWork(storage,Buffers.FaceCorners,record.FaceCorners),
    };
    const std::array work{seeds[0].Work,seeds[1].Work};
    EncodeElementMembershipWork(r,chain,seeds);
    EncodeSortElementWork(r,chain,work);
    chain.Submit();
    for (const auto &domain:work) CheckElementWork(storage,domain);

    // Only vertices incident to retired lines can hold line roots. A point
    // mesh can have many loose vertices outside these corner blocks.
    std::vector<uint32_t> vertex_blocks;
    ForEachWorkElement(storage,work[1],[&](uint32_t corner) {
        vertex_blocks.push_back(Buffers.FaceCorners.Get({corner,1u})[0]/MeshElementBlockSize);
    });
    std::ranges::sort(vertex_blocks);
    vertex_blocks.erase(std::unique(vertex_blocks.begin(),vertex_blocks.end()),vertex_blocks.end());
    Buffers.VertexCorners.Buffer.CaptureWriteElements(vertex_blocks,sizeof(uvec2)*MeshElementBlockSize);
    Buffers.OutgoingHalfedges.Buffer.CaptureWriteElements(vertex_blocks,sizeof(uint32_t)*MeshElementBlockSize);
    for (const auto block:vertex_blocks) {
        const auto roots=Buffers.VertexCorners.GetMutable({block*MeshElementBlockSize,MeshElementBlockSize});
        Buffers.VertexFans.Release(roots);
        std::ranges::fill(roots,uvec2{InvalidOffset,0u});
        std::ranges::fill(Buffers.OutgoingHalfedges.GetMutable({block*MeshElementBlockSize,MeshElementBlockSize}),InvalidOffset);
    }
    EraseElements(id,Domain::Edge,storage,work[0]);
    EraseElements(id,Domain::Halfedge,storage,work[1]);
    auto &writable=WriteRecord(id);
    writable.EdgeData={};
    writable.FaceCorners={};
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
    const auto id = AcquireId(std::move(record));
    CaptureSelectionSummary(id);
    return id;
}

uint32_t MeshStore::CloneMesh(const Mesh &mesh, const MeshPipelines &pipelines) {
    const auto src_id = mesh.GetStoreId();
    // The current packed clone emitter must reject fragmented input before it
    // acquires an output record. Local topology allocation does not use cloning.
    ForEachArena(Buffers, [&](auto &, const ArenaInfo &info, auto &&ranges) {
        if (const auto *allocation = ranges(Records.at(src_id)))
            DenseRange(Buffers, info, *allocation);
    });
    const auto id = AcquireId(Record{Records.at(src_id)});
    DerivedRecords[id] = DerivedRecords.at(src_id);
    DerivedRecords[id].NormalRevision = ++NextNormalRevision;
    const auto &src = Records[src_id];
    auto &dst = Records[id];
    auto &dst_derived = DerivedRecords[id];
    dst.SectorBlockCount=0u;
    struct Copy { mtl::Buffer *Buffer; uint64_t Source, Destination, Bytes; };
    std::vector<Copy> copies;
    // Allocate every destination before recording copies, so virtual growth and
    // history capture finish before the GPU reads either side.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (&arena.Buffer == &Buffers.VertexCorners.Buffer) return;
        const auto *source_allocation = ranges(src);
        if (!source_allocation) return;
        auto *target_allocation = ranges(dst);
        const auto source = DenseRange(Buffers, info, *source_allocation);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*target_allocation)>, ElementSetRef> && !requires { arena.Blocks; }) {
            WithDomain(Buffers, info.Elements, [&](const auto &owner) { arena.Mirror({0, info.BlockIndexed ? owner.Capacity() / MeshElementBlockSize : owner.Capacity()}); });
        } else {
            *target_allocation = arena.Allocate(source.Count);
        }
        const auto target = DenseRange(Buffers, info, *target_allocation);
        constexpr auto stride = sizeof(typename decltype(arena.Get(Range{}))::element_type);
        const uint64_t bytes = uint64_t(source.Count) * stride;
        if (!bytes) return;
        arena.Buffer.CaptureWrite(uint64_t(target.Offset) * stride, bytes);
        copies.push_back({&arena.Buffer, uint64_t(source.Offset) * stride, uint64_t(target.Offset) * stride, bytes});
    });
    const auto &ctx = BufferContext().Ctx;
    SyncMirrors();
    ClearVertexRoots(Buffers,dst.Vertices);
    for (const auto domain : SelectionDomains) {
        WithDomain(Buffers, domain, [&](const auto &owner) {
            const auto from = owner.Dense(DomainSet(src, domain));
            const auto to = owner.Dense(DomainSet(dst, domain));
            const auto count = ElementArena<uint32_t>::BlockCount(from.Count);
            if (!count) return;
            auto &bits = SelectionArena(Buffers, domain);
            const Range target{to.Offset / MeshElementBlockSize, count};
            CaptureRange(bits, target);
            copies.push_back({&bits.Buffer, uint64_t(from.Offset / MeshElementBlockSize) * sizeof(MeshArenas::SelectionBlock),
                              uint64_t(target.Offset) * sizeof(MeshArenas::SelectionBlock), uint64_t(count) * sizeof(MeshArenas::SelectionBlock)});
            // Equal block contents in the same order give the clone equal aggregates and roots.
            copies.push_back({&AggregateArena(Buffers, domain).Buffer, uint64_t(from.Offset / MeshElementBlockSize) * sizeof(SelectionAggregate),
                              uint64_t(target.Offset) * sizeof(SelectionAggregate), uint64_t(count) * sizeof(SelectionAggregate)});
        });
    }
    std::ranges::copy(Buffers.SelectionRoots.Get({3u * src_id, 3u}), Buffers.SelectionRoots.GetMutable({3u * id, 3u}).begin());
    // Authored layers and normal-sector layers clone through the same sparse payload ownership.
    // Allocate all destination payloads before reading addresses: growth may move an arena.
    const auto copy_attribute = [&](auto &attribute, Range source, uint32_t destination, uint32_t entries = 1u) {
        using Value = typename std::remove_cvref_t<decltype(attribute)>::Block::value_type;
        std::vector<uint32_t> blocks;
        for (uint32_t i = 0u; i < source.Count; i += MeshElementBlockSize)
            if (attribute.PayloadBlock((source.Offset + i) / MeshElementBlockSize))
                blocks.push_back((destination + i) / MeshElementBlockSize);
        attribute.Attach(blocks, {}, entries);
        for (const auto block : blocks) {
            const auto i = block * MeshElementBlockSize - destination;
            const auto count = std::min(MeshElementBlockSize, source.Count - i);
            for (uint32_t e = 0u; e < entries; ++e) {
                const auto from = attribute.Payload(source.Offset + i, count, e);
                const auto to = attribute.Payload(destination + i, count, e);
                const uint64_t bytes = uint64_t(count) * sizeof(Value);
                attribute.Values.Buffer.CaptureWrite(uint64_t(to.Offset) * sizeof(Value), bytes);
                copies.push_back({&attribute.Values.Buffer, uint64_t(from.Offset) * sizeof(Value), uint64_t(to.Offset) * sizeof(Value), bytes});
            }
        }
        return blocks;
    };
    ForEachAttribute(Buffers, [&](auto &attribute, Domain domain, uint32_t, const char *, const char *, const char *, auto &&entries) {
        if (const auto count = entries(src)) WithDomain(Buffers, domain, [&](const auto &arena) {
            copy_attribute(attribute, arena.Dense(DomainSet(src, domain)), arena.First(DomainSet(dst, domain)), count);
        });
    });
    const auto source_corners = Buffers.FaceCorners.Dense(src.FaceCorners);
    const Range sector_source{source_corners.Offset, ElementArena<uint32_t>::BlockCount(source_corners.Count) * MeshElementBlockSize};
    const auto destination_corners = Buffers.FaceCorners.First(dst.FaceCorners);
    const auto sector_targets = copy_attribute(Buffers.CornerSectors, sector_source, destination_corners);
    copy_attribute(Buffers.NormalSectors, sector_source, destination_corners);
    dst.SectorBlockCount = uint32_t(sector_targets.size());
    // The clone's fans fill one run in vertex order.
    const auto source_vertices=Buffers.Vertices.Dense(src.Vertices), target_vertices=Buffers.Vertices.Dense(dst.Vertices);
    const auto roots=Buffers.VertexCorners.Get(source_vertices);
    uint64_t fan_items=0u;
    for (const auto root : roots) fan_items+=root.y;
    if (fan_items>=InvalidOffset) throw std::length_error("Cloned fan items exceed their address space.");
    const auto fans=Buffers.VertexFans.Items.Allocate(uint32_t(fan_items));
    Buffers.VertexFans.Items.Buffer.CaptureWrite(uint64_t(fans.Offset)*sizeof(uvec2),uint64_t(fans.Count)*sizeof(uvec2));
    auto target_roots=Buffers.VertexCorners.GetMutable(target_vertices);
    std::vector<ReferencePairCopy> fan_ranges;
    for (uint32_t i=0u,next=fans.Offset; i<source_vertices.Count; ++i) {
        if (!roots[i].y) continue;
        target_roots[i]={next,roots[i].y};
        if (!fan_ranges.empty() && uint64_t(fan_ranges.back().Source)+fan_ranges.back().Count==roots[i].x) {
            fan_ranges.back().Count+=roots[i].y;
        } else fan_ranges.push_back({roots[i].x,next,roots[i].y});
        next+=roots[i].y;
    }
    auto *command = ctx.Queue->commandBuffer();
    ctx.OrderAfterGpuWork(command);
    auto *encoder = command->blitCommandEncoder();
    for (const auto &copy : copies) encoder->copyFromBuffer(**copy.Buffer, copy.Source, **copy.Buffer, copy.Destination, copy.Bytes);
    encoder->endEncoding();
    const auto &rebase_indices = pipelines[MeshPass::CloneRebaseIndices];
    EncodeRebaseIndices(command, rebase_indices, *Buffers.FaceCorners.Buffer, Buffers.FaceCorners.Dense(dst.FaceCorners), Buffers.Vertices.First(dst.Vertices) - Buffers.Vertices.First(src.Vertices));
    const auto corner_delta = Buffers.FaceCorners.First(dst.FaceCorners) - Buffers.FaceCorners.First(src.FaceCorners);
    EncodeCopyReferencePairs(command, pipelines[MeshPass::CloneCopyReferencePairs], *Buffers.VertexFans.Items.Buffer, fan_ranges, corner_delta,
        Buffers.FaceTriangles.First(dst.FaceData) - Buffers.FaceTriangles.First(src.FaceData));
    const auto rebase = [&](auto &arena, Range range, uint32_t delta, uint32_t stride = 1u) {
        EncodeRebaseIndices(command, rebase_indices, *arena.Buffer, range, delta, stride);
    };
    if (!dst_derived.SelectionBaseline.empty()) {
        const auto domain = SelectionDomain(dst_derived.SelectionBaselineElement);
        const auto delta = WithDomain(Buffers, domain, [&](const auto &owner) {
            return owner.First(DomainSet(dst, domain)) / MeshElementBlockSize - owner.First(DomainSet(src, domain)) / MeshElementBlockSize;
        });
        for (auto &[block, words] : dst_derived.SelectionBaseline) block += delta;
    }
    rebase(Buffers.OutgoingHalfedges, Buffers.Vertices.Dense(dst.Vertices), corner_delta);
    rebase(Buffers.OppositeHalfedges, Buffers.FaceCorners.Dense(dst.FaceCorners), corner_delta);
    rebase(Buffers.HalfedgeEdges, Buffers.FaceCorners.Dense(dst.FaceCorners), Buffers.EdgeHalfedges.First(dst.EdgeData) - Buffers.EdgeHalfedges.First(src.EdgeData));
    rebase(Buffers.HalfedgeFaces, Buffers.FaceCorners.Dense(dst.FaceCorners), Buffers.FaceTriangles.First(dst.FaceData) - Buffers.FaceTriangles.First(src.FaceData));
    rebase(Buffers.FaceTriangles, Buffers.FaceTriangles.Dense(dst.FaceData), Buffers.Triangles.First(dst.TriangleData) - Buffers.Triangles.First(src.TriangleData));
    for (uint32_t c = 0; c < 3u; ++c)
        EncodeRebaseIndices(command, rebase_indices, *Buffers.Triangles.Buffer, {c, Buffers.Triangles.Count(dst.TriangleData)}, corner_delta, 3u, uint64_t(Buffers.Triangles.First(dst.TriangleData)) * sizeof(uvec3));
    rebase(Buffers.EdgeHalfedges, Buffers.EdgeHalfedges.Dense(dst.EdgeData), corner_delta);
    rebase(Buffers.FaceRanges, {2u * Buffers.FaceTriangles.First(dst.FaceData), 2u * Buffers.FaceTriangles.Count(dst.FaceData)}, corner_delta);
    for (const auto block : sector_targets)
        rebase(Buffers.CornerSectors.Values, Buffers.CornerSectors.Payload(block * MeshElementBlockSize, MeshElementBlockSize), corner_delta);
    command->commit();
    command->waitUntilCompleted();
    if (command->status() == MTL::CommandBufferStatusError) throw std::runtime_error("GPU mesh clone failed.");
    return id;
}

void MeshStore::Release(uint32_t id) {
    if (id >= Records.size() || !Records[id].Alive) return;
    Buffers.Vertices.ForEachBlock(Records[id].Vertices,[&](uint32_t b, const auto &) {
        Buffers.VertexFans.Release(Buffers.VertexCorners.Get({b*MeshElementBlockSize,MeshElementBlockSize}));
    });
    // Released blocks name no fan runs, so their next owner inherits empty roots.
    ClearVertexRoots(Buffers,Records[id].Vertices);
    auto &record = WriteRecord(id);
    auto &derived = DerivedRecords.at(id);
    Buffers.FaceCorners.ForEachBlock(record.FaceCorners, [&](uint32_t block, const auto &) {
        Buffers.CornerSectors.Release(block);
        Buffers.NormalSectors.Release(block);
    });
    // Released blocks hold no selection, so every dead slot's mask bit is clear for the blocks' next owner.
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto masks = SelectionArena(Buffers, SelectionDomains[d]).Buffer.template GetSpan<MeshArenas::SelectionBlock>();
        std::vector<uint32_t> selected;
        for (const auto block : GetBlockList(id, SelectionDomains[d]).Blocks)
            if (block < masks.size() && std::ranges::any_of(masks[block], [](uint32_t word) { return word != 0u; })) selected.push_back(block);
        EditSelectionBlocks(SelectionElements[d], selected, [](uint32_t, auto &words) { words = {}; });
    }
    ReleaseBlockLists(id);
    std::ranges::fill(Buffers.SelectionRoots.GetMutable({3u * id, 3u}), SelectionAggregate{});
    ForEachAttribute(Buffers, [&](auto &attributes, Domain domain, uint32_t, const char *, const char *, const char *, auto &&entries) {
        if (!entries(record)) return;
        const auto set = DomainSet(record, domain);
        WithDomain(Buffers, domain, [&](const auto &arena) {
            arena.ForEachBlock(set, [&](uint32_t block, const auto &) { attributes.Release(block); });
        });
    });
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (info.Mirror) return;
        if (const auto *range = ranges(record))
            if constexpr (requires { arena.Release(*range); }) arena.Release(*range);
    });
    record = {};
    derived = {};
    RenderStale.push_back(id);
    if (Tracked) Tracked->Free.Write(FreeIds.size(), 1);
    FreeIds.emplace_back(id);
}

void MeshStore::Clear() {
    if (Tracked) {
        Tracked->Entries.Write(0, Records.size());
        Tracked->Free.Write(0, FreeIds.size());
        Tracked->AllDirty = true;
    }
    Buffers.VertexFans.Reset();
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.Reset(); });
    Records.clear();
    DerivedRecords.clear();
    {
        const std::scoped_lock lock{BlockListLock};
        BlockLists.Reset();
        BlockListEntries.clear();
        RetiredBlockLists.clear();
    }
    FreeIds.clear();
    RenderStale.clear();
}

VertexEdgeIncidence MeshStore::GetVertexEdgeIncidence(uint32_t id) const {
    return {GetConnectivity(id)};
}

uint32_t MeshStore::AcquireId(Record &&record) {
    if (!FreeIds.empty()) {
        const auto reused = FreeIds.back();
        if (Tracked) Tracked->Free.Write(FreeIds.size() - 1, 1);
        FreeIds.pop_back();
        WriteRecord(reused) = std::move(record);
        std::ranges::fill(Buffers.SelectionRoots.GetMutable({3u * reused, 3u}), SelectionAggregate{});
        return reused;
    }
    if (Tracked) {
        Tracked->Entries.Write(Records.size(), 1);
        Tracked->Dirty.push_back(uint32_t(Records.size()));
    }
    Records.emplace_back(std::move(record));
    DerivedRecords.emplace_back();
    const auto id = uint32_t(Records.size() - 1);
    Buffers.SelectionRoots.Mirror({0, 3u * uint32_t(Records.size())});
    std::ranges::fill(Buffers.SelectionRoots.GetMutable({3u * id, 3u}), SelectionAggregate{});
    return id;
}
