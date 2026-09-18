#include "numeric/uvec2.h"
#include "numeric/vec2.h"

#include "MeshStore.h"

#include "CornerNormalOffset.h"
#include "Profile.h"
#include "ScratchChunks.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/FanItemEncoding.h"
#include "project/store/History.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include <bit>
#include <format>

namespace {
constexpr uint32_t ClassTagShift{uint32_t(CornerClassEncoding::TagShift)}, ClassIndexMask{uint32_t(CornerClassEncoding::IndexMask)};
constexpr uint32_t UniformFaceOffset{uint32_t(CornerClassEncoding::UniformFaceOffset)};
constexpr uint32_t FanLoopShift{uint32_t(FanItemEncoding::LoopShift)};

constexpr uint32_t ElementIndex(Element element) {
    return element == Element::Vertex ? 0u : element == Element::Edge ? 1u :
                                                                        2u;
}

constexpr bool IsSharp(uint8_t sharpness) { return sharpness != 0; }

// Outgoing halfedges, opposites, the two bit tables, the samples at their bound, then an n-gon mesh's face starts.
constexpr uint32_t ConnectivityWords(uint32_t vertices, uint32_t halfedges, uint32_t faces, bool face_starts) {
    return vertices + halfedges + 3 * BitWords(halfedges) + (face_starts ? faces : 0u);
}
// CSR offsets then one item per halfedge, since every halfedge of a face-topology mesh belongs to a face loop.
constexpr uint32_t FanAdjacencyWords(uint32_t vertices, uint32_t halfedges) { return vertices + 1 + halfedges; }
// CSR offsets then two items per edge, one per endpoint.
constexpr uint32_t EdgeAdjacencyWords(uint32_t vertices, uint32_t edges) { return vertices + 1 + 2 * edges; }

template<typename... R> auto Ptrs(R &...ranges) { return std::array{&ranges...}; }
auto ArrayPtrs(auto &ranges) {
    return std::apply([](auto &...r) { return std::array{&r...}; }, ranges);
}
constexpr auto NoRanges = [](auto &, auto &) { return std::array<Range *, 0>{}; };

// The change bits a tracked arena reports and its history name, with a derived arena left unnamed.
// A mirror arena shares its ranges with a master arena earlier in the roster and allocates nothing of its own.
struct ArenaInfo {
    uint32_t Bits{};
    const char *Name{};
    bool Mirror{false};
    bool Tracked() const { return Name != nullptr; }
};

// The arena roster: calls f(arena, info, ranges) for every arena, where ranges(record, derived) returns pointers to the ranges a mesh holds in it.
// Tracked arenas come first in their history order, then the derived arenas and the mirrors, each after its master.
void ForEachArena(MeshArenas &b, auto &&f) {
    using enum MeshStore::ChangeBits;
    f(b.Vertices, ArenaInfo{GeometryChanged, "Vertices"}, [](auto &e, auto &) { return Ptrs(e.Vertices); });
    f(b.FaceFirstTriangles, ArenaInfo{TopologyChanged, "FaceFirstTriangle"}, [](auto &e, auto &) { return Ptrs(e.FaceData); });
    f(b.ElementPrimitives, ArenaInfo{AttributesChanged, "ElementPrimitive"}, [](auto &e, auto &) { return Ptrs(e.ElementPrimitives); });
    f(b.PrimitiveMaterials, ArenaInfo{AttributesChanged, "PrimitiveMaterial"}, [](auto &e, auto &) { return Ptrs(e.PrimitiveMaterials); });
    f(b.BoneDeform, ArenaInfo{DeformChanged, "BoneDeform"}, [](auto &e, auto &) { return Ptrs(e.BoneDeform); });
    f(b.MorphTargets, ArenaInfo{DeformChanged, "MorphTarget"}, [](auto &e, auto &) { return Ptrs(e.MorphTargets); });
    f(b.SelectionBits, ArenaInfo{SelectionChanged, "SelectionBits"}, [](auto &e, auto &) { return ArrayPtrs(e.SelectionBits); });
    f(b.SelectionSummary, ArenaInfo{SelectionChanged, "SelectionSummary"}, [](auto &e, auto &) { return Ptrs(e.SelectionSummary); });
    f(b.TriangleFaceIds, ArenaInfo{TopologyChanged, "TriangleFaceId"}, [](auto &e, auto &) { return Ptrs(e.TriangleFaceIds); });
    f(b.FaceCorners, ArenaInfo{TopologyChanged, "FaceCorner"}, [](auto &e, auto &) { return Ptrs(e.FaceCorners); });
    f(b.Connectivity, ArenaInfo{TopologyChanged, "Connectivity"}, [](auto &e, auto &) { return Ptrs(e.Connectivity, e.ConnectivityEdges, e.ConnectivityHalfedgeToEdge); });
    f(b.EdgeSharpness, ArenaInfo{ShadingChanged, "EdgeSharpness"}, [](auto &e, auto &) { return Ptrs(e.EdgeSharpness); });
    f(b.CustomCornerMasks, ArenaInfo{ShadingChanged, "CustomCornerMask"}, [](auto &e, auto &) { return Ptrs(e.CustomCornerMasks); });
    f(b.CustomCornerNormals, ArenaInfo{ShadingChanged, "CustomCornerNormal"}, [](auto &e, auto &) { return Ptrs(e.CustomCornerNormals); });
    f(b.CornerTangents, ArenaInfo{AttributesChanged, "CornerTangent"}, [](auto &e, auto &) { return Ptrs(e.CornerTangents); });
    f(b.CornerColors, ArenaInfo{AttributesChanged, "CornerColor"}, [](auto &e, auto &) { return Ptrs(e.CornerColors); });
    f(b.CornerUvs, ArenaInfo{AttributesChanged, "CornerUv"}, [](auto &e, auto &) { return ArrayPtrs(e.CornerUvs); });
    f(b.PointNormals, ArenaInfo{ShadingChanged, "PointNormal"}, [](auto &e, auto &) { return Ptrs(e.PointNormals); });
    f(b.TetPositions, ArenaInfo{0, "TetPosition"}, NoRanges);
    f(b.TetEdgeIndices, ArenaInfo{0, "TetEdgeIndex"}, NoRanges);
    f(b.FaceSharpness, ArenaInfo{ShadingChanged, "FaceSharpness", true}, [](auto &e, auto &) { return Ptrs(e.FaceData); });
    f(b.SoundVertices, ArenaInfo{}, NoRanges);
    f(b.Adjacency, ArenaInfo{}, [](auto &, auto &d) { return Ptrs(d.VertexFanAdjacency, d.VertexEdgeAdjacency, d.SeamFans); });
    f(b.CornerClasses, ArenaInfo{}, [](auto &, auto &d) { return Ptrs(d.CornerClasses); });
    f(b.BaseSeamNormals, ArenaInfo{}, [](auto &, auto &d) { return Ptrs(d.BaseSeamNormals); });
    f(b.SelectionBaseline, ArenaInfo{}, [](auto &, auto &d) { return Ptrs(d.SelectionBaseline); });
    f(b.BaseVertexNormals, ArenaInfo{.Mirror = true}, [](auto &e, auto &) { return Ptrs(e.Vertices); });
    f(b.BaseFaceNormals, ArenaInfo{.Mirror = true}, [](auto &e, auto &) { return Ptrs(e.FaceData); });
}

template<typename Arena> using ArenaValue = typename decltype(std::declval<const Arena &>().Get(Range{}))::value_type;
} // namespace

MeshArenas::MeshArenas(mtl::BufferContext &ctx)
    : Vertices{ctx, SlotType::VertexBuffer},
      FaceFirstTriangles{ctx, SlotType::ObjectIdBuffer},
      FaceSharpness{ctx, SlotType::Buffer},
      FaceCorners{ctx, SlotType::IndexBuffer},
      TriangleFaceIds{ctx, SlotType::ObjectIdBuffer},
      Connectivity{ctx, SlotType::Buffer},
      SelectionBits{ctx, SlotType::Buffer},
      SelectionSummary{ctx, SlotType::Buffer},
      EdgeSharpness{ctx, SlotType::Buffer},
      CustomCornerMasks{ctx, SlotType::Buffer},
      CustomCornerNormals{ctx, SlotType::Buffer},
      PointNormals{ctx, SlotType::Buffer},
      CornerTangents{ctx, SlotType::CornerTangentBuffer},
      CornerColors{ctx, SlotType::CornerColorBuffer},
      CornerUvs{ctx, SlotType::CornerUvBuffer},
      ElementPrimitives{ctx, SlotType::ElementPrimitiveBuffer},
      PrimitiveMaterials{ctx, SlotType::PrimitiveMaterialBuffer},
      BoneDeform{ctx, SlotType::BoneDeformBuffer},
      MorphTargets{ctx, SlotType::MorphTargetBuffer},
      TetPositions{ctx, SlotType::Buffer},
      TetEdgeIndices{ctx, SlotType::Buffer},
      SoundVertices{ctx, SlotType::Buffer},
      Adjacency{ctx, SlotType::Buffer},
      CornerClasses{ctx, SlotType::Buffer},
      BaseSeamNormals{ctx, SlotType::Buffer},
      SelectionBaseline{ctx, SlotType::Buffer},
      BaseVertexNormals{ctx, SlotType::Buffer},
      BaseFaceNormals{ctx, SlotType::Buffer} {}

struct MeshStore::HistoryState {
    struct Extent {
        uint64_t Begin, End;
        uint32_t Id, Bits;
    };
    store::Records Entries, Free;
    std::unordered_map<mtl::Buffer *, std::vector<Extent>> Ranges;
    bool RangesDirty{true};

    HistoryState(MeshStore &mesh, store::History &history) : Entries(mesh.Records), Free(mesh.FreeIds) {
        Entries.Trie.CollectChanged = true;
        history.Track(Entries, "mesh.entries", 0);
        history.Track(Free, "mesh.free", 0);
    }

    void Index(MeshStore &mesh) {
        for (auto &[buffer, ranges] : Ranges) ranges.clear();
        for (uint32_t id = 0; id < mesh.Records.size(); ++id) {
            const auto &record = mesh.Records[id];
            if (!record.Alive) continue;
            const auto &derived = mesh.DerivedRecords[id];
            ForEachArena(mesh.Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
                if (!info.Tracked()) return;
                constexpr auto Stride = sizeof(ArenaValue<std::remove_cvref_t<decltype(arena)>>);
                for (const auto *range : ranges(record, derived)) {
                    if (range->Count) Ranges[&arena.Buffer].push_back({uint64_t(range->Offset) * Stride, uint64_t(range->Offset + range->Count) * Stride, id, info.Bits});
                }
            });
        }
        for (auto &[buffer, ranges] : Ranges) std::ranges::sort(ranges, {}, &Extent::Begin);
        RangesDirty = false;
    }
};

MeshStore::MeshStore(mtl::BufferContext &ctx)
    : Buffers{ctx},
      SlotTable{
          .Vertices = Buffers.Vertices.Buffer.Slot,
          .FaceFirstTriangle = Buffers.FaceFirstTriangles.Buffer.Slot,
          .FaceSharpness = Buffers.FaceSharpness.Buffer.Slot,
          .SelectionBits = Buffers.SelectionBits.Buffer.Slot,
          .EdgeSharpness = Buffers.EdgeSharpness.Buffer.Slot,
          .CustomCornerMask = Buffers.CustomCornerMasks.Buffer.Slot,
          .CustomCornerNormal = Buffers.CustomCornerNormals.Buffer.Slot,
          .CornerTangent = Buffers.CornerTangents.Buffer.Slot,
          .CornerColor = Buffers.CornerColors.Buffer.Slot,
          .CornerUv = Buffers.CornerUvs.Buffer.Slot,
          .ElementPrimitive = Buffers.ElementPrimitives.Buffer.Slot,
          .PrimitiveMaterial = Buffers.PrimitiveMaterials.Buffer.Slot,
          .BoneDeform = Buffers.BoneDeform.Buffer.Slot,
          .MorphTarget = Buffers.MorphTargets.Buffer.Slot,
          .TetPosition = Buffers.TetPositions.Buffer.Slot,
          .TetEdgeIndex = Buffers.TetEdgeIndices.Buffer.Slot,
          .SoundVertex = Buffers.SoundVertices.Buffer.Slot,
          .Adjacency = Buffers.Adjacency.Buffer.Slot,
          .CornerClass = Buffers.CornerClasses.Buffer.Slot,
          .BaseSeamNormal = Buffers.BaseSeamNormals.Buffer.Slot,
          .BaseVertexNormal = Buffers.BaseVertexNormals.Buffer.Slot,
          .BaseFaceNormal = Buffers.BaseFaceNormals.Buffer.Slot,
      } {}
MeshStore::~MeshStore() = default;

void MeshStore::Track(store::History &history) {
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (!info.Tracked()) return;
        // A mirror owns no allocator, so only its bytes carry history.
        if (info.Mirror) arena.Buffer.Track(history, std::string{"mesh."} + info.Name);
        else arena.Track(history, std::string{"mesh."} + info.Name);
    });
    Tracked = std::make_unique<HistoryState>(*this, history);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&) {
        if (info.Tracked()) Tracked->Ranges.try_emplace(&arena.Buffer);
    });
}

namespace {
void CaptureRange(const auto &arena, Range range) {
    using Value = ArenaValue<std::remove_cvref_t<decltype(arena)>>;
    arena.Buffer.CaptureWrite(uint64_t(range.Offset) * sizeof(Value), uint64_t(range.Count) * sizeof(Value));
}

// Requires selected indices in ascending order.
void CaptureSelected(const mtl::Buffer &buffer, Range range, uint32_t stride, std::span<const uint32_t> bits) {
    const auto *history = buffer.History();
    if (!history) return;
    const uint64_t page_size = history->PageBytes;
    uint64_t first = 0, end = 0;
    ForEachSelected(bits, range.Count, [&](uint32_t index) {
        const uint64_t offset = uint64_t(range.Offset + index) * stride;
        const auto page = offset / page_size, last = (offset + stride + page_size - 1) / page_size;
        if (first != end && page > end) {
            buffer.CaptureWrite(first * page_size, (end - first) * page_size);
            first = end = 0;
        }
        if (first == end) first = page;
        end = last;
    });
    if (first != end) buffer.CaptureWrite(first * page_size, (end - first) * page_size);
}
} // namespace

void MeshStore::CaptureVertexEdit(uint32_t id) {
    if (!Tracked) return;
    CaptureSelected(Buffers.Vertices.Buffer, Records.at(id).Vertices, sizeof(Vertex), GetSelectionBits(id, Element::Vertex));
}

void MeshStore::CaptureSelectionWrite(uint32_t id) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    for (const auto range : record.SelectionBits) CaptureRange(Buffers.SelectionBits, range);
    CaptureRange(Buffers.SelectionSummary, record.SelectionSummary);
}

void MeshStore::CaptureSharpnessWrite(uint32_t id, EditSharpnessOperation operation) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    switch (operation) {
        case EditSharpnessOperation::SetSelectedFaces:
            CaptureSelected(Buffers.FaceSharpness.Buffer, record.FaceData, 1, GetSelectionBits(id, Element::Face));
            break;
        case EditSharpnessOperation::SetSelectedEdges:
            CaptureSelected(Buffers.EdgeSharpness.Buffer, record.EdgeSharpness, 1, GetSelectionBits(id, Element::Edge));
            break;
        case EditSharpnessOperation::SetVertexEdges: {
            const auto edges = GetVertexEdgeAdjacency(id);
            ForEachSelected(GetSelectionBits(id, Element::Vertex), record.Vertices.Count, [&](uint32_t vertex) {
                for (const auto edge : edges.Incident(vertex)) Buffers.EdgeSharpness.Buffer.CaptureWrite(record.EdgeSharpness.Offset + edge, 1);
            });
            break;
        }
        case EditSharpnessOperation::SetAllFaces:
            CaptureRange(Buffers.FaceSharpness, record.FaceData);
            break;
        case EditSharpnessOperation::SmoothAll:
        case EditSharpnessOperation::SmoothByAngle:
            CaptureRange(Buffers.FaceSharpness, record.FaceData);
            CaptureRange(Buffers.EdgeSharpness, record.EdgeSharpness);
            break;
    }
}

void MeshStore::CaptureConnectivityWrite(uint32_t id) {
    if (Tracked) CaptureRange(Buffers.Connectivity, Records.at(id).Connectivity);
}

void MeshStore::CaptureWeldWrite(uint32_t id) {
    if (!Tracked) return;
    const auto &record = Records.at(id);
    CaptureRange(Buffers.Vertices, record.Vertices);
    CaptureRange(Buffers.FaceCorners, record.FaceCorners);
    CaptureRange(Buffers.BoneDeform, record.BoneDeform);
    CaptureRange(Buffers.MorphTargets, record.MorphTargets);
}

MeshStore::Record &MeshStore::WriteRecord(uint32_t id) {
    if (Tracked) {
        Tracked->Entries.Write(id, 1);
        Tracked->RangesDirty = true;
    }
    return Records.at(id);
}

std::vector<MeshStore::Change> MeshStore::TakeChanges() {
    if (!Tracked) return {};
    if (Tracked->RangesDirty) Tracked->Index(*this);
    struct ChangedRange {
        uint32_t Id, Bits;
        Range Vertices{};
    };
    std::vector<ChangedRange> changed;
    for (const auto id : Tracked->Entries.Trie.TakeChanged()) changed.push_back({uint32_t(id), EntryChanged});
    for (const auto &[buffer, ranges] : Tracked->Ranges) {
        const auto page_size = buffer->History()->PageBytes;
        for (const auto page : buffer->History()->Trie.TakeChanged()) {
            const uint64_t begin = page * page_size, end = begin + page_size;
            auto it = std::ranges::upper_bound(ranges, begin, {}, &HistoryState::Extent::End);
            for (; it != ranges.end() && it->Begin < end; ++it) {
                Range vertices{};
                if (it->Bits == GeometryChanged) {
                    const auto first = (std::max(begin, it->Begin) - it->Begin) / sizeof(Vertex);
                    const auto last = (std::min(end, it->End) - it->Begin + sizeof(Vertex) - 1) / sizeof(Vertex);
                    vertices = {uint32_t(first), uint32_t(last - first)};
                }
                changed.push_back({it->Id, it->Bits, vertices});
            }
        }
    }
    std::ranges::sort(changed, [](const auto &a, const auto &b) { return std::pair{a.Id, a.Vertices.Offset} < std::pair{b.Id, b.Vertices.Offset}; });
    std::vector<Change> changes;
    for (const auto &range : changed) {
        if (changes.empty() || changes.back().StoreId != range.Id) changes.push_back({range.Id, 0});
        auto &change = changes.back();
        change.Bits |= range.Bits;
        if (!range.Vertices.Count) continue;
        auto &vertices = change.VertexRanges;
        if (!vertices.empty() && range.Vertices.Offset <= vertices.back().Offset + vertices.back().Count) {
            vertices.back().Count = std::max(vertices.back().Offset + vertices.back().Count, range.Vertices.Offset + range.Vertices.Count) - vertices.back().Offset;
        } else vertices.push_back(range.Vertices);
    }
    return changes;
}

void MeshStore::SyncMirrors(uint32_t id) {
    auto &record = Records.at(id);
    auto &derived = DerivedRecords.at(id);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (!info.Mirror) return;
        for (const auto *range : ranges(record, derived)) arena.Mirror(*range);
    });
}

void MeshStore::FinishRestore() {
    if (!Tracked->Entries.Trie.ChangedSlots.empty()) Tracked->RangesDirty = true;
    for (const auto id : Tracked->Entries.Trie.ChangedSlots) {
        if (id < DerivedRecords.size()) ReleaseDerived(id);
    }
    DerivedRecords.resize(Records.size());
    for (const auto id : Tracked->Entries.Trie.ChangedSlots) {
        if (id >= Records.size() || !Records[id].Alive) continue;
        SyncMirrors(id);
        FillBaseVertexNormalMirror(Records[id].Vertices, Records[id].PointNormals);
    }
}

void MeshStore::FillBaseVertexNormalMirror(Range vertices, Range point_normals) {
    const auto normals = Buffers.BaseVertexNormals.GetMutable(vertices);
    if (point_normals.Count > 0) std::ranges::copy(Buffers.PointNormals.Get(point_normals), normals.begin());
    else std::ranges::fill(normals, vec3{0});
}

// Derived arena offsets follow rebuild order, so sort by store id for a deterministic layout.
void MeshStore::RebuildDerived(std::span<Mesh> meshes) {
    std::ranges::sort(meshes, {}, &Mesh::GetStoreId);
    for (const auto &mesh : meshes) {
        ReleaseDerived(mesh.GetStoreId());
        BuildVertexAdjacency(mesh);
        UpdateCornerClassification(mesh);
    }
}

uint32_t MeshStore::GetCornerClassOffset(uint32_t id) const {
    const auto &derived = DerivedRecords.at(id);
    if (derived.CornerClasses.Count > 0) return derived.CornerClasses.Offset;
    return derived.UniformCornerClass == CornerClass::Face ? UniformFaceOffset : InvalidOffset;
}

std::span<Vertex> MeshStore::EditVertices(uint32_t id) { return Buffers.Vertices.GetMutable(Records.at(id).Vertices); }
std::span<uint32_t> MeshStore::EditPrimitiveMaterials(uint32_t id) { return Buffers.PrimitiveMaterials.GetMutable(Records.at(id).PrimitiveMaterials); }
std::span<uint8_t> MeshStore::EditFaceSharpness(uint32_t id) { return Buffers.FaceSharpness.GetMutable(Records.at(id).FaceData); }
std::span<uint8_t> MeshStore::EditEdgeSharpness(uint32_t id) { return Buffers.EdgeSharpness.GetMutable(Records.at(id).EdgeSharpness); }

void MeshStore::SetCustomCornerNormals(uint32_t id, std::span<const uvec2> masks, std::span<const vec2> packed) {
    auto &record = WriteRecord(id);
    Buffers.CustomCornerMasks.Release(record.CustomCornerMasks);
    Buffers.CustomCornerNormals.Release(record.CustomCornerNormals);
    record.CustomCornerMasks = Buffers.CustomCornerMasks.Allocate(masks);
    record.CustomCornerNormals = Buffers.CustomCornerNormals.Allocate(packed);
}

void MeshStore::SetMorphShadingAuthored(uint32_t id, bool authored) { DerivedRecords.at(id).MorphShadingAuthored = authored; }

TetBuffers MeshStore::AllocateTets(std::span<const vec3> positions, std::span<const uint32_t> edge_indices) {
    return {Buffers.TetPositions.Allocate(positions), Buffers.TetEdgeIndices.Allocate(edge_indices)};
}

void MeshStore::ReleaseTets(TetBuffers tets) {
    Buffers.TetPositions.Release(tets.Positions);
    Buffers.TetEdgeIndices.Release(tets.EdgeIndices);
}

Range MeshStore::AllocateSoundVertices(std::span<const uint32_t> vertices) { return Buffers.SoundVertices.Allocate(vertices); }
void MeshStore::ReleaseSoundVertices(Range range) { Buffers.SoundVertices.Release(range); }

void MeshStore::EnsureSelectionBits(const Mesh &mesh) {
    auto &record = Records.at(mesh.GetStoreId());
    auto &derived = DerivedRecords.at(mesh.GetStoreId());
    const std::array counts{mesh.VertexCount(), mesh.EdgeCount(), mesh.FaceCount()};
    bool domains_sized = true;
    for (uint32_t i = 0; i < counts.size(); ++i) {
        domains_sized &= record.SelectionBits[i].Count >= BitWords(counts[i]);
    }
    if (!domains_sized || record.SelectionSummary.Count == 0) WriteRecord(mesh.GetStoreId());
    if (!domains_sized) {
        for (const auto range : record.SelectionBits) Buffers.SelectionBits.Release(range);
        for (uint32_t i = 0; i < counts.size(); ++i) {
            record.SelectionBits[i] = Buffers.SelectionBits.Allocate(BitWords(counts[i]));
            std::ranges::fill(Buffers.SelectionBits.GetMutable(record.SelectionBits[i]), 0u);
        }
        if (record.SelectionSummary.Count > 0) Buffers.SelectionSummary.GetMutable(record.SelectionSummary)[0] = {};
    }
    // One trailing word preserves the active handle alongside the largest authoritative domain.
    const uint32_t baseline_words = BitWords(std::ranges::max(counts)) + 1u;
    if (derived.SelectionBaseline.Count < baseline_words) {
        Buffers.SelectionBaseline.Release(derived.SelectionBaseline);
        derived.SelectionBaseline = Buffers.SelectionBaseline.Allocate(baseline_words);
        std::ranges::fill(Buffers.SelectionBaseline.GetMutable(derived.SelectionBaseline), 0u);
    }
    if (record.SelectionSummary.Count == 0) {
        record.SelectionSummary = Buffers.SelectionSummary.Allocate(1);
        Buffers.SelectionSummary.GetMutable(record.SelectionSummary)[0] = {};
    }
}

std::span<const uint32_t> MeshStore::GetSelectionBits(uint32_t id, Element element) const { return Buffers.SelectionBits.Get(Records.at(id).SelectionBits[ElementIndex(element)]); }
uint32_t MeshStore::GetSelectionBitOffset(uint32_t id, Element element) const { return Records.at(id).SelectionBits[ElementIndex(element)].Offset * 32; }
SlottedRange MeshStore::GetSelectionBitsRange(uint32_t id, Element element) const {
    const auto range = Records.at(id).SelectionBits[ElementIndex(element)];
    return range.Count > 0 ? Buffers.SelectionBits.Slotted(range) : SlottedRange{};
}
SlottedRange MeshStore::GetSelectionBaselineRange(uint32_t id) const {
    const auto range = DerivedRecords.at(id).SelectionBaseline;
    return range.Count > 0 ? Buffers.SelectionBaseline.Slotted(range) : SlottedRange{};
}
EditSelectionStorage MeshStore::GetEditSelectionStorage(uint32_t id) const {
    const auto summary = Records.at(id).SelectionSummary;
    return {
        GetSelectionBitsRange(id, Element::Vertex), GetSelectionBitsRange(id, Element::Edge), GetSelectionBitsRange(id, Element::Face),
        summary.Count > 0 ? Buffers.SelectionSummary.Slotted(summary) : SlottedRange{}
    };
}
const EditSelectionSummary &MeshStore::GetSelectionSummary(uint32_t id) const { return Buffers.SelectionSummary.Get(Records.at(id).SelectionSummary)[0]; }

namespace {
VertexAdjacency SliceAdjacency(std::span<const uint32_t> words, uint32_t bucket_count) {
    if (words.empty()) return {};
    return {words.first(bucket_count + 1), words.subspan(bucket_count + 1)};
}
} // namespace

vec3 ComposeCornerNormal(std::span<const uint32_t> classes, CornerClass uniform_class, uint32_t ci, std::span<const uint32_t> indices, std::span<const uint32_t> face_ids, const CornerNormalSources &sources) {
    const auto value = classes.empty() ? uint32_t(uniform_class) << ClassTagShift : classes[ci];
    switch (CornerClass(value >> ClassTagShift)) {
        case CornerClass::Face: return sources.FaceNormals[face_ids[ci / 3] - 1];
        case CornerClass::Seam: return sources.SeamNormals[value & ClassIndexMask];
        default: return sources.VertexNormals[indices[ci]];
    }
}

void MeshStore::UpdateCornerClassification(const Mesh &mesh) {
    const profile::CpuScope scope{"CornerClassification"};
    const auto id = mesh.GetStoreId();
    auto &record = Records.at(id);
    auto &derived = DerivedRecords.at(id);
    if (record.TriangleCount == 0) return;
    const auto sharp_faces = Buffers.FaceSharpness.Get(record.FaceData);
    const auto sharp_edges = Buffers.EdgeSharpness.Get(record.EdgeSharpness);
    const auto [any_face_sharp, all_faces_sharp] = GetFaceSharpnessSummary(id);
    const bool any_sharp = any_face_sharp || std::ranges::any_of(sharp_edges, IsSharp);
    // Uniform classification avoids allocating a per-corner class buffer.
    if (!any_sharp || all_faces_sharp) {
        derived.UniformCornerClass = all_faces_sharp ? CornerClass::Face : CornerClass::Vertex;
        Buffers.CornerClasses.Release(derived.CornerClasses);
        Buffers.Adjacency.Release(derived.SeamFans);
        Buffers.BaseSeamNormals.Release(derived.BaseSeamNormals);
        derived.CornerClasses = derived.SeamFans = derived.BaseSeamNormals = {};
        derived.SeamCornerCount = 0;
        return;
    }
    if (derived.CornerClasses.Count == 0) derived.CornerClasses = Buffers.CornerClasses.Allocate(record.TriangleCount * 3);
    const auto classes = Buffers.CornerClasses.GetMutable(derived.CornerClasses);
    const auto &c = mesh.GetConnectivity();
    const auto face_sharp = [&](Mesh::FH fh) { return *fh < sharp_faces.size() && IsSharp(sharp_faces[*fh]); };
    const auto edge_sharp = [&](Mesh::HH hh) { const auto eh = mesh.GetEdge(hh); return *eh < sharp_edges.size() && IsSharp(sharp_edges[*eh]); };

    // Assign seam sectors to vertices at a discontinuity and the vertex normal to all other vertices.
    static thread_local std::vector<uint8_t> touched;
    touched.assign(mesh.VertexCount(), 0);
    for (uint32_t ei = 0; ei < sharp_edges.size(); ++ei) {
        if (!sharp_edges[ei]) continue;
        const auto hh = mesh.GetHalfedge(Mesh::EH{ei}, 0);
        if (const auto to = mesh.GetToVertex(hh)) touched[*to] = 1;
        if (const auto from = mesh.GetFromVertex(hh)) touched[*from] = 1;
    }
    for (uint32_t fi = 0; fi < sharp_faces.size(); ++fi) {
        if (!sharp_faces[fi]) continue;
        for (const auto vh : mesh.fv_range(Mesh::FH{fi})) touched[*vh] = 1;
    }

    static thread_local std::vector<uint32_t> seam_offsets, seam_items;
    const auto add_incident = [&](Mesh::FH fh, uint32_t k) { seam_items.push_back(*fh | (k << FanLoopShift)); };
    const auto loop_position = [&](Mesh::FH fh, Mesh::HH h_v) {
        uint32_t k = 0;
        for (auto h = c.FaceHalfedge(*fh); h != h_v; h = c.Next(h)) ++k;
        return k;
    };
    seam_offsets.clear();
    seam_items.clear();
    seam_offsets.push_back(0);

    // Collect the seam sector around `h_in`'s corner vertex, walking both ways from `fh` until a sharp edge, sharp face, or boundary cuts the fan.
    const auto collect_sector = [&](Mesh::FH fh, Mesh::HH h_in, uint32_t k_in) {
        constexpr uint32_t MaxFan{64};
        add_incident(fh, k_in);
        bool full_loop = false;
        auto h = h_in;
        for (uint32_t i = 0; i < MaxFan; ++i) {
            const auto out = c.Next(h); // Advances within the current face while retaining the corner vertex.
            if (edge_sharp(out)) break;
            const auto opp = c.Opposites[*out];
            if (!opp) break;
            const auto nf = c.FaceOf(opp);
            if (!nf) break;
            if (nf == fh) {
                full_loop = true;
                break;
            }
            if (face_sharp(nf)) break;
            add_incident(nf, loop_position(nf, opp));
            h = opp;
        }
        if (!full_loop) {
            h = h_in;
            for (uint32_t i = 0; i < MaxFan; ++i) {
                if (edge_sharp(h)) break;
                const auto opp = c.Opposites[*h];
                if (!opp) break;
                const auto nf = c.FaceOf(opp);
                if (!nf || nf == fh || face_sharp(nf)) break;
                // Continue from the halfedge entering the corner vertex within `nf`.
                auto prev = opp;
                for (auto walk = c.Next(opp); walk != opp; walk = c.Next(walk)) prev = walk;
                add_incident(nf, loop_position(nf, prev));
                h = prev;
            }
        }
        seam_offsets.push_back(uint32_t(seam_items.size()));
    };

    uint32_t ci = 0;
    static thread_local std::vector<uint32_t> face_classes; // Per-face corner classes in vertex order, emitted in fan order.
    for (const auto fh : mesh.faces()) {
        const auto tri_count = mesh.GetValence(fh) - 2;
        if (face_sharp(fh)) {
            for (uint32_t t = 0; t < tri_count * 3; ++t) classes[ci++] = uint32_t(CornerClass::Face) << ClassTagShift;
            continue;
        }
        face_classes.clear();
        uint32_t k = 0;
        for (const auto hh : mesh.fh_range(fh)) {
            const auto vh = mesh.GetToVertex(hh);
            if (touched[*vh]) {
                const auto s = uint32_t(seam_offsets.size() - 1);
                collect_sector(fh, hh, k);
                face_classes.push_back((uint32_t(CornerClass::Seam) << ClassTagShift) | s);
            } else {
                face_classes.push_back(uint32_t(CornerClass::Vertex) << ClassTagShift);
            }
            ++k;
        }
        for (uint32_t i = 1; i + 1 < face_classes.size(); ++i) {
            classes[ci++] = face_classes[0];
            classes[ci++] = face_classes[i];
            classes[ci++] = face_classes[i + 1];
        }
    }

    derived.SeamCornerCount = uint32_t(seam_offsets.size() - 1);
    const auto seam_words = derived.SeamCornerCount > 0 ? uint32_t(seam_offsets.size() + seam_items.size()) : 0u;
    if (derived.SeamFans.Count != seam_words) {
        Buffers.Adjacency.Release(derived.SeamFans);
        derived.SeamFans = Buffers.Adjacency.Allocate(seam_words);
    }
    if (seam_words > 0) {
        const auto out = Buffers.Adjacency.GetMutable(derived.SeamFans);
        std::ranges::copy(seam_offsets, out.begin());
        std::ranges::copy(seam_items, out.begin() + seam_offsets.size());
    }
    if (derived.BaseSeamNormals.Count != derived.SeamCornerCount) {
        Buffers.BaseSeamNormals.Release(derived.BaseSeamNormals);
        derived.BaseSeamNormals = Buffers.BaseSeamNormals.Allocate(derived.SeamCornerCount);
    }
}

std::span<const vec3> MeshStore::GetCornerNormals(const Mesh &mesh) const {
    return GetCornerNormals(mesh, mesh.CreateTriangleIndices());
}

std::span<const vec3> MeshStore::GetCornerNormals(const Mesh &mesh, std::span<const uint32_t> indices) const {
    const auto id = mesh.GetStoreId();
    const auto &record = Records.at(id);
    const auto &derived = DerivedRecords.at(id);
    static thread_local std::vector<vec3> corners;
    corners.resize(size_t{record.TriangleCount} * 3);
    if (corners.empty()) return corners;
    const CornerNormalSources sources{Buffers.BaseVertexNormals.Get(record.Vertices), Buffers.BaseSeamNormals.Get(derived.BaseSeamNormals), Buffers.BaseFaceNormals.Get(record.FaceData)};
    const auto classes = Buffers.CornerClasses.Get(derived.CornerClasses);
    const auto face_ids = Buffers.TriangleFaceIds.Get(record.TriangleFaceIds);
    for (uint32_t ci = 0; ci < corners.size(); ++ci) {
        corners[ci] = ComposeCornerNormal(classes, derived.UniformCornerClass, ci, indices, face_ids, sources);
    }
    const auto masks = Buffers.CustomCornerMasks.Get(record.CustomCornerMasks);
    if (masks.empty()) return corners;
    const auto packed = Buffers.CustomCornerNormals.Get(record.CustomCornerNormals);
    const auto vertices = Buffers.Vertices.Get(record.Vertices);
    size_t next = 0;
    for (size_t w = 0; w < masks.size(); ++w) {
        for (auto word = masks[w].x; word != 0; word &= word - 1) {
            const auto i = w * 32 + std::countr_zero(word);
            corners[i] = DecodeNormalOffset(packed[next++], ComputeCornerFrame(corners[i], indices, vertices, i));
        }
    }
    return corners;
}

SharpnessSummary MeshStore::GetFaceSharpnessSummary(uint32_t id) const {
    const auto s = Buffers.FaceSharpness.Get(Records.at(id).FaceData);
    return {std::ranges::any_of(s, IsSharp), !s.empty() && std::ranges::all_of(s, IsSharp)};
}

namespace {
void WriteVertices(std::span<Vertex> dst, std::span<const vec3> positions) {
    for (uint32_t i = 0; i < positions.size(); ++i) dst[i] = {.Position = positions[i]};
}
} // namespace

void MeshStore::AllocateConnectivity(uint32_t id, uint32_t vertex_count, uint32_t halfedge_count, uint32_t face_count, bool face_starts) {
    auto &record = WriteRecord(id);
    record.ConnectivityVertices = vertex_count;
    record.ConnectivityHalfedges = halfedge_count;
    record.ConnectivityFaces = face_count;
    record.ConnectivityFaceStarts = face_starts;
    record.Connectivity = Buffers.Connectivity.Allocate(ConnectivityWords(vertex_count, halfedge_count, face_count, face_starts));
}

namespace {
auto SliceConnectivity(const auto &record, auto run) {
    const auto vertices = record.ConnectivityVertices, halfedges = record.ConnectivityHalfedges;
    const auto words = BitWords(halfedges);
    constexpr bool IsConst = std::is_const_v<typename decltype(run)::element_type>;
    using Handle = std::conditional_t<IsConst, const he::HH, he::HH>;
    using Face = std::conditional_t<IsConst, const MeshConnectivity::Face, MeshConnectivity::Face>;
    const auto handles = [](auto words) { return std::span{reinterpret_cast<Handle *>(words.data()), words.size()}; };
    return std::tuple{
        handles(run.subspan(0, vertices)),
        handles(run.subspan(vertices, halfedges)),
        run.subspan(vertices + halfedges, words),
        run.subspan(vertices + halfedges + words, words),
        run.subspan(vertices + halfedges + 2 * words, words),
        record.ConnectivityFaceStarts ?
            std::span{reinterpret_cast<Face *>(run.data() + vertices + halfedges + 3 * words), record.ConnectivityFaces} :
            std::span<Face>{},
    };
}
} // namespace

ConnectivityStorage MeshStore::GetConnectivityStorage(uint32_t id) {
    const auto &record = Records.at(id);
    return std::apply([](auto... spans) { return ConnectivityStorage{spans...}; }, SliceConnectivity(record, Buffers.Connectivity.GetMutable(record.Connectivity)));
}

void MeshStore::PlaceConnectivity(uint32_t id, const BuiltConnectivity &built) {
    auto &record = WriteRecord(id);
    record.ConnectivityEdgeCount = built.EdgeCount;
    // Only a non-manifold mesh keeps an edge list, which the bit ranks cannot answer for.
    if (built.Edges.empty()) return;
    record.ConnectivityEdges = Buffers.Connectivity.Allocate(uint32_t(built.Edges.size()));
    record.ConnectivityHalfedgeToEdge = Buffers.Connectivity.Allocate(uint32_t(built.HalfedgeToEdge.size()));
    const auto edges = Buffers.Connectivity.GetMutable(record.ConnectivityEdges);
    const auto halfedge_to_edge = Buffers.Connectivity.GetMutable(record.ConnectivityHalfedgeToEdge);
    for (uint32_t i = 0; i < built.Edges.size(); ++i) edges[i] = *built.Edges[i];
    for (uint32_t i = 0; i < built.HalfedgeToEdge.size(); ++i) halfedge_to_edge[i] = *built.HalfedgeToEdge[i];
}

void MeshStore::SetConnectivityEdgeCount(uint32_t id, uint32_t edge_count) { WriteRecord(id).ConnectivityEdgeCount = edge_count; }

MeshConnectivity MeshStore::GetConnectivity(uint32_t id) const {
    const auto &record = Records.at(id);
    if (record.Connectivity.Count == 0) return {.VertexCount = record.ConnectivityVertices, .EdgeCount = record.ConnectivityEdgeCount, .FaceCount = record.ConnectivityFaces};
    const auto [outgoing, opposites, bits, ranks, samples, faces] = SliceConnectivity(record, Buffers.Connectivity.Get(record.Connectivity));
    // A non-manifold mesh reads its edges from the list instead of the bit ranks.
    const bool explicit_edges = record.ConnectivityEdges.Count > 0;
    const auto edges = explicit_edges ? Buffers.Connectivity.Get(record.ConnectivityEdges) : std::span<const uint32_t>{};
    const auto halfedge_to_edge = explicit_edges ? Buffers.Connectivity.Get(record.ConnectivityHalfedgeToEdge) : std::span<const uint32_t>{};
    return {
        .VertexCount = record.ConnectivityVertices,
        .OutgoingHalfedges = outgoing,
        .Opposites = opposites,
        .EdgeFirstBits = explicit_edges ? std::span<const uint32_t>{} : bits,
        .EdgeFirstRanks = explicit_edges ? std::span<const uint32_t>{} : ranks,
        .HalfedgeToEdge = {reinterpret_cast<const he::EH *>(halfedge_to_edge.data()), halfedge_to_edge.size()},
        .EdgeCount = record.ConnectivityEdgeCount,
        .Edges = {reinterpret_cast<const he::HH *>(edges.data()), edges.size()},
        .EdgeSamples = explicit_edges ? std::span<const uint32_t>{} : samples.first(BitWords(record.ConnectivityEdgeCount)),
        .FaceCount = record.ConnectivityFaces,
        .Faces = faces,
    };
}

uint32_t MeshStore::CreateMeshSource(const MeshData &data) {
    const auto vertices = Buffers.Vertices.Allocate(data.Positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), data.Positions);
    const auto id = AcquireId({.Vertices = vertices, .Alive = true});
    SyncMirrors(id);
    // A face mesh's corners are its face loops.
    // An edge mesh has no weld and receives its corners in CreateMesh.
    if (data.FaceCount() > 0) WriteRecord(id).FaceCorners = Buffers.FaceCorners.Allocate(std::span<const uint32_t>{data.FaceCorners});
    return id;
}

void MeshStore::CreateDeformSource(uint32_t id, const std::optional<ArmatureDeformData> &deform, const std::optional<MorphTargetData> &morph) {
    auto &record = WriteRecord(id);
    const uint32_t vertex_count = record.Vertices.Count;
    if (vertex_count == 0) return;
    if (deform) {
        record.BoneDeform = Buffers.BoneDeform.Allocate(vertex_count);
        auto bone_deform = Buffers.BoneDeform.GetMutable(record.BoneDeform);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            bone_deform[i] = {.Joints = deform->Joints[i], .Weights = deform->Weights[i]};
        }
    }
    if (morph && morph->TargetCount > 0) {
        record.MorphTargetCount = morph->TargetCount;
        const uint32_t total = record.MorphTargetCount * vertex_count;
        record.MorphTargets = Buffers.MorphTargets.Allocate(total);
        auto morph_targets = Buffers.MorphTargets.GetMutable(record.MorphTargets);
        const bool has_normal_deltas = !morph->NormalDeltas.empty();
        for (uint32_t i = 0; i < total; ++i) {
            morph_targets[i] = {
                .PositionDelta = morph->PositionDeltas[i],
                .NormalDelta = has_normal_deltas ? morph->NormalDeltas[i] : vec3{0},
            };
        }
        record.DefaultMorphWeights = morph->DefaultWeights;
        record.DefaultMorphWeights.resize(record.MorphTargetCount, 0.f);
    }
}

void MeshStore::ShrinkMeshSource(uint32_t id, uint32_t welded_vertices) {
    auto &record = WriteRecord(id);
    Buffers.Vertices.Shrink(record.Vertices, welded_vertices);
    Buffers.BoneDeform.Shrink(record.BoneDeform, welded_vertices);
    Buffers.MorphTargets.Shrink(record.MorphTargets, record.MorphTargetCount * welded_vertices);
}

uint32_t MeshStore::AllocateVertexBuffer(std::span<const vec3> positions, const MeshVertexAttributes &attrs) {
    const auto vertices = Buffers.Vertices.Allocate(positions.size());
    WriteVertices(Buffers.Vertices.GetMutable(vertices), positions);
    // Face-less meshes keep authored normals as primary point normals, mirrored for shader reads.
    const auto point_normals = attrs.Normals ? Buffers.PointNormals.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    const auto id = AcquireId({.Vertices = vertices, .PointNormals = point_normals, .Alive = true});
    SyncMirrors(id);
    FillBaseVertexNormalMirror(vertices, point_normals);
    return id;
}

void MeshStore::PlanCreate(const MeshData &data, const MeshPrimitives &primitives, bool has_deform, uint32_t morph_target_count, const MeshVertexAttributes &attrs) {
    const uint32_t vertices = data.Positions.size();
    const uint32_t faces = data.FaceCount();
    const uint32_t halfedges = data.HalfedgeCount();
    const uint32_t triangles = uint32_t(data.FaceCorners.size()) - 2u * faces;
    const uint32_t edges = (halfedges + 1) / 2; // manifold estimate: edges ≈ halfedges / 2
    Buffers.Vertices.PlanAdditional(vertices);
    Buffers.BaseVertexNormals.PlanAdditional(vertices);
    Buffers.FaceFirstTriangles.PlanAdditional(faces);
    Buffers.FaceSharpness.PlanAdditional(faces);
    Buffers.BaseFaceNormals.PlanAdditional(faces);
    Buffers.TriangleFaceIds.PlanAdditional(triangles);
    Buffers.CornerClasses.PlanAdditional(triangles * 3);
    Buffers.FaceCorners.PlanAdditional(halfedges);
    Buffers.EdgeSharpness.PlanAdditional(edges);
    Buffers.Adjacency.PlanAdditional(FanAdjacencyWords(vertices, halfedges) + EdgeAdjacencyWords(vertices, edges));
    Buffers.Connectivity.PlanAdditional(ConnectivityWords(vertices, halfedges, faces, faces > 0 && halfedges != 3 * faces));
    Buffers.PrimitiveMaterials.PlanAdditional(primitives.MaterialIndices.size());
    if (has_deform) Buffers.BoneDeform.PlanAdditional(vertices);
    if (morph_target_count > 0) Buffers.MorphTargets.PlanAdditional(morph_target_count * vertices);
    // Point and line meshes index their primitive per vertex, triangle meshes per face.
    Buffers.ElementPrimitives.PlanAdditional(faces > 0 ? faces : uint32_t(primitives.ElementPrimitiveIndices.size()));
    if (triangles > 0) {
        const uint32_t corners = triangles * 3;
        if (attrs.Tangents) Buffers.CornerTangents.PlanAdditional(corners);
        if (attrs.Colors0) Buffers.CornerColors.PlanAdditional(corners);
        for (const auto *uvs : {&attrs.TexCoords0, &attrs.TexCoords1, &attrs.TexCoords2, &attrs.TexCoords3}) {
            if (*uvs) Buffers.CornerUvs.PlanAdditional(corners);
        }
    } else if (attrs.Colors0) {
        Buffers.CornerColors.PlanAdditional(vertices);
    }
}

void MeshStore::PlanClone(const Mesh &mesh) {
    const auto &record = Records.at(mesh.GetStoreId());
    const auto &derived = DerivedRecords.at(mesh.GetStoreId());
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &, auto &&ranges) {
        for (const auto *range : ranges(record, derived)) arena.PlanAdditional(range->Count);
    });
}

void MeshStore::CommitReserves() {
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.CommitPlanned(); });
}

namespace {
// Fill `out` with CSR incidence over `bucket_count` buckets: (bucket_count + 1) item-start offsets, then the items.
// `emit(add)` calls add(bucket, item) in a fixed order and runs twice (count, then scatter).
void WriteCsr(std::span<uint32_t> out, uint32_t bucket_count, auto &&emit) {
    const auto offsets = out.first(bucket_count + 1);
    const auto items = out.subspan(bucket_count + 1);
    std::ranges::fill(offsets, 0u);
    emit([&](uint32_t bucket, uint32_t) { ++offsets[bucket + 1]; });
    for (uint32_t b = 0; b < bucket_count; ++b) offsets[b + 1] += offsets[b];
    std::vector<uint32_t> cursors(offsets.begin(), offsets.end() - 1);
    emit([&](uint32_t bucket, uint32_t item) { items[cursors[bucket]++] = item; });
}
} // namespace

bool BuildsFanAdjacencyOnGpu(const Mesh &mesh) {
    const auto &c = mesh.GetConnectivity();
    return c.FaceCount > 0 && c.Faces.empty();
}

bool BuildsEdgeAdjacencyOnGpu(const Mesh &mesh) {
    return BuildsFanAdjacencyOnGpu(mesh) && mesh.GetConnectivity().HalfedgeToEdge.empty();
}

namespace {
// Calls `add(vertex, item)` once per vertex-fan incidence in table order.
void EmitFanIncidence(const Mesh &mesh, auto &&add) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t face = 0; face < c.FaceCount; ++face) {
        const auto first = *c.FaceHalfedge(face), last = c.FaceEnd(face);
        for (auto h = first; h < last; ++h) add(corners[h], face | ((h - first) << FanLoopShift));
    }
}

// Call `add(vertex, item)` once per vertex-edge incidence, in the order the table records it.
void EmitEdgeIncidence(const Mesh &mesh, auto &&add) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t edge = 0; edge < c.EdgeCount; ++edge) {
        const auto h = c.EdgeHalfedge(edge);
        const auto opposite = c.Opposites[*h];
        add(corners[*(opposite ? opposite : c.Previous(h))], edge);
        add(corners[*h], edge);
    }
}
} // namespace

void MeshStore::BuildVertexAdjacency(const Mesh &mesh) {
    const profile::CpuScope scope{"VertexAdjacency"};
    const auto id = mesh.GetStoreId();
    auto &record = Records.at(id);
    auto &derived = DerivedRecords.at(id);
    const uint32_t vertex_count = mesh.VertexCount();
    if (record.TriangleCount > 0) {
        derived.VertexFanAdjacency = Buffers.Adjacency.Allocate(FanAdjacencyWords(vertex_count, mesh.HalfEdgeCount()));
        if (!BuildsFanAdjacencyOnGpu(mesh)) {
            WriteCsr(Buffers.Adjacency.GetMutable(derived.VertexFanAdjacency), vertex_count, [&](auto &&add) { EmitFanIncidence(mesh, add); });
        }
    }
    if (mesh.EdgeCount() > 0) {
        derived.VertexEdgeAdjacency = Buffers.Adjacency.Allocate(EdgeAdjacencyWords(vertex_count, mesh.EdgeCount()));
        if (!BuildsEdgeAdjacencyOnGpu(mesh)) {
            WriteCsr(Buffers.Adjacency.GetMutable(derived.VertexEdgeAdjacency), vertex_count, [&](auto &&add) { EmitEdgeIncidence(mesh, add); });
        }
    }
}

std::string MeshStore::CheckVertexAdjacency(const Mesh &mesh) const {
    const auto vertex_count = mesh.VertexCount();
    const auto check = [&](std::string_view name, Range range, auto &&emit) {
        if (range.Count == 0) return std::string{};
        std::vector<uint32_t> reference(range.Count);
        WriteCsr(reference, vertex_count, emit);
        const auto stored = Buffers.Adjacency.Get(range);
        for (uint32_t i = 0; i < range.Count; ++i) {
            if (reference[i] == stored[i]) continue;
            const bool offset = i <= vertex_count;
            return std::format(
                "mesh {} {} {} {} differs: {} against {}",
                mesh.GetStoreId(), name, offset ? "offset" : "item", offset ? i : i - vertex_count - 1, stored[i], reference[i]
            );
        }
        return std::string{};
    };
    const auto &derived = DerivedRecords.at(mesh.GetStoreId());
    if (auto fan = check("fan", derived.VertexFanAdjacency, [&](auto &&add) { EmitFanIncidence(mesh, add); }); !fan.empty()) return fan;
    return check("edge", derived.VertexEdgeAdjacency, [&](auto &&add) { EmitEdgeIncidence(mesh, add); });
}

void MeshStore::CreateMesh(uint32_t id, const MeshData &data, const MeshVertexAttributes &attrs, const MeshPrimitives &primitives, const CornerLayers &layers, bool has_authored_normals) {
    const profile::CpuScope scope{"CreateMesh"};
    const uint32_t face_count = data.FaceCount();

    // Source creation and welding completed the vertex-domain arena ranges.
    auto &record = WriteRecord(id);
    record.HasAuthoredNormals = has_authored_normals;
    // A face-less mesh keeps its authored normals as primary point normals, mirrored for shader reads.
    record.PointNormals = attrs.Normals ? Buffers.PointNormals.Allocate(std::span<const vec3>{*attrs.Normals}) : Range{};
    FillBaseVertexNormalMirror(record.Vertices, record.PointNormals);

    const auto write_primitive_tables = [&](uint32_t element_count, uint32_t primitive_count) {
        record.ElementPrimitives = Buffers.ElementPrimitives.Allocate(element_count);
        auto fp_span = Buffers.ElementPrimitives.GetMutable(record.ElementPrimitives);
        if (!primitives.ElementPrimitiveIndices.empty()) std::ranges::copy(primitives.ElementPrimitiveIndices, fp_span.begin());
        else std::ranges::fill(fp_span, 0u);

        record.PrimitiveMaterials = Buffers.PrimitiveMaterials.Allocate(primitive_count);
        auto pm_span = Buffers.PrimitiveMaterials.GetMutable(record.PrimitiveMaterials);
        if (!primitives.MaterialIndices.empty()) std::ranges::copy(primitives.MaterialIndices, pm_span.begin());
        else std::ranges::fill(pm_span, 0u);
    };

    if (face_count > 0) {
        record.FaceData = Buffers.FaceFirstTriangles.Allocate(face_count);
        SyncMirrors(id);
        auto first_tri_span = Buffers.FaceFirstTriangles.GetMutable(record.FaceData);
        uint32_t tri_offset = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            first_tri_span[fi] = tri_offset;
            tri_offset += data.FaceSize(fi) - 2u;
        }
        record.TriangleCount = tri_offset;

        if (!layers.Tangents.empty()) record.CornerTangents = Buffers.CornerTangents.Allocate(std::span<const vec4>{layers.Tangents});
        if (!layers.Colors.empty()) record.CornerColors = Buffers.CornerColors.Allocate(std::span<const vec4>{layers.Colors});
        for (uint32_t set = 0; set < layers.Uvs.size(); ++set) {
            if (!layers.Uvs[set].empty()) record.CornerUvs[set] = Buffers.CornerUvs.Allocate(std::span<const vec2>{layers.Uvs[set]});
        }

        record.TriangleFaceIds = Buffers.TriangleFaceIds.Allocate(tri_offset);
        auto tri_face_span = Buffers.TriangleFaceIds.GetMutable(record.TriangleFaceIds);
        uint32_t ti = 0;
        for (uint32_t fi = 0; fi < face_count; ++fi) {
            const auto n_tris = data.FaceSize(fi) - 2u;
            for (size_t t = 0; t < n_tris; ++t) tri_face_span[ti++] = fi + 1;
        }

        const auto primitive_count = !primitives.MaterialIndices.empty() ?
            (primitives.ElementPrimitiveIndices.empty() ? 1u : *std::ranges::max_element(primitives.ElementPrimitiveIndices) + 1u) :
            1u;
        write_primitive_tables(face_count, primitive_count);

        // Faces come sorted by primitive, so each primitive's triangles are one contiguous range.
        {
            const auto fp = Buffers.ElementPrimitives.Get(record.ElementPrimitives);
            const auto fft = Buffers.FaceFirstTriangles.Get(record.FaceData);
            auto &ranges = record.PrimitiveTriangleRanges;
            uint32_t current_prim = fp[0];
            uint32_t range_first_tri = fft[0];
            for (uint32_t fi = 1; fi < face_count; ++fi) {
                if (fp[fi] != current_prim) {
                    ranges.push_back({current_prim, range_first_tri, fft[fi] - range_first_tri});
                    current_prim = fp[fi];
                    range_first_tri = fft[fi];
                }
            }
            ranges.push_back({current_prim, range_first_tri, record.TriangleCount - range_first_tri});
        }
    } else if (!primitives.ElementPrimitiveIndices.empty()) {
        // Point and line meshes carry one color and one primitive index per vertex.
        if (attrs.Colors0) record.CornerColors = Buffers.CornerColors.Allocate(std::span<const vec4>{*attrs.Colors0});
        // Primitive indices are source-wide, so the material table spans every primitive of the source mesh.
        const auto primitive_count = primitives.MaterialIndices.empty() ? 1u : uint32_t(primitives.MaterialIndices.size());
        write_primitive_tables(primitives.ElementPrimitiveIndices.size(), primitive_count);
    }

    // Store face loops or edge endpoints as the canonical halfedge corner indices.
    if (face_count == 0 && !data.Edges.empty()) {
        record.FaceCorners = Buffers.FaceCorners.Allocate(uint32_t(data.Edges.size()) * 2u);
        auto corners = Buffers.FaceCorners.GetMutable(record.FaceCorners);
        for (uint32_t e = 0; e < data.Edges.size(); ++e) {
            corners[e * 2u] = data.Edges[e][1];
            corners[e * 2u + 1u] = data.Edges[e][0];
        }
    }

    // The sharpness stores start smooth.
    const Mesh mesh{*this, id};
    record.EdgeSharpness = Buffers.EdgeSharpness.Allocate(mesh.EdgeCount());
    std::ranges::fill(Buffers.FaceSharpness.GetMutable(record.FaceData), uint8_t{0});
    std::ranges::fill(Buffers.EdgeSharpness.GetMutable(record.EdgeSharpness), uint8_t{0});
    BuildVertexAdjacency(mesh);
}

uint32_t MeshStore::CloneMesh(const Mesh &mesh) {
    const auto src_id = mesh.GetStoreId();
    const auto id = AcquireId(Record{Records.at(src_id)});
    DerivedRecords[id] = DerivedRecords.at(src_id);
    const auto &src = Records[src_id];
    const auto &src_derived = DerivedRecords[src_id];
    auto &dst = Records[id];
    auto &dst_derived = DerivedRecords[id];
    // Every range the source holds clones into fresh storage, and each mirror copies over the master's cloned range.
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        const auto sources = ranges(src, src_derived);
        const auto targets = ranges(dst, dst_derived);
        for (size_t i = 0; i < sources.size(); ++i) {
            if (info.Mirror) {
                arena.Mirror(*targets[i]);
                std::ranges::copy(arena.Get(*sources[i]), arena.GetMutable(*targets[i]).begin());
            } else {
                *targets[i] = arena.Clone(*sources[i]);
            }
        }
    });
    return id;
}

void MeshStore::ReleaseDerived(uint32_t id) {
    // A restore can leave derived records past the restored record count, so the persistent side reads as empty here.
    Record none{};
    auto &derived = DerivedRecords.at(id);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (info.Tracked() || info.Mirror) return;
        for (const auto *range : ranges(none, derived)) arena.Release(*range);
    });
    derived = {};
}

void MeshStore::Release(uint32_t id) {
    if (id >= Records.size() || !Records[id].Alive) return;
    auto &record = WriteRecord(id);
    auto &derived = DerivedRecords.at(id);
    ForEachArena(Buffers, [&](auto &arena, const ArenaInfo &info, auto &&ranges) {
        if (info.Mirror) return;
        for (const auto *range : ranges(record, derived)) arena.Release(*range);
    });
    record = {};
    derived = {};
    if (Tracked) Tracked->Free.Write(FreeIds.size(), 1);
    FreeIds.emplace_back(id);
}

void MeshStore::Clear() {
    if (Tracked) {
        Tracked->Entries.Write(0, Records.size());
        Tracked->Free.Write(0, FreeIds.size());
        Tracked->RangesDirty = true;
    }
    ForEachArena(Buffers, [](auto &arena, const ArenaInfo &, auto &&) { arena.Reset(); });
    Records.clear();
    DerivedRecords.clear();
    FreeIds.clear();
}

VertexAdjacency MeshStore::GetVertexEdgeAdjacency(uint32_t id) const {
    return SliceAdjacency(Buffers.Adjacency.Get(DerivedRecords.at(id).VertexEdgeAdjacency), Records.at(id).Vertices.Count);
}

uint32_t MeshStore::AcquireId(Record &&record) {
    if (!FreeIds.empty()) {
        const auto reused = FreeIds.back();
        if (Tracked) Tracked->Free.Write(FreeIds.size() - 1, 1);
        FreeIds.pop_back();
        WriteRecord(reused) = std::move(record);
        return reused;
    }
    if (Tracked) {
        Tracked->Entries.Write(Records.size(), 1);
        Tracked->RangesDirty = true;
    }
    Records.emplace_back(std::move(record));
    DerivedRecords.emplace_back();
    return uint32_t(Records.size() - 1);
}
