#pragma once

#include "gpu/Element.h"
#include "gpu/Vertex.h"
#include "mesh/ElementView.h"

#include "state/Entity.h"

#include <vector>

struct TriangleVertexView;

struct TrianglePoint {
    vec3 Position{0}, Weights{0};
};
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c);

namespace he {
constexpr uint32_t null{std::numeric_limits<uint32_t>::max()};

constexpr std::string_view label(Element element) {
    switch (element) {
        case Element::Vertex: return "vertex";
        case Element::Edge: return "edge";
        case Element::Face: return "face";
        case Element::None: return "none";
    }
}

namespace tag {
struct Vertex {};
struct Edge {};
struct Face {};

struct Halfedge {};
} // namespace tag

template<typename Tag>
struct Handle {
    uint32_t Index{null};

    uint32_t operator*() const { return Index; }
    auto operator<=>(const Handle &) const = default;
    explicit operator bool() const { return Index != null; }
};

using VH = Handle<tag::Vertex>;
using HH = Handle<tag::Halfedge>;
using EH = Handle<tag::Edge>;
using FH = Handle<tag::Face>;

// A record's live handles in one domain, ascending.
template<typename H>
struct HandleRange {
    struct Iterator {
        std::span<const uint32_t> Table;
        std::span<const MeshElementBlock> Blocks;
        uint32_t Local{null}, Word{}, Bits{};
        H operator*() const { return {Local * MeshElementBlockSize + Word * 32u + uint32_t(std::countr_zero(Bits))}; }
        Iterator &operator++() {
            Bits &= Bits - 1u;
            if (!Bits) {
                ++Word;
                Seek();
            }
            return *this;
        }
        bool operator==(const Iterator &other) const {
            return Local == other.Local && (Local == null || (Word == other.Word && Bits == other.Bits));
        }
        void Seek() {
            while (Local != null) {
                if (Table[Local] != null) {
                    for (; Word < MeshElementBlockWords; ++Word) {
                        Bits = Blocks[Table[Local]].Live[Word];
                        if (Bits) return;
                    }
                }
                Word = 0;
                if (++Local == Table.size()) Local = null;
            }
        }
    };
    std::span<const uint32_t> Table;
    std::span<const MeshElementBlock> Blocks;
    Iterator begin() const {
        Iterator it{Table, Blocks, Table.empty() ? null : 0u};
        it.Seek();
        return it;
    }
    Iterator end() const { return {Table, Blocks}; }
};
} // namespace he

inline std::vector<uint32_t> HandleList(const auto &range) {
    std::vector<uint32_t> handles;
    for (const auto h : range) handles.push_back(*h);
    return handles;
}

static constexpr uint32_t InvalidStoreId{~0u};

struct MeshStore;

// A record's connectivity: each domain's table entries, which resolve its handles to arena addresses, and the arenas' values at those addresses.
// Every argument and stored reference is a record-relative handle.
struct MeshConnectivity {
    std::span<const uint32_t> VertexTable, HalfedgeTable, EdgeTable, FaceTable;
    std::span<const MeshElementBlock> VertexBlocks, HalfedgeBlocks, EdgeBlocks, FaceBlocks;
    uint32_t VertexCount{}, HalfedgeCount{}, EdgeCount{}, FaceCount{};
    std::span<const uint32_t> OutgoingValues, OppositeValues, HalfedgeEdgeValues, HalfedgeFaceValues, EdgeHalfedgeValues;
    std::span<const uvec2> FaceRangeValues, VertexCornerValues;
    std::span<const uvec2> FanItems; // (corner, face) in fan runs, indexed by a vertex's incoming run

    ElementAddress VertexAddress(uint32_t v) const { return Resolve<MeshElementBlockSize>(VertexTable, ElementHandle{v}); }
    ElementAddress HalfedgeAddress(uint32_t h) const { return Resolve<MeshElementBlockSize>(HalfedgeTable, ElementHandle{h}); }
    ElementAddress EdgeAddress(uint32_t e) const { return Resolve<MeshElementBlockSize>(EdgeTable, ElementHandle{e}); }
    ElementAddress FaceAddress(uint32_t f) const { return Resolve<MeshElementBlockSize>(FaceTable, ElementHandle{f}); }
    bool HasVertexBlock(uint32_t v) const { return v / MeshElementBlockSize < VertexTable.size() && VertexTable[v / MeshElementBlockSize] != he::null; }

    he::HH Outgoing(he::VH v) const { return {OutgoingValues[VertexAddress(*v).Value()]}; }
    he::HH Opposite(he::HH hh) const { return {OppositeValues[HalfedgeAddress(*hh).Value()]}; }
    he::EH Edge(he::HH hh) const { return {HalfedgeEdgeValues[HalfedgeAddress(*hh).Value()]}; }
    he::HH EdgeHalfedge(uint32_t edge) const { return {EdgeHalfedgeValues[EdgeAddress(edge).Value()]}; }
    // A face's loop range [first, end) belongs to that face alone.
    uvec2 FaceRange(uint32_t face) const { return FaceRangeValues[FaceAddress(face).Value()]; }
    he::HH FaceHalfedge(uint32_t face) const { return {FaceRange(face).x}; }
    // The vertex's run of fan items: the first and the count.
    uvec2 Incoming(he::VH v) const { return VertexCornerValues[VertexAddress(*v).Value()]; }

    // The halfedge's face, or empty for a wire corner.
    he::FH FaceOf(he::HH hh) const { return {HalfedgeFaceValues[HalfedgeAddress(*hh).Value()]}; }

    he::HH Next(he::HH hh) const {
        const auto face = FaceOf(hh);
        if (!face) return {};
        const auto range = FaceRange(*face);
        return he::HH(*hh + 1 < range.y ? *hh + 1 : range.x);
    }

    he::HH Previous(he::HH hh) const {
        const auto face = FaceOf(hh);
        if (!face) return Opposite(hh);
        const auto range = FaceRange(*face);
        return he::HH(*hh == range.x ? range.y - 1u : *hh - 1u);
    }
};

// Borrows connectivity and vertex data from MeshStore.
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    Mesh() = default;
    Mesh(const MeshStore &store, uint32_t store_id);

    uint32_t VertexCount() const { return C.VertexCount; }
    VH VertexAt(uint32_t ordinal) const;
    uint32_t VertexOrdinal(VH vertex) const;
    uint32_t EdgeCount() const { return C.EdgeCount; }
    EH EdgeAt(uint32_t ordinal) const;
    FH FaceAt(uint32_t ordinal) const;
    uint32_t FaceOrdinal(FH f) const;
    uint32_t FaceCount() const { return C.FaceCount; }
    uint32_t HalfEdgeCount() const { return C.HalfedgeCount; }
    bool HasClosedSurface() const { return FaceCount() && uint64_t(HalfEdgeCount()) == 2ull * EdgeCount(); }

    const MeshStore &GetStore() const { return *Store; }
    uint32_t GetStoreId() const { return StoreId; }
    const MeshConnectivity &GetConnectivity() const { return C; }
    ElementView<uvec3> DerivedTriangles() const;
    TriangleVertexView TriangleVertices() const;
    HandleValues<Vertex> VertexValues() const;
    uint32_t TriangleIndexCount() const;

    he::HandleRange<VH> vertices() const { return {C.VertexTable, C.VertexBlocks}; }
    he::HandleRange<EH> edges() const { return {C.EdgeTable, C.EdgeBlocks}; }
    he::HandleRange<FH> faces() const { return {C.FaceTable, C.FaceBlocks}; }
    uint32_t ElementCount(Element element) const {
        return element == Element::Vertex ? VertexCount() : element == Element::Edge ? EdgeCount() :
            element == Element::Face                                                 ? FaceCount() :
                                                                                       0u;
    }

private:
    const MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    MeshConnectivity C{};
};

// Whether the entity is a mesh entity, which holds a geometry procedure.
bool HasMesh(const state::Registry &, state::Entity);
// Whether an instance of the entity draws a mesh record: a cooked mesh entity's, or the record a procedure's instances place.
bool DrawsMesh(const state::Registry &, state::Entity);
// The store record an entity's instances draw: its evaluated procedure output, its owned record, or the vertex record of a bone or joint.
std::optional<uint32_t> DrawnStoreId(const state::Registry &, state::Entity);
// TryGetMesh returns empty for an entity drawing no mesh record.
Mesh GetMesh(const state::Registry &, state::Entity);
std::optional<Mesh> TryGetMesh(const state::Registry &, state::Entity);
