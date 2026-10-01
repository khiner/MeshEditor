#pragma once

#include <algorithm>
#include <array>
#include <bit>

#include "gpu/Element.h"
#include "gpu/Vertex.h"
#include "mesh/ElementView.h"

#include "gpu/MeshElementBlock.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "state/Entity.h"

#include <optional>
#include <span>
#include <vector>

struct TriangleVertexView;

struct TrianglePoint {
    vec3 Position{0}, Weights{0};
};
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c);

namespace he {
constexpr uint32_t null{std::numeric_limits<uint32_t>::max()};

constexpr uint8_t ElementMask(Element element) { return uint8_t(element); }
constexpr bool ElementMaskContains(uint8_t mask, Element element) { return (mask & ElementMask(element)) != 0; }
constexpr void SetElementMask(uint8_t &mask, Element element, bool enabled) {
    if (enabled) mask |= ElementMask(element);
    else mask &= ~ElementMask(element);
}

constexpr std::array Elements{Element::Vertex, Element::Edge, Element::Face};

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

// Enumerates live canonical handles in membership order, borrowing the GPU metadata.
template<typename H>
struct HandleRange {
    struct Iterator {
        std::span<const MeshElementBlock> Blocks;
        uint32_t Block{null}, Word{}, Bits{};
        H operator*() const { return {Block * MeshElementBlockSize + Word * 32u + uint32_t(std::countr_zero(Bits))}; }
        Iterator &operator++() {
            Bits &= Bits - 1u;
            if (!Bits) { ++Word; Seek(); }
            return *this;
        }
        bool operator==(const Iterator &other) const {
            return Block == other.Block && (Block == null || (Word == other.Word && Bits == other.Bits));
        }
        void Seek() {
            while (Block != null) {
                for (; Word < MeshElementBlockWords; ++Word) {
                    Bits = Blocks[Block].Live[Word];
                    if (Bits) return;
                }
                Block = Blocks[Block].Next;
                Word = 0;
            }
        }
    };
    std::span<const MeshElementBlock> Blocks;
    uint32_t FirstBlock{null};
    Iterator begin() const { Iterator it{Blocks, FirstBlock}; it.Seek(); return it; }
    Iterator end() const { return {Blocks}; }
};
} // namespace he

static constexpr uint32_t InvalidStoreId{~0u};

struct MeshStore;

struct MeshConnectivity {
    struct Face {
        he::HH Halfedge;
        uint32_t End;
    };

    std::span<const MeshElementBlock> VertexBlocks, HalfedgeBlocks, EdgeBlocks, FaceBlocks;
    uint32_t VertexFirst{}, VertexCount{0};
    uint32_t HalfedgeFirst{}, HalfedgeCount{}, EdgeFirst{}, FaceFirst{};
    std::span<const he::HH> OutgoingHalfedges;
    std::span<const he::HH> Opposites;
    // Edges number by ascending first halfedge.
    std::span<const he::EH> HalfedgeToEdge;
    std::span<const he::FH> HalfedgeToFace;
    uint32_t EdgeCount{0};
    std::span<const he::HH> Edges;
    uint32_t FaceCount{0};
    // Each face owns its loop range independently of neighboring faces.
    std::span<const Face> Faces;
    std::span<const uvec2> VertexCorners;
    std::span<const uvec2> FanItems;

    he::HH FaceHalfedge(uint32_t face) const { return Faces[face].Halfedge; }
    uint32_t FaceEnd(uint32_t face) const { return Faces[face].End; }

    he::HH EdgeHalfedge(uint32_t edge) const { return Edges[edge]; }
    he::EH Edge(he::HH hh) const { return HalfedgeToEdge[*hh]; }

    // Returns the halfedge's owner, or empty for edge-only meshes.
    he::FH FaceOf(he::HH hh) const {
        if (FaceCount == 0) return {};
        return HalfedgeToFace[*hh];
    }

    he::HH Next(he::HH hh) const {
        const auto face = FaceOf(hh);
        if (!face) return {};
        const auto first = *FaceHalfedge(*face);
        const auto last = FaceEnd(*face);
        return he::HH(*hh + 1 < last ? *hh + 1 : first);
    }

    he::HH Previous(he::HH hh) const {
        const auto face = FaceOf(hh);
        if (!face) return {};
        const auto first = *FaceHalfedge(*face);
        return he::HH(*hh == first ? FaceEnd(*face) - 1u : *hh - 1u);
    }
};

struct VertexEdgeIncidence {
    MeshConnectivity C;
    struct Iterator {
        using difference_type = std::ptrdiff_t;
        using value_type = uint32_t;
        const MeshConnectivity *C{};
        uint32_t Item{}, Remaining{}, Side{}, Edge{he::null};
        uint32_t operator*() const { return Edge; }
        Iterator &operator++();
        bool operator==(const Iterator &other) const { return Edge == other.Edge && Remaining == other.Remaining && Side == other.Side; }
    };
    struct Range {
        const MeshConnectivity *C;
        uint32_t First, Count;
        Iterator begin() const { return ++Iterator{C, First, Count}; }
        Iterator end() const { return {C}; }
    };
    Range Incident(uint32_t v) const { return {&C, C.VertexCorners[v].x,C.VertexCorners[v].y}; }
};

// Borrows connectivity and vertex data from MeshStore.
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    Mesh() = default;
    Mesh(const MeshStore &store, uint32_t store_id);
    // Packed input/export view.
    // Direct handle reads use GetToVertex.
    std::span<const uint32_t> CornerVertices() const;

    uint32_t VertexCount() const { return C.VertexCount; }
    uint32_t VertexFirst() const { return C.VertexFirst; }
    VH VertexAt(uint32_t ordinal) const;
    uint32_t VertexOrdinal(VH vertex) const;
    uint32_t EdgeCount() const { return C.EdgeCount; }
    uint32_t EdgeFirst() const { return C.EdgeFirst; }
    uint32_t FaceFirst() const { return C.FaceFirst; }
    uint32_t HalfedgeFirst() const { return C.HalfedgeFirst; }
    EH EdgeAt(uint32_t ordinal) const;
    FH FaceAt(uint32_t ordinal) const;
    uint32_t FaceOrdinal(FH f) const;
    uint32_t FaceCount() const { return C.FaceCount; }
    // The MeshPrimitiveTopology its live elements draw as: faces, else edges, else points.
    uint32_t PrimitiveTopology() const {
        return uint32_t(FaceCount() ? MeshPrimitiveTopology::Triangle : EdgeCount() ? MeshPrimitiveTopology::Line : MeshPrimitiveTopology::Point);
    }
    uint32_t HalfEdgeCount() const { return C.HalfedgeCount; }
    bool HasClosedSurface() const { return FaceCount() && uint64_t(HalfEdgeCount()) == 2ull * EdgeCount(); }

    const vec3 &GetPosition(VH) const;
    const vec3 &GetNormal(VH) const;
    vec3 GetNormal(FH) const;

    uint32_t GetStoreId() const { return StoreId; }
    const MeshConnectivity &GetConnectivity() const { return C; }
    ElementView<uvec3> DerivedTriangles() const;
    TriangleVertexView TriangleVertices() const;
    uint32_t TriangleIndexCount() const;

    // Incident edges derived from the canonical incoming-corner lists.
    VertexEdgeIncidence GetVertexEdgeIncidence() const;

    HH GetHalfedge(EH eh, uint32_t i) const {
        const auto h0 = C.EdgeHalfedge(*eh);
        return i == 0 ? h0 : (i == 1 && h0 ? C.Opposites[*h0] : HH{});
    }
    HH GetOppositeHalfedge(HH hh) const { return C.Opposites[*hh]; }
    EH GetEdge(HH hh) const { return C.Edge(hh); }
    FH GetFace(HH hh) const { return C.FaceOf(hh); }
    VH GetFromVertex(HH) const;
    VH GetToVertex(HH hh) const { return VH(Corners[*hh]); }

    uint32_t GetValence(FH) const;

    vec3 CalcFaceCentroid(FH) const;
    // Discrete mean curvature (1/length) averaged over the one-ring normal curvatures.
    // 1/R on a sphere of radius R, zero on a flat or boundary vertex.
    // `edge_sharpness` is indexed by canonical edge handle, 1 where shading is discontinuous.
    // A sharp edge is where the surface turns rather than curves, so it has no curvature.
    float CalcMeanCurvature(VH, std::span<const uint8_t> edge_sharpness) const;
    VH FindNearestVertex(vec3) const;

    // Dense ordinals for CPU export/solver inputs.
    // Canonical corner and draw arenas store handles.
    std::vector<uint32_t> CreateTriangleIndices() const;
    void WriteTriangleIndices(std::span<uint32_t> dest) const;

    he::HandleRange<VH> vertices() const { return {C.VertexBlocks, VertexCount() ? VertexFirst() / MeshElementBlockSize : he::null}; }
    he::HandleRange<EH> edges() const { return {C.EdgeBlocks, EdgeCount() ? EdgeFirst() / MeshElementBlockSize : he::null}; }
    he::HandleRange<FH> faces() const { return {C.FaceBlocks, FaceCount() ? FaceFirst() / MeshElementBlockSize : he::null}; }
    uint32_t ElementCount(Element element) const {
        return element == Element::Vertex ? VertexCount() : element == Element::Edge ? EdgeCount() :
            element == Element::Face                                                 ? FaceCount() :
                                                                                       0u;
    }

    struct CirculatorBase {
        const Mesh *M{};
        HH CurrentHalfedge{}, StartHalfedge{};

        CirculatorBase() = default;
        CirculatorBase(const Mesh *m, HH current, HH start)
            : M(m), CurrentHalfedge(current), StartHalfedge(start) {}

        auto &operator++(this auto &self) {
            self.CurrentHalfedge = self.advance();
            if (self.CurrentHalfedge == self.StartHalfedge) self.CurrentHalfedge = HH{};
            return self;
        }

        auto operator++(this auto &self, int) {
            auto tmp = self;
            ++self;
            return tmp;
        }

        bool operator==(this auto const &self, const auto &other) { return self.CurrentHalfedge == other.CurrentHalfedge; }
    };

    struct FaceVertexIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = VH;
        using CirculatorBase::CirculatorBase;

        VH operator*() const { return M->GetToVertex(CurrentHalfedge); }
        HH advance() const { return M->C.Next(CurrentHalfedge); }
    };
    struct FaceVertexRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        FaceVertexIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceVertexIterator end() const { return {Mesh, HH{}, StartHalfedge}; }
    };
    FaceVertexRange fv_range(FH fh) const { return {this, C.FaceHalfedge(*fh)}; }

    struct VertexOutgoingHalfedgeIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const {
            const auto opp = M->C.Opposites[*CurrentHalfedge];
            return opp ? M->C.Next(opp) : HH{};
        }
    };
    struct VertexOutgoingHalfedgeRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        VertexOutgoingHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; }
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh && *vh >= C.VertexFirst && *vh - C.VertexFirst < C.VertexCount ? C.OutgoingHalfedges[*vh] : HH{}};
    }

    struct FaceHalfedgeIterator : CirculatorBase {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const { return M->C.Next(CurrentHalfedge); }
    };
    struct FaceHalfedgeRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        FaceHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; }
    };
    FaceHalfedgeRange fh_range(FH fh) const { return {this, C.FaceHalfedge(*fh)}; }

private:
    const MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    MeshConnectivity C{};
    std::span<const uint32_t> Corners{};
};

// GetMesh requires a mesh entity, and reads its preview while a staged operator has one.
// TryGetMesh returns empty for other entities.
Mesh GetMesh(const state::Scene &, state::Entity);
std::optional<Mesh> TryGetMesh(const state::Scene &, state::Entity);
bool HasMesh(const state::Scene &, state::Entity);
// The store record an entity's instances draw: its preview, its mesh, or the vertex record of a bone or joint.
std::optional<uint32_t> DrawnStoreId(const state::Scene &, state::Entity);
