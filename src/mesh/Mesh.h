#pragma once

#include <algorithm>
#include <array>
#include <bit>

#include "gpu/AABB.h"
#include "gpu/Element.h"
#include "gpu/Vertex.h"

#include <entt/entity/fwd.hpp>

#include <optional>
#include <span>
#include <vector>

namespace he { // half-edge
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

    constexpr Element GetElement() const {
        if constexpr (std::is_same_v<Tag, tag::Vertex>) return Element::Vertex;
        if constexpr (std::is_same_v<Tag, tag::Edge>) return Element::Edge;
        if constexpr (std::is_same_v<Tag, tag::Face>) return Element::Face;
        return Element::None;
    }
};

using VH = Handle<tag::Vertex>;
using HH = Handle<tag::Halfedge>;
using EH = Handle<tag::Edge>;
using FH = Handle<tag::Face>;

struct AnyHandle {
    AnyHandle(Element element = Element::None, uint32_t index = null) : Element(element), Index(index) {}
    template<typename Tag> AnyHandle(Handle<Tag> h) : Element(h.GetElement()), Index(*h) {}

    Element Element;
    uint32_t Index;

    uint32_t operator*() const { return Index; }
    bool operator==(const AnyHandle &other) const { return Element == other.Element && Index == other.Index; }
    operator bool() const { return Index != null; }

    bool operator==(VH vh) const { return Element == Element::Vertex && Index == *vh; }
    bool operator==(EH eh) const { return Element == Element::Edge && Index == *eh; }
    bool operator==(FH fh) const { return Element == Element::Face && Index == *fh; }

    operator VH() const { return {Element == Element::Vertex ? Index : null}; }
    operator EH() const { return {Element == Element::Edge ? Index : null}; }
    operator FH() const { return {Element == Element::Face ? Index : null}; }
};

struct AnyHandleHash {
    size_t operator()(const AnyHandle &h) const { return std::hash<uint32_t>{}(uint32_t(h.Element)) ^ (std::hash<uint32_t>{}(h.Index) << 1); }
};
} // namespace he

namespace std {
template<typename Tag>
struct hash<he::Handle<Tag>> {
    size_t operator()(const he::Handle<Tag> &h) const noexcept {
        return std::hash<uint32_t>{}(*h);
    }
};
} // namespace std

static constexpr uint32_t InvalidStoreId{~0u};

struct MeshStore;

// Half-edge connectivity for one mesh, as views into the store's arena, which is where it is built.
// Obtain one via MeshStore::GetConnectivity(StoreId).
struct MeshConnectivity {
    struct Face {
        he::HH Halfedge; // One of the boundary halfedges
    };

    uint32_t VertexCount{0};
    std::span<const he::HH> OutgoingHalfedges;
    std::span<const he::HH> Opposites; // Each halfedge's opposite, one per corner
    // One bit per halfedge, set when it is the first halfedge of its edge, with a running count of the
    // bits before each word. Edges number by ascending first halfedge, so an edge index is a bit rank.
    std::span<const uint32_t> EdgeFirstBits, EdgeFirstRanks;
    // Holds each halfedge's edge instead when a halfedge's first is neither itself nor its opposite,
    // which only a non-manifold edge produces.
    std::span<const he::EH> HalfedgeToEdge;
    uint32_t EdgeCount{0};
    // Each edge's first halfedge. Empty when the bits answer, where a sample every 32 edges bounds
    // the scan that finds the n-th set bit.
    std::span<const he::HH> Edges;
    std::span<const uint32_t> EdgeSamples; // Word index holding every 32nd edge's first halfedge
    uint32_t FaceCount{0};
    // Each face's first halfedge. Empty for a triangle mesh, whose face f starts at halfedge 3f.
    std::span<const Face> Faces;

    he::HH FaceHalfedge(uint32_t face) const { return Faces.empty() ? he::HH(face * 3u) : Faces[face].Halfedge; }
    uint32_t FaceEnd(uint32_t face) const { return face + 1 < FaceCount ? *FaceHalfedge(face + 1) : uint32_t(Opposites.size()); }

    he::HH EdgeHalfedge(uint32_t edge) const {
        if (!Edges.empty()) return Edges[edge];
        auto word = EdgeSamples[edge / 32u];
        while (word + 1u < EdgeFirstRanks.size() && EdgeFirstRanks[word + 1u] <= edge) ++word;
        auto bits = EdgeFirstBits[word];
        for (auto remaining = edge - EdgeFirstRanks[word]; remaining > 0u; --remaining) bits &= bits - 1u;
        return he::HH(word * 32u + uint32_t(std::countr_zero(bits)));
    }

    he::EH Edge(he::HH hh) const {
        if (!HalfedgeToEdge.empty()) return HalfedgeToEdge[*hh];
        const auto opposite = Opposites[*hh];
        const uint32_t first = opposite && *opposite < *hh ? *opposite : *hh;
        const auto word = first / 32u;
        return he::EH(EdgeFirstRanks[word] + uint32_t(std::popcount(EdgeFirstBits[word] & ((1u << (first % 32u)) - 1u))));
    }

    // A face's halfedges are the contiguous run its first halfedge starts, in face order, so a
    // halfedge's face is the run containing it. An edge-only mesh has no faces, so its halfedges
    // belong to none.
    he::FH FaceOf(he::HH hh) const {
        if (FaceCount == 0) return {};
        if (Faces.empty()) return he::FH(*hh / 3u);
        const auto after = std::upper_bound(Faces.begin(), Faces.end(), *hh, [](uint32_t h, const Face &f) { return h < *f.Halfedge; });
        return he::FH(uint32_t(after - Faces.begin()) - 1u);
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

// The arena spans a connectivity build fills, sized from the source counts before the build runs.
// `EdgeSamples` and `Edges` come sized at their bound, since the edge count only falls out of the build.
struct ConnectivityStorage {
    std::span<he::HH> OutgoingHalfedges, Opposites;
    std::span<uint32_t> EdgeFirstBits, EdgeFirstRanks, EdgeSamples;
    std::span<MeshConnectivity::Face> Faces; // Empty for a triangle mesh
};

// What a build returns for the caller to place: the edge count, and the edge list plus its inverse
// where a non-manifold edge produced one.
struct BuiltConnectivity {
    uint32_t EdgeCount{0};
    std::vector<he::HH> Edges;
    std::vector<he::EH> HalfedgeToEdge;
};

// Build connectivity into `storage` from polygon faces (concatenated vertex-index loops, face `f`
// spanning [offsets[f], offsets[f + 1]) of corners), or from edge pairs.
// An empty `face_offsets` means every face is a triangle, where face `f` spans corners [3f, 3f + 3).
BuiltConnectivity BuildConnectivity(std::span<const uint32_t> face_offsets, std::span<const uint32_t> face_corners, uint32_t vertex_count, const ConnectivityStorage &);
BuiltConnectivity BuildConnectivity(std::span<const std::array<uint32_t, 2>> edges, uint32_t vertex_count, const ConnectivityStorage &);

// CSR incidence over a mesh's vertices: Offsets holds one entry per vertex plus a terminator, and Incident(v) spans vertex v's items.
struct VertexAdjacency {
    std::span<const uint32_t> Offsets;
    std::span<const uint32_t> Items;
    std::span<const uint32_t> Incident(uint32_t v) const { return Items.subspan(Offsets[v], Offsets[v + 1] - Offsets[v]); }
};

// Lightweight, copyable view over a mesh: its connectivity (owned by MeshStore) plus its vertex data (read via StoreId).
// Holds no ownership, the MeshStore entry is released when the entity's MeshHandle is destroyed.
// Obtain one via MeshStore::GetMesh(StoreId).
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    Mesh() = default;
    Mesh(const MeshStore &store, uint32_t store_id);
    // The mesh's corner vertex indices, one per halfedge, from the store's canonical arena.
    std::span<const uint32_t> CornerVertices() const { return Corners; }

    uint32_t VertexCount() const { return C.VertexCount; }
    uint32_t EdgeCount() const { return C.EdgeCount; }
    uint32_t FaceCount() const { return C.FaceCount; }
    uint32_t HalfEdgeCount() const { return C.Opposites.size(); }

    const vec3 &GetPosition(VH) const;
    const vec3 &GetNormal(VH) const;
    vec3 GetNormal(FH) const;
    std::span<const Vertex> GetVerticesSpan() const;
    AABB CalcAABB() const; // Local-space

    uint32_t GetStoreId() const { return StoreId; }
    const MeshConnectivity &GetConnectivity() const { return C; }
    uint32_t TriangleIndexCount() const; // Cached triangle count * 3

    // CSR vertex-to-edge incidence, empty when the mesh has no edges.
    VertexAdjacency GetVertexEdgeAdjacency() const; // Items are edge indices

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
    // `edge_sharpness` is the mesh's canonical per-edge sharpness, 1 where shading is discontinuous.
    // A sharp edge is where the surface turns rather than curves, so it has no curvature.
    float CalcMeanCurvature(VH, std::span<const uint8_t> edge_sharpness) const;
    std::vector<float> CalcMeanCurvatures(std::span<const uint8_t> edge_sharpness) const; // The above for every vertex, in vertex order.
    // Volume the surface encloses, in the mesh's own units. Empty unless it is closed and manifold.
    std::optional<double> CalcEnclosedVolume() const;
    VH FindNearestVertex(vec3) const;

    std::vector<uint32_t> CreateTriangleIndices() const;
    void WriteTriangleIndices(std::span<uint32_t> dest) const;
    void WriteEdgeIndices(std::span<uint32_t> dest) const;

    struct VertexIterator {
        uint32_t Index;
        VH operator*() const { return {Index}; }
        VertexIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const VertexIterator &) const = default;
    };
    struct VertexRange {
        uint32_t Count;
        VertexIterator begin() const { return {0}; }
        VertexIterator end() const { return {Count}; }
    };
    VertexRange vertices() const { return {VertexCount()}; }

    struct EdgeIterator {
        uint32_t Index;
        EH operator*() const { return {Index}; }
        EdgeIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const EdgeIterator &) const = default;
    };
    struct EdgeRange {
        uint32_t Count;
        EdgeIterator begin() const { return {0}; }
        EdgeIterator end() const { return {Count}; }
    };
    EdgeRange edges() const { return {EdgeCount()}; }

    struct FaceIterator {
        uint32_t Index;
        FH operator*() const { return {Index}; }
        FaceIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const FaceIterator &) const = default;
    };
    struct FaceRange {
        uint32_t Count;
        FaceIterator begin() const { return {0}; }
        FaceIterator end() const { return {Count}; }
    };
    FaceRange faces() const { return {FaceCount()}; }

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
        operator bool() const { return bool(CurrentHalfedge); }
    };
    struct FaceVertexRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        FaceVertexIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceVertexIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    FaceVertexRange fv_range(FH fh) const { return {this, C.FaceHalfedge(*fh)}; }
    // Iterator positioned at the first vertex of a face.
    FaceVertexIterator cfv_iter(FH fh) const { return {this, C.FaceHalfedge(*fh), C.FaceHalfedge(*fh)}; }

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
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh && *vh < C.OutgoingHalfedges.size() ? C.OutgoingHalfedges[*vh] : HH{}};
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
        const Mesh *Mesh; // Always valid, never null
        HH StartHalfedge;
        FaceHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    FaceHalfedgeRange fh_range(FH fh) const { return {this, C.FaceHalfedge(*fh)}; }

private:
    const MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    MeshConnectivity C{};
    std::span<const uint32_t> Corners{};
};

// Resolve an entity's MeshHandle to a Mesh view via the registry's MeshStore.
// GetMesh asserts the entity has a mesh, TryGetMesh returns nullopt when it doesn't.
Mesh GetMesh(const entt::registry &, entt::entity);
std::optional<Mesh> TryGetMesh(const entt::registry &, entt::entity);
bool HasMesh(const entt::registry &, entt::entity);

// Surface length per texture coordinate unit, mesh-local, from its summed triangle area against the same triangles' area in texture coordinates.
// A node instancing the mesh multiplies by its world scale to get meters. Zero when the mesh carries no coordinates for that set.
float LocalLengthPerUv(const entt::registry &, entt::entity mesh_entity, uint32_t uv_set);
