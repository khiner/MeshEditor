#pragma once

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

// Tag types for type-safe handles
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

// Type-erased handle with comparison/conversion to typed handles
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

    // Implicit conversion to typed handles
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

// Plain half-edge connectivity for one mesh, a registry component the Mesh view below reads.
// `in_place_delete` keeps element addresses stable, so a Mesh view's cached pointer into the pool survives other entities' deletions.
struct MeshConnectivity {
    static constexpr bool in_place_delete = true;

    struct Halfedge {
        he::VH Vertex;
        he::HH Next;
        he::HH Opposite;
        he::FH Face; // Left face (invalid for boundary halfedges)
    };
    struct Face {
        he::HH Halfedge; // One of the boundary halfedges
    };

    uint32_t VertexCount{0};
    std::vector<he::HH> OutgoingHalfedges;
    std::vector<Halfedge> Halfedges;
    std::vector<he::EH> HalfedgeToEdge;
    std::vector<he::HH> Edges; // Maps edge index to its halfedge at index 0
    std::vector<Face> Faces;
};

// Build connectivity from polygon faces (vertex-index lists), edge pairs, or a vertex count only (no topology).
MeshConnectivity BuildConnectivity(std::span<const std::vector<uint32_t>> faces, uint32_t vertex_count);
MeshConnectivity BuildConnectivity(std::span<const std::array<uint32_t, 2>> edges, uint32_t vertex_count);

// Lightweight, copyable view over a mesh: its connectivity (owned by MeshStore) plus its vertex data (read via StoreId).
// Holds no ownership, the MeshStore entry is released when the entity's MeshHandle is destroyed.
// Obtain one via MeshStore::GetMesh(StoreId).
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    Mesh() = default;
    Mesh(const MeshStore &store, uint32_t store_id, const MeshConnectivity &c) : Store(&store), StoreId(store_id), C(&c) {}

    uint32_t VertexCount() const { return C->VertexCount; }
    uint32_t EdgeCount() const { return C->Edges.size(); }
    uint32_t FaceCount() const { return C->Faces.size(); }
    uint32_t HalfEdgeCount() const { return C->Halfedges.size(); }

    const vec3 &GetPosition(VH) const;
    const vec3 &GetNormal(VH) const;
    vec3 GetNormal(FH) const;
    std::span<const Vertex> GetVerticesSpan() const;

    uint32_t GetStoreId() const { return StoreId; }
    const MeshConnectivity &GetConnectivity() const { return *C; }
    uint32_t TriangleIndexCount() const; // Cached triangle count * 3

    // Halfedge navigation
    HH GetHalfedge(EH eh, uint32_t i) const {
        const auto h0 = C->Edges[*eh];
        return i == 0 ? h0 : (i == 1 && h0 ? C->Halfedges[*h0].Opposite : HH{});
    }
    HH GetOppositeHalfedge(HH hh) const { return C->Halfedges[*hh].Opposite; }
    EH GetEdge(HH hh) const { return C->HalfedgeToEdge[*hh]; }
    FH GetFace(HH hh) const { return C->Halfedges[*hh].Face; }
    VH GetFromVertex(HH) const;
    VH GetToVertex(HH hh) const { return C->Halfedges[*hh].Vertex; }

    // Valence
    bool Empty() const { return VertexCount() == 0; }
    uint32_t GetValence(VH) const;
    uint32_t GetValence(FH) const;

    // Geometric queries
    vec3 CalcFaceCentroid(FH) const;
    float CalcEdgeLength(HH) const;
    float CalcFaceArea(FH) const;
    VH FindNearestVertex(vec3) const;

    std::vector<uint32_t> CreateTriangleIndices() const; // Allocates + returns triangulated face indices
    void WriteTriangleIndices(std::span<uint32_t> dest) const; // Write triangulated face indices into dest
    void WriteEdgeIndices(std::span<uint32_t> dest) const; // Write edge line segment indices into dest

    // Iterators
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
        HH advance() const { return M->C->Halfedges[*CurrentHalfedge].Next; }
        operator bool() const { return bool(CurrentHalfedge); }
    };
    struct FaceVertexRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        FaceVertexIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceVertexIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    // Iterate the vertices of a face.
    // Use for range-based for loops:
    //   for (auto vh : fv_range(fh)) { ... }
    FaceVertexRange fv_range(FH fh) const { return {this, C->Faces[*fh].Halfedge}; }
    // Iterator positioned at the first vertex of a face.
    // Use for manual iterator control with pre-increment:
    //   auto it = cfv_iter(fh); auto v0 = **it; auto v1 = **(++it);
    FaceVertexIterator cfv_iter(FH fh) const { return {this, C->Faces[*fh].Halfedge, C->Faces[*fh].Halfedge}; }

    struct VertexOutgoingHalfedgeIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const {
            const auto opp = M->C->Halfedges[*CurrentHalfedge].Opposite;
            return opp ? M->C->Halfedges[*opp].Next : HH{};
        }
    };
    struct VertexOutgoingHalfedgeRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        VertexOutgoingHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh && *vh < C->OutgoingHalfedges.size() ? C->OutgoingHalfedges[*vh] : HH{}};
    }

    struct FaceHalfedgeIterator : CirculatorBase {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const { return M->C->Halfedges[*CurrentHalfedge].Next; }
    };
    struct FaceHalfedgeRange {
        const Mesh *Mesh; // Always valid, never null
        HH StartHalfedge;
        FaceHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    FaceHalfedgeRange fh_range(FH fh) const { return {this, C->Faces[*fh].Halfedge}; }

private:
    const MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    const MeshConnectivity *C{};
};

// Resolve an entity's MeshHandle to a Mesh view via the registry's MeshStore.
// GetMesh asserts the entity has a mesh, TryGetMesh returns nullopt when it doesn't.
Mesh GetMesh(const entt::registry &, entt::entity);
std::optional<Mesh> TryGetMesh(const entt::registry &, entt::entity);
bool HasMesh(const entt::registry &, entt::entity);
