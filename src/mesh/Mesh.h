#pragma once

#include "gpu/Element.h"
#include "gpu/Vertex.h"

#include <array>
#include <limits>
#include <span>
#include <string_view>
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

// Most recently selected element per mesh (remembered even when not selected).
struct MeshActiveElement {
    uint32_t Handle;
};

// Half-edge polymesh data structure
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    Mesh(MeshStore &, uint32_t store_id, std::vector<std::vector<uint32_t>> &&faces);
    Mesh(MeshStore &, uint32_t store_id, std::vector<std::array<uint32_t, 2>> &&edges, uint32_t vertex_count);
    Mesh(MeshStore &, uint32_t store_id, uint32_t vertex_count);
    Mesh(MeshStore &, uint32_t store_id, const Mesh &src);

    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&) noexcept;
    Mesh &operator=(Mesh &&) noexcept;
    ~Mesh();

    uint32_t VertexCount() const { return VertexCountValue; }
    uint32_t EdgeCount() const { return Edges.size(); }
    uint32_t FaceCount() const { return Faces.size(); }
    uint32_t HalfEdgeCount() const { return Halfedges.size(); }

    const vec3 &GetPosition(VH) const;
    const vec3 &GetNormal(VH) const;
    vec3 GetNormal(FH) const;
    std::span<const Vertex> GetVerticesSpan() const;

    uint32_t GetStoreId() const { return StoreId; }
    uint32_t TriangleIndexCount() const; // Cached triangle count * 3

    // Halfedge navigation
    HH GetHalfedge(EH eh, uint32_t i) const {
        const auto h0 = Edges[*eh];
        return i == 0 ? h0 : (i == 1 && h0 ? Halfedges[*h0].Opposite : HH{});
    }
    HH GetOppositeHalfedge(HH hh) const { return Halfedges[*hh].Opposite; }
    EH GetEdge(HH hh) const { return HalfedgeToEdge[*hh]; }
    FH GetFace(HH hh) const { return Halfedges[*hh].Face; }
    VH GetFromVertex(HH) const;
    VH GetToVertex(HH hh) const { return Halfedges[*hh].Vertex; }

    // Valence
    bool Empty() const { return VertexCount() == 0; }
    uint32_t GetValence(VH) const;
    uint32_t GetValence(FH) const;

    // Geometric queries
    vec3 CalcFaceCentroid(FH) const;
    float CalcEdgeLength(HH) const;
    float CalcFaceArea(FH) const;
    VH FindNearestVertex(vec3) const;

    // Topological queries
    bool VertexBelongsToFace(VH, FH) const;
    bool VertexBelongsToEdge(VH, EH) const;
    bool VertexBelongsToFaceEdge(VH, FH, EH) const;
    bool EdgeBelongsToFace(EH, FH) const;

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
        HH advance() const { return M->Halfedges[*CurrentHalfedge].Next; }
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
    FaceVertexRange fv_range(FH fh) const { return {this, Faces[*fh].Halfedge}; }
    // Iterator positioned at the first vertex of a face.
    // Use for manual iterator control with pre-increment:
    //   auto it = cfv_iter(fh); auto v0 = **it; auto v1 = **(++it);
    FaceVertexIterator cfv_iter(FH fh) const { return {this, Faces[*fh].Halfedge, Faces[*fh].Halfedge}; }

    struct VertexOutgoingHalfedgeIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const {
            const auto opp = M->Halfedges[*CurrentHalfedge].Opposite;
            return opp ? M->Halfedges[*opp].Next : HH{};
        }
    };
    struct VertexOutgoingHalfedgeRange {
        const Mesh *Mesh;
        HH StartHalfedge;
        VertexOutgoingHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh && *vh < OutgoingHalfedges.size() ? OutgoingHalfedges[*vh] : HH{}};
    }

    struct FaceHalfedgeIterator : CirculatorBase {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const { return M->Halfedges[*CurrentHalfedge].Next; }
    };
    struct FaceHalfedgeRange {
        const Mesh *Mesh; // Always valid, never null
        HH StartHalfedge;
        FaceHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    FaceHalfedgeRange fh_range(FH fh) const { return {this, Faces[*fh].Halfedge}; }

private:
    struct Halfedge {
        VH Vertex; // To vertex
        HH Next; // Next halfedge in face
        HH Opposite; // Opposite halfedge
        FH Face; // Left face (invalid for boundary halfedges)
    };

    struct Face {
        HH Halfedge; // One of the boundary halfedges
    };

    // Vertex data (ranges stay stable until a mesh resizes its vertex count)
    MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    uint32_t VertexCountValue{0};
    std::vector<HH> OutgoingHalfedges;
    std::vector<Halfedge> Halfedges;
    std::vector<EH> HalfedgeToEdge;
    std::vector<HH> Edges; // Maps edge index to its halfedge at index 0
    std::vector<Face> Faces;
};
