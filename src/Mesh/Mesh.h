#pragma once

#include "Handle.h"
#include "MeshData.h"
#include "numeric/vec4.h"

#include <span>
#include <unordered_map>
#include <vector>

using uint = uint32_t;

struct MeshStore;

// Half-edge polymesh data structure
struct Mesh {
    using VH = he::VH;
    using HH = he::HH;
    using EH = he::EH;
    using FH = he::FH;

    static constexpr vec4 DefaultMeshColor{0.7, 0.7, 0.7, 1};

    Mesh(MeshStore &, uint32_t store_id, std::vector<std::vector<uint>> &&faces);

    Mesh(const Mesh &) = delete;
    Mesh &operator=(const Mesh &) = delete;
    Mesh(Mesh &&) noexcept;
    Mesh &operator=(Mesh &&) noexcept;
    ~Mesh();

    uint VertexCount() const { return Vertices.size(); }
    uint EdgeCount() const { return Edges.size(); }
    uint FaceCount() const { return Faces.size(); }
    uint HalfEdgeCount() const { return Halfedges.size(); }

    const vec3 &GetPosition(VH vh) const { return Vertices[*vh].Position; }

    const vec3 &GetNormal(VH vh) const { return Vertices[*vh].Normal; }
    const vec3 &GetNormal(FH fh) const { return FaceNormals[*fh]; }

    uint32_t GetStoreId() const { return StoreId; }

    MeshData ToMeshData() const;

    // Halfedge navigation
    HH GetHalfedge(EH eh, uint i) const {
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
    uint GetValence(VH) const;
    uint GetValence(FH) const;

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

    std::vector<uint> CreateTriangleIndices() const; // Triangulated face indices
    std::vector<uint> CreateTriangulatedFaceIndices() const; // Triangle fan for each face
    std::vector<uint> CreateEdgeIndices() const; // Edge line segment indices

    // Iterators
    struct VertexIterator {
        uint Index;
        VH operator*() const { return {Index}; }
        VertexIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const VertexIterator &) const = default;
    };
    struct VertexRange {
        uint Count;
        VertexIterator begin() const { return {0}; }
        VertexIterator end() const { return {Count}; }
    };
    VertexRange vertices() const { return {VertexCount()}; }

    struct EdgeIterator {
        uint Index;
        EH operator*() const { return {Index}; }
        EdgeIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const EdgeIterator &) const = default;
    };
    struct EdgeRange {
        uint Count;
        EdgeIterator begin() const { return {0}; }
        EdgeIterator end() const { return {Count}; }
    };
    EdgeRange edges() const { return {EdgeCount()}; }

    struct FaceIterator {
        uint Index;
        FH operator*() const { return {Index}; }
        FaceIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator==(const FaceIterator &) const = default;
    };
    struct FaceRange {
        uint Count;
        FaceIterator begin() const { return {0}; }
        FaceIterator end() const { return {Count}; }
    };
    FaceRange faces() const { return {FaceCount()}; }

    struct CirculatorBase {
        const Mesh *M{};
        HH CurrentHalfedge{};
        HH StartHalfedge{};

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
        operator bool() const { return CurrentHalfedge; }
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
    static constexpr uint32_t InvalidStoreId{~0u};
    MeshStore *Store{};
    uint32_t StoreId{InvalidStoreId};
    std::span<const Vertex3D> Vertices;
    std::span<const vec3> FaceNormals;
    std::vector<HH> OutgoingHalfedges;
    std::vector<Halfedge> Halfedges;
    std::vector<EH> HalfedgeToEdge; // Separate mapping for better cache locality
    std::vector<HH> Edges; // Maps edge index to one of its halfedges (index 0)
    std::vector<Face> Faces;

    vec4 Color{DefaultMeshColor};

    void ComputeVertexNormals();
    void ComputeFaceNormals();
};
