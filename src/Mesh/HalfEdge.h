#pragma once

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

namespace he {

struct VertexHandle {
    int Index{-1};

    int operator*() const { return Index; }
    bool operator==(const VertexHandle &) const = default;

    bool IsValid() const { return Index >= 0; }
};

struct HalfedgeHandle {
    int Index{-1};

    int operator*() const { return Index; }
    bool operator==(const HalfedgeHandle &) const = default;

    bool IsValid() const { return Index >= 0; }
};

struct EdgeHandle {
    int Index{-1};

    int operator*() const { return Index; }
    bool operator==(const EdgeHandle &) const = default;

    bool IsValid() const { return Index >= 0; }
};

struct FaceHandle {
    int Index{-1};

    int operator*() const { return Index; }
    bool operator==(const FaceHandle &) const = default;

    bool IsValid() const { return Index >= 0; }
};

using VH = VertexHandle;
using HH = HalfedgeHandle;
using EH = EdgeHandle;
using FH = FaceHandle;

// Core half-edge mesh structure
struct PolyMesh {
    PolyMesh() = default;
    PolyMesh(const PolyMesh &) = default;
    PolyMesh(PolyMesh &&) = default;
    PolyMesh &operator=(const PolyMesh &) = default;
    PolyMesh &operator=(PolyMesh &&) = default;

    // Element counts
    uint VertexCount() const { return static_cast<uint>(Positions.size()); }
    uint EdgeCount() const { return static_cast<uint>(Edges.size()); }
    uint FaceCount() const { return static_cast<uint>(Faces.size()); }
    uint HalfEdgeCount() const { return static_cast<uint>(Halfedges.size()); }

    // Add elements
    VH AddVertex(const vec3 &);
    FH AddFace(const std::vector<VH> &);

    // Position access
    const vec3 &GetPosition(VH vh) const { return Positions[*vh]; }
    const float *GetPositionData() const { return &Positions[0][0]; }

    // Normal access
    const vec3 &GetNormal(VH vh) const { return Normals[*vh]; }
    const vec3 &GetNormal(FH fh) const { return Faces[*fh].Normal; }
    void UpdateNormals();

    // Color access
    vec4 GetColor(FH fh) const { return Faces[*fh].Color; }
    void SetColor(FH fh, const vec4 &color) { Faces[*fh].Color = color; }

    // Halfedge navigation
    HH GetHalfedge(EH, uint i) const;
    HH GetOppositeHalfedge(HH) const;
    EH GetEdge(HH hh) const { return Halfedges[*hh].Edge; }
    FH GetFace(HH hh) const { return Halfedges[*hh].Face; }
    VH GetFromVertex(HH) const;
    VH GetToVertex(HH hh) const { return Halfedges[*hh].Vertex; }

    // Valence
    uint GetValence(VH) const;
    uint GetValence(FH) const;

    // Geometric queries
    vec3 CalcFaceCentroid(FH) const;
    float CalcEdgeLength(HH) const;

    // Iterators - simple range-based for loop support
    struct VertexIterator {
        int Index;
        VH operator*() const { return VH(Index); }
        VertexIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator!=(const VertexIterator &other) const { return Index != other.Index; }
    };
    struct VertexRange {
        uint Count;
        VertexIterator begin() const { return {0}; }
        VertexIterator end() const { return {static_cast<int>(Count)}; }
    };
    VertexRange vertices() const { return {VertexCount()}; }

    struct EdgeIterator {
        int Index;
        EH operator*() const { return EH(Index); }
        EdgeIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator!=(const EdgeIterator &other) const { return Index != other.Index; }
    };
    struct EdgeRange {
        uint Count;
        EdgeIterator begin() const { return {0}; }
        EdgeIterator end() const { return {static_cast<int>(Count)}; }
    };
    EdgeRange edges() const { return {EdgeCount()}; }

    struct FaceIterator {
        int Index;
        FH operator*() const { return FH(Index); }
        FaceIterator &operator++() {
            ++Index;
            return *this;
        }
        bool operator!=(const FaceIterator &other) const { return Index != other.Index; }
    };
    struct FaceRange {
        uint Count;
        FaceIterator begin() const { return {0}; }
        FaceIterator end() const { return {static_cast<int>(Count)}; }
    };
    FaceRange faces() const { return {FaceCount()}; }

    // Circulator-style ranges
    struct FaceVertexIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = VH;

        const PolyMesh *Mesh{};
        HH CurrentHalfedge{};
        HH StartHalfedge{};
        bool First{true};

        FaceVertexIterator() = default;
        FaceVertexIterator(const PolyMesh *m, HH current, HH start, bool first)
            : Mesh(m), CurrentHalfedge(current), StartHalfedge(start), First(first) {}

        VH operator*() const { return Mesh->GetToVertex(CurrentHalfedge); }
        FaceVertexIterator &operator++();
        FaceVertexIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const FaceVertexIterator &other) const {
            return !First && !other.First && CurrentHalfedge == other.CurrentHalfedge;
        }

        bool IsValid() const { return First || CurrentHalfedge != StartHalfedge; }
    };
    struct FaceVertexRange {
        const PolyMesh *Mesh;
        HH StartHalfedge;
        FaceVertexIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge, true}; }
        FaceVertexIterator end() const { return {Mesh, StartHalfedge, StartHalfedge, false}; }
    };
    FaceVertexRange fv_range(FH fh) const { return {this, Faces[*fh].Halfedge}; }
    FaceVertexIterator cfv_iter(FH fh) const { return {this, Faces[*fh].Halfedge, Faces[*fh].Halfedge, true}; }

    struct VertexOutgoingHalfedgeIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = HH;

        const PolyMesh *Mesh{};
        HH CurrentHalfedge{};
        HH StartHalfedge{};
        bool First{true};

        VertexOutgoingHalfedgeIterator() = default;
        VertexOutgoingHalfedgeIterator(const PolyMesh *m, HH current, HH start, bool first)
            : Mesh(m), CurrentHalfedge(current), StartHalfedge(start), First(first) {}

        HH operator*() const { return CurrentHalfedge; }
        VertexOutgoingHalfedgeIterator &operator++();
        VertexOutgoingHalfedgeIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const VertexOutgoingHalfedgeIterator &other) const {
            return !First && !other.First && CurrentHalfedge == other.CurrentHalfedge;
        }
    };
    struct VertexOutgoingHalfedgeRange {
        const PolyMesh *Mesh;
        HH StartHalfedge;
        VertexOutgoingHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge, true}; }
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, StartHalfedge, StartHalfedge, false}; }
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh.IsValid() && *vh < static_cast<int>(OutgoingHalfedges.size()) ? OutgoingHalfedges[*vh] : HH{}};
    }

    struct FaceHalfedgeIterator {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = HH;
        using pointer = const HH *;
        using reference = HH;

        const PolyMesh *Mesh{};
        HH CurrentHalfedge{};
        HH StartHalfedge{};
        bool First{true};

        FaceHalfedgeIterator() = default;
        FaceHalfedgeIterator(const PolyMesh *m, HH current, HH start, bool first)
            : Mesh(m), CurrentHalfedge(current), StartHalfedge(start), First(first) {}

        HH operator*() const { return CurrentHalfedge; }
        FaceHalfedgeIterator &operator++();
        FaceHalfedgeIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator==(const FaceHalfedgeIterator &other) const {
            return !First && !other.First && CurrentHalfedge == other.CurrentHalfedge;
        }
        bool operator!=(const FaceHalfedgeIterator &other) const {
            return First || CurrentHalfedge != other.CurrentHalfedge;
        }
    };
    struct FaceHalfedgeRange {
        const PolyMesh *Mesh;
        HH StartHalfedge;
        FaceHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge, true}; }
        FaceHalfedgeIterator end() const { return {Mesh, StartHalfedge, StartHalfedge, false}; }
    };
    FaceHalfedgeRange fh_range(FH fh) const { return {this, Faces[*fh].Halfedge}; }

private:
    struct Halfedge {
        VH Vertex; // To vertex
        HH Next; // Next halfedge in face
        HH Opposite; // Opposite halfedge
        EH Edge; // Parent edge
        FH Face; // Left face (invalid for boundary halfedges)
    };

    struct Edge {
        HH Halfedge; // One of the two halfedges (index 0)
    };

    struct Face {
        HH Halfedge; // One of the boundary halfedges
        vec3 Normal{0};
        vec4 Color{1, 1, 1, 1};
    };

    // Vertex data
    std::vector<vec3> Positions;
    std::vector<vec3> Normals;
    std::vector<HH> OutgoingHalfedges;
    std::vector<Halfedge> Halfedges;
    std::vector<Edge> Edges;
    std::vector<Face> Faces;

    // Map to track halfedges by their vertex pairs for opposite finding
    std::unordered_map<uint64_t, HH> HalfedgeMap;

    void ComputeVertexNormals();
    void ComputeFaceNormals();
};

// obj/ply
std::optional<PolyMesh> ReadMesh(const std::filesystem::path &);
} // namespace he
