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
    PolyMesh(std::vector<vec3> &&vertices, std::vector<std::vector<uint>> &&faces);

    // Element counts
    uint VertexCount() const { return static_cast<uint>(Positions.size()); }
    uint EdgeCount() const { return static_cast<uint>(Edges.size()); }
    uint FaceCount() const { return static_cast<uint>(Faces.size()); }
    uint HalfEdgeCount() const { return static_cast<uint>(Halfedges.size()); }

    // Position access
    const vec3 &GetPosition(VH vh) const { return Positions[*vh]; }
    const float *GetPositionData() const { return &Positions[0][0]; }

    // Normal access
    const vec3 &GetNormal(VH vh) const { return Normals[*vh]; }
    const vec3 &GetNormal(FH fh) const { return Faces[*fh].Normal; }

    // Color access
    vec4 GetColor(FH fh) const { return Faces[*fh].Color; }
    void SetColor(FH fh, const vec4 &color) { Faces[*fh].Color = color; }

    // Halfedge navigation
    HH GetHalfedge(EH, uint i) const;
    HH GetOppositeHalfedge(HH) const;
    EH GetEdge(HH hh) const { return HalfedgeToEdge[*hh]; }
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

    struct CirculatorBase {
        const PolyMesh *Mesh{};
        HH CurrentHalfedge{};
        HH StartHalfedge{};

        CirculatorBase() = default;
        CirculatorBase(const PolyMesh *m, HH current, HH start)
            : Mesh(m), CurrentHalfedge(current), StartHalfedge(start) {}

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

        bool operator==(this auto const &self, const auto &other) {
            return self.CurrentHalfedge == other.CurrentHalfedge;
        }

        bool operator!=(this auto const &self, const auto &other) {
            return !(self == other);
        }
    };

    struct FaceVertexIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = VH;

        using CirculatorBase::CirculatorBase;

        VH operator*() const { return Mesh->GetToVertex(CurrentHalfedge); }
        HH advance() const { return Mesh->Halfedges[*CurrentHalfedge].Next; }
        bool IsValid() const { return CurrentHalfedge.IsValid(); }
    };
    struct FaceVertexRange {
        const PolyMesh *Mesh;
        HH StartHalfedge;
        FaceVertexIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        FaceVertexIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    FaceVertexRange fv_range(FH fh) const { return {this, Faces[*fh].Halfedge}; }
    FaceVertexIterator cfv_iter(FH fh) const { return {this, Faces[*fh].Halfedge, Faces[*fh].Halfedge}; }

    struct VertexOutgoingHalfedgeIterator : CirculatorBase {
        using difference_type = std::ptrdiff_t;
        using value_type = HH;

        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const {
            const auto opp = Mesh->Halfedges[*CurrentHalfedge].Opposite;
            return opp.IsValid() ? Mesh->Halfedges[*opp].Next : HH{};
        }
    };
    struct VertexOutgoingHalfedgeRange {
        const PolyMesh *Mesh;
        HH StartHalfedge;
        VertexOutgoingHalfedgeIterator begin() const { return {Mesh, StartHalfedge, StartHalfedge}; }
        VertexOutgoingHalfedgeIterator end() const { return {Mesh, HH{}, StartHalfedge}; } // Invalid HH as sentinel
    };
    VertexOutgoingHalfedgeRange voh_range(VH vh) const {
        return {this, vh.IsValid() && *vh < static_cast<int>(OutgoingHalfedges.size()) ? OutgoingHalfedges[*vh] : HH{}};
    }

    struct FaceHalfedgeIterator : CirculatorBase {
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = HH;

        using CirculatorBase::CirculatorBase;

        HH operator*() const { return CurrentHalfedge; }
        HH advance() const { return Mesh->Halfedges[*CurrentHalfedge].Next; }
    };
    struct FaceHalfedgeRange {
        const PolyMesh *Mesh; // Always valid, never null
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
        vec3 Normal{0};
        vec4 Color{1, 1, 1, 1};
    };

    // Vertex data
    std::vector<vec3> Positions;
    std::vector<vec3> Normals;
    std::vector<HH> OutgoingHalfedges;
    std::vector<Halfedge> Halfedges;
    std::vector<EH> HalfedgeToEdge; // Separate mapping for better cache locality
    std::vector<HH> Edges; // Maps edge index to one of its halfedges (index 0)
    std::vector<Face> Faces;

    void ComputeVertexNormals();
    void ComputeFaceNormals();
};

// obj/ply
std::optional<PolyMesh> ReadMesh(const std::filesystem::path &);
} // namespace he
