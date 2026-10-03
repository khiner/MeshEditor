#pragma once
#include "gpu/Vertex.h"
#include "mesh/ElementAttributeView.h"

#include <array>
#include <optional>
#include <span>

struct GpuBuffers;
struct MeshBuffers;
struct Mesh;
namespace state { struct Scene; }

// The initial build visits each owner's mesh once, and the owners build concurrently.
// Thereafter these operations touch only changed finest meshlets and their spatial ancestors.
void BuildMeshletSpatial(state::Scene &, std::span<MeshBuffers *const>);
void ReplaceMeshletSpatial(state::Scene &, MeshBuffers &, std::span<const uint32_t> removed, std::span<const uint32_t> added);
void RefitMeshletSpatial(state::Scene &, MeshBuffers &, std::span<const uint32_t> changed);

struct SpatialSurfacePoint {
    std::array<uint32_t,3> Vertices{}; // Canonical vertex handles.
    vec3 Weights{0};
};
SpatialSurfacePoint ClosestMeshletPoint(const GpuBuffers &,const MeshBuffers &,
    std::span<const Vertex>,TriangleVertexView,vec3 point);
std::optional<double> MeshletEnclosedVolume(const GpuBuffers &,const MeshBuffers &,const Mesh &);
