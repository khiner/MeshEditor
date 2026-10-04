#pragma once
#include "mesh/MeshStore.h"
#include "gpu/Vertex.h"
#include "mesh/ElementAttributeView.h"

#include <array>
#include <optional>
#include <span>

struct Mesh;
namespace state { struct Scene; }

// The initial build visits each owner's mesh once, and the owners build concurrently.
// Thereafter these operations touch only changed finest meshlets and their spatial ancestors.
void BuildMeshletSpatial(state::Scene &, std::span<MeshStore::Record *const>);
void ReplaceMeshletSpatial(state::Scene &, const MeshStore::Record &, std::span<const uint32_t> removed, std::span<const uint32_t> added);
void RefitMeshletSpatial(state::Scene &, const MeshStore::Record &, std::span<const uint32_t> changed);

struct SpatialSurfacePoint {
    std::array<uint32_t,3> Vertices{}; // Canonical vertex handles.
    vec3 Weights{0};
};
SpatialSurfacePoint ClosestMeshletPoint(const RenderArenas &,const MeshStore::Record &,
    std::span<const Vertex>,TriangleVertexView,vec3 point);
std::optional<double> MeshletEnclosedVolume(const RenderArenas &,const MeshStore::Record &,const Mesh &);
