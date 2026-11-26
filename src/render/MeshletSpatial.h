#pragma once
#include "mesh/MeshStore.h"

struct Mesh;
namespace state {
struct Registry;
} // namespace state

// The initial build visits each owner's mesh once, and the owners build concurrently.
// Thereafter a replace touches only changed finest meshlets and their spatial ancestors, and so does a refit of a small part of a tree, which otherwise builds the tree again.
void BuildMeshletSpatial(state::Registry &, std::span<MeshStore::Record *const>);
void ReplaceMeshletSpatial(state::Registry &, const MeshStore::Record &, std::span<const uint32_t> removed, std::span<const uint32_t> added);
void RefitMeshletSpatial(state::Registry &, const MeshStore::Record &, std::span<const uint32_t> changed);

struct SpatialSurfacePoint {
    std::array<uint32_t, 3> Vertices{}; // Canonical vertex handles.
    vec3 Weights{0};
};
SpatialSurfacePoint ClosestMeshletPoint(const RenderArenas &, const MeshStore::Record &, HandleValues<Vertex>, TriangleVertexView, vec3 point);
std::optional<double> MeshletEnclosedVolume(const RenderArenas &, const MeshStore::Record &, const Mesh &);
