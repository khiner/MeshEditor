#include "mesh/LevelSet.h"
#include "mesh/MeshStore.h"
#include <cmath>

uint32_t Lattice::Count() const {
    return uint32_t(Size[0]) * uint32_t(Size[1]) * uint32_t(Size[2]) + uint32_t(Size[0] + 1) * uint32_t(Size[1] + 1) * uint32_t(Size[2] + 1);
}
std::vector<vec3> Lattice::Points() const {
    std::vector<vec3> points;
    points.reserve(Count());
    for (const int32_t w : {1, 0}) {
        const int32_t extra = w == 0 ? 1 : 0;
        for (int32_t x = 0; x < Size[0] + extra; ++x) {
            for (int32_t y = 0; y < Size[1] + extra; ++y) {
                for (int32_t z = 0; z < Size[2] + extra; ++z) points.push_back(Origin + Spacing * (vec3{float(x), float(y), float(z)} + (w == 1 ? 0.f : -.5f)));
            }
        }
    }
    return points;
}

std::optional<Lattice> LatticeOver(vec3 min, vec3 max, float edge_length) {
    const auto dim = max - min;
    if (!std::isfinite(edge_length) || !(edge_length > 0.f) || !std::isfinite(dim.x + dim.y + dim.z) || !(dim.x > 0.f && dim.y > 0.f && dim.z > 0.f)) return {};
    Lattice l{.Origin = min};
    for (int axis = 0; axis < 3; ++axis) {
        const auto size = std::floor(dim[axis] / edge_length + 1.f);
        if (!(size < 1024.f)) return {};
        l.Size[axis] = int32_t(size);
        l.Spacing[axis] = dim[axis] / float(l.Size[axis] - 1);
    }
    if (l.Size[0] < 2 || l.Size[1] < 2 || l.Size[2] < 2) return {};
    return l;
}

MeshTopologyTask LevelSetTask(const MeshStore &meshes, uint32_t id, const Lattice &lattice, uint32_t attribute, float level) {
    return {
        .SourceId = id,
        .Op = MeshTopologyOp::LevelSet,
        .Param1 = level,
        .Param0Attribute = meshes.AttributeRef(attribute),
        .Flags = TopologyFlagSelectAll,
        .LatticeSize = {uint32_t(lattice.Size[0]), uint32_t(lattice.Size[1]), uint32_t(lattice.Size[2])},
    };
}
