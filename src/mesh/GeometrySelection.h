#pragma once

#include "gpu/Element.h"
#include <stdexcept>
#include <vector>

// Explicit, sorted canonical handles. Empty domains select nothing.
struct GeometrySelection {
    std::vector<uint32_t> Vertices, Edges, Faces;
    const std::vector<uint32_t> &Get(Element element) const {
        switch (element) {
            case Element::Vertex: return Vertices;
            case Element::Edge: return Edges;
            case Element::Face: return Faces;
            default: throw std::invalid_argument("Geometry selection requires an element domain.");
        }
    }
};

struct MeshStore;
void ValidateGeometrySelection(const MeshStore &, uint32_t id, const GeometrySelection &);
