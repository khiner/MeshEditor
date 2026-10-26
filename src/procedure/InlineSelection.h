#pragma once

#include "mesh/GeometrySelection.h"

namespace procedure {
// Persist selection masks in input-block order, the edit domain they were captured in, and the active element's handle.
// Absent or empty masks select everything.
struct InlineSelection {
    std::vector<SelectionBlock> Vertices, Edges, Faces;
    Element Domain{Element::Vertex};
    uint32_t Active{InvalidOffset};
};
} // namespace procedure
