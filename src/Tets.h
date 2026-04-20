#pragma once

#include <memory>
#include <vector>

#include "numeric/vec3.h"

struct Mesh;

// See https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html
struct TetGenOptions {
    // (-Y) Input boundary edges and faces of the PLC are preserved in the generated tetrahedral mesh.
    // Steiner points appear only in the interior space of the PLC.
    bool PreserveSurface{false};
    // (-q) Adds new points to improve the mesh quality.
    bool Quality{false};
};

class tetgenio;

std::unique_ptr<tetgenio> GenerateTets(const Mesh &, vec3 scale, TetGenOptions);

struct TetMeshData {
    std::vector<vec3> Positions;
    std::vector<uint32_t> EdgeIndices;
};

// positions are divided by `scale`.
TetMeshData BuildTetMeshData(const tetgenio &, vec3 scale);
