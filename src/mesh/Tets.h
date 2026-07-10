#pragma once

#include "mesh/TetMeshData.h"

#include "numeric/vec3.h"

// See https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html
struct TetGenOptions {
    // (-Y) Input boundary edges and faces of the PLC are preserved in the generated tetrahedral mesh.
    // Steiner points appear only in the interior space of the PLC.
    bool PreserveSurface{false};
    // (-q) Adds new points to improve the mesh quality.
    bool Quality{false};
    // Fraction of surface triangles kept for tetrahedralization. Below 1, the surface is quadric-simplified first.
    float SimplifyRatio{1};
};

class tetgenio;

std::unique_ptr<tetgenio> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, TetGenOptions);

// positions are divided by `scale`.
TetMeshData BuildTetMeshData(const tetgenio &, vec3 scale);
