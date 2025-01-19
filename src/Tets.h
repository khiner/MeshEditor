#pragma once

#include <memory>

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

std::unique_ptr<tetgenio> GenerateTets(const Mesh &, TetGenOptions);
