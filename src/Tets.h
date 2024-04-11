#pragma once

#include <memory>
#include <string>

#include "mesh/Mesh.h"

// See https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html
struct TetGenOptions {
    // (-Y) Input boundary edges and faces of the PLC are preserved in the generated tetrahedral mesh.
    // Steiner points appear only in the interior space of the PLC.
    bool PreserveSurface{false};
    // (-q) Adds new points to improve the mesh quality.
    bool Quality{false};

    std::string CreateFlags() const;
};

class tetgenio;

struct Tets {
    Tets(Tets &&);
    Tets(std::unique_ptr<tetgenio>);
    ~Tets();

    std::unique_ptr<tetgenio> TetGen;
    uint MeshEntity{0};

    const tetgenio &operator*() const { return *TetGen; }

    Mesh GenerateMesh() const;
};

Tets GenerateTets(const Mesh &, TetGenOptions);
