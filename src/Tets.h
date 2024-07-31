#pragma once

#include <memory>
#include <string>

#include "Mesh/Mesh.h"

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
    Tets(const Mesh &, TetGenOptions);
    Tets(Tets &&);
    Tets(std::unique_ptr<tetgenio>);
    ~Tets();

    Tets &operator=(Tets &&) noexcept;

    std::unique_ptr<tetgenio> TetGen;

    static Tets CreateTets(const Mesh &, TetGenOptions);

    const tetgenio &operator*() const { return *TetGen; }
    const tetgenio *operator->() const { return TetGen.get(); }

    Mesh CreateMesh() const;
};
