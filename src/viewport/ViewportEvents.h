#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"

#include <filesystem>

// The mesh's edit selection after a geometry change.
// A restore keeps it, a rebuild drops it, and a topology operator carries it onto the output and derives the rest.
enum class EditSelectionAfter : uint8_t {
    Keep,
    Reset,
    Derive,
};
struct MeshGeometryDirty {
    EditSelectionAfter Selection{EditSelectionAfter::Reset};
};
struct MeshPositionsChanged {};

struct PendingSetEditMode {
    Element Mode;
};

struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};
