#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"

#include <filesystem>

struct MeshGeometryDirty {
    bool ResetSelection{true};
};
struct MeshPositionsChanged {};

struct PendingSetEditMode {
    Element Mode;
};

struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};
