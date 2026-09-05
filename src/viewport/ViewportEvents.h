#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"

#include <filesystem>

struct MeshGeometryDirty {};
struct MeshPositionsChanged {};
struct MeshShadingDirty {};
struct LightWireframeDirty {};

struct PendingSetEditMode {
    Element Mode;
};
struct PendingShaderRecompile {};

struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};
