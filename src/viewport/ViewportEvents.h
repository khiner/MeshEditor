#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"

#include <filesystem>

// Reactive tags and deferred requests, consumed and cleared once per frame.

struct MeshGeometryDirty {}; // Request overlay + element-state buffer refresh after mesh geometry changes
struct LightWireframeDirty {};

struct ProfileNextProcessComponentEvents {};

struct PendingSetEditMode {
    Element Mode;
};
struct PendingShaderRecompile {};

// Pending mesh import (file load + texture uploads), consumed once the GPU work completes.
struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};
