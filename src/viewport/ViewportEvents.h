#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"

#include <filesystem>

// Reactive tags and deferred requests consumed (and cleared) by ProcessComponentEvents.

struct MeshGeometryDirty {}; // Request overlay + element-state buffer refresh after mesh geometry changes
struct LightWireframeDirty {};

struct ProfileNextProcessComponentEvents {}; // Profile the next ProcessComponentEvents pass.

struct PendingSetStudioEnvironment {
    uint32_t Index;
};
struct PendingSetEditMode {
    Element Mode;
};
struct PendingShaderRecompile {};

// Pending mesh import (file load + texture uploads).
// Apply emits this; ProcessComponentEvents performs the GPU work, then removes it.
struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};
