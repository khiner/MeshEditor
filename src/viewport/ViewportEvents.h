#pragma once

#include "gpu/Element.h"
#include "object/ObjectCreateInfo.h"
#include "viewport/ViewportSchema.h"

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
    bool RenderReady{}; // Canonical meshlets and normals already repaired or restored.
};
struct MeshPositionsChanged {};

struct PendingSetEditMode {
    Element Mode;
};

struct PendingImportMesh {
    std::filesystem::path Path;
    MeshInstanceCreateInfo Info;
};

// The lighting a presentation shows: the material preview, or the rendered scene lit by its image light or by its punctual lights.
enum class PresentationLighting : uint8_t {
    MaterialPreview,
    SceneImageLight,
    SceneLights,
};

// The view a recorded action asks for, which the active viewport takes in its next frame step.
// Navigation and replay remove it, so only a live action sets a view up.
struct PendingViewSetup {
    std::optional<float> WorldOpacity; // The imported world's opacity under rendered lighting.
    state::Entity LookThroughEntity{state::Null}; // The scene camera to look through, or null.
    std::optional<PresentationLighting> Presentation;
};
