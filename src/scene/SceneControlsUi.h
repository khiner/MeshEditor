#pragma once

#include "state/Entity.h"

#include <string>

// ImGui controls that view/edit scene and entity state (rendered outside the viewport image).
// Edits apply to the object, and for the viewport's active object with Alt held, to the selection.
void RenderEntityControls(state::Registry &, state::Entity object);
void RenderInteractionControls(state::Registry &); // Interaction mode, and in Edit mode the element type and the selected elements' controls.
void RenderSelectionActions(state::Registry &); // Actions on the selected objects.
void RenderControls(state::Registry &); // The viewport's scene and its objects, the viewport's display and view camera, and the audio device.
void RenderDocumentControls(state::Registry &); // The document's settings: simulation, audio synthesis, texture filtering, the material variant, and import metadata.
void DrawViewportStyle(state::Registry &); // Viewport theme colors and sizes.
// The scene's name, or "Scene N" from its source index when the name is empty.
std::string SceneLabel(const state::Registry &, state::Entity scene);
