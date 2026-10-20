#pragma once
#include "state/Entity.h"

// Draw the viewport-global audio synthesis controls.
void DrawGlobalSynthControls(state::Registry &);

// Draws audio bank and fixed-pool utilization.
void DrawAudioDebug(const state::Registry &);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
void DrawObjectAudioControls(state::Registry &, state::Entity sound_entity, state::Entity mesh_entity);

// Draw the in-flight modal solve jobs as a progress overlay anchored to the current window's lower-left corner.
// Call inside the viewport panel.
void DrawModalJobsOverlay(state::Registry &);
void DrawAudioDeviceControls(state::Registry &);

struct ContactSurface;
struct AcousticMaterial;

void DrawContactSurfaceControls(state::Registry &, state::Entity sound_entity, const ContactSurface &, const AcousticMaterial &);
void DrawSurfaceSynthControls(state::Registry &);
void DrawSurfaceContactDebug(const state::Registry &);
