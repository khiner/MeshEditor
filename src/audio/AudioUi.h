#pragma once
#include "state/Entity.h"

// Draw the viewport-global audio synthesis controls.
void DrawGlobalSynthControls(state::Scene &, state::Entity viewport);

// Draws audio bank and fixed-pool utilization.
void DrawAudioDebug(const state::Scene &);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
void DrawObjectAudioControls(state::Scene &, state::Entity viewport, state::Entity sound_entity, state::Entity mesh_entity);

// Draw the in-flight modal solve jobs as a progress overlay anchored to the current window's lower-left corner. Call inside the viewport window.
void DrawModalJobsOverlay(state::Scene &);
void DrawAudioDeviceControls(state::Scene &, state::Entity);

struct ContactSurface;
struct AcousticMaterial;
/***** User interface *****/

void DrawContactSurfaceControls(state::Scene &, state::Entity sound_entity, const ContactSurface &, const AcousticMaterial &);
void DrawSurfaceSynthControls(state::Scene &, state::Entity viewport);
void DrawSurfaceContactDebug(const state::Scene &);
