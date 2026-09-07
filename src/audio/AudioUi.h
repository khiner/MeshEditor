#pragma once
#include "entt_fwd.h"

// Draw the viewport-global audio synthesis controls.
void DrawGlobalSynthControls(entt::registry &, entt::entity viewport);

// Draws audio bank and fixed-pool utilization.
void DrawAudioDebug(const entt::registry &);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
void DrawObjectAudioControls(entt::registry &, entt::entity viewport, entt::entity sound_entity, entt::entity mesh_entity);

// Draw the in-flight modal solve jobs as a progress overlay anchored to the current window's lower-left corner. Call inside the viewport window.
void DrawModalJobsOverlay(entt::registry &);
void DrawAudioDeviceControls(entt::registry &, entt::entity);

struct ContactSurface;
struct AcousticMaterial;
/***** User interface *****/

void DrawContactSurfaceControls(entt::registry &, entt::entity sound_entity, const ContactSurface &, const AcousticMaterial &);
void DrawSurfaceSynthControls(entt::registry &, entt::entity viewport);
void DrawSurfaceContactDebug(const entt::registry &);
