#include "SurfaceContact.h"

// The surface-contact model's entry points where the model itself is absent: each does nothing and reports nothing, so a scene sounds by collision alone.
// src/audio/surface/ holds the implementations these stand in for, which SURFACE_AUDIO=1 at configure time selects instead.

void SurfaceAudioStateDelete::operator()(SurfaceAudioState *) const {}
void SurfaceRenderScratchDelete::operator()(SurfaceRenderScratch *) const {}

SurfaceAudioStatePtr MakeSurfaceAudioState() { return {}; }

void SurfaceAdoptVoices(ModalAudio &, ModalBank &, uint32_t) {}
uint32_t SurfaceVoiceCount(const ModalAudio &, uint32_t) { return 0; }
bool SurfaceRenderObject(ModalAudio &, ModalRenderScratch &, ModalBank &, uint32_t, std::span<const uint32_t>, float *, uint32_t) { return false; }
void SurfaceSilenceObject(ModalAudio &, uint32_t) {}
uint32_t SurfaceActiveVoices(const ModalAudio &) { return 0; }

void SurfaceInstallBank(ModalAudio &) {}
void RegisterSurfaceContactHandlers(entt::registry &) {}
void SurfaceUpdateContacts(entt::registry &) {}
float SurfaceRoughnessOf(const entt::registry &, entt::entity) { return 0.f; }
entt::entity ContactSurfaceNode(const entt::registry &, entt::entity, entt::entity body) { return body; }

void DrawContactSurfaceControls(entt::registry &, entt::entity) {}
void DrawSurfaceSynthControls(entt::registry &, entt::entity) {}
void DrawSurfaceContactDebug(const entt::registry &) {}
