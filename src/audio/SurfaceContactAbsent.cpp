#include "SurfaceContact.h"

void SurfaceAudioStateDelete::operator()(SurfaceAudioState *) const {}
void SurfaceRenderScratchDelete::operator()(SurfaceRenderScratch *) const {}

SurfaceAudioStatePtr MakeSurfaceAudioState() { return {}; }

void SurfaceAdoptVoices(ModalAudio &, ModalBank &, uint32_t) {}
uint32_t SurfaceVoiceCount(const ModalAudio &, uint32_t) { return 0; }
bool SurfaceRenderObject(ModalAudio &, ModalRenderScratch &, ModalBank &, uint32_t, std::span<const uint32_t>, float *, uint32_t) { return false; }
void SurfaceSilenceObject(ModalAudio &, uint32_t) {}
uint32_t SurfaceActiveVoices(const ModalAudio &) { return 0; }

void SurfaceInstallBank(ModalAudio &) {}
void RegisterSurfaceContactHandlers(state::Scene &) {}
void SurfaceUpdateContacts(state::Scene &) {}
float SurfaceRoughnessOf(const state::Scene &, state::Entity) { return 0.f; }
state::Entity ContactSurfaceNode(const state::Scene &, state::Entity, state::Entity body) { return body; }
