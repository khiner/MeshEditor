#pragma once

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <memory>
#include <span>

// SURFACE_AUDIO selects either the surface-contact model or inert entry points at link time.

struct ModalAudio;
struct ModalBank;
struct ModalRenderScratch;

// The selected implementation defines the deleter for this opaque state.
struct SurfaceAudioState;
struct SurfaceAudioStateDelete {
    void operator()(SurfaceAudioState *) const;
};
using SurfaceAudioStatePtr = std::unique_ptr<SurfaceAudioState, SurfaceAudioStateDelete>;

// Per-renderer scratch allocated on the first sustained contact.
struct SurfaceRenderScratch;
struct SurfaceRenderScratchDelete {
    void operator()(SurfaceRenderScratch *) const;
};
using SurfaceRenderScratchPtr = std::unique_ptr<SurfaceRenderScratch, SurfaceRenderScratchDelete>;

SurfaceAudioStatePtr MakeSurfaceAudioState();

/***** Audio thread *****/

// Synchronizes bank voices with the latest contact set once per callback.
void SurfaceAdoptVoices(ModalAudio &, ModalBank &, uint32_t frame_count);
// Returns the sustained-contact count for object `o`.
uint32_t SurfaceVoiceCount(const ModalAudio &, uint32_t object);
// Renders object `o` and returns false when modal-only rendering is sufficient.
bool SurfaceRenderObject(ModalAudio &, ModalRenderScratch &, ModalBank &, uint32_t object, std::span<const uint32_t> impacts, float *out, uint32_t frame_count);
// Removes all voices for an inactive object.
void SurfaceSilenceObject(ModalAudio &, uint32_t object);
// Returns the active sustained-contact count across the bank.
uint32_t SurfaceActiveVoices(const ModalAudio &);

/***** Main thread *****/

// Releases state that references the replaced bank.
void SurfaceInstallBank(ModalAudio &);
void RegisterSurfaceContactHandlers(entt::registry &);
// Recomputes edited surface state and publishes the current contacts.
void SurfaceUpdateContacts(entt::registry &);
// Returns combined RMS asperity height in meters, or zero without the model.
float SurfaceRoughnessOf(const entt::registry &, entt::entity node);
// Returns the collider or nearest ancestor with an acoustic surface, otherwise the body.
entt::entity ContactSurfaceNode(const entt::registry &, entt::entity collider, entt::entity body);

/***** User interface *****/

void DrawContactSurfaceControls(entt::registry &, entt::entity sound_entity);
void DrawSurfaceSynthControls(entt::registry &, entt::entity viewport);
void DrawSurfaceContactDebug(const entt::registry &);
