#pragma once

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <memory>
#include <span>

// The surface-contact audio model, as the core modal path calls it.
//
// Two bodies resting or sliding on one another sound from the roughness of the surfaces in contact, which is a separate model from the force pulse a collision drives the modes with.
// That model lives entirely in src/audio/surface/, and SURFACE_AUDIO=1 at configure time selects it.
// Every declaration here has two implementations and the link takes exactly one: the model itself, or SurfaceContactAbsent.cpp, whose entry points do nothing and report nothing.

struct ModalAudio;
struct ModalBank;
struct ModalRenderScratch;

// The model's state, held by ModalAudio for as long as the audio system lives.
// The deleter is defined alongside whichever implementation is linked, so the core never needs the type.
struct SurfaceAudioState;
struct SurfaceAudioStateDelete {
    void operator()(SurfaceAudioState *) const;
};
using SurfaceAudioStatePtr = std::unique_ptr<SurfaceAudioState, SurfaceAudioStateDelete>;

// One renderer's working memory for the coupled kernel, allocated on its first sustained contact.
struct SurfaceRenderScratch;
struct SurfaceRenderScratchDelete {
    void operator()(SurfaceRenderScratch *) const;
};
using SurfaceRenderScratchPtr = std::unique_ptr<SurfaceRenderScratch, SurfaceRenderScratchDelete>;

// Null without the model.
SurfaceAudioStatePtr MakeSurfaceAudioState();

/***** Audio thread *****/

// Bring the bank's voices in line with the newest published contact set, once per callback.
void SurfaceAdoptVoices(ModalAudio &, ModalBank &, uint32_t frame_count);
// Sustained contacts object `o` holds.
uint32_t SurfaceVoiceCount(const ModalAudio &, uint32_t object);
// Render object `o` with its sustained contacts, its impacts included.
// False when it holds none, leaving it to the modal-only kernel.
bool SurfaceRenderObject(ModalAudio &, ModalRenderScratch &, ModalBank &, uint32_t object, std::span<const uint32_t> impacts, float *out, uint32_t frame_count);
// Drop every voice on an object that has fallen silent.
void SurfaceSilenceObject(ModalAudio &, uint32_t object);
// Sustained contacts live across the whole bank, for display.
uint32_t SurfaceActiveVoices(const ModalAudio &);

/***** Main thread *****/

// Release the pools and published sets, which address the bank being replaced.
void SurfaceInstallBank(ModalAudio &);
// Register the model's components, reactive tracking, and viewport controls.
void RegisterSurfaceContactHandlers(entt::registry &);
// Re-derive whatever the frame's edits changed, then publish this step's contacts as voices.
void SurfaceUpdateContacts(entt::registry &);
// Combined rms asperity height of a node's acoustic surface, m.
// Zero is ideally smooth, which is every surface without the model.
float SurfaceRoughnessOf(const entt::registry &, entt::entity node);
// The node whose acoustic surface a contact reads: the collider node it touched, or the nearest ancestor of it carrying one.
// The body itself when none does, which is every contact without the model.
entt::entity ContactSurfaceNode(const entt::registry &, entt::entity collider, entt::entity body);

/***** User interface *****/

// An object's acoustic surface, inside its modal model editor.
void DrawContactSurfaceControls(entt::registry &, entt::entity sound_entity);
// The viewport-global sustained-contact controls, inside the modal synthesis section.
void DrawSurfaceSynthControls(entt::registry &, entt::entity viewport);
// Where the surface model's fixed-size pools stand against demand, inside the audio debug view.
void DrawSurfaceContactDebug(const entt::registry &);
