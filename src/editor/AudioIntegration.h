#pragma once
#include "audio/AudioSystem.h"

enum class EventPass;
struct AudioDeviceResource;

void RegisterAudioComponentHandlers(state::Registry &);
// Follows the output configuration and the device's reported rate, placing the render threads with a reopened unit.
void ReconcileOutputDevice(state::Registry &, AudioDeviceResource &);
// Install finished modal solves and apply pending model rescales.
void ApplyCompletedModalSolves(state::Registry &, EventPass);
// Retune scaled objects and refresh listener gains after world-transform updates.
void UpdateModalPlacement(state::Registry &);
// Strike the objects hit by the displayed frame's contacts.
void UpdateAudioContacts(state::Registry &);

// Create the modal audio context and register its component handlers.
// Must run before a scene loads, so that loading one populates the bank.
// The output device is separate, and capture works without it.
void InitAudioSystem(state::Registry &);
// Destroy the modal audio context after any output device has stopped.
void DeinitAudioSystem(state::Registry &);
bool HasPendingModalSolves(const state::Registry &);
void CancelModalSolves(state::Registry &);
void RemoveAudioComponents(state::Registry &, state::Entity sound_entity);

void ApplyModalModel(state::Registry &, state::Entity sound_entity, const std::filesystem::path &);

// Assign sample[i] to mesh_vertices[i].
// Used by RealImpact initial load and mic swap.
// Unreferenced samples are released by the component handler.
void SetVertexSamples(
    state::Registry &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::span<LoadedSample>
);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path.
void AssignVertexSample(
    state::Registry &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices, const std::filesystem::path &, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`.
// Removes audio components if the sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    state::Registry &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

void Stop(state::Registry &, state::Entity sound_entity);
void SetModel(state::Registry &, state::Entity sound_entity, SoundVerticesModel);
