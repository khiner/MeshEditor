#pragma once
#include "audio/AudioSystem.h"

enum class EventPass;

void RegisterAudioComponentHandlers(state::Scene &);
// Install finished modal solves and apply pending model rescales.
void ApplyCompletedModalSolves(state::Scene &, EventPass);
// Strike the objects hit by the displayed frame's contacts.
void UpdateAudioContacts(state::Scene &);
// Drop the cleared scene's bank slots, samples, warm-start data, and in-flight solves.
void ClearAudioScene(state::Scene &);

// Create the modal audio context and register its component handlers.
// Must run before a scene loads, so that loading one populates the bank.
// The output device is separate, and capture works without it.
void InitAudioSystem(state::Scene &);
// Destroy the modal audio context after any output device has stopped.
void DeinitAudioSystem(state::Scene &);
bool HasPendingModalSolves(const state::Scene &);
void CancelModalSolves(state::Scene &);
void RemoveAudioComponents(state::Scene &, state::Entity sound_entity);

void ApplyModalModel(state::Scene &, state::Entity sound_entity, const std::filesystem::path &);

// Assign sample[i] to mesh_vertices[i]. Used by RealImpact initial load and mic swap.
// Unreferenced samples are released by the component handler.
void SetVertexSamples(
    state::Scene &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::span<LoadedSample>
);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path.
void AssignVertexSample(
    state::Scene &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::filesystem::path, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`.
// Removes audio components if the sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    state::Scene &, state::Entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

void Stop(state::Scene &, state::Entity sound_entity);
void SetModel(state::Scene &, state::Entity sound_entity, SoundVerticesModel);
