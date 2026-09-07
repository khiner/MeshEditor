#pragma once
#include "audio/AudioSystem.h"

void RegisterAudioComponentHandlers(entt::registry &);

// Create the modal audio context and register its component handlers.
// Must run before a scene loads, so that loading one populates the bank.
// The output device is separate, and capture works without it.
void InitAudioSystem(entt::registry &);
// Destroy the modal audio context after any output device has stopped.
void DeinitAudioSystem(entt::registry &);
bool HasPendingModalSolves(const entt::registry &);
void RemoveAudioComponents(entt::registry &, entt::entity sound_entity);

// Apply a modal solve result file (relative to ModalModelsDir()) to the sound entity.
void ApplyModalModel(entt::registry &, entt::entity sound_entity, const std::filesystem::path &relative_path);

// Assign sample[i] to mesh_vertices[i]. Used by RealImpact initial load and mic swap.
// Unreferenced samples are released by the component handler.
void SetVertexSamples(
    entt::registry &, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::span<LoadedSample>
);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path.
void AssignVertexSample(
    entt::registry &, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::filesystem::path, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`.
// Removes audio components if the sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    entt::registry &, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

void Stop(entt::registry &, entt::entity sound_entity);
void SetModel(entt::registry &, entt::entity sound_entity, SoundVerticesModel);
