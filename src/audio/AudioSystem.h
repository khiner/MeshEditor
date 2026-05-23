#pragma once

#include "Action.h" // action::Emit
#include "AudioBuffer.h"
#include "AudioTypes.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

struct FaustDSP;

struct Recording {
    Recording(uint32_t frame_count) : Frames(frame_count) {}
    std::vector<float> Frames;
    uint32_t Frame{0};
    bool Complete() const { return Frame == Frames.size(); }
    void Record(float value) {
        if (!Complete()) Frames[Frame++] = value;
    }
};

// Called from audio device callback.
void ProcessAudio(FaustDSP &, entt::registry &, entt::entity viewport, AudioBuffer);

void RegisterAudioComponentHandlers(entt::registry &, entt::entity viewport);
void RemoveAudioComponents(entt::registry &, entt::entity sound_entity);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
// `selection_bits` is the raw SelectionBitset pointer (used in Edit mode for SampleOpVertices); may be null.
void DrawObjectAudioControls(
    entt::registry &, entt::entity viewport, entt::entity sound_entity, entt::entity mesh_entity,
    const uint32_t *selection_bits, action::Emit
);

// {path, frames} pair — path is an fs::path used as a dedup key in the scene-level sample store.
// For on-disk audio this is the absolute file path; for synthetic sources (e.g. RealImpact) it is a
// URI-style virtual key (see RealImpact::LoadSamples) that cannot be mistaken for a real file.
using LoadedSample = std::pair<std::filesystem::path, std::vector<float>>;

// Assign sample[i] to mesh_vertices[i]. Used by RealImpact initial load and mic swap.
// Any existing samples at those vertices are refcount-released.
void SetVertexSamples(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::vector<LoadedSample> &&
);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path; existing entries are reused and refcounted.
void AssignVertexSample(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::filesystem::path, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`. Removes audio components if the
// sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

// Decode any miniaudio-supported audio file to mono float frames at `SampleRate`. Returns empty on failure.
std::vector<float> LoadAudioFrames(const std::string &file_path);

void Stop(entt::registry &, entt::entity viewport, entt::entity sound_entity);
void SetModel(entt::registry &, entt::entity viewport, entt::entity sound_entity, SoundVerticesModel);
void SetVertex(entt::registry &, entt::entity viewport, entt::entity sound_entity, uint32_t vertex);
