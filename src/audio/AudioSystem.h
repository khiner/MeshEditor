#pragma once

#include "AcousticMaterial.h"
#include "AudioBuffer.h"

#include "entt_fwd.h"

#include <filesystem>
#include <span>
#include <utility>
#include <vector>

struct FaustDSP;

enum class SoundVerticesModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    Samples,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transform the object's geometry into a tetrahedral volume mesh
    // - use FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimate the dominant modes (frequencies/amplitudes/T60s) using an eigensolver
    // - simulate the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

// If an entity has this component, user has opened the modal model create/edit pane.
struct ModalModelCreateInfo {
    AcousticMaterial Material{materials::acoustic::All.front()};
    uint32_t NumVertices{10};
    bool CopySoundVertices{true}; // Only used if excitable component is already present.
    bool QualityTets{false};
};

// Called from audio device callback.
void ProcessAudio(FaustDSP &, entt::registry &, entt::entity scene_entity, AudioBuffer);

void RegisterAudioComponentHandlers(entt::registry &, entt::entity scene_entity);
void RemoveAudioComponents(entt::registry &, entt::entity sound_entity);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
// `selection_bits` is the raw SelectionBitset pointer (used in Edit mode for SampleOpVertices); may be null.
void DrawObjectAudioControls(
    entt::registry &, entt::entity scene_entity, entt::entity sound_entity, entt::entity mesh_entity,
    const uint32_t *selection_bits
);

// {path, frames} pair — path is an fs::path used as a dedup key in the scene-level sample store.
// For on-disk audio this is the absolute file path; for synthetic sources (e.g. RealImpact) it is a
// URI-style virtual key (see RealImpact::LoadSamples) that cannot be mistaken for a real file.
using LoadedSample = std::pair<std::filesystem::path, std::vector<float>>;

// Replace all per-vertex samples on a sound object (used by RealImpact mic swap).
// One pair per vertex in SoundVertices (parallel indexing). Any existing samples are refcount-released.
void SetVertexSamples(entt::registry &, entt::entity scene_entity, entt::entity sound_entity, std::vector<LoadedSample> &&);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path; existing entries are reused and refcounted.
void AssignVertexSample(
    entt::registry &, entt::entity scene_entity, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::filesystem::path, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`. Removes audio components if the
// sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    entt::registry &, entt::entity scene_entity, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

// Decode any miniaudio-supported audio file to mono float frames at `SampleRate`. Returns empty on failure.
std::vector<float> LoadAudioFrames(const std::string &file_path);
