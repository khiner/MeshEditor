#pragma once

#include "AcousticMaterial.h"
#include "AudioBuffer.h"

#include "entt_fwd.h"

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
void ProcessAudio(FaustDSP &, entt::registry &, AudioBuffer);

void RegisterAudioComponentHandlers(entt::registry &, entt::entity scene_entity);
void RemoveAudioComponents(entt::registry &, entt::entity sound_entity);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
void DrawObjectAudioControls(entt::registry &, entt::entity scene_entity, entt::entity sound_entity, entt::entity mesh_entity);

// Replace all per-vertex sample buffers on a sound object (used by RealImpact mic swap).
void SetVertexSamples(entt::registry &, entt::entity, std::vector<std::vector<float>> &&);
// Add (or replace) a sample for a single mesh vertex on a sound object.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
void AddVertexSample(entt::registry &, entt::entity sound_entity, uint32_t mesh_vertex, std::vector<float> &&frames);
// Remove the sample for a single mesh vertex. Removes audio components if empty and no modal model.
void RemoveVertexSample(entt::registry &, entt::entity sound_entity, uint32_t mesh_vertex);

// Decode any miniaudio-supported audio file to mono float frames at `SampleRate`. Returns empty on failure.
std::vector<float> LoadAudioFrames(const std::string &file_path);
