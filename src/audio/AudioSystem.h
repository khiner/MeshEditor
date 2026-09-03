#pragma once

#include "AudioTypes.h"
#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <span>
#include <vector>

struct Recording {
    Recording(uint32_t frame_count) : Frames(frame_count) {}
    std::vector<float> Frames;
    uint32_t Frame{0};
    bool Complete() const { return Frame == Frames.size(); }
    void Record(float value) {
        if (!Complete()) Frames[Frame++] = value;
    }
};

// Render the master mix, which is far-field pressure in pascals at 1 m.
// `monitor` converts the result to device units as the final stage, mapping 20 Pa (120 dB SPL) to full scale and soft-limiting what still crests past it.
// Capture paths leave it unset and convert the pressure themselves.
void ProcessAudio(entt::registry &, entt::entity viewport, float *output, uint32_t frame_count, bool monitor = false);

// Convert captured pressure to device units in place, at the monitor level.
// `limiter` holds the envelope across calls, one per stream monitored.
void MonitorFrames(entt::registry &, std::span<float>, MonitorLimiter &);

// Capture of the master output, for muxing into a video recording.
// The device thread writes and the main thread drains, so one of each and no locking.
// Returns the device sample rate, or 0 when there is no device to capture.
uint32_t BeginAudioCapture(entt::registry &);
void EndAudioCapture(entt::registry &);
// Append everything the device has produced since the last call.
void DrainAudioCapture(entt::registry &, std::vector<float> &);

// Render exactly `frame_count` frames of the master output on the calling thread, for capture with no device, where the result is deterministic and aligned to whatever advanced the simulation.
void RenderAudioOffline(entt::registry &, entt::entity viewport, std::vector<float> &, uint32_t frame_count);

void RegisterAudioComponentHandlers(entt::registry &);

// Create the modal audio context and register its component handlers.
// Must run before a scene loads, so that loading one populates the bank.
// The output device is separate, and capture works without it.
void InitAudioSystem(entt::registry &);
// Destroy the modal audio context after any output device has stopped.
void DeinitAudioSystem(entt::registry &);
void RemoveAudioComponents(entt::registry &, entt::entity sound_entity);

// Draw the viewport-global audio synthesis controls.
void DrawGlobalSynthControls(entt::registry &, entt::entity viewport);

// Draw a live view of the audio model: what the bank holds, and where the fixed-size pools stand against demand.
void DrawAudioDebug(const entt::registry &);

// Rebuild the entity's ContactDynamics from its MassProperties, ModalModes, and mesh (surface curvature).
// Removes ContactDynamics when the inputs are missing.
void UpdateContactDynamics(entt::registry &, entt::entity sound_entity);

// Density ratio of the mesh's current acoustic material to the density the entity's modal model was
// solved at. Mass properties are stored at the solved density and scale linearly by this. 1 when unknown.
double ModalDensityRatio(const entt::registry &, entt::entity sound_entity);

// Apply a modal solve result file (relative to ModalModelsDir()) to the sound entity.
void ApplyModalModel(entt::registry &, entt::entity sound_entity, const std::filesystem::path &relative_path);

// Draw the Audio controls for a sound object entity (has SoundVerticesModel).
void DrawObjectAudioControls(entt::registry &, entt::entity viewport, entt::entity sound_entity, entt::entity mesh_entity);

// Draw the in-flight modal solve jobs as a progress overlay anchored to the current window's
// lower-left corner. Call inside the viewport window.
void DrawModalJobsOverlay(entt::registry &);

// {path, frames} pair, where path is the dedup key in the scene-level sample store.
// On-disk audio uses its absolute file path. A synthetic source (e.g. RealImpact) uses a URI-style
// virtual key that cannot be mistaken for a real file.
using LoadedSample = std::pair<std::filesystem::path, std::vector<float>>;

// Assign sample[i] to mesh_vertices[i]. Used by RealImpact initial load and mic swap.
// Any existing samples at those vertices are refcount-released.
void SetVertexSamples(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::vector<LoadedSample> &&
);

// Assign one sample (path + frames) to every mesh vertex in `mesh_vertices`.
// Creates SoundVertices / VertexSamples / SoundVerticesModel::Samples if missing.
// The sample store deduplicates by path, reusing and refcounting existing entries.
void AssignVertexSample(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices, std::filesystem::path, std::vector<float> &&frames
);

// Remove samples from every mesh vertex in `mesh_vertices`.
// Removes audio components if the sound object ends up empty and has no modal model.
void RemoveVertexSamples(
    entt::registry &, entt::entity viewport, entt::entity sound_entity,
    std::span<const uint32_t> mesh_vertices
);

uint32_t DeviceSampleRate(const entt::registry &);

// Decode any CoreAudio-supported audio file to mono float frames at `sample_rate`. Returns empty on failure.
std::vector<float> LoadAudioFrames(const std::string &file_path, uint32_t sample_rate);

void Stop(entt::registry &, entt::entity sound_entity);
void SetModel(entt::registry &, entt::entity sound_entity, SoundVerticesModel);
