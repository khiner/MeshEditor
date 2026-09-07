#pragma once

#include "AudioSamples.h"
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
// `monitor` maps 20 Pa to full scale and soft-limits higher pressures for device output.
// Capture paths pass null and convert pressure separately.
void ProcessAudio(entt::registry &, entt::entity viewport, float *output, uint32_t frame_count, bool monitor = false);

// Convert captured pressure to device units in place, at the monitor level.
// `limiter` preserves the envelope across calls for one monitored stream.
void MonitorFrames(entt::registry &, std::span<float>, MonitorLimiter &);

// Capture of the master output, for muxing into a video recording.
// The device thread writes and the main thread drains, so one of each and no locking.
// Returns the device sample rate, or 0 when there is no device to capture.
uint32_t BeginAudioCapture(entt::registry &);
void EndAudioCapture(entt::registry &);
// Append everything the device has produced since the last call.
void DrainAudioCapture(entt::registry &, std::vector<float> &);

void RenderAudioOffline(entt::registry &, entt::entity viewport, std::vector<float> &, uint32_t frame_count);

// Rebuild the entity's ContactDynamics from its MassProperties, ModalModes, and mesh (surface curvature).
// Removes ContactDynamics when the inputs are missing.
void UpdateContactDynamics(entt::registry &, entt::entity sound_entity);

// Ratio of current acoustic density to solved modal density, or one when unknown.
// Stored mass properties scale linearly by this ratio.
double ModalDensityRatio(const entt::registry &, entt::entity sound_entity);

uint32_t DeviceSampleRate(const entt::registry &);

// Decode any CoreAudio-supported audio file to mono float frames at `sample_rate`. Returns empty on failure.
std::vector<float> LoadAudioFrames(const std::string &file_path, uint32_t sample_rate);
