#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "numeric/vec2.h"
#include "numeric/vec3.h"

/*
Loads and provides access to a [RealImpact](https://github.com/samuel-clarke/RealImpact) dataset for a single object.
Holds all listener point data except the audio samples.
There are 2.3G of audio sample data for each object.
  - 15 microphones * 4 distances * 10 angles = 600 listener points for each impact point
  - 5 impact points per listener points = 3000 impact samples
  - ~4.37s at 48kHz = 208,592 frames for each recording

The `ListenerPoints` vector holds the first 600 listener points, representing all unique listener positions.
The values vary first by MicId, then by DistanceMm, and finally by AngleDeg.
This mirrors the chronological order of the recording process, where the microphone array records all mics simultaneously,
storing from bottom to top, then the array moves back away from the object along a line at the same angle, and finally the
array rotates to the next angle, moves back to the original distance, and repeats.
Then, the hammer is positioned at the next impact point, and the process repeats.
*/

namespace RealImpact {
namespace fs = std::filesystem;

static constexpr uint NumListenerPoints = 600; // Number of unique listener positions
static constexpr uint NumImpactVertices = 5; // Number of recorded impact points on each object
static constexpr uint NumMics = 15; // Number of microphones in the vertical microphone array
static constexpr float MicBarLengthMm = 1890 - 70; // Height of the microphone array, in mm
// Authors use a Dayton Audio EMM6 calibrated measurement microphone.
// Measurements from https://www.amazon.com/Dayton-Audio-EMM-6-Measurement-Microphone/dp/B002KI8X40
static constexpr float MicLengthMm = 190.5, MicWidthMm = 22.352;

struct ListenerPoint {
    const long Index; // [0, NumListenerPoints - 1]
    const long MicId; // [0, NumMics - 1], bottom -> top
    const long DistanceMm; // Distance from the microphone to the object, in (whole) mm
    const long AngleDeg; // Angle of the listener relative to the object, in (whole) degrees

    // Optionally add half the mic length to the distance so that placing a mic mesh with its origin
    // at the returned point results in the front of the mic head at the correct distance.
    // Pass `false` to get the mic head position (listener position) instead of the mic center.
    vec3 GetPosition(vec3 world_up = {0, 1, 0}, bool mic_center = false) const;
};

// Load the audio sample frames (at 48kHz) for each impact vertex at this listener point.
std::array<std::vector<float>, NumImpactVertices> LoadSamples(const fs::path &directory, long listener_point_index);
std::optional<std::string> FindObjectName(const fs::path &start_path);
std::optional<std::string_view> FindMaterialName(std::string_view);
std::vector<ListenerPoint> LoadListenerPoints(const fs::path &directory);
// World positions of the impact points
std::array<vec3, NumImpactVertices> LoadPositions(const fs::path &directory);

} // namespace RealImpact
