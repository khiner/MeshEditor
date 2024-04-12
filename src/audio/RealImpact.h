#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "numeric/vec2.h"
#include "numeric/vec3.h"

namespace fs = std::filesystem;

/*
Loads and provides access to a [RealImpact](https://github.com/samuel-clarke/RealImpact) dataset for a single object.
Holds all listener point data except the audio samples.
There are 2.3G of audio sample data for each object.
  - 15 microphones * 4 distances * 10 angles = 600 listener points for each impact point
  - 5 impact points per listener points = 3000 impact samples
  - ~4.37s at 48kHz = 209,549 frames for each recording

The `ListenerPoints` vector holds the first 600 listener points, representing all unique listener positions.
The values vary first by MicId, then by DistanceMm, and finally by AngleDeg.
This mirrors the chronological order of the recording process, where the microphone array records all mics simultaneously,
storing from bottom to top, then the array moves back away from the object along a line at the same angle, and finally the
array rotates to the next angle, moves back to the original distance, and repeats.
Then, the hammer is positioned at the next impact point, and the process repeats.
*/

struct RealImpact;
struct RealImpactListenerPoint {
    const long Index; // [0, NumListenerPoints - 1]
    const long MicId; // [0, NumMics - 1], bottom -> top
    const long DistanceMm; // Distance from the microphone to the object, in (whole) mm
    const long AngleDeg; // Angle of the listener relative to the object, in (whole) degrees

    const uint ObjectEntityId; // ID of the object entity in the scene

    // Optionally add half the mic length to the distance so that placing a mic mesh with its origin
    // at the returned point results in the front of the mic head at the correct distance.
    // Pass `false` to get the mic head position (listener position) instead of the mic center.
    vec3 GetPosition(vec3 world_up = {0, 1, 0}, bool mic_center = false) const;

    // Load the audio sample frames (at 48kHz) for each impact vertex at this listener point.
    std::unordered_map<uint, std::vector<float>> LoadImpactSamples(const RealImpact &) const;
};

struct RealImpact {
    static constexpr uint NumListenerPoints = 600; // Number of unique listener positions
    static constexpr uint NumImpactVertices = 5; // Number of recorded impact points on each object
    static constexpr uint NumMics = 15; // Number of microphones in the vertical microphone array
    static constexpr float MicBarLengthMm = 1890 - 70; // Height of the microphone array, in mm
    // Authors use a Dayton Audio EMM6 calibrated measurement microphone.
    // Measurements from https://www.amazon.com/Dayton-Audio-EMM-6-Measurement-Microphone/dp/B002KI8X40
    static constexpr float MicLengthMm = 190.5, MicWidthMm = 22.352;
    // Even though the authors provide `.mtl` and `.png` material files, they don't provide the material name.
    // However, most object names include the material name, and the rest are easy to guess.
    static const std::unordered_map<std::string, std::string> MaterialNameForObjName;

    const fs::path Directory;
    const fs::path ObjPath; // Path to the .obj file
    const std::string ObjectName;
    uint VertexIndices[NumImpactVertices]; // Unique vertex indices of the impact points in the .obj file
    vec3 ImpactPositions[NumImpactVertices]; // World positions of the impact points
    std::optional<std::string> MaterialName;

    uint ObjectEntityId; // ID of the object entity in the scene

    RealImpact(const fs::path &directory);

    std::vector<RealImpactListenerPoint> LoadListenerPoints() const;
};
