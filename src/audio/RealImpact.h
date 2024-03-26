#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "numeric/vec2.h"
#include "numeric/vec3.h"

namespace fs = std::filesystem;

/*
Loads and provides access to a [RealImpact](https://github.com/samuel-clarke/RealImpact) dataset for a single object.
Holds all listener point data except the audio samples.
There are 2.3G of audio sample data across all listener points for each object.
  - 5 impact points recorded at 600 field points = 3000 impact samples, with 209,549 frames each (~4.37s at 48kHz).
  - 600 comes from: 15 microphones * 4 distances * 10 angles.
The `ListenerPoints` vector holds the first 600 listener points, representing all unique listener positions.
The values vary first by MicId, then by DistanceMm, and finally by AngleDeg.
This mirrors the chronological order of the recording process, where the microphone array records all mics simultaneously,
storing from bottom to top, then the array moves back away from the object along a line at the same angle, and finally the
array rotates to the next angle, moves back to the original distance, and repeats.
Load samples as needed with `LoadSamples`.
*/
struct RealImpact {
    struct ListenerPoint {
        const long MicId; // [0, NumMics - 1], bottom -> top
        const long DistanceMm; // Distance from the microphone to the object, in (whole) mm
        const long AngleDeg; // Angle of the listener relative to the object, in (whole) degrees
    };

    static constexpr uint NumMics = 15; // Number of microphones in the vertical microphone array
    static constexpr float MicBarLengthMm = 1890 - 70; // Height of the microphone array, in mm
    // Authors use a Dayton Audio EMM6 calibrated measurement microphone.
    // Measurements from https://www.amazon.com/Dayton-Audio-EMM-6-Measurement-Microphone/dp/B002KI8X40
    static constexpr float MicLengthMm = 190.5, MicWidthMm = 22.352;

    const fs::path Directory;
    const fs::path ObjPath; // Path to the .obj file
    std::vector<ListenerPoint> ListenerPoints;

    RealImpact(const fs::path &directory);

    // Load the audio samples for the given listener point.
    std::vector<float> LoadSamples(const ListenerPoint &) const;

    // Optionally add half the mic length to the distance so that placing a mic mesh with its origin
    // at the returned point results in the front of the mic head at the correct distance.
    // Pass `false` to get the mic head position (listener position) instead of the mic center.
    static vec3 GetPosition(const ListenerPoint &, vec3 world_up, bool mic_center = false);
};
