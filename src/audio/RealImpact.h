#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Loads and provides access to a [RealImpact](https://github.com/samuel-clarke/RealImpact) dataset for a single object.
struct RealImpact {
    // Holds all listener point data except the audio samples.
    // There are 2.3G of audio sample data across all listener points for each object.
    //   - 5 impact points recorded at 600 field points = 3000 impact samples, with 209,549 frames each (~4.37s at 48kHz).
    //   - 600 comes from: 15 microphones * 4 distances * 10 angles.
    // The `ListenerPoints` vector holds the first 600 listener points, representing all unique listener positions.
    // They vary first by MicId, then by DistanceMm, and finally by AngleDeg.
    // (This mirrors the chronological order of the recording process,where the vertical microphone array records from bottom to top,
    // then the array moves back away from the object along a line at the same angle, and finally the array rotates to the next angle,
    // moves back to the original distance, and repeats.)
    // Load samples as needed with `LoadSamples`.
    struct ListenerPoint {
        const long MicId; // Accesses the same recording across all files
        const long DistanceMm; // Distance from the microphone to the object, in (whole) mm
        const long AngleDeg; // Angle of the listener relative to the object, in (whole) degrees
    };

    const fs::path Directory;
    const fs::path ObjPath; // Path to the .obj file
    std::vector<ListenerPoint> ListenerPoints;

    RealImpact(const fs::path &directory);

    // Load the audio samples for the given listener point.
    std::vector<float> LoadSamples(const ListenerPoint &) const;
};
