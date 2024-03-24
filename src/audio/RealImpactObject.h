#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "numeric/vec3.h"

namespace fs = std::filesystem;

struct RealImpactObject {
    inline static const std::string SampleDataFileName = "deconvolved_0db.npy";
    inline static const std::string ListenerXYZsFileName = "listenerXYZ.npy";

    // Holds all listener point data except the audio samples.
    // There are 2.3G of audio sample data across all listener points for each object.
    // (5 impact points recorded at 600 field points (= 3000 impact samples) with 209,549 frames each (~4.37s at 48kHz).)
    // Load samples as needed with `LoadSamples`.
    struct ListenerPoint {
        const uint Index; // Accesses the same recording across all files
        const long AngleDeg; // Angle of the listener relative to the object, in (whole) degrees
        const long DistanceMm; // Distance from the microphone to the object, in (whole) mm
        const vec3 Position; // Position of the listener
    };

    const fs::path Directory;
    const fs::path ObjPath; // Path to the .obj file
    std::vector<ListenerPoint> ListenerPoints;

    RealImpactObject(const fs::path &directory);

    // Load the audio samples for the given listener index.
    std::vector<float> LoadSamples(const ListenerPoint &) const;
};
