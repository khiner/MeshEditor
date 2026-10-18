#pragma once

#include <filesystem>
#include <map>
#include <unordered_map>
#include <vector>

namespace state {
struct Scene;
} // namespace state

// Decoded frames keyed by an audio file path or a synthetic source URI.
using LoadedSample = std::pair<std::filesystem::path, std::vector<float>>;

// Sample assignments per sound object, keyed by mesh vertex handle.
struct VertexSamples {
    std::map<uint32_t, std::filesystem::path> PathByVertex;
};

// Runtime decoded cache.
// Snapshots store source references, never audio frames.
struct AudioSamples {
    std::unordered_map<std::filesystem::path, std::vector<float>> ByPath;
};
inline void Reset(AudioSamples &samples, state::Scene &) { samples.ByPath.clear(); }

struct SamplePlayback {
    uint32_t Frame{0};
    bool Stopped{true};

    void Stop() { Stopped = true; }
    void Play() {
        Frame = 0;
        Stopped = false;
    }
};
