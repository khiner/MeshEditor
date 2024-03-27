#pragma once

#include <vector>

#include "SoundObject.h"

struct RealImpact;
struct RealImpactListenerPoint;

struct RealImpactSoundObject : SoundObject {
    RealImpactSoundObject(const RealImpact &, const RealImpactListenerPoint &);

    std::vector<std::vector<float>> ImpactSamples;
    uint CurrentVertexIndex{0}, CurrentFrame{0};

    void Strike(uint vertex_index, float force = 1.0) override;
    void ProduceAudio(DeviceData, float *output, uint frame_count) override;
};
