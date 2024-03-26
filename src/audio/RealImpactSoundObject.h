#pragma once

#include <vector>

#include "SoundObject.h"

struct RealImpact;
struct RealImpactListenerPoint;

struct RealImpactSoundObject : SoundObject {
    RealImpactSoundObject(const RealImpact &, const RealImpactListenerPoint &);

    std::vector<std::vector<float>> ImpactSamples;

    void Strike(uint vertex_index, float force = 1.0) override;
    void ProduceAudio(DeviceData *, float *output, uint frame_count) const override;
};
