#pragma once

#include "RealImpact.h"
#include "SoundObject.h"

using uint = unsigned int;

struct RealImpactSoundObject : SoundObject {
    RealImpactSoundObject(vec3 listener_position, RealImpact real_impact)
        : SoundObject(listener_position), RealImpact(real_impact) {}

    RealImpact RealImpact;

    void Strike(uint vertex_index, float force = 1.0) override {} // SoundObject
    void ProduceAudio(DeviceData *, float *output, uint frame_count); // AudioSource
};
