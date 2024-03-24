#pragma once

#include "AudioSource.h"

#include "numeric/vec3.h"

using uint = unsigned int;

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    SoundObject(vec3 listener_position) : ListenerPosition(listener_position) {}
    virtual ~SoundObject() = default;

    vec3 ListenerPosition;

    virtual void ProduceAudio(DeviceData *, float *output, uint frame_count) override = 0; // AudioSource

    virtual void Strike(uint vertex_index, float force = 1.0) = 0;
};
