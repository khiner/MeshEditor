#pragma once

using uint = unsigned int;

struct DeviceData {
    const uint SampleRate;
    const uint ChannelCount;
};

struct AudioSource {
    AudioSource() = default;
    virtual ~AudioSource() = default;

    // Fill the output buffer with floats between -1.0 and 1.0.
    virtual void ProduceAudio(DeviceData *, float *output, uint frame_count) = 0;
};
