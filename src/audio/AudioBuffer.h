#pragma once

#include <cstdint>

struct AudioBuffer {
    const uint32_t SampleRate;
    const uint32_t ChannelCount, FrameCount;
    const float *Input;
    float *Output;
};
