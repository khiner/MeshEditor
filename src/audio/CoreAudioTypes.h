#pragma once

#include <AudioToolbox/AudioToolbox.h>

inline AudioStreamBasicDescription MonoFloatAudioFormat(double sample_rate) {
    return {
        .mSampleRate = sample_rate,
        .mFormatID = kAudioFormatLinearPCM,
        .mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagsNativeEndian,
        .mBytesPerPacket = sizeof(float),
        .mFramesPerPacket = 1,
        .mBytesPerFrame = sizeof(float),
        .mChannelsPerFrame = 1,
        .mBitsPerChannel = 8 * sizeof(float),
        .mReserved = 0,
    };
}
