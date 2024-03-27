#include "RealImpactSoundObject.h"

#include "RealImpact.h"

RealImpactSoundObject::RealImpactSoundObject(const RealImpact &parent, const RealImpactListenerPoint &listener_point)
    : SoundObject(listener_point.GetPosition()), ImpactSamples(listener_point.LoadImpactSamples(parent)) {}

void RealImpactSoundObject::ProduceAudio(DeviceData, float *output, uint frame_count) {
    if (CurrentVertexIndex >= ImpactSamples.size()) return;

    const auto &impact_samples = ImpactSamples[CurrentVertexIndex];
    for (uint i = 0; i < frame_count; ++i) {
        output[i] = CurrentFrame < impact_samples.size() ? impact_samples[CurrentFrame++] : 0.0f;
    }
}

void RealImpactSoundObject::Strike(uint vertex_index, float force) {
    CurrentVertexIndex = vertex_index;
    CurrentFrame = 0;
    (void)force; // Unused
}
