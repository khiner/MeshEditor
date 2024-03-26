#include "RealImpactSoundObject.h"

#include "RealImpact.h"

RealImpactSoundObject::RealImpactSoundObject(const RealImpact &parent, const RealImpactListenerPoint &listener_point)
    : SoundObject(listener_point.GetPosition()), ImpactSamples(listener_point.LoadImpactSamples(parent)) {}

void RealImpactSoundObject::ProduceAudio(DeviceData *, float *output, uint frame_count) const {
}

void RealImpactSoundObject::Strike(uint vertex_index, float force) {
    std::vector<float> impact_samples = ImpactSamples.at(vertex_index);
}
