#include "animation/Keyframes.h"

#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "armature/ArmatureComponents.h"
#include "scene/Entity.h"
#include "state/Scene.h"

#include <algorithm>

namespace {
// Every key is kept. Loop repeats are added while they land inside the timeline range.
void AppendClipKeys(const auto &clip, const TimelineRange &range, std::vector<float> &frames) {
    const float period = clip.DurationSeconds * range.Fps;
    for (const auto &channel : clip.Channels) {
        for (const float t : channel.TimesSeconds) {
            float f = 1.f + t * range.Fps;
            do frames.push_back(f);
            while (period >= 1.f && (f += period) <= float(range.EndFrame));
        }
    }
}
} // namespace

std::vector<float> CollectKeyframes(const state::Scene &r, state::Entity viewport) {
    const auto &range = r.get<const TimelineRange>(viewport);
    std::vector<float> frames;
    const auto append_active = [&](const auto *animation) {
        if (animation && animation->ActiveClipIndex < animation->Clips.size()) AppendClipKeys(animation->Clips[animation->ActiveClipIndex], range, frames);
    };
    for (const auto e : r.view<const Selected>()) {
        append_active(r.try_get<const NodeTransformAnimation>(e));
        append_active(r.try_get<const MorphWeightAnimation>(e));
        if (const auto *armature = r.try_get<const ArmatureObject>(e)) append_active(r.try_get<const ArmatureAnimation>(armature->Entity));
    }
    std::ranges::sort(frames);
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    return frames;
}
