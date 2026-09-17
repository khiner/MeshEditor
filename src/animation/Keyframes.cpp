#include "animation/Keyframes.h"

#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/Clips.h"
#include "armature/ArmatureComponents.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "scene/Entity.h"
#include "state/Scene.h"
#include "state/Schema.h"

#include <algorithm>

namespace {
void AppendClipKeys(const AnimationClip &clip, float fps, std::vector<float> &frames, auto &&include) {
    for (const auto &channel : clip.Channels) {
        if (!include(channel)) continue;
        for (const float t : channel.Times) frames.emplace_back(1.f + t * fps);
    }
}
} // namespace

std::vector<float> CollectKeyframes(const state::Scene &r, state::Entity viewport) {
    const float fps = r.get<const TimelineRange>(viewport).Fps;
    std::vector<float> frames;
    const auto append = [&](state::Entity e, auto &&include) {
        if (const auto *clips = r.try_get<const AnimationClips>(e)) {
            if (const auto *clip = animation::ActiveClip(r, viewport, *clips)) AppendClipKeys(*clip, fps, frames, include);
        }
    };
    const auto all = [](const AnimationChannel &) { return true; };
    for (const auto e : r.view<const Selected>()) {
        append(e, all);
        if (const auto *armature = r.try_get<const ArmatureObject>(e)) append(armature->Entity, all);
    }
    // Material keys show for the material slot the active mesh displays.
    if (const auto active = FindActiveEntity(r); active != state::Null) {
        const auto *instance = r.try_get<const Instance>(active);
        if (const auto material = instance ? DisplayedMaterial(r, instance->Entity) : std::nullopt) append(viewport, [&](const AnimationChannel &channel) { return channel.Target.Component == state::Key<MaterialStore>() && channel.Target.Index == *material; });
    }
    std::ranges::sort(frames);
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    return frames;
}
