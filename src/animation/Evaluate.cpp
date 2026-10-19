#include "animation/Evaluate.h"

#include "animation/Clips.h"
#include "animation/Fields.h"
#include "state/Registry.h"

namespace animation {
void Evaluate(state::Registry &r, float seconds, bool persistent) {
    std::vector<state::Entity> stale;
    for (const auto entity : r.view<const PosedLocal, const Transform>()) {
        if (!PosedTransformComponents(r, entity)) stale.emplace_back(entity);
    }
    for (const auto entity : stale) r.remove<PosedLocal>(entity);

    std::vector<float> value;
    for (const auto [entity, clips] : r.view<const AnimationClips>().each()) {
        const auto *clip = ActiveClip(r, clips);
        if (!clip) continue;
        // The node's pose, seeded from its Transform so unanimated components follow edits to it.
        std::optional<Transform> pose;
        for (const auto &channel : clip->Channels) {
            const bool posing = channel.Target.Component == state::Key<PosedLocal>();
            if (!posing && !persistent) continue;
            if (posing && !pose) {
                const auto *transform = r.try_get<const Transform>(entity);
                if (!transform) break;
                pose = *transform;
            }
            value.resize(channel.Target.Count);
            EvaluateChannel(channel, seconds, value);
            if (posing) std::memcpy(reinterpret_cast<std::byte *>(&*pose) + channel.Target.Offset, value.data(), value.size() * sizeof(float));
            else WriteField(r, entity, channel.Target, value);
        }
        // Publish pose writes only on changes.
        if (!pose) continue;
        if (const auto *posed = r.try_get<const PosedLocal>(entity)) {
            if (posed->Value != *pose) r.replace<PosedLocal>(entity, *pose);
        } else {
            r.emplace<PosedLocal>(entity, *pose);
        }
    }
}
} // namespace animation
