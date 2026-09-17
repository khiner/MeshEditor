#include "animation/Evaluate.h"

#include "animation/Clips.h"
#include "animation/Fields.h"
#include "state/Scene.h"

#include <vector>

namespace animation {
void Evaluate(state::Scene &r, state::Entity viewport, float seconds, bool persistent) {
    // Drop the pose of a node the active animation no longer poses.
    std::vector<state::Entity> stale;
    for (const auto entity : r.view<const PosedLocal, const Transform>())
        if (!PosedTransformComponents(r, viewport, entity)) stale.emplace_back(entity);
    for (const auto entity : stale) r.remove<PosedLocal>(entity);

    std::vector<float> value;
    for (const auto [entity, clips] : r.view<const AnimationClips>().each()) {
        const auto *clip = ActiveClip(r, viewport, clips);
        if (!clip) continue;
        bool node_posed = false;
        for (const auto &channel : clip->Channels) {
            const bool pose = channel.Target.Component == state::Key<PosedLocal>();
            if (!pose && !persistent) continue;
            // Seed the pose from the Transform so unanimated components follow edits to it.
            if (pose && !std::exchange(node_posed, true)) {
                if (!r.all_of<Transform>(entity)) break;
                r.emplace_or_replace<PosedLocal>(entity, r.get<const Transform>(entity));
            }
            value.resize(channel.Target.Count);
            EvaluateChannel(channel, seconds, value);
            WriteField(r, entity, channel.Target, value);
        }
    }
}
} // namespace animation
