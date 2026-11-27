#include "animation/Evaluate.h"

#include "TransformMath.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "scene/Instance.h"

namespace animation {
void Evaluate(state::Registry &r, float seconds, bool persistent) {
    std::vector<state::Entity> stale;
    for (const auto entity : r.view<const PosedLocal, const Transform>()) {
        if (!PosedTransformComponents(r, entity)) stale.emplace_back(entity);
    }
    for (const auto entity : stale) r.remove<PosedLocal>(entity);

    std::vector<float> value;
    for (const auto entity : AnimatedEntities(r)) {
        const auto *clip = ActiveClip(r, *ClipsOf(r, entity));
        if (!clip) continue;
        // The node's pose, seeded from its Transform so unanimated components follow edits to it.
        std::optional<Transform> pose;
        for (const auto &channel : clip->Channels) {
            const bool posing = channel.Target.Component == state::Key<PosedLocal>();
            if (!posing && !persistent) continue;
            if (posing && !pose) {
                const auto *transform = r.try_get<const Transform>(SourceOfCopy(r, entity));
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
        if (const auto *edited = r.try_get<const UnkeyedPose>(entity)) {
            if (edited->Components & TranslationBit) pose->P = edited->Value.P;
            if (edited->Components & RotationBit) pose->R = edited->Value.R;
            if (edited->Components & ScaleBit) pose->S = edited->Value.S;
        }
        if (const auto *posed = r.try_get<const PosedLocal>(entity)) {
            if (posed->Value != *pose) r.replace<PosedLocal>(entity, *pose);
        } else {
            r.emplace<PosedLocal>(entity, *pose);
        }
    }
}
} // namespace animation
