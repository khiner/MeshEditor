#include "animation/Clips.h"
#include "TransformMath.h"

#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"

namespace animation {
namespace {
// The left endpoint of the keyframe interval containing `t`.
uint32_t FindKeyframe(std::span<const float> times, float t) {
    if (times.size() <= 1 || t <= times.front()) return 0;
    if (t >= times.back()) return times.size() - 2;
    return std::distance(times.begin(), std::ranges::upper_bound(times, t)) - 1;
}
} // namespace

const AnimationClips *ClipsOf(const state::Registry &r, state::Entity e) {
    return r.try_get<const AnimationClips>(HierarchySource(r, e));
}
std::vector<state::Entity> AnimatedEntities(const state::Registry &r) {
    auto entities = SortedEntities(r.view<const AnimationClips>());
    for (const auto e : r.view<const HierarchyMember>())
        if (ClipsOf(r, e)) entities.push_back(e);
    std::ranges::sort(entities);
    return entities;
}

void PlaceChannel(const state::Registry &r, state::Entity e, AnimationChannel &channel) {
    if (channel.Target.Component != state::Key<PosedLocal>()) return;
    ForEachHierarchyPlacement(r, e, [&](state::Entity placed, state::Entity source) {
        const auto delta = ToTransform(ToMatrix(r.get<const Transform>(placed)) * Inverse(ToMatrix(r.get<const Transform>(source))));
        const auto apply = [&](std::span<float> value, bool tangent) {
            if (channel.Target.Offset == offsetof(Transform, P) && value.size() == 3) {
                const auto moved = delta.R * (delta.S * vec3{value[0], value[1], value[2]}) + (tangent ? vec3{} : delta.P);
                std::copy_n(&moved.x, 3, value.begin());
            } else if (channel.Target.Offset == offsetof(Transform, S) && value.size() == 3) {
                const auto scaled = delta.S * vec3{value[0], value[1], value[2]};
                std::copy_n(&scaled.x, 3, value.begin());
            } else if (channel.Target.Offset == offsetof(Transform, R) && value.size() == 4) StoreQuat(value.data(), delta.R * LoadQuat(value.data()));
        };
        apply(channel.Rest, false);
        const auto count = channel.Target.Count;
        if (!count) return;
        for (size_t offset = 0; offset + count <= channel.Values.size(); offset += count) apply(std::span{channel.Values}.subspan(offset, count), channel.Interp == AnimationInterpolation::CubicSpline && (offset / count) % 3 != 1);
    });
}

const AnimationClip *ActiveClip(const state::Registry &r, const AnimationClips &clips) {
    return clips.Find(r.get<const Animations>(state::Document).Active);
}

void EvaluateChannel(const AnimationChannel &channel, float seconds, std::span<float> out) {
    const uint32_t n = channel.Target.Count;
    if (n == 0 || n != out.size() || channel.Times.empty()) return;
    const bool rotation = channel.Target.Kind == ValueKind::Quaternion && n == 4;
    const auto k = FindKeyframe(channel.Times, seconds);
    const auto k1 = std::min<uint32_t>(k + 1, channel.Times.size() - 1);
    const float t0 = channel.Times[k], t1 = channel.Times[k1];
    const float dt = t1 - t0;
    const float alpha = dt > 0 ? std::clamp((seconds - t0) / dt, 0.f, 1.f) : 0.f;
    if (channel.Interp == AnimationInterpolation::Step || channel.Target.Kind == ValueKind::Bool || dt <= 0) {
        const auto stride = channel.Interp == AnimationInterpolation::CubicSpline ? 3 * n : n;
        const auto offset = channel.Interp == AnimationInterpolation::CubicSpline ? n : 0;
        std::copy_n(channel.Values.data() + (alpha >= 1.f ? k1 : k) * stride + offset, n, out.begin());
        return;
    }
    if (channel.Interp == AnimationInterpolation::Linear) {
        const float *v0 = channel.Values.data() + k * n, *v1 = channel.Values.data() + k1 * n;
        if (rotation) return StoreQuat(out.data(), Slerp(LoadQuat(v0), LoadQuat(v1), alpha));
        for (uint32_t i = 0; i < n; ++i) out[i] = v0[i] + (v1[i] - v0[i]) * alpha;
        return;
    }
    // Cubic keys store the in tangent, value, and out tangent per component.
    const float a2 = alpha * alpha, a3 = a2 * alpha;
    const float h00 = 2 * a3 - 3 * a2 + 1, h10 = (a3 - 2 * a2 + alpha) * dt, h01 = -2 * a3 + 3 * a2, h11 = (a3 - a2) * dt;
    const float *kf0 = channel.Values.data() + k * 3 * n, *kf1 = channel.Values.data() + k1 * 3 * n;
    for (uint32_t i = 0; i < n; ++i) out[i] = h00 * kf0[n + i] + h10 * kf0[2 * n + i] + h01 * kf1[n + i] + h11 * kf1[i];
    if (rotation) StoreQuat(out.data(), Normalize(LoadQuat(out.data())));
}

float LastKeySeconds(const state::Registry &r, std::optional<uint32_t> animation) {
    float end = 0;
    for (const auto [_, clips] : r.view<const AnimationClips>().each()) {
        for (const auto &clip : clips.Clips) {
            if (animation && clip.Animation != *animation) continue;
            for (const auto &channel : clip.Channels) {
                if (!channel.Times.empty()) end = std::max(end, channel.Times.back());
            }
        }
    }
    return end;
}

uint8_t PosedTransformComponents(const state::Registry &r, state::Entity e) {
    const auto *clips = r.all_of<Animations>(state::Document) ? ClipsOf(r, e) : nullptr;
    const auto *clip = clips ? ActiveClip(r, *clips) : nullptr;
    if (!clip) return 0;
    uint8_t mask = 0;
    for (const auto &channel : clip->Channels) {
        if (channel.Target.Component != state::Key<PosedLocal>()) continue;
        if (channel.Target.Offset == offsetof(Transform, P)) mask |= TranslationBit;
        else if (channel.Target.Offset == offsetof(Transform, R)) mask |= RotationBit;
        else if (channel.Target.Offset == offsetof(Transform, S)) mask |= ScaleBit;
    }
    return mask;
}
} // namespace animation
