#include "animation/Keying.h"

#include "animation/AnimationTimeline.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "state/Scene.h"

#include <algorithm>
#include <cmath>

namespace animation {
namespace {
// Keys closer than this in seconds replace each other.
constexpr float KeyEpsilon{1e-4f};

// The index of the key at `seconds`, or the index to insert one at and false.
std::pair<size_t, bool> FindKey(const std::vector<float> &times, float seconds) {
    const auto it = std::ranges::lower_bound(times, seconds - KeyEpsilon);
    const size_t i = size_t(it - times.begin());
    return {i, i < times.size() && std::abs(times[i] - seconds) <= KeyEpsilon};
}

// The channel of `target` in `clip`, or null.
auto *FindChannel(auto &clip, const ChannelTarget &target) {
    const auto it = std::ranges::find(clip.Channels, target, &AnimationChannel::Target);
    return it != clip.Channels.end() ? &*it : nullptr;
}

// Writes `value` as the key at `seconds`, replacing a key within KeyEpsilon.
void SetKey(AnimationChannel &channel, float seconds, std::span<const float> value) {
    const uint32_t n = uint32_t(value.size());
    const auto [i, replace] = FindKey(channel.Times, seconds);
    const bool cubic = channel.Interp == AnimationInterpolation::CubicSpline;
    const uint32_t stride = cubic ? 3 * n : n;
    if (!replace) {
        channel.Times.insert(channel.Times.begin() + i, seconds);
        channel.Values.insert(channel.Values.begin() + i * stride, stride, 0.f);
    }
    // Cubic keys take zero tangents around the value.
    std::ranges::copy(value, channel.Values.begin() + i * stride + (cubic ? n : 0));
}

// Returns each emptied channel's field to rest, drops empty channels and clips, publishes the update, and removes the component when no clip remains.
void PublishClips(state::Scene &r, state::Entity entity) {
    for (const auto &clip : r.get<const AnimationClips>(entity).Clips)
        for (const auto &channel : clip.Channels)
            if (channel.Times.empty()) WriteField(r, entity, channel.Target, channel.Rest);
    r.patch<AnimationClips>(entity, [](auto &clips) {
        for (auto &clip : clips.Clips) std::erase_if(clip.Channels, [](const auto &channel) { return channel.Times.empty(); });
        std::erase_if(clips.Clips, [](const auto &clip) { return clip.Channels.empty(); });
    });
    if (r.get<const AnimationClips>(entity).Clips.empty()) r.remove<AnimationClips>(entity);
}

// Whether the field's value differs from its channel at `seconds`. A field the entity lacks never differs.
bool FieldChanged(const state::Scene &r, state::Entity e, const AnimationChannel &channel, float seconds) {
    std::vector<float> current(channel.Target.Count), animated(channel.Target.Count);
    if (!ReadField(r, e, channel.Target, current)) return false;
    EvaluateChannel(channel, seconds, animated);
    return !SameValue(channel.Target, current, animated);
}

// Calls `fn` with each active channel whose field value differs from the channel at `seconds`.
// Stops when `fn` returns true.
void ForEachChanged(const state::Scene &r, state::Entity viewport, float seconds, auto &&fn) {
    for (const auto [entity, clips] : r.view<const AnimationClips>().each()) {
        const auto *clip = ActiveClip(r, viewport, clips);
        if (!clip) continue;
        for (const auto &channel : clip->Channels)
            if (FieldChanged(r, entity, channel, seconds) && fn(KeyTarget{entity, channel.Target})) return;
    }
}
} // namespace

float FrameSeconds(const state::Scene &r, state::Entity viewport, int frame) {
    return float(std::max(0, frame - 1)) / r.get<const TimelineRange>(viewport).Fps;
}

ChannelState QueryChannel(const state::Scene &r, state::Entity viewport, const KeyTarget &key, float seconds) {
    const auto *clips = r.try_get<const AnimationClips>(key.Entity);
    const auto *clip = clips ? ActiveClip(r, viewport, *clips) : nullptr;
    const auto *channel = clip ? FindChannel(*clip, key.Target) : nullptr;
    if (!channel) return {};
    return {.HasChannel = true, .KeyAtFrame = FindKey(channel->Times, seconds).second, .Changed = FieldChanged(r, key.Entity, *channel, seconds)};
}

void InsertKey(state::Scene &r, state::Entity viewport, const KeyTarget &key, float seconds) {
    std::vector<float> value(key.Target.Count);
    if (value.empty() || !ReadField(r, key.Entity, key.Target, value)) return;
    auto &animations = r.edit<Animations>(viewport);
    if (animations.Names.empty()) {
        animations.Names.emplace_back("Animation");
        animations.Active = 0;
    }
    auto &clips = r.get_or_emplace<AnimationClips>(key.Entity);
    auto *clip = clips.Find(animations.Active);
    if (!clip) clip = &clips.Clips.emplace_back(AnimationClip{.Animation = animations.Active});
    auto *channel = FindChannel(*clip, key.Target);
    if (!channel) channel = &clip->Channels.emplace_back(AnimationChannel{.Target = key.Target, .Interp = AnimationInterpolation::Linear, .Rest = value});
    SetKey(*channel, seconds, value);
    PublishClips(r, key.Entity);
}

bool DeleteKey(state::Scene &r, state::Entity viewport, const KeyTarget &key, float seconds) {
    auto *clips = r.try_edit<AnimationClips>(key.Entity);
    auto *clip = clips ? clips->Find(r.get<const Animations>(viewport).Active) : nullptr;
    auto *channel = clip ? FindChannel(*clip, key.Target) : nullptr;
    if (!channel) return false;
    const auto [i, found] = FindKey(channel->Times, seconds);
    if (!found) return false;
    const auto stride = channel->Values.size() / channel->Times.size();
    channel->Times.erase(channel->Times.begin() + i);
    channel->Values.erase(channel->Values.begin() + i * stride, channel->Values.begin() + (i + 1) * stride);
    PublishClips(r, key.Entity);
    return true;
}

bool DeleteKeys(state::Scene &r, state::Entity viewport, state::Entity entity, float seconds) {
    const auto *clips = r.try_get<const AnimationClips>(entity);
    const auto *clip = clips ? ActiveClip(r, viewport, *clips) : nullptr;
    if (!clip) return false;
    std::vector<ChannelTarget> targets;
    targets.reserve(clip->Channels.size());
    for (const auto &channel : clip->Channels) targets.emplace_back(channel.Target);
    bool any = false;
    for (const auto &target : targets) any |= DeleteKey(r, viewport, {entity, target}, seconds);
    return any;
}

bool AnyChanged(const state::Scene &r, state::Entity viewport, float seconds) {
    bool any = false;
    ForEachChanged(r, viewport, seconds, [&](const KeyTarget &) { return any = true; });
    return any;
}

void RestoreRest(state::Scene &r, uint32_t from, uint32_t to) {
    for (const auto [entity, clips] : r.view<const AnimationClips>().each()) {
        const auto *leaving = clips.Find(from);
        if (!leaving) continue;
        const auto *entering = clips.Find(to);
        for (const auto &channel : leaving->Channels)
            if (!entering || !FindChannel(*entering, channel.Target)) WriteField(r, entity, channel.Target, channel.Rest);
    }
}

void RecordChanged(state::Scene &r, state::Entity viewport, float seconds) {
    std::vector<KeyTarget> changed;
    ForEachChanged(r, viewport, seconds, [&](const KeyTarget &key) {
        changed.emplace_back(key);
        return false;
    });
    for (const auto &key : changed) InsertKey(r, viewport, key, seconds);
}
} // namespace animation
