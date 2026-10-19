#include "action/Animation.h"

#include "action/ScopeResolve.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "selection/Selection.h"
#include "timeline/Timeline.h"
#include "viewport/InteractionComponents.h"

namespace action::animation {
namespace {
using ::animation::KeyTarget;

std::vector<state::Entity> TargetEntities(const state::Registry &r, const Target &target) {
    std::vector<state::Entity> entities;
    const bool pose = r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose;
    ForEachTarget(
        target,
        [&] { return pose ? FindActiveBone(r) : FindActiveEntity(r); },
        [&](auto &&fn) {
            if (pose) {
                for (const auto e : r.view<const BoneSelection>()) fn(e);
            } else {
                for (const auto e : r.view<const Selected>()) {
                    if (r.all_of<Transform>(e)) fn(e);
                }
            }
        },
        [&](state::Entity e) { entities.emplace_back(e); }
    );
    return entities;
}

std::vector<KeyTarget> Targets(const state::Registry &r, const KeyScope &keys) {
    std::vector<KeyTarget> targets;
    for (const auto e : TargetEntities(r, keys.Target)) {
        if (keys.Channel) targets.emplace_back(e, *keys.Channel);
        else {
            for (const auto &target : ::animation::TransformTargets(r, e)) targets.emplace_back(e, target);
        }
    }
    return targets;
}

float CurrentSeconds(const state::Registry &r) {
    return ::animation::FrameSeconds(r, r.get<const TimelinePlayback>(state::Document).CurrentFrame);
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    std::visit(
        overloaded{
            [&](const InsertKey &a) {
                const float seconds = CurrentSeconds(r);
                for (const auto &target : Targets(r, a.Keys)) ::animation::InsertKey(r, target, seconds);
            },
            [&](const DeleteKey &a) {
                const float seconds = CurrentSeconds(r);
                if (a.Keys.Channel) {
                    for (const auto &target : Targets(r, a.Keys)) ::animation::DeleteKey(r, target, seconds);
                    return;
                }
                for (const auto e : TargetEntities(r, a.Keys.Target)) ::animation::DeleteKeys(r, e, seconds);
            },
            [&](RecordChanged) { ::animation::RecordChanged(r, CurrentSeconds(r)); },
            [&](const AddAnimation &a) {
                r.patch<Animations>(state::Document, [&](auto &animations) {
                    animations.Names.emplace_back(a.Name);
                    animations.Active = uint32_t(animations.Names.size() - 1);
                });
            },
            [&](const RenameAnimation &a) {
                r.patch<Animations>(state::Document, [&](auto &animations) {
                    if (a.Index < animations.Names.size()) animations.Names[a.Index] = a.Name;
                });
            },
            [&](const SelectAnimation &a) {
                const auto &animations = r.get<const Animations>(state::Document);
                if (a.Index >= animations.Names.size()) return;
                if (a.Index != animations.Active) ::animation::RestoreRest(r, animations.Active, a.Index);
                r.patch<Animations>(state::Document, [&](auto &animations) { animations.Active = a.Index; });
                const float end = ::animation::LastKeySeconds(r, a.Index);
                r.patch<TimelineRange>(state::Document, [&](auto &range) { range.EndFrame = std::max(range.StartFrame, int(std::ceil(end * range.Fps))); });
                JumpToStartFrame(r);
            },
        },
        action
    );
}
} // namespace action::animation
