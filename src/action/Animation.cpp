#include "action/Animation.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"
#include "selection/Selection.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"

#include <cmath>
#include <vector>

namespace action::animation {
namespace {
using ::animation::KeyTarget;

// Entities a scope names: the given entity, or the bones in Pose mode and the objects otherwise.
std::vector<state::Entity> ScopeEntities(const state::Scene &r, state::Entity viewport, Scope scope, state::Entity entity) {
    std::vector<state::Entity> entities;
    const bool pose = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose;
    switch (scope) {
        case Scope::Entity: entities.emplace_back(entity); break;
        case Scope::Active:
            if (const auto e = pose ? FindActiveBone(r) : FindActiveEntity(r); e != state::Null) entities.emplace_back(e);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta:
            if (pose) {
                for (const auto e : r.view<const BoneSelection>()) entities.emplace_back(e);
            } else {
                for (const auto e : r.view<const Selected>())
                    if (r.all_of<Transform>(e)) entities.emplace_back(e);
            }
            break;
    }
    return entities;
}

std::vector<KeyTarget> Targets(const state::Scene &r, state::Entity viewport, const KeyScope &keys) {
    std::vector<KeyTarget> targets;
    for (const auto e : ScopeEntities(r, viewport, keys.Scope, keys.Entity)) {
        if (keys.Target) targets.emplace_back(e, *keys.Target);
        else
            for (const auto &target : ::animation::TransformTargets(r, e)) targets.emplace_back(e, target);
    }
    return targets;
}

float CurrentSeconds(const state::Scene &r, state::Entity viewport) {
    return ::animation::FrameSeconds(r, viewport, r.get<const TimelinePlayback>(viewport).CurrentFrame);
}
} // namespace

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    std::visit(
        overloaded{
            [&](const InsertKey &a) {
                const float seconds = CurrentSeconds(r, viewport);
                for (const auto &target : Targets(r, viewport, a.Keys)) ::animation::InsertKey(r, viewport, target, seconds);
            },
            [&](const DeleteKey &a) {
                const float seconds = CurrentSeconds(r, viewport);
                if (a.Keys.Target) {
                    for (const auto &target : Targets(r, viewport, a.Keys)) ::animation::DeleteKey(r, viewport, target, seconds);
                    return;
                }
                for (const auto e : ScopeEntities(r, viewport, a.Keys.Scope, a.Keys.Entity)) ::animation::DeleteKeys(r, viewport, e, seconds);
            },
            [&](RecordChanged) { ::animation::RecordChanged(r, viewport, CurrentSeconds(r, viewport)); },
            [&](const AddAnimation &a) {
                r.patch<Animations>(viewport, [&](auto &animations) {
                    animations.Names.emplace_back(a.Name);
                    animations.Active = uint32_t(animations.Names.size() - 1);
                });
            },
            [&](const RenameAnimation &a) {
                r.patch<Animations>(viewport, [&](auto &animations) {
                    if (a.Index < animations.Names.size()) animations.Names[a.Index] = a.Name;
                });
            },
            [&](const SelectAnimation &a) {
                const auto &animations = r.get<const Animations>(viewport);
                if (a.Index >= animations.Names.size()) return;
                if (a.Index != animations.Active) ::animation::RestoreRest(r, animations.Active, a.Index);
                r.patch<Animations>(viewport, [&](auto &animations) { animations.Active = a.Index; });
                const float end = ::animation::LastKeySeconds(r, a.Index);
                r.patch<TimelineRange>(viewport, [&](auto &range) { range.EndFrame = std::max(range.StartFrame, int(std::ceil(end * range.Fps))); });
                JumpToStartFrame(r, viewport);
            },
        },
        action
    );
}
} // namespace action::animation
