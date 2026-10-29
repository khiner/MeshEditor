#include "action/Scene.h"
#include "SortUnique.h"
#include "Variant.h"
#include "action/Errors.h"
#include "action/ScopeResolve.h"
#include "object/ObjectOps.h"
#include "procedure/Network.h"
#include "scene/SceneGraph.h"
#include "selection/Selection.h"
#include "state/Registry.h"
#include "viewport/ViewportScene.h"

#include <format>

namespace action::scene {
namespace {
// The distinct membership roots addressed by the action.
std::vector<state::Entity> TargetRoots(const state::Registry &r, const Target &target) {
    std::vector<state::Entity> roots;
    ForEachTarget(target, [&] { return FindActiveEntity(r); }, [&](auto &&fn) { for (const auto e : SelectedIn(r)) fn(e); }, [&](state::Entity e) {
        const auto root = MembershipRoot(r, e);
        if (root != state::Null) roots.push_back(root); });
    SortUnique(roots);
    return roots;
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    std::visit(
        overloaded{
            [&](const Add &a) {
                if (r.view<const Scene>().size() >= MaxScenes) return Fail(r, std::format("Cannot add more than {} scenes", MaxScenes));
                const auto e = r.create();
                r.emplace<Scene>(e);
                EmplaceUniqueName(r, e, a.Name);
            },
            [&](const Remove &a) {
                if (!r.all_of<Scene>(a.Scene)) return;
                if (r.view<const Scene>().size() == 1) return Fail(r, "Cannot remove the only scene");
                // A root in another scene leaves this one, and a root only in this one goes with its descendants.
                std::vector<state::Entity> destroyed;
                const auto add_subtree = [&](this const auto &self, state::Entity e) -> void {
                    if (!r.all_of<SubElementOf>(e)) destroyed.push_back(e);
                    for (const auto child : Children{&r, e}) self(child);
                };
                for (const auto root : SortedEntities(r.view<const SceneMembership>())) {
                    const auto &scenes = r.get<const SceneMembership>(root).Scenes;
                    if (!std::ranges::contains(scenes, a.Scene)) continue;
                    if (scenes.size() == 1) add_subtree(root);
                    else r.patch<SceneMembership>(root, [&](auto &m) { std::erase(m.Scenes, a.Scene); });
                }
                Destroy(r, destroyed);
                // A viewport showing the scene shows the lowest-index scene that remains.
                const auto remaining = *std::ranges::find_if(r.view<const Scene>(), [&](state::Entity e) { return e != a.Scene; });
                for (const auto [viewport, shown] : r.view<const ViewportScene>().each()) {
                    if (shown.Scene == a.Scene) r.replace<ViewportScene>(viewport, remaining);
                }
                procedure::RemoveNetworkSubnets(r, a.Scene);
                r.destroy(a.Scene);
            },
            [&](const Rename &a) {
                if (!r.all_of<Scene>(a.Scene)) return;
                r.remove<Name>(a.Scene);
                EmplaceUniqueName(r, a.Scene, a.Name);
            },
            [&](const LinkSelected &a) {
                if (!r.all_of<Scene>(a.Scene)) return;
                for (const auto root : TargetRoots(r, a.Target)) {
                    if (!std::ranges::contains(r.get<const SceneMembership>(root).Scenes, a.Scene)) r.patch<SceneMembership>(root, [&](auto &m) { m.Scenes.push_back(a.Scene); });
                }
            },
            [&](const UnlinkSelected &a) {
                for (const auto root : TargetRoots(r, a.Target)) {
                    const auto &scenes = r.get<const SceneMembership>(root).Scenes;
                    if (scenes.size() > 1 && std::ranges::contains(scenes, a.Scene)) r.patch<SceneMembership>(root, [&](auto &m) { std::erase(m.Scenes, a.Scene); });
                }
            },
        },
        action
    );
}
} // namespace action::scene
