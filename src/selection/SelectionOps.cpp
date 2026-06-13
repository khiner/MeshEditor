#include "selection/SelectionOps.h"

#include "armature/ArmatureComponents.h" // BoneActive, BoneIndex, BoneSubPartOf
#include "scene/Entity.h" // Selected, Active, SubElementOf
#include "selection/BoneSelection.h" // BoneSelection
#include "selection/SelectionQueries.h" // SelectionHit, ResolveHits

#include <entt/entity/registry.hpp>

using std::ranges::contains, std::ranges::find;

void Select(entt::registry &r, entt::entity e) {
    r.clear<Selected>();
    if (e != entt::null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void SelectBone(entt::registry &r, entt::entity e) {
    r.clear<BoneSelection>();
    if (e != entt::null) {
        r.clear<BoneActive>();
        r.emplace<BoneActive>(e);
        r.emplace<BoneSelection>(e);
    }
}

std::vector<SelectionHit> ResolveHits(entt::registry &r, const std::vector<entt::entity> &raw, bool bone_mode, bool merge_parts) {
    std::vector<SelectionHit> hits;
    for (const auto e : raw) {
        if (bone_mode && r.all_of<BoneIndex>(e)) {
            if (auto it = find(hits, e, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(e, BoneSel::Body);
            else if (merge_parts) it->Part = {};
        } else if (bone_mode && r.all_of<BoneSubPartOf>(e)) {
            const auto &sub = r.get<BoneSubPartOf>(e);
            if (auto it = find(hits, sub.BoneEntity, &SelectionHit::Entity); it == hits.end()) hits.emplace_back(sub.BoneEntity, sub.IsTip ? BoneSel::Tip : BoneSel::Root);
            else if (merge_parts) it->Part = {};
        } else if (!bone_mode) {
            if (const auto target = r.all_of<SubElementOf>(e) ? r.get<SubElementOf>(e).Parent : e; !contains(hits, target, &SelectionHit::Entity)) hits.emplace_back(target);
        }
    }
    return hits;
}
