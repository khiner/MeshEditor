#include "scene/SceneGraph.h"
#include "TransformMath.h"
#include "animation/Clips.h"
#include "mesh/MeshComponents.h"
#include "render/GpuBuffers.h"
#include "scene/Entity.h"
#include "scene/SceneGraphOps.h"

namespace {
// Points the parent's child list at `first_child`.
void SetFirstChild(state::Scene &r, state::Entity parent, state::Entity first_child) {
    if (r.all_of<SceneChildren>(parent)) r.patch<SceneChildren>(parent, [first_child](auto &c) { c.FirstChild = first_child; });
    else r.emplace<SceneChildren>(parent, SceneChildren{.FirstChild = first_child});
}

// Links the parentless child ahead of `next_sibling` under the parent, leaving the parent's head to the caller.
void LinkAhead(state::Scene &r, state::Entity child, state::Entity parent, state::Entity next_sibling) {
    if (r.all_of<SceneChildren>(child)) r.patch<SceneChildren>(child, [next_sibling](auto &c) { c.NextSibling = next_sibling; });
    else r.emplace<SceneChildren>(child, SceneChildren{.NextSibling = next_sibling});
    r.emplace<SceneParent>(child, parent);
}

state::Entity FirstChildOf(const state::Scene &r, state::Entity parent) {
    const auto *children = r.try_get<const SceneChildren>(parent);
    return children ? children->FirstChild : state::Null;
}

// The placed instance slot holding the node's world transform, or UINT32_MAX when its WorldTransform component holds it.
// An armature part's slot holds its display transform.
uint32_t WorldSlot(const state::Scene &r, state::Entity e) {
    const auto *render = r.try_get<const RenderInstance>(e);
    return render && !r.all_of<SubElementOf>(e) ? render->BufferIndex : UINT32_MAX;
}

// The node's world transform at its home, `slot` from WorldSlot.
const Transform *WorldTransformAt(const state::Scene &r, state::Entity e, uint32_t slot) {
    if (slot != UINT32_MAX) return r.Context.get<const GpuBuffers>().Instances.TransformBuffer.GetSpan<Transform>({slot, 1u}).data();
    return r.try_get<const WorldTransform>(e);
}

void SetWorldTransformAt(state::Scene &r, state::Entity e, uint32_t slot, const Transform &world) {
    if (slot != UINT32_MAX) r.Context.get<GpuBuffers>().Instances.TransformBuffer.GetMutableSpan<Transform>({slot, 1u})[0] = world;
    else r.emplace_or_replace<WorldTransform>(e, world);
    reactive(r, state::Change::WorldTransform).emplace(e);
}
} // namespace

const Transform *WorldTransformOf(const state::Scene &r, state::Entity e) { return WorldTransformAt(r, e, WorldSlot(r, e)); }
void SetWorldTransform(state::Scene &r, state::Entity e, const Transform &world) { SetWorldTransformAt(r, e, WorldSlot(r, e), world); }

mat4 GetParentDelta(const state::Scene &r, state::Entity e) {
    const auto *parent = r.try_get<const SceneParent>(e);
    return parent ? ToMatrix(*WorldTransformOf(r, parent->Parent)) : I4;
}

ChildrenIterator &ChildrenIterator::operator++() {
    if (Current != state::Null) {
        const auto *children = R->try_get<const SceneChildren>(Current);
        Current = children ? children->NextSibling : state::Null;
    }
    return *this;
}

ChildrenIterator Children::begin() const {
    if (ParentEntity == state::Null) return {R, state::Null};
    return {R, FirstChildOf(*R, ParentEntity)};
}

state::Entity ParentOrNull(const state::Scene &r, state::Entity e) {
    const auto *parent = r.try_get<const SceneParent>(e);
    return parent ? parent->Parent : state::Null;
}

void ClearParents(state::Scene &r, std::span<const state::Entity> children) {
    // Each detached child beside its parent, grouped by parent and ordered by child index within a group.
    std::vector<std::pair<state::Entity, state::Entity>> detached;
    for (const auto child : children)
        if (const auto *parent = r.try_get<const SceneParent>(child)) detached.emplace_back(parent->Parent, child);
    if (detached.empty()) return;
    std::ranges::sort(detached, {}, [](const auto &link) { return std::pair{state::Index(link.first), state::Index(link.second)}; });
    detached.erase(std::unique(detached.begin(), detached.end()), detached.end());
    for (auto group = detached.begin(); group != detached.end();) {
        const auto parent = group->first;
        const auto group_end = std::find_if(group, detached.end(), [parent](const auto &link) { return link.first != parent; });
        const auto detaching = [&](state::Entity child) {
            const auto it = std::lower_bound(group, group_end, state::Index(child), [](const auto &link, uint32_t index) { return state::Index(link.second) < index; });
            return it != group_end && it->second == child;
        };
        auto remaining = group_end - group;
        auto previous = state::Null;
        for (auto child = FirstChildOf(r, parent); child != state::Null && remaining > 0;) {
            const auto next = r.get<const SceneChildren>(child).NextSibling;
            if (detaching(child)) {
                if (previous == state::Null) SetFirstChild(r, parent, next);
                else r.patch<SceneChildren>(previous, [next](auto &c) { c.NextSibling = next; });
                r.patch<SceneChildren>(child, [](auto &c) { c.NextSibling = state::Null; });
                r.remove<SceneParent>(child);
                --remaining;
            } else {
                previous = child;
            }
            child = next;
        }
        group = group_end;
    }
}

const Transform *ComposedLocal(const state::Scene &r, state::Entity e) {
    if (const auto *posed = r.try_get<const PosedLocal>(e)) return &posed->Value;
    return r.try_get<const Transform>(e);
}

void CommitEditedLocal(state::Scene &r, state::Entity e, const Transform &edited) {
    if (!r.all_of<PosedLocal>(e)) {
        if (r.all_of<Transform>(e)) r.replace<Transform>(e, edited);
        return;
    }
    r.patch<PosedLocal>(e, [&](PosedLocal &posed) { posed.Value = edited; });
    if (!r.all_of<Transform>(e)) return;
    const auto posed = animation::PosedTransformComponents(r, animation::AnimationsViewport(r), e);
    r.patch<Transform>(e, [&](Transform &t) {
        if (!(posed & animation::TranslationBit)) t.P = edited.P;
        if (!(posed & animation::RotationBit)) t.R = edited.R;
        if (!(posed & animation::ScaleBit)) t.S = edited.S;
    });
}

Transform ComposeWorldTransform(const state::Scene &r, state::Entity e) {
    if (const auto *world = WorldTransformOf(r, e)) return *world;
    const auto parent = ParentOrNull(r, e);
    return ToTransform((parent != state::Null ? ToMatrix(ComposeWorldTransform(r, parent)) : I4) * ToMatrix(*ComposedLocal(r, e)));
}

void RecomputeWorldTransforms(state::Scene &r, std::span<const state::Entity> roots, std::span<const state::Entity> owned, std::span<const state::Entity> held) {
    // Roots in ascending depth, so every parent is current before a dirty descendant composes under it.
    std::vector<std::pair<uint32_t, state::Entity>> ordered;
    ordered.reserve(roots.size());
    for (const auto root : roots) {
        if (ContainsInIndexOrder(owned, root)) continue;
        uint32_t depth = 0;
        for (auto parent = ParentOrNull(r, root); parent != state::Null; parent = ParentOrNull(r, parent)) ++depth;
        ordered.emplace_back(depth, root);
    }
    std::ranges::sort(ordered, {}, [](const auto &entry) { return std::pair{entry.first, state::Index(entry.second)}; });

    // Each pending node with the index of its parent's world matrix, or NoParent at a root without a parent.
    static constexpr uint32_t NoParent{UINT32_MAX};
    struct Pending {
        state::Entity Entity;
        uint32_t ParentMatrix;
    };
    std::vector<Pending> pending;
    std::vector<mat4> matrices;
    const auto &placed = reactive(r, state::Change::RenderInstanceCreated);
    for (const auto &[_, root] : ordered) {
        matrices.clear();
        const auto parent = ParentOrNull(r, root);
        if (parent != state::Null) matrices.push_back(ToMatrix(*WorldTransformOf(r, parent)));
        pending.push_back({root, parent != state::Null ? 0u : NoParent});
        while (!pending.empty()) {
            const auto [e, parent_matrix] = pending.back();
            pending.pop_back();
            const auto *local = ComposedLocal(r, e);
            if (!local) continue;
            const Transform world = parent_matrix == NoParent ? *local : ToTransform(matrices[parent_matrix] * ToMatrix(*local));
            // An unchanged world transform leaves its subtree current, and an instance slot placed this settle holds no earlier value.
            const auto slot = WorldSlot(r, e);
            if (const auto *current = placed.contains(e) ? nullptr : WorldTransformAt(r, e, slot); current && *current == world) continue;
            SetWorldTransformAt(r, e, slot, world);
            if (ContainsInIndexOrder(held, e) || FirstChildOf(r, e) == state::Null) continue;
            matrices.push_back(ToMatrix(world));
            const auto matrix = uint32_t(matrices.size() - 1);
            for (const auto child : Children{&r, e})
                if (!ContainsInIndexOrder(owned, child)) pending.push_back({child, matrix});
        }
    }
}

void SetParent(state::Scene &r, state::Entity child, state::Entity parent) {
    LinkAhead(r, child, parent, FirstChildOf(r, parent));
    SetFirstChild(r, parent, child);
}

bool SetParentKeepWorld(state::Scene &r, std::span<const state::Entity> children, state::Entity parent) {
    // The parent and its ancestors, none of which can become its child.
    std::vector<state::Entity> lineage;
    for (auto e = parent; e != state::Null; e = ParentOrNull(r, e)) lineage.push_back(e);
    bool looped = false;
    std::vector<state::Entity> linked;
    linked.reserve(children.size());
    for (const auto child : children) {
        if (std::ranges::contains(lineage, child)) looped |= child != parent;
        else linked.push_back(child);
    }
    if (linked.empty()) return looped;

    const auto parent_world_inv = Inverse(ToMatrix(ComposeWorldTransform(r, parent)));
    std::vector<Transform> locals;
    locals.reserve(linked.size());
    for (const auto child : linked) locals.push_back(ToTransform(parent_world_inv * ToMatrix(ComposeWorldTransform(r, child))));
    ClearParents(r, linked);
    // Each child links at the head, so the parent's head is written once after the last.
    auto first_child = FirstChildOf(r, parent);
    for (size_t i = 0; i < linked.size(); ++i) {
        LinkAhead(r, linked[i], parent, first_child);
        r.emplace_or_replace<Transform>(linked[i], locals[i]);
        first_child = linked[i];
    }
    SetFirstChild(r, parent, first_child);
    return looped;
}
