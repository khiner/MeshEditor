#include "scene/SceneGraph.h"

#include "TransformMath.h"
#include "animation/Clips.h"
#include "mesh/MeshComponents.h"
#include "render/GpuBuffers.h"
#include "render/RenderChanges.h"
#include "scene/Entity.h"
#include "scene/SceneChanges.h"

namespace {
void SetFirstChild(state::Registry &r, state::Entity parent, state::Entity first_child) {
    if (r.all_of<SceneChildren>(parent)) r.patch<SceneChildren>(parent, [first_child](auto &c) { c.FirstChild = first_child; });
    else r.emplace<SceneChildren>(parent, SceneChildren{.FirstChild = first_child});
}

// Links the parentless child ahead of `next_sibling` under the parent, leaving the parent's head to the caller.
// The child joins its new root's scenes.
void LinkAhead(state::Registry &r, state::Entity child, state::Entity parent, state::Entity next_sibling) {
    if (r.all_of<SceneChildren>(child)) r.patch<SceneChildren>(child, [next_sibling](auto &c) { c.NextSibling = next_sibling; });
    else r.emplace<SceneChildren>(child, SceneChildren{.NextSibling = next_sibling});
    r.emplace<SceneParent>(child, parent);
    r.remove<SceneMembership>(child);
}

// The world transform of a node with the local transform under the parent's world matrix, or the local transform itself at a root.
Transform ComposeUnder(const mat4 *parent_world, const Transform &local) { return parent_world ? ToTransform(*parent_world * ToMatrix(local)) : local; }

state::Entity FirstChildOf(const state::Registry &r, state::Entity parent) {
    const auto *children = r.try_get<const SceneChildren>(parent);
    return children ? children->FirstChild : state::Null;
}

// Return the node's transform slot, or UINT32_MAX when WorldTransform stores it.
// An armature part's slot contains its display transform.
uint32_t WorldSlot(const state::Registry &r, state::Entity e) {
    const auto *render = r.try_get<const RenderInstance>(e);
    return render && !r.all_of<SubElementOf>(e) ? render->BufferIndex : UINT32_MAX;
}

const Transform *WorldTransformAt(const state::Registry &r, state::Entity e, uint32_t slot) {
    if (slot != UINT32_MAX) return r.Resources.get<const GpuBuffers>().Instances.TransformBuffer.GetSpan<Transform>({slot, 1u}).data();
    return r.try_get<const WorldTransform>(e);
}

void SetWorldTransformAt(state::Registry &r, state::Entity e, uint32_t slot, const Transform &world) {
    if (slot != UINT32_MAX) r.Resources.get<GpuBuffers>().Instances.TransformBuffer.GetMutableSpan<Transform>({slot, 1u})[0] = world;
    else r.emplace_or_replace<WorldTransform>(e, world);
    r.Resources.get<SceneChanges>().WorldTransform.emplace(e);
}
} // namespace

const Transform *WorldTransformOf(const state::Registry &r, state::Entity e) { return WorldTransformAt(r, e, WorldSlot(r, e)); }
void SetWorldTransform(state::Registry &r, state::Entity e, const Transform &world) { SetWorldTransformAt(r, e, WorldSlot(r, e), world); }

mat4 GetParentDelta(const state::Registry &r, state::Entity e) {
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

state::Entity ParentOrNull(const state::Registry &r, state::Entity e) {
    const auto *parent = r.try_get<const SceneParent>(e);
    return parent ? parent->Parent : state::Null;
}

state::Entity MembershipRoot(const state::Registry &r, state::Entity node) {
    for (auto e = node; e != state::Null;) {
        if (const auto *parent = r.try_get<const SceneParent>(e)) e = parent->Parent;
        else if (r.all_of<SceneMembership>(e)) return e;
        else if (const auto *sub = r.try_get<const SubElementOf>(e)) e = sub->Parent;
        else return state::Null;
    }
    return state::Null;
}

const SceneMembership *RootMembership(const state::Registry &r, state::Entity node) {
    const auto root = MembershipRoot(r, node);
    return root != state::Null ? r.try_get<const SceneMembership>(root) : nullptr;
}

bool InScene(const state::Registry &r, state::Entity node, state::Entity scene) {
    const auto *membership = RootMembership(r, node);
    return membership && std::ranges::contains(membership->Scenes, scene);
}

void ClearParents(state::Registry &r, std::span<const state::Entity> children) {
    // Each detached child beside its parent, grouped by parent and ordered by child index within a group.
    struct Link {
        state::Entity Parent, Child;
        bool operator==(const Link &) const = default;
    };
    std::vector<Link> detached;
    for (const auto child : children) {
        if (const auto *parent = r.try_get<const SceneParent>(child)) detached.emplace_back(parent->Parent, child);
    }
    if (detached.empty()) return;
    std::ranges::sort(detached, {}, [](const auto &link) { return std::pair{state::Index(link.Parent), state::Index(link.Child)}; });
    detached.erase(std::ranges::unique(detached).begin(), detached.end());
    // Each new root keeps the scenes of the root it leaves.
    std::vector<std::pair<state::Entity, SceneMembership>> memberships;
    for (const auto &link : detached) {
        if (const auto *membership = RootMembership(r, link.Parent)) memberships.emplace_back(link.Child, *membership);
    }
    for (auto group = detached.begin(); group != detached.end();) {
        const auto parent = group->Parent;
        const auto group_end = std::ranges::find_if(group, detached.end(), [parent](const auto &link) { return link.Parent != parent; });
        const auto detaching = [&](state::Entity child) {
            const auto it = std::ranges::lower_bound(group, group_end, state::Index(child), {}, [](const auto &link) { return state::Index(link.Child); });
            return it != group_end && it->Child == child;
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
    for (auto &[child, membership] : memberships) r.emplace<SceneMembership>(child, std::move(membership));
}

const Transform *ComposedLocal(const state::Registry &r, state::Entity e) {
    if (const auto *posed = r.try_get<const PosedLocal>(e)) return &posed->Value;
    return r.try_get<const Transform>(e);
}

void CommitEditedLocal(state::Registry &r, state::Entity e, const Transform &edited) {
    if (!r.all_of<PosedLocal>(e)) {
        if (r.all_of<Transform>(e)) r.replace<Transform>(e, edited);
        return;
    }
    r.patch<PosedLocal>(e, [&](PosedLocal &posed) { posed.Value = edited; });
    if (!r.all_of<Transform>(e)) return;
    const auto posed = animation::PosedTransformComponents(r, e);
    r.patch<Transform>(e, [&](Transform &t) {
        if (!(posed & animation::TranslationBit)) t.P = edited.P;
        if (!(posed & animation::RotationBit)) t.R = edited.R;
        if (!(posed & animation::ScaleBit)) t.S = edited.S;
    });
}

Transform ComposeTransform(const state::Registry &r, state::Entity e, state::Entity frame, std::span<const state::Entity> owned) {
    if (e == frame) return {};
    if (ContainsInIndexOrder(owned, e)) return *WorldTransformOf(r, e);
    const auto parent = ParentOrNull(r, e);
    const auto &local = *ComposedLocal(r, e);
    if (parent == frame) return local;
    const auto parent_transform = ToMatrix(ComposeTransform(r, parent, frame, owned));
    return ComposeUnder(&parent_transform, local);
}

void RecomputeWorldTransforms(state::Registry &r, std::span<const state::Entity> roots, std::span<const state::Entity> owned, std::span<const state::Entity> held) {
    // Roots in ascending depth, so every parent is current before a dirty descendant composes under it.
    struct Root {
        uint32_t Depth;
        state::Entity Entity;
    };
    std::vector<Root> ordered;
    ordered.reserve(roots.size());
    for (const auto root : roots) {
        if (ContainsInIndexOrder(owned, root)) continue;
        uint32_t depth = 0;
        for (auto parent = ParentOrNull(r, root); parent != state::Null; parent = ParentOrNull(r, parent)) ++depth;
        ordered.emplace_back(depth, root);
    }
    std::ranges::sort(ordered, {}, [](const auto &root) { return std::pair{root.Depth, state::Index(root.Entity)}; });

    // Each pending node with the index of its parent's world matrix, or NoParent at a root without a parent.
    static constexpr uint32_t NoParent{UINT32_MAX};
    struct Pending {
        state::Entity Entity;
        uint32_t ParentMatrix;
    };
    std::vector<Pending> pending;
    std::vector<mat4> matrices;
    const auto &placed = r.Resources.get<const RenderChanges>().InstanceCreated;
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
            const auto world = ComposeUnder(parent_matrix == NoParent ? nullptr : &matrices[parent_matrix], *local);
            // Skip unchanged subtrees; always initialize newly placed instance slots.
            const auto slot = WorldSlot(r, e);
            if (const auto *current = placed.contains(e) ? nullptr : WorldTransformAt(r, e, slot); current && *current == world) continue;
            SetWorldTransformAt(r, e, slot, world);
            if (ContainsInIndexOrder(held, e) || FirstChildOf(r, e) == state::Null) continue;
            matrices.push_back(ToMatrix(world));
            const auto matrix = uint32_t(matrices.size() - 1);
            for (const auto child : Children{&r, e}) {
                if (!ContainsInIndexOrder(owned, child)) pending.push_back({child, matrix});
            }
        }
    }
}

void SetParent(state::Registry &r, state::Entity child, state::Entity parent) {
    LinkAhead(r, child, parent, FirstChildOf(r, parent));
    SetFirstChild(r, parent, child);
}

bool SetParentKeepWorld(state::Registry &r, std::span<const state::Entity> children, state::Entity parent, std::span<const state::Entity> owned) {
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

    const auto parent_world_inv = Inverse(ToMatrix(ComposeTransform(r, parent, state::Null, owned)));
    std::vector<Transform> locals;
    locals.reserve(linked.size());
    for (const auto child : linked) locals.push_back(ToTransform(parent_world_inv * ToMatrix(ComposeTransform(r, child, state::Null, owned))));
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
