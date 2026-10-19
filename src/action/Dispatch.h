#pragma once

#include "action/Nodes.h"
#include "action/ScopeResolve.h"
#include "action/Updatable.h"
#include "armature/ArmatureComponents.h"
#include "scene/SceneGraph.h"
#include "selection/Selection.h"

namespace action {
// Use the innermost field range containing the offset.
// Inherit enclosing bounds when the field has none.
struct FieldRange {
    uint16_t Offset, Size;
    std::string Path;
    FieldSpec Spec;
};
template<typename T> void CollectFields(T &value, uint16_t base, const std::string &prefix, const FieldSpec &enclosing, std::vector<FieldRange> &out) {
    field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
        using F = std::remove_cvref_t<decltype(member)>;
        const auto offset = uint16_t(base + (reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&value)));
        const auto path = prefix + std::string{field::Name<T, I>};
        FieldSpec spec = Spec<T, I>;
        if (!spec.Bounded()) {
            spec.Min = enclosing.Min;
            spec.Max = enclosing.Max;
        }
        out.push_back({offset, uint16_t(sizeof(F)), path, spec});
        if constexpr (field::Walkable<F>) CollectFields(member, offset, path + ".", spec, out);
    });
}
template<typename C> const std::vector<FieldRange> &Fields() {
    static const auto fields = [] {
        std::vector<FieldRange> out;
        if constexpr (field::Walkable<C>) {
            C value{};
            CollectFields(value, 0, "", FieldSpec{}, out);
        }
        return out;
    }();
    return fields;
}
inline const FieldRange NoField{};
template<typename C> const FieldRange &FieldAt(uint16_t offset) {
    const FieldRange *found = &NoField;
    for (const auto &range : Fields<C>()) {
        if (offset >= range.Offset && offset < range.Offset + range.Size) found = &range;
    }
    return *found;
}
struct ComponentField {
    std::string Component;
    const FieldRange &Field;
};
ComponentField UpdatedField(state::TypeKey, uint16_t offset);

template<typename C>
struct ComponentUpdateTraits {
    static bool Has(const state::Registry &r, state::Entity e) { return r.all_of<C>(e); }
    static state::Entity Active(const state::Registry &r) { return ActiveWith<C>(r); }
    static void ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) { ForEachSelectedWith<C>(r, fn); }
    static void Read(const state::Registry &r, state::Entity e, uint16_t offset, void *dst, size_t size) {
        std::memcpy(dst, reinterpret_cast<const std::byte *>(&r.get<const C>(e)) + offset, size);
    }
    static void Write(state::Registry &r, state::Entity e, uint16_t offset, const void *src, size_t size) {
        r.patch<C>(e, [&](C &c) { std::memcpy(reinterpret_cast<std::byte *>(&c) + offset, src, size); });
    }
    static const FieldSpec &Bounds(const state::Registry &, state::Entity, uint16_t offset) { return FieldAt<C>(offset).Spec; }
};
template<typename C>
struct UpdateTraits : ComponentUpdateTraits<C> {};

// Edit the Inset or Bevel node appended at the start of the viewport drag.
template<typename C>
    requires std::same_as<C, mesh::Inset> || std::same_as<C, mesh::Bevel>
struct UpdateTraits<C> : ComponentUpdateTraits<C> {
    static void ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) {
        nodes::ForEachResultNode(r, [&](state::Entity, state::Entity node) {
            if (r.all_of<C>(node)) fn(node);
        });
    }
};

// A bone delta edits the active bone, and a selection edit fans out to the selected bones.
template<>
struct UpdateTraits<BoneDelta> : ComponentUpdateTraits<BoneDelta> {
    static state::Entity Active(const state::Registry &r) {
        const auto e = FindActiveBone(r);
        return e != state::Null && Has(r, e) ? e : state::Null;
    }
    static void ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) {
        for (const auto e : r.view<BoneSelection>()) {
            if (Has(r, e)) fn(e);
        }
    }
};

// A pose edit also persists its unanimated components in the node's Transform.
template<>
struct UpdateTraits<PosedLocal> : ComponentUpdateTraits<PosedLocal> {
    static void Write(state::Registry &r, state::Entity e, uint16_t offset, const void *src, size_t size) {
        PosedLocal edited = r.get<const PosedLocal>(e);
        std::memcpy(reinterpret_cast<std::byte *>(&edited) + offset, src, size);
        CommitEditedLocal(r, e, edited.Value);
    }
};

// A node layout edit addresses the edited procedure's active node and fans out to its selected nodes.
template<>
struct UpdateTraits<procedure::NodeLayout> : ComponentUpdateTraits<procedure::NodeLayout> {
    static state::Entity Active(const state::Registry &);
    static void ForEachSelected(state::Registry &, const std::function<void(state::Entity)> &);
};

// Resolve PrimitiveShape through the instance mesh.
// Selection edits affect meshes with the active shape type; rebuild them after applying changes.
template<>
struct UpdateTraits<PrimitiveShape> {
    static bool Has(const state::Registry &, state::Entity);
    static state::Entity Active(const state::Registry &);
    static void ForEachSelected(state::Registry &, const std::function<void(state::Entity)> &);
    static void Read(const state::Registry &, state::Entity, uint16_t offset, void *dst, size_t size);
    static void Write(state::Registry &, state::Entity, uint16_t offset, const void *src, size_t size);
    static const FieldSpec &Bounds(const state::Registry &, state::Entity, uint16_t offset);
};

// Cache each target's initial field value for the duration of a drag.
template<typename Field>
Field FieldGestureStart(state::Registry &r, state::Entity e, state::TypeId comp, uint16_t offset, auto &&read) {
    static_assert(sizeof(Field) <= sizeof(DragFieldStart::Bytes));
    Field start;
    if (const auto *snap = r.try_get<DragFieldStart>(e); snap && snap->Comp == comp && snap->Offset == offset) {
        std::memcpy(&start, snap->Bytes.data(), sizeof(Field));
        return start;
    }
    read(start);
    DragFieldStart s{comp, offset, uint16_t(sizeof(Field)), {}};
    std::memcpy(s.Bytes.data(), &start, sizeof(Field));
    r.emplace_or_replace<DragFieldStart>(e, s);
    return start;
}
} // namespace action
