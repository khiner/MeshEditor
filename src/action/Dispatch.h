#pragma once

#include "action/Core.h"
#include "mesh/PrimitiveType.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"

#include <cstring>
#include <functional>

// Field access for Update on each updatable component.
namespace action {
// Reads and writes a field at a byte offset of a component held on object entities.
template<typename C>
struct UpdateTraits {
    static bool Has(const state::Scene &r, state::Entity e) { return r.all_of<C>(e); }
    static state::Entity Active(const state::Scene &r) {
        const auto e = FindActiveEntity(r);
        return e != state::Null && Has(r, e) ? e : state::Null;
    }
    static void ForEachSelected(state::Scene &r, const std::function<void(state::Entity)> &fn) {
        for (const auto e : r.view<Selected>())
            if (Has(r, e)) fn(e);
    }
    static void Read(const state::Scene &r, state::Entity e, uint16_t offset, void *dst, size_t size) {
        std::memcpy(dst, reinterpret_cast<const std::byte *>(&r.get<const C>(e)) + offset, size);
    }
    static void Write(state::Scene &r, state::Entity e, uint16_t offset, const void *src, size_t size) {
        r.patch<C>(e, [&](C &c) { std::memcpy(reinterpret_cast<std::byte *>(&c) + offset, src, size); });
    }
};

// PrimitiveShape lives on the mesh entity behind an instance.
// Fields address the current shape alternative, the selection covers meshes sharing the active shape, and each write rebuilds the mesh.
template<>
struct UpdateTraits<PrimitiveShape> {
    static bool Has(const state::Scene &, state::Entity);
    static state::Entity Active(const state::Scene &);
    static void ForEachSelected(state::Scene &, const std::function<void(state::Entity)> &);
    static void Read(const state::Scene &, state::Entity, uint16_t offset, void *dst, size_t size);
    static void Write(state::Scene &, state::Entity, uint16_t offset, const void *src, size_t size);
};

// Cache each target's initial field value for the duration of a drag.
template<typename Field>
Field FieldGestureStart(state::Scene &r, state::Entity e, state::TypeId comp, uint16_t offset, auto &&read) {
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
