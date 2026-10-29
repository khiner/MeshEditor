#include "action/Core.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/AnimationData.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "gpu/ViewportTheme.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "scene/CameraLens.h"
#include "scene/HierarchyInstance.h"
#include "snapshot/SnapshotRoles.h"
#include "timeline/Timeline.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

namespace action {
ComponentField UpdatedField(state::TypeKey key, uint16_t offset) {
    std::string component;
    const FieldRange *field = &NoField;
    ForUpdatable(key, [&]<typename C> {
        component = state::LeafName<C>();
        field = &FieldAt<C>(offset);
    });
    return {std::move(component), *field};
}

std::optional<Core> ReadUpdate(const state::Registry &r, state::Entity entity, state::TypeKey key, uint16_t offset) {
    std::optional<Core> result;
    ForUpdatable(key, [&]<typename C> {
        const auto *component = r.try_get<const C>(entity);
        if (!component || snapshot::SnapshotTable()[state::Type<C>()].Kind == snapshot::Role::Workspace) return;
        const auto visit = [&](this const auto &self, const auto &value) -> void {
            using V = std::remove_cvref_t<decltype(value)>;
            const auto at = reinterpret_cast<const std::byte *>(&value) - reinterpret_cast<const std::byte *>(component);
            if constexpr (std::constructible_from<Core, Update<V>>) {
                if (at == offset) result = Update<V>{entity, key, offset, value};
            } else if constexpr (field::Walkable<V>) {
                field::ForEach(value, [&](const auto &member, auto) { self(member); });
            }
        };
        visit(*component);
    });
    return result;
}

namespace {
template<typename Field>
Field ClampField(Field v, const FieldSpec &spec) {
    const auto lo = LowerBound<Field>(spec), hi = UpperBound<Field>(spec);
    if constexpr (VectorField<Field>) return Min(Max(v, lo), hi);
    else return std::clamp(v, lo, hi);
}

// Writes the field of the component on each entity the target resolves to.
template<typename C, typename Field>
void ApplyEntityUpdate(state::Registry &r, const Update<Field> &a) {
    using Traits = UpdateTraits<C>;
    if constexpr (std::same_as<C, procedure::ObjectInfo> && std::same_as<Field, state::Entity>) {
        std::vector<state::Entity> nodes;
        ForEachTarget(a.Target, [&] { return Traits::Active(r); }, [&](auto &&fn) { Traits::ForEachSelected(r, fn); }, [&](state::Entity e) { if (Traits::Has(r, e)) nodes.push_back(e); });
        if (!procedure::ObjectReferencesAcyclic(r, nodes, a.Value)) return Fail(r, "Object references would create a geometry dependency cycle");
    }
    if constexpr (std::same_as<C, procedure::InstanceOnPoints> && std::same_as<Field, state::Entity>) {
        if (a.Value != state::Null && !r.all_of<ObjectKind>(a.Value)) return Fail(r, "Choose an object hierarchy as the prototype");
        bool cycle = false;
        ForEachTarget(a.Target, [&] { return Traits::Active(r); }, [&](auto &&fn) { Traits::ForEachSelected(r, fn); }, [&](state::Entity node) {
            const auto *owner = r.try_get<const procedure::ProcedureNode>(node);
            if (!owner) return;
            for (const auto [object, instance] : r.view<const Instance>().each())
                if (instance.Entity == owner->Procedure && HierarchyParentCycle(r, a.Value, object)) cycle = true; });
        if (cycle) return Fail(r, "The prototype would recursively instance its producer");
    }
    const auto write = [&](state::Entity e, Field value) {
        constexpr bool placement = std::same_as<C, Transform> || std::same_as<C, PosedLocal> || std::same_as<C, Visibility>;
        if constexpr (std::same_as<Field, state::Entity>) std::tie(e, value) = HierarchyReferenceTarget(r, e, value, placement);
        else e = HierarchyEditTarget(r, e, placement);
        if constexpr (DeltaField<Field> && !RotationField<Field>) value = ClampField(value, Traits::Bounds(r, e, a.Offset));
        Traits::Write(r, e, a.Offset, &value, sizeof(Field));
    };
    if constexpr (DeltaField<Field>) {
        if (std::holds_alternative<OnSelectedDelta>(a.Target)) {
            // Offset each selected target by the active target's change from its drag start.
            const auto active = Traits::Active(r);
            if (active == state::Null) return;
            const auto start = [&](state::Entity e) {
                return FieldGestureStart<Field>(r, e, state::Type<C>(), a.Offset, [&](Field &v) { Traits::Read(r, e, a.Offset, &v, sizeof(Field)); });
            };
            const auto active_start = start(active);
            Traits::ForEachSelected(r, [&](state::Entity e) {
                if (e == active) {
                    write(e, a.Value);
                } else if constexpr (std::integral<Field>) {
                    // Accumulate in a wider signed type so an unsigned field can't wrap on a downward delta.
                    const auto value = int64_t(start(e)) + int64_t(a.Value) - int64_t(active_start);
                    write(e, Field(std::clamp<int64_t>(value, std::numeric_limits<Field>::min(), std::numeric_limits<Field>::max())));
                } else if constexpr (std::integral<Limit<Field>>) {
                    using L = Limit<Field>;
                    auto value = start(e);
                    for (int c = 0; c < Components<Field>; ++c) {
                        value[c] = L(std::clamp<int64_t>(int64_t(value[c]) + int64_t(a.Value[c]) - int64_t(active_start[c]), std::numeric_limits<L>::min(), std::numeric_limits<L>::max()));
                    }
                    write(e, value);
                } else if constexpr (RotationField<Field>) {
                    // Rotate each target by the active target's rotation since the drag start.
                    write(e, Normalize(a.Value * Conjugate(active_start) * start(e)));
                } else {
                    write(e, start(e) + (a.Value - active_start));
                }
            });
            return;
        }
    }
    ForEachTarget(
        a.Target,
        [&] { return Traits::Active(r); },
        [&](auto &&fn) { Traits::ForEachSelected(r, fn); },
        [&](state::Entity e) {
            if (Traits::Has(r, e)) write(e, a.Value);
        }
    );
}

template<typename Field>
void ApplyUpdate(state::Registry &r, const Update<Field> &a) {
    ForUpdatable(a.ComponentType, [&]<typename C> { ApplyEntityUpdate<C>(r, a); });
}
} // namespace

void Apply(state::Registry &r, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, a); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}

bool IsRecordable(const Core &action) {
    return std::visit(
        overloaded{
            []<typename Field>(const Update<Field> &a) {
                bool recorded = false;
                ForUpdatable(a.ComponentType, [&]<typename C> { recorded = snapshot::SnapshotTable()[state::Type<C>()].Kind != snapshot::Role::Workspace; });
                return recorded;
            },
            [](const DestroyEntity &) { return true; },
        },
        action
    );
}
} // namespace action
