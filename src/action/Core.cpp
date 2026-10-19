#include "action/Core.h"
#include "action/Dispatch.h"
#include "animation/AnimationData.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "gpu/ViewportTheme.h"
#include "procedure/NodeTheme.h"
#include "procedure/Procedure.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
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

namespace {
template<typename Field>
Field ClampField(Field v, const FieldSpec &spec) {
    const auto lo = LowerBound<Field>(spec), hi = UpperBound<Field>(spec);
    if constexpr (VectorField<Field>) return Min(Max(v, lo), hi);
    else return std::clamp(v, lo, hi);
}

// Writes the field of the view value on the viewport the target names.
template<typename C, typename Field>
void ApplyViewUpdate(state::Registry &r, const Update<Field> &a) {
    ForEachComponentTarget<C>(r, a.Target, [&](C &component) {
        auto value = a.Value;
        if constexpr (DeltaField<Field> && !RotationField<Field>) value = ClampField(value, FieldAt<C>(a.Offset).Spec);
        std::memcpy(reinterpret_cast<std::byte *>(&component) + a.Offset, &value, sizeof(Field));
    });
}

// Writes the field of the component on each entity the target resolves to.
template<typename C, typename Field>
void ApplyEntityUpdate(state::Registry &r, const Update<Field> &a) {
    using Traits = UpdateTraits<C>;
    const auto write = [&](state::Entity e, Field value) {
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
    ForUpdatable(a.ComponentType, [&]<typename C> {
        if constexpr (ViewportState::Holds<C>) ApplyViewUpdate<C>(r, a);
        else ApplyEntityUpdate<C>(r, a);
    });
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
                bool tracked = false;
                ForUpdatable(a.ComponentType, [&]<typename C> {
                    if constexpr (!ViewportState::Holds<C>) tracked = snapshot::SnapshotTable()[state::Type<C>()].History;
                });
                return tracked;
            },
            [](const DestroyEntity &) { return true; },
        },
        action
    );
}
} // namespace action
