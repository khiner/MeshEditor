#include "action/Core.h"
#include "CameraTypes.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "action/Updatable.h"
#include "animation/AnimationData.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModes.h"
#include "gpu/ViewportTheme.h"
#include "numeric/VectorMath.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "scene/WorldTransform.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportInteractionState.h"

#include <algorithm>

namespace action {
ComponentField UpdatedField(state::TypeKey key, uint16_t offset) {
    std::string component;
    const FieldRange *field = &NoField;
    ForUpdatable(state::Slot(key), [&]<typename C> {
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

template<typename Field>
void ApplyUpdate(state::Scene &r, state::Entity viewport, const Update<Field> &a) {
    ForUpdatable(state::Slot(a.ComponentType), [&]<typename C> {
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
            a.Target, viewport,
            [&] { return Traits::Active(r); },
            [&](auto &&fn) { Traits::ForEachSelected(r, fn); },
            [&](state::Entity e) {
                if (Traits::Has(r, e)) write(e, a.Value);
            }
        );
    });
}
} // namespace

void Apply(state::Scene &r, state::Entity viewport, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}
} // namespace action
