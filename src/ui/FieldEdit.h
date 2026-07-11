#pragma once

// ui::Edit — terse, single-line wrappers around ImGui controls that read a field, run the widget,
// and feed any change into action gesture grouping: a drag stages each frame and commits one
// `action::UpdateOf<Ms...>` on release, and an instantaneous edit stages and commits in a single frame.
//
// Bound to a registry, optionally an entity:
//   `ui::Edit{R}`     — reads/writes target the registry's active entity (reads resolve via
//                       FindActiveEntity(R) at Run time). Emits Update<T> with Scope::Active. While Alt is
//                       held a number drag deltas every selected entity (Scope::SelectedDelta) and a discrete
//                       or click-to-edit typed value copies the value (Scope::Selected).
//   `ui::Edit{R, E}`  — reads from E and writes carry E. Emits Update<T> with Scope::Entity.
//
// `HasEntity` is a template bool (deduced via the constructor) so only the relevant code path instantiates.

#include "action/Build.h"
#include "action/Emit.h"
#include "scene/Entity.h" // FindActiveEntity
#include "selection/SelectionComponents.h" // Selected

#include <imgui_internal.h> // TempInputIsActive

#include <array>
#include <functional>

namespace ui {

// House drag widgets — use instead of ImGui::DragFloat* so the modifiers are consistent: Alt stays free for
// app use (not ImGui's fine-control) and Shift is fine control (Blender-style 0.05x).
inline bool DragFloat(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat(label, v, ImGui::GetIO().KeyShift ? speed * 0.05f : speed, lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat2(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat2(label, v, ImGui::GetIO().KeyShift ? speed * 0.05f : speed, lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat3(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat3(label, v, ImGui::GetIO().KeyShift ? speed * 0.05f : speed, lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat4(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat4(label, v, ImGui::GetIO().KeyShift ? speed * 0.05f : speed, lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}

// Resolve the active-form scope from the Alt modifier. No Alt → the active entity. Alt → apply to all
// selected, where a delta-capable number drag deltas each from its own start and anything else copies.
inline action::Scope ScopeFromAlt(bool delta_capable = false) {
    if (!ImGui::GetIO().KeyAlt) return action::Scope::Active;
    return delta_capable ? action::Scope::SelectedDelta : action::Scope::Selected;
}

namespace detail {
// State for the one in-progress widget gesture (ImGui has a single active item, so one slot each).
// GestureScope is captured at gesture start and frozen so a mid-drag Alt toggle doesn't flip it.
inline action::Scope GestureScope{action::Scope::Active};
// Field value at gesture start (type-erased, ≤16 bytes covers any Update field), so a SelectedDelta drag deltas from it.
inline std::array<std::byte, 16> GestureStartValue{};
// Set once the gesture entered a click-to-edit text input: its value applies only on commit, and an Alt edit copies it rather than deltaing.
inline bool GestureTyped{false};
// Reverts to the start value on an aborted edit (Escape / no-op release); set when the widget activates.
inline std::function<void()> GestureCancel;

// A composite gesture (several sub-widgets emitting one action) is staging an uncommitted value.
inline bool CompositeGestureOpen{false};
} // namespace detail

// Group a composite editor (several sub-widgets emitting one whole-value action, so ui::Edit's per-field
// grouping doesn't apply) into one record per drag: stage while editing, commit when the drag releases.
template<typename MakeAction>
void Gesture(bool changed, MakeAction &&make) {
    if (changed) {
        action::EmitStaged(make());
        detail::CompositeGestureOpen = true;
    } else if (detail::CompositeGestureOpen && !ImGui::IsAnyItemActive()) {
        action::Commit();
        detail::CompositeGestureOpen = false;
    }
}

// Walk a chain Ms... from the outermost object down to the leaf field.
template<auto M, auto... Rest>
decltype(auto) ReadChain(const auto &obj) {
    if constexpr (sizeof...(Rest) == 0) return obj.*M;
    else return ReadChain<Rest...>(obj.*M);
}

template<typename F>
consteval ImGuiDataType ImGuiDt() {
    if constexpr (std::same_as<F, int8_t>) return ImGuiDataType_S8;
    else if constexpr (std::same_as<F, uint8_t>) return ImGuiDataType_U8;
    else if constexpr (std::same_as<F, int16_t>) return ImGuiDataType_S16;
    else if constexpr (std::same_as<F, uint16_t>) return ImGuiDataType_U16;
    else if constexpr (std::same_as<F, int32_t>) return ImGuiDataType_S32;
    else if constexpr (std::same_as<F, uint32_t>) return ImGuiDataType_U32;
    else if constexpr (std::same_as<F, int64_t>) return ImGuiDataType_S64;
    else if constexpr (std::same_as<F, uint64_t>) return ImGuiDataType_U64;
    else if constexpr (std::same_as<F, float>) return ImGuiDataType_Float;
    else if constexpr (std::same_as<F, double>) return ImGuiDataType_Double;
    else static_assert(false, "ImGuiDt: unsupported scalar type");
}

// Map a field's FieldLimits to ImGui drag bounds. Unbounded → (0,0), ImGui's "no clamp". A one-sided limit
// uses ±FLT_MAX on the open side so ImGui still clamps the bounded side.
template<auto... Ms>
constexpr std::pair<float, float> DragBounds() {
    if constexpr (!HasLimits<Ms...>) return {0.f, 0.f};
    else {
        using L = FieldLimits<Ms...>;
        float lo = -FLT_MAX, hi = FLT_MAX;
        if constexpr (HasMin<Ms...>) lo = float(L::Min);
        if constexpr (HasMax<Ms...>) hi = float(L::Max);
        return {lo, hi};
    }
}

template<bool HasEntity, auto... Prefix>
struct Edit {
    entt::registry &R;
    [[no_unique_address]] std::conditional_t<HasEntity, entt::entity, std::monostate> E{};

    // Edit for a nested field, so callers don't repeat the outer member on each call.
    template<auto... More>
    Edit<HasEntity, Prefix..., More...> Sub() const {
        if constexpr (HasEntity) return {R, E};
        else return {R};
    }

    entt::entity ReadFrom() const {
        if constexpr (HasEntity) return E;
        else return FindActiveEntity(R);
    }

    template<typename T, typename Reg>
    const T &GetConst(Reg &r, entt::entity e) { return r.template get<const T>(e); }

    // Read field, hand a mutable copy to `widget`, and feed the result into gesture grouping: a drag stages
    // each frame and commits one record on release, an instantaneous edit stages+commits in one frame, and
    // an aborted edit (Escape) reverts to the gesture's start value.
    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool delta_capable = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        Field v = ReadChain<Prefix..., Ms...>(GetConst<action::detail::first_class<Prefix..., Ms...>>(R, ReadFrom()));
        const Field original = v;
        const bool changed = widget(v);

        if constexpr (!HasEntity) {
            if (ImGui::IsItemHovered() && R.template view<Selected>().size() > 1) ImGui::SetItemTooltip("Hold Alt to apply to all selected");
        }

        // Capture the scope + start value once at gesture begin, so all of its steps share one target.
        auto capture_gesture = [&] {
            detail::GestureTyped = false;
            std::memcpy(detail::GestureStartValue.data(), &original, sizeof(Field));
            if constexpr (!HasEntity) {
                detail::GestureScope = ScopeFromAlt(delta_capable && action::DeltaField<Field>);
                // Drop any baseline an interrupted prior drag left. Mutation outside an Apply handler.
                if (detail::GestureScope == action::Scope::SelectedDelta) R.template clear<action::DragFieldStart>();
            }
        };

        auto gesture_start = [] { Field s; std::memcpy(&s, detail::GestureStartValue.data(), sizeof(Field)); return s; };

        auto update = [&](const Field &val) {
            if constexpr (HasEntity) return action::UpdateOf<Prefix..., Ms...>(E, val);
            else {
                if constexpr (action::DeltaField<Field>) {
                    // Alt+drag records each selected entity's delta from its own start. A click-to-edit
                    // typed value is an absolute copy instead, so it falls through to the scope below.
                    if (detail::GestureScope == action::Scope::SelectedDelta && !detail::GestureTyped) {
                        return action::UpdateOf<Prefix..., Ms...>(action::Scope::SelectedDelta, Field(val - gesture_start()));
                    }
                }
                // A typed value over an Alt+drag gesture copies the absolute value to every selected entity.
                const auto scope = detail::GestureScope == action::Scope::SelectedDelta ? action::Scope::Selected : detail::GestureScope;
                return action::UpdateOf<Prefix..., Ms...>(scope, val);
            }
        };

        // Stage the value as action(s).
        // A numeric vector widget edits one component at a time, so it stages only the touched component(s).
        // A delta drag adds each from its start. Anything else sets it absolutely.
        auto stage = [&] {
            if constexpr (std::same_as<Field, vec2> || std::same_as<Field, vec3> || std::same_as<Field, vec4>) {
                if (delta_capable) {
                    const Field start = gesture_start();
                    const auto w = update(v); // resolves scope + target entity + component type + base offset
                    const bool delta = w.Scope == action::Scope::SelectedDelta;
                    for (typename Field::length_type i = 0; i < Field::length(); ++i) {
                        if (v[i] == start[i]) continue;
                        const uint16_t off = uint16_t(w.Offset + i * sizeof(float));
                        action::EmitStaged(action::Update<float>{w.Scope, w.Entity, w.ComponentType, off, delta ? v[i] - start[i] : v[i]});
                        // Revert a delta step (Escape) by re-zeroing the component on each selected entity.
                        if (delta) detail::GestureCancel = [c = w.ComponentType, e = w.Entity, off] { action::EmitCancel(action::Update<float>{action::Scope::SelectedDelta, e, c, off, 0.f}); };
                    }
                    return;
                }
            }
            action::EmitStaged(update(v));
        };

        // Capture the gesture's start update so an aborted edit can revert.
        if (ImGui::IsItemActivated()) {
            capture_gesture();
            detail::GestureCancel = [revert = update(original)] { action::EmitCancel(revert); };
        }
        // Latch a click-to-edit text input (vs a drag): its value applies only when the edit commits,
        // not per keypress, and an Alt edit copies it to the selection rather than deltaing.
        if (ImGui::TempInputIsActive(ImGui::GetItemID())) detail::GestureTyped = true;

        if (ImGui::IsItemDeactivatedAfterEdit()) { // Gesture committed: release / Enter / instantaneous widget.
            if (changed) stage();
            action::Commit();
            detail::GestureCancel = nullptr;
            return changed;
        }
        if (ImGui::IsItemDeactivated()) { // Gesture aborted (Escape / no-op release): revert to start, record nothing.
            if (detail::GestureCancel) {
                detail::GestureCancel();
                detail::GestureCancel = nullptr;
            }
            return false;
        }
        if (changed) {
            // A click-to-edit text input applies only when it commits (handled above), not per keypress.
            if (ImGui::IsItemActive() && detail::GestureTyped) return changed;
            // An instantaneous widget (combo selection) may change without a prior activation this frame.
            // Capture the scope now so its single emit honors the modifier.
            if (!ImGui::IsItemActive()) capture_gesture();
            stage();
            // An edit on an item that isn't held (e.g. a combo selection) is instantaneous: commit now
            // rather than waiting for a deactivation event that may not come.
            if (!ImGui::IsItemActive()) action::Commit();
        }
        return changed;
    }

    template<auto... Ms>
    bool Check(const char *label) {
        return Run<Ms...>([&](bool &v) { return ImGui::Checkbox(label, &v); });
    }

    // Drag bounds come from the field's FieldLimits (none → unbounded).
    template<auto... Ms>
    bool Drag(const char *label, float speed = 1.f, const char *fmt = "%.3f") {
        constexpr auto bounds = DragBounds<Prefix..., Ms...>();
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ui::DragFloat(label, &v, speed, bounds.first, bounds.second, fmt);
            else if constexpr (std::same_as<F, vec3>) return ui::DragFloat3(label, &v.x, speed, bounds.first, bounds.second, fmt);
            else if constexpr (std::same_as<F, vec4>) return ui::DragFloat4(label, &v.x, speed, bounds.first, bounds.second, fmt);
            else static_assert(false, "Edit::Drag: field type must be float, vec3, or vec4");
        },
                          /*delta_capable=*/true);
    }

    // Slider bounds come from the field's FieldLimits, which must declare both Min and Max.
    template<auto... Ms>
    bool Slider(const char *label, const char *fmt = nullptr, ImGuiSliderFlags flags = 0) {
        static_assert(HasMin<Prefix..., Ms...> && HasMax<Prefix..., Ms...>, "Edit::Slider: field must declare FieldLimits with both Min and Max");
        using L = FieldLimits<Prefix..., Ms...>;
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ImGui::SliderFloat(label, &v, F(L::Min), F(L::Max), fmt ? fmt : "%.3f", flags);
            else if constexpr (std::same_as<F, vec3>) return ImGui::SliderFloat3(label, &v.x, F(L::Min), F(L::Max), fmt ? fmt : "%.3f", flags);
            else if constexpr (std::same_as<F, double> || std::integral<F>) {
                F lof = F(L::Min), hif = F(L::Max);
                return ImGui::SliderScalar(label, ImGuiDt<F>(), &v, &lof, &hif, fmt, flags);
            } else static_assert(false, "Edit::Slider: unsupported field type");
        },
                          /*delta_capable=*/true);
    }

    template<auto... Ms>
    bool Input(const char *label, const char *fmt = nullptr) {
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ImGui::InputFloat(label, &v, 0.f, 0.f, fmt ? fmt : "%.3f");
            else if constexpr (std::same_as<F, double>) return ImGui::InputDouble(label, &v, 0.0, 0.0, fmt ? fmt : "%.3f");
            else static_assert(false, "Edit::Input: only float/double supported");
        });
    }

    // ColorEdit3 for vec3, ColorEdit4 for vec4 — picked by field type.
    template<auto... Ms>
    bool Color(const char *label) {
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, vec3>) return ImGui::ColorEdit3(label, &v.x);
            else if constexpr (std::same_as<F, vec4>) return ImGui::ColorEdit4(label, &v.x);
            else static_assert(false, "Edit::Color: field must be vec3 or vec4");
        });
    }

    // Combo over a contiguous enum represented by a packed C-string ("A\0B\0C\0").
    template<auto... Ms>
    bool Enum(const char *label, const char *items) {
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            static_assert(std::is_enum_v<F>, "Edit::Enum: field must be an enum");
            int i = int(v);
            if (!ImGui::Combo(label, &i, items)) return false;
            v = F(i);
            return true;
        });
    }

    // Write a value the caller has already produced (e.g. from a bitmask widget, optional toggle).
    // Skips the read-widget step; useful where a simple read/widget mapping doesn't fit.
    template<auto... Ms>
    void Set(action::detail::last_field<Ms...> v) const {
        if constexpr (HasEntity) action::Emit(action::UpdateOf<Ms...>(E, std::move(v)));
        else action::Emit(action::UpdateOf<Ms...>(ScopeFromAlt(false), std::move(v)));
    }
};

Edit(entt::registry &) -> Edit<false>;
Edit(entt::registry &, entt::entity) -> Edit<true>;

} // namespace ui
