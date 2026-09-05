#pragma once

// Wrap ImGui field controls in one action gesture per edit.
// Edit{R} targets the active entity and applies Alt-modified edits to the selection.
// Edit{R, E} targets E explicitly.

#include "action/Build.h"
#include "action/Emit.h"
#include "numeric/Angles.h"
#include "scene/Entity.h" // FindActiveEntity
#include "selection/SelectionComponents.h" // Selected

#include <imgui_internal.h> // TempInputIsActive

namespace ui {

// Reserve Alt for selection edits and use Shift for 0.05x drag precision.
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

// Apply Alt-modified drags as per-entity deltas and other Alt-modified edits as copied values.
inline action::Scope ScopeFromAlt(bool delta_capable = false) {
    if (!ImGui::GetIO().KeyAlt) return action::Scope::Active;
    return delta_capable ? action::Scope::SelectedDelta : action::Scope::Selected;
}

namespace detail {
// Preserve gesture state because ImGui permits one active item.
inline action::Scope GestureScope{action::Scope::Active};
inline std::array<std::byte, 16> GestureStartValue{};
inline bool GestureTyped{false};
inline std::function<void()> GestureCancel;

inline bool CompositeGestureOpen{false};
} // namespace detail

// Group a composite editor into one recorded action per drag.
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

template<auto M, auto... Rest>
decltype(auto) ReadChain(auto &obj) {
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

// Map FieldLimits to ImGui bounds, using (0,0) for an unbounded field and FLT_MAX for an open endpoint.
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

template<auto... Ms, typename Field>
bool SliderField(const char *label, Field &value, const char *fmt, ImGuiSliderFlags flags) {
    static_assert(HasMin<Ms...> && HasMax<Ms...>, "SliderField: field must declare FieldLimits with both Min and Max");
    using L = FieldLimits<Ms...>;
    using F = std::remove_cvref_t<Field>;
    if constexpr (std::same_as<F, float>) return ImGui::SliderFloat(label, &value, F(L::Min), F(L::Max), fmt ? fmt : "%.3f", flags);
    else if constexpr (std::same_as<F, vec3>) return ImGui::SliderFloat3(label, &value.x, F(L::Min), F(L::Max), fmt ? fmt : "%.3f", flags);
    else if constexpr (std::same_as<F, double> || std::integral<F>) {
        F lo = F(L::Min), hi = F(L::Max);
        return ImGui::SliderScalar(label, ImGuiDt<F>(), &value, &lo, &hi, fmt, flags);
    } else static_assert(false, "SliderField: unsupported field type");
}

template<typename Component>
struct Replace {
    const Component &Current;
};

struct UpdateFields {};

template<bool HasEntity, typename Policy = UpdateFields, auto... Prefix>
struct Edit {
    entt::registry &R;
    [[no_unique_address]] std::conditional_t<HasEntity, entt::entity, std::monostate> E{};
    [[no_unique_address]] Policy Write{};

    // Return an editor with additional nested field members.
    template<auto... More>
    Edit<HasEntity, Policy, Prefix..., More...> Sub() const {
        return {R, E, Write};
    }

    entt::entity ReadFrom() const {
        if constexpr (HasEntity) return E;
        else return FindActiveEntity(R);
    }

    template<typename T, typename Reg>
    const T &GetConst(Reg &r, entt::entity e) { return r.template get<const T>(e); }

    // Run a widget and group its staged values into one committed action or one cancellation.
    template<auto... Ms, typename Widget>
    bool RunUpdate(Widget widget, bool delta_capable) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        Field v = ReadChain<Prefix..., Ms...>(GetConst<action::detail::first_class<Prefix..., Ms...>>(R, ReadFrom()));
        const Field original = v;
        const bool changed = widget(v);

        if constexpr (!HasEntity) {
            if (ImGui::IsItemHovered() && R.template view<Selected>().size() > 1) ImGui::SetItemTooltip("Hold Alt to apply to all selected");
        }

        // Freeze the scope and initial value for the full gesture.
        auto capture_gesture = [&] {
            detail::GestureTyped = false;
            std::memcpy(detail::GestureStartValue.data(), &original, sizeof(Field));
            if constexpr (!HasEntity) {
                detail::GestureScope = ScopeFromAlt(delta_capable && action::DeltaField<Field>);
                // Clear the transient baseline from an interrupted selection drag.
                if (detail::GestureScope == action::Scope::SelectedDelta) R.template clear<action::DragFieldStart>();
            }
        };

        auto gesture_start = [] { Field s; std::memcpy(&s, detail::GestureStartValue.data(), sizeof(Field)); return s; };

        auto update = [&](const Field &val) {
            if constexpr (HasEntity) return action::UpdateOf<Prefix..., Ms...>(E, val);
            else {
                if constexpr (action::DeltaField<Field>) {
                    // Apply Alt-drag values as offsets from each selected entity's initial value.
                    if (detail::GestureScope == action::Scope::SelectedDelta && !detail::GestureTyped) {
                        return action::UpdateOf<Prefix..., Ms...>(action::Scope::SelectedDelta, Field(val - gesture_start()));
                    }
                }
                // Apply typed Alt-edits as absolute values to the selection.
                const auto scope = detail::GestureScope == action::Scope::SelectedDelta ? action::Scope::Selected : detail::GestureScope;
                return action::UpdateOf<Prefix..., Ms...>(scope, val);
            }
        };

        // Stage only the vector components modified by the widget.
        auto stage = [&] {
            if constexpr (std::same_as<Field, vec2> || std::same_as<Field, vec3> || std::same_as<Field, vec4>) {
                if (delta_capable) {
                    const Field start = gesture_start();
                    const auto w = update(v);
                    const bool delta = w.Scope == action::Scope::SelectedDelta;
                    for (size_t i = 0; i < Field::ComponentCount; ++i) {
                        if (v[i] == start[i]) continue;
                        const uint16_t off = uint16_t(w.Offset + i * sizeof(float));
                        action::EmitStaged(action::Update<float>{w.Scope, w.Entity, w.ComponentType, off, delta ? v[i] - start[i] : v[i]});
                        // Cancel a selection delta by applying a zero offset.
                        if (delta) detail::GestureCancel = [c = w.ComponentType, e = w.Entity, off] { action::EmitCancel(action::Update<float>{action::Scope::SelectedDelta, e, c, off, 0.f}); };
                    }
                    return;
                }
            }
            action::EmitStaged(update(v));
        };

        // Capture the initial update for cancellation.
        if (ImGui::IsItemActivated()) {
            capture_gesture();
            detail::GestureCancel = [revert = update(original)] { action::EmitCancel(revert); };
        }
        // Apply click-to-edit text only when the edit commits.
        if (ImGui::TempInputIsActive(ImGui::GetItemID())) detail::GestureTyped = true;

        if (ImGui::IsItemDeactivatedAfterEdit()) {
            if (changed) stage();
            action::Commit();
            detail::GestureCancel = nullptr;
            return changed;
        }
        if (ImGui::IsItemDeactivated()) {
            if (detail::GestureCancel) {
                detail::GestureCancel();
                detail::GestureCancel = nullptr;
            }
            return false;
        }
        if (changed) {
            // Delay typed edits until commit.
            if (ImGui::IsItemActive() && detail::GestureTyped) return changed;
            // Capture the modifier scope for instantaneous widgets.
            if (!ImGui::IsItemActive()) capture_gesture();
            stage();
            // Commit instantaneous widget changes immediately.
            if (!ImGui::IsItemActive()) action::Commit();
        }
        return changed;
    }

    template<auto... Ms, typename Widget>
    bool RunReplace(Widget widget) const {
        using Component = action::detail::first_class<Prefix..., Ms...>;
        static_assert(HasEntity && std::same_as<Policy, Replace<Component>>);
        auto value = ReadChain<Prefix..., Ms...>(Write.Current);
        const bool changed = widget(value);
        Gesture(changed, [&] {
            auto replacement = Write.Current;
            ReadChain<Prefix..., Ms...>(replacement) = std::move(value);
            return action::Replace<Component>{.Entity = E, .Value = std::move(replacement)};
        });
        return changed;
    }

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool delta_capable = false) {
        if constexpr (std::same_as<Policy, UpdateFields>) return RunUpdate<Ms...>(std::move(widget), delta_capable);
        else return RunReplace<Ms...>(std::move(widget));
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
        return Run<Ms...>([&](auto &value) { return SliderField<Prefix..., Ms...>(label, value, fmt, flags); }, /*delta_capable=*/true);
    }

    // Slider over an angle field stored in radians, displayed in degrees.
    // Bounds come from the field's FieldLimits (radians), which must declare both Min and Max.
    template<auto... Ms>
    bool SliderAngle(const char *label, const char *fmt = "%.0f deg") {
        static_assert(HasMin<Prefix..., Ms...> && HasMax<Prefix..., Ms...>, "Edit::SliderAngle: field must declare FieldLimits with both Min and Max");
        using L = FieldLimits<Prefix..., Ms...>;
        return Run<Ms...>([&](float &v) { return ImGui::SliderAngle(label, &v, numeric::Degrees(float(L::Min)), numeric::Degrees(float(L::Max)), fmt); },
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
    void Set(action::detail::last_field<Prefix..., Ms...> value) const {
        if constexpr (std::same_as<Policy, UpdateFields>) {
            if constexpr (HasEntity) action::Emit(action::UpdateOf<Prefix..., Ms...>(E, std::move(value)));
            else action::Emit(action::UpdateOf<Prefix..., Ms...>(ScopeFromAlt(false), std::move(value)));
        } else {
            using Component = action::detail::first_class<Prefix..., Ms...>;
            static_assert(HasEntity && std::same_as<Policy, Replace<Component>>);
            auto replacement = Write.Current;
            ReadChain<Prefix..., Ms...>(replacement) = std::move(value);
            action::Emit(action::Replace<Component>{.Entity = E, .Value = std::move(replacement)});
        }
    }
};

Edit(entt::registry &) -> Edit<false>;
Edit(entt::registry &, entt::entity) -> Edit<true>;
template<typename Component>
Edit(entt::registry &, entt::entity, Replace<Component>) -> Edit<true, Replace<Component>>;

} // namespace ui
