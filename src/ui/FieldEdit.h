#pragma once

// Wrap ImGui field controls in one action gesture per edit.
// Edit{R} targets the active entity and applies Alt-modified edits to the selection.
// Edit{R, E} targets E explicitly.
// PatchEdit{E, value} edits fields of a value the caller holds and patches them onto E.
// ValueEdit{value} edits fields of a value the caller holds in place.

#include "action/Build.h"
#include "action/Emit.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "numeric/Angles.h"
#include "scene/Entity.h" // FindActiveEntity
#include "state/Scene.h"

#include <imgui.h>
#include <optional>

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
action::Scope ScopeFromAlt(bool delta_capable = false);

// Keying state of a field on its entity, or nothing for a field channels cannot animate there.
std::optional<animation::ChannelState> QueryKey(const state::Scene &, state::Entity, const ChannelTarget &);

namespace detail {
inline bool CompositeGestureOpen{false};

// Commits, cancels, or continues the gesture of the last item, returning the scope to stage a change with.
// Returns nothing when the change is not staged.
std::optional<action::Scope> FieldGesture(state::Scene &, bool changed, bool selection, bool delta_capable);

// The channel target of the field Ms... walks to, or a zero-count target for a field channels cannot animate.
template<auto... Ms>
ChannelTarget Channel(uint16_t index = 0) {
    if constexpr (animation::KeyableField<action::detail::last_field<Ms...>>) {
        if (const auto target = animation::Target<Ms...>(index); animation::IsChannelStore(target.Component)) return target;
    }
    return {};
}
} // namespace detail

// Tints the widgets drawn while alive by the field's channel state: green animated, yellow keyed on this frame, orange changed.
struct KeyTint {
    explicit KeyTint(const std::optional<animation::ChannelState> &);
    ~KeyTint();
    int Pushed{0};
};

// Draws the keying decorator beside the last item: a dot for no channel, a diamond for a channel, filled for a key on this frame.
// Clicking keys or unkeys the field at the current frame.
void KeyDecorator(state::Entity, const ChannelTarget &, const animation::ChannelState &);
inline void KeyDecorator(const state::Scene &r, state::Entity entity, const ChannelTarget &target) {
    if (const auto state = QueryKey(r, entity, target)) KeyDecorator(entity, target, *state);
}

namespace detail {
// Runs `widget` over `v` and stages its change in the item's gesture through `emit(scope, v)`.
// A field with a channel target is tinted by its keying state on `entity` and followed by its keying decorator.
template<typename Field, typename Widget, typename Emit>
bool RunField(state::Scene &r, state::Entity entity, const ChannelTarget &channel, Field v, Widget widget, bool selection, bool delta_capable, Emit emit) {
    const auto state = channel.Count ? QueryKey(r, entity, channel) : std::nullopt;
    const bool changed = [&] {
        const KeyTint tint{state};
        return widget(v);
    }();
    if (const auto scope = FieldGesture(r, changed, selection, delta_capable)) emit(*scope, v);
    if (state) KeyDecorator(entity, channel, *state);
    return changed;
}
} // namespace detail

// Group a composite editor into one recorded action per drag.
template<typename MakeAction>
void Gesture(bool changed, MakeAction &&make) {
    if (changed) {
        action::Emit(make(), action::Phase::Stage);
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

// Widgets over an editor's Run<Ms...>(widget, delta_capable), which reads the field, runs the widget, and stages a change.
template<typename Editor, auto... Prefix>
struct FieldWidgets {
    template<auto... Ms>
    bool Check(const char *label) {
        return Self().template Run<Ms...>([&](bool &v) { return ImGui::Checkbox(label, &v); });
    }

    // Drag bounds come from the field's FieldLimits (none → unbounded).
    template<auto... Ms>
    bool Drag(const char *label, float speed = 1.f, const char *fmt = "%.3f") {
        constexpr auto bounds = DragBounds<Prefix..., Ms...>();
        return Self().template Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ui::DragFloat(label, &v, speed, bounds.first, bounds.second, fmt);
            else if constexpr (std::same_as<F, vec2>) return ui::DragFloat2(label, &v.x, speed, bounds.first, bounds.second, fmt);
            else if constexpr (std::same_as<F, vec3>) return ui::DragFloat3(label, &v.x, speed, bounds.first, bounds.second, fmt);
            else if constexpr (std::same_as<F, vec4>) return ui::DragFloat4(label, &v.x, speed, bounds.first, bounds.second, fmt);
            else static_assert(false, "Edit::Drag: field type must be float or a float vector");
        },
                                          /*delta_capable=*/true);
    }

    // Slider bounds come from the field's FieldLimits, which must declare both Min and Max.
    template<auto... Ms>
    bool Slider(const char *label, const char *fmt = nullptr, ImGuiSliderFlags flags = 0) {
        return Self().template Run<Ms...>([&](auto &value) { return SliderField<Prefix..., Ms...>(label, value, fmt, flags); }, /*delta_capable=*/true);
    }

    // Slider over a float field with explicit bounds.
    template<auto... Ms>
    bool Slider(const char *label, float lo, float hi, const char *fmt = "%.3f") {
        return Self().template Run<Ms...>([&](float &v) { return ImGui::SliderFloat(label, &v, lo, hi, fmt); }, /*delta_capable=*/true);
    }

    // Slider over an angle field stored in radians, displayed in degrees.
    // Bounds come from the field's FieldLimits (radians), which must declare both Min and Max.
    template<auto... Ms>
    bool SliderAngle(const char *label, const char *fmt = "%.0f deg") {
        static_assert(HasMin<Prefix..., Ms...> && HasMax<Prefix..., Ms...>, "Edit::SliderAngle: field must declare FieldLimits with both Min and Max");
        using L = FieldLimits<Prefix..., Ms...>;
        return Self().template Run<Ms...>([&](float &v) { return ImGui::SliderAngle(label, &v, numeric::Degrees(float(L::Min)), numeric::Degrees(float(L::Max)), fmt); },
                                          /*delta_capable=*/true);
    }

    // ColorEdit3 for vec3, ColorEdit4 for vec4, picked by field type.
    template<auto... Ms>
    bool Color(const char *label) {
        return Self().template Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, vec3>) return ImGui::ColorEdit3(label, &v.x);
            else if constexpr (std::same_as<F, vec4>) return ImGui::ColorEdit4(label, &v.x);
            else static_assert(false, "Edit::Color: field must be vec3 or vec4");
        });
    }

    // Combo over a contiguous enum represented by a packed C-string ("A\0B\0C\0").
    template<auto... Ms>
    bool Enum(const char *label, const char *items) {
        return Self().template Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            static_assert(std::is_enum_v<F>, "Edit::Enum: field must be an enum");
            int i = int(v);
            if (!ImGui::Combo(label, &i, items)) return false;
            v = F(i);
            return true;
        });
    }

private:
    Editor &Self() { return static_cast<Editor &>(*this); }
};

// Edits component fields on an entity through Update actions.
template<bool HasEntity, auto... Prefix>
struct Edit : FieldWidgets<Edit<HasEntity, Prefix...>, Prefix...> {
    using Target = std::conditional_t<HasEntity, state::Entity, std::monostate>;
    Edit(state::Scene &r, Target e = {}) : R{r}, E{e} {}

    state::Scene &R;
    [[no_unique_address]] Target E;

    // Return an editor with additional nested field members.
    template<auto... More>
    Edit<HasEntity, Prefix..., More...> Sub() const { return {R, E}; }

    state::Entity ReadFrom() const {
        if constexpr (HasEntity) return E;
        else return FindActiveEntity(R);
    }

    // Run a widget over the field and stage its change in the item's gesture.
    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool delta_capable = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        using C = action::detail::first_class<Prefix..., Ms...>;
        const auto target = ReadFrom();
        return detail::RunField(
            R, target, detail::Channel<Prefix..., Ms...>(), ReadChain<Prefix..., Ms...>(R.template get<const C>(target)), widget,
            !HasEntity, delta_capable && action::DeltaField<Field>, [&](action::Scope scope, const Field &v) {
                if constexpr (HasEntity) action::Emit(action::UpdateOn<Prefix..., Ms...>(E, v), action::Phase::Stage);
                else action::Emit(action::UpdateOf<Prefix..., Ms...>(scope, v), action::Phase::Stage);
            }
        );
    }

    // Write a value the caller has already produced (e.g. from a bitmask widget, optional toggle).
    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const {
        if constexpr (HasEntity) action::Emit(action::UpdateOn<Prefix..., Ms...>(E, std::move(value)));
        else action::Emit(action::UpdateOf<Prefix..., Ms...>(ScopeFromAlt(false), std::move(value)));
    }
};

Edit(state::Scene &) -> Edit<false>;
Edit(state::Scene &, state::Entity) -> Edit<true>;

// Edits fields of `Current`, a value the caller holds, patching each change onto E.
// The component is created from defaults when E lacks it.
template<typename Component, auto... Prefix>
struct PatchEdit : FieldWidgets<PatchEdit<Component, Prefix...>, Prefix...> {
    PatchEdit(state::Entity e, const Component &current) : E{e}, Current{current} {}

    state::Entity E;
    const Component &Current;

    template<auto... More>
    PatchEdit<Component, Prefix..., More...> Sub() const { return {E, Current}; }

    template<auto... Ms>
    auto Action(action::detail::last_field<Prefix..., Ms...> value) const {
        using F = action::detail::last_field<Prefix..., Ms...>;
        return action::PatchFields<Component, F>{E, {action::detail::FieldOffset<Prefix..., Ms...>()}, {std::move(value)}};
    }

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool = false) {
        auto value = ReadChain<Prefix..., Ms...>(Current);
        const bool changed = widget(value);
        Gesture(changed, [&] { return Action<Ms...>(std::move(value)); });
        return changed;
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const { action::Emit(Action<Ms...>(std::move(value))); }
};

// Edits fields of `Value` in place, accumulating whether any changed.
template<typename T, auto... Prefix>
struct ValueEdit : FieldWidgets<ValueEdit<T, Prefix...>, Prefix...> {
    ValueEdit(T &value, bool &changed) : Value{value}, Changed{changed} {}

    T &Value;
    bool &Changed;

    template<auto... More>
    ValueEdit<T, Prefix..., More...> Sub() const { return {Value, Changed}; }

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool = false) {
        const bool changed = widget(ReadChain<Prefix..., Ms...>(Value));
        Changed |= changed;
        return changed;
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) {
        ReadChain<Prefix..., Ms...>(Value) = std::move(value);
        Changed = true;
    }
};
template<typename T> ValueEdit(T &, bool &) -> ValueEdit<T>;
} // namespace ui
