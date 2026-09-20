#pragma once

#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

// Wrap ImGui field controls in one action gesture per edit.
// Edit{R} targets the active entity and applies Alt-modified edits to the selection.
// Edit{R, E} targets E explicitly.
// PatchEdit{value} edits fields of a value the caller holds and patches them onto the active entity.
// ValueEdit{value} edits fields of a value the caller holds in place.

#include "Field.h"
#include "Variant.h"
#include "action/Build.h"
#include "action/Dispatch.h"
#include "action/Emit.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "numeric/Angles.h"
#include "scene/Entity.h" // FindActiveEntity
#include "scene/RotationUi.h"
#include "state/Scene.h"

#include <imgui.h>

#include <algorithm>
#include <format>
#include <iterator>
#include <optional>
#include <utility>

namespace ui {
using numeric::Degrees;

// Reserve Alt for selection edits and use Shift for 0.05x drag precision.
inline float DragSpeed(float speed) { return ImGui::GetIO().KeyShift ? speed * 0.05f : speed; }
inline bool DragFloat(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat(label, v, DragSpeed(speed), lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat2(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat2(label, v, DragSpeed(speed), lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat3(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat3(label, v, DragSpeed(speed), lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}
inline bool DragFloat4(const char *label, float *v, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
    return ImGui::DragFloat4(label, v, DragSpeed(speed), lo, hi, fmt, ImGuiSliderFlags_NoSpeedTweaks);
}

// Apply Alt-modified drags as per-entity deltas and other Alt-modified edits as copied values.
action::Target TargetFromAlt(bool delta_capable = false);

// Keying state of a field on its entity, or nothing for a field channels cannot animate there.
std::optional<animation::ChannelState> QueryKey(const state::Scene &, state::Entity, const ChannelTarget &);

namespace detail {
inline bool CompositeGestureOpen{false};

// Commits, cancels, or continues the gesture of the last item, returning the target to stage a change with.
// Returns nothing when the change is not staged.
std::optional<action::Target> FieldGesture(state::Scene &, bool changed, bool selection, bool delta_capable);

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
// Clicking keys or unkeys the field at the current frame, on the active entity as OnActive and on any other entity by identity.
void KeyDecorator(const state::Scene &, state::Entity, const ChannelTarget &, const animation::ChannelState &);
inline void KeyDecorator(const state::Scene &r, state::Entity entity, const ChannelTarget &target) {
    if (const auto state = QueryKey(r, entity, target)) KeyDecorator(r, entity, target, *state);
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
    if (const auto target = FieldGesture(r, changed, selection, delta_capable)) emit(*target, v);
    if (state) KeyDecorator(r, entity, channel, *state);
    return changed;
}
} // namespace detail

// Records the last item's gesture: a drag finishes on release, and a widget that changes without staying active finishes at once.
inline void NoteGesture(bool edited, bool &changed, bool &finished) {
    changed |= edited;
    finished |= ImGui::IsItemDeactivatedAfterEdit() || (edited && !ImGui::IsItemActive());
    if (ImGui::IsItemDeactivated() && !ImGui::IsItemDeactivatedAfterEdit()) action::Cancel();
}

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

template<typename F>
inline constexpr bool DrawableField = std::is_arithmetic_v<F> || std::is_enum_v<F> || action::VectorField<F> || std::same_as<F, quat>;

namespace detail {
inline constexpr const char *FloatFormats[]{"%.0f", "%.1f", "%.2f", "%.3f", "%.4f", "%.5f", "%.6f"};
constexpr const char *Format(const FieldSpec &spec) { return FloatFormats[std::min<size_t>(spec.Digits, std::size(FloatFormats) - 1)]; }
inline std::string InDegrees(const char *label) { return std::format("{} (deg)", label); }
} // namespace detail

// Edits a rotation in the representation the widget last showed: a quaternion, XYZ Euler degrees, or an axis and angle.
inline bool RotationWidget(const char *label, quat &rotation) {
    using namespace ImGui;
    auto *storage = GetStateStorage();
    PushID(label);
    const auto mode_key = GetID("mode"), active_key = GetID("active"), value_key = GetID("value");
    int mode = storage->GetInt(mode_key, 0);
    static constexpr const char *Modes[]{"Quaternion (XYZW)", "XYZ Euler (deg)", "Axis Angle (deg)"};
    if (Combo("##mode", &mode, Modes, IM_ARRAYSIZE(Modes))) storage->SetInt(mode_key, mode);
    SameLine();
    TextUnformatted(label);
    auto ui = ToUiVariant(rotation, size_t(mode));
    // A drag keeps editing the stored numbers, keyed by index, rather than the round trip through the rotation.
    const auto numbers = [&](auto &&f) {
        std::visit([&](auto &v) {
            for (int i = 0; i < int(sizeof(v.Value) / sizeof(float)); ++i) f(value_key + ImGuiID(i), v.Value[i]);
        },
                   ui);
    };
    if (storage->GetBool(active_key)) numbers([&](ImGuiID key, float &x) { x = storage->GetFloat(key); });
    bool changed = false, active = false;
    std::visit(
        overloaded{
            [&](RotationQuat &v) {
                changed = DragFloat4("##value", &v.Value[0], 0.01f);
                active = IsItemActive();
            },
            [&](RotationEuler &v) {
                changed = DragFloat3("##value", &v.Value[0], 1.f);
                active = IsItemActive();
            },
            [&](RotationAxisAngle &v) {
                changed = DragFloat3("##axis", &v.Value[0], 0.01f);
                active = IsItemActive();
                changed |= DragFloat("##angle", &v.Value.w, 1.f);
                active |= IsItemActive();
            },
        },
        ui
    );
    storage->SetBool(active_key, active);
    if (active) numbers([&](ImGuiID key, float x) { storage->SetFloat(key, x); });
    if (changed) rotation = ToRotation(ui);
    PopID();
    return changed;
}

// An enum without enumerators drags its underlying integer, and a radian field drags in degrees.
template<typename F>
bool DrawField(const char *label, F &v, const FieldSpec &spec) {
    if constexpr (std::same_as<F, bool>) {
        return ImGui::Checkbox(label, &v);
    } else if constexpr (std::is_enum_v<F>) {
        constexpr auto &values = field::EnumValues<F>;
        constexpr auto &names = field::EnumLabels<F>;
        if constexpr (values.empty()) {
            auto underlying = std::to_underlying(v);
            if (!DrawField(label, underlying, spec)) return false;
            v = F(underlying);
            return true;
        } else {
            int i = int(std::ranges::find(values, v) - values.begin());
            if (!ImGui::Combo(label, &i, names.data(), int(names.size()))) return false;
            v = values[size_t(i)];
            return true;
        }
    } else if constexpr (std::same_as<F, quat>) {
        return RotationWidget(label, v);
    } else if constexpr (action::ScalarField<F> || action::VectorField<F>) {
        using C = action::Limit<F>;
        C lo = action::LowerBound<F>(spec), hi = action::UpperBound<F>(spec);
        const char *format = std::floating_point<C> ? detail::Format(spec) : nullptr;
        if constexpr (std::same_as<F, float>) {
            if (spec.Unit == FieldUnit::Radians) {
                float degrees = Degrees(v);
                lo = Degrees(lo);
                hi = Degrees(hi);
                if (!ImGui::DragFloat(detail::InDegrees(label).c_str(), &degrees, DragSpeed(spec.Speed), lo, hi, format, ImGuiSliderFlags_NoSpeedTweaks | ImGuiSliderFlags_AlwaysClamp)) return false;
                v = numeric::Radians(degrees);
                return true;
            }
        }
        return ImGui::DragScalarN(label, ImGuiDt<C>(), &v, action::Components<F>, DragSpeed(spec.Speed), spec.Bounded() ? &lo : nullptr, spec.Bounded() ? &hi : nullptr, format, ImGuiSliderFlags_NoSpeedTweaks | ImGuiSliderFlags_AlwaysClamp);
    } else {
        static_assert(false, "DrawField: unsupported field type");
    }
}

// Widgets over an editor's Run<Ms...>(widget, delta_capable), which reads the field, runs the widget, and stages a change.
// A label defaults to the field's name, and bounds, speed, and format come from the field's spec.
template<typename Editor, auto... Prefix>
struct FieldWidgets {
    template<auto... Ms> using Field = action::detail::last_field<Prefix..., Ms...>;
    template<auto... Ms> static const char *LabelOr(const char *label) { return label ? label : field::LabelOf<action::detail::last_v<Prefix..., Ms...>>.c_str(); }
    template<auto... Ms> static consteval FieldSpec SpecOf() { return field::ChainSpec<Prefix..., Ms...>(); }

    template<auto... Ms>
    bool Draw(const char *label = nullptr) {
        return Self().template Run<Ms...>([&](auto &v) { return DrawField(LabelOr<Ms...>(label), v, SpecOf<Ms...>()); }, /*delta_capable=*/true);
    }

    template<auto... Ms>
    bool Check(const char *label = nullptr) {
        static_assert(std::same_as<Field<Ms...>, bool>, "Edit::Check: field must be a bool");
        return Draw<Ms...>(label);
    }

    template<auto... Ms>
    bool Drag(const char *label = nullptr) {
        static_assert(action::DeltaField<Field<Ms...>>, "Edit::Drag: field must be numeric");
        return Draw<Ms...>(label);
    }

    // A null format takes the spec's, and a radian field slides in degrees.
    template<auto... Ms>
    bool Slider(const char *label = nullptr, const char *fmt = nullptr, ImGuiSliderFlags flags = 0) {
        constexpr auto spec = SpecOf<Ms...>();
        static_assert(spec.HasMin() && spec.HasMax(), "Edit::Slider: field must declare a spec with both Min and Max");
        using F = Field<Ms...>;
        using C = action::Limit<F>;
        const char *format = fmt ? fmt : std::floating_point<C> ? detail::Format(spec) :
                                                                  nullptr;
        return Self().template Run<Ms...>([&](F &v) {
            if constexpr (std::same_as<F, float> && spec.Unit == FieldUnit::Radians) {
                return ImGui::SliderAngle(detail::InDegrees(LabelOr<Ms...>(label)).c_str(), &v, Degrees(float(spec.Min)), Degrees(float(spec.Max)), format, flags);
            } else {
                const C lo = C(spec.Min), hi = C(spec.Max);
                return ImGui::SliderScalarN(LabelOr<Ms...>(label), ImGuiDt<C>(), &v, action::Components<F>, &lo, &hi, format, flags);
            }
        },
                                          /*delta_capable=*/true);
    }

    template<auto... Ms>
    bool Slider(const char *label, float lo, float hi, const char *fmt = "%.3f") {
        return Self().template Run<Ms...>([&](float &v) { return ImGui::SliderFloat(LabelOr<Ms...>(label), &v, lo, hi, fmt); }, /*delta_capable=*/true);
    }

    // ColorEdit3 for vec3, ColorEdit4 for vec4, picked by field type.
    template<auto... Ms>
    bool Color(const char *label = nullptr) {
        return Self().template Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, vec3>) return ImGui::ColorEdit3(LabelOr<Ms...>(label), &v.x);
            else if constexpr (std::same_as<F, vec4>) return ImGui::ColorEdit4(LabelOr<Ms...>(label), &v.x);
            else static_assert(false, "Edit::Color: field must be vec3 or vec4");
        });
    }

    template<auto... Ms>
    bool Enum(const char *label = nullptr) {
        static_assert(std::is_enum_v<Field<Ms...>>, "Edit::Enum: field must be an enum");
        return Draw<Ms...>(label);
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

    template<typename C> state::Entity ReadFrom() const {
        if constexpr (HasEntity) return E;
        else return action::UpdateTraits<C>::Active(R);
    }
    // The target an edit records: the gesture's target for an active-entity editor, otherwise the entity, or OnViewport for the viewport.
    action::Target Recorded(action::Target gesture) const {
        if constexpr (!HasEntity) return gesture;
        else return E == animation::AnimationsViewport(R) ? action::Target{action::OnViewport{}} : action::Target{E};
    }

    // Run a widget over the field and stage its change in the item's gesture.
    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool delta_capable = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        using C = action::detail::first_class<Prefix..., Ms...>;
        const auto entity = ReadFrom<C>();
        return detail::RunField(
            R, entity, detail::Channel<Prefix..., Ms...>(), ReadChain<Prefix..., Ms...>(R.template get<const C>(entity)), widget,
            !HasEntity, delta_capable && action::DeltaField<Field>,
            [&](action::Target gesture, const Field &v) { action::Emit(action::UpdateOf<Prefix..., Ms...>(Recorded(gesture), v), action::Phase::Stage); }
        );
    }

    // Write a value the caller has already produced (e.g. from a bitmask widget, optional toggle).
    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const {
        action::Emit(action::UpdateOf<Prefix..., Ms...>(Recorded(TargetFromAlt(false)), std::move(value)));
    }
};

Edit(state::Scene &) -> Edit<false>;
Edit(state::Scene &, state::Entity) -> Edit<true>;

// Edits fields of `Current`, a value the caller holds, patching each change onto the active entity.
template<typename Component, auto... Prefix>
struct PatchEdit : FieldWidgets<PatchEdit<Component, Prefix...>, Prefix...> {
    explicit PatchEdit(const Component &current) : Current{current} {}

    const Component &Current;

    template<auto... More>
    PatchEdit<Component, Prefix..., More...> Sub() const { return PatchEdit<Component, Prefix..., More...>{Current}; }

    template<auto... Ms>
    auto Action(action::detail::last_field<Prefix..., Ms...> value) const {
        using F = action::detail::last_field<Prefix..., Ms...>;
        return action::PatchFields<Component, F>{{action::detail::FieldOffset<Prefix..., Ms...>()}, {std::move(value)}};
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

// Edits fields of `Value` in place, accumulating whether any changed, and whether a gesture finished when `finished` is given.
template<typename T, auto... Prefix>
struct ValueEdit : FieldWidgets<ValueEdit<T, Prefix...>, Prefix...> {
    ValueEdit(T &value, bool &changed, bool *finished = nullptr) : Value{value}, Changed{changed}, Finished{finished} {}

    T &Value;
    bool &Changed;
    bool *Finished;

    template<auto... More>
    ValueEdit<T, Prefix..., More...> Sub() const { return {Value, Changed, Finished}; }

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool = false) {
        const bool changed = widget(ReadChain<Prefix..., Ms...>(Value));
        if (Finished) NoteGesture(changed, Changed, *Finished);
        else Changed |= changed;
        return changed;
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) {
        ReadChain<Prefix..., Ms...>(Value) = std::move(value);
        Changed = true;
    }
};
template<typename T> ValueEdit(T &, bool &, bool * = nullptr) -> ValueEdit<T>;

// A type with an editor of its own draws through `DrawEditor(editor, std::type_identity<T>{})` over any editor.
template<typename T>
concept HasEditor = requires(ValueEdit<T> &e) { DrawEditor(e, std::type_identity<T>{}); };
} // namespace ui
