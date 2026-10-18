#pragma once

// Wrap ImGui field controls in one action gesture per edit.
// Edit{R} targets the active entity and applies Alt-modified edits to the selection.
// Edit{R, E} targets E explicitly.
// PatchEdit{value} patches edits onto the active entity.
// ValueEdit{value} edits the supplied value in place.
// DrawFields<C>(editor) draws every field of C through an editor from its Specs table.

#include "action/Build.h"
#include "action/Dispatch.h"
#include "action/Emit.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "animation/Keying.h"
#include "numeric/Angles.h"
#include "scene/RotationUi.h"

#include <imgui.h>

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
std::optional<animation::ChannelState> QueryKey(const state::Registry &, state::Entity, const ChannelTarget &);

// The hovered field or item. Every pointer refers to static storage.
struct FieldInfo {
    const char *Label{}, *Path{}, *Type{}, *Help{}, *Status{};
    const FieldSpec *Spec{};
    bool Selection{}; // Alt applies the edit to every selected entity
    std::optional<animation::ChannelState> Key{};
};
inline FieldInfo Hovered{}; // The field or described item under the mouse
inline void Describe(const char *label, const char *help) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) Hovered = {.Label = label, .Help = help};
}

namespace detail {
inline bool CompositeGestureOpen{false};
// The keying state and selection scope of the field being run.
inline std::optional<animation::ChannelState> RunKey{};
inline bool RunSelection{false};

// Commits, cancels, or continues the gesture of the last item, returning the target to stage a change with.
// Returns nothing when the change is not staged.
std::optional<action::Target> FieldGesture(bool changed, bool selection, bool delta_capable);

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
void KeyDecorator(const state::Registry &, state::Entity, const ChannelTarget &, const animation::ChannelState &);
inline void KeyDecorator(const state::Registry &r, state::Entity entity, const ChannelTarget &target) {
    if (const auto state = QueryKey(r, entity, target)) KeyDecorator(r, entity, target, *state);
}

namespace detail {
// Runs `widget` over `v` and stages its change in the item's gesture through `emit(scope, v)`.
// A field with a channel target is tinted by its keying state on `entity` and followed by its keying decorator.
bool RunField(state::Registry &r, state::Entity entity, const ChannelTarget &channel, auto v, auto widget, bool selection, bool delta_capable, auto emit) {
    const auto state = channel.Count ? QueryKey(r, entity, channel) : std::nullopt;
    RunKey = state;
    RunSelection = selection;
    const bool changed = [&] {
        const KeyTint tint{state};
        return widget(v);
    }();
    if (const auto target = FieldGesture(changed, selection, delta_capable)) emit(*target, v);
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
void Gesture(bool changed, auto &&make) {
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
    if constexpr (sizeof...(Rest) == 0) return field::Get<M>(obj);
    else return ReadChain<Rest...>(field::Get<M>(obj));
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

// The number conversion, from the spec's Format or its Digits, followed by its unit symbol, so a 3-digit meter value prints as "%.3f m".
template<typename C> std::array<char, 24> NumberFormat(const FieldSpec &spec) {
    std::array<char, 24> out{};
    const auto *symbol = UnitSymbol(spec.Unit).data();
    if (spec.Format) snprintf(out.data(), out.size(), "%s%s", spec.Format, symbol);
    else if constexpr (std::floating_point<C>) snprintf(out.data(), out.size(), "%%.%uf%s", unsigned(spec.Digits.value_or(UnitDigits(spec.Unit))), symbol);
    else snprintf(out.data(), out.size(), "%s%s", std::signed_integral<C> ? "%d" : "%u", symbol);
    return out;
}

template<typename T> struct OptionalTraits {
    static constexpr bool Is = false;
    using Value = T;
};
template<typename T> struct OptionalTraits<std::optional<T>> {
    static constexpr bool Is = true;
    using Value = T;
};

template<typename T> T InitialValue(const FieldSpec &spec) {
    if constexpr (action::ScalarField<T> || action::VectorField<T>) {
        using L = action::Limit<T>;
        if (!std::isnan(spec.Initial)) return T(L(spec.Initial));
        return T(std::clamp(L{}, action::LowerBound<T>(spec), action::UpperBound<T>(spec)));
    } else {
        return T{};
    }
}

template<typename F>
inline constexpr bool DrawableField = std::is_arithmetic_v<F> || std::is_enum_v<F> || action::VectorField<F> || std::same_as<F, quat>;
template<typename T> inline constexpr bool DrawableField<std::optional<T>> = DrawableField<T>;

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

// An enum without enumerators edits as its underlying integer, a radian field in degrees, and an optional field behind a presence checkbox.
template<typename F>
bool DrawField(const char *label, F &v, const FieldSpec &spec) {
    if constexpr (OptionalTraits<F>::Is) {
        ImGui::PushID(label);
        bool present = v.has_value();
        bool changed = ImGui::Checkbox(present ? "##present" : label, &present);
        if (changed) {
            if (present) v = InitialValue<typename F::value_type>(spec);
            else v.reset();
        }
        if (v) {
            ImGui::SameLine();
            changed |= DrawField(label, *v, spec);
        }
        ImGui::PopID();
        return changed;
    } else if constexpr (std::same_as<F, bool>) {
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
            auto i = int(std::ranges::find(values, v) - values.begin());
            if (!ImGui::Combo(label, &i, names.data(), int(names.size()))) return false;
            v = values[size_t(i)];
            return true;
        }
    } else if constexpr (std::same_as<F, quat>) {
        ImGui::BeginGroup();
        const bool changed = RotationWidget(label, v);
        ImGui::EndGroup();
        return changed;
    } else if constexpr (action::ScalarField<F> || action::VectorField<F>) {
        using C = action::Limit<F>;
        if constexpr (std::integral<F>) {
            if (spec.Widget == FieldWidget::Checkbox) {
                bool on = v != 0;
                if (!ImGui::Checkbox(label, &on)) return false;
                v = on ? 1 : 0;
                return true;
            }
        }
        if constexpr (std::same_as<F, vec3> || std::same_as<F, vec4>) {
            if (spec.Widget == FieldWidget::Color) {
                if constexpr (std::same_as<F, vec3>) return ImGui::ColorEdit3(label, &v.x);
                else return ImGui::ColorEdit4(label, &v.x);
            }
        }
        C lo = action::LowerBound<F>(spec), hi = action::UpperBound<F>(spec);
        const bool radians = spec.Unit == FieldUnit::Radians;
        const auto number_format = NumberFormat<C>(spec);
        const char *format = number_format.data();
        const bool slider = spec.Widget == FieldWidget::Slider || (spec.Widget == FieldWidget::Auto && spec.HasMin() && spec.HasMax());
        const ImGuiSliderFlags scale = spec.Scale == FieldScale::Logarithmic ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None;
        if constexpr (std::same_as<F, float>) {
            if (radians) {
                lo = Degrees(lo);
                hi = Degrees(hi);
                if (slider) return ImGui::SliderAngle(label, &v, lo, hi, format, scale | ImGuiSliderFlags_AlwaysClamp);
                float degrees = Degrees(v);
                if (!ImGui::DragFloat(label, &degrees, DragSpeed(spec.Speed), lo, hi, format, scale | ImGuiSliderFlags_NoSpeedTweaks | ImGuiSliderFlags_AlwaysClamp)) return false;
                v = numeric::Radians(degrees);
                return true;
            }
        }
        if (slider) return ImGui::SliderScalarN(label, ImGuiDt<C>(), &v, action::Components<F>, &lo, &hi, format, scale | ImGuiSliderFlags_AlwaysClamp);
        return ImGui::DragScalarN(label, ImGuiDt<C>(), &v, action::Components<F>, DragSpeed(spec.Speed), spec.Bounded() ? &lo : nullptr, spec.Bounded() ? &hi : nullptr, format, scale | ImGuiSliderFlags_NoSpeedTweaks | ImGuiSliderFlags_AlwaysClamp);
    } else {
        static_assert(false, "DrawField: unsupported field type");
    }
}

// Widgets over an editor's Run<Ms...>(widget, delta_capable), which reads the field, runs the widget, and stages a change.
template<auto... Prefix>
struct FieldWidgets {
    template<auto... Ms> using Field = action::detail::last_field<Prefix..., Ms...>;

    template<auto... Ms>
    bool Draw(this auto &&self, const char *label = nullptr) {
        using F = Field<Ms...>;
        constexpr auto &spec = field::ChainSpecOf<Prefix..., Ms...>;
        static_assert(spec.Widget != FieldWidget::Color || std::same_as<F, vec3> || std::same_as<F, vec4>, "Edit::Draw: a color field must be a vec3 or vec4");
        static_assert(spec.Widget != FieldWidget::Slider || (spec.HasMin() && spec.HasMax()), "Edit::Draw: a slider field must declare both Min and Max");
        return self.template Draw<Ms...>([&](const char *chain_label, F &v) { return DrawField(label ? label : chain_label, v, spec); });
    }

    template<auto... Ms, typename Widget>
        requires std::invocable<Widget &, const char *, Field<Ms...> &>
    bool Draw(this auto &&self, Widget widget) {
        using F = Field<Ms...>;
        constexpr auto &spec = field::ChainSpecOf<Prefix..., Ms...>;
        detail::RunKey.reset();
        detail::RunSelection = false;
        return self.template Run<Ms...>([&](F &v) {
            const bool changed = widget(field::ChainLabel<Prefix..., Ms...>.c_str(), v);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                Hovered = {
                    .Label = field::ChainLabel<Prefix..., Ms...>.c_str(),
                    .Path = field::ChainPath<Prefix..., Ms...>.c_str(),
                    .Type = field::TypeName<typename OptionalTraits<F>::Value>.c_str(),
                    .Help = spec.Help,
                    .Spec = &spec,
                    .Selection = detail::RunSelection,
                    .Key = detail::RunKey,
                };
                if constexpr (requires { self.Status; }) Hovered.Status = self.Status;
            }
            return changed;
        },
                                        /*delta_capable=*/true);
    }

    template<auto M>
    void Section() {
        ImGui::SeparatorText(field::ChainLabel<Prefix..., M>.c_str());
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            constexpr auto &spec = field::ChainSpecOf<Prefix..., M>;
            Hovered = {.Label = field::ChainLabel<Prefix..., M>.c_str(), .Path = field::ChainPath<Prefix..., M>.c_str(), .Help = spec.Help};
        }
    }
};

// Edits component fields on an entity through Update actions.
template<bool HasEntity, auto... Prefix>
struct Edit : FieldWidgets<Prefix...> {
    using Target = std::conditional_t<HasEntity, state::Entity, std::monostate>;
    Edit(state::Registry &r, Target e = {}) : R{r}, E{e} {}

    state::Registry &R;
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

    template<auto... Ms>
    auto Read() const {
        using C = action::detail::first_class<Prefix..., Ms...>;
        return ReadChain<Prefix..., Ms...>(R.template get<const C>(ReadFrom<C>()));
    }

    // Run a widget over the field and stage its change in the item's gesture.
    template<auto... Ms>
    bool Run(auto widget, bool delta_capable = false) {
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

Edit(state::Registry &) -> Edit<false>;
Edit(state::Registry &, state::Entity) -> Edit<true>;

template<typename Component, auto... Prefix>
struct PatchEdit : FieldWidgets<Prefix...> {
    explicit PatchEdit(const Component &current) : Current{current} {}

    const Component &Current;

    template<auto... More>
    PatchEdit<Component, Prefix..., More...> Sub() const { return PatchEdit<Component, Prefix..., More...>{Current}; }

    template<auto... Ms>
    auto Action(action::detail::last_field<Prefix..., Ms...> value) const {
        using F = action::detail::last_field<Prefix..., Ms...>;
        return action::PatchFields<Component, F>{{action::detail::FieldOffset<Prefix..., Ms...>()}, {std::move(value)}};
    }

    template<auto... Ms>
    auto Read() const { return ReadChain<Prefix..., Ms...>(Current); }

    template<auto... Ms>
    bool Run(auto widget, bool = false) {
        auto value = Read<Ms...>();
        const bool changed = widget(value);
        Gesture(changed, [&] { return Action<Ms...>(std::move(value)); });
        return changed;
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const { action::Emit(Action<Ms...>(std::move(value))); }
};

// Edits fields of `Value` in place, accumulating whether any changed, and whether a gesture finished when `finished` is given.
template<typename T, auto... Prefix>
struct ValueEdit : FieldWidgets<Prefix...> {
    ValueEdit(T &value, bool &changed, bool *finished = nullptr) : Value{value}, Changed{changed}, Finished{finished} {}

    T &Value;
    bool &Changed;
    bool *Finished;

    template<auto... More>
    ValueEdit<T, Prefix..., More...> Sub() const { return {Value, Changed, Finished}; }

    template<auto... Ms>
    auto Read() const { return ReadChain<Prefix..., Ms...>(Value); }

    template<auto... Ms>
    bool Run(auto widget, bool = false) {
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

// A type with its own editor draws through `DrawEditor(editor, std::type_identity<T>{})`, declared in the type's namespace.
template<typename T>
concept HasEditor = requires(ValueEdit<T> &e) { DrawEditor(e, std::type_identity<T>{}); };

// Draws every field of C except the Skip members through the editor, in declaration order.
template<typename C, auto... Skip> void DrawFields(auto &&e);

namespace detail {
template<typename C, size_t I, auto... Skip>
void DrawMember(auto &e) {
    if constexpr (!(... || (field::IndexOf<Skip> == I))) {
        constexpr field::At<C, I> member{};
        using F = field::At<C, I>::Field;
        if constexpr (requires { e.template Leaf<member>(); }) {
            e.template Leaf<member>();
        } else if constexpr (requires(decltype(e.template Sub<member>()) &sub) { DrawEditor(sub, std::type_identity<F>{}); }) {
            ImGui::PushID(int(I));
            auto sub = e.template Sub<member>();
            DrawEditor(sub, std::type_identity<F>{});
            ImGui::PopID();
        } else if constexpr (field::Walkable<F>) {
            e.template Section<member>();
            ImGui::PushID(int(I));
            DrawFields<F>(e.template Sub<member>());
            ImGui::PopID();
        } else if constexpr (DrawableField<F>) {
            e.template Draw<member>();
        } else {
            static_assert(false, "DrawFields: this field has no widget, give the editor a Leaf for it or skip it");
        }
    }
}
} // namespace detail

template<typename C, auto... Skip> void DrawFields(auto &&e) {
    [&]<size_t... Is>(std::index_sequence<Is...>) { (..., detail::DrawMember<C, Is, Skip...>(e)); }(std::make_index_sequence<field::detail::Count<C>()>{});
}
} // namespace ui
