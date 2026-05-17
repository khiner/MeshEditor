#pragma once

// ui::Edit — terse, single-line wrappers around ImGui controls that read a field, run the
// widget, and dispatch `action::UpdateOf<Ms...>` if the user changed it. Bound to (registry,
// apply, entity) at construction; one instance per UI block.

#include "action/Update.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt/entity/registry.hpp>
#include <imgui.h>

#include <concepts>
#include <type_traits>

namespace ui {

// Walk a chain Ms... from the outermost object down to the leaf field.
template<auto M, auto... Rest>
decltype(auto) ReadChain(const auto &obj) {
    if constexpr (sizeof...(Rest) == 0) return obj.*M;
    else return ReadChain<Rest...>(obj.*M);
}

template<class F>
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

// Wraps a Self* so `ui::Edit f{R, ui::Applier{this}, E}` works without a separate apply lambda.
template<class T>
struct Applier {
    T *Self;
    template<class A> void operator()(A &&a) const { Self->Apply(std::forward<A>(a)); }
};
template<class T> Applier(T *) -> Applier<T>;

template<class Apply, auto... Prefix>
struct Edit {
    entt::registry &R;
    Apply A;
    entt::entity E;

    // Edit for a nested field, so callers don't repeat the outer member on each call.
    template<auto... More> Edit<Apply, Prefix..., More...> Sub() const { return {R, A, E}; }

    // Read field, hand a mutable copy to `widget`, dispatch update if the widget returns true.
    template<auto... Ms, class Widget>
    bool Run(Widget widget) {
        action::detail::last_field<Prefix..., Ms...> v = ReadChain<Prefix..., Ms...>(R.get<const action::detail::first_class<Prefix..., Ms...>>(E));
        if (!widget(v)) return false;

        A(action::UpdateOf<Prefix..., Ms...>(E, v));
        return true;
    }

    template<auto... Ms>
    bool Check(const char *label) {
        return Run<Ms...>([&](bool &v) { return ImGui::Checkbox(label, &v); });
    }

    template<auto... Ms>
    bool Drag(const char *label, float speed = 1.f, float lo = 0.f, float hi = 0.f, const char *fmt = "%.3f") {
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ImGui::DragFloat(label, &v, speed, lo, hi, fmt);
            else if constexpr (std::same_as<F, vec3>) return ImGui::DragFloat3(label, &v.x, speed, lo, hi, fmt);
            else if constexpr (std::same_as<F, vec4>) return ImGui::DragFloat4(label, &v.x, speed, lo, hi, fmt);
            else static_assert(false, "Edit::Drag: field type must be float, vec3, or vec4");
        });
    }

    template<auto... Ms>
    bool Slider(const char *label, auto lo, auto hi, const char *fmt = nullptr) {
        return Run<Ms...>([&](auto &v) {
            using F = std::remove_reference_t<decltype(v)>;
            if constexpr (std::same_as<F, float>) return ImGui::SliderFloat(label, &v, F(lo), F(hi), fmt ? fmt : "%.3f");
            else if constexpr (std::same_as<F, vec3>) return ImGui::SliderFloat3(label, &v.x, F(lo), F(hi), fmt ? fmt : "%.3f");
            else if constexpr (std::integral<F>) {
                F lof = F(lo), hif = F(hi);
                return ImGui::SliderScalar(label, ImGuiDt<F>(), &v, &lof, &hif);
            } else static_assert(false, "Edit::Slider: unsupported field type");
        });
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
    void Set(action::detail::last_field<Ms...> v) const { A(action::UpdateOf<Ms...>(E, std::move(v))); }
};

template<class A>
Edit(entt::registry &, const A &, entt::entity) -> Edit<A>;

} // namespace ui
