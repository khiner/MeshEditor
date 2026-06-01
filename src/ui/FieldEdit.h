#pragma once

// ui::Edit — terse, single-line wrappers around ImGui controls that read a field, run the widget,
// and feed any change into action gesture grouping: a drag stages each frame and commits one
// `action::UpdateOf<Ms...>` on release; an instantaneous edit stages and commits in a single frame.
//
// Bound to a registry, optionally an entity:
//   `ui::Edit{R}`     — reads/writes target the registry's active entity. Emits UpdateActive<T>
//                       (reads resolve via FindActiveEntity(R) at Run time).
//   `ui::Edit{R, E}`  — reads from E and writes carry E. Emits Update<T>.
//
// `HasEntity` is a template bool (deduced via the constructor) so only the relevant code path
// instantiates — adding a `ui::Edit{R}` site doesn't force `UpdateActive<T>` for every T
// the entity-bound form already uses.

#include "action/Build.h"
#include "action/Emit.h"
#include "scene/Entity.h" // FindActiveEntity

#include <imgui.h>

#include <functional>

namespace ui {

namespace detail {
// Reverts the one in-progress widget gesture to its start value (ImGui has a single active item, so
// one slot suffices). Set when a widget activates, consumed on an aborted edit (Escape / no-op release).
inline std::function<void()> GestureCancel;
// A composite-editor gesture (several sub-widgets, one action) is staging an uncommitted value.
inline bool CompositeGestureOpen = false;
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

    template<class T, class Reg>
    const T &GetConst(Reg &r, entt::entity e) { return r.template get<const T>(e); }

    // Read field, hand a mutable copy to `widget`, and feed the result into gesture grouping: a drag
    // stages each frame and commits one record on release; an instantaneous edit stages+commits in one
    // frame; an aborted edit (Escape) reverts to the gesture's start value.
    template<auto... Ms, class Widget>
    bool Run(Widget widget) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        Field v = ReadChain<Prefix..., Ms...>(GetConst<action::detail::first_class<Prefix..., Ms...>>(R, ReadFrom()));
        const Field original = v;
        const bool changed = widget(v);

        // Build the field's update — entity-bound (Update) or active (UpdateActive).
        auto update = [&](const Field &val) {
            if constexpr (HasEntity) return action::UpdateOf<Prefix..., Ms...>(E, val);
            else return action::UpdateOf<Prefix..., Ms...>(val);
        };

        // Capture the gesture's start update so an aborted edit can revert.
        if (ImGui::IsItemActivated()) detail::GestureCancel = [revert = update(original)] { action::EmitCancel(revert); };

        if (ImGui::IsItemDeactivatedAfterEdit()) { // Gesture committed: release / Enter / instantaneous widget.
            if (changed) action::EmitStaged(update(v));
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
            action::EmitStaged(update(v));
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
    void Set(action::detail::last_field<Ms...> v) const {
        if constexpr (HasEntity) action::Emit(action::UpdateOf<Ms...>(E, std::move(v)));
        else action::Emit(action::UpdateOf<Ms...>(std::move(v)));
    }
};

Edit(entt::registry &) -> Edit<false>;
Edit(entt::registry &, entt::entity) -> Edit<true>;

} // namespace ui
