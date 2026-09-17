#pragma once

// MaterialEdit{R, index} edits fields of a material in the material buffer.

#include "action/Object.h"
#include "animation/Clips.h"
#include "render/GpuBufferOps.h"
#include "ui/FieldEdit.h"

namespace ui {
// Edits fields of material `Index` through UpdateMaterial actions.
// Material channels key on the scene viewport.
template<auto... Prefix>
struct MaterialEdit : FieldWidgets<MaterialEdit<Prefix...>, Prefix...> {
    MaterialEdit(state::Scene &r, uint32_t index) : R{r}, Index{index} {}

    state::Scene &R;
    uint32_t Index;

    template<auto... More>
    MaterialEdit<Prefix..., More...> Sub() const { return {R, Index}; }

    template<auto... Ms>
    auto Action(action::detail::last_field<Prefix..., Ms...> value) const {
        static_assert(std::same_as<action::detail::first_class<Prefix..., Ms...>, PBRMaterial>, "MaterialEdit edits PBRMaterial fields");
        return action::object::UpdateMaterial<action::detail::last_field<Prefix..., Ms...>>{Index, action::detail::FieldOffset<Prefix..., Ms...>(), std::move(value)};
    }

    template<auto... Ms, typename Widget>
    bool Run(Widget widget, bool = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        return detail::RunField(
            R, animation::AnimationsViewport(R), detail::Channel<Prefix..., Ms...>(uint16_t(Index)), ReadChain<Prefix..., Ms...>(GetMaterials(R)[Index]), widget,
            false, false, [&](action::Scope, const Field &v) { action::Emit(Action<Ms...>(v), action::Phase::Stage); }
        );
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const { action::Emit(Action<Ms...>(std::move(value))); }
};

MaterialEdit(state::Scene &, uint32_t) -> MaterialEdit<>;
} // namespace ui
