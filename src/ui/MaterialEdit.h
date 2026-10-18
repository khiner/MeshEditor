#pragma once

#include "action/Object.h"
#include "render/GpuBufferOps.h"
#include "render/TextureRefs.h"
#include "ui/FieldEdit.h"

#include <algorithm>
#include <format>

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

    template<auto... Ms>
    auto Read() const { return ReadChain<Prefix..., Ms...>(GetMaterials(R)[Index]); }

    template<auto... Ms>
    bool Run(auto widget, bool = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        return detail::RunField(
            R, animation::AnimationsViewport(R), detail::Channel<Prefix..., Ms...>(uint16_t(Index)), Read<Ms...>(), widget,
            false, false, [&](action::Target, const Field &v) { action::Emit(Action<Ms...>(v), action::Phase::Stage); }
        );
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const { action::Emit(Action<Ms...>(std::move(value))); }
};

MaterialEdit(state::Scene &, uint32_t) -> MaterialEdit<>;

template<auto... Prefix>
void DrawEditor(MaterialEdit<Prefix...> &e, std::type_identity<TextureInfo>) {
    const auto textures = GetTextureRefs(e.R);
    e.template Run<&TextureInfo::Slot>([&](uint32_t &slot) {
        const auto match = std::ranges::find(textures, slot, &TextureRef::SamplerSlot);
        const auto preview = match != textures.end() ? match->Name :
            slot == InvalidSlot                      ? std::string{"None"} :
                                                       std::format("Missing slot {}", slot);
        constexpr auto &label = field::ChainLabel<Prefix...>;
        const bool open = ImGui::BeginCombo(label.c_str(), preview.c_str());
        if (ImGui::IsItemHovered()) {
            Hovered = {
                .Label = label.c_str(),
                .Path = field::ChainPath<Prefix..., &TextureInfo::Slot>.c_str(),
                .Type = field::TypeName<uint32_t>.c_str(),
                .Help = field::ChainSpecOf<Prefix...>.Help,
                .Spec = &field::ChainSpecOf<Prefix..., &TextureInfo::Slot>,
                .Key = detail::RunKey,
            };
        }
        if (!open) return false;
        bool changed = false;
        if (ImGui::Selectable("None", slot == InvalidSlot)) {
            slot = InvalidSlot;
            changed = true;
        }
        for (const auto &texture : textures) {
            if (ImGui::Selectable(texture.Name.c_str(), slot == texture.SamplerSlot)) {
                slot = texture.SamplerSlot;
                changed = true;
            }
        }
        ImGui::EndCombo();
        return changed;
    });
    ImGui::Indent();
    e.template Draw<&TextureInfo::TexCoord>();
    e.template Draw<&TextureInfo::UvOffset>();
    e.template Draw<&TextureInfo::UvScale>();
    e.template Draw<&TextureInfo::UvRotation>();
    ImGui::Unindent();
}
} // namespace ui
