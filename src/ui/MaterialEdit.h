#pragma once

#include "action/Object.h"
#include "render/GpuBufferOps.h"
#include "render/MaterialComponents.h"
#include "render/TextureRefs.h"
#include "ui/FieldEdit.h"

#include <format>

namespace ui {
// Edits fields of the material entity's record through UpdateMaterial actions.
// The material's animation channels belong to its entity.
template<auto... Prefix>
struct MaterialEdit : FieldWidgets<Prefix...> {
    MaterialEdit(state::Registry &r, state::Entity material) : R{r}, Material{material}, Index{r.get<const ::Material>(material).Index} {}

    state::Registry &R;
    state::Entity Material;
    uint32_t Index;

    template<auto... More>
    MaterialEdit<Prefix..., More...> Sub() const { return {R, Material}; }

    template<auto... Ms>
    auto Action(action::detail::last_field<Prefix..., Ms...> value) const {
        static_assert(std::same_as<action::detail::first_class<Prefix..., Ms...>, PBRMaterial>, "MaterialEdit edits PBRMaterial fields");
        return action::object::UpdateMaterial<action::detail::last_field<Prefix..., Ms...>>{Material, action::detail::FieldOffset<Prefix..., Ms...>(), std::move(value)};
    }

    template<auto... Ms>
    auto Read() const { return ReadChain<Prefix..., Ms...>(GetMaterials(R)[Index]); }

    template<auto... Ms>
    bool Run(auto widget, bool = false) {
        using Field = action::detail::last_field<Prefix..., Ms...>;
        return detail::RunField(
            R, Material, detail::Channel<Prefix..., Ms...>(), Read<Ms...>(), widget,
            false, false, [&](action::Target, const Field &v) { action::Emit(Action<Ms...>(v), action::Phase::Stage); }
        );
    }

    template<auto... Ms>
    void Set(action::detail::last_field<Prefix..., Ms...> value) const { action::Emit(Action<Ms...>(std::move(value))); }
};

MaterialEdit(state::Registry &, state::Entity) -> MaterialEdit<>;

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
