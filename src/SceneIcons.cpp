#include "SceneIcons.h"

#include "Paths.h"
#include "SceneBuffers.h"
#include "SceneComponents.h" // SceneOneShotGpu
#include "ScenePipelines.h" // ColorSubresourceRange
#include "SceneTextures.h"
#include "SceneVulkanResources.h"
#include "SvgResource.h"

#include <entt/entity/registry.hpp>

void LoadSceneIcons(entt::registry &R, entt::entity scene_entity) {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(scene_entity);
    auto &Buffers = R.get<SceneBuffers>(scene_entity);
    auto &icons = R.emplace<SceneIcons>(scene_entity);
    const auto svg_path = Paths::Res() / "svg";
    auto batch = BeginTextureUploadBatch(Vk.Device, *one_shot.Pool, Buffers.Ctx);
    const auto RenderBitmap = [&Vk, &batch](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(Vk, batch, data, width, height, Format::Color, ColorSubresourceRange);
    };

    const std::pair<std::unique_ptr<SvgResource> *, std::string_view> entries[] = {
        {&icons.Transform.Select, "select.svg"},
        {&icons.Transform.SelectBox, "select_box.svg"},
        {&icons.Transform.Move, "move.svg"},
        {&icons.Transform.Rotate, "rotate.svg"},
        {&icons.Transform.Scale, "scale.svg"},
        {&icons.Transform.Universal, "transform.svg"},
        {&icons.Shading.Wireframe, "shading_wire.svg"},
        {&icons.Shading.Solid, "shading_solid.svg"},
        {&icons.Shading.MaterialPreview, "shading_texture.svg"},
        {&icons.Shading.Rendered, "shading_rendered.svg"},
        {&icons.Overlay, "overlay.svg"},
        {&icons.Anim.Play, "play.svg"},
        {&icons.Anim.Pause, "pause.svg"},
        {&icons.Anim.JumpStart, "jump_start.svg"},
        {&icons.Anim.JumpEnd, "jump_end.svg"},
    };
    for (const auto &[svg, name] : entries) {
        *svg = std::make_unique<SvgResource>(Vk.Device, RenderBitmap, svg_path / name);
    }

    SubmitTextureUploadBatch(batch, Vk.Queue, *one_shot.Fence, Vk.Device);
}
