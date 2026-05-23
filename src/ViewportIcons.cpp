#include "ViewportIcons.h"

#include "GpuBuffers.h"
#include "Paths.h"
#include "Pipelines.h" // ColorSubresourceRange
#include "Textures.h"
#include "ViewportComponents.h" // OneShotGpu

void LoadViewportIcons(entt::registry &r, entt::entity viewport) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &icons = r.emplace<ViewportIcons>(viewport);
    const auto svg_path = Paths::Res() / "svg";
    auto batch = BeginTextureUploadBatch(vk.Device, *one_shot.Pool, buffers.Ctx);
    const auto RenderBitmap = [&vk, &batch](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(vk, batch, data, width, height, Format::Color, ColorSubresourceRange);
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
        *svg = std::make_unique<SvgResource>(vk.Device, RenderBitmap, svg_path / name);
    }

    SubmitTextureUploadBatch(batch, vk.Queue, *one_shot.Fence, vk.Device);
}
