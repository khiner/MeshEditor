#include "metal/ImGuiRenderer.h"
#include "metal/AutoreleaseScope.h"
#include "metal/ImGuiTexture.h"

#include "metal/MetalCpp.h"

namespace mtl {
namespace {
constexpr uint32_t VertexBinding{0u}, ProjectionBinding{1u}, ImageBinding{0u}, SamplerBinding{0u};
} // namespace

ImGuiRenderer::ImGuiRenderer(LibraryCache &libraries, MTL::PixelFormat target)
    : Ctx(libraries.Ctx),
      Pipeline(MakeRenderPipeline(libraries, {"ImGui.metal", "ImGuiVertex"}, FunctionRef{"ImGui.metal", "ImGuiFragment"}, PassFormats{{target}}, {Blend}, DepthState{.Test = false, .Write = false})),
      Sampler(CreateSampler(Ctx, MTL::SamplerMinMagFilterLinear, MTL::SamplerMipFilterNotMipmapped, MTL::SamplerAddressModeClampToEdge)) {
    auto &io = ImGui::GetIO();
    io.BackendRendererName = "MeshEditor Metal 4";
    io.BackendFlags |= BackendFlags;
}

ImGuiRenderer::~ImGuiRenderer() {
    const AutoreleaseScope pool;
    // The platform backend's viewport data goes with the renderer, as ImGui's renderer backends do it.
    ImGui::DestroyPlatformWindows();
    for (auto *texture : ImGui::GetPlatformIO().Textures) {
        if (texture->BackendUserData) {
            texture->SetTexID(ImTextureID_Invalid);
            texture->BackendUserData = nullptr;
            texture->SetStatus(ImTextureStatus_Destroyed);
        }
    }
    Textures.clear();
    auto &io = ImGui::GetIO();
    io.BackendRendererName = nullptr;
    io.BackendFlags &= ~BackendFlags;
}

void ImGuiRenderer::UpdateTexture(ImTextureData *texture) {
    const AutoreleaseScope pool;
    if (texture->Status == ImTextureStatus_WantCreate) {
        auto created = std::make_unique<Texture>(CreateTexture2D(Ctx, MTL::PixelFormatRGBA8Unorm, {uint32_t(texture->Width), uint32_t(texture->Height)}, MTL::TextureUsageShaderRead, 1, MTL::StorageModeShared));
        Upload(*created, 0, {static_cast<const std::byte *>(texture->GetPixels()), size_t(texture->GetSizeInBytes())}, uint32_t(texture->GetPitch()));
        texture->SetTexID(ImGuiTextureId(**created));
        texture->BackendUserData = created.get();
        texture->SetStatus(ImTextureStatus_OK);
        Textures.push_back(std::move(created));
    } else if (texture->Status == ImTextureStatus_WantUpdates) {
        const auto &created = *static_cast<Texture *>(texture->BackendUserData);
        for (const auto &rect : texture->Updates) {
            created.Handle->replaceRegion(MTL::Region::Make2D(rect.x, rect.y, rect.w, rect.h), 0, texture->GetPixelsAt(rect.x, rect.y), uint32_t(texture->GetPitch()));
        }
        texture->SetStatus(ImTextureStatus_OK);
    } else if (texture->Status == ImTextureStatus_WantDestroy && texture->UnusedFrames > 0) {
        // The texture leaves residency after the commands that sample it complete.
        auto *created = static_cast<Texture *>(texture->BackendUserData);
        std::erase_if(Textures, [&](const auto &owned) { return owned.get() == created; });
        texture->SetTexID(ImTextureID_Invalid);
        texture->BackendUserData = nullptr;
        texture->SetStatus(ImTextureStatus_Destroyed);
    }
}

void ImGuiRenderer::Render(ImDrawData *draw_data, RenderEncoder encoder) {
    const AutoreleaseScope pool;
    const auto fb_width = int(draw_data->DisplaySize.x * draw_data->FramebufferScale.x + 0.5f);
    const auto fb_height = int(draw_data->DisplaySize.y * draw_data->FramebufferScale.y + 0.5f);
    if (fb_width <= 0 || fb_height <= 0 || draw_data->CmdLists.Size == 0) return;
    if (draw_data->Textures) {
        for (auto *texture : *draw_data->Textures) {
            if (texture->Status != ImTextureStatus_OK) UpdateTexture(texture);
        }
    }

    auto &command = *encoder.Cmd;
    const float left = draw_data->DisplayPos.x, right = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    const float top = draw_data->DisplayPos.y, bottom = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    const float projection[4][4]{
        {2.f / (right - left), 0.f, 0.f, 0.f},
        {0.f, 2.f / (top - bottom), 0.f, 0.f},
        {0.f, 0.f, 1.f, 0.f},
        {(right + left) / (left - right), (top + bottom) / (bottom - top), 0.f, 1.f},
    };
    const auto bind_state = [&] {
        Pipeline.Bind(encoder);
        SetCullMode(encoder, MTL::CullModeNone);
        encoder->setViewport({0.0, 0.0, double(fb_width), double(fb_height), 0.0, 1.0});
        command.SetBytes(projection, ProjectionBinding);
        command.SetSampler(Sampler.get(), SamplerBinding);
    };
    bind_state();

    const auto clip_offset = draw_data->DisplayPos, clip_scale = draw_data->FramebufferScale;
    for (const auto *list : draw_data->CmdLists) {
        const auto vertices = command.Upload(std::as_bytes(std::span{list->VtxBuffer.Data, size_t(list->VtxBuffer.Size)}));
        const auto indices = command.Upload(std::as_bytes(std::span{list->IdxBuffer.Data, size_t(list->IdxBuffer.Size)}));
        const auto index_bytes = uint64_t(list->IdxBuffer.Size) * sizeof(ImDrawIdx);
        for (const auto &cmd : list->CmdBuffer) {
            if (cmd.UserCallback) {
                if (cmd.UserCallback == ImDrawCallback_ResetRenderState) bind_state();
                else cmd.UserCallback(list, &cmd);
                continue;
            }
            const ImVec2 clip_min{(cmd.ClipRect.x - clip_offset.x) * clip_scale.x, (cmd.ClipRect.y - clip_offset.y) * clip_scale.y};
            const ImVec2 clip_max{(cmd.ClipRect.z - clip_offset.x) * clip_scale.x, (cmd.ClipRect.w - clip_offset.y) * clip_scale.y};
            const auto x0 = uint32_t(std::max(clip_min.x, 0.f)), y0 = uint32_t(std::max(clip_min.y, 0.f));
            const auto x1 = uint32_t(std::min(clip_max.x, float(fb_width))), y1 = uint32_t(std::min(clip_max.y, float(fb_height)));
            if (x1 <= x0 || y1 <= y0 || cmd.ElemCount == 0) continue;
            encoder->setScissorRect({x0, y0, x1 - x0, y1 - y0});
            if (const auto texture = cmd.GetTexID(); texture != ImTextureID_Invalid) command.SetTexture(reinterpret_cast<MTL::Texture *>(texture), ImageBinding);
            command.SetAddress(vertices + uint64_t(cmd.VtxOffset) * sizeof(ImDrawVert), VertexBinding);
            const auto index_offset = uint64_t(cmd.IdxOffset) * sizeof(ImDrawIdx);
            encoder->drawIndexedPrimitives(
                MTL::PrimitiveTypeTriangle, cmd.ElemCount, sizeof(ImDrawIdx) == 2 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32,
                indices + index_offset, index_bytes - index_offset
            );
        }
    }
}
} // namespace mtl
