#pragma once

#include "metal/Command.h"
#include "metal/Image.h"
#include "metal/Shader.h"

#include <imgui.h>

namespace mtl {
// Draws the current ImGui context's draw data into a render pass whose color target has the renderer's pixel format.
// Owns the textures the context's atlas requests, which draw data references as MTL::Texture pointers.
struct ImGuiRenderer {
    // The capabilities a context declares when this renderer draws its output.
    static constexpr ImGuiBackendFlags BackendFlags{ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset};

    ImGuiRenderer(LibraryCache &, MTL::PixelFormat target);
    ~ImGuiRenderer();
    ImGuiRenderer(const ImGuiRenderer &) = delete;
    ImGuiRenderer &operator=(const ImGuiRenderer &) = delete;

    // Uploads the draw data into the encoder's command and records its draws.
    void Render(ImDrawData *, RenderEncoder);

private:
    void UpdateTexture(ImTextureData *);

    const Context &Ctx;
    RenderPipeline Pipeline;
    NS::SharedPtr<MTL::SamplerState> Sampler;
    std::vector<std::unique_ptr<Texture>> Textures;
};
} // namespace mtl
