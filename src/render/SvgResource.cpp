#include "render/SvgResource.h"

#include "metal/ImGuiTexture.h"
#include "metal/Image.h"

#include <cmath>

#include "imgui.h"
#include "lunasvg.h"

namespace {
lunasvg::Bitmap RenderDocumentToBitmap(const lunasvg::Document &doc, float scale = 1.f) {
    const int width = std::ceil(doc.width()) * scale;
    const int height = std::ceil(doc.height()) * scale;
    if (width == 0 || height == 0) return {};

    lunasvg::Bitmap bitmap{width, height};
    doc.render(bitmap, {scale, 0, 0, scale, 0, 0});
    return bitmap;
}
} // namespace

struct SvgResource::Impl {
    Impl(const mtl::Context &ctx, const std::filesystem::path &path) {
        if ((Document = lunasvg::Document::loadFromFile(path))) {
            if (auto bitmap = RenderDocumentToBitmap(*Document, Scale); !bitmap.isNull()) {
                const uint32_t width = bitmap.width(), height = bitmap.height();
                Image = mtl::CreateTexture2D(ctx, mtl::Format::Color, {width, height}, MTL::TextureUsageShaderRead);
                mtl::Upload(Image, 0, {reinterpret_cast<const std::byte *>(bitmap.data()), width * height * 4u}, width * 4u);
            }
        }
    }

    std::unique_ptr<lunasvg::Document> Document;
    mtl::Texture Image;

private:
    static constexpr float Scale{1.5}; // Scale factor for rendering SVG to bitmap
};

SvgResource::SvgResource(const mtl::Context &ctx, const std::filesystem::path &path)
    : Imp(std::make_unique<SvgResource::Impl>(ctx, path)) {}
SvgResource::~SvgResource() = default;

void SvgResource::DrawIcon(vec2 size) const {
    if (Imp->Image) ImGui::Image(mtl::ImGuiTextureId(*Imp->Image), {size.x, size.y});
}

std::unique_ptr<SvgResource> LoadSvg(const mtl::Context &ctx, const std::filesystem::path &path) {
    return std::make_unique<SvgResource>(ctx, path);
}
