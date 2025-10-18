#include "SvgResource.h"

#include "imgui.h"
#include "lunasvg.h"

#include <cstdio> // sscanf

namespace {

lunasvg::Bitmap RenderDocumentToBitmap(const lunasvg::Document &doc, float scale = 1.f) {
    const int width = std::ceil(doc.width()) * scale;
    const int height = std::ceil(doc.height()) * scale;
    if (width == 0 || height == 0) return {};

    lunasvg::Bitmap bitmap{width, height};
    doc.render(bitmap, {scale, 0, 0, scale, 0, 0});
    return bitmap;
}

// Nearest ancestor (including self) that has href/xlink:href; empty if none.
inline lunasvg::Element FindLinkAncestor(lunasvg::Element e) {
    for (auto cur = e; cur; cur = cur.parentElement()) {
        if (cur.hasAttribute("xlink:href")) return cur;
    }
    return {};
}

} // namespace

struct SvgResource::Impl {
    Impl(vk::Device device, BitmapToImage render, fs::path path) {
        if ((Document = lunasvg::Document::loadFromFile(path))) {
            if (auto bitmap = RenderDocumentToBitmap(*Document, Scale); !bitmap.isNull()) {
                const uint32_t width = bitmap.width(), height = bitmap.height();
                const uint32_t size = width * height * 4; // RGBA8
                Image = std::make_unique<mvk::ImageResource>(render({reinterpret_cast<const std::byte *>(bitmap.data()), size}, width, height));
                Texture = std::make_unique<mvk::ImGuiTexture>(device, *Image->View);
            }
        }
    }

    // Returns the clicked link path.
    std::optional<fs::path> Render() {
        using namespace ImGui;
        if (!Document) return {};

        // Draw with document viewport aspect
        const ImVec2 doc_size{Document->width(), Document->height()};
        const auto disp_w = std::min(GetContentRegionAvail().x, doc_size.x * Scale);
        Texture->Render({disp_w, disp_w * (doc_size.y / doc_size.x)});
        if (!IsItemHovered()) return {};

        const auto item_pos = GetItemRectMin();
        const auto mouse_px = GetMousePos() - item_pos;
        const auto vp_to_disp = disp_w / doc_size.x;
        const auto vp_size = mouse_px / vp_to_disp;
        if (const auto hit = Document->elementFromPoint(vp_size.x, vp_size.y)) {
            if (const auto link = FindLinkAncestor(hit)) {
                // Map bbox (viewBox/user) -> viewport -> display px
                const auto viewBox = Document->documentElement().getAttribute("viewBox");
                ImVec2 vb_pos, vb_size;
                std::sscanf(viewBox.c_str(), "%f%f%f%f", &vb_pos.x, &vb_pos.y, &vb_size.x, &vb_size.y);

                // Map bbox -> viewport
                const float s = std::min(doc_size.x / vb_size.x, doc_size.y / vb_size.y);
                const auto bb = link.getBoundingBox();
                const auto bb_pos = ImVec2{bb.x, bb.y} * s * vp_to_disp, bb_size = ImVec2{bb.w, bb.h} * s * vp_to_disp;
                const auto p0 = item_pos + bb_pos - vb_pos * s + (doc_size - vb_size * s) * 0.5f;
                GetWindowDrawList()->AddRect(p0, p0 + bb_size, IM_COL32(0, 255, 0, 255));
                if (IsMouseClicked(ImGuiMouseButton_Left)) {
                    if (const auto href = link.getAttribute("xlink:href"); !href.empty()) return fs::path{href};
                }
            }
        }
        return {};
    }

    std::unique_ptr<lunasvg::Document> Document;
    std::unique_ptr<mvk::ImageResource> Image;
    std::unique_ptr<mvk::ImGuiTexture> Texture;

private:
    static constexpr float Scale{1.5f};
};

SvgResource::SvgResource(vk::Device device, BitmapToImage render, fs::path path)
    : Path(std::move(path)), Imp(std::make_unique<SvgResource::Impl>(device, std::move(render), Path)) {}
SvgResource::~SvgResource() = default;

std::optional<fs::path> SvgResource::Render() { return Imp->Render(); }
