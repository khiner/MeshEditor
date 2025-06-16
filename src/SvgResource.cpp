#include "SvgResource.h"

#include "imgui.h"
#include "lunasvg.h"

namespace {
// Find the deepest descendant element with the given attribute containing the given point.
std::optional<lunasvg::Element> FindElementAtPoint(const lunasvg::Element &element, const std::string &attribute, ImVec2 point, float scale = 1.f) {
    constexpr auto contains = [](const lunasvg::Box &box, ImVec2 point, float scale) {
        return box.x * scale <= point.x && point.x <= (box.x + box.w) * scale &&
            box.y * scale <= point.y && point.y <= (box.y + box.h) * scale;
    };

    std::optional<lunasvg::Element> found;
    if (contains(element.getBoundingBox(), point, scale) && element.hasAttribute(attribute)) {
        found = element;
    }
    for (const auto &childNode : element.children()) {
        if (auto descendent = FindElementAtPoint(childNode.toElement(), attribute, point, scale)) {
            found = *descendent;
        }
    }
    return found;
}

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
    Impl(vk::Device device, BitmapToImage render, fs::path path) {
        if ((Document = lunasvg::Document::loadFromFile(path))) {
            if (auto bitmap = RenderDocumentToBitmap(*Document, Scale); !bitmap.isNull()) {
                const auto width = uint32_t(bitmap.width()), height = uint32_t(bitmap.height());
                const auto size = width * height * 4; // 4 bytes per pixel
                Image = std::make_unique<mvk::ImageResource>(render({reinterpret_cast<const std::byte *>(bitmap.data()), size}, width, height));
                Texture = std::make_unique<mvk::ImGuiTexture>(device, *Image->View);
            }
        }
    }

    // Returns the clicked link path.
    std::optional<fs::path> Render() {
        using namespace ImGui;

        const auto doc = Document->documentElement();
        const auto doc_box = doc.getBoundingBox();
        const auto display_width = std::min(GetContentRegionAvail().x, doc_box.w * Scale);
        Texture->Render({display_width, display_width * doc_box.h / doc_box.w});
        if (IsItemHovered()) {
            static constexpr std::string LinkAttribute = "xlink:href";
            const auto display_scale = display_width / doc_box.w;
            const auto offset = GetItemRectMin();
            if (auto element = FindElementAtPoint(doc, LinkAttribute, GetMousePos() - offset, display_scale)) {
                const auto box = element->getBoundingBox();
                GetWindowDrawList()->AddRect(
                    offset + ImVec2{box.x, box.y} * display_scale,
                    offset + ImVec2{box.x + box.w, box.y + box.h} * display_scale,
                    IM_COL32(0, 255, 0, 255)
                );
                if (IsMouseClicked(ImGuiMouseButton_Left)) return element->getAttribute(LinkAttribute);
            }
        }
        return {};
    }

    std::unique_ptr<lunasvg::Document> Document;
    std::unique_ptr<mvk::ImageResource> Image;
    std::unique_ptr<mvk::ImGuiTexture> Texture;

private:
    static constexpr float Scale{1.5}; // Scale factor for rendering SVG to bitmap.
};

SvgResource::SvgResource(vk::Device device, BitmapToImage render, fs::path path)
    : Path(std::move(path)), Imp(std::make_unique<SvgResource::Impl>(device, std::move(render), Path)) {}
SvgResource::~SvgResource() = default;

std::optional<fs::path> SvgResource::Render() { return Imp->Render(); }
