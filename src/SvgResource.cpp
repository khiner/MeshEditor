#include "SvgResource.h"

#include "imgui.h"
#include "lunasvg.h"

namespace {
// Find the deepest link element whose bounding box contains the point.
std::optional<lunasvg::Element> FindLinkAtPoint(const lunasvg::Element &root, ImVec2 point, const std::string &attribute) {
    static const auto contains = [](const lunasvg::Box &box, ImVec2 point) {
        return box.x <= point.x && point.x <= (box.x + box.w) && box.y <= point.y && point.y <= (box.y + box.h);
    };
    std::optional<lunasvg::Element> found;
    if (root.hasAttribute(attribute) && contains(root.getGlobalBoundingBox(), point)) found = root;
    for (const auto &childNode : root.children()) {
        if (auto descendent = FindLinkAtPoint(childNode.toElement(), point, attribute)) found = *descendent;
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
    Impl(vk::Device device, BitmapToImage render, std::filesystem::path path) {
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
    std::optional<std::filesystem::path> Draw() {
        using namespace ImGui;

        const auto doc = Document->documentElement();
        const auto doc_width = Document->width(), doc_height = Document->height();
        if (doc_width <= 0.f || doc_height <= 0.f) return {};

        const auto display_width = std::min(GetContentRegionAvail().x, doc_width * Scale);
        const auto &t = *Texture;
        ImGui::Image(ImTextureID(VkDescriptorSet(t.DescriptorSet)), {display_width, display_width * doc_height / doc_width}, {t.Uv0.x, t.Uv0.y}, {t.Uv1.x, t.Uv1.y});
        if (IsItemHovered()) {
            static constexpr std::string LinkAttribute{"xlink:href"};
            const auto display_scale = display_width / doc_width;
            const auto offset = GetItemRectMin();
            const auto local_point = (GetMousePos() - offset) / display_scale;
            auto link = FindLinkAtPoint(doc, local_point, LinkAttribute);
            if (link) {
                const auto box = link->getGlobalBoundingBox();
                if (box.w > 0.f && box.h > 0.f) {
                    GetWindowDrawList()->AddRect(
                        offset + ImVec2{box.x, box.y} * display_scale,
                        offset + ImVec2{box.x + box.w, box.y + box.h} * display_scale,
                        IM_COL32(0, 255, 0, 255)
                    );
                }
                if (IsMouseClicked(ImGuiMouseButton_Left)) return link->getAttribute(LinkAttribute);
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

SvgResource::SvgResource(vk::Device device, BitmapToImage render, std::filesystem::path path)
    : Path(std::move(path)), Imp(std::make_unique<SvgResource::Impl>(device, std::move(render), Path)) {}
SvgResource::~SvgResource() = default;

std::optional<std::filesystem::path> SvgResource::Draw() { return Imp->Draw(); }
void SvgResource::DrawIcon(vec2 size) const {
    if (const auto *t = Imp->Texture.get()) ImGui::Image(ImTextureID(VkDescriptorSet(t->DescriptorSet)), {size.x, size.y}, {t->Uv0.x, t->Uv0.y}, {t->Uv1.x, t->Uv1.y});
}
