#include "selection/SelectionBitset.h"
#include "mesh/Mesh.h"

#include <cstring>

namespace selection {

void SelectAll(uint32_t *bits, uint32_t offset, uint32_t count) {
    if (count == 0) return;
    const uint32_t word_count = (count + 31) / 32;
    auto *u32 = bits + offset / 32;
    memset(u32, 0xFF, word_count * sizeof(uint32_t));
    if (const uint32_t rem = count & 31) u32[word_count - 1] = (1u << rem) - 1u;
}

uint32_t CountSelected(const uint32_t *bits, uint32_t offset, uint32_t count) {
    if (count == 0) return 0;
    uint32_t total = 0;
    const uint32_t first_word = offset / 32, last_word = (offset + count + 31) / 32;
    for (uint32_t w = first_word; w < last_word; ++w) {
        uint32_t word = bits[w];
        // Mask off bits outside [offset, offset+count)
        if (w == first_word && (offset & 31)) word &= ~((1u << (offset & 31)) - 1u);
        if (w == last_word - 1) {
            const uint32_t end_bit = (offset + count) & 31;
            if (end_bit) word &= (1u << end_bit) - 1u;
        }
        total += __builtin_popcount(word);
    }
    return total;
}

std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count) {
    std::vector<uint32_t> result;
    ForEachSelected(bits, offset, count, [&](uint32_t handle) { result.emplace_back(handle); });
    return result;
}

std::vector<uint32_t> ConvertSelectionElement(const uint32_t *bits, uint32_t offset, uint32_t count, const Mesh &mesh, Element from_element, Element to_element) {
    if (from_element == Element::None || count == 0) return {};

    std::vector<uint32_t> result;
    if (from_element == to_element) {
        ForEachSelected(bits, offset, count, [&](uint32_t h) { result.emplace_back(h); });
        return result;
    }
    const auto selected = [&](uint32_t handle) { return handle < count && IsSelected(bits, offset, handle); };
    if (from_element == Element::Face) {
        if (to_element == Element::Edge) {
            ForEachSelected(bits, offset, count, [&](uint32_t f) {
                for (const auto heh : mesh.fh_range(he::FH{f})) result.emplace_back(*mesh.GetEdge(heh));
            });
        } else if (to_element == Element::Vertex) {
            ForEachSelected(bits, offset, count, [&](uint32_t f) {
                for (const auto vh : mesh.fv_range(he::FH{f})) result.emplace_back(*vh);
            });
        }
    } else if (from_element == Element::Edge) {
        if (to_element == Element::Vertex) {
            ForEachSelected(bits, offset, count, [&](uint32_t e) {
                const auto heh = mesh.GetHalfedge(he::EH{e}, 0);
                result.emplace_back(*mesh.GetFromVertex(heh));
                result.emplace_back(*mesh.GetToVertex(heh));
            });
        } else if (to_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto heh : mesh.fh_range(fh)) {
                    if (!selected(*mesh.GetEdge(heh))) {
                        all_selected = false;
                        break;
                    }
                }
                if (all_selected) result.emplace_back(*fh);
            }
        }
    } else if (from_element == Element::Vertex) {
        if (to_element == Element::Edge) {
            for (const auto eh : mesh.edges()) {
                if (const auto heh = mesh.GetHalfedge(eh, 0); selected(*mesh.GetFromVertex(heh)) && selected(*mesh.GetToVertex(heh))) {
                    result.emplace_back(*eh);
                }
            }
        } else if (to_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto vh : mesh.fv_range(fh)) {
                    if (!selected(*vh)) {
                        all_selected = false;
                        break;
                    }
                }
                if (all_selected) result.emplace_back(*fh);
            }
        }
    }
    return result;
}

uint32_t GetElementCount(const Mesh &mesh, Element element) {
    if (element == Element::Vertex) return mesh.VertexCount();
    if (element == Element::Edge) return mesh.EdgeCount();
    if (element == Element::Face) return mesh.FaceCount();
    return 0;
}

} // namespace selection
