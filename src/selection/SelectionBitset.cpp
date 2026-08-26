#include "selection/SelectionBitset.h"
#include "mesh/Mesh.h"

#include <cstring>

namespace selection {

void SelectAll(std::span<uint32_t> bits, uint32_t count) {
    if (count == 0) return;
    const uint32_t word_count = (count + 31) / 32;
    memset(bits.data(), 0xFF, word_count * sizeof(uint32_t));
    if (const uint32_t rem = count & 31) bits[word_count - 1] = (1u << rem) - 1u;
}

uint32_t CountSelected(std::span<const uint32_t> bits, uint32_t count) {
    if (count == 0) return 0;
    const uint32_t last_word = (count + 31) / 32;
    uint32_t total = 0;
    for (uint32_t w = 0; w + 1 < last_word; ++w) total += __builtin_popcount(bits[w]);
    uint32_t last = bits[last_word - 1];
    if (const uint32_t end_bit = count & 31) last &= (1u << end_bit) - 1u;
    return total + __builtin_popcount(last);
}

std::vector<uint32_t> ConvertSelectionElement(std::span<const uint32_t> bits, uint32_t count, const Mesh &mesh, Element from_element, Element to_element) {
    if (from_element == Element::None || count == 0) return {};

    std::vector<uint32_t> result;
    if (from_element == to_element) {
        ForEachSelected(bits, count, [&](uint32_t h) { result.emplace_back(h); });
        return result;
    }
    const auto selected = [&](uint32_t handle) { return handle < count && IsSelected(bits, handle); };
    if (from_element == Element::Face) {
        if (to_element == Element::Edge) {
            ForEachSelected(bits, count, [&](uint32_t f) {
                for (const auto heh : mesh.fh_range(he::FH{f})) result.emplace_back(*mesh.GetEdge(heh));
            });
        } else if (to_element == Element::Vertex) {
            ForEachSelected(bits, count, [&](uint32_t f) {
                for (const auto vh : mesh.fv_range(he::FH{f})) result.emplace_back(*vh);
            });
        }
    } else if (from_element == Element::Edge) {
        if (to_element == Element::Vertex) {
            ForEachSelected(bits, count, [&](uint32_t e) {
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
