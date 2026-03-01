#include "SceneSelection.h"
#include "Entity.h"
#include "MeshComponents.h"
#include "MeshInstance.h"
#include "mesh/Mesh.h"

#include <entt/entity/registry.hpp>

namespace scene_selection {

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
        total += static_cast<uint32_t>(__builtin_popcount(word));
    }
    return total;
}

std::vector<uint32_t> ScanBitsetRange(const uint32_t *bits, uint32_t offset, uint32_t count) {
    std::vector<uint32_t> result;
    const uint32_t first_word = offset / 32, last_word = (offset + count + 31) / 32;
    for (uint32_t w = first_word; w < last_word; ++w) {
        uint32_t word = bits[w];
        while (word) {
            const uint32_t global_idx = w * 32 + static_cast<uint32_t>(__builtin_ctz(word));
            if (global_idx >= offset && global_idx < offset + count) result.emplace_back(global_idx - offset);
            word &= word - 1;
        }
    }
    return result;
}

std::vector<uint32_t> ConvertSelectionElement(std::span<const uint32_t> handles, const Mesh &mesh, Element from_element, Element to_element) {
    if (from_element == Element::None || handles.empty()) return {};
    if (from_element == to_element) return {handles.begin(), handles.end()};

    std::vector<uint32_t> result;
    if (from_element == Element::Face) {
        if (to_element == Element::Edge) {
            for (auto f : handles) {
                for (const auto heh : mesh.fh_range(he::FH{f})) result.emplace_back(*mesh.GetEdge(heh));
            }
        } else if (to_element == Element::Vertex) {
            for (auto f : handles) {
                for (const auto vh : mesh.fv_range(he::FH{f})) result.emplace_back(*vh);
            }
        }
    } else if (from_element == Element::Edge) {
        if (to_element == Element::Vertex) {
            for (auto eh_raw : handles) {
                const auto heh = mesh.GetHalfedge(he::EH{eh_raw}, 0);
                result.emplace_back(*mesh.GetFromVertex(heh));
                result.emplace_back(*mesh.GetToVertex(heh));
            }
        } else if (to_element == Element::Face) {
            const std::unordered_set<uint32_t> handle_set{handles.begin(), handles.end()};
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto heh : mesh.fh_range(fh)) {
                    if (!handle_set.contains(*mesh.GetEdge(heh))) { all_selected = false; break; }
                }
                if (all_selected) result.emplace_back(*fh);
            }
        }
    } else if (from_element == Element::Vertex) {
        const std::unordered_set<uint32_t> handle_set{handles.begin(), handles.end()};
        if (to_element == Element::Edge) {
            for (const auto eh : mesh.edges()) {
                const auto heh = mesh.GetHalfedge(eh, 0);
                if (handle_set.contains(*mesh.GetFromVertex(heh)) && handle_set.contains(*mesh.GetToVertex(heh))) {
                    result.emplace_back(*eh);
                }
            }
        } else if (to_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto vh : mesh.fv_range(fh)) {
                    if (!handle_set.contains(*vh)) { all_selected = false; break; }
                }
                if (all_selected) result.emplace_back(*fh);
            }
        }
    }
    return result;
}

std::unordered_map<entt::entity, entt::entity> ComputePrimaryEditInstances(const entt::registry &r, bool include_frozen) {
    std::unordered_map<entt::entity, entt::entity> primaries;
    const auto active = FindActiveEntity(r);
    for (const auto [e, mi, ok, ri] : r.view<const MeshInstance, const Selected, const ObjectKind, const RenderInstance>().each()) {
        if (ok.Value != ObjectType::Mesh) continue;
        if (!include_frozen && r.all_of<Frozen>(e)) continue;
        auto &primary = primaries[mi.MeshEntity];
        if (primary == entt::entity{} || e == active) primary = e;
    }
    return primaries;
}

bool HasFrozenInstance(const entt::registry &r, entt::entity mesh_entity) {
    for (const auto [e, mi] : r.view<const MeshInstance, const Frozen>().each()) {
        if (mi.MeshEntity == mesh_entity) return true;
    }
    return false;
}

std::unordered_set<entt::entity> GetSelectedMeshEntities(const entt::registry &r) {
    std::unordered_set<entt::entity> entities;
    for (const auto [e, mi] : r.view<const MeshInstance, const Selected>().each()) entities.emplace(mi.MeshEntity);
    return entities;
}

uint32_t GetElementCount(const Mesh &mesh, Element element) {
    if (element == Element::Vertex) return mesh.VertexCount();
    if (element == Element::Edge) return mesh.EdgeCount();
    if (element == Element::Face) return mesh.FaceCount();
    return 0;
}

} // namespace scene_selection
