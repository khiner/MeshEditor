#include "SceneSelection.h"
#include "Entity.h"
#include "MeshComponents.h"
#include "MeshInstance.h"
#include "mesh/Mesh.h"
#include "scene_impl/SceneInternalTypes.h"

#include <entt/entity/registry.hpp>

namespace scene_selection {

std::unordered_set<uint32_t> ConvertSelectionElement(const MeshSelection &selection, const Mesh &mesh, Element from_element, Element to_element) {
    if (from_element == Element::None || selection.Handles.empty()) return {};
    if (from_element == to_element) return selection.Handles;

    const auto &handles = selection.Handles;
    std::unordered_set<uint32_t> new_handles;
    if (from_element == Element::Face) {
        if (to_element == Element::Edge) {
            for (auto f : handles) {
                for (const auto heh : mesh.fh_range(he::FH{f})) new_handles.emplace(*mesh.GetEdge(heh));
            }
        } else if (to_element == Element::Vertex) {
            for (auto f : handles) {
                for (const auto vh : mesh.fv_range(he::FH{f})) new_handles.emplace(*vh);
            }
        }
    } else if (from_element == Element::Edge) {
        if (to_element == Element::Vertex) {
            for (auto eh_raw : handles) {
                const auto heh = mesh.GetHalfedge(he::EH{eh_raw}, 0);
                new_handles.emplace(*mesh.GetFromVertex(heh));
                new_handles.emplace(*mesh.GetToVertex(heh));
            }
        } else if (to_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto heh : mesh.fh_range(fh)) {
                    if (!handles.contains(*mesh.GetEdge(heh))) {
                        all_selected = false;
                        break;
                    }
                }
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    } else if (from_element == Element::Vertex) {
        if (to_element == Element::Edge) {
            for (const auto eh : mesh.edges()) {
                const auto heh = mesh.GetHalfedge(eh, 0);
                if (handles.contains(*mesh.GetFromVertex(heh)) && handles.contains(*mesh.GetToVertex(heh))) {
                    new_handles.emplace(*eh);
                }
            }
        } else if (to_element == Element::Face) {
            for (const auto fh : mesh.faces()) {
                bool all_selected = true;
                for (const auto vh : mesh.fv_range(fh)) {
                    if (!handles.contains(*vh)) {
                        all_selected = false;
                        break;
                    }
                }
                if (all_selected) new_handles.emplace(*fh);
            }
        }
    }
    return new_handles;
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
