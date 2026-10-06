#include "mesh/EditVisibility.h"
#include "Profile.h"
#include "mesh/MeshStore.h"
#include "metal/Dispatch.h"
#include "state/Scene.h"
#include <set>

bool EditVisibility(state::Scene &r, std::span<const uint32_t> ids, Element element, EditVisibilityOperation operation) {
    constexpr std::array elements{Element::Vertex, Element::Edge, Element::Face};
    const auto source = uint32_t(std::ranges::find(elements, element) - elements.begin());
    if (source >= 3u || ids.empty()) return false;
    const profile::CpuScope scope{"EditVisibility"};
    const bool reveal = operation == EditVisibilityOperation::Reveal || operation == EditVisibilityOperation::RevealSelected;
    const bool select = operation == EditVisibilityOperation::RevealSelected;
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.EnsureSelectionState(r, chain, ids);
    std::vector<MeshStore::SelectionUpdate> updates;
    uint64_t changed = 0u;
    for (const auto id : ids) {
        const Mesh mesh{meshes, id};
        const auto &c = mesh.GetConnectivity();
        const auto incidence = mesh.GetVertexEdgeIncidence();
        const std::array hidden{meshes.GetHiddenElements(id, elements[0]), meshes.GetHiddenElements(id, elements[1]), meshes.GetHiddenElements(id, elements[2])};
        std::array<std::set<uint32_t>, 3> changes;
        const auto hide = [&](uint32_t d, uint32_t h) { if (!hidden[d].Contains(h)) changes[d].insert(h); };
        const auto is_hidden = [&](uint32_t d, uint32_t h) { return hidden[d].Contains(h) || changes[d].contains(h); };
        const auto unselected = [&](uint32_t d, auto &&visit) {
            const auto selection = meshes.GetSelectedElements(id, elements[d]);
            const auto &a = meshes.Arenas();
            const auto membership = (d == 0u ? a.Vertices.Blocks : d == 1u ? a.EdgeHalfedges.Blocks :
                                                                             a.FaceTriangles.Blocks)
                                        .Buffer.GetSpan<MeshElementBlock>();
            selection.Tree.Visit(SelectionIndexMask::Unselected, false, [&](uint32_t block) {
                for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) {
                    const auto word = block * MeshElementBlockWords + w;
                    for (auto bits = membership[block].Live[w] & ~(selection.Bits[word] | hidden[d].Bits[word]); bits; bits &= bits - 1u)
                        visit(word * 32u + uint32_t(std::countr_zero(bits)));
                }
                return true;
            });
        };
        // Every radial user is visited, including nonmanifold and same-winding faces.
        const auto edge_faces = [&](uint32_t edge, auto &&visit) {
            const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
            if (!c.FaceOf(h)) return;
            auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
            if (c.VertexCorners[a].y > c.VertexCorners[b].y) std::swap(a, b);
            const auto fan = c.VertexCorners[a];
            for (uint32_t i = 0u; i < fan.y; ++i) {
                const auto item = c.FanItems[fan.x + i];
                if (item.y == InvalidOffset) continue;
                const he::HH corner{item.x};
                if (*mesh.GetEdge(corner) == edge || *mesh.GetEdge(c.Next(corner)) == edge) visit(item.y);
            }
        };
        if (reveal) {
            for (uint32_t d = 0u; d < 3u; ++d) hidden[d].ForEach([&](uint32_t h) { changes[d].insert(h); });
        } else {
            if (operation == EditVisibilityOperation::HideUnselected) unselected(source, [&](uint32_t h) { hide(source, h); });
            else meshes.GetSelectedElements(id, element).ForEach([&](uint32_t h) { hide(source, h); });
            std::set<uint32_t> vertices, edges;
            if (source == 0u) {
                for (const auto v : changes[0]) {
                    for (const auto e : incidence.Incident(v)) hide(1u, e);
                    const auto fan = c.VertexCorners[v];
                    for (uint32_t i = 0u; i < fan.y; ++i)
                        if (const auto face = c.FanItems[fan.x + i].y; face != InvalidOffset) hide(2u, face);
                }
            } else if (source == 1u) {
                for (const auto e : changes[1]) {
                    edge_faces(e, [&](uint32_t f) { hide(2u, f); });
                    const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
                    vertices.insert(*mesh.GetFromVertex(h));
                    vertices.insert(*mesh.GetToVertex(h));
                }
            } else {
                for (const auto f : changes[2])
                    for (const auto h : mesh.fh_range(he::FH{f})) {
                        vertices.insert(*mesh.GetToVertex(h));
                        edges.insert(*mesh.GetEdge(h));
                    }
                for (const auto e : edges) {
                    bool visible = false;
                    edge_faces(e, [&](uint32_t f) { visible |= !is_hidden(2u, f); });
                    if (!visible) hide(1u, e);
                }
            }
            const auto hide_isolated = [&](uint32_t v) {
                for (const auto e : incidence.Incident(v))
                    if (!is_hidden(1u, e)) return;
                hide(0u, v);
            };
            for (const auto v : vertices) hide_isolated(v);
            if (operation == EditVisibilityOperation::HideUnselected) {
                if (source == 2u && !mesh.FaceCount()) unselected(1u, [&](uint32_t e) { hide(1u, e); });
                if (source != 0u) unselected(0u, hide_isolated);
            }
        }
        if (std::ranges::all_of(changes, [](const auto &domain) { return domain.empty(); })) continue;
        auto &update = updates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id, .Source = element});
        for (uint32_t d = 0u; d < 3u; ++d) {
            changed += changes[d].size();
            std::vector<std::pair<uint32_t, uint32_t>> words;
            for (const auto h : changes[d]) {
                if (words.empty() || words.back().first != h / 32u) words.emplace_back(h / 32u, 0u);
                words.back().second |= 1u << (h % 32u);
            }
            std::vector<uint32_t> blocks;
            for (const auto [word, bits] : words) {
                if (blocks.empty() || blocks.back() != word / MeshElementBlockWords) blocks.push_back(word / MeshElementBlockWords);
                update.Seeds.push_back({d, word, bits});
            }
            const auto write = [&](bool set) {
                return [&, set, at = words.begin()](uint32_t block, auto &out) mutable {
                    while (at != words.end() && at->first / MeshElementBlockWords == block) {
                        const auto [word, bits] = *at++;
                        if (set) out[word % MeshElementBlockWords] |= bits;
                        else out[word % MeshElementBlockWords] &= ~bits;
                    }
                };
            };
            meshes.EditHiddenBlocks(elements[d], blocks, write(!reveal));
            if (!reveal || (select && d == source)) meshes.EditSelectionBlocks(elements[d], blocks, write(reveal));
        }
        auto &summary = meshes.WriteSelectionSummary(id);
        summary.ActiveHandle = InvalidOffset;
    }
    if (updates.empty()) return false;
    profile::RecordCounter("VisibilityChangedElements", changed);
    meshes.UpdateSelection(r, chain, updates);
    chain.Submit();
    for (const auto &update : updates) meshes.PublishSelectionSummary(update.StoreId);
    return true;
}
