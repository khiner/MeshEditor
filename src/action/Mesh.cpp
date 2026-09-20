#include "action/Mesh.h"

#include "gpu/MeshTopologyOp.h"

#include "TransformMath.h"
#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/PrimitiveType.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"
#include "numeric/VectorMath.h"
#include "object/ObjectOps.h"
#include "project/Project.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/MeshBuffers.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

#include <format>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <unordered_map>

namespace {
// The edit-mode meshes with a selection in the viewport's edit element domain.
std::vector<state::Entity> SelectedEditMeshes(const state::Scene &r, state::Entity viewport) {
    std::vector<state::Entity> result;
    const auto element = r.get<const EditMode>(viewport).Value;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : r.view<const MeshElementSelection>()) {
        if (!HasMesh(r, e)) continue;
        const auto &summary = meshes.GetSelectionSummary(r.get<const MeshHandle>(e).StoreId);
        if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(e);
    }
    return result;
}

// Runs the tasks and replaces each entity's mesh with its output, or draws the output as a preview while the session previews.
void RunTasks(state::Scene &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    const auto outputs = RunMeshTopology(r, tasks);
    const bool preview = project::Session(r).Previewing;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (outputs[i] == InvalidStoreId) continue;
        const auto e = mesh_entities[i];
        if (preview) {
            // The base handle stays for the commit to adopt or the restore to return to.
            r.remove<PrimitiveShape, MeshActiveElement>(e);
            r.emplace_or_replace<MeshPreview>(e, MeshPreview{outputs[i]});
        } else {
            // Releasing the handle frees the source record, and the new handle takes the entity through the new-mesh path.
            r.remove<MeshHandle, PrimitiveShape, MeshActiveElement>(e);
            r.emplace<MeshHandle>(e, MeshHandle{outputs[i]});
        }
        r.emplace_or_replace<MeshGeometryDirty>(e, EditSelectionAfter::Derive);
    }
}

// Runs the task `make` builds for each mesh, skipping the meshes it returns nothing for.
void RunPerMesh(state::Scene &r, std::span<const state::Entity> mesh_entities, auto &&make) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshTopologyTask> tasks;
    std::vector<state::Entity> entities;
    for (const auto e : mesh_entities) {
        if (std::optional<MeshTopologyTask> task = make(e, Mesh{meshes, r.get<const MeshHandle>(e).StoreId})) {
            tasks.push_back(std::move(*task));
            entities.push_back(e);
        }
    }
    RunTasks(r, entities, tasks);
}

void RunOperator(state::Scene &r, std::span<const state::Entity> mesh_entities, MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) { return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags}; });
}

// The lowest and highest selected vertex of a mesh, from its vertex mask.
std::pair<uint32_t, uint32_t> SelectedVertexSpan(const MeshStore &meshes, uint32_t id) {
    const auto bits = meshes.GetSelectionBits(id, Element::Vertex);
    uint32_t first = InvalidOffset, last = InvalidOffset;
    for (uint32_t w = 0; w < bits.size(); ++w) {
        if (!bits[w]) continue;
        if (first == InvalidOffset) first = w * 32 + std::countr_zero(bits[w]);
        last = w * 32 + 31 - std::countl_zero(bits[w]);
    }
    return {first, last};
}

// Moves the selected faces of each mesh into a new mesh object placed over the source's primary instance.
void SeparateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    std::vector<MeshTopologyTask> tasks;
    for (const auto e : mesh_entities) tasks.push_back({.SourceId = r.get<const MeshHandle>(e).StoreId, .Op = MeshTopologyOp::KeepSelectedFaces});
    const auto outputs = RunMeshTopology(r, tasks);
    const auto primaries = ::selection::ComputePrimaryEditInstances(r);
    std::vector<state::Entity> separated;
    for (size_t i = 0; i < mesh_entities.size(); ++i) {
        if (outputs[i] == InvalidStoreId) continue;
        const auto e = mesh_entities[i];
        const auto primary = primaries.find(e);
        const auto instance = primary != primaries.end() ? primary->second : state::Null;
        ::AddMesh(r, outputs[i], MeshInstanceCreateInfo{
                                     .Name = std::format("{}.001", instance != state::Null ? GetName(r, instance) : "Mesh"),
                                     .Transform = instance != state::Null ? Transform{r.get<const WorldTransform>(instance)} : Transform{},
                                     .Select = MeshInstanceCreateInfo::SelectBehavior::None,
                                 });
        separated.push_back(e);
    }
    RunOperator(r, separated, MeshTopologyOp::DeleteFaces);
}

// The lowest selected edge of a mesh, or the active one when the active element is an edge.
std::optional<uint32_t> ActiveOrFirstSelectedEdge(const state::Scene &r, state::Entity mesh_entity, const Mesh &mesh) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto bits = meshes.GetSelectionBits(mesh.GetStoreId(), Element::Edge);
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity); active && active->Handle < mesh.EdgeCount() && (bits[active->Handle / 32] >> (active->Handle % 32)) & 1u) return active->Handle;
    for (uint32_t w = 0; w < bits.size(); ++w) {
        if (bits[w]) return w * 32 + std::countr_zero(bits[w]);
    }
    return {};
}

// The ring of edges across quads from `edge`, walked both ways until a non-quad, a boundary, or the ring closes.
std::vector<uint32_t> EdgeRing(const Mesh &mesh, uint32_t edge) {
    std::vector<uint32_t> ring{edge};
    std::vector<uint8_t> visited(mesh.EdgeCount(), 0);
    visited[edge] = 1;
    const auto &c = mesh.GetConnectivity();
    const auto start = mesh.GetHalfedge(Mesh::EH{edge}, 0);
    for (const auto side : {start, c.Opposites[*start]}) {
        auto h = side;
        while (h) {
            const auto face = c.FaceOf(h);
            if (!face || mesh.GetValence(face) != 4) break;
            const auto across = c.Next(c.Next(h));
            const auto e = mesh.GetEdge(across);
            if (visited[*e]) break;
            visited[*e] = 1;
            ring.push_back(*e);
            h = c.Opposites[*across];
        }
    }
    return ring;
}

// Each closed loop of boundary edges, selected ones or all of them, as its vertices in the boundary's own direction.
std::vector<std::vector<uint32_t>> BoundaryChains(const MeshStore &meshes, const Mesh &mesh, bool selected_only) {
    const auto bits = meshes.GetSelectionBits(mesh.GetStoreId(), Element::Edge);
    const auto &c = mesh.GetConnectivity();
    // The boundary halfedge leaving each vertex, lowest first where a vertex has several.
    std::vector<uint32_t> leaving(mesh.VertexCount(), InvalidOffset);
    for (uint32_t e = 0; e < mesh.EdgeCount(); ++e) {
        if (selected_only && !((bits[e / 32] >> (e % 32)) & 1u)) continue;
        const auto h = mesh.GetHalfedge(Mesh::EH{e}, 0);
        if (c.Opposites[*h]) continue;
        const auto from = *mesh.GetFromVertex(h);
        leaving[from] = std::min(leaving[from], *h);
    }
    std::vector<std::vector<uint32_t>> loops;
    std::vector<uint8_t> used(mesh.HalfEdgeCount(), 0);
    for (uint32_t v = 0; v < leaving.size(); ++v) {
        if (leaving[v] == InvalidOffset || used[leaving[v]]) continue;
        std::vector<uint32_t> loop;
        auto h = leaving[v];
        bool closed = false;
        for (uint32_t step = 0; step <= mesh.HalfEdgeCount(); ++step) {
            if (used[h]) {
                closed = h == leaving[v];
                break;
            }
            used[h] = 1;
            loop.push_back(*mesh.GetFromVertex(Mesh::HH{h}));
            h = leaving[*mesh.GetToVertex(Mesh::HH{h})];
            if (h == InvalidOffset) break;
        }
        if (closed && loop.size() >= 3) loops.push_back(std::move(loop));
    }
    return loops;
}

// Each closed boundary loop as the vertex loop of the face that fills it, wound against the boundary.
std::vector<std::vector<uint32_t>> BoundaryLoops(const MeshStore &meshes, const Mesh &mesh, bool selected_only) {
    auto loops = BoundaryChains(meshes, mesh, selected_only);
    for (auto &loop : loops) std::ranges::reverse(loop);
    return loops;
}

// A face list task over `loops`, each a run of vertex indices.
// `positions` append as new vertices, which the loops name by indices past the source count.
MeshTopologyTask FaceListTask(uint32_t source, std::span<const std::vector<uint32_t>> loops, std::span<const vec3> positions = {}) {
    MeshTopologyTask task{.SourceId = source, .Op = MeshTopologyOp::AddFaces};
    task.List.push_back(uint32_t(positions.size()));
    for (const auto &p : positions) {
        task.List.push_back(std::bit_cast<uint32_t>(p.x));
        task.List.push_back(std::bit_cast<uint32_t>(p.y));
        task.List.push_back(std::bit_cast<uint32_t>(p.z));
    }
    task.List.push_back(uint32_t(loops.size()));
    for (const auto &loop : loops) {
        task.List.push_back(uint32_t(loop.size()));
        task.List.insert(task.List.end(), loop.begin(), loop.end());
    }
    return task;
}

// Bridges the two closed loops of selected boundary edges, pairing each vertex of the longer with its share of the shorter.
void BridgeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        auto chains = BoundaryChains(meshes, mesh, true);
        if (chains.size() != 2) return {};
        if (chains[0].size() < chains[1].size()) std::swap(chains[0], chains[1]);
        const auto &a = chains[0], &b = chains[1];
        // The strip runs along the longer loop and against the shorter one, starting at the shorter's nearest vertex.
        uint32_t start = 0;
        float best = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < b.size(); ++j) {
            if (const auto d = Distance2(mesh.GetPosition(Mesh::VH{a[0]}), mesh.GetPosition(Mesh::VH{b[j]})); d < best) {
                best = d;
                start = j;
            }
        }
        const auto na = uint32_t(a.size()), nb = uint32_t(b.size());
        const auto at_b = [&](uint32_t steps) { return b[(start + nb - steps % nb) % nb]; };
        std::vector<std::vector<uint32_t>> faces;
        for (uint32_t i = 0; i < na; ++i) {
            const uint32_t j0 = (i * nb) / na, j1 = ((i + 1) * nb) / na;
            if (j0 == j1) {
                faces.push_back({a[(i + 1) % na], a[i], at_b(j0)});
                continue;
            }
            faces.push_back({a[(i + 1) % na], a[i], at_b(j0), at_b(j0 + 1)});
            for (uint32_t j = j0 + 1; j < j1; ++j) faces.push_back({a[(i + 1) % na], at_b(j), at_b(j + 1)});
        }
        return FaceListTask(mesh.GetStoreId(), faces);
    });
}

// Fills one closed loop of selected boundary edges with a Coons patch of quads, `span` edges along its first side.
void GridFillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t span) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto chains = BoundaryChains(meshes, mesh, true);
        if (chains.size() != 1 || chains[0].size() % 2 != 0 || chains[0].size() < 4) return {};
        const auto &loop = chains[0];
        const auto length = uint32_t(loop.size());
        const uint32_t s = std::clamp(span == 0 ? std::max(length / 4, 1u) : span, 1u, length / 2 - 1), t = length / 2 - s;
        // Nodes run along the first side (u) and up the second (v), with the loop's four sides as the rails.
        const auto rail = [&](uint32_t k) { return mesh.GetPosition(Mesh::VH{loop[k % length]}); };
        std::vector<uint32_t> node((s + 1) * (t + 1), InvalidOffset);
        std::vector<vec3> positions;
        const auto index = [&](uint32_t i, uint32_t j) { return j * (s + 1) + i; };
        for (uint32_t i = 0; i <= s; ++i) {
            node[index(i, 0)] = loop[i];
            node[index(i, t)] = loop[(2 * s + t - i) % length];
        }
        for (uint32_t j = 0; j <= t; ++j) {
            node[index(s, j)] = loop[s + j];
            node[index(0, j)] = loop[(2 * s + 2 * t - j) % length];
        }
        for (uint32_t j = 1; j < t; ++j) {
            for (uint32_t i = 1; i < s; ++i) {
                const float u = float(i) / float(s), v = float(j) / float(t);
                const vec3 bottom = rail(i), top = rail(2 * s + t - i), right = rail(s + j), left = rail(2 * s + 2 * t - j);
                const vec3 p00 = rail(0), p10 = rail(s), p11 = rail(s + t), p01 = rail(2 * s + t);
                const vec3 p = bottom * (1.f - v) + top * v + left * (1.f - u) + right * u -
                    (p00 * ((1.f - u) * (1.f - v)) + p10 * (u * (1.f - v)) + p01 * ((1.f - u) * v) + p11 * (u * v));
                node[index(i, j)] = mesh.VertexCount() + uint32_t(positions.size());
                positions.push_back(p);
            }
        }
        std::vector<std::vector<uint32_t>> faces;
        for (uint32_t j = 0; j < t; ++j) {
            for (uint32_t i = 0; i < s; ++i) faces.push_back({node[index(i + 1, j)], node[index(i, j)], node[index(i, j + 1)], node[index(i + 1, j + 1)]});
        }
        return FaceListTask(mesh.GetStoreId(), faces, positions);
    });
}

void FillHolesSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t sides) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        auto loops = BoundaryLoops(meshes, mesh, false);
        std::erase_if(loops, [&](const auto &loop) { return sides > 0 && loop.size() > sides; });
        if (loops.empty()) return {};
        return FaceListTask(mesh.GetStoreId(), loops);
    });
}

// The convex hull of the selected vertices as outward triangles, by quickhull.
// Each face keeps the points outside it, and adding a face's farthest point replaces only the faces that point sees.
void ConvexHullSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        std::vector<uint32_t> points;
        ForEachSelected(meshes.GetSelectionBits(mesh.GetStoreId(), Element::Vertex), mesh.VertexCount(), [&](uint32_t v) { points.push_back(v); });
        if (points.size() < 4) return {};
        const auto at = [&](uint32_t v) { return mesh.GetPosition(Mesh::VH{v}); };
        // A starting tetrahedron from the first point, the farthest from it, the farthest from that line, and the farthest from that plane.
        std::array<uint32_t, 4> seed{points[0], points[0], points[0], points[0]};
        float best = 0.f;
        for (const auto v : points)
            if (const auto d = Distance2(at(v), at(seed[0])); d > best) {
                best = d;
                seed[1] = v;
            }
        const float extent = std::sqrt(best);
        best = 0.f;
        for (const auto v : points)
            if (const auto d = Length(Cross(at(seed[1]) - at(seed[0]), at(v) - at(seed[0]))); d > best) {
                best = d;
                seed[2] = v;
            }
        best = 0.f;
        const auto seed_normal = Cross(at(seed[1]) - at(seed[0]), at(seed[2]) - at(seed[0]));
        for (const auto v : points)
            if (const auto d = std::abs(Dot(seed_normal, at(v) - at(seed[0]))); d > best) {
                best = d;
                seed[3] = v;
            }
        if (best < 1e-12f) return {};

        struct Face {
            std::array<uint32_t, 3> V;
            // The face across each edge V[k] to V[k + 1].
            std::array<uint32_t, 3> Across{};
            vec3 Normal;
            float Offset, Tolerance;
            std::vector<uint32_t> Outside;
            bool Alive{true};
        };
        std::vector<Face> faces;
        // A point counts as outside a face beyond a sliver of the point set's extent, so coplanar points stay inside.
        const float tolerance = 1e-6f * extent;
        const auto make = [&](std::array<uint32_t, 3> v) {
            Face face{.V = v, .Normal = Cross(at(v[1]) - at(v[0]), at(v[2]) - at(v[0]))};
            face.Offset = Dot(face.Normal, at(v[0]));
            face.Tolerance = Length(face.Normal) * tolerance;
            return face;
        };
        const auto height = [&](const Face &face, uint32_t p) { return Dot(face.Normal, at(p)) - face.Offset; };
        const auto outside = [&](const Face &face, uint32_t p) { return height(face, p) > face.Tolerance; };
        const auto outward = [&](std::array<uint32_t, 3> tri, uint32_t inside) {
            const auto n = Cross(at(tri[1]) - at(tri[0]), at(tri[2]) - at(tri[0]));
            return Dot(n, at(inside) - at(tri[0])) > 0.f ? std::array{tri[0], tri[2], tri[1]} : tri;
        };
        // The slot of the edge `from` to `to` on a face, or three when the face lacks it.
        const auto edge_slot = [](const Face &face, uint32_t from, uint32_t to) {
            uint32_t j = 0;
            while (j < 3 && !(face.V[j] == from && face.V[(j + 1) % 3] == to)) ++j;
            return j;
        };
        // A point waits on the first face from `first` that sees it, or falls inside.
        const auto assign = [&](uint32_t v, uint32_t first) {
            for (uint32_t i = first; i < faces.size(); ++i) {
                if (outside(faces[i], v)) {
                    faces[i].Outside.push_back(v);
                    return;
                }
            }
        };
        std::vector<uint32_t> pending;
        const auto enqueue = [&](uint32_t first) {
            for (uint32_t i = first; i < faces.size(); ++i)
                if (!faces[i].Outside.empty()) pending.push_back(i);
        };
        faces.push_back(make(outward({seed[0], seed[1], seed[2]}, seed[3])));
        faces.push_back(make(outward({seed[0], seed[1], seed[3]}, seed[2])));
        faces.push_back(make(outward({seed[0], seed[2], seed[3]}, seed[1])));
        faces.push_back(make(outward({seed[1], seed[2], seed[3]}, seed[0])));
        // The tetrahedron's faces meet across each shared edge, in opposite directions.
        for (uint32_t a = 0; a < 4; ++a) {
            for (uint32_t k = 0; k < 3; ++k) {
                for (uint32_t b = 0; b < 4; ++b) {
                    if (edge_slot(faces[b], faces[a].V[(k + 1) % 3], faces[a].V[k]) < 3) faces[a].Across[k] = b;
                }
            }
        }
        for (const auto v : points)
            if (std::ranges::find(seed, v) == seed.end()) assign(v, 0);
        enqueue(0);

        struct HorizonEdge {
            uint32_t From, To, Neighbor, NeighborEdge;
        };
        std::vector<uint32_t> visible, stack;
        std::vector<HorizonEdge> horizon;
        // The search that last reached each face.
        std::vector<uint32_t> visited(faces.size(), 0u);
        uint32_t search = 0;
        while (!pending.empty()) {
            const auto start = pending.back();
            pending.pop_back();
            if (!faces[start].Alive || faces[start].Outside.empty()) continue;
            const auto p = *std::ranges::max_element(faces[start].Outside, {}, [&](uint32_t v) { return height(faces[start], v); });
            // The faces the point sees form one connected region, whose boundary edges are the horizon.
            visible.clear();
            horizon.clear();
            ++search;
            stack.assign(1, start);
            visited[start] = search;
            while (!stack.empty()) {
                const auto f = stack.back();
                stack.pop_back();
                visible.push_back(f);
                for (uint32_t k = 0; k < 3; ++k) {
                    const auto n = faces[f].Across[k];
                    const bool sees = outside(faces[n], p);
                    if (!sees) horizon.push_back({faces[f].V[k], faces[f].V[(k + 1) % 3], n, 0});
                    if (visited[n] == search) continue;
                    visited[n] = search;
                    if (sees) stack.push_back(n);
                }
            }
            for (auto &edge : horizon) edge.NeighborEdge = edge_slot(faces[edge.Neighbor], edge.To, edge.From);
            // Each horizon edge fans to the point, and consecutive fans meet along the point's spokes.
            const uint32_t first_new = uint32_t(faces.size());
            std::unordered_map<uint32_t, uint32_t> fan_from, fan_to;
            for (uint32_t i = 0; i < horizon.size(); ++i) {
                const auto &edge = horizon[i];
                faces.push_back(make({edge.From, edge.To, p}));
                fan_from[edge.From] = first_new + i;
                fan_to[edge.To] = first_new + i;
            }
            // A horizon that is not one simple loop means the tolerance split a nearly coplanar region, and the hull is abandoned.
            if (fan_from.size() != horizon.size() || fan_to.size() != horizon.size()) return {};
            visited.resize(faces.size(), 0u);
            for (uint32_t i = 0; i < horizon.size(); ++i) {
                const auto &edge = horizon[i];
                auto &face = faces[first_new + i];
                face.Across = {edge.Neighbor, fan_from.at(edge.To), fan_to.at(edge.From)};
                faces[edge.Neighbor].Across[edge.NeighborEdge] = first_new + i;
            }
            // The visible faces' outside points move to the new faces.
            for (const auto f : visible) {
                auto &face = faces[f];
                face.Alive = false;
                for (const auto v : face.Outside)
                    if (v != p) assign(v, first_new);
                face.Outside.clear();
            }
            enqueue(first_new);
        }
        std::vector<std::vector<uint32_t>> hull;
        for (const auto &face : faces)
            if (face.Alive) hull.push_back({face.V[0], face.V[1], face.V[2]});
        if (hull.empty()) return {};
        return FaceListTask(mesh.GetStoreId(), hull);
    });
}

// Rotates each selected edge with two faces: dissolves it, then connects the vertices following its ends around the joined face.
// Dissolves each selected edge and connects the far vertices of its two faces, which keep their numbering through the dissolve.
void EdgeRotateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshTopologyTask> connects;
    std::vector<state::Entity> entities;
    for (const auto e : mesh_entities) {
        const auto id = r.get<const MeshHandle>(e).StoreId;
        const Mesh mesh{meshes, id};
        const auto &c = mesh.GetConnectivity();
        MeshTopologyTask connect{.SourceId = id, .Op = MeshTopologyOp::ConnectVertices, .Flags = TopologyFlagListSelects, .List = {0}};
        ForEachSelected(meshes.GetSelectionBits(id, Element::Edge), mesh.EdgeCount(), [&](uint32_t edge) {
            const auto h = mesh.GetHalfedge(Mesh::EH{edge}, 0);
            const auto opposite = c.Opposites[*h];
            if (!opposite) return;
            connect.List.push_back(*mesh.GetToVertex(c.Next(h)));
            connect.List.push_back(*mesh.GetToVertex(c.Next(opposite)));
            connect.List[0] += 2;
        });
        if (connect.List[0] == 0) continue;
        connects.push_back(std::move(connect));
        entities.push_back(e);
    }
    if (entities.empty()) return;
    RunOperator(r, entities, MeshTopologyOp::DissolveEdges, 0.f, 0.f, TopologyFlagKeepVertices);
    for (size_t i = 0; i < entities.size(); ++i) connects[i].SourceId = r.get<const MeshHandle>(entities[i]).StoreId;
    RunTasks(r, entities, connects);
}

void FillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto loops = BoundaryLoops(meshes, mesh, true);
        if (loops.empty()) return {};
        return FaceListTask(mesh.GetStoreId(), loops);
    });
}

// Subdivides the ring through each mesh's active edge.
void LoopCutSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t cuts) {
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto edge = ActiveOrFirstSelectedEdge(r, e, mesh);
        if (!edge) return {};
        MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = float(std::clamp(cuts, 1u, 32u)), .Flags = TopologyFlagLoopCutSelect | TopologyFlagListSelects, .List = EdgeRing(mesh, *edge)};
        task.List.insert(task.List.begin(), uint32_t(task.List.size()));
        return task;
    });
}

// Extrudes the selection in `steps` layers, each moved through the step's transform once more than the last.
void ExtrudeSteps(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t steps, const mat3 &rotation, vec3 translation, vec3 center) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{
            .SourceId = mesh.GetStoreId(),
            .Op = MeshTopologyOp::ExtrudeRegion,
            .Flags = TopologyFlagTransformCopies,
            .Steps = std::clamp(steps, 1u, 256u),
            .CopyRotation = rotation,
            // Rotating about a center is a rotation about the origin followed by the center's own displacement.
            .CopyTranslation = center - rotation * center + translation,
        };
    });
}

// Cuts each mesh along a plane in its own space, then deletes the faces on the cleared sides.
void BisectSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, vec3 point, vec3 normal, bool clear_inner, bool clear_outer) {
    const auto n = Normalize(normal);
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagPlaneCuts | TopologyFlagLoopCutSelect, .PlaneNormal = n, .PlaneOffset = Dot(n, point)};
    });
    for (const bool inner : {true, false}) {
        if (inner ? !clear_inner : !clear_outer) continue;
        const auto side = inner ? n : -n;
        RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
            return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DeleteFaces, .Flags = TopologyFlagPlaneSide, .PlaneNormal = side, .PlaneOffset = Dot(side, point)};
        });
    }
}

// Mirrors the kept side across the mesh origin and welds the vertices on the plane.
void SymmetrizeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint8_t axis, bool negative) {
    vec3 normal{0.f};
    normal[axis % 3] = negative ? -1.f : 1.f;
    BisectSelected(r, mesh_entities, vec3{0.f}, normal, true, false);
    mat3 mirror{};
    for (int i = 0; i < 3; ++i) mirror[i][i] = i == axis % 3 ? -1.f : 1.f;
    // The whole kept side duplicates, and the plane's vertices weld afterward.
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DuplicateFaces, .Flags = TopologyFlagTransformCopies | TopologyFlagFlipCopies | TopologyFlagSelectAll, .CopyRotation = mirror};
    });
    RunOperator(r, mesh_entities, MeshTopologyOp::MergeByDistance, 1e-5f, 0.f, TopologyFlagSelectAll);
}

// Cuts every edge whose screen segment crosses the knife segment, at the crossing.
void KnifeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, vec2 start, vec2 end, const RenderView &view) {
    const auto primaries = ::selection::ComputePrimaryEditInstances(r);
    const float aspect = view.Extent.y > 0.f ? view.Extent.x / view.Extent.y : 1.f;
    const auto view_projection = view.Camera.Projection(aspect) * view.Camera.View();
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto primary = primaries.find(e);
        if (primary == primaries.end()) return {};
        return MeshTopologyTask{
            .SourceId = mesh.GetStoreId(),
            .Op = MeshTopologyOp::Subdivide,
            .Param0 = 1.f,
            .Flags = TopologyFlagScreenCuts | TopologyFlagLoopCutSelect,
            .ScreenTransform = view_projection * ToMatrix(Transform{r.get<const WorldTransform>(primary->second)}),
            .Extent = view.Extent,
            .KnifeStart = start,
            .KnifeEnd = end,
        };
    });
}

void MergeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, action::mesh::MergeMode mode, float distance) {
    using Mode = action::mesh::MergeMode;
    if (mode == Mode::Collapse) return RunOperator(r, mesh_entities, MeshTopologyOp::MergeCollapse);
    if (mode == Mode::ByDistance) return RunOperator(r, mesh_entities, MeshTopologyOp::MergeByDistance, distance);
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto id = mesh.GetStoreId();
        const auto [first, last] = SelectedVertexSpan(meshes, id);
        if (first == InvalidOffset || first == last) return {};
        const auto &summary = meshes.GetSelectionSummary(id);
        const auto target = mode == Mode::Last ? last : first;
        const vec3 position = mode == Mode::Center ? summary.PositionSum / float(std::max(summary.SelectedVertexCount, 1u)) : mesh.GetPosition(Mesh::VH{target});
        return MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::MergeAtTarget, .TargetVertex = target, .TargetPosition = position};
    });
}
} // namespace

namespace action::mesh {
void CommitPreviews(state::Scene &r) {
    std::vector<std::pair<state::Entity, uint32_t>> previews;
    for (const auto [e, preview] : r.view<const MeshPreview>().each()) previews.emplace_back(e, preview.StoreId);
    for (const auto [e, id] : previews) {
        r.remove<MeshPreview>(e);
        // Releasing the handle frees the base record, and the preview's render data stays with its record.
        r.remove<MeshHandle>(e);
        r.emplace<MeshHandle>(e, MeshHandle{id});
    }
}

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    // A restart has restored the base already, so any preview still present belongs to an earlier operator and becomes the source.
    CommitPreviews(r);
    const auto targets = SelectedEditMeshes(r, viewport);
    const auto latch_translate = [&] { r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate); };
    std::visit(
        overloaded{
            [&](const Delete &a) { RunOperator(r, targets, MeshTopologyOp(uint32_t(a.Mode))); },
            [&](const Merge &a) { MergeSelected(r, targets, a.Mode, std::max(a.Distance, 0.f)); },
            [&](const Extrude &a) {
                using Mode = ExtrudeMode;
                const auto op = a.Mode == Mode::Edges ? MeshTopologyOp::ExtrudeEdges : a.Mode == Mode::FacesIndividual ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                         MeshTopologyOp::ExtrudeRegion;
                RunOperator(r, targets, op);
                latch_translate();
            },
            [&](Duplicate) {
                RunOperator(r, targets, MeshTopologyOp::DuplicateFaces);
                latch_translate();
            },
            [&](Split) { RunOperator(r, targets, MeshTopologyOp::SplitFaces); },
            [&](Separate) { SeparateSelected(r, targets); },
            [&](const Subdivide &a) { RunOperator(r, targets, MeshTopologyOp::Subdivide, float(std::clamp(a.Cuts, 1u, 32u))); },
            [&](Triangulate) { RunOperator(r, targets, MeshTopologyOp::Triangulate); },
            [&](TrisToQuads) { RunOperator(r, targets, MeshTopologyOp::TrisToQuads); },
            [&](const Poke &a) { RunOperator(r, targets, MeshTopologyOp::Poke, a.Offset); },
            [&](FlipNormals) { RunOperator(r, targets, MeshTopologyOp::FlipNormals); },
            [&](EdgeSplit) { RunOperator(r, targets, MeshTopologyOp::EdgeSplit); },
            [&](const Inset &a) { RunOperator(r, targets, a.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, std::max(a.Thickness, 0.f), a.Depth, a.Even ? 1u : 0u); },
            [&](Fill) { FillSelected(r, targets); },
            [&](const LoopCut &a) { LoopCutSelected(r, targets, a.Cuts); },
            [&](const Spin &a) {
                if (Dot(a.Axis, a.Axis) <= 0.f) return;
                const auto axis = Normalize(a.Axis);
                ExtrudeSteps(r, targets, a.Steps, ToMat3(AngleAxis(a.Angle / float(std::max(a.Steps, 1u)), axis)), axis * (a.Offset / float(std::max(a.Steps, 1u))), a.Center);
            },
            [&](const ExtrudeRepeat &a) { ExtrudeSteps(r, targets, a.Steps, mat3{1.f}, a.Offset, vec3{0.f}); },
            [&](const Bisect &a) {
                if (Dot(a.Normal, a.Normal) <= 0.f) return;
                BisectSelected(r, targets, a.Point, a.Normal, a.ClearInner, a.ClearOuter);
            },
            [&](const Symmetrize &a) { SymmetrizeSelected(r, targets, uint8_t(a.Axis), a.Negative); },
            [&](const Solidify &a) { RunOperator(r, targets, MeshTopologyOp::Solidify, a.Thickness); },
            [&](ConnectVertices) { RunOperator(r, targets, MeshTopologyOp::ConnectVertices); },
            [&](const Knife &a) { KnifeSelected(r, targets, a.Start, a.End, *a.View); },
            [&](BridgeEdgeLoops) { BridgeSelected(r, targets); },
            [&](const GridFill &a) { GridFillSelected(r, targets, a.Span); },
            [&](const FillHoles &a) { FillHolesSelected(r, targets, a.Sides); },
            [&](ConvexHull) { ConvexHullSelected(r, targets); },
            [&](EdgeRotate) { EdgeRotateSelected(r, targets); },
            [&](const Bevel &a) { RunOperator(r, targets, a.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, std::max(a.Width, 0.f), float(std::clamp(a.Segments, 1u, 16u))); },
            [&](Rip) {
                RunOperator(r, targets, MeshTopologyOp::EdgeSplit, 0.f, 0.f, TopologyFlagRipSelectCopies);
                latch_translate();
            },
            [&](const Dissolve &a) {
                using Mode = DissolveMode;
                switch (a.Mode) {
                    case Mode::Vertices: return RunOperator(r, targets, MeshTopologyOp::DissolveVertices);
                    case Mode::Edges: return RunOperator(r, targets, MeshTopologyOp::DissolveEdges);
                    case Mode::Faces: return RunOperator(r, targets, MeshTopologyOp::DissolveFaces);
                    case Mode::Limited: return RunOperator(r, targets, MeshTopologyOp::DissolveLimited, std::clamp(a.Angle, 0.f, std::numbers::pi_v<float>));
                    case Mode::Degenerate: return RunOperator(r, targets, MeshTopologyOp::DissolveDegenerate, std::max(a.Distance, 0.f));
                }
            },
        },
        action
    );
}
} // namespace action::mesh
