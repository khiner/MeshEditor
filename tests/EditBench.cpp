#include "Paths.h"
#include "Profile.h"
#include "TestMeshOrdinals.h"
#include "TestPaths.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "action/Object.h"
#include "action/View.h"
#include "editor/Engine.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/SpatialFaceWork.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/Shader.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportRenderGpu.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
double Milliseconds(auto &&run) {
    const auto start = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

using Result = std::expected<void, std::string>;

double P95(std::span<double> samples) {
    std::ranges::sort(samples);
    return samples[size_t(std::ceil(0.95 * double(samples.size()))) - 1u];
}

void ReportLibraries(const mtl::LibraryCache &cache, const char *phase) {
    std::printf("pipeline_archive,%u,%u,%s\n", cache.ArchiveHitCount(), cache.CompileMissCount(), phase);
    std::printf("shader_libraries,%u,%u,%s\n", cache.BinaryLibraryLoadCount(), cache.SourceLibraryCompileCount(), phase);
}

Result Capture(state::Scene &r, const char *variable) {
    const char *path = std::getenv(variable);
    if (!path) return {};
    const auto &image = r.Context.get<const RenderTargets>().Resources->FinalColorImage;
    const auto rgba = ReadbackImageRgba8(r.Context.get<const mtl::Context>(), image, 0u, 0u, image.Extent);
    const auto file = std::unique_ptr<FILE, decltype(&std::fclose)>{std::fopen(path, "wb"), &std::fclose};
    if (!file) return std::unexpected{"failed to open frame capture"};
    if (std::fwrite(rgba.data(), 1u, rgba.size(), file.get()) != rgba.size()) return std::unexpected{"failed to write frame capture"};
    return {};
}

enum class SingleEdit { None,
                        SeparateSelected,
                        SpatialPlane,
                        SpatialCut,
                        EdgeSplit,
                        BevelEdge,
                        BevelVertex,
                        DissolveEdge,
                        EdgeRotate,
                        ExtrudeEdge,
                        SubdivideEdge,
                        LoopCut,
                        FillFace,
                        FillHoles,
                        DeleteEdge,
                        DeleteOnlyEdgeFaces,
                        DeleteVertex,
                        DissolveVertex,
                        DissolveFace,
                        DissolveLimited,
                        SharpFace,
                        MergeDistance,
                        MergeCenter,
                        MergeCollapse,
                        DissolveDegenerate };
struct EditSpec {
    SingleEdit Op;
    const char *Argument, *Label;
};
constexpr std::array EditSpecs{
    EditSpec{SingleEdit::None, "inset", "inset"},
    EditSpec{SingleEdit::SeparateSelected, "separate-selected", "separate_selected"},
    EditSpec{SingleEdit::SpatialPlane, "spatial-plane", "spatial_plane"},
    EditSpec{SingleEdit::SpatialCut, "spatial-cut", "spatial_cut"},
    EditSpec{SingleEdit::EdgeSplit, "edge-split", "edge_split"},
    EditSpec{SingleEdit::BevelEdge, "bevel-edge", "bevel_edge"},
    EditSpec{SingleEdit::BevelVertex, "bevel-vertex", "bevel_vertex"},
    EditSpec{SingleEdit::DissolveEdge, "dissolve-edge", "dissolve_edge"},
    EditSpec{SingleEdit::EdgeRotate, "edge-rotate", "edge_rotate"},
    EditSpec{SingleEdit::ExtrudeEdge, "extrude-edge", "extrude_edge"},
    EditSpec{SingleEdit::SubdivideEdge, "subdivide-edge", "subdivide_edge"},
    EditSpec{SingleEdit::LoopCut, "loop-cut", "loop_cut"},
    EditSpec{SingleEdit::FillFace, "fill-face", "fill_face"},
    EditSpec{SingleEdit::FillHoles, "fill-holes", "fill_holes"},
    EditSpec{SingleEdit::DeleteEdge, "delete-edge", "delete_edge"},
    EditSpec{SingleEdit::DeleteOnlyEdgeFaces, "delete-only-edge-faces", "delete_only_edge_faces"},
    EditSpec{SingleEdit::DeleteVertex, "delete-vertex", "delete_vertex"},
    EditSpec{SingleEdit::DissolveVertex, "dissolve-vertex", "dissolve_vertex"},
    EditSpec{SingleEdit::DissolveFace, "dissolve-face", "dissolve_face"},
    EditSpec{SingleEdit::DissolveLimited, "dissolve-limited", "dissolve_limited"},
    EditSpec{SingleEdit::SharpFace, "sharp-face", "sharp_face"},
    EditSpec{SingleEdit::MergeDistance, "merge-distance", "merge_distance"},
    EditSpec{SingleEdit::MergeCenter, "merge-center", "merge_center"},
    EditSpec{SingleEdit::MergeCollapse, "merge-collapse", "merge_collapse"},
    EditSpec{SingleEdit::DissolveDegenerate, "dissolve-degenerate", "dissolve_degenerate"},
};
// The selected face and its valence stay fixed while the unselected mesh grows.
// Select-all replaces the operator's selection with every element of its domain, and inset then insets each face individually.
// Timings include the production project/event/history path.
// Audits and rendering are separate.
Result Bench(uint32_t slices, const std::filesystem::path &scene, uint32_t updates, bool render, bool refit_probe, uint32_t viewport_width, uint32_t viewport_height, bool position, bool join, const EditSpec &edit, bool select_all) {
    const auto single_edit = edit.Op;
    if (uint32_t(position) + uint32_t(join) + uint32_t(single_edit != SingleEdit::None) > 1u) return std::unexpected{"choose one benchmark operation"};
    const bool filling = single_edit == SingleEdit::FillFace || single_edit == SingleEdit::FillHoles;
    const bool merging = single_edit == SingleEdit::MergeDistance || single_edit == SingleEdit::MergeCenter ||
        single_edit == SingleEdit::MergeCollapse || single_edit == SingleEdit::DissolveDegenerate;
    if (select_all && (join || filling || merging)) return std::unexpected{"join, fill and merge benchmarks count a paired or filled selection"};
    const TestDir dir{"mesheditor-edit-bench"};
    Paths::Init(MESHEDITOR_BUILD_DIR, dir.Path);
    Engine engine{false};
    auto &r = engine.R;
    auto &p = *engine.P;
    const auto result = [&]() -> Result {
        profile::Enabled = true;
        profile::Init(r.Context.get<const mtl::Context>());
        profile::Enabled = false;
        if (!p.Begin(dir)) return std::unexpected{"project begin failed"};
        r.Context.get<ViewportExtent>().Value = {viewport_width, viewport_height};
        p.Settle();
        std::fprintf(stderr, "Preparing benchmark scene...\n");
        const auto load_begin = std::chrono::steady_clock::now();
        if (scene.empty()) p.Do(action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
        else {
            p.Do(action::MakeAction(action::io::LoadGltf{scene}));
            state::Entity largest = state::Null;
            uint32_t count = 0u;
            for (const auto [object, instance] : r.view<const Instance>().each()) {
                if (!r.all_of<MeshHandle>(instance.Entity)) continue;
                const auto mesh = GetMesh(r, instance.Entity);
                if (mesh.FaceCount() > count) {
                    largest = object;
                    count = mesh.FaceCount();
                }
            }
            if (largest == state::Null) return std::unexpected{"scene contains no face mesh"};
            p.Do(action::MakeAction(action::selection::Select{largest}));
        }
        std::fprintf(stderr, "Scene prepared in %.3f s; entering Edit mode...\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin).count());
        profile::ClearStats();
        profile::Enabled = true;
        const auto enter_ms = Milliseconds([&] { p.Do(action::MakeAction(action::view::SetInteractionMode{InteractionMode::Edit})); });
        profile::ReportCpuPhase("Edit mode entry");
        profile::Report();
        profile::ClearStats();
        std::fprintf(stderr, "Entered Edit mode in %.3f ms; switching to face selection...\n", enter_ms);
        const auto mode_ms = Milliseconds([&] { p.Do(action::MakeAction(action::view::SetEditMode{.Mode = Element::Face})); });
        profile::ReportCpuPhase("Face selection mode");
        profile::Enabled = false;
        std::fprintf(stderr, "Face selection mode prepared in %.3f ms.\n", mode_ms);
        const auto entity = GetActiveMeshEntity(r);
        auto &meshes = r.Context.get<MeshStore>();
        const auto original = GetMesh(r, entity);
        const auto faces = original.FaceCount(), vertices = original.VertexCount();
        const auto find_face = [&]() -> std::optional<he::FH> {
            if (filling) {
                const auto boundary = meshes.GetBoundaryEdges(original.GetStoreId());
                const auto incidence = original.GetVertexEdgeIncidence();
                for (uint32_t delta = 0u; delta < std::min(faces, 4096u); ++delta) {
                    const auto candidate = original.FaceAt((faces / 2u + delta) % faces);
                    if (std::ranges::all_of(original.fh_range(candidate), [&](auto h) {
                            if (!original.GetOppositeHalfedge(h)) return false;
                            if (single_edit != SingleEdit::FillHoles) return true;
                            for (const auto edge : incidence.Incident(*original.GetToVertex(h)))
                                if (boundary.Contains(edge)) return false;
                            return true;
                        })) return candidate;
                }
                return std::nullopt;
            }
            return original.FaceAt(faces / 2u);
        }();
        if (!find_face) return std::unexpected{"benchmark could not find an interior face to fill"};
        const auto face = *find_face;
        if (const char *distance_text = std::getenv("MESHEDITOR_EDIT_BENCH_FACE_CAMERA")) {
            const float distance = std::strtof(distance_text, nullptr);
            if (!(std::isfinite(distance) && distance > 0.f)) return std::unexpected{"invalid face camera distance"};
            const vec3 center = original.CalcFaceCentroid(face);
            const auto lens = r.get<ViewCamera>(engine.Viewport).Data;
            ClearLookThrough(r, engine.Viewport);
            const auto eye = center + vec3{0.2f * distance, distance, 0.5f * distance};
            r.replace<ViewCamera>(engine.Viewport, ViewCamera{eye, center, lens});
        }
        const auto added = original.GetValence(face);
        const auto selected_face_halfedge = *original.fh_range(face).begin();
        std::vector<uint32_t> fill_edges;
        if (filling) {
            for (const auto h : original.fh_range(face)) fill_edges.push_back(*original.GetEdge(h));
        }
        std::vector<uint32_t> selected{original.FaceOrdinal(face)};
        const bool pair_faces = join || single_edit == SingleEdit::DissolveFace || single_edit == SingleEdit::DissolveLimited;
        if (pair_faces) {
            // The generated terrain emits each grid quad as consecutive triangles.
            const auto mate = original.FaceOrdinal(face) ^ 1u;
            for (const auto halfedge : original.fh_range(face)) {
                const auto opposite = original.GetOppositeHalfedge(halfedge);
                if (!opposite) continue;
                if (join && (original.GetValence(face) != 3u || original.GetValence(original.GetFace(opposite)) != 3u || original.FaceOrdinal(original.GetFace(opposite)) != mate)) continue;
                selected.push_back(original.FaceOrdinal(original.GetFace(opposite)));
                break;
            }
            if (selected.size() != 2u) return std::unexpected{"paired-face benchmark requires adjacent faces"};
        }
        const std::array<uint32_t, 1> alternate{(selected[0] + (pair_faces ? 17u : 1u)) % faces};
        std::fprintf(stderr, "Selecting %zu face(s) from %u vertices / %u faces...\n", selected.size(), vertices, faces);
        profile::ClearStats();
        profile::Enabled = true;
        const auto select = [&](std::span<const uint32_t> ordinals, Element element = Element::Face) {
            ApplyEditSelectionLists(r, std::array{std::pair{entity, ordinals}}, element);
            p.Settle();
        };
        const auto initial_select_ms = Milliseconds([&] { select(selected); });
        profile::ReportCpuPhase("Initial selection replacement");
        profile::Report();
        profile::Enabled = false;
        if (meshes.GetSelectionSummary(original.GetStoreId()).SelectedCount != selected.size()) return std::unexpected{"expected selected faces"};
        select(alternate);
        const auto select_ms = Milliseconds([&] { select(selected); });
        if (meshes.GetSelectionSummary(original.GetStoreId()).SelectedCount != selected.size()) return std::unexpected{"expected selected faces after sparse selection"};
        const bool face_edit = single_edit == SingleEdit::SeparateSelected || single_edit == SingleEdit::SpatialPlane ||
            single_edit == SingleEdit::SpatialCut ||
            single_edit == SingleEdit::DissolveFace || single_edit == SingleEdit::DissolveLimited ||
            single_edit == SingleEdit::SharpFace;
        if (filling) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces}));
        if (filling) {
            const auto boundary = meshes.GetBoundaryEdges(original.GetStoreId());
            uint32_t on_hole = 0u;
            for (const auto edge : fill_edges) on_hole += boundary.Contains(edge);
            std::fprintf(stderr, "Hole boundary index: %u edges total, %u/%zu hole edges indexed.\n", boundary.Count(), on_hole, fill_edges.size());
        }
        if (position || (single_edit != SingleEdit::None && !face_edit)) {
            const bool use_vertex = position || single_edit == SingleEdit::BevelVertex ||
                single_edit == SingleEdit::DeleteVertex || single_edit == SingleEdit::DissolveVertex || merging;
            const auto element = use_vertex ? Element::Vertex : Element::Edge;
            p.Do(action::MakeAction(action::view::SetEditMode{.Mode = element}));
            const auto current = GetMesh(r, entity);
            std::vector<uint32_t> selected_elements;
            if (filling) {
                for (const auto handle : fill_edges) selected_elements.push_back(test::EdgeOrdinal(current, he::EH{handle}));
            } else {
                const auto first = use_vertex ? original.VertexOrdinal(original.GetToVertex(selected_face_halfedge)) :
                                                test::EdgeOrdinal(original, original.GetEdge(selected_face_halfedge));
                selected_elements.push_back(first);
            }
            if (single_edit == SingleEdit::EdgeSplit) {
                const auto next = original.GetConnectivity().Next(selected_face_halfedge);
                selected_elements.push_back(test::EdgeOrdinal(original, original.GetEdge(next)));
            }
            if (merging) selected_elements.push_back(original.VertexOrdinal(original.GetFromVertex(selected_face_halfedge)));
            select(selected_elements, element);
            const auto selected_count = meshes.GetSelectionSummary(original.GetStoreId()).SelectedCount;
            if (selected_count != selected_elements.size()) std::fprintf(stderr, "operator selection: expected %zu, got %u\n", selected_elements.size(), selected_count);
            if (selected_count != selected_elements.size()) return std::unexpected{"unexpected operator selection"};
            if (filling) {
                const auto selected_edges = meshes.GetSelectedElements(original.GetStoreId(), Element::Edge);
                for (const auto edge : fill_edges)
                    if (!selected_edges.Contains(edge)) return std::unexpected{"fill did not select a hole edge"};
            }
        }
        if (select_all) {
            p.Do(action::MakeAction(action::selection::SelectAll{}));
            p.Settle();
        }
        const auto selected_count = meshes.GetSelectionSummary(original.GetStoreId()).SelectedCount;
        if (select_all && (selected_count != GetMesh(r, entity).ElementCount(r.get<const EditMode>(engine.Viewport).Value))) return std::unexpected{"select-all left elements unselected"};
        // Each face of an individual inset gains one face and one vertex per corner.
        const auto inset_added = select_all ? original.HalfEdgeCount() : added;
        std::fprintf(stderr, "Rendering the baseline...\n");
        SubmitViewport(r, engine.Viewport);
        WaitForRender(r);
        // Include the renderer's first meshlet and LOD publication in the
        // baseline used for replay, as well as the edit selection.
        p.History.Commit("benchmark selection", {});
        const auto base = *p.History.Present;
        const auto allocated = [&] { return uint64_t(r.Context.get<const mtl::Context>().Device->currentAllocatedSize()); };
        const auto row = [&](std::string_view name, double ms, size_t count) {
            std::printf("mesh,%u,%u,%zu,%.*s,%.6f,%llu\n", vertices, faces, count, int(name.size()), name.data(), ms, (unsigned long long)allocated());
        };
        const auto draw = [&](std::string_view name, size_t count) {
            const auto ms = Milliseconds([&] { SubmitViewport(r, engine.Viewport); WaitForRender(r); });
            row(name, ms, count);
        };
        const auto render_step = [&](std::string_view name) -> Result {
            if (!render) return {};
            if (r.Context.get<const PendingRenderRequest>().Value == RenderRequest::None)
                return std::unexpected{std::string{name} + " did not request a rendered frame"};
            draw(name, selected_count);
            return {};
        };
        const auto &cache = r.Context.get<const mtl::LibraryCache>();
        ReportLibraries(cache, "baseline");
        profile::ClearStats();
        profile::Enabled = !std::getenv("MESHEDITOR_EDIT_BENCH_UNPROFILED");
        row("selection_initial_replace", initial_select_ms, selected.size());
        row("selection", select_ms, selected.size());
        if (single_edit == SingleEdit::SpatialPlane || single_edit == SingleEdit::SpatialCut) {
            const vec3 normal = Normalize(vec3{0.3f, 0.8f, 0.5f});
            const char *plane_offset = std::getenv("MESHEDITOR_SPATIAL_PLANE_OFFSET");
            float offset = plane_offset ? std::strtof(plane_offset, nullptr) : Dot(normal, original.CalcFaceCentroid(face));
            // Runs one spatial face query on its own chain and returns its candidate meshlets, candidate triangles and exact faces.
            const auto spatial_counts = [&](const MeshTopologyTask &task) {
                mtl::ComputeChain chain{meshes.BufferContext()};
                SpatialFaceWork work{r, chain, task};
                chain.Submit();
                work.RecordFaces(r, chain);
                chain.Submit();
                return std::array{work.CandidateMeshlets, work.CandidateTriangles, work.Count};
            };
            if (single_edit == SingleEdit::SpatialPlane) {
                const MeshTopologyTask task{.SourceId = original.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts, .PlaneNormal = normal, .PlaneOffset = offset};
                std::array<uint32_t, 3> counts{};
                const auto elapsed = Milliseconds([&] { counts = spatial_counts(task); });
                std::printf("spatial_plane_work,%u,%u,%u\n", counts[0], counts[1], counts[2]);
                row("spatial_plane", elapsed, counts[2]);
                return {};
            }
            if (const char *fraction_text = std::getenv("MESHEDITOR_SPATIAL_NEAR_MAX")) {
                const float fraction = std::strtof(fraction_text, nullptr);
                if (!(std::isfinite(fraction) && fraction > 0.f && fraction < 1.f)) return std::unexpected{"invalid near-max fraction"};
                const auto &arena = meshes.Arenas().Vertices;
                const auto vertices_data = arena.Buffer.GetSpan<Vertex>();
                const auto set = meshes.Get(original.GetStoreId()).Vertices;
                float lo = INFINITY, hi = -INFINITY;
                arena.ForEach(set, [&](uint32_t vertex, uint32_t) {
                    const float coordinate = Dot(normal, vertices_data[vertex].Position);
                    lo = std::min(lo, coordinate);
                    hi = std::max(hi, coordinate);
                });
                offset = hi - fraction * (hi - lo);
                std::printf("spatial_cut_setup,%.9g,%.9g,%.9g\n", lo, hi, offset);
                const MeshTopologyTask probe{.SourceId = original.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts, .PlaneNormal = normal, .PlaneOffset = offset};
                const auto [meshlets, triangles, candidate_faces] = spatial_counts(probe);
                std::printf("spatial_cut_candidates,%u,%u,%u\n", meshlets, triangles, candidate_faces);
                if (!(candidate_faces > 0u && candidate_faces <= 4096u)) return std::unexpected{"near-max cut needs 1 to 4096 candidate faces"};
                profile::ClearStats();
            }
            const auto elapsed = Milliseconds([&] {
                p.Do(action::MakeAction(action::mesh::Bisect{.Point = normal * offset, .Normal = normal}));
            });
            const auto after = GetMesh(r, entity);
            if (after.GetStoreId() != original.GetStoreId()) return std::unexpected{"spatial cut replaced its canonical mesh"};
            row("spatial_cut", elapsed, after.FaceCount() - faces);
            if (std::getenv("MESHEDITOR_SPATIAL_VERIFY")) {
                const auto edited = *p.History.Present;
                const auto changed = after.FaceCount();
                draw("spatial_render_commit", changed - faces);
                const auto undo_ms = Milliseconds([&] { p.Navigate(base); });
                if (GetMesh(r, entity).FaceCount() != faces) return std::unexpected{"spatial cut undo did not restore faces"};
                row("spatial_undo", undo_ms, changed - faces);
                draw("spatial_render_undo", changed - faces);
                const auto redo_ms = Milliseconds([&] { p.Navigate(edited); });
                if (GetMesh(r, entity).FaceCount() != changed) return std::unexpected{"spatial cut redo did not restore edit"};
                row("spatial_redo", redo_ms, changed - faces);
                draw("spatial_render_redo", changed - faces);
                if (std::getenv("MESHEDITOR_SPATIAL_VERIFY_REOPEN")) {
                    if (!p.Save()) return std::unexpected{"spatial cut save failed"};
                    if (!p.Close()) return std::unexpected{"spatial cut close failed"};
                    if (!p.Open(dir)) return std::unexpected{"spatial cut reopen failed"};
                    if (GetMesh(r, GetActiveMeshEntity(r)).FaceCount() != changed) return std::unexpected{"spatial cut reopen did not restore the edit"};
                    draw("spatial_render_reopen", changed - faces);
                }
            }
            return {};
        }
        if (single_edit == SingleEdit::SeparateSelected) {
            const auto entity_count = r.view<const MeshHandle>().size();
            const auto elapsed = Milliseconds([&] { p.Do(action::MakeAction(action::mesh::Separate{})); });
            if (r.view<const MeshHandle>().size() != entity_count + 1u) return std::unexpected{"separate did not create one mesh object"};
            if (GetMesh(r, entity).FaceCount() != faces - selected_count) return std::unexpected{"separate did not remove selected source faces"};
            uint32_t output = InvalidStoreId;
            for (const auto e : r.view<const MeshHandle>()) {
                const auto id = r.get<const MeshHandle>(e).StoreId;
                if (id != original.GetStoreId()) output = id;
            }
            if (!(output != InvalidStoreId && Mesh{meshes, output}.FaceCount() == selected_count)) return std::unexpected{"separate emitted unrelated faces"};
            row("separate_selected", elapsed, selected_count);
            return {};
        }
        const auto counts = [&] {
            const auto mesh = GetMesh(r, entity);
            return std::array{mesh.VertexCount(), mesh.FaceCount()};
        };
        const auto source_counts = counts();
        const bool staged = !join && single_edit == SingleEdit::None;
        std::pair<const char *, double> edit_phase{"commit", 0.0};
        if (!staged) {
            const auto label = join ? "join" : edit.Label;
            const auto pair_distance = [&] {
                return Length(original.GetPosition(original.GetToVertex(selected_face_halfedge)) - original.GetPosition(original.GetFromVertex(selected_face_halfedge))) + 1e-4f;
            };
            const auto toggle_sharp = [&] {
                p.Do(action::MakeAction(action::object::SetSelectedSharp{.Element = Element::Face, .Sharp = !bool(meshes.Arenas().FaceSharpness.Get({*face, 1u})[0])}));
            };
            const auto edit_ms = Milliseconds([&] {
                if (join) p.Do(action::MakeAction(action::mesh::TrisToQuads{}));
                else if (single_edit == SingleEdit::EdgeSplit) p.Do(action::MakeAction(action::mesh::EdgeSplit{}));
                else if (single_edit == SingleEdit::DissolveEdge) p.Do(action::MakeAction(action::mesh::Dissolve{action::mesh::DissolveMode::Edges}));
                else if (single_edit == SingleEdit::EdgeRotate) p.Do(action::MakeAction(action::mesh::EdgeRotate{}));
                else if (single_edit == SingleEdit::ExtrudeEdge) p.Do(action::MakeAction(action::mesh::Extrude{action::mesh::ExtrudeMode::Edges}));
                else if (single_edit == SingleEdit::SubdivideEdge) p.Do(action::MakeAction(action::mesh::Subdivide{.Cuts = 1u}));
                else if (single_edit == SingleEdit::LoopCut) p.Do(action::MakeAction(action::mesh::LoopCut{1u}));
                else if (single_edit == SingleEdit::FillFace) p.Do(action::MakeAction(action::mesh::Fill{}));
                else if (single_edit == SingleEdit::FillHoles) p.Do(action::MakeAction(action::mesh::FillHoles{4u}));
                else if (single_edit == SingleEdit::DeleteEdge) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Edges}));
                else if (single_edit == SingleEdit::DeleteOnlyEdgeFaces) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyEdgesAndFaces}));
                else if (single_edit == SingleEdit::DeleteVertex) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Vertices}));
                else if (single_edit == SingleEdit::DissolveVertex) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Vertices}));
                else if (single_edit == SingleEdit::DissolveFace) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Faces}));
                else if (single_edit == SingleEdit::DissolveLimited) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = 3.14159f}));
                else if (single_edit == SingleEdit::MergeDistance) {
                    p.Do(action::MakeAction(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = pair_distance()}));
                } else if (single_edit == SingleEdit::MergeCenter) {
                    p.Do(action::MakeAction(action::mesh::Merge{.Mode = action::mesh::MergeMode::Center}));
                } else if (single_edit == SingleEdit::MergeCollapse) {
                    p.Do(action::MakeAction(action::mesh::Merge{.Mode = action::mesh::MergeMode::Collapse}));
                } else if (single_edit == SingleEdit::DissolveDegenerate) {
                    p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Degenerate, .Distance = pair_distance()}));
                } else if (single_edit == SingleEdit::SharpFace) toggle_sharp();
                else p.Do(action::MakeAction(action::mesh::Bevel{.Width = 0.01f, .Vertices = single_edit == SingleEdit::BevelVertex}));
            });
            const auto edited = *p.History.Present;
            const auto edited_counts = counts();
            if (GetMesh(r, entity).GetStoreId() != original.GetStoreId()) return std::unexpected{"operator replaced the canonical mesh"};
            if (filling && (edited_counts[1] != faces)) return std::unexpected{"fill did not restore the deleted face"};
            if (join && (edited_counts[1] != faces - 1u)) return std::unexpected{"selected triangle pair did not join"};
            if (merging && (edited_counts[0] != vertices - 1u)) return std::unexpected{"selected vertex pair did not merge"};
            if (single_edit == SingleEdit::SharpFace) {
                const auto reverse_ms = Milliseconds(toggle_sharp);
                row("sharp_face_warm_reverse", reverse_ms, selected_count);
                p.Navigate(edited);
            }
            if (auto result = render_step(std::string{"render_"} + label); !result) return result;
            edit_phase = {label, edit_ms};
        } else {
            std::vector<double> warm_preview_ms;
            if (updates > 1u) warm_preview_ms.reserve(updates - 1u);
            for (uint32_t i = 0; i < updates; ++i) {
                const auto elapsed = Milliseconds([&] {
                    if (position) action::Emit(action::view::TransformElements{{.P = {0.0001f * float(i + 1), 0.f, 0.f}}}, action::Phase::Stage);
                    else action::Emit(action::mesh::Inset{.Thickness = 0.0001f * float(i + 1), .Individual = select_all}, action::Phase::Stage);
                    p.Frame(action::Drain());
                });
                const auto mesh = GetMesh(r, entity);
                if (mesh.FaceCount() != faces + (position ? 0u : inset_added) || mesh.VertexCount() != vertices + (position ? 0u : inset_added)) return std::unexpected{"edit counts changed unexpectedly"};
                row("update" + std::to_string(i), elapsed, selected_count);
                if (i == 0u && std::getenv("MESHEDITOR_EDIT_BENCH_PROFILE_FIRST")) {
                    std::puts("First preview profile");
                    profile::Report();
                    profile::ClearStats();
                }
                if (i == 0u && refit_probe && !position) {
                    std::vector<Range> ranges;
                    meshes.GetSelectedElements(original.GetStoreId(), Element::Vertex).ForEach([&](uint32_t handle) {
                        ranges.push_back({handle, 1u});
                    });
                    if (ranges.empty()) return std::unexpected{"inset produced no selected vertices for refit probe"};
                    std::fprintf(stderr, "Probing position refit for %zu selected vertices...\n", ranges.size());
                    std::vector<double> refit_ms;
                    refit_ms.reserve(20u);
                    for (uint32_t sample = 0u; sample < 20u; ++sample) {
                        const auto duration = Milliseconds([&] {
                            mtl::ComputeChain chain{meshes.BufferContext()};
                            RefreshEditedPositions(r, chain, std::array{MeshVertexChanges{entity, ranges}});
                            chain.Submit();
                        });
                        row("refit" + std::to_string(sample), duration, ranges.size());
                        if (sample) refit_ms.push_back(duration);
                    }
                    row("refit_p95", P95(refit_ms), ranges.size());
                }
                if (i) warm_preview_ms.push_back(elapsed);
                if (render) {
                    const auto request = r.Context.get<const PendingRenderRequest>().Value;
                    if (request == RenderRequest::None) return std::unexpected{"edit preview did not request a rendered frame"};
                    if (i && (request != RenderRequest::Reuse)) return std::unexpected{"warmed edit preview requested a full scene rebuild"};
                    draw("render" + std::to_string(i), selected_count);
                    if (i == 1u)
                        if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE"); !result) return result;
                }
            }
            if (!warm_preview_ms.empty()) row("warm_preview_p95", P95(warm_preview_ms), selected_count);
            edit_phase.second = Milliseconds([&] { action::Commit(); p.Frame(action::Drain()); });
        }
        const auto edited = *p.History.Present;
        if (staged && edited == base) return std::unexpected{"edit was not committed"};
        const auto edited_counts = counts();
        const auto undo_ms = Milliseconds([&] { p.Navigate(base); });
        if (counts() != source_counts) return std::unexpected{"undo did not restore source"};
        if (auto result = render_step("render_undo"); !result) return result;
        const auto redo_ms = Milliseconds([&] { p.Navigate(edited); });
        if (counts() != edited_counts) return std::unexpected{"redo did not restore edit"};
        if (auto result = render_step("render_redo"); !result) return result;
        if (staged) {
            const auto replay_error = p.History.ValidateReplay();
            if (!replay_error.empty()) return std::unexpected{replay_error};
        }
        const auto exit_ms = Milliseconds([&] { p.Do(action::MakeAction(action::view::SetInteractionMode{InteractionMode::Object})); });
        if (staged && !r.Context.get<const GpuSceneState>().EditWork.empty()) return std::unexpected{"Edit mode exit retained geometry work"};
        if (r.Context.get<const GpuBuffers>().MeshOf(original.GetStoreId()).DirtyGroupRoot != InvalidOffset) return std::unexpected{"edit mode exit left stale LOD groups"};
        if (position && (r.Context.get<const GpuBuffers>().MeshOf(original.GetStoreId()).PositionDirtyRoot != InvalidOffset)) return std::unexpected{"position edit exit retained coarse dirt"};
        if (auto result = render_step("render_exit_mode"); !result) return result;
        if (staged && render) {
            if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE_EXIT"); !result) return result;
            if (std::getenv("MESHEDITOR_EDIT_BENCH_CAPTURE_FINE")) {
                r.patch<ViewportDisplay>(engine.Viewport, [](auto &display) { display.LodErrorPixels = 0.f; });
                p.Frame(action::Drain());
                SubmitViewport(r, engine.Viewport);
                WaitForRender(r);
                if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE_FINE"); !result) return result;
            }
        }
        for (const auto &[name, ms] : {edit_phase, {"undo", undo_ms}, {"redo", redo_ms}, {"exit_mode", exit_ms}})
            row(name, ms, selected_count);
        return {};
    }();
    if (result) {
        ReportLibraries(r.Context.get<const mtl::LibraryCache>(), "completed");
        profile::Report();
    }
    profile::Deinit();
    profile::Enabled = false;
    if (!result) return result;
    std::string why;
    if (!p.Audit(why)) return std::unexpected{why};
    if (!r.Context.get<action::Errors>().Messages.empty()) return std::unexpected{"action error"};
    if (!p.Close()) return std::unexpected{"project close failed"};
    return {};
}
} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2 || argc == 6 || argc > 11) {
        std::fprintf(stderr, "usage: MeshEditorEditBench <sphere-slices|scene.gltf> [updates=3] [render=0] [refit_probe=0] [viewport_width viewport_height] [position=0] [join=0] [op] [selection=one]\noperators:");
        for (const auto &spec : EditSpecs) std::fprintf(stderr, " %s", spec.Argument);
        std::fputs("\nselection: one (a single element for the operator) or all (every element of the operator's domain)\n", stderr);
        return 2;
    }
    const auto fail = [](std::string_view message) {
        std::fprintf(stderr, "%.*s\n", int(message.size()), message.data());
        return 1;
    };
    const auto scene = std::filesystem::path{argv[1]};
    const bool file = scene.extension() == ".gltf" || scene.extension() == ".glb";
    // Slices, updates, render, refit probe, width, height, position, join.
    std::array<uint32_t, 8> numbers{0u, 3u, 0u, 0u, 128u, 128u, 0u, 0u};
    for (int i = 1; i < std::min(argc, 9); ++i) {
        if (i == 1 && file) continue;
        const std::string_view input{argv[i]};
        const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), numbers[i - 1]);
        if (error != std::errc{} || end != input.data() + input.size()) return fail("invalid unsigned argument");
    }
    const auto [slices, updates, render, refit_probe, width, height, position, join] = numbers;
    if ((!file && slices < 8u) || updates == 0u) return fail("need at least eight slices and one update");
    if (width == 0u || height == 0u) return fail("viewport dimensions must be positive");
    const std::string_view op = argc > 9 ? argv[9] : "inset";
    const auto spec = std::ranges::find(EditSpecs, op, &EditSpec::Argument);
    if (spec == EditSpecs.end()) return fail("unknown benchmark operator");
    const std::string_view selection = argc > 10 ? argv[10] : "one";
    if (selection != "one" && selection != "all") return fail("selection must be one or all");
    std::printf("kind,vertices,faces,selected,phase,milliseconds,device_bytes\n");
    const auto result = Bench(slices, file ? scene : std::filesystem::path{}, updates, render != 0u, refit_probe != 0u, width, height, position != 0u, join != 0u, *spec, selection == "all");
    return result ? 0 : fail(result.error());
}
