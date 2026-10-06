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
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/Primitives.h"
#include "mesh/SpatialFaceWork.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/Shader.h"
#include "object/ObjectOps.h"
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
                        ExtrudeRegion,
                        ExtrudeVertex,
                        NewEdge,
                        DuplicateVertex,
                        DuplicateEdge,
                        DuplicateFace,
                        SplitVertex,
                        SplitEdge,
                        SplitFace,
                        SubdivideEdge,
                        LoopCut,
                        FillFace,
                        FillHoles,
                        Bridge,
                        GridFill,
                        SpaceEvenly,
                        RelaxEdgeLoops,
                        Flatten,
                        CurveBetweenSelected,
                        Circularize,
                        DeleteLoose,
                        DeleteEdge,
                        DeleteOnlyEdgeFaces,
                        DeleteFaces,
                        DeleteOnlyFaces,
                        DeleteVertex,
                        DissolveVertex,
                        DissolveFace,
                        DissolveLimited,
                        DissolveDelimited,
                        SharpFace,
                        MergeDistance,
                        MergeCenter,
                        MergeCorners,
                        MergeCollapse,
                        SmoothVertices,
                        ShrinkFatten,
                        ToSphere,
                        PushPull,
                        Shear,
                        Warp,
                        Bend,
                        Randomize,
                        VertexSlide,
                        EdgeSlide,
                        RecalculateNormals,
                        BeautifyFaces,
                        Unsubdivide,
                        Decimate,
                        SnapSymmetry,
                        Hide,
                        DuplicateObject,
                        MakePlanarFaces,
                        RotateUVs,
                        SplitNonplanarFaces,
                        SplitConcaveFaces,
                        Wireframe,
                        DissolveDegenerate };
struct EditSpec {
    SingleEdit Op;
    const char *Argument;
    Element Domain = Element::Face;
};
constexpr std::array EditSpecs{
    EditSpec{SingleEdit::None, "inset"},
    EditSpec{SingleEdit::SeparateSelected, "separate-selected"},
    EditSpec{SingleEdit::SpatialPlane, "spatial-plane"},
    EditSpec{SingleEdit::SpatialCut, "spatial-cut"},
    EditSpec{SingleEdit::EdgeSplit, "edge-split", Element::Edge},
    EditSpec{SingleEdit::BevelEdge, "bevel-edge", Element::Edge},
    EditSpec{SingleEdit::BevelVertex, "bevel-vertex", Element::Vertex},
    EditSpec{SingleEdit::DissolveEdge, "dissolve-edge", Element::Edge},
    EditSpec{SingleEdit::EdgeRotate, "edge-rotate", Element::Edge},
    EditSpec{SingleEdit::ExtrudeEdge, "extrude-edge", Element::Edge},
    EditSpec{SingleEdit::ExtrudeRegion, "extrude-region"},
    EditSpec{SingleEdit::ExtrudeVertex, "extrude-vertex", Element::Vertex},
    EditSpec{SingleEdit::NewEdge, "new-edge", Element::Vertex},
    EditSpec{SingleEdit::DuplicateVertex, "duplicate-vertex", Element::Vertex},
    EditSpec{SingleEdit::DuplicateEdge, "duplicate-edge", Element::Edge},
    EditSpec{SingleEdit::DuplicateFace, "duplicate-face"},
    EditSpec{SingleEdit::SplitVertex, "split-vertex", Element::Vertex},
    EditSpec{SingleEdit::SplitEdge, "split-edge", Element::Edge},
    EditSpec{SingleEdit::SplitFace, "split-face"},
    EditSpec{SingleEdit::SubdivideEdge, "subdivide-edge", Element::Edge},
    EditSpec{SingleEdit::LoopCut, "loop-cut", Element::Edge},
    EditSpec{SingleEdit::FillFace, "fill-face", Element::Edge},
    EditSpec{SingleEdit::FillHoles, "fill-holes", Element::Edge},
    EditSpec{SingleEdit::Bridge, "bridge", Element::Edge},
    EditSpec{SingleEdit::GridFill, "grid-fill", Element::Edge},
    EditSpec{SingleEdit::SpaceEvenly, "space-evenly", Element::Edge},
    EditSpec{SingleEdit::RelaxEdgeLoops, "relax-edge-loops", Element::Edge},
    EditSpec{SingleEdit::Flatten, "flatten", Element::Edge},
    EditSpec{SingleEdit::CurveBetweenSelected, "curve-between-selected", Element::Vertex},
    EditSpec{SingleEdit::Circularize, "circularize", Element::Edge},
    EditSpec{SingleEdit::DeleteLoose, "delete-loose", Element::Vertex},
    EditSpec{SingleEdit::DeleteEdge, "delete-edge", Element::Edge},
    EditSpec{SingleEdit::DeleteOnlyEdgeFaces, "delete-only-edge-faces", Element::Edge},
    EditSpec{SingleEdit::DeleteFaces, "delete-faces"},
    EditSpec{SingleEdit::DeleteOnlyFaces, "delete-only-faces"},
    EditSpec{SingleEdit::DeleteVertex, "delete-vertex", Element::Vertex},
    EditSpec{SingleEdit::DissolveVertex, "dissolve-vertex", Element::Vertex},
    EditSpec{SingleEdit::DissolveFace, "dissolve-face"},
    EditSpec{SingleEdit::DissolveLimited, "dissolve-limited"},
    EditSpec{SingleEdit::DissolveDelimited, "dissolve-delimited"},
    EditSpec{SingleEdit::SharpFace, "sharp-face"},
    EditSpec{SingleEdit::MergeDistance, "merge-distance", Element::Vertex},
    EditSpec{SingleEdit::MergeCenter, "merge-center", Element::Vertex},
    EditSpec{SingleEdit::MergeCorners, "merge-corners", Element::Vertex},
    EditSpec{SingleEdit::MergeCollapse, "merge-collapse", Element::Vertex},
    EditSpec{SingleEdit::DissolveDegenerate, "dissolve-degenerate", Element::Vertex},
    EditSpec{SingleEdit::RecalculateNormals, "recalculate-normals"},
    EditSpec{SingleEdit::SnapSymmetry, "snap-symmetry"},
    EditSpec{SingleEdit::Hide, "hide"},
    EditSpec{SingleEdit::DuplicateObject, "duplicate-object"},
    EditSpec{SingleEdit::Unsubdivide, "unsubdivide"},
    EditSpec{SingleEdit::Decimate, "decimate"},
    EditSpec{SingleEdit::BeautifyFaces, "beautify-faces"},
    EditSpec{SingleEdit::ToSphere, "to-sphere"},
    EditSpec{SingleEdit::PushPull, "push-pull"},
    EditSpec{SingleEdit::Shear, "shear"},
    EditSpec{SingleEdit::Warp, "warp"},
    EditSpec{SingleEdit::Bend, "bend"},
    EditSpec{SingleEdit::Randomize, "randomize"},
    EditSpec{SingleEdit::VertexSlide, "vertex-slide"},
    EditSpec{SingleEdit::EdgeSlide, "edge-slide", Element::Edge},
    EditSpec{SingleEdit::ShrinkFatten, "shrink-fatten", Element::Vertex},
    EditSpec{SingleEdit::SmoothVertices, "smooth-vertices", Element::Vertex},
    EditSpec{SingleEdit::RotateUVs, "rotate-uvs"},
    EditSpec{SingleEdit::MakePlanarFaces, "planar-faces"},
    EditSpec{SingleEdit::SplitNonplanarFaces, "split-nonplanar"},
    EditSpec{SingleEdit::SplitConcaveFaces, "split-concave"},
    EditSpec{SingleEdit::Wireframe, "wireframe"},
};
// The selected face and its valence stay fixed while the unselected mesh grows.
// Select-all replaces the operator's selection with every element of its domain, and inset then insets each face individually.
// Timings include the production project/event/history path.
// Audits and rendering are separate.
Result Bench(uint32_t slices, const std::filesystem::path &scene, uint32_t updates, bool render, bool refit_probe, uint32_t viewport_width, uint32_t viewport_height, bool position, bool join, const EditSpec &edit, bool select_all) {
    const auto single_edit = edit.Op;
    std::string label = join ? "join" : edit.Argument;
    std::ranges::replace(label, '-', '_');
    if (uint32_t(position) + uint32_t(join) + uint32_t(single_edit != SingleEdit::None) > 1u) return std::unexpected{"choose one benchmark operation"};
    const bool filling = single_edit == SingleEdit::FillFace || single_edit == SingleEdit::FillHoles;
    const bool merging = single_edit == SingleEdit::MergeDistance || single_edit == SingleEdit::MergeCenter || single_edit == SingleEdit::MergeCorners ||
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
        if (scene.empty() && single_edit == SingleEdit::BeautifyFaces) {
            MeshSource source;
            const uint32_t count = std::max(1u, slices * slices / 4u);
            for (uint32_t i = 0u; i < count; ++i) {
                const uint32_t base = uint32_t(source.Data.Positions.size());
                const vec3 offset{5.f * float(i % 256u), 5.f * float(i / 256u), 0.f};
                for (const vec3 p : std::array{vec3{0, 0, 0}, vec3{3, 0, 0}, vec3{2, 1, 0}, vec3{0, 2, 0}}) source.Data.Positions.push_back(p + offset);
                source.Data.AddFace(std::array{base, base + 1u, base + 3u});
                source.Data.AddFace(std::array{base + 1u, base + 2u, base + 3u});
            }
            const auto id = CreateMesh(r, std::move(source)).StoreId;
            const auto [entity, instance] = AddMesh(r, id, MeshInstanceCreateInfo{});
            p.Settle();
            p.Do(action::MakeAction(action::selection::Select{instance}));
        } else if (scene.empty() && single_edit == SingleEdit::RecalculateNormals) {
            MeshSource source;
            const auto cube = primitive::CreateMesh(primitive::Cuboid{});
            const uint32_t count = std::max(1u, slices * slices / 12u);
            for (uint32_t i = 0u; i < count; ++i) {
                const uint32_t base = uint32_t(source.Data.Positions.size());
                const vec3 offset{3.f * float(i % 256u), 3.f * float(i / 256u), 0.f};
                for (const auto p : cube.Positions) source.Data.Positions.push_back(p + offset);
                for (uint32_t f = 0u; f < cube.FaceCount(); ++f) {
                    std::vector<uint32_t> face;
                    for (const auto v : cube.Face(f)) face.push_back(base + v);
                    source.Data.AddFace(face);
                }
            }
            const auto id = CreateMesh(r, std::move(source)).StoreId;
            const auto [entity, instance] = AddMesh(r, id, MeshInstanceCreateInfo{});
            p.Settle();
            p.Do(action::MakeAction(action::selection::Select{instance}));
        } else if (scene.empty() && (single_edit == SingleEdit::Bridge || single_edit == SingleEdit::GridFill || single_edit == SingleEdit::SpaceEvenly || single_edit == SingleEdit::RelaxEdgeLoops || single_edit == SingleEdit::Flatten || single_edit == SingleEdit::CurveBetweenSelected || single_edit == SingleEdit::Circularize || single_edit == SingleEdit::DissolveDelimited || single_edit == SingleEdit::RotateUVs || single_edit == SingleEdit::VertexSlide || single_edit == SingleEdit::EdgeSlide)) {
            MeshSource source{.Data = primitive::CreateMesh(primitive::UVSphere{.Slices = slices, .Stacks = slices / 2})};
            if (single_edit == SingleEdit::Bridge) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {1, 0, 2}, {0, 1, 2}, {1, 1, 2}});
                source.Data.Edges = {{first, first + 1u}, {first + 2u, first + 3u}};
            } else if (single_edit == SingleEdit::GridFill) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {2, 1, 2}, {2, 2, 2}, {1, 2, 2}, {0, 2, 2}, {0, 1, 2}});
                for (uint32_t i = 0u; i < 8u; ++i) source.Data.Edges.push_back({first + i, first + (i + 1u) % 8u});
            } else if (single_edit == SingleEdit::Circularize) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {2, 0, 2}, {3, 1, 2}, {2, 3, 2}, {-1, 2, 2}});
                for (uint32_t i = 0u; i < 5u; ++i) source.Data.Edges.push_back({first + i, first + (i + 1u) % 5u});
            } else if (single_edit == SingleEdit::CurveBetweenSelected) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {1, 1, 2}, {2, 0, 2}, {3, -1, 2}, {4, 0, 2}});
                for (uint32_t i = 0u; i < 4u; ++i) source.Data.Edges.push_back({first + i, first + i + 1u});
            } else if (single_edit == SingleEdit::Flatten) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {2, 0, 2}, {2, 2, 3}, {0, 2, 2}});
                for (uint32_t i = 0u; i < 3u; ++i) source.Data.Edges.push_back({first + i, first + i + 1u});
            } else if (single_edit == SingleEdit::SpaceEvenly || single_edit == SingleEdit::RelaxEdgeLoops) {
                const auto first = uint32_t(source.Data.Positions.size());
                source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {.25f, 1, 2}, {2, 0, 2}, {3, 2, 2}, {5, 0, 2}});
                for (uint32_t i = 0u; i < 4u; ++i) source.Data.Edges.push_back({first + i, first + i + 1u});
            }
            auto &uvs = source.Attrs.TexCoords0.emplace();
            for (const auto position : source.Data.Positions) uvs.push_back({position.x, position.y});
            const auto id = CreateMesh(r, std::move(source)).StoreId;
            const auto [entity, instance] = AddMesh(r, id, MeshInstanceCreateInfo{});
            p.Settle();
            p.Do(action::MakeAction(action::selection::Select{instance}));
        } else if (scene.empty()) p.Do(action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
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
        const bool pair_faces = join || single_edit == SingleEdit::DissolveFace || (single_edit == SingleEdit::DissolveLimited || single_edit == SingleEdit::DissolveDelimited) || single_edit == SingleEdit::BeautifyFaces;
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
        if (filling) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces}));
        if (filling) {
            const auto boundary = meshes.GetBoundaryEdges(original.GetStoreId());
            uint32_t on_hole = 0u;
            for (const auto edge : fill_edges) on_hole += boundary.Contains(edge);
            std::fprintf(stderr, "Hole boundary index: %u edges total, %u/%zu hole edges indexed.\n", boundary.Count(), on_hole, fill_edges.size());
        }
        if (position || (single_edit != SingleEdit::None && edit.Domain != Element::Face)) {
            const auto element = position ? Element::Vertex : edit.Domain;
            const bool use_vertex = element == Element::Vertex;
            p.Do(action::MakeAction(action::view::SetEditMode{.Mode = element}));
            const auto current = GetMesh(r, entity);
            std::vector<uint32_t> selected_elements;
            if (single_edit == SingleEdit::Bridge && scene.empty()) {
                selected_elements = {current.EdgeCount() - 2u, current.EdgeCount() - 1u};
            } else if (single_edit == SingleEdit::GridFill && scene.empty()) {
                for (uint32_t i = 8u; i > 0u; --i) selected_elements.push_back(current.EdgeCount() - i);
            } else if (single_edit == SingleEdit::Circularize && scene.empty()) {
                for (uint32_t i = 5u; i > 0u; --i) selected_elements.push_back(current.EdgeCount() - i);
            } else if (single_edit == SingleEdit::CurveBetweenSelected && scene.empty()) {
                for (uint32_t i = 0u; i < 5u; i += 2u) selected_elements.push_back(current.VertexCount() - 5u + i);
            } else if (single_edit == SingleEdit::Flatten && scene.empty()) {
                for (uint32_t i = 3u; i > 0u; --i) selected_elements.push_back(current.EdgeCount() - i);
            } else if ((single_edit == SingleEdit::SpaceEvenly || single_edit == SingleEdit::RelaxEdgeLoops) && scene.empty()) {
                for (uint32_t i = 4u; i > 0u; --i) selected_elements.push_back(current.EdgeCount() - i);
            } else if (filling) {
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
            if (merging && single_edit != SingleEdit::MergeCorners) selected_elements.push_back(original.VertexOrdinal(original.GetFromVertex(selected_face_halfedge)));
            if (single_edit == SingleEdit::NewEdge || single_edit == SingleEdit::MergeCorners) {
                const auto &c = original.GetConnectivity();
                selected_elements.push_back(original.VertexOrdinal(original.GetToVertex(c.Next(c.Next(selected_face_halfedge)))));
            }
            select(selected_elements, element);
            if (single_edit == SingleEdit::DeleteLoose) {
                p.Do(action::MakeAction(action::mesh::Extrude{action::mesh::ExtrudeMode::Vertices}));
                const auto copy = meshes.GetSelectedElements(original.GetStoreId(), Element::Vertex).First();
                if (!copy) return std::unexpected{"loose-delete setup did not extrude a vertex"};
                selected_elements.push_back(*copy - original.VertexFirst());
                select(selected_elements, Element::Vertex);
            }
            const auto selected_count = meshes.GetSelectionSummary(original.GetStoreId()).SelectedCount;
            if (selected_count != selected_elements.size()) std::fprintf(stderr, "operator selection: expected %zu, got %u\n", selected_elements.size(), selected_count);
            if (selected_count != selected_elements.size()) return std::unexpected{"unexpected operator selection"};
            if (filling) {
                const auto selected_edges = meshes.GetSelectedElements(original.GetStoreId(), Element::Edge);
                for (const auto edge : fill_edges)
                    if (!selected_edges.Contains(edge)) return std::unexpected{"fill did not select a hole edge"};
            }
        }
        if (single_edit == SingleEdit::MakePlanarFaces || single_edit == SingleEdit::SplitNonplanarFaces || single_edit == SingleEdit::SplitConcaveFaces) {
            if (original.GetValence(face) < 4u) return std::unexpected{"planar benchmark needs a polygon with at least four corners"};
            // Deform one corner before timing so flattening and splitting do real work.
            vec3 offset = original.GetNormal(face) * 0.01f;
            if (single_edit == SingleEdit::SplitConcaveFaces) {
                if (original.GetValence(face) != 4u) return std::unexpected{"concave benchmark requires a quad"};
                std::array<vec3, 4> points;
                uint32_t i = 0u;
                for (const auto v : original.fv_range(face)) points[i++] = original.GetPosition(v);
                offset = (points[1] + points[3]) * 0.375f + points[2] * 0.25f - points[0];
            }
            p.Do(action::MakeAction(action::view::SetEditMode{.Mode = Element::Vertex}));
            const std::array corner{original.VertexOrdinal(original.GetToVertex(selected_face_halfedge))};
            select(corner, Element::Vertex);
            action::Emit(action::view::TransformElements{{.P = offset}}, action::Phase::Stage);
            p.Frame(action::Drain());
            action::Commit();
            p.Frame(action::Drain());
            p.Do(action::MakeAction(action::view::SetEditMode{.Mode = Element::Face}));
            select(selected);
        }
        if (single_edit == SingleEdit::SnapSymmetry) {
            action::Emit(action::view::TransformElements{{.P = {.0001f, 0.f, 0.f}}}, action::Phase::Stage);
            p.Frame(action::Drain());
            action::Commit();
            p.Frame(action::Drain());
        }
        if (single_edit == SingleEdit::RecalculateNormals) p.Do(action::MakeAction(action::mesh::FlipNormals{}));
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
        if (single_edit == SingleEdit::DuplicateObject) {
            p.Do(action::MakeAction(action::mesh::Subdivide{2u}));
            p.Do(action::MakeAction(action::view::SetInteractionMode{InteractionMode::Object}));
            const auto source = GetMesh(r, entity);
            const auto before = r.view<const MeshHandle>().size();
            profile::ClearStats();
            const auto elapsed = Milliseconds([&] { p.Do(action::MakeAction(action::object::Duplicate{})); });
            if (r.view<const MeshHandle>().size() != before + 1u) return std::unexpected{"object duplicate did not create one mesh"};
            const auto clone = GetMesh(r, GetActiveMeshEntity(r));
            if (clone.GetStoreId() == source.GetStoreId() || clone.VertexCount() != source.VertexCount() || clone.FaceCount() != source.FaceCount())
                return std::unexpected{"object duplicate did not preserve edited geometry"};
            row("duplicate_object", elapsed, source.FaceCount());
            const auto undo_ms = Milliseconds([&] { p.Undo(); });
            if (r.view<const MeshHandle>().size() != before) return std::unexpected{"object duplicate undo failed"};
            row("duplicate_object_undo", undo_ms, source.FaceCount());
            const auto redo_ms = Milliseconds([&] { p.Redo(); });
            if (r.view<const MeshHandle>().size() != before + 1u) return std::unexpected{"object duplicate redo failed"};
            row("duplicate_object_redo", redo_ms, source.FaceCount());
            if (render) draw("duplicate_object_render", source.FaceCount());
            return {};
        }
        const auto counts = [&] {
            const auto mesh = GetMesh(r, entity);
            return std::array{mesh.VertexCount(), mesh.FaceCount()};
        };
        const auto source_counts = counts();
        const auto source_edges = GetMesh(r, entity).EdgeCount();
        const bool splitting = single_edit == SingleEdit::SplitVertex || single_edit == SingleEdit::SplitEdge || single_edit == SingleEdit::SplitFace;
        const bool duplicating = single_edit == SingleEdit::DuplicateVertex || single_edit == SingleEdit::DuplicateEdge || single_edit == SingleEdit::DuplicateFace;
        const std::array duplicate_counts{meshes.GetSelectedElements(original.GetStoreId(), Element::Vertex).Count(), meshes.GetSelectedElements(original.GetStoreId(), Element::Edge).Count(), meshes.GetSelectedElements(original.GetStoreId(), Element::Face).Count()};
        const bool staged = !join && single_edit == SingleEdit::None;
        std::pair<const char *, double> edit_phase{"commit", 0.0};
        if (!staged) {
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
                else if (single_edit == SingleEdit::ExtrudeRegion) p.Do(action::MakeAction(action::mesh::Extrude{}));
                else if (single_edit == SingleEdit::ExtrudeVertex) p.Do(action::MakeAction(action::mesh::Extrude{action::mesh::ExtrudeMode::Vertices}));
                else if (splitting) p.Do(action::MakeAction(action::mesh::Split{}));
                else if (duplicating) p.Do(action::MakeAction(action::mesh::Duplicate{}));
                else if (single_edit == SingleEdit::NewEdge) p.Do(action::MakeAction(action::mesh::Fill{}));
                else if (single_edit == SingleEdit::SubdivideEdge) p.Do(action::MakeAction(action::mesh::Subdivide{.Cuts = 1u}));
                else if (single_edit == SingleEdit::LoopCut) p.Do(action::MakeAction(action::mesh::LoopCut{1u}));
                else if (single_edit == SingleEdit::FillFace) p.Do(action::MakeAction(action::mesh::Fill{}));
                else if (single_edit == SingleEdit::Bridge) p.Do(action::MakeAction(action::mesh::BridgeEdgeLoops{}));
                else if (single_edit == SingleEdit::GridFill) p.Do(action::MakeAction(action::mesh::GridFill{2u}));
                else if (single_edit == SingleEdit::SpaceEvenly) p.Do(action::MakeAction(action::mesh::SpaceEvenly{}));
                else if (single_edit == SingleEdit::RelaxEdgeLoops) p.Do(action::MakeAction(action::mesh::RelaxEdgeLoops{}));
                else if (single_edit == SingleEdit::Flatten) p.Do(action::MakeAction(action::mesh::Flatten{}));
                else if (single_edit == SingleEdit::CurveBetweenSelected) p.Do(action::MakeAction(action::mesh::CurveBetweenSelected{}));
                else if (single_edit == SingleEdit::Circularize) p.Do(action::MakeAction(action::mesh::Circularize{}));
                else if (single_edit == SingleEdit::FillHoles) p.Do(action::MakeAction(action::mesh::FillHoles{4u}));
                else if (single_edit == SingleEdit::DeleteLoose) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Loose}));
                else if (single_edit == SingleEdit::DeleteEdge) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Edges}));
                else if (single_edit == SingleEdit::DeleteOnlyEdgeFaces) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyEdgesAndFaces}));
                else if (single_edit == SingleEdit::DeleteFaces) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Faces}));
                else if (single_edit == SingleEdit::DeleteOnlyFaces) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces}));
                else if (single_edit == SingleEdit::DeleteVertex) p.Do(action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::Vertices}));
                else if (single_edit == SingleEdit::DissolveVertex) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Vertices}));
                else if (single_edit == SingleEdit::RecalculateNormals) p.Do(action::MakeAction(action::mesh::RecalculateNormals{}));
                else if (single_edit == SingleEdit::SnapSymmetry) p.Do(action::MakeAction(action::mesh::SnapSymmetry{.Threshold = .001f}));
                else if (single_edit == SingleEdit::Hide) p.Do(action::MakeAction(action::mesh::Hide{}));
                else if (single_edit == SingleEdit::Unsubdivide) p.Do(action::MakeAction(action::mesh::Unsubdivide{1u}));
                else if (single_edit == SingleEdit::Decimate) p.Do(action::MakeAction(action::mesh::Decimate{}));
                else if (single_edit == SingleEdit::BeautifyFaces) p.Do(action::MakeAction(action::mesh::BeautifyFaces{}));
                else if (single_edit == SingleEdit::ToSphere) p.Do(action::MakeAction(action::mesh::ToSphere{}));
                else if (single_edit == SingleEdit::PushPull) p.Do(action::MakeAction(action::mesh::PushPull{.Distance = .01f}));
                else if (single_edit == SingleEdit::Shear) p.Do(action::MakeAction(action::mesh::Shear{}));
                else if (single_edit == SingleEdit::Warp) p.Do(action::MakeAction(action::mesh::Warp{.Angle = 1.f}));
                else if (single_edit == SingleEdit::Bend) p.Do(action::MakeAction(action::mesh::Bend{.Clamp = false}));
                else if (single_edit == SingleEdit::Randomize) p.Do(action::MakeAction(action::mesh::Randomize{.Amount = .01f, .Uniform = .5f, .Normal = .5f, .Seed = 7u}));
                else if (single_edit == SingleEdit::VertexSlide) p.Do(action::MakeAction(action::mesh::VertexSlide{.Factor = .1f, .Even = true}));
                else if (single_edit == SingleEdit::EdgeSlide) p.Do(action::MakeAction(action::mesh::EdgeSlide{.Factor = .1f, .Even = true}));
                else if (single_edit == SingleEdit::ShrinkFatten) p.Do(action::MakeAction(action::mesh::ShrinkFatten{.Distance = .01f, .Even = true}));
                else if (single_edit == SingleEdit::SmoothVertices) p.Do(action::MakeAction(action::mesh::SmoothVertices{}));
                else if (single_edit == SingleEdit::RotateUVs) p.Do(action::MakeAction(action::mesh::RotateUVs{}));
                else if (single_edit == SingleEdit::MakePlanarFaces) p.Do(action::MakeAction(action::mesh::MakePlanarFaces{}));
                else if (single_edit == SingleEdit::SplitNonplanarFaces) p.Do(action::MakeAction(action::mesh::SplitNonplanarFaces{}));
                else if (single_edit == SingleEdit::SplitConcaveFaces) p.Do(action::MakeAction(action::mesh::SplitConcaveFaces{}));
                else if (single_edit == SingleEdit::Wireframe) p.Do(action::MakeAction(action::mesh::Wireframe{}));
                else if (single_edit == SingleEdit::DissolveFace) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Faces}));
                else if ((single_edit == SingleEdit::DissolveLimited || single_edit == SingleEdit::DissolveDelimited)) p.Do(action::MakeAction(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = 3.14159f, .DelimitMaterials = single_edit == SingleEdit::DissolveDelimited, .DelimitSharpEdges = single_edit == SingleEdit::DissolveDelimited, .DelimitUVs = single_edit == SingleEdit::DissolveDelimited}));
                else if (single_edit == SingleEdit::MergeDistance) {
                    p.Do(action::MakeAction(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = pair_distance()}));
                } else if (single_edit == SingleEdit::MergeCenter || single_edit == SingleEdit::MergeCorners) {
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
            if (splitting && (edited_counts[0] < source_counts[0] || edited_counts[1] != source_counts[1] || GetMesh(r, entity).EdgeCount() < source_edges))
                return std::unexpected{"split removed source geometry"};
            if (duplicating && (edited_counts[0] != source_counts[0] + duplicate_counts[0] || edited_counts[1] != source_counts[1] + duplicate_counts[2] || GetMesh(r, entity).EdgeCount() != source_edges + duplicate_counts[1]))
                return std::unexpected{"duplicate did not copy exactly the selected geometry"};
            if (single_edit == SingleEdit::DeleteOnlyFaces && (edited_counts[0] != source_counts[0] || edited_counts[1] != source_counts[1] - selected.size() || GetMesh(r, entity).EdgeCount() != source_edges))
                return std::unexpected{"only-faces deletion did not preserve vertices and edges"};
            if (single_edit == SingleEdit::DeleteLoose && (edited_counts[0] + 1u != source_counts[0] || edited_counts[1] != source_counts[1] || GetMesh(r, entity).EdgeCount() + 1u != source_edges))
                return std::unexpected{"loose deletion did not remove exactly the extruded edge and vertex"};
            if (single_edit == SingleEdit::Bridge && scene.empty() && !select_all &&
                (edited_counts[0] != source_counts[0] || edited_counts[1] != source_counts[1] + 1u || GetMesh(r, entity).EdgeCount() != source_edges + 2u))
                return std::unexpected{"bridge did not join the two loose edges"};
            if (single_edit == SingleEdit::GridFill && scene.empty() && !select_all &&
                (edited_counts[0] != source_counts[0] + 1u || edited_counts[1] != source_counts[1] + 4u || GetMesh(r, entity).EdgeCount() != source_edges + 4u))
                return std::unexpected{"grid fill did not create one vertex and four quads inside the loose boundary"};
            if (single_edit == SingleEdit::Circularize && scene.empty() && !select_all) {
                const auto mesh = GetMesh(r, entity);
                if (edited_counts != source_counts || mesh.EdgeCount() != source_edges)
                    return std::unexpected{"circle fitting changed topology"};
                if (Length(mesh.GetPosition(mesh.VertexAt(mesh.VertexCount() - 5u)) - vec3{-.45437264f, .26652074f, 2.f}) > 1e-5f)
                    return std::unexpected{"circle fitting differs from the nonlinear least-squares reference"};
            }
            if (single_edit == SingleEdit::CurveBetweenSelected && scene.empty() && !select_all) {
                const auto mesh = GetMesh(r, entity);
                if (edited_counts != source_counts || mesh.EdgeCount() != source_edges)
                    return std::unexpected{"curve fitting changed topology"};
                for (uint32_t i = 1u; i < 4u; i += 2u)
                    if (Length(mesh.GetPosition(mesh.VertexAt(mesh.VertexCount() - 5u + i)) - vec3{float(i), 0, 2}) > 1e-5f)
                        return std::unexpected{"curve fitting did not interpolate the selected control points"};
            }
            if (single_edit == SingleEdit::Flatten && scene.empty() && !select_all) {
                const auto mesh = GetMesh(r, entity);
                if (edited_counts != source_counts || mesh.EdgeCount() != source_edges)
                    return std::unexpected{"flatten changed topology"};
                if (Length(mesh.GetPosition(mesh.VertexAt(mesh.VertexCount() - 4u)) - vec3{0.06480586011f, 0.06480586011f, 1.755084981f}) > 1e-5f)
                    return std::unexpected{"flatten differs from the least-squares reference"};
            }
            if ((single_edit == SingleEdit::SpaceEvenly || single_edit == SingleEdit::RelaxEdgeLoops) && scene.empty() && !select_all) {
                if (edited_counts != source_counts || GetMesh(r, entity).EdgeCount() != source_edges)
                    return std::unexpected{"edge spacing changed topology"};
                const auto mesh = GetMesh(r, entity);
                const auto v = mesh.VertexAt(mesh.VertexCount() - 4u);
                const auto expected = single_edit == SingleEdit::SpaceEvenly ? vec3{1.05305076f, .56604123f, 2.f} : vec3{.63188285f, .5f, 2.f};
                if (Length(mesh.GetPosition(v) - expected) > 1e-5f)
                    return std::unexpected{"edge curve edit differs from the cubic reference"};
            }
            if (single_edit == SingleEdit::NewEdge && (edited_counts != source_counts || GetMesh(r, entity).EdgeCount() != source_edges + 1u))
                return std::unexpected{"edge creation did not add exactly one loose edge"};
            if (single_edit == SingleEdit::ExtrudeVertex &&
                (edited_counts[0] != source_counts[0] + 1u || edited_counts[1] != source_counts[1] || GetMesh(r, entity).EdgeCount() != source_edges + 1u))
                return std::unexpected{"vertex extrusion did not add one vertex and one loose edge"};
            if (single_edit == SingleEdit::ExtrudeEdge && !select_all &&
                (edited_counts[0] != source_counts[0] + 2u || edited_counts[1] != source_counts[1] + 1u || GetMesh(r, entity).EdgeCount() != source_edges + 3u))
                return std::unexpected{"edge extrusion did not add two vertices, three edges, and a quad"};
            if (single_edit == SingleEdit::ExtrudeRegion && !select_all && scene.empty() &&
                (edited_counts[0] != source_counts[0] + 4u || edited_counts[1] != source_counts[1] + 4u || GetMesh(r, entity).EdgeCount() != source_edges + 8u))
                return std::unexpected{"region extrusion did not replace the selected quad and create its four sides"};
            if (GetMesh(r, entity).GetStoreId() != original.GetStoreId()) return std::unexpected{"operator replaced the canonical mesh"};
            if (filling && (edited_counts[1] != faces)) return std::unexpected{"fill did not restore the deleted face"};
            if (join && (edited_counts[1] != faces - 1u)) return std::unexpected{"selected triangle pair did not join"};
            if (merging && (edited_counts[0] != vertices - 1u)) return std::unexpected{"selected vertex pair did not merge"};
            if (single_edit == SingleEdit::MergeCorners && scene.empty() && edited_counts[1] != faces - 1u)
                return std::unexpected{"welding opposite quad corners did not remove the collapsed face"};
            if (single_edit == SingleEdit::Wireframe && !select_all && scene.empty() &&
                (edited_counts[0] != vertices + 4u * added || edited_counts[1] != faces + 4u * added - 1u))
                return std::unexpected{"wireframe did not emit the expected one-face struts"};
            if ((single_edit == SingleEdit::SplitNonplanarFaces || single_edit == SingleEdit::SplitConcaveFaces) && !select_all && scene.empty() &&
                (edited_counts[0] != vertices || edited_counts[1] != faces + 1u))
                return std::unexpected{"polygon split did not divide the deformed quad"};
            if (single_edit == SingleEdit::RecalculateNormals) {
                if (edited_counts != source_counts) return std::unexpected{"normal recalculation changed element counts"};
                if (edited == base) return std::unexpected{"normal recalculation did not repair the flipped face"};
            }
            if (single_edit == SingleEdit::SnapSymmetry) {
                if (edited_counts != source_counts) return std::unexpected{"symmetry snap changed topology"};
                if (edited == base) return std::unexpected{"symmetry snap did not change positions"};
            }
            if (single_edit == SingleEdit::Hide) {
                if (edited_counts != source_counts) return std::unexpected{"hide changed topology"};
                if (meshes.GetHiddenElements(original.GetStoreId(), Element::Face).Count() != selected_count)
                    return std::unexpected{"hide did not hide selected faces"};
            }
            if (scene.empty() && !select_all && (single_edit == SingleEdit::DissolveLimited || single_edit == SingleEdit::DissolveDelimited) &&
                (edited_counts[0] != vertices || edited_counts[1] + 1u != faces)) return std::unexpected{"limited dissolve did not join the selected faces"};
            if (single_edit == SingleEdit::Unsubdivide || single_edit == SingleEdit::Decimate) {
                if (edited_counts[0] >= vertices) return std::unexpected{"simplification did not remove selected vertices"};
            }
            if (single_edit == SingleEdit::BeautifyFaces) {
                if (edited_counts != source_counts) return std::unexpected{"beautification changed element counts"};
                if (edited == base) return std::unexpected{"beautification did not rotate the diagonal"};
            }
            constexpr std::array position_edits{
                SingleEdit::ToSphere,
                SingleEdit::PushPull,
                SingleEdit::Shear,
                SingleEdit::Warp,
                SingleEdit::Bend,
                SingleEdit::Randomize,
                SingleEdit::VertexSlide,
                SingleEdit::EdgeSlide,
                SingleEdit::ShrinkFatten,
            };
            if (std::ranges::find(position_edits, single_edit) != position_edits.end()) {
                if (edited_counts != source_counts) return std::unexpected{label + " changed topology"};
                if (edited == base) return std::unexpected{label + " did not change positions"};
            }
            if (single_edit == SingleEdit::RotateUVs) {
                if (edited_counts[0] != vertices || edited_counts[1] != faces) return std::unexpected{"UV rotation changed topology"};
                if (!(meshes.Get(original.GetStoreId()).CornerAttributes & MeshAttributeBit_TexCoord0)) return std::unexpected{"UV rotation requires UV0"};
                if (edited == base) return std::unexpected{"UV rotation did not change attributes"};
            }
            if (single_edit == SingleEdit::MakePlanarFaces && !select_all) {
                const auto mesh = GetMesh(r, entity);
                const auto center = mesh.CalcFaceCentroid(face), normal = mesh.GetNormal(face);
                for (const auto v : mesh.fv_range(face))
                    if (std::abs(Dot(mesh.GetPosition(v) - center, normal)) > 1e-5f) return std::unexpected{"selected face did not become planar"};
            }
            if (single_edit == SingleEdit::SharpFace) {
                const auto reverse_ms = Milliseconds(toggle_sharp);
                row("sharp_face_warm_reverse", reverse_ms, selected_count);
                p.Navigate(edited);
            }
            if (auto result = render_step(std::string{"render_"} + label); !result) return result;
            if (render)
                if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE"); !result) return result;
            edit_phase = {label.c_str(), edit_ms};
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
        if (r.Context.get<const MeshStore>().Get(original.GetStoreId()).DirtyGroupRoot != InvalidOffset) return std::unexpected{"edit mode exit left stale LOD groups"};
        if (position && (r.Context.get<const MeshStore>().Get(original.GetStoreId()).PositionDirtyRoot != InvalidOffset)) return std::unexpected{"position edit exit retained coarse dirt"};
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
