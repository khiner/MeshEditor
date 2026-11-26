#include "BenchTime.h"
#include "MeshWalk.h"
#include "Paths.h"
#include "Profile.h"
#include "TestPaths.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshPipelines.h"
#include "mesh/Neighborhood.h"
#include "mesh/Primitives.h"
#include "mesh/SpatialFaceWork.h"
#include "metal/MetalCpp.h"
#include "object/ObjectOps.h"
#include "procedure/Macros.h"
#include "procedure/Procedure.h"
#include "project/Recording.h"
#include "render/Instance.h"
#include "render/Textures.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"

#include <charconv>
#include <print>

namespace {
using Result = std::expected<void, std::string>;

Mesh::EH Edge(const Mesh &mesh, Mesh::HH h) { return mesh.GetConnectivity().Edge(h); }

double P95(std::span<double> samples) {
    std::ranges::sort(samples);
    return samples[size_t(std::ceil(0.95 * double(samples.size()))) - 1u];
}

void ReportLibraries(const mtl::LibraryCache &cache, const char *phase) {
    std::println("pipeline_archive,{},{},{}", cache.ArchiveHitCount(), cache.CompileMissCount(), phase);
    std::println("shader_libraries,{},{},{}", cache.BinaryLibraryLoadCount(), cache.SourceLibraryCompileCount(), phase);
}

Result Capture(state::Registry &r, const char *variable) {
    const char *path = std::getenv(variable);
    if (!path) return {};
    const auto &image = r.get<const Viewport>(FirstViewport(r)).Targets.Screen->FinalColorImage;
    const auto rgba = ReadbackImageRgba8(r.Resources.get<const mtl::Context>(), image, 0u, 0u, image.Extent);
    const auto file = std::unique_ptr<FILE, decltype(&std::fclose)>{std::fopen(path, "wb"), &std::fclose};
    if (!file) return std::unexpected{"failed to open frame capture"};
    if (std::fwrite(rgba.data(), 1u, rgba.size(), file.get()) != rgba.size()) return std::unexpected{"failed to write frame capture"};
    return {};
}

// An operator the benchmark prepares an input or a selection for, or Plain for one applied to the selected elements of a sphere.
enum class SingleEdit { Plain,
                        Inset,
                        SeparateSelected,
                        SpatialPlane,
                        SpatialCut,
                        EdgeSplit,
                        NewEdge,
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
                        DissolveFace,
                        DissolveLimited,
                        DissolveDelimited,
                        SharpFace,
                        MergeDistance,
                        MergeCenter,
                        MergeCorners,
                        MergeCollapse,
                        VertexSlide,
                        EdgeSlide,
                        RecalculateNormals,
                        BeautifyFaces,
                        SnapSymmetry,
                        DuplicateObject,
                        MakePlanarFaces,
                        RotateUVs,
                        SplitNonplanarFaces,
                        SplitConcaveFaces,
                        DissolveDegenerate,
                        Position,
                        Join };
template<typename A> action::Action Edit(A a) { return action::MakeAction(std::move(a)); }
// `Make` builds the action of an edit applied in one step, or is null for an edit the benchmark drives itself.
// `Stage` emits step `i` of an edit applied as a gesture, over one element or over all.
// `Kinds` are the nodes of a chain appended in Object mode, whose gesture changes a parameter of its first or last node.
struct EditSpec {
    SingleEdit Op;
    const char *Argument;
    Element Domain = Element::Face;
    action::Action (*Make)() = nullptr;
    void (*Stage)(uint32_t i, bool all) = nullptr;
    std::span<const uint16_t> Kinds{};
};
constexpr auto Smooth = procedure::KindOf<action::mesh::SmoothVertices>, Fatten = procedure::KindOf<action::mesh::ShrinkFatten>, Poke = procedure::KindOf<action::mesh::Poke>;
constexpr std::array PositionChain{Smooth, Fatten, Smooth, Fatten, Smooth, Fatten, Smooth, Fatten};
constexpr std::array TopologyChain{Poke, Poke, Poke};
constexpr std::array MixedChain{Smooth, Poke, Fatten, Smooth, Poke, Fatten};
constexpr std::array EditSpecs{
    EditSpec{SingleEdit::Inset, "inset", Element::Face, nullptr, [](uint32_t i, bool all) { action::Emit(action::mesh::Inset{.Thickness = 0.0001f * float(i + 1), .Individual = all}, action::Phase::Stage); }},
    EditSpec{SingleEdit::Position, "position", Element::Vertex, nullptr, [](uint32_t i, bool) { action::Emit(action::view::TransformElements{{.P = {0.0001f * float(i + 1), 0.f, 0.f}}}, action::Phase::Stage); }},
    EditSpec{SingleEdit::Join, "join", Element::Face, [] { return Edit(action::mesh::TrisToQuads{}); }},
    EditSpec{SingleEdit::Plain, "position-chain", Element::Face, nullptr, nullptr, PositionChain},
    EditSpec{SingleEdit::Plain, "topology-chain", Element::Face, nullptr, nullptr, TopologyChain},
    EditSpec{SingleEdit::Plain, "mixed-chain", Element::Face, nullptr, nullptr, MixedChain},
    EditSpec{SingleEdit::SeparateSelected, "separate-selected"},
    EditSpec{SingleEdit::SpatialPlane, "spatial-plane"},
    EditSpec{SingleEdit::SpatialCut, "spatial-cut"},
    EditSpec{SingleEdit::EdgeSplit, "edge-split", Element::Edge, [] { return Edit(action::mesh::EdgeSplit{}); }},
    EditSpec{SingleEdit::Plain, "bevel-edge", Element::Edge, [] { return Edit(action::mesh::Bevel{.Width = 0.01f}); }},
    EditSpec{SingleEdit::Plain, "bevel-vertex", Element::Vertex, [] { return Edit(action::mesh::Bevel{.Width = 0.01f, .Vertices = true}); }},
    EditSpec{SingleEdit::Plain, "dissolve-edge", Element::Edge, [] { return Edit(action::mesh::Dissolve{action::mesh::DissolveMode::Edges}); }},
    EditSpec{SingleEdit::Plain, "edge-rotate", Element::Edge, [] { return Edit(action::mesh::EdgeRotate{}); }},
    EditSpec{SingleEdit::Plain, "extrude-edge", Element::Edge, [] { return Edit(action::mesh::Extrude{action::mesh::ExtrudeMode::Edges}); }},
    EditSpec{SingleEdit::Plain, "extrude-region", Element::Face, [] { return Edit(action::mesh::Extrude{}); }},
    EditSpec{SingleEdit::Plain, "extrude-vertex", Element::Vertex, [] { return Edit(action::mesh::Extrude{action::mesh::ExtrudeMode::Vertices}); }},
    EditSpec{SingleEdit::NewEdge, "new-edge", Element::Vertex, [] { return Edit(action::mesh::Fill{}); }},
    EditSpec{SingleEdit::Plain, "duplicate-vertex", Element::Vertex, [] { return Edit(action::mesh::Duplicate{}); }},
    EditSpec{SingleEdit::Plain, "duplicate-edge", Element::Edge, [] { return Edit(action::mesh::Duplicate{}); }},
    EditSpec{SingleEdit::Plain, "duplicate-face", Element::Face, [] { return Edit(action::mesh::Duplicate{}); }},
    EditSpec{SingleEdit::Plain, "split-vertex", Element::Vertex, [] { return Edit(action::mesh::Split{}); }},
    EditSpec{SingleEdit::Plain, "split-edge", Element::Edge, [] { return Edit(action::mesh::Split{}); }},
    EditSpec{SingleEdit::Plain, "split-face", Element::Face, [] { return Edit(action::mesh::Split{}); }},
    EditSpec{SingleEdit::Plain, "subdivide-edge", Element::Edge, [] { return Edit(action::mesh::Subdivide{.Cuts = 1u}); }},
    EditSpec{SingleEdit::Plain, "loop-cut", Element::Edge, [] { return Edit(action::nodes::AppendMacro{procedure::LoopCutMacro, {{1u, 0u, 0u, 0u}}}); }},
    EditSpec{SingleEdit::FillFace, "fill-face", Element::Edge, [] { return Edit(action::mesh::Fill{}); }},
    EditSpec{SingleEdit::FillHoles, "fill-holes", Element::Edge, [] { return Edit(action::nodes::AppendMacro{procedure::FillHolesMacro, {{4u, 0u, 0u, 0u}}}); }},
    EditSpec{SingleEdit::Bridge, "bridge", Element::Edge, [] { return Edit(action::mesh::BridgeEdgeLoops{}); }},
    EditSpec{SingleEdit::GridFill, "grid-fill", Element::Edge, [] { return Edit(action::mesh::GridFill{2u}); }},
    EditSpec{SingleEdit::SpaceEvenly, "space-evenly", Element::Edge, [] { return Edit(action::mesh::SpaceEvenly{}); }},
    EditSpec{SingleEdit::RelaxEdgeLoops, "relax-edge-loops", Element::Edge, [] { return Edit(action::mesh::RelaxEdgeLoops{}); }},
    EditSpec{SingleEdit::Flatten, "flatten", Element::Edge, [] { return Edit(action::mesh::Flatten{}); }},
    EditSpec{SingleEdit::CurveBetweenSelected, "curve-between-selected", Element::Vertex, [] { return Edit(action::mesh::CurveBetweenSelected{}); }},
    EditSpec{SingleEdit::Circularize, "circularize", Element::Edge, [] { return Edit(action::mesh::Circularize{}); }},
    EditSpec{SingleEdit::DeleteLoose, "delete-loose", Element::Vertex, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::Loose}); }},
    EditSpec{SingleEdit::Plain, "delete-edge", Element::Edge, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::Edges}); }},
    EditSpec{SingleEdit::Plain, "delete-only-edge-faces", Element::Edge, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::OnlyEdgesAndFaces}); }},
    EditSpec{SingleEdit::Plain, "delete-faces", Element::Face, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::Faces}); }},
    EditSpec{SingleEdit::Plain, "delete-only-faces", Element::Face, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces}); }},
    EditSpec{SingleEdit::Plain, "delete-vertex", Element::Vertex, [] { return Edit(action::mesh::Delete{action::mesh::DeleteMode::Vertices}); }},
    EditSpec{SingleEdit::Plain, "dissolve-vertex", Element::Vertex, [] { return Edit(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Vertices}); }},
    EditSpec{SingleEdit::DissolveFace, "dissolve-face", Element::Face, [] { return Edit(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Faces}); }},
    EditSpec{SingleEdit::DissolveLimited, "dissolve-limited", Element::Face, [] { return Edit(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = 3.14159f}); }},
    EditSpec{SingleEdit::DissolveDelimited, "dissolve-delimited", Element::Face, [] { return Edit(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = 3.14159f, .DelimitMaterials = true, .DelimitSharpEdges = true, .DelimitUVs = true}); }},
    EditSpec{SingleEdit::SharpFace, "sharp-face"},
    EditSpec{SingleEdit::MergeDistance, "merge-distance", Element::Vertex},
    EditSpec{SingleEdit::MergeCenter, "merge-center", Element::Vertex, [] { return Edit(action::mesh::Merge{.Mode = action::mesh::MergeMode::Center}); }},
    EditSpec{SingleEdit::MergeCorners, "merge-corners", Element::Vertex, [] { return Edit(action::mesh::Merge{.Mode = action::mesh::MergeMode::Center}); }},
    EditSpec{SingleEdit::MergeCollapse, "merge-collapse", Element::Vertex, [] { return Edit(action::mesh::Merge{.Mode = action::mesh::MergeMode::Collapse}); }},
    EditSpec{SingleEdit::DissolveDegenerate, "dissolve-degenerate", Element::Vertex},
    EditSpec{SingleEdit::RecalculateNormals, "recalculate-normals", Element::Face, [] { return Edit(action::mesh::RecalculateNormals{}); }},
    EditSpec{SingleEdit::SnapSymmetry, "snap-symmetry", Element::Face, [] { return Edit(action::mesh::SnapSymmetry{.Threshold = .001f}); }},
    EditSpec{SingleEdit::Plain, "hide", Element::Face, [] { return Edit(action::mesh::Hide{}); }},
    EditSpec{SingleEdit::DuplicateObject, "duplicate-object"},
    EditSpec{SingleEdit::Plain, "unsubdivide", Element::Face, [] { return Edit(action::mesh::Unsubdivide{1u}); }},
    EditSpec{SingleEdit::Plain, "simplify", Element::Face, [] { return Edit(action::mesh::Simplify{.Ratio = .5f}); }},
    EditSpec{SingleEdit::BeautifyFaces, "beautify-faces", Element::Face, [] { return Edit(action::mesh::BeautifyFaces{}); }},
    EditSpec{SingleEdit::Plain, "to-sphere", Element::Face, [] { return Edit(action::mesh::ToSphere{}); }},
    EditSpec{SingleEdit::Plain, "push-pull", Element::Face, [] { return Edit(action::mesh::PushPull{.Distance = .01f}); }},
    EditSpec{SingleEdit::Plain, "shear", Element::Face, [] { return Edit(action::mesh::Shear{}); }},
    EditSpec{SingleEdit::Plain, "warp", Element::Face, [] { return Edit(action::mesh::Warp{.Angle = 1.f}); }},
    EditSpec{SingleEdit::Plain, "bend", Element::Face, [] { return Edit(action::mesh::Bend{.Clamp = false}); }},
    EditSpec{SingleEdit::Plain, "randomize", Element::Face, [] { return Edit(action::mesh::Randomize{.Amount = .01f, .Uniform = .5f, .Normal = .5f, .Seed = 7u}); }},
    EditSpec{SingleEdit::VertexSlide, "vertex-slide", Element::Face, [] { return Edit(action::mesh::VertexSlide{.Factor = .1f, .Even = true}); }},
    EditSpec{SingleEdit::EdgeSlide, "edge-slide", Element::Edge, [] { return Edit(action::mesh::EdgeSlide{.Factor = .1f, .Even = true}); }},
    EditSpec{SingleEdit::Plain, "shrink-fatten", Element::Vertex, [] { return Edit(action::mesh::ShrinkFatten{.Distance = .01f, .Even = true}); }},
    EditSpec{SingleEdit::Plain, "smooth-vertices", Element::Vertex, [] { return Edit(action::mesh::SmoothVertices{}); }},
    EditSpec{SingleEdit::RotateUVs, "rotate-uvs", Element::Face, [] { return Edit(action::mesh::CycleCornerAttribute{}); }},
    EditSpec{SingleEdit::MakePlanarFaces, "planar-faces", Element::Face, [] { return Edit(action::mesh::MakePlanarFaces{}); }},
    EditSpec{SingleEdit::SplitNonplanarFaces, "split-nonplanar", Element::Face, [] { return Edit(action::mesh::SplitFaces{action::mesh::SplitTarget::Planar}); }},
    EditSpec{SingleEdit::SplitConcaveFaces, "split-concave", Element::Face, [] { return Edit(action::mesh::SplitFaces{action::mesh::SplitTarget::Convex}); }},
    EditSpec{SingleEdit::Plain, "wireframe", Element::Face, [] { return Edit(action::mesh::Wireframe{}); }},
};

// One run: the mesh, the edit, and how much of the mesh the edit covers.
struct Run {
    uint32_t Slices{};
    std::filesystem::path Scene{};
    uint32_t Updates{3u};
    bool Render{}, RefitProbe{};
    uint32_t Width{128u}, Height{128u};
    const EditSpec *Edit{};
    // `one` element or `all` elements of the operator's domain, or the `first` or `last` node of a chain.
    std::string_view Selection{"one"};
    // Also times a restore of the edit after the history evicts its state, which saves the project.
    bool Cold{};
};
// The phases a run timed, which the table reads in place of printed rows.
struct Report {
    uint32_t Faces{};
    std::vector<std::pair<std::string, double>> Phases;
};
// A run's engine and the mesh it edits.
struct Session {
    state::Registry &R;
    project::Project &P;
    const Run &Config;
    Report *Table;
    state::Entity Entity{state::Null};
    uint32_t Vertices{}, Faces{};
    size_t Selected{};
    // The history node and mesh storage state the edit starts from, and the mesh's vertex and face counts there.
    uint32_t Base{};
    store::Hash128 State{};
    std::array<uint32_t, 2> Source{};
};

// Progress on stderr, which a table run omits.
template<typename... Args> void Note(const Session &s, std::format_string<Args...> format, Args &&...args) {
    if (!s.Table) std::println(stderr, format, std::forward<Args>(args)...);
}
std::array<uint32_t, 2> Counts(const Session &s) {
    const auto mesh = GetMesh(s.R, s.Entity);
    return {mesh.VertexCount(), mesh.FaceCount()};
}
void Row(const Session &s, std::string_view name, double ms, size_t count) {
    if (s.Table) s.Table->Phases.emplace_back(std::string{name}, ms);
    else std::println("mesh,{},{},{},{},{:.6f},{}", s.Vertices, s.Faces, count, name, ms, uint64_t(s.R.Resources.get<const mtl::Context>().Device->currentAllocatedSize()));
}
void Draw(const Session &s, std::string_view name, size_t count) {
    Row(s, name, Milliseconds([&] { SubmitViewports(s.R); WaitForRender(s.R); }), count);
}
Result RenderStep(const Session &s, std::string_view name) {
    if (!s.Config.Render) return {};
    if (!s.R.get<const Viewport>(FirstViewport(s.R)).RenderRequested) return std::unexpected{std::string{name} + " did not request a rendered frame"};
    Draw(s, name, s.Selected);
    return {};
}

// Renders the mesh and commits the history node the edit starts from.
void Baseline(Session &s) {
    Note(s, "Rendering the baseline...");
    SubmitViewports(s.R);
    WaitForRender(s.R);
    s.P.History.Commit("benchmark selection", {});
    s.Base = *s.P.History.Present;
    s.State = s.R.Resources.get<MeshStore>().StateHash();
    s.Source = Counts(s);
    if (!s.Table) ReportLibraries(s.R.Resources.get<const mtl::LibraryCache>(), "baseline");
    profile::ClearStats();
    profile::Enable(!s.Table && !std::getenv("MESHEDITOR_EDIT_BENCH_UNPROFILED"));
}
// Fails for an edit that left the history node or the mesh storage state it started from.
Result Changed(const Session &s) {
    if (*s.P.History.Present == s.Base || s.R.Resources.get<MeshStore>().StateHash() == s.State) return std::unexpected{std::string{s.Config.Edit->Argument} + " changed nothing"};
    return {};
}

// Times each step of a gesture with the frame that cooks it, then the commit.
// `probe` runs after the first step, outside the timings.
Result Gesture(Session &s, auto &&stage, auto &&probe) {
    std::vector<double> warm;
    for (uint32_t i = 0; i < s.Config.Updates; ++i) {
        const auto elapsed = Milliseconds([&] {
            stage(i);
            s.P.Frame();
        });
        Row(s, "update" + std::to_string(i), elapsed, s.Selected);
        if (i) warm.push_back(elapsed);
        if (i == 0u && !s.Table && std::getenv("MESHEDITOR_EDIT_BENCH_PROFILE_FIRST")) {
            std::println("First preview profile");
            profile::Report();
            profile::ClearStats();
        }
        if (i == 0u)
            if (auto result = probe(); !result) return result;
        if (auto result = RenderStep(s, "render" + std::to_string(i)); !result) return result;
        if (s.Config.Render && i == 1u)
            if (auto result = Capture(s.R, "MESHEDITOR_EDIT_BENCH_CAPTURE"); !result) return result;
    }
    if (!warm.empty()) Row(s, "warm_preview_p95", P95(warm), s.Selected);
    Row(s, "commit", Milliseconds([&] { action::Commit(); s.P.Frame(); }), s.Selected);
    return Changed(s);
}

// Times navigation to the baseline and back to the edit, then with Cold a restore of the edit once the history evicted its state.
// A gesture's history also replays to the same state.
Result Navigation(Session &s) {
    auto &p = s.P;
    const auto edited = *p.History.Present;
    const auto edited_counts = Counts(s);
    Row(s, "undo", Milliseconds([&] { p.Navigate(s.Base); }), s.Selected);
    if (Counts(s) != s.Source) return std::unexpected{"undo did not restore source"};
    if (auto result = RenderStep(s, "render_undo"); !result) return result;
    Row(s, "redo", Milliseconds([&] { p.Navigate(edited); }), s.Selected);
    if (Counts(s) != edited_counts) return std::unexpected{"redo did not restore edit"};
    if (auto result = RenderStep(s, "render_redo"); !result) return result;
    if (s.Config.Edit->Stage || !s.Config.Edit->Kinds.empty()) {
        if (const auto replay_error = p.History.ValidateReplay(); !replay_error.empty()) return std::unexpected{replay_error};
    }
    if (!s.Config.Cold) return {};
    if (!p.Save()) return std::unexpected{"project save failed"};
    p.Navigate(s.Base);
    p.History.Evict(0);
    if (p.History.Nodes[edited].Hot) return std::unexpected{"eviction kept the edit's state"};
    Row(s, "cold_redo", Milliseconds([&] { p.Navigate(edited); }), s.Selected);
    return {};
}

// Adds the mesh the edit runs on: the operator's own input, the scene file's largest mesh, or a UV sphere.
Result PlaceScene(Session &s) {
    auto &r = s.R;
    auto &p = s.P;
    const auto &scene = s.Config.Scene;
    const auto slices = s.Config.Slices;
    const auto single_edit = s.Config.Edit->Op;
    Note(s, "Preparing benchmark scene...");
    const auto load_begin = std::chrono::steady_clock::now();
    const auto place = [&](MeshSource source) {
        const auto [entity, instance] = AddMesh(r, CreateMesh(r, std::move(source)).StoreId, MeshInstanceCreateInfo{});
        p.Settle();
        project::Do(p, action::MakeAction(action::selection::Select{instance}));
    };
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
        place(std::move(source));
    } else if (scene.empty() && single_edit == SingleEdit::RecalculateNormals) {
        MeshSource source;
        const auto cube = primitive::CreateMesh(primitive::Cube{}).Data;
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
        place(std::move(source));
    } else if (scene.empty() && (single_edit == SingleEdit::Bridge || single_edit == SingleEdit::GridFill || single_edit == SingleEdit::SpaceEvenly || single_edit == SingleEdit::RelaxEdgeLoops || single_edit == SingleEdit::Flatten || single_edit == SingleEdit::CurveBetweenSelected || single_edit == SingleEdit::Circularize || single_edit == SingleEdit::DissolveDelimited || single_edit == SingleEdit::RotateUVs || single_edit == SingleEdit::VertexSlide || single_edit == SingleEdit::EdgeSlide)) {
        MeshSource source{.Data = primitive::CreateMesh(primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}).Data};
        struct Strip {
            SingleEdit Edit;
            bool Closed;
            std::vector<vec3> Points;
        };
        const Strip strips[]{
            {SingleEdit::GridFill, true, {{0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {2, 1, 2}, {2, 2, 2}, {1, 2, 2}, {0, 2, 2}, {0, 1, 2}}},
            {SingleEdit::Circularize, true, {{0, 0, 2}, {2, 0, 2}, {3, 1, 2}, {2, 3, 2}, {-1, 2, 2}}},
            {SingleEdit::CurveBetweenSelected, false, {{0, 0, 2}, {1, 1, 2}, {2, 0, 2}, {3, -1, 2}, {4, 0, 2}}},
            {SingleEdit::Flatten, false, {{0, 0, 2}, {2, 0, 2}, {2, 2, 3}, {0, 2, 2}}},
            {SingleEdit::SpaceEvenly, false, {{0, 0, 2}, {.25f, 1, 2}, {2, 0, 2}, {3, 2, 2}, {5, 0, 2}}},
            {SingleEdit::RelaxEdgeLoops, false, {{0, 0, 2}, {.25f, 1, 2}, {2, 0, 2}, {3, 2, 2}, {5, 0, 2}}},
        };
        const auto first = uint32_t(source.Data.Positions.size());
        if (single_edit == SingleEdit::Bridge) {
            source.Data.Positions.insert(source.Data.Positions.end(), {{0, 0, 2}, {1, 0, 2}, {0, 1, 2}, {1, 1, 2}});
            source.Data.Edges = {{first, first + 1u}, {first + 2u, first + 3u}};
        } else if (const auto strip = std::ranges::find(strips, single_edit, &Strip::Edit); strip != std::end(strips)) {
            const auto n = uint32_t(strip->Points.size());
            source.Data.Positions.append_range(strip->Points);
            for (uint32_t i = 0u; i + (strip->Closed ? 0u : 1u) < n; ++i) source.Data.Edges.push_back({first + i, first + (i + 1u) % n});
        }
        auto &uvs = source.Attrs.TexCoords0.emplace();
        for (const auto position : source.Data.Positions) uvs.push_back({position.x, position.y});
        place(std::move(source));
    } else if (scene.empty()) project::Do(p, action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
    else {
        project::Do(p, action::MakeAction(action::io::LoadGltf{scene}));
        state::Entity largest = state::Null;
        uint32_t count = 0u;
        for (const auto [object, instance] : r.view<const Instance>().each()) {
            if (!HasMesh(r, instance.Entity)) continue;
            const auto mesh = GetMesh(r, instance.Entity);
            if (mesh.FaceCount() > count) {
                largest = object;
                count = mesh.FaceCount();
            }
        }
        if (largest == state::Null) return std::unexpected{"scene contains no face mesh"};
        project::Do(p, action::MakeAction(action::selection::Select{largest}));
    }
    Note(s, "Scene prepared in {:.3f} s.", std::chrono::duration<double>(std::chrono::steady_clock::now() - load_begin).count());
    return {};
}

// Times a gesture that changes a parameter of the first or last node of a chain appended to the mesh's procedure.
Result BenchChain(Session &s) {
    using namespace action::mesh;
    auto &r = s.R;
    auto &p = s.P;
    s.Entity = GetActiveMeshEntity(r);
    std::vector<state::Entity> nodes;
    for (const auto kind : s.Config.Edit->Kinds) {
        const auto display = r.get<const procedure::GeometryProcedure>(s.Entity).DisplayNode;
        const auto result = *procedure::ResultSocket(r, display);
        project::Do(p, action::MakeAction(action::nodes::AddNode{s.Entity, action::nodes::KindNode{kind}}));
        nodes.push_back(r.get<const procedure::NetworkNodes>(s.Entity).Last);
        project::Do(p, action::MakeAction(action::nodes::Link{s.Entity, display, result, nodes.back(), 0u}));
        project::Do(p, action::MakeAction(action::nodes::SetDisplay{s.Entity, nodes.back()}));
    }
    const auto node = s.Config.Selection == "first" ? nodes.front() : nodes.back();
    const auto kind = r.get<const procedure::ProcedureNode>(node).Kind;
    const auto counts = Counts(s);
    s.Vertices = counts[0];
    s.Selected = s.Faces = counts[1];
    Baseline(s);
    const auto stage = [&](uint32_t i) {
        const float value = .01f * float(i + 1);
        if (kind == Smooth) action::Emit(action::UpdateOf<&SmoothVertices::Factor>(node, value), action::Phase::Stage);
        else if (kind == Fatten) action::Emit(action::UpdateOf<&ShrinkFatten::Distance>(node, value), action::Phase::Stage);
        else action::Emit(action::UpdateOf<&action::mesh::Poke::Offset>(node, value), action::Phase::Stage);
    };
    if (auto result = Gesture(s, stage, [] { return Result{}; }); !result) return result;
    return Navigation(s);
}

// Times an Edit mode operator on a selection of the mesh: a gesture's steps and commit, or one committed action.
Result BenchEdit(Session &s) {
    auto &r = s.R;
    auto &p = s.P;
    const auto &run = s.Config;
    const auto &edit = *run.Edit;
    const auto &scene = run.Scene;
    const auto single_edit = edit.Op;
    const bool select_all = run.Selection == "all", join = single_edit == SingleEdit::Join, verbose = !s.Table;
    std::string label = edit.Argument;
    std::ranges::replace(label, '-', '_');
    const bool filling = single_edit == SingleEdit::FillFace || single_edit == SingleEdit::FillHoles;
    const bool merging = single_edit == SingleEdit::MergeDistance || single_edit == SingleEdit::MergeCenter || single_edit == SingleEdit::MergeCorners ||
        single_edit == SingleEdit::MergeCollapse || single_edit == SingleEdit::DissolveDegenerate;
    if (select_all && (join || filling || merging)) return std::unexpected{"join, fill and merge benchmarks count a paired or filled selection"};
    profile::ClearStats();
    profile::Enable(verbose);
    const auto enter_ms = Milliseconds([&] { project::Do(p, action::MakeAction(action::view::SetInteractionMode{InteractionMode::Edit})); });
    if (verbose) {
        profile::ReportCpuPhase("Edit mode entry");
        profile::Report();
    }
    profile::ClearStats();
    Note(s, "Entered Edit mode in {:.3f} ms; switching to face selection...", enter_ms);
    const auto mode_ms = Milliseconds([&] { project::Do(p, action::MakeAction(action::view::SetEditMode{.Mode = Element::Face})); });
    if (verbose) profile::ReportCpuPhase("Face selection mode");
    profile::Enable(false);
    Note(s, "Face selection mode prepared in {:.3f} ms.", mode_ms);
    const auto entity = s.Entity = GetActiveMeshEntity(r);
    auto &meshes = r.Resources.get<MeshStore>();
    const auto original = GetMesh(r, entity);
    const auto drawn = [&] { return *DrawnStoreId(r, entity); };
    const auto faces = original.FaceCount(), vertices = original.VertexCount();
    s.Vertices = vertices;
    s.Faces = faces;
    const auto find_face = [&]() -> std::optional<he::FH> {
        if (filling) {
            const auto &pipelines = GetMeshPipelines(r);
            for (uint32_t delta = 0u; delta < std::min(faces, 4096u); ++delta) {
                const auto candidate = original.FaceAt((faces / 2u + delta) % faces);
                if (!std::ranges::all_of(test::FaceHalfedges(original, candidate), [&](auto h) { return bool(test::Opposite(original, h)); })) continue;
                if (single_edit != SingleEdit::FillHoles) return candidate;
                const auto corners = std::ranges::to<std::vector>(test::FaceVertices(original, candidate) | std::views::transform([](he::VH v) { return *v; }));
                const auto edges = VertexEdges(meshes, pipelines, original.GetStoreId(), corners);
                if (std::ranges::all_of(edges.Items, [&](const auto &edge) { return bool(test::Opposite(original, test::Halfedge(original, he::EH{edge.Edge}))); })) return candidate;
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
        const vec3 center = test::Centroid(original, face);
        const auto eye = center + vec3{0.2f * distance, distance, 0.5f * distance};
        r.patch<ViewCameraState>(FirstViewport(r), [&](ViewCameraState &view) {
            const auto lens = view.Active.Data;
            ClearLookThrough(view);
            view.Active = ViewCamera{eye, center, lens};
        });
    }
    const auto selected_face_halfedge = *test::FaceHalfedges(original, face).begin();
    std::vector<uint32_t> fill_edges;
    if (filling) {
        for (const auto h : test::FaceHalfedges(original, face)) fill_edges.push_back(*Edge(original, h));
    }
    std::vector<uint32_t> selected{original.FaceOrdinal(face)};
    const bool pair_faces = join || single_edit == SingleEdit::DissolveFace || (single_edit == SingleEdit::DissolveLimited || single_edit == SingleEdit::DissolveDelimited) || single_edit == SingleEdit::BeautifyFaces;
    if (pair_faces) {
        // The generated terrain emits each grid quad as consecutive triangles.
        const auto mate = original.FaceOrdinal(face) ^ 1u;
        for (const auto halfedge : test::FaceHalfedges(original, face)) {
            const auto opposite = test::Opposite(original, halfedge);
            if (!opposite) continue;
            if (join && (test::Valence(original, face) != 3u || test::Valence(original, test::Face(original, opposite)) != 3u || original.FaceOrdinal(test::Face(original, opposite)) != mate)) continue;
            selected.push_back(original.FaceOrdinal(test::Face(original, opposite)));
            break;
        }
        if (selected.size() != 2u) return std::unexpected{"paired-face benchmark requires adjacent faces"};
    }
    const std::array<uint32_t, 1> alternate{(selected[0] + (pair_faces ? 17u : 1u)) % faces};
    Note(s, "Selecting {} face(s) from {} vertices / {} faces...", selected.size(), vertices, faces);
    profile::ClearStats();
    profile::Enable(verbose);
    const auto select = [&](std::span<const uint32_t> ordinals, Element element = Element::Face) {
        ApplyEditSelectionLists(r, std::array{std::pair{entity, ordinals}}, element);
        p.Settle();
    };
    const auto initial_select_ms = Milliseconds([&] { select(selected); });
    if (verbose) {
        profile::ReportCpuPhase("Initial selection replacement");
        profile::Report();
    }
    profile::Enable(false);
    if (meshes.GetSelectionSummary(drawn()).SelectedCount != selected.size()) return std::unexpected{"expected selected faces"};
    select(alternate);
    const auto select_ms = Milliseconds([&] { select(selected); });
    if (meshes.GetSelectionSummary(drawn()).SelectedCount != selected.size()) return std::unexpected{"expected selected faces after sparse selection"};
    if (filling) project::Do(p, action::MakeAction(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces}));
    if (filling) {
        const auto current = GetMesh(r, entity);
        const auto boundary = [&](uint32_t edge) { return !test::Opposite(current, test::Halfedge(current, he::EH{edge})); };
        uint32_t total = 0u, on_hole = 0u;
        for (const auto edge : current.edges()) total += boundary(*edge);
        for (const auto edge : fill_edges) on_hole += boundary(edge);
        Note(s, "Hole boundary: {} edges total, {}/{} hole edges on it.", total, on_hole, fill_edges.size());
    }
    if (edit.Domain != Element::Face) {
        const auto element = edit.Domain;
        const bool use_vertex = element == Element::Vertex;
        const auto edge_ordinal = [&](const Mesh &mesh, he::EH e) { return meshes.LiveElementOrdinal(mesh.GetStoreId(), MeshStore::ElementDomain::Edge, ElementHandle{*e}); };
        project::Do(p, action::MakeAction(action::view::SetEditMode{.Mode = element}));
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
            for (const auto handle : fill_edges) selected_elements.push_back(edge_ordinal(current, he::EH{handle}));
        } else {
            const auto first = use_vertex ? original.VertexOrdinal(test::ToVertex(original, selected_face_halfedge)) :
                                            edge_ordinal(original, Edge(original, selected_face_halfedge));
            selected_elements.push_back(first);
        }
        if (single_edit == SingleEdit::EdgeSplit) {
            const auto next = original.GetConnectivity().Next(selected_face_halfedge);
            selected_elements.push_back(edge_ordinal(original, Edge(original, next)));
        }
        if (merging && single_edit != SingleEdit::MergeCorners) selected_elements.push_back(original.VertexOrdinal(test::FromVertex(original, selected_face_halfedge)));
        if (single_edit == SingleEdit::NewEdge || single_edit == SingleEdit::MergeCorners) {
            const auto &c = original.GetConnectivity();
            selected_elements.push_back(original.VertexOrdinal(test::ToVertex(original, c.Next(c.Next(selected_face_halfedge)))));
        }
        select(selected_elements, element);
        if (single_edit == SingleEdit::DeleteLoose) {
            project::Do(p, action::MakeAction(action::mesh::Extrude{action::mesh::ExtrudeMode::Vertices}));
            std::optional<uint32_t> copy;
            meshes.GetSelectedElements(drawn(), Element::Vertex).ForEach([&](uint32_t v) { if (!copy) copy = v; });
            if (!copy) return std::unexpected{"loose-delete setup did not extrude a vertex"};
            selected_elements.push_back(*copy);
            select(selected_elements, Element::Vertex);
        }
        const auto selected_count = meshes.GetSelectionSummary(drawn()).SelectedCount;
        if (selected_count != selected_elements.size()) std::println(stderr, "operator selection: expected {}, got {}", selected_elements.size(), selected_count);
        if (selected_count != selected_elements.size()) return std::unexpected{"unexpected operator selection"};
        if (filling) {
            const auto selected_edges = meshes.GetSelectedElements(drawn(), Element::Edge);
            for (const auto edge : fill_edges)
                if (!selected_edges.Contains(edge)) return std::unexpected{"fill did not select a hole edge"};
        }
    }
    if (single_edit == SingleEdit::MakePlanarFaces || single_edit == SingleEdit::SplitNonplanarFaces || single_edit == SingleEdit::SplitConcaveFaces) {
        if (test::Valence(original, face) < 4u) return std::unexpected{"planar benchmark needs a polygon with at least four corners"};
        vec3 offset = test::Normal(original, face) * 0.01f;
        if (single_edit == SingleEdit::SplitConcaveFaces) {
            if (test::Valence(original, face) != 4u) return std::unexpected{"concave benchmark requires a quad"};
            std::array<vec3, 4> points;
            uint32_t i = 0u;
            for (const auto v : test::FaceVertices(original, face)) points[i++] = test::Position(original, v);
            offset = (points[1] + points[3]) * 0.375f + points[2] * 0.25f - points[0];
        }
        project::Do(p, action::MakeAction(action::view::SetEditMode{.Mode = Element::Vertex}));
        const std::array corner{original.VertexOrdinal(test::ToVertex(original, selected_face_halfedge))};
        select(corner, Element::Vertex);
        action::Emit(action::view::TransformElements{{.P = offset}}, action::Phase::Stage);
        p.Frame();
        action::Commit();
        p.Frame();
        project::Do(p, action::MakeAction(action::view::SetEditMode{.Mode = Element::Face}));
        select(selected);
    }
    if (single_edit == SingleEdit::SnapSymmetry) {
        action::Emit(action::view::TransformElements{{.P = {.0001f, 0.f, 0.f}}}, action::Phase::Stage);
        p.Frame();
        action::Commit();
        p.Frame();
    }
    if (single_edit == SingleEdit::RecalculateNormals) project::Do(p, action::MakeAction(action::mesh::FlipNormals{}));
    if (select_all) {
        project::Do(p, action::MakeAction(action::selection::SelectAll{}));
        p.Settle();
    }
    const auto selected_count = meshes.GetSelectionSummary(drawn()).SelectedCount;
    if (select_all && (selected_count != GetMesh(r, entity).ElementCount(r.get<const EditMode>(state::Document).Value))) return std::unexpected{"select-all left elements unselected"};
    s.Selected = selected_count;
    Baseline(s);
    const auto base = s.Base;
    Row(s, "selection_initial_replace", initial_select_ms, selected.size());
    Row(s, "selection", select_ms, selected.size());
    if (single_edit == SingleEdit::SpatialPlane || single_edit == SingleEdit::SpatialCut) {
        const vec3 normal = Normalize(vec3{0.3f, 0.8f, 0.5f});
        const char *plane_offset = std::getenv("MESHEDITOR_SPATIAL_PLANE_OFFSET");
        float offset = plane_offset ? std::strtof(plane_offset, nullptr) : Dot(normal, test::Centroid(original, face));
        std::println("spatial_work,visited_nodes,candidate_blocks,candidate_faces,exact_faces");
        const auto spatial_counts = [&](const MeshTopologyTask &task) {
            mtl::ComputeChain chain{meshes.BufferContext()};
            SpatialFaceWork work{r, chain, task};
            chain.Submit();
            work.RecordFaces(r, chain);
            chain.Submit();
            return std::array{work.VisitedNodes, work.CandidateBlocks, work.CandidateCount, work.Count};
        };
        if (single_edit == SingleEdit::SpatialPlane) {
            const MeshTopologyTask task{.SourceId = drawn(), .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = normal, .PlaneOffset = offset};
            std::array<uint32_t, 4> counts{};
            const auto elapsed = Milliseconds([&] { counts = spatial_counts(task); });
            std::println("spatial_plane_work,{},{},{},{}", counts[0], counts[1], counts[2], counts[3]);
            Row(s, "spatial_plane", elapsed, counts[3]);
            return {};
        }
        if (const char *fraction_text = std::getenv("MESHEDITOR_SPATIAL_NEAR_MAX")) {
            const float fraction = std::strtof(fraction_text, nullptr);
            if (!(std::isfinite(fraction) && fraction > 0.f && fraction < 1.f)) return std::unexpected{"invalid near-max fraction"};
            const auto &arena = meshes.Arenas().Vertices;
            const auto vertices_data = arena.Buffer.GetSpan<Vertex>();
            const auto set = meshes.Get(drawn()).Vertices;
            float lo = INFINITY, hi = -INFINITY;
            arena.ForEach(set, [&](ElementHandle, ElementAddress address) {
                const float coordinate = Dot(normal, vertices_data[address.Value()].Position);
                lo = std::min(lo, coordinate);
                hi = std::max(hi, coordinate);
            });
            offset = hi - fraction * (hi - lo);
            std::println("spatial_cut_setup,{:.9g},{:.9g},{:.9g}", lo, hi, offset);
            const MeshTopologyTask probe{.SourceId = drawn(), .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = normal, .PlaneOffset = offset};
            const auto [nodes, blocks, candidate_faces, exact_faces] = spatial_counts(probe);
            std::println("spatial_cut_candidates,{},{},{},{}", nodes, blocks, candidate_faces, exact_faces);
            if (!(exact_faces > 0u && exact_faces <= 4096u)) return std::unexpected{"near-max cut needs 1 to 4096 exact faces"};
            profile::ClearStats();
        }
        const auto elapsed = Milliseconds([&] {
            project::Do(p, action::MakeAction(action::mesh::Bisect{.Point = normal * offset, .Normal = normal}));
        });
        const auto after = GetMesh(r, entity);
        Row(s, "spatial_cut", elapsed, after.FaceCount() - faces);
        if (std::getenv("MESHEDITOR_SPATIAL_VERIFY")) {
            const auto edited = *p.History.Present;
            const auto changed = after.FaceCount();
            Draw(s, "spatial_render_commit", changed - faces);
            const auto undo_ms = Milliseconds([&] { p.Navigate(base); });
            if (GetMesh(r, entity).FaceCount() != faces) return std::unexpected{"spatial cut undo did not restore faces"};
            Row(s, "spatial_undo", undo_ms, changed - faces);
            Draw(s, "spatial_render_undo", changed - faces);
            const auto redo_ms = Milliseconds([&] { p.Navigate(edited); });
            if (GetMesh(r, entity).FaceCount() != changed) return std::unexpected{"spatial cut redo did not restore edit"};
            Row(s, "spatial_redo", redo_ms, changed - faces);
            Draw(s, "spatial_render_redo", changed - faces);
            if (std::getenv("MESHEDITOR_SPATIAL_VERIFY_REOPEN")) {
                if (!p.Save()) return std::unexpected{"spatial cut save failed"};
                if (!p.Close()) return std::unexpected{"spatial cut close failed"};
                if (!p.Open(Paths::UserData())) return std::unexpected{"spatial cut reopen failed"};
                if (GetMesh(r, GetActiveMeshEntity(r)).FaceCount() != changed) return std::unexpected{"spatial cut reopen did not restore the edit"};
                Draw(s, "spatial_render_reopen", changed - faces);
            }
        }
        return {};
    }
    if (single_edit == SingleEdit::SeparateSelected) {
        Row(s, "separate_selected", Milliseconds([&] { project::Do(p, action::MakeAction(action::mesh::Separate{})); }), selected_count);
        return Changed(s);
    }
    if (single_edit == SingleEdit::DuplicateObject) {
        project::Do(p, action::MakeAction(action::mesh::Subdivide{2u}));
        project::Do(p, action::MakeAction(action::view::SetInteractionMode{InteractionMode::Object}));
        const auto source = GetMesh(r, entity);
        const auto before = r.view<const EvaluatedMesh>().size();
        profile::ClearStats();
        const auto elapsed = Milliseconds([&] { project::Do(p, action::MakeAction(action::object::Duplicate{})); });
        if (r.view<const EvaluatedMesh>().size() != before + 1u) return std::unexpected{"object duplicate did not create one mesh"};
        const auto clone = GetMesh(r, GetActiveMeshEntity(r));
        if (clone.GetStoreId() == source.GetStoreId() || clone.VertexCount() != source.VertexCount() || clone.FaceCount() != source.FaceCount())
            return std::unexpected{"object duplicate did not preserve edited geometry"};
        Row(s, "duplicate_object", elapsed, source.FaceCount());
        const auto undo_ms = Milliseconds([&] { p.Undo(); });
        if (r.view<const EvaluatedMesh>().size() != before) return std::unexpected{"object duplicate undo failed"};
        Row(s, "duplicate_object_undo", undo_ms, source.FaceCount());
        const auto redo_ms = Milliseconds([&] { p.Redo(); });
        if (r.view<const EvaluatedMesh>().size() != before + 1u) return std::unexpected{"object duplicate redo failed"};
        Row(s, "duplicate_object_redo", redo_ms, source.FaceCount());
        if (run.Render) Draw(s, "duplicate_object_render", source.FaceCount());
        return {};
    }
    if (edit.Stage) {
        const auto probe = [&]() -> Result {
            if (!run.RefitProbe || single_edit != SingleEdit::Inset) return {};
            std::vector<Range> ranges;
            meshes.GetSelectedElements(drawn(), Element::Vertex).ForEach([&](uint32_t handle) {
                ranges.push_back({handle, 1u});
            });
            if (ranges.empty()) return std::unexpected{"inset produced no selected vertices for refit probe"};
            std::println(stderr, "Probing position refit for {} selected vertices...", ranges.size());
            std::vector<double> refit_ms;
            refit_ms.reserve(20u);
            for (uint32_t sample = 0u; sample < 20u; ++sample) {
                const auto duration = Milliseconds([&] {
                    mtl::ComputeChain chain{meshes.BufferContext()};
                    RefreshEditedPositions(r, chain, std::array{MeshVertexChanges{entity, ranges}});
                    chain.Submit();
                });
                Row(s, "refit" + std::to_string(sample), duration, ranges.size());
                if (sample) refit_ms.push_back(duration);
            }
            Row(s, "refit_p95", P95(refit_ms), ranges.size());
            return {};
        };
        if (auto result = Gesture(s, [&](uint32_t i) { edit.Stage(i, select_all); }, probe); !result) return result;
    } else {
        const auto pair_distance = [&] {
            return Length(test::Position(original, test::ToVertex(original, selected_face_halfedge)) - test::Position(original, test::FromVertex(original, selected_face_halfedge))) + 1e-4f;
        };
        const auto toggle_sharp = [&] {
            project::Do(p, action::MakeAction(action::object::SetSelectedSharp{.Element = Element::Face, .Sharp = !bool(meshes.Arenas().FaceSharpness.Get({test::Address(original, face), 1u})[0])}));
        };
        const auto edit_ms = Milliseconds([&] {
            if (single_edit == SingleEdit::MergeDistance) project::Do(p, Edit(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = pair_distance()}));
            else if (single_edit == SingleEdit::DissolveDegenerate) project::Do(p, Edit(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Degenerate, .Distance = pair_distance()}));
            else if (single_edit == SingleEdit::SharpFace) toggle_sharp();
            else project::Do(p, edit.Make());
        });
        Row(s, label, edit_ms, selected_count);
        if (auto result = Changed(s); !result) return result;
        if (single_edit == SingleEdit::SharpFace) {
            const auto edited = *p.History.Present;
            Row(s, "sharp_face_warm_reverse", Milliseconds(toggle_sharp), selected_count);
            p.Navigate(edited);
        }
        if (auto result = RenderStep(s, "render_" + label); !result) return result;
        if (run.Render)
            if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE"); !result) return result;
    }
    if (auto result = Navigation(s); !result) return result;
    Row(s, "exit_mode", Milliseconds([&] { project::Do(p, action::MakeAction(action::view::SetInteractionMode{InteractionMode::Object})); }), selected_count);
    if (edit.Stage && !r.view<const MeshEditWork>().empty()) return std::unexpected{"Edit mode exit retained geometry work"};
    if (meshes.HasDirtyGroups(meshes.Get(drawn()))) return std::unexpected{"edit mode exit left stale LOD groups"};
    if (single_edit == SingleEdit::Position && meshes.HasDirtyPositions(meshes.Get(drawn()))) return std::unexpected{"position edit exit retained coarse dirt"};
    if (auto result = RenderStep(s, "render_exit_mode"); !result) return result;
    if (edit.Stage && run.Render) {
        if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE_EXIT"); !result) return result;
        if (std::getenv("MESHEDITOR_EDIT_BENCH_CAPTURE_FINE")) {
            r.patch<ViewportDisplay>(FirstViewport(r), [](ViewportDisplay &display) { display.LodErrorPixels = 0.f; });
            p.Frame();
            SubmitViewports(r);
            WaitForRender(r);
            if (auto result = Capture(r, "MESHEDITOR_EDIT_BENCH_CAPTURE_FINE"); !result) return result;
        }
    }
    return {};
}

// Timings include the production project, event and history path, and exclude audits and rendering.
// A table run fills `table` and prints nothing.
Result Bench(const Run &run, Report *table) {
    const TestDir dir{"mesheditor-edit-bench"};
    Paths::Init(MESHEDITOR_BUILD_DIR, dir.Path);
    Engine engine{false};
    auto &r = engine.R;
    auto &p = *engine.P;
    Session s{r, p, run, table};
    const auto body = [&]() -> Result {
        if (!table) {
#ifndef PROFILE_BUILD
            std::println("Build with PROFILE=1 for profile tables.");
#endif
            profile::Enable(true);
            profile::Init(r.Resources.get<const mtl::Context>());
            profile::Enable(false);
        }
        if (!p.New(dir, true)) return std::unexpected{"project creation failed"};
        r.replace<ViewportPanel>(FirstViewport(r), ViewportPanel{.Extent = {run.Width, run.Height}});
        p.Settle();
        if (auto result = PlaceScene(s); !result) return result;
        return run.Edit->Kinds.empty() ? BenchEdit(s) : BenchChain(s);
    };
    // An operator's exception reports before engine teardown, since teardown waits for every GPU command and a command that never completes blocks later output.
    const auto result = [&]() -> Result {
        try {
            return body();
        } catch (const std::exception &e) {
            std::println(stderr, "{}", e.what());
            return std::unexpected{"the benchmark threw"};
        }
    }();
    if (!table) {
        if (result) {
            ReportLibraries(r.Resources.get<const mtl::LibraryCache>(), "completed");
            profile::Report();
        }
        profile::Deinit();
    }
    profile::Enable(false);
    if (!result) return result;
    if (table) table->Faces = s.Faces;
    std::string why;
    if (!p.Audit(why)) return std::unexpected{why};
    if (!r.Resources.get<action::Errors>().Messages.empty()) return std::unexpected{"action error"};
    if (auto error = p.History.TakeIntegrityError(); !error.empty()) return std::unexpected{error};
    if (!p.Close()) return std::unexpected{"project close failed"};
    return {};
}

// The runs a call without arguments times: a gesture on the first and on the last node of each chain, and a drag of every vertex in Edit mode without and with rendering.
struct TableRow {
    uint32_t Slices;
    std::string_view Op, Selection;
    bool Render{};
};
constexpr TableRow Table[]{
    {40u, "position-chain", "first"},
    {40u, "position-chain", "last"},
    {16u, "topology-chain", "first"},
    {16u, "topology-chain", "last"},
    {16u, "mixed-chain", "first"},
    {16u, "mixed-chain", "last"},
    {700u, "position-chain", "first"},
    {700u, "position-chain", "last"},
    {128u, "topology-chain", "first"},
    {128u, "topology-chain", "last"},
    {64u, "position", "all"},
    {256u, "position", "all"},
    {64u, "position", "all", true},
    {256u, "position", "all", true},
};
} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const auto fail = [](std::string_view message) {
        std::println(stderr, "{}", message);
        return 1;
    };
    const auto spec_of = [](std::string_view op) { return std::ranges::find(EditSpecs, op, &EditSpec::Argument); };
    if (argc == 1) {
        for (const auto &row : Table) {
            Report report;
            if (const auto result = Bench({.Slices = row.Slices, .Updates = 11u, .Render = row.Render, .Edit = &*spec_of(row.Op), .Selection = row.Selection, .Cold = true}, &report); !result) return fail(result.error());
            const auto phase = [&](std::string_view name) { return std::ranges::find(report.Phases, name, [](const auto &timed) -> std::string_view { return timed.first; })->second; };
            // The median over the warm steps, which follow the phase's step zero.
            const auto warm = [&](std::string_view prefix) {
                std::vector<double> steps;
                for (const auto &[name, ms] : report.Phases) {
                    const auto step = std::string_view{name}.substr(std::min(name.size(), prefix.size()));
                    if (name.starts_with(prefix) && !step.empty() && step != "0" && std::ranges::all_of(step, [](char c) { return c >= '0' && c <= '9'; })) steps.push_back(ms);
                }
                return Median(std::move(steps));
            };
            std::println("{} {}, {} faces: step {:.3f} ms{}, commit {:.3f} ms, undo {:.3f} ms, redo {:.3f} ms, cold redo {:.3f} ms", row.Op, row.Selection, report.Faces, warm("update"), row.Render ? std::format(" and frame {:.3f} ms", warm("render")) : "", phase("commit"), phase("undo"), phase("redo"), phase("cold_redo"));
        }
        return 0;
    }
    if (argc == 6 || argc > 11 || argv[1][0] == '-') {
        std::print(stderr, "usage: MeshEditorEditBench <sphere-slices|scene.gltf> [updates=3] [render=0] [refit_probe=0] [viewport_width viewport_height] [position=0] [join=0] [op] [selection]\nWithout arguments, times the standard table of chain gestures and Edit mode drags.\noperators:");
        for (const auto &spec : EditSpecs) std::print(stderr, " {}", spec.Argument);
        std::println(stderr, "\nselection: one (a single element for the operator, the default) or all (every element of the operator's domain), and for a chain the first or the last (the default) node its gesture changes");
        return 2;
    }
    const auto scene = std::filesystem::path{argv[1]};
    const bool file = scene.extension() == ".gltf" || scene.extension() == ".glb";
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
    if (position + join + uint32_t(argc > 9 && std::string_view{argv[9]} != "inset") > 1u) return fail("choose one benchmark operation");
    const auto spec = spec_of(position ? "position" : join ? "join" :
                                  argc > 9                 ? argv[9] :
                                                             "inset");
    if (spec == EditSpecs.end()) return fail("unknown benchmark operator");
    const bool chain = !spec->Kinds.empty();
    const std::string_view selection = argc > 10 ? argv[10] : chain ? "last" :
                                                                      "one";
    if (chain ? selection != "first" && selection != "last" : selection != "one" && selection != "all") return fail("selection must be one or all, or first or last for a chain");
    std::println("kind,vertices,faces,selected,phase,milliseconds,device_bytes");
    const auto result = Bench({.Slices = slices, .Scene = file ? scene : std::filesystem::path{}, .Updates = updates, .Render = render != 0u, .RefitProbe = refit_probe != 0u, .Width = width, .Height = height, .Edit = &*spec, .Selection = selection, .Cold = std::getenv("MESHEDITOR_EDIT_BENCH_COLD") != nullptr}, nullptr);
    return result ? 0 : fail(result.error());
}
