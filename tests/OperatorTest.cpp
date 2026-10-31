#include "MeshChecks.h"
#include "ProcedureFixture.h"
#include "TestMeshOrdinals.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "render/GpuBuffers.h"
#include "scene/SceneGraph.h"
#include "viewport/ViewCameraOps.h"

#include <format>
#include <limits>
#include <numbers>
#include <numeric>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::Link;

bool Facing(const Mesh &mesh, he::FH face, vec3 eye) { return face && Dot(mesh.GetNormal(face), eye - mesh.CalcFaceCentroid(face)) > 0.f; }
// The midpoints of the edges between two faces looking toward the eye.
std::vector<vec3> FacingEdgeCenters(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        if (Facing(mesh, mesh.GetFace(h), eye) && Facing(mesh, mesh.GetFace(mesh.GetOppositeHalfedge(h)), eye)) centers.push_back(.5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h))));
    }
    return centers;
}

struct ViewportRow {
    std::string_view Name;
    Element Domain;
    uint32_t Clicks;
    uint16_t Kind;
    std::vector<float> Values;
    void (*Gesture)(ProcedureFixture &, float value, bool first);
    // Run the operator on the clone at the clicked handles and return its selected elements.
    GeometrySelection (*Kernel)(ProcedureFixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};
// Run the topology task on the clone, publish its emitted selection, and return that selection.

// Each viewport edit appends one node cooking like the kernel at the clicked elements, and a drag commits as one history node.
void ViewportOperators(ProcedureFixture &f, Fixture &other) {
    using D = MeshStore::ElementDomain;
    static const std::array<ViewportRow, 4> rows{{
        {"Extrude", Element::Face, 2u, KindOf<action::mesh::Extrude>, {0.f}, [](ProcedureFixture &f, float, bool) { f.Do(action::mesh::Extrude{}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::ExtrudeRegion, .Selection = clicked}); }},
        {"BevelDrag", Element::Edge, 1u, KindOf<action::mesh::Bevel>, {.05f, .1f, .2f, .3f}, [](ProcedureFixture &f, float width, bool first) {
             if (first) f.Stage(action::mesh::Bevel{.Width = width});
             else f.Stage(UpdateOf<&action::mesh::Bevel::Width>(action::OnSelected{}, width)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float width) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::BevelEdges, .Param0 = width, .Steps = 1u, .Selection = clicked}); }},
        {"InsetDrag", Element::Face, 1u, KindOf<action::mesh::Inset>, {.1f, .25f}, [](ProcedureFixture &f, float thickness, bool first) {
             if (first) f.Stage(action::mesh::Inset{.Thickness = thickness});
             else f.Stage(UpdateOf<&action::mesh::Inset::Thickness>(action::OnSelected{}, thickness)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float thickness) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::InsetRegion, .Param0 = thickness, .Flags = TopologyFlagEvenOffset, .Selection = clicked}); }},
        {"TransformDrag", Element::Vertex, 1u, KindOf<TransformElements>, {.5f, 1.f, 1.5f}, [](ProcedureFixture &f, float z, bool) { f.Stage(action::view::TransformElements{{.P = {0.f, 0.f, z}}}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float z) {
             const PositionOperationTarget target{.StoreId = clone, .Selection = clicked};
             expect(!ExecuteGeometryPositions(f.R, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {0.f, 0.f, z}}}}).empty());
             return clicked; }},
    }};
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            const auto &r = f.R;
            const auto &meshes = f.Store();
            const auto domains = [](GeometrySelection &selection) {
                return std::array{std::tuple{D::Vertex, Element::Vertex, &selection.Vertices}, std::tuple{D::Edge, Element::Edge, &selection.Edges}, std::tuple{D::Face, Element::Face, &selection.Faces}};
            };
            const auto mesh = f.AddMesh(primitive::Cube{});
            f.EnterEdit(row.Domain);
            f.Render();
            {
                // The row clicks the centers of the first elements of its domain whose faces all look toward the eye.
                const Mesh cube{meshes, f.Output(mesh)};
                const auto eye = f.View().CameraPosition;
                std::vector<vec3> centers;
                if (row.Domain == Element::Vertex) {
                    for (const auto v : cube.vertices())
                        if (std::ranges::all_of(cube.voh_range(v), [&](he::HH h) { return Facing(cube, cube.GetFace(h), eye); })) centers.push_back(cube.GetPosition(v));
                } else if (row.Domain == Element::Edge) {
                    centers = FacingEdgeCenters(cube, eye);
                } else {
                    for (const auto face : cube.faces())
                        if (Facing(cube, face, eye)) centers.push_back(cube.CalcFaceCentroid(face));
                }
                expect(fatal(centers.size() >= row.Clicks));
                for (uint32_t i = 0u; i < row.Clicks; ++i) f.Click(centers[i], i > 0u);
            }
            GeometrySelection ordinals;
            for (const auto [domain, element, list] : domains(ordinals))
                meshes.GetSelectedElements(f.Output(mesh), element).ForEach([&](uint32_t handle) { list->push_back(meshes.LiveElementOrdinal(f.Output(mesh), domain, ElementHandle{handle})); });
            expect(ordinals.Get(row.Domain).size() == row.Clicks);
            const auto source = f.SourceOf(mesh);
            const auto start = f.P->History.Present.value();
            const auto node_count = uint32_t(r.get<const CookPlan>(mesh).Nodes.size());
            auto previous = f.Shaped(mesh);
            for (uint32_t i = 0u; i < row.Values.size(); ++i) {
                row.Gesture(f, row.Values[i], i == 0u);
                expect(r.get<const CookPlan>(mesh).Nodes.size() == node_count + 1u);
                const auto shape = f.Shaped(mesh);
                expect(shape != previous);
                previous = shape;
            }
            if (f.P->HasStaged()) f.CommitGesture();
            const auto &procedure = r.get<const CookPlan>(mesh);
            const auto [node, kind] = procedure.Nodes.back();
            expect(procedure.Nodes.size() == node_count + 1u && kind == row.Kind);
            expect(r.all_of<PublishSelection>(node) && !r.any_of<NodeError>(node));
            expect(f.P->History.Nodes[f.P->History.Present.value()].Parent == start);
            const auto selection = EditorGeometrySelection(meshes, f.Output(mesh));
            const auto selected = Describe(meshes, f.Output(mesh), selection);
            expect(meshes.GetSelectionSummary(f.Output(mesh)).SelectedCount == selection.Get(row.Domain).size());
            f.Do(action::view::SetInteractionMode{InteractionMode::Object});
            expect(f.Shaped(mesh) == previous);
            expect(f.SourceOf(mesh) == source);
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                GeometrySelection clicked;
                for (const auto [domain, element, list] : domains(clicked)) {
                    for (const auto ordinal : ordinals.Get(element)) list->push_back(meshes.LiveElementAt(clone, domain, ordinal).Value);
                    std::ranges::sort(*list);
                }
                return row.Kernel(f, clone, clicked, row.Values.back());
            });
            expect(kernel.Output == previous);
            expect(kernel.Selected == selected);
        });
    }
}

// A gesture the mesh rejects appends no node and leaves the history where it was.
constexpr uint16_t NoKind = std::numeric_limits<uint16_t>::max();
// The record an operator leaves, from an independent reference: each vertex's position by ordinal, the vertex positions as a set, or every face as a loop of source vertex indices matched by position.
struct Oracle {
    std::vector<vec3> Positions{}, Points{};
    std::vector<std::vector<uint32_t>> Faces{};
    // The node leaves the record as it was, cooking an identity or, faulted, passing its input through.
    bool Unchanged{}, Faulted{};
};
// A primitive or sourced object in World, a preparation, the edit domain and the elements selected in it, an operator gesture, the node kind the gesture appends, and the record its node cooks, as a fingerprint of its shape and marks or as an Oracle.
// A row selects its clicked Targets, else its Select ordinals in Domain with an edge as its two vertex ordinals, else every element.
// A row with Targets leaves World at the identity, where local positions are world positions.
struct OperatorRow {
    std::string_view Name;
    std::variant<PrimitiveParameters, MeshSource (*)()> Shape;
    Element Domain;
    uint16_t Kind;
    uint64_t Cooks;
    void (*Gesture)(ProcedureFixture &);
    std::vector<vec3> (*Targets)(const Mesh &, vec3 eye){};
    Transform World{};
    void (*Prepare)(ProcedureFixture &){};
    std::vector<uint32_t> Select{};
    Oracle Expected{};
};
// A two by two grid of quads three units wide in the XZ plane facing +Y, with each vertex's X and Z as its UV and a color of its own.
MeshSource AttributedGrid() {
    MeshSource source;
    auto &uvs = source.Attrs.TexCoords0.emplace();
    auto &colors = source.Attrs.Colors0.emplace();
    source.Attrs.Colors0ComponentCount = 4u;
    for (uint32_t z = 0u; z < 3u; ++z) {
        for (uint32_t x = 0u; x < 3u; ++x) {
            source.Data.Positions.emplace_back(3.f * (float(x) - 1.f), 0.f, 3.f * (float(z) - 1.f));
            uvs.emplace_back(float(x), float(z));
            colors.emplace_back(float(x) / 2.f, float(z) / 2.f, .25f, 1.f);
        }
    }
    for (uint32_t z = 0u; z < 2u; ++z) {
        for (uint32_t x = 0u; x < 2u; ++x) {
            const auto v = z * 3u + x;
            source.Data.AddFace(std::array{v, v + 3u, v + 4u, v + 1u});
        }
    }
    return source;
}
// The sharp edges and flat faces by their corner positions, and each face's corners in loop order from its least position, as position, first UV and color, each sorted.
struct Marks {
    std::vector<std::array<Corner, 2>> SharpEdges;
    std::vector<std::vector<Corner>> FlatFaces;
    std::vector<std::vector<std::array<float, 9>>> Corners;
    bool operator==(const Marks &) const = default;
};
Marks MarksOf(const MeshStore &meshes, uint32_t id) {
    const Mesh mesh{meshes, id};
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto point = [&](he::VH v) {
        const auto p = mesh.GetPosition(v);
        return Corner{p.x, p.y, p.z};
    };
    Marks marks;
    for (const auto e : mesh.edges()) {
        if (!a.EdgeSharpness.Get({test::Address(mesh, e), 1u})[0]) continue;
        const auto h = mesh.GetHalfedge(e, 0u);
        const auto from = point(mesh.GetFromVertex(h)), to = point(mesh.GetToVertex(h));
        marks.SharpEdges.push_back({std::min(from, to), std::max(from, to)});
    }
    const bool uvs = record.CornerAttributes & MeshAttributeBit_TexCoord0, colors = record.CornerAttributes & MeshAttributeBit_Color0;
    for (const auto face : mesh.faces()) {
        auto &corners = marks.Corners.emplace_back();
        for (const auto h : mesh.fh_range(face)) {
            const auto p = mesh.GetPosition(mesh.GetToVertex(h));
            const auto uv = uvs ? a.CornerUvs[0].Get(record.CornerUvs[0], *h) : vec2{};
            const auto color = colors ? a.CornerColors.Get(record.CornerColors, *h) : vec4{};
            corners.push_back({p.x, p.y, p.z, uv.x, uv.y, color.x, color.y, color.z, color.w});
        }
        std::ranges::rotate(corners, std::ranges::min_element(corners));
        if (!a.FaceSharpness.Get({test::Address(mesh, face), 1u})[0]) continue;
        auto &flat = marks.FlatFaces.emplace_back();
        for (const auto v : mesh.fv_range(face)) flat.push_back(point(v));
        std::ranges::sort(flat);
    }
    std::ranges::sort(marks.SharpEdges);
    std::ranges::sort(marks.FlatFaces);
    std::ranges::sort(marks.Corners);
    return marks;
}
// A hash of the shape and marks of the record the mesh draws, which counts zeros of either sign as one value.
uint64_t Fingerprint(ProcedureFixture &f, state::Entity mesh) {
    const auto shape = f.Shaped(mesh);
    const auto marks = MarksOf(f.Store(), f.Output(mesh));
    uint64_t hash{14695981039346656037ull};
    const auto mix = [&](float value) {
        const auto bits = value == 0.f ? 0u : std::bit_cast<uint32_t>(value);
        for (uint32_t i = 0u; i < 4u; ++i) hash = (hash ^ ((bits >> (8u * i)) & 0xffu)) * 1099511628211ull;
    };
    const auto mix_points = [&](std::span<const Corner> points) {
        mix(float(points.size()));
        for (const auto &p : points) std::ranges::for_each(p, mix);
    };
    mix_points(shape.Vertices);
    for (const auto &edge : shape.Edges) mix_points(edge);
    for (const auto &face : shape.Faces) mix_points(face);
    for (const auto &edge : marks.SharpEdges) mix_points(edge);
    for (const auto &face : marks.FlatFaces) mix_points(face);
    for (const auto &face : marks.Corners) {
        mix(float(face.size()));
        for (const auto &corner : face) std::ranges::for_each(corner, mix);
    }
    return hash;
}
// The centroids of the first face and a face sharing an edge with it.
std::vector<vec3> AdjacentFaceCenters(const Mesh &mesh, vec3) {
    const auto first = *mesh.faces().begin();
    for (const auto h : mesh.fh_range(first)) {
        if (const auto opposite = mesh.GetOppositeHalfedge(h); opposite && mesh.GetFace(opposite)) return {mesh.CalcFaceCentroid(first), mesh.CalcFaceCentroid(mesh.GetFace(opposite))};
    }
    return {};
}
// The centroid of the first face looking toward the eye.
std::vector<vec3> FacingFaceCenter(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (Facing(mesh, face, eye)) return {mesh.CalcFaceCentroid(face)};
    }
    return {};
}
// The midpoint of the first edge between two faces looking toward the eye.
std::vector<vec3> FacingEdgeCenter(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(1) | std::ranges::to<std::vector>(); }
// Two edges meeting at a cube's corner nearest the eye, which form a chain of three vertices.
std::vector<vec3> CornerEdgeCenters(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(2) | std::ranges::to<std::vector>(); }
// The first and third equator vertices of a sphere about the origin facing the eye, in order around the equator, with one vertex between them.
std::vector<vec3> EquatorVertices(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> equator;
    for (const auto v : mesh.vertices()) {
        const auto p = mesh.GetPosition(v);
        if (std::abs(p.y) < 1e-4f && Dot(p, eye - p) > 0.f) equator.push_back(p);
    }
    if (equator.size() < 3u) return {};
    std::ranges::sort(equator, {}, [](vec3 p) { return std::atan2(p.z, p.x); });
    return {equator[0], equator[2]};
}

// Two adjacent corners of the first face looking toward the eye, the lower handle first.
std::vector<vec3> FacingSide(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye)) continue;
        auto corners = mesh.fv_range(face) | std::views::take(2) | std::ranges::to<std::vector>();
        std::ranges::sort(corners);
        return {mesh.GetPosition(corners[0]), mesh.GetPosition(corners[1])};
    }
    return {};
}
// Two opposite corners of the first quad looking toward the eye.
std::vector<vec3> FacingDiagonal(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye) || mesh.GetValence(face) != 4u) continue;
        const auto corners = mesh.fv_range(face) | std::ranges::to<std::vector>();
        return {mesh.GetPosition(corners[0]), mesh.GetPosition(corners[2])};
    }
    return {};
}
// The midpoints of the edges with one face.
std::vector<vec3> BoundaryEdgeCenters(const Mesh &mesh, vec3) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u), opposite = mesh.GetOppositeHalfedge(h);
        if (bool(mesh.GetFace(h)) != (opposite && mesh.GetFace(opposite))) centers.push_back(.5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h))));
    }
    return centers;
}
// The midpoints of the first edge and of its reflection through the origin.
std::vector<vec3> OppositeEdgeCenters(const Mesh &mesh, vec3) {
    const auto h = mesh.GetHalfedge(*mesh.edges().begin(), 0u);
    const auto center = .5f * (mesh.GetPosition(mesh.GetFromVertex(h)) + mesh.GetPosition(mesh.GetToVertex(h)));
    return {center, -center};
}

// Shades every face smooth.
void ShadedSmooth(ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{true}); }
// Randomly offsets every vertex.
template<float Amount> void Randomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Randomize{.Amount = Amount, .Seed = 3u});
}
// Subdivides every edge three times.
void Subdivided(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 3u});
}
// Deletes every face, leaving the edges.
void Wires(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::OnlyFaces});
}
// Dissolves five consecutive triangles of an eight-segment disc into one polygon, concave at the disc's center.
void Notched(ProcedureFixture &f) {
    f.Render();
    const Mesh mesh{f.Store(), f.Output(f.R.get<const Instance>(FindActiveEntity(f.R)).Entity)};
    std::vector<vec3> centers;
    for (const auto face : mesh.faces()) centers.push_back(mesh.CalcFaceCentroid(face));
    std::ranges::sort(centers, {}, [](vec3 p) { return std::atan2(p.z, p.x); });
    for (uint32_t i = 0u; i < 5u; ++i) f.Click(centers[i], i > 0u);
    f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Faces});
}
// Triangulates every face and randomly offsets every vertex.
void TriangulatedRandomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Triangulate{});
    f.Do(action::mesh::Randomize{.Amount = .3f, .Seed = 3u});
}
// Deletes the first edge between two faces looking toward the eye with its faces, which opens a boundary loop.
void Opened(ProcedureFixture &f) {
    f.Render();
    const auto mesh = f.R.get<const Instance>(FindActiveEntity(f.R)).Entity;
    f.Click(FacingEdgeCenter(Mesh{f.Store(), f.Output(mesh)}, f.View().CameraPosition).front(), false);
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::Edges});
}

// Closeness within the precision of a reference read off another implementation.
bool Near(vec3 actual, vec3 expected) { return Length(actual - expected) <= 2e-4f * std::max(1.f, Length(expected)); }
// Each face as a loop of indices into `source`, the vertex positions before the operator, which position operators leave and topology operators keep.
std::vector<std::vector<uint32_t>> Loops(const Mesh &mesh, std::span<const vec3> source) {
    std::vector<std::vector<uint32_t>> loops;
    for (const auto face : mesh.faces()) {
        auto &loop = loops.emplace_back();
        for (const auto v : mesh.fv_range(face)) {
            const auto p = mesh.GetPosition(v);
            const auto found = std::ranges::find_if(source, [&](vec3 q) { return Near(p, q); });
            expect(found != source.end()) << "vertex" << p.x << p.y << p.z << "is not a source vertex";
            loop.push_back(uint32_t(found - source.begin()));
        }
    }
    return loops;
}
// The loops each rotated to start at its least index, sorted, which compares oriented loops.
std::vector<std::vector<uint32_t>> Canonical(std::vector<std::vector<uint32_t>> loops) {
    for (auto &loop : loops) std::ranges::rotate(loop, std::ranges::min_element(loop));
    std::ranges::sort(loops);
    return loops;
}
// The ordinal of the first edge between each pair of vertex ordinals.
std::vector<uint32_t> EdgeOrdinals(const Mesh &mesh, std::span<const uint32_t> pairs) {
    std::vector<uint32_t> ordinals;
    for (size_t i = 0u; i + 1u < pairs.size(); i += 2u) {
        uint32_t ordinal = 0u;
        for (const auto e : mesh.edges()) {
            const auto h = mesh.GetHalfedge(e, 0u);
            if (std::minmax(mesh.VertexOrdinal(mesh.GetFromVertex(h)), mesh.VertexOrdinal(mesh.GetToVertex(h))) == std::minmax(pairs[i], pairs[i + 1u])) {
                ordinals.push_back(ordinal);
                break;
            }
            ++ordinal;
        }
    }
    expect(ordinals.size() == pairs.size() / 2u);
    return ordinals;
}
std::vector<vec3> Sorted(std::vector<vec3> points) {
    std::ranges::sort(points, {}, [](vec3 p) { return Corner{p.x, p.y, p.z}; });
    return points;
}
// The vertex ordinals `first` through `last`.
std::vector<uint32_t> Ordinals(uint32_t first, uint32_t last) {
    std::vector<uint32_t> ordinals(last - first + 1u);
    std::iota(ordinals.begin(), ordinals.end(), first);
    return ordinals;
}
// The position whose world placement by `world` is `p`.
vec3 Local(const Transform &world, vec3 p) {
    const auto local = Conjugate(world.R) * (p - world.P);
    return {local.x / world.S.x, local.y / world.S.y, local.z / world.S.z};
}
MeshSource Sourced(std::vector<vec3> positions, std::vector<std::vector<uint32_t>> faces = {}, std::vector<std::array<uint32_t, 2>> edges = {}) {
    MeshSource source{.Weld = false};
    source.Data.Positions = std::move(positions);
    for (const auto &face : faces) source.Data.AddFace(face);
    source.Data.Edges = std::move(edges);
    return source;
}
// Every vertex's position, with UVs and colors from its position.
MeshSource Attributed(MeshSource source) {
    auto &uvs = source.Attrs.TexCoords0.emplace();
    auto &colors = source.Attrs.Colors0.emplace();
    source.Attrs.Colors0ComponentCount = 4u;
    for (const auto p : source.Data.Positions) {
        uvs.push_back({p.x / 3.f, p.y / 3.f});
        colors.push_back({p.x / 3.f, p.y / 3.f, .5f, 1.f});
    }
    return source;
}
// A `size` by `size` grid of unit quads on the XY plane, its vertex at (x, y) at index x + (size + 1) y.
MeshSource Grid(uint32_t size, bool triangles = false) {
    std::vector<vec3> positions;
    std::vector<std::vector<uint32_t>> faces;
    const auto width = size + 1u;
    for (uint32_t y = 0u; y < width; ++y)
        for (uint32_t x = 0u; x < width; ++x) positions.push_back({float(x), float(y), 0.f});
    for (uint32_t y = 0u; y < size; ++y)
        for (uint32_t x = 0u; x < size; ++x) {
            const auto v = y * width + x;
            if (triangles) {
                faces.push_back({v, v + 1u, v + width + 1u});
                faces.push_back({v, v + width + 1u, v + width});
            } else faces.push_back({v, v + 1u, v + width + 1u, v + width});
        }
    return Sourced(std::move(positions), std::move(faces));
}
// A far point no operator selects, so a fit leaves it.
constexpr vec3 Far{20.f, 20.f, 20.f};
// Three quad strips of two quads, or the same strips as triangles, with the middle row at y = 0.
MeshSource SlideQuads() { return Sourced({{-.2f, -1, 0}, {2, -2, 0}, {4.5f, -1, 0}, {0, 0, 0}, {2, 0, 0}, {4, 0, 0}, {-.5f, 2, 0}, {2, 3, 0}, {5, 1, 0}}, {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}}); }
MeshSource SlideTriangles(std::array<float, 9> heights = {}) {
    auto source = Sourced({{-.2f, -1, 0}, {2, -2, 0}, {4.5f, -1, 0}, {0, 0, 0}, {2, 0, 0}, {4, 0, 0}, {-.5f, 2, 0}, {2, 3, 0}, {5, 1, 0}}, {{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}, {3, 4, 7}, {3, 7, 6}, {4, 5, 8}, {4, 8, 7}});
    for (uint32_t i = 0u; i < heights.size(); ++i) source.Data.Positions[i].z = heights[i];
    return source;
}
// Three square rings of radius 1, 2 and 4 joined by quads.
MeshSource Rings() {
    std::vector<vec3> positions;
    std::vector<std::vector<uint32_t>> faces;
    for (const float radius : {1.f, 2.f, 4.f}) positions.insert(positions.end(), {{radius, radius, 0}, {-radius, radius, 0}, {-radius, -radius, 0}, {radius, -radius, 0}});
    for (uint32_t ring = 0u; ring < 2u; ++ring)
        for (uint32_t i = 0u; i < 4u; ++i) faces.push_back({ring * 4u + i, ring * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + i});
    return Sourced(std::move(positions), std::move(faces));
}
// Three separate triangles along +Y, each with a slide vertex, and an isolated vertex.
MeshSource ThreeTriangles() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 4, 0}, {4, 4, 0}, {-1, 6, 0}, {0, 8, 0}, {1, 8, 0}, {0, 10, 0}, {0, 12, 0}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}); }
// Three triangles sharing the edge from the origin to (2, 0, 0), one of them opening the fan into a nonmanifold edge.
MeshSource Fan3() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}, {{0, 1, 2}, {1, 0, 3}, {0, 2, 4}}); }
MeshSource NonmanifoldFan() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 2, 1}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}, {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}, {5, 6, 7}, {7, 6, 5}}); }
MeshSource NonmanifoldEdge() { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}}, {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}}); }
// A quad whose four vertices sit on the axes two units out, 252 selected vertices at its center, and a far point, so a radial edit's mean radius crosses a reduction tile.
MeshSource Cross() {
    std::vector<vec3> positions{{-2.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 0.f}};
    positions.insert(positions.end(), 252u, vec3{});
    positions.push_back({9.f, 8.f, 7.f});
    return Sourced(std::move(positions), {{0, 2, 1, 3}});
}
// Columns of five vertices in the plane y = 1 over 53 heights, which span two reduction tiles.
MeshSource Columns() {
    std::vector<vec3> positions;
    for (uint32_t z = 0u; z < 53u; ++z)
        for (const float x : {-3.f, -2.f, 0.f, 2.f, 3.f}) positions.push_back({x, 1.f, float(z)});
    return Sourced(std::move(positions));
}
std::vector<vec3> Scaled(MeshSource source, uint32_t count, auto &&scale) {
    auto expected = source.Data.Positions;
    for (uint32_t i = 0u; i < count; ++i) expected[i] *= scale(Length(expected[i]));
    return expected;
}
// The positions a wireframe of the unit quad leaves for a wire of `thickness` at `offset`, with its inner and outer ring insets even or not, with or without the boundary ring, replacing the quad or keeping it.
std::vector<vec3> WirePoints(float thickness, float offset, bool even, bool boundary, bool replace) {
    const float radius = thickness * .5f, mid = radius * offset, inset = even ? radius : radius / std::sqrt(2.f);
    std::vector<vec3> points;
    for (const vec3 p : {vec3{-1.f, -1.f, 0.f}, vec3{1.f, -1.f, 0.f}, vec3{1.f, 1.f, 0.f}, vec3{-1.f, 1.f, 0.f}}) {
        points.push_back(p + vec3{0.f, 0.f, mid - radius});
        points.push_back(p + vec3{0.f, 0.f, mid + radius});
        points.push_back({p.x * (1.f - inset), p.y * (1.f - inset), mid});
        if (boundary) points.push_back({p.x * (1.f + inset), p.y * (1.f + inset), mid});
        if (!replace) points.push_back(p);
    }
    return points;
}
// A 66-gon folded along the line between its first vertex and the opposite one.
MeshSource Ridge() {
    std::vector<vec3> positions;
    std::vector<uint32_t> loop;
    for (uint32_t i = 0u; i < 66u; ++i) {
        const float angle = -.5f * std::numbers::pi_v<float> + float(i) * (2.f * std::numbers::pi_v<float> / 66.f);
        const float x = i == 0u || i == 33u ? 0.f : std::cos(angle);
        positions.push_back({x, std::sin(angle), std::abs(x)});
        loop.push_back(i);
    }
    return Sourced(std::move(positions), {loop});
}
// Two cubes with three faces of each wound inward, a nonmanifold fan, an open fan and a lone triangle.
MeshSource Windings() {
    std::vector<vec3> positions{{-1.f, -1.f, -1.f}, {1.f, -1.f, -1.f}, {1.f, 1.f, -1.f}, {-1.f, 1.f, -1.f}, {-1.f, -1.f, 1.f}, {1.f, -1.f, 1.f}, {1.f, 1.f, 1.f}, {-1.f, 1.f, 1.f}};
    std::array<std::vector<uint32_t>, 6> cube{{{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}};
    for (const auto i : {0u, 2u, 5u}) std::ranges::reverse(cube[i]);
    std::vector<std::vector<uint32_t>> faces{cube.begin(), cube.end()};
    for (uint32_t i = 0u; i < 8u; ++i) positions.push_back(positions[i] + vec3{5.f, 1.f, 0.f});
    for (auto face : cube) {
        for (auto &v : face) v += 8u;
        faces.push_back(face);
    }
    positions.insert(positions.end(), {{10, 0, 0}, {12, 0, 0}, {10, 1, 0}, {10, 0, 1}, {10, -1, -1}, {0, 4, 0}, {1, 4, 0}, {0, 5, 0}, {0, 4, 1}, {0, 8, 0}, {1, 8, 0}, {2, 8, 0}});
    faces.insert(faces.end(), {{16, 17, 18}, {17, 16, 19}, {16, 17, 20}, {21, 22, 23}, {21, 22, 24}, {25, 26, 27}});
    return Sourced(std::move(positions), std::move(faces));
}
// Looks straight down the world Z axis, which the bend and warp references take as the viewport plane.
void LookDown(ProcedureFixture &f) {
    f.R.patch<ViewCameraState>(FirstViewport(f.R), [](auto &view) { view.Active.Orientation = {}; });
}
const Transform BendWorld{.P = {3, -1, 2}, .R = numeric::AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}), .S = {-2, .5f, 1.5f}};
const Transform SlideWorld{.P = {2, -1, 3}, .R = numeric::AngleAxis(.6f, Normalize(vec3{1, 2, 3})), .S = {3, -2, .5f}};
const Transform VertexSlideWorld{.P = {2, -1, 3}, .R = numeric::AngleAxis(.4f, vec3{0, 0, 1}) * numeric::AngleAxis(-.2f, vec3{0, 1, 0}) * numeric::AngleAxis(.3f, vec3{1, 0, 0}), .S = {-3, .5f, 2}};
// A chain of four vertices whose world positions bend onto a quarter circle of radius two.
MeshSource BendChain() {
    std::vector<vec3> positions;
    for (const auto p : {vec3{-1, 0, 0}, vec3{1, 0, 1}, vec3{2, 0, 2}, vec3{3, 0, 3}}) positions.push_back(Local(BendWorld, p));
    return Sourced(std::move(positions), {}, {{0, 1}, {1, 2}, {2, 3}});
}

// The source's positions with those at `indices` replaced in order.
std::vector<vec3> Replaced(MeshSource source, std::initializer_list<uint32_t> indices, std::initializer_list<vec3> values) {
    auto positions = source.Data.Positions;
    auto value = values.begin();
    for (const auto i : indices) positions[i] = *value++;
    return positions;
}
// Sets the mesh's active element to the vertex at `ordinal` and commits it.
void Activate(ProcedureFixture &f, uint32_t ordinal) {
    const auto mesh = f.R.get<const Instance>(FindActiveEntity(f.R)).Entity;
    f.R.emplace_or_replace<MeshActiveElement>(mesh, *Mesh{f.Store(), f.Output(mesh)}.VertexAt(ordinal));
    f.Commit("Set active vertex");
}
// Eight vertices with a symmetric partner within .2 of each one's mirror across x = 0, where the last competes with the first for the second's partner.
MeshSource Symmetric() { return Sourced({{-1, 0, 0}, {1.08f, .04f, .02f}, {-2, 1, .2f}, {2.02f, 1.05f, .25f}, {.02f, 2, .3f}, {4, 2, 3}, {.1f, 4, .6f}, {-1.02f, 0, 0}}, {{0, 1, 4}, {2, 3, 5}}); }
// Four wires: two separate edges to slide along, a zero-length edge and a fork.
MeshSource SlideWires() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {4, 3, 0}, {7, 0, 0}, {7, 0, 0}, {9, 0, 0}, {0, 9, 0}}, {}, {{0, 1}, {2, 3}, {4, 5}, {4, 6}}); }
// A fan of triangles over a convex polygon with `count` vertices, then a triangle and a quad outside the selection on the fan's first edge.
MeshSource Fanned(std::vector<vec3> polygon) {
    const auto n = uint32_t(polygon.size());
    std::vector<std::vector<uint32_t>> faces;
    for (uint32_t i = 1u; i + 1u < n; ++i) faces.push_back({0u, i, i + 1u});
    polygon.insert(polygon.end(), {{-7, 0, 0}, {-7, -3, 0}});
    faces.push_back({1u, 0u, n});
    faces.push_back({0u, n - 1u, n, n + 1u});
    return Sourced(std::move(polygon), std::move(faces));
}
// Adds `source` as an object, selects it, and returns its mesh entity and instance.
std::pair<state::Entity, state::Entity> AddSourced(ProcedureFixture &f, MeshSource source, const Transform &world = {}) {
    const auto [mesh, instance] = AddMesh(f.R, CreateMesh(f.R, std::move(source)).StoreId, MeshInstanceCreateInfo{.Transform = world});
    f.Commit("Add mesh");
    f.Do(action::selection::Select{instance});
    return {mesh, instance};
}
// Selects the ordinals in the mesh's domain and commits the selection.
void SelectOrdinals(ProcedureFixture &f, state::Entity mesh, std::span<const uint32_t> ordinals, Element domain) {
    ApplyEditSelectionLists(f.R, std::array{std::pair{mesh, ordinals}}, domain);
    f.Commit("Select elements");
}
// The node each operator appends cooks the shape, sharp marks, UVs and colors recorded for it at the same selection.
// Operators that edit in their mesh's world frame place the object at a rotated and unevenly scaled transform.
void RecordedOperators(ProcedureFixture &f, Fixture &other) {
    using namespace action::mesh;
    static const Transform Placed{.P = {.5f, .25f, -.5f}, .R = numeric::AngleAxis(.4f, Normalize(vec3{1.f, 2.f, 3.f})), .S = {1.f, 1.5f, .75f}};
    static const std::vector<OperatorRow> rows{
        {"LimitedDissolveFaces", primitive::Circle{.Radius = 5.f, .Segments = 8u, .Fill = primitive::FillType::Fan}, Element::Face, KindOf<Dissolve>, 0x9d7f60a219ef7cf2ull, [](ProcedureFixture &f) { f.Do(Dissolve{.Mode = DissolveMode::Limited}); }, AdjacentFaceCenters},
        {"ShrinkFattenFace", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<ShrinkFatten>, 0x2ff00fd8faa7342ull, [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .5f}); }, FacingFaceCenter},
        {"SpaceEvenly", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<SpaceEvenly>, 0xfd909392906a58faull, [](ProcedureFixture &f) { f.Do(SpaceEvenly{}); }, CornerEdgeCenters},
        {"RelaxEdgeLoops", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<RelaxEdgeLoops>, 0x7adaf411cc5a0b7bull, [](ProcedureFixture &f) { f.Do(RelaxEdgeLoops{}); }, CornerEdgeCenters},
        {"CurveBetweenSelected", primitive::UVSphere{.Radius = 5.f, .Slices = 8u, .Stacks = 4u}, Element::Vertex, KindOf<CurveBetweenSelected>, 0x48b8cce36aef7394ull, [](ProcedureFixture &f) { f.Do(CurveBetweenSelected{}); }, EquatorVertices},
        {"Circularize", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Face, KindOf<Circularize>, 0xe34644a8884ae607ull, [](ProcedureFixture &f) { f.Do(Circularize{}); }, FacingFaceCenter},
        {"MakePlanarFaces", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<MakePlanarFaces>, 0x7e7be948d50e0430ull, [](ProcedureFixture &f) { f.Do(MakePlanarFaces{}); }, {}, {}, Randomized<.3f>},
        {"FlattenView", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Flatten>, 0xc5a379f30a5ba0f2ull, [](ProcedureFixture &f) { f.R.patch<ViewCameraState>(FirstViewport(f.R), [](auto &v) { v.Active.Orientation = ViewCamera::OrientationFromAway(Normalize(vec3{1.f, 2.f, 2.f})); }); f.Do(action::mesh::Flatten{.Method = FlattenMethod::View}); }, {}, Placed},
        {"ToSphere", primitive::Cone{.Radius = 2.f, .Height = 4.f, .Slices = 8u}, Element::Vertex, KindOf<procedure::ToSphere>, 0x2470285c3aa67a25ull, [](ProcedureFixture &f) { f.Do(action::mesh::ToSphere{}); }, {}, Placed},
        {"PushPull", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::PushPull>, 0x9d362c76fc3b4222ull, [](ProcedureFixture &f) { f.Do(action::mesh::PushPull{.Distance = .5f}); }, {}, Placed},
        {"ShearLocal", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Shear>, 0x845dc6891da1c7f2ull, [](ProcedureFixture &f) { f.Do(action::mesh::Shear{.Local = true}); }, {}, Placed},
        {"Warp", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Warp>, 0x29fcc570816d46efull, [](ProcedureFixture &f) { f.R.patch<ViewCameraState>(FirstViewport(f.R), [](auto &v) { v.Active.Orientation = numeric::AngleAxis(.3f, Normalize(vec3{1.f, 1.f, 0.f})); }); f.Do(action::mesh::Warp{.Angle = 1.f, .Center = {.5f, 0.f, 0.f}}); }, {}, Placed},
        {"Bend", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::Bend>, 0x193360401e7f9919ull, [](ProcedureFixture &f) { f.R.patch<ViewCameraState>(FirstViewport(f.R), [](auto &v) { v.Active.Orientation = numeric::AngleAxis(.3f, Normalize(vec3{0.f, 1.f, 1.f})); }); f.Do(action::mesh::Bend{.Angle = .8f, .Radius = 2.f, .Clamp = false}); }, {}, Placed},
        {"VertexSlide", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::VertexSlide>, 0x815064164d8c48c2ull, [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .3f, .Direction = Normalize(vec3{1.f, .2f, .1f})}); }, {}, Placed},
        {"EdgeSlide", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<procedure::EdgeSlide>, 0x72da6642c85c8ac2ull, [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .5f, .Direction = {0.f, 1.f, 0.f}}); }, FacingEdgeCenter},
        {"SnapSymmetry", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<SnapSymmetry>, 0xe29e1f318642b782ull, [](ProcedureFixture &f) { f.Do(SnapSymmetry{.Threshold = .5f}); }, {}, {}, Randomized<.05f>},
        {"VertexSlideEvenActive", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::VertexSlide>, 0x6a6a3848f7831314ull, [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .3f, .Direction = Normalize(vec3{1.f, .2f, .1f}), .Even = true}); }, FacingSide, {}, Randomized<.3f>},
        {"EdgeSlideEvenActive", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::EdgeSlide>, 0xb1f7e0a1da31983dull, [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .5f, .Direction = {0.f, 1.f, 0.f}, .Even = true}); }, FacingSide, {}, Randomized<.3f>},
        {"LoopCutActive", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<LoopCut>, 0x10b3e9b664300622ull, [](ProcedureFixture &f) { f.Do(LoopCut{.Cuts = 2u}); }, CornerEdgeCenters},
        {"SplitNonplanarCube", primitive::Cube{}, Element::Face, KindOf<SplitNonplanarFaces>, 0xba7ee32f43a24e84ull, [](ProcedureFixture &f) { f.Do(SplitNonplanarFaces{}); }, {}, {}, Randomized<.3f>},
        {"SplitConcaveDisc", primitive::Circle{.Radius = 5.f, .Segments = 8u, .Fill = primitive::FillType::Fan}, Element::Face, KindOf<SplitConcaveFaces>, 0xcfa4b6519fb148d2ull, [](ProcedureFixture &f) { f.Do(SplitConcaveFaces{}); }, {}, {}, Notched},
        {"ConnectDiagonal", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<ConnectVertices>, 0xb136037f318c5ac2ull, [](ProcedureFixture &f) { f.Do(ConnectVertices{}); }, FacingDiagonal},
        {"Fill", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<Fill>, 0x7c06b3169d079ce2ull, [](ProcedureFixture &f) { f.Do(Fill{}); }, FacingDiagonal},
        {"BridgeEdgeLoops", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<BridgeEdgeLoops>, 0xc676e92a61a1aa82ull, [](ProcedureFixture &f) { f.Do(BridgeEdgeLoops{}); }, OppositeEdgeCenters, {}, Wires},
        {"GridFill", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<GridFill>, 0x6384c60043138282ull, [](ProcedureFixture &f) { f.Do(GridFill{}); }, BoundaryEdgeCenters, {}, Opened},
        {"EdgeRotate", primitive::IcoSphere{.Radius = 3.f, .Subdivisions = 1u}, Element::Edge, KindOf<EdgeRotate>, 0xcad4838bfffb586full, [](ProcedureFixture &f) { f.Do(EdgeRotate{}); }, FacingEdgeCenter},
        {"BeautifyFaces", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<BeautifyFaces>, 0xc9c17ac413c1468bull, [](ProcedureFixture &f) { f.Do(BeautifyFaces{}); }, {}, {}, TriangulatedRandomized},
        {"Unsubdivide", primitive::Plane{}, Element::Vertex, KindOf<Unsubdivide>, 0x1161622a953771d2ull, [](ProcedureFixture &f) { f.Do(Unsubdivide{}); }, {}, {}, Subdivided},
        {"RotateUVs", AttributedGrid, Element::Face, KindOf<RotateUVs>, 0x8c705a4d446978b2ull, [](ProcedureFixture &f) { f.Do(RotateUVs{}); }, AdjacentFaceCenters},
        {"ReverseUVs", AttributedGrid, Element::Face, KindOf<ReverseUVs>, 0xd794f9f5f0c51232ull, [](ProcedureFixture &f) { f.Do(ReverseUVs{}); }, AdjacentFaceCenters},
        {"RotateColors", AttributedGrid, Element::Face, KindOf<RotateColors>, 0xc1f62ebd4e569352ull, [](ProcedureFixture &f) { f.Do(RotateColors{.CounterClockwise = true}); }, AdjacentFaceCenters},
        {"ReverseColors", AttributedGrid, Element::Face, KindOf<ReverseColors>, 0xf240377728750772ull, [](ProcedureFixture &f) { f.Do(ReverseColors{}); }, AdjacentFaceCenters},
        {"MarkSharpEdges", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<procedure::MarkSharp>, 0xaf74c97303be8d62ull, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Edge, .Sharp = true}); }, CornerEdgeCenters},
        {"MarkSharpVertices", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::MarkSharp>, 0x7d0d307e27f220e2ull, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Vertex, .Sharp = true}); }, FacingSide},
        {"SmoothFace", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::MarkSharp>, 0x9ab8278bab26cbe2ull, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSharp{.Element = Element::Face, .Sharp = false}); }, FacingFaceCenter},
        {"Knife", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Knife>, 0xd5a12f0a93af004full, [](ProcedureFixture &f) {
             f.Render();
             const auto &view = f.View();
             const auto pixel = [&](vec3 p) {
                 const auto clip = view.ViewProj * vec4{p.x, p.y, p.z, 1.f};
                 return vec2{(clip.x / clip.w + 1.f) * .5f * view.ViewportSize.x, (1.f - clip.y / clip.w) * .5f * view.ViewportSize.y};
             };
             f.Do(action::mesh::Knife{pixel(Placed.P + vec3{-4.f, .3f, 0.f}), pixel(Placed.P + vec3{4.f, -.2f, 0.f})}); }, {}, Placed},
        {"ShadeSmooth", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::ShadeSmooth>, 0x402f5a2f4c86f02ull, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{true}); }},
        {"ShadeFlat", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::ShadeSmooth>, 0x26de72431ea25cf2ull, [](ProcedureFixture &f) { f.Do(action::object::SetSelectedSmoothShading{false}); }, {}, {}, ShadedSmooth},
        {"ShadeSmoothByAngle", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::ShadeSmoothByAngle>, 0xa655f6a012146ff2ull, [](ProcedureFixture &f) { f.Do(action::object::ShadeSelectedSmoothByAngle{.6f}); }},
        // Fits against independent least-squares, SciPy spline and Blender references, each leaving the unselected far point.
        {.Name = "Circularize", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 3, 0}, {-1, 2, 0}, Far}, {{0, 1, 2, 3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<Circularize>, .Gesture = [](ProcedureFixture &f) { f.Do(Circularize{}); }, .Expected = {.Positions = {{-.4543723752f, .2665205426f, 0}, {1.74700349f, -.2120457217f, 0}, {2.882409607f, 1.733702031f, 0}, {1.382753314f, 3.41480654f, 0}, {-.6794913643f, 2.508038512f, 0}, Far}}},
        {.Name = "CircularizeOpen", .Shape = +[] { return Sourced({{2, 0, 0}, {1.2f, 1, 0}, {0, 2, 0}, {-1, 1.1f, 0}, {-2, 0, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<Circularize>, .Gesture = [](ProcedureFixture &f) { f.Do(Circularize{.Method = CircleFit::Contract, .Regular = false}); }, .Expected = {.Positions = {{1.08173014f, .2805637001f, 0}, {1.074599406f, .9587836917f, 0}, {.001674496185f, 1.730462079f, 0}, {-1, 1.1f, 0}, {-1.065226088f, .2831478019f, 0}, Far}}},
        {.Name = "CurveBetweenSelected", .Shape = +[] { return Sourced({{0, 0, 0}, {.5f, 1, 0}, {1, 0, .2f}, {2, 1, .5f}, {3, -1, 0}, {4, 0, 0}, {5, 1, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}); }, .Domain = Element::Vertex, .Kind = KindOf<CurveBetweenSelected>, .Gesture = [](ProcedureFixture &f) { f.Do(CurveBetweenSelected{}); }, .Select = {1, 3, 5}, .Expected = {.Positions = {{0, 0, 0}, {.5f, 1, 0}, {1.203125f, 1.09375f, .34375f}, {2, 1, .5f}, {2.953125f, .59375f, .34375f}, {4, 0, 0}, {5, 1, 0}, Far}}},
        {.Name = "Flatten", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {2, 2, 1}, {0, 2, 0}, {10, 0, 0}, {10, 2, 0}, {11, 2, 2}, {10, 0, 2}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::Flatten>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Flatten{}); }, .Expected = {.Positions = {{.06480586011f, .06480586011f, -.2449150189f}, {1.941974115f, -.05802588532f, .2192920636f}, {2.051245911f, 2.051245911f, .8063308916f}, {-.05802588532f, 1.941974115f, .2192920636f}, {9.755084981f, .06480586011f, .06480586011f}, {10.21929206f, 1.941974115f, -.05802588532f}, {10.80633089f, 2.051245911f, 2.051245911f}, {10.21929206f, -.05802588532f, 1.941974115f}, Far}}},
        {.Name = "RelaxEdgeLoops", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {3, 0, 0}, {3, 2, 0}, {0, 2, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}); }, .Domain = Element::Edge, .Kind = KindOf<RelaxEdgeLoops>, .Gesture = [](ProcedureFixture &f) { f.Do(RelaxEdgeLoops{.EvenSpacing = false}); }, .Expected = {.Positions = {{.257891029f, .292481363f, 0}, {.982795715f, -.248387098f, 0}, {2.63177109f, .308072478f, 0}, {2.78709674f, 1.73870969f, 0}, {.611708343f, 1.09812188f, 0}, Far}}},
        {.Name = "SpaceEvenlyLinear", .Shape = +[] { return Sourced({{0, 0, 0}, {.25f, 1, 0}, {2, 0, 0}, {3, 2, 0}, {5, 0, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<SpaceEvenly>, .Gesture = [](ProcedureFixture &f) { f.Do(SpaceEvenly{.Interpolation = EdgeLoopInterpolation::Linear}); }, .Expected = {.Positions = {{0, 0, 0}, {1.11557984f, .50538290f, 0}, {2.45127296f, .90254611f, 0}, {3.56619334f, 1.43380666f, 0}, {5, 0, 0}, Far}}},
        {.Name = "SpaceEvenlyCubic", .Shape = +[] { return Sourced({{0, 0, 0}, {.25f, 1, 0}, {2, 0, 0}, {3, 2, 0}, {5, 0, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<SpaceEvenly>, .Gesture = [](ProcedureFixture &f) { f.Do(SpaceEvenly{.Interpolation = EdgeLoopInterpolation::Cubic}); }, .Expected = {.Positions = {{0, 0, 0}, {1.05305076f, .56604123f, 0}, {2.54712796f, .72494555f, 0}, {3.43722606f, 2.06138992f, 0}, {5, 0, 0}, Far}}},
        {.Name = "SmoothVertices", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 2, 0}, {4, 0, 0}, {9, 9, 9}}, {}, {{0, 1}, {1, 2}}); }, .Domain = Element::Vertex, .Kind = KindOf<SmoothVertices>, .Gesture = [](ProcedureFixture &f) { f.Do(SmoothVertices{.Factor = .5f, .Repeat = 2u, .X = false}); }, .Select = {1, 3}, .Expected = {.Positions = {{0, 0, 0}, {1, .5f, 0}, {4, 0, 0}, {9, 9, 9}}}},
        {.Name = "SmoothZero", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 2, 0}, {4, 0, 0}, {9, 9, 9}}, {}, {{0, 1}, {1, 2}}); }, .Domain = Element::Vertex, .Kind = KindOf<SmoothVertices>, .Gesture = [](ProcedureFixture &f) { f.Do(SmoothVertices{.Factor = 0.f}); }, .Select = {1, 3}, .Expected = {.Unchanged = true}},
        {.Name = "MakePlanarFaces", .Shape = +[] { return Sourced({{-1, -1, 1}, {1, -1, -1}, {1, 1, 1}, {-1, 1, -1}, {3, 0, 2}, {4, 0, 2}, {3, 1, 2}}, {{0, 1, 2, 3}, {4, 5, 6}}); }, .Domain = Element::Face, .Kind = KindOf<MakePlanarFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(MakePlanarFaces{}); }, .Expected = {.Positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {3, 0, 2}, {4, 0, 2}, {3, 1, 2}}}},
        // bpy.ops.transform.shrink_fatten(value=.2), Blender 5.2.2 d13f752e3b9c, where the unselected third face changes vertex normals but not face-mode normals, and every user of a nonmanifold edge contributes to its normal and shell factor.
        {.Name = "ShrinkFattenFaces", .Shape = Fan3, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Select = {0, 1}, .Expected = {.Positions = {{0, .141421363f, .141421363f}, {2, .141421363f, .141421363f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenFacesEven", .Shape = Fan3, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Select = {0, 1}, .Expected = {.Positions = {{0, .2f, .2f}, {2, .2f, .2f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenVertices", .Shape = Fan3, .Domain = Element::Vertex, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Select = {0, 1, 2}, .Expected = {.Positions = {{.033676531f, .121543162f, .155219704f}, {2, .141421363f, .141421363f}, {.100692429f, 1, .172803462f}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenVerticesEven", .Shape = Fan3, .Domain = Element::Vertex, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Select = {0, 1, 2}, .Expected = {.Positions = {{.043392085f, .156607896f, .2f}, {2, .2f, .2f}, {.116539836f, 1, .2f}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenNonmanifold", .Shape = NonmanifoldFan, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Expected = {.Positions = {{0, .039776370f, .196004689f}, {2, .033695214f, .197141156f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 1.91055727f, 1.17888546f}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}}},
        {.Name = "ShrinkFattenNonmanifoldEven", .Shape = NonmanifoldFan, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Expected = {.Positions = {{0, .093217827f, .459346354f}, {2, .086327799f, .505079567f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 1.91055727f, 1.17888546f}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}}},
        // Radial edits about the selection center, whose mean radius the 252 central vertices dilute, and the far point unselected.
        {.Name = "ToSphere", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::ToSphere{.25f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Positions = Scaled(Cross(), 4u, [](float radius) { return (radius * .75f + (6.f / 257.f) * .25f) / radius; })}},
        {.Name = "PushPull", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::PushPull>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::PushPull{-.4f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Positions = Scaled(Cross(), 4u, [](float radius) { return (radius + .4f) / radius; })}},
        {.Name = "ToSphereZero", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::ToSphere{0.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        {.Name = "ToSphereNegative", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::ToSphere{-1.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        {.Name = "PushPullZero", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::PushPull>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::PushPull{0.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        // Blender 5.2.2 edge_slide with its positive side up, over quad strips, triangle strips with forked rails, a pentagon, a quad turn, a transformed nonplanar fork and a closed loop.
        {.Name = "EdgeSlide", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.125f, .5f, 0}, {2, .75f, 0}, {4.25f, .25f, 0}})}},
        {.Name = "EdgeSlideBack", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = -.25f, .Direction = {0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.05f, -.25f, 0}, {2, -.5f, 0}, {4.125f, -.25f, 0}})}},
        {.Name = "EdgeSlideEvenFlipped", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}, .Even = true, .Flipped = true}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.22286583f, .89146334f, 0}, {2, -.11564839f, 0}, {4.85387611f, .85387623f, 0}})}},
        {.Name = "EdgeSlideUnclamped", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = -.25f, .Direction = {0, 1, 0}, .Clamp = false}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{.125f, -.5f, 0}, {2, -.75f, 0}, {3.75f, -.25f, 0}})}},
        {.Name = "EdgeSlideEvenZero", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = 0.f, .Direction = {0, 1, 0}, .Even = true}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.1304878f, .5219512f, 0}, {2, 1.49251878f, 0}, {4.17364359f, -.34728706f, 0}})}},
        {.Name = "EdgeSlideTriangles", .Shape = +[] { return SlideTriangles(); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideTriangles(), {3, 4, 5}, {{.5f, .75f, 0}, {2.375f, .5f, 0}, {4.25f, .25f, 0}})}},
        {.Name = "EdgeSlidePentagon", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {1.5f, 3, 0}, {0, 2, 0}}, {{0, 1, 2, 3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}}); }, .Select = {0, 1, 1, 2}, .Expected = {.Positions = {{0, .5f, 0}, {1.72951460f, .65300971f, 0}, {2.625f, 1.5f, 0}, {1.5f, 3, 0}, {0, 2, 0}}}},
        {.Name = "EdgeSlideQuadTurn", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {0, 2, 0}}, {{0, 1, 2, 3}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}}); }, .Select = {0, 1, 1, 2}, .Expected = {.Positions = {{0, .5f, 0}, {1.5f, .5f, 0}, {2.25f, 2, 0}, {0, 2, 0}}}},
        {.Name = "EdgeSlideTransformed", .Shape = +[] { return SlideTriangles({.4f, -.2f, .5f, 0, 0, 0, .3f, .6f, -.3f}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = SlideWorld.R * vec3{0, -1, 0}}); }, .World = SlideWorld, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideTriangles({.4f, -.2f, .5f, 0, 0, 0, .3f, .6f, -.3f}), {3, 4, 5}, {{.5f, .75f, .15f}, {2.375f, .5f, .0375f}, {4.25f, .25f, -.075f}})}},
        {.Name = "EdgeSlideNonmanifold", .Shape = NonmanifoldEdge, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{}); }, .Select = {0, 1}, .Expected = {.Unchanged = true}},
        {.Name = "EdgeSlideLoop", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {1, 1, 0}}); }, .Select = {4, 5, 5, 6, 6, 7, 7, 4}, .Expected = {.Positions = Replaced(Rings(), {4, 5, 6, 7}, {{2.5f, 2.5f, 0}, {-2.5f, 2.5f, 0}, {-2.5f, -2.5f, 0}, {2.5f, -2.5f, 0}})}},
        {.Name = "EdgeSlideLoopEven", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {1, 1, 0}, .Even = true}); }, .Select = {4, 5, 5, 6, 6, 7, 7, 4}, .Expected = {.Positions = Replaced(Rings(), {4, 5, 6, 7}, {{2.875f, 2.875f, 0}, {-2.875f, 2.875f, 0}, {-2.875f, -2.875f, 0}, {2.875f, -2.875f, 0}})}},
        {.Name = "EdgeSlideBranch", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::EdgeSlide{}); }, .Select = {4, 5, 4, 7, 4, 8}, .Expected = {.Unchanged = true}},
        // Blender 5.2.2 vert_slide with an explicit world direction and correct_uv=false, the first selected vertex its reference.
        {.Name = "VertexSlide", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = {1, 0, 0}}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{.5f, 0, 0}, {1, 4, 0}, {.25f, 8, 0}})}},
        {.Name = "VertexSlideEvenFlipped", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = {1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{1.5f, 0, 0}, {3.5f, 4, 0}, {.5f, 8, 0}})}},
        {.Name = "VertexSlideEvenZero", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = 0.f, .Direction = {1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{2, 0, 0}, {4, 4, 0}, {1, 8, 0}})}},
        {.Name = "VertexSlideUnclamped", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = -.5f, .Direction = {1, 0, 0}, .Even = true, .Clamp = false}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{-1, 0, 0}, {-1, 4, 0}, {-1, 8, 0}})}},
        {.Name = "VertexSlideTransformed", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = {-.3f, 1, .2f}, .Even = true}); }, .World = VertexSlideWorld, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{0, .5f, 0}, {-.223606795f, 4.447213650f, 0}, {0, 8.5f, 0}})}},
        // The active vertex is the even slide's reference, both endpoints of one edge meet at its middle, and a zero-length edge or a fork leaves its vertex.
        {.Name = "VertexSlideActive", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { Activate(f, 2u); f.Do(action::mesh::VertexSlide{.Factor = .25f, .Even = true}); }, .Select = {0, 2, 7}, .Expected = {.Positions = Replaced(SlideWires(), {0, 2}, {{1, 0, 0}, {1, 3, 0}})}},
        {.Name = "VertexSlideBothEnds", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{}); }, .Select = {0, 1}, .Expected = {.Positions = Replaced(SlideWires(), {0, 1}, {{1, 0, 0}, {1, 0, 0}})}},
        {.Name = "VertexSlideDegenerate", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Direction = {-1, 0, 0}, .Flipped = true}); }, .Select = {4, 7}, .Expected = {.Positions = SlideWires().Data.Positions}},
        {.Name = "VertexSlideDegenerateEven", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Direction = {-1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {4, 7}, .Expected = {.Positions = SlideWires().Data.Positions}},
        {.Name = "VertexSlideZeroDirection", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Direction = {}}); }, .Select = {0, 2, 7}, .Expected = {.Unchanged = true}},
        {.Name = "VertexSlideNan", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::VertexSlide{.Factor = std::numeric_limits<float>::quiet_NaN()}); }, .Select = {0, 2, 7}, .Expected = {.Unchanged = true}},
        // Blender 5.2.2 snap to symmetry: interpolate toward positive partners, unselected partners included, center eligible vertices and leave the unmatched, where the first canonical pair owns its partner and the competing last vertex stays.
        {.Name = "SnapSymmetry", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f) { f.Do(SnapSymmetry{.Threshold = .2f, .Factor = .25f, .Negative = false}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Positions = {{-1.06f, .03f, .015f}, {1.06f, .03f, .015f}, {-2.015f, 1.0375f, .2375f}, {2.015f, 1.0375f, .2375f}, {0, 2, .3f}, {4, 2, 3}, {.1f, 4, .6f}, {-1.02f, 0, 0}}}},
        {.Name = "SnapSymmetryNegative", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f) { f.Do(SnapSymmetry{.Threshold = .2f, .Factor = 0.f, .Negative = true, .Center = false}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Positions = Replaced(Symmetric(), {1, 3}, {{1, 0, 0}, {2, 1, .2f}})}},
        {.Name = "SnapSymmetryZeroThreshold", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f) { f.Do(SnapSymmetry{.Threshold = 0.f}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Unchanged = true}},
        // A bend about the world plane onto a quarter circle, which stays put near zero angle, and a warp that continues downward beyond its bounds, with the world XY plane as the viewport plane.
        {.Name = "Bend", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Bend{.Angle = std::numbers::pi_v<float> / 2.f, .Radius = 2.f}); }, .World = BendWorld, .Expected = {.Positions = [] {
                                                                                                                                                                         std::vector<vec3> positions;
                                                                                                                                                                         const float root2 = std::sqrt(2.f);
                                                                                                                                                                         for (const auto p : {vec3{-1, 0, 0}, vec3{root2, root2 - 2.f, 1}, vec3{2, -2, 2}, vec3{2, -3, 3}}) positions.push_back(Local(BendWorld, p));
                                                                                                                                                                         return positions;
                                                                                                                                                                     }()}},
        {.Name = "BendSmallAngle", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Bend{.Angle = 1e-6f, .Radius = 2.f, .Clamp = false}); }, .World = BendWorld, .Expected = {.Positions = BendChain().Data.Positions}},
        {.Name = "BendZeroRadius", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Bend{.Radius = 0.f}); }, .World = BendWorld, .Expected = {.Unchanged = true}},
        {.Name = "Warp", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Warp{.Angle = std::numbers::pi_v<float>, .AutoRange = false, .Min = 2.f, .Max = -2.f}); }, .Expected = {.Positions = [] {
                                                                                                                                                   std::vector<vec3> positions;
                                                                                                                                                   for (uint32_t i = 0u; i < 265u; ++i) positions.push_back(std::array{vec3{-1, -1, 0}, vec3{-1, 0, 0}, vec3{0, 1, 0}, vec3{1, 0, 0}, vec3{1, -1, 0}}[i % 5u] + vec3{0, 0, float(i / 5u)});
                                                                                                                                                   return positions;
                                                                                                                                               }()}},
        {.Name = "WarpAuto", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Warp{.Angle = std::numbers::pi_v<float>, .AutoRange = true, .Min = 2.f, .Max = -2.f}); }, .Expected = {.Positions = [] {
                                                                                                                                                       std::vector<vec3> positions;
                                                                                                                                                       const float root3 = std::sqrt(3.f) * .5f;
                                                                                                                                                       for (uint32_t i = 0u; i < 265u; ++i) positions.push_back(std::array{vec3{-1, 0, 0}, vec3{-root3, .5f, 0}, vec3{0, 1, 0}, vec3{root3, .5f, 0}, vec3{1, 0, 0}}[i % 5u] + vec3{0, 0, float(i / 5u)});
                                                                                                                                                       return positions;
                                                                                                                                                   }()}},
        {.Name = "WarpFlat", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Warp{.Angle = 0.f}); }, .Expected = {.Positions = [] {
                                                                                                                                                       std::vector<vec3> positions;
                                                                                                                                                       for (uint32_t i = 0u; i < 265u; ++i) positions.push_back({0, 1, float(i / 5u)});
                                                                                                                                                       return positions;
                                                                                                                                                   }()}},
        {.Name = "WarpEmptyRange", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f) { LookDown(f); f.Do(action::mesh::Warp{.AutoRange = false, .Min = 1.f, .Max = 1.f}); }, .Expected = {.Unchanged = true}},
        // bmesh.ops.unsubdivide, beautify_fill and connect_verts_concave partitions, Blender 5.2.2 d13f752e3b9c, as oriented loops of source vertices.
        {.Name = "UnsubdivideOnce", .Shape = +[] { return Grid(3u); }, .Domain = Element::Vertex, .Kind = KindOf<Unsubdivide>, .Gesture = [](ProcedureFixture &f) { f.Do(Unsubdivide{1u}); }, .Expected = {.Faces = {{9, 4, 1, 6}, {0, 1, 4}, {14, 9, 6, 11}, {9, 12, 4}, {9, 14, 12}, {1, 3, 6}, {11, 15, 14}, {3, 11, 6}}}},
        {.Name = "UnsubdivideTwice", .Shape = +[] { return Grid(3u); }, .Domain = Element::Vertex, .Kind = KindOf<Unsubdivide>, .Gesture = [](ProcedureFixture &f) { f.Do(Unsubdivide{2u}); }, .Expected = {.Faces = {{9, 14, 4}, {0, 1, 4}, {14, 9, 11}, {9, 4, 1}, {11, 9, 1, 3}, {11, 15, 14}}}},
        {.Name = "BeautifyArea", .Shape = +[] { return Fanned({{-5, 0, 0}, {-4, -3, 0}, {-1, -4, 0}, {3, -3, 0}, {6, 0, 0}, {5, 4, 0}, {2, 6, 0}, {-2, 5, 0}}); }, .Domain = Element::Face, .Kind = KindOf<BeautifyFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(BeautifyFaces{BeautifyMethod::Area}); }, .Select = {0, 1, 2, 3, 4, 5}, .Expected = {.Faces = {{0, 1, 2}, {0, 2, 7}, {2, 3, 7}, {3, 4, 5}, {3, 5, 6}, {3, 6, 7}, {1, 0, 8}, {0, 7, 8, 9}}}},
        {.Name = "BeautifyAngle", .Shape = +[] { return Fanned({{0, 0, 0}, {3, 0, .3f}, {4, 1, -.7f}, {3, 3, .2f}, {1, 4, -.4f}, {-1, 2, .6f}}); }, .Domain = Element::Face, .Kind = KindOf<BeautifyFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(BeautifyFaces{BeautifyMethod::Angle}); }, .Select = {0, 1, 2, 3}, .Expected = {.Faces = {{0, 1, 3}, {0, 3, 4}, {0, 4, 5}, {1, 2, 3}, {1, 0, 6}, {0, 5, 6, 7}}}},
        {.Name = "SplitConcaveFaces", .Shape = +[] { return Sourced({{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {3, 4, 0}, {3, 1, 0}, {1, 1, 0}, {1, 4, 0}, {0, 4, 0}, {10, 0, 0}, {13, 0, 0}, {13, 1, 0}, {11, 1, 0}, {11, 3, 0}, {10, 3, 0}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13}}); }, .Domain = Element::Face, .Kind = KindOf<SplitConcaveFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(SplitConcaveFaces{}); }, .Select = {0}, .Expected = {.Faces = {{0, 5, 6, 7}, {0, 1, 4, 5}, {1, 2, 3, 4}, {8, 9, 10, 11, 12, 13}}}},
        {.Name = "ConnectVertices", .Shape = +[] { return Sourced({{2, 0, 0}, {1, 2, 0}, {-1, 2, 0}, {-2, 0, 0}, {-1, -2, 0}, {1, -2, 0}}, {{0, 1, 2, 3, 4, 5}}); }, .Domain = Element::Vertex, .Kind = KindOf<ConnectVertices>, .Gesture = [](ProcedureFixture &f) { f.Do(ConnectVertices{}); }, .Select = {0, 2, 4}, .Expected = {.Faces = {{0, 1, 2}, {2, 3, 4}, {0, 4, 5}, {0, 2, 4}}}},
        // A hexagon folded ninety degrees splits along its crease, and a nonplanar polygon has no valence limit.
        {.Name = "SplitNonplanarFaces", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 1}, {1, 2, 1}, {0, 2, 0}, {-1, 2, 1}, {-1, 0, 1}, {3, 0, 0}, {4, 0, 0}, {4, 1, 0}, {3, 1, 0}, {2, 0, 0}, {2, 2, 2}, {3, 2, 1}}, {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9}, {2, 1, 10, 11}, {8, 12, 9}}); }, .Domain = Element::Face, .Kind = KindOf<SplitNonplanarFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(SplitNonplanarFaces{}); }, .Select = {0, 1, 3}, .Expected = {.Faces = {{0, 1, 2, 3}, {0, 3, 4, 5}, {6, 7, 8, 9}, {2, 1, 10, 11}, {8, 12, 9}}}},
        {.Name = "SplitNonplanarPolygon", .Shape = Ridge, .Domain = Element::Face, .Kind = KindOf<SplitNonplanarFaces>, .Gesture = [](ProcedureFixture &f) { f.Do(SplitNonplanarFaces{}); }, .Expected = {.Faces = {Ordinals(0u, 33u), [] { auto loop = Ordinals(33u, 65u); loop.push_back(0u); return loop; }()}}},
        {.Name = "SplitNonplanarConcave", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {1, 1, 1}, {0, 2, 0}}, {{0, 1, 2, 3, 4}}); }, .Domain = Element::Face, .Kind = KindOf<SplitNonplanarFaces>, .Cooks = 0xd6e602f619f437f5ull, .Gesture = [](ProcedureFixture &f) { f.Do(SplitNonplanarFaces{}); }},
        // A closed or nonmanifold input passes through decimation, and a half-ratio triangle grid ends at 35 or 36 triangles in Blender 5.2.2.
        {.Name = "DecimateClosed", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 1, 2}, {1, 0, 3}, {1, 3, 2}, {0, 2, 3}}); }, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Decimate>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Decimate{0.f}); }, .Expected = {.Faulted = true}},
        {.Name = "DecimateNonmanifold", .Shape = NonmanifoldEdge, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Decimate>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Decimate{0.f}); }, .Expected = {.Faulted = true}},
        {.Name = "DecimateGrid", .Shape = +[] { return Grid(6u, true); }, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Decimate>, .Cooks = 0xb10f96ab40612913ull, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Decimate{.5f}); }},
        // A wireframe of the unit quad takes the analytic wire positions for each option, and a cube's wireframe closes.
        {.Name = "WireframeQuad", .Shape = +[] { return Sourced({{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {{0, 1, 2, 3}}); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Wireframe{.Thickness = .2f, .Offset = .5f}); }, .Expected = {.Points = WirePoints(.2f, .5f, true, true, true)}},
        {.Name = "WireframeQuadKept", .Shape = +[] { return Sourced({{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {{0, 1, 2, 3}}); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Wireframe{.Thickness = .2f, .Offset = -1.f, .Replace = false, .Boundary = false, .Even = false}); }, .Expected = {.Points = WirePoints(.2f, -1.f, false, false, false)}},
        {.Name = "WireframeQuadRelative", .Shape = +[] { return Sourced({{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {{0, 1, 2, 3}}); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Wireframe{.Thickness = .1f, .Even = false, .Relative = true}); }, .Expected = {.Points = WirePoints(.2f, 0.f, false, true, true)}},
        {.Name = "WireframeCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Cooks = 0x6b7bb935470e34full, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Wireframe{.Thickness = .2f}); }},
        {.Name = "WireframeCubeFace", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Cooks = 0xd6c9af0ed7205eaaull, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Wireframe{.Thickness = .1f}); }, .Select = {0}},
        // A spin of an open grid and of a closed tetrahedron, whose UVs and colors turn with each step.
        {.Name = "SpinGrid", .Shape = +[] { return Attributed(Grid(2u)); }, .Domain = Element::Face, .Kind = KindOf<procedure::Spin>, .Cooks = 0xde7b333f3d2cc3aaull, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Spin{.Steps = 3u, .Angle = std::numbers::pi_v<float> * 1.5f, .Center = {.5f, .5f, 0.f}, .Offset = 6.f}); }},
        {.Name = "SpinTetrahedron", .Shape = +[] { return Attributed(Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}})); }, .Domain = Element::Face, .Kind = KindOf<procedure::Spin>, .Cooks = 0xc810f125bae3db2eull, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Spin{.Steps = 3u, .Angle = std::numbers::pi_v<float> * 1.5f, .Center = {.5f, .5f, 0.f}, .Offset = 6.f}); }},
        // bmesh.ops.recalc_face_normals, Blender 5.2.2 d13f752e3b9c, flips faces 0, 2, 5, 6 and 15 of the selected cubes and fans, inside out every selected face.
        {.Name = "RecalculateNormals", .Shape = Windings, .Domain = Element::Face, .Kind = KindOf<RecalculateNormals>, .Cooks = 0xbb180aedcd280622ull, .Gesture = [](ProcedureFixture &f) { f.Do(RecalculateNormals{}); }, .Select = {0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17}},
        {.Name = "RecalculateNormalsInside", .Shape = Windings, .Domain = Element::Face, .Kind = KindOf<RecalculateNormals>, .Cooks = 0x909b16c440ec2e42ull, .Gesture = [](ProcedureFixture &f) { f.Do(RecalculateNormals{true}); }, .Select = {0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17}},
        {.Name = "PointHull", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}); }, .Domain = Element::Vertex, .Kind = KindOf<ConvexHull>, .Cooks = 0x6abda59ec196d3b5ull, .Gesture = [](ProcedureFixture &f) { f.Do(ConvexHull{}); }},
        // A shear by NaN cooks an identity, while a shear along one axis, a separation that leaves the cube whole and face layer edits on absent layers append no node.
        {.Name = "ShearSameAxis", .Shape = primitive::Cube{}, .Domain = Element::Vertex, .Kind = NoKind, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Shear{.6f, ShearAxis::X, ShearAxis::X}); }},
        {.Name = "ShearNan", .Shape = primitive::Cube{}, .Domain = Element::Vertex, .Kind = KindOf<procedure::Shear>, .Gesture = [](ProcedureFixture &f) { f.Do(action::mesh::Shear{std::numeric_limits<float>::quiet_NaN()}); }, .Expected = {.Unchanged = true}},
        {.Name = "SeparateLooseCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f) { f.Do(Separate{SeparateMode::LooseParts}); }},
        {.Name = "SeparateMaterialCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f) { f.Do(Separate{SeparateMode::Material}); }},
        {.Name = "AbsentFaceLayers", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f) {
             f.Do(RotateColors{});
             f.Do(ReverseColors{});
             f.Do(RotateUVs{3u});
             f.Do(ReverseUVs{3u});
         }},
    };
    // Adds an active cube at Placed and a selected second cube, the tool, and returns the active object, its mesh and the tool.
    const auto add_cubes = [](ProcedureFixture &f) {
        auto &r = f.R;
        f.Do(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = Placed})});
        const auto active = FindActiveEntity(r);
        const Transform world{.P = {1.f, .5f, -.25f}, .R = numeric::AngleAxis(.6f, Normalize(vec3{0.f, 1.f, 1.f})), .S = {.75f, 1.f, 1.25f}};
        f.Do(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = world, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
        expect(fatal(FindActiveEntity(r) == active));
        state::Entity tool = state::Null;
        for (const auto e : r.view<const Instance>())
            if (e != active) tool = e;
        return std::tuple{active, r.get<const Instance>(active).Entity, tool};
    };
    // The enclosed volume of the record the mesh draws.
    const auto volume = [](ProcedureFixture &f, state::Entity mesh) { return test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0); };
    // A boolean with another object appends a Boolean node in the active object's space and an Object Info node that emits the other object's mesh placed by that object, which cooks the recorded difference.
    Scenario(f, other, "BooleanObject", [&](ProcedureFixture &f) {
        auto &r = f.R;
        const auto [active, mesh, tool] = add_cubes(f);
        const auto node_count = uint32_t(r.get<const CookPlan>(mesh).Nodes.size());
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(fatal(procedure.Nodes.size() == node_count + 2u));
        const auto [boolean, boolean_kind] = procedure.Nodes[node_count];
        const auto [info, info_kind] = procedure.Nodes[node_count + 1u];
        expect(boolean_kind == KindOf<procedure::Boolean> && info_kind == KindOf<procedure::ObjectInfo> && r.get<const procedure::ObjectInfo>(info).Object == tool);
        expect(procedure.Inputs[procedure.InputFirst[node_count] + 1u].FromNode == node_count + 1u && ActiveIndex(r, procedure) == node_count);
        expect(!r.any_of<NodeError>(boolean) && !r.any_of<NodeError>(info));
        expect(r.get<const procedure::Boolean>(boolean).Space == active);
        expect(Fingerprint(f, mesh) == 0x38e0585db10ca082ull);
        // The difference follows both objects: moving either away leaves the cube whole, and moving it back cuts it again.
        for (const auto moved : {tool, active}) {
            const auto placed = r.get<const Transform>(moved).P;
            f.Do(UpdateOf<&Transform::P>(moved, placed + vec3{10.f, 0.f, 0.f}));
            expect(std::abs(volume(f, mesh) - 8.0) < 1e-4);
            f.Do(UpdateOf<&Transform::P>(moved, placed));
            expect(Fingerprint(f, mesh) == 0x38e0585db10ca082ull);
        }
        // A generated Mesh 2 is in this procedure's mesh space, so it stays in place relative to Mesh 1 wherever Space moves: a bar through the cube cuts a 1 x 1 x 2 tunnel.
        const auto bar = f.Append(mesh, KindOf<primitive::Cube>);
        f.Do(UpdateOf<&primitive::Cube::HalfExtents>(bar, vec3{.5f, .5f, 2.f}));
        f.Do(Link{mesh, bar, f.Out(bar), boolean, 1u});
        expect(!r.any_of<NodeError>(boolean) && std::abs(volume(f, mesh) - 6.0) < 1e-4);
        // Its placement is the identity wherever the objects move, so moving Space cooks nothing.
        const auto cooks = Cooks(r);
        const auto placed = r.get<const Transform>(active).P;
        f.Do(UpdateOf<&Transform::P>(active, placed + vec3{10.f, 0.f, 0.f}));
        expect(Cooks(r) == cooks && std::abs(volume(f, mesh) - 6.0) < 1e-4);
        f.Do(UpdateOf<&Transform::P>(active, placed));
    });
    // Moving a common parent of both objects leaves the transform between their spaces unchanged bit for bit, so the procedure does not cook, and moving one of them cooks it.
    Scenario(f, other, "BooleanCommonParent", [&](ProcedureFixture &f) {
        auto &r = f.R;
        const auto parent = f.AddEmpty("Parent");
        const auto [active, mesh, tool] = add_cubes(f);
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        f.Do(action::selection::ExtendActive{parent});
        f.Do(action::object::ParentToActive{});
        expect(fatal(ParentOrNull(r, active) == parent && ParentOrNull(r, tool) == parent));
        const auto cut = Fingerprint(f, mesh);
        const auto cooks = Cooks(r);
        f.Do(UpdateOf<&Transform::P>(parent, vec3{3.f, -2.f, 1.f}));
        f.Do(UpdateOf<&Transform::R>(parent, numeric::AngleAxis(.7f, Normalize(vec3{1.f, 0.f, 1.f}))));
        f.Do(UpdateOf<&Transform::S>(parent, vec3{1.5f, 1.5f, 1.5f}));
        expect(Cooks(r) == cooks && Fingerprint(f, mesh) == cut);
        f.Do(UpdateOf<&Transform::P>(tool, r.get<const Transform>(tool).P + vec3{10.f, 0.f, 0.f}));
        expect(Cooks(r) == cooks + 1u && std::abs(volume(f, mesh) - 8.0) < 1e-4);
    });
    // A pose on an object placing an operand re-cooks the Boolean, and the next pass, which only reads the posed world transform the first published, does not cook again.
    Scenario(f, other, "BooleanPose", [&](ProcedureFixture &f) {
        auto &r = f.R;
        const auto [active, mesh, tool] = add_cubes(f);
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        const auto cut = Fingerprint(f, mesh);
        const auto cooks = Cooks(r);
        auto posed = r.get<const Transform>(tool);
        posed.P += vec3{10.f, 0.f, 0.f};
        r.emplace<PosedLocal>(tool, posed);
        f.Commit("Pose tool");
        expect(Cooks(r) == cooks + 1u && std::abs(volume(f, mesh) - 8.0) < 1e-4);
        f.P->Settle();
        expect(Cooks(r) == cooks + 1u);
        r.remove<PosedLocal>(tool);
        f.Commit("Release pose");
        expect(Cooks(r) == cooks + 2u && Fingerprint(f, mesh) == cut);
    });
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            if (const auto *shape = std::get_if<PrimitiveParameters>(&row.Shape)) {
                f.Do(action::object::AddMeshPrimitive{*shape, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = row.World})});
            } else {
                AddSourced(f, std::get<MeshSource (*)()>(row.Shape)(), row.World);
            }
            const auto mesh = r.get<const Instance>(FindActiveEntity(r)).Entity;
            f.EnterEdit(row.Domain);
            if (row.Prepare) row.Prepare(f);
            if (row.Targets) {
                f.Render();
                const auto targets = row.Targets(Mesh{meshes, f.Output(mesh)}, f.View().CameraPosition);
                expect(fatal(!targets.empty()));
                for (uint32_t i = 0u; i < targets.size(); ++i) f.Click(targets[i], i > 0u);
            } else if (!row.Select.empty()) {
                const auto selected = row.Domain == Element::Edge ? EdgeOrdinals(Mesh{meshes, f.Output(mesh)}, row.Select) : row.Select;
                ApplyEditSelectionLists(r, std::array{std::pair{mesh, std::span<const uint32_t>{selected}}}, row.Domain);
                f.Commit("Select elements");
            } else {
                f.Do(action::selection::SelectAll{});
            }
            std::vector<vec3> source;
            for (const Mesh before{meshes, f.Output(mesh)}; const auto v : before.vertices()) source.push_back(before.GetPosition(v));
            const auto shape = f.Shaped(mesh);
            const auto node_count = r.get<const CookPlan>(mesh).Nodes.size();
            const auto present = f.P->History.Present.value();
            row.Gesture(f);
            const auto &procedure = r.get<const CookPlan>(mesh);
            if (row.Kind == NoKind) {
                expect(procedure.Nodes.size() == node_count && f.P->History.Present == present);
                expect(f.Shaped(mesh) == shape);
                return;
            }
            expect(fatal(procedure.Nodes.size() == node_count + 1u));
            const auto [node, kind] = procedure.Nodes.back();
            const auto &expected = row.Expected;
            expect(kind == row.Kind && r.any_of<NodeError>(node) == expected.Faulted);
            if (expected.Unchanged || expected.Faulted) {
                expect(f.Shaped(mesh) == shape);
                return;
            }
            if (expected.Positions.empty() && expected.Points.empty() && expected.Faces.empty()) expect(Fingerprint(f, mesh) == row.Cooks) << std::format("fingerprint {:#x}", Fingerprint(f, mesh));
            const Mesh result{meshes, f.Output(mesh)};
            if (!expected.Positions.empty()) {
                expect(result.VertexCount() == expected.Positions.size());
                for (uint32_t i = 0u; i < std::min<size_t>(result.VertexCount(), expected.Positions.size()); ++i) {
                    const auto p = result.GetPosition(result.VertexAt(i));
                    expect(Near(p, expected.Positions[i])) << "vertex" << i << p.x << p.y << p.z;
                }
            }
            if (!expected.Points.empty()) {
                std::vector<vec3> actual;
                for (const auto v : result.vertices()) actual.push_back(result.GetPosition(v));
                const auto sorted = Sorted(actual), points = Sorted(expected.Points);
                expect(sorted.size() == points.size());
                for (uint32_t i = 0u; i < std::min(sorted.size(), points.size()); ++i) expect(Near(sorted[i], points[i])) << "point" << i << sorted[i].x << sorted[i].y << sorted[i].z;
            }
            if (!expected.Faces.empty()) expect(Canonical(Loops(result, source)) == Canonical(expected.Faces));
        });
    }
}

std::string_view DomainName(Element domain) { return domain == Element::Vertex ? "Vertices" : domain == Element::Edge ? "Edges" :
                                                                                                                        "Faces"; }

// Two unit quads side by side and an unconnected vertex.
MeshSource TwoQuads() { return Sourced({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}}, {{0, 1, 4, 3}, {1, 2, 5, 4}}); }
// Blender 5.2.2 mesh.hide over the two quads: the elements each mode hides when hiding the selection or everything else, with vertex 1, face 0 or the edge from 1 to 4 selected.
struct VisibilityReference {
    Element Mode;
    bool Unselected;
    std::vector<uint32_t> Vertices, Faces;
    std::vector<std::array<uint32_t, 2>> Edges;
};
void EditVisibility(ProcedureFixture &f, Fixture &other) {
    static const std::array<VisibilityReference, 6> references{{
        {Element::Vertex, false, {1}, {0, 1}, {{0, 1}, {1, 2}, {1, 4}}},
        {Element::Vertex, true, {0, 2, 3, 4, 5, 6}, {0, 1}, {{0, 1}, {0, 3}, {1, 2}, {1, 4}, {2, 5}, {3, 4}, {4, 5}}},
        {Element::Edge, false, {}, {0, 1}, {{1, 4}}},
        {Element::Edge, true, {0, 2, 3, 5, 6}, {0, 1}, {{0, 1}, {0, 3}, {1, 2}, {2, 5}, {3, 4}, {4, 5}}},
        {Element::Face, false, {0, 3}, {0}, {{0, 1}, {0, 3}, {3, 4}}},
        {Element::Face, true, {2, 5, 6}, {1}, {{1, 2}, {2, 5}, {4, 5}}},
    }};
    for (const auto &reference : references) {
        Scenario(f, other, std::format("Hide{}{}", reference.Unselected ? "Unselected" : "Selected", DomainName(reference.Mode)), [&reference](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto mesh = AddSourced(f, TwoQuads()).first;
            f.EnterEdit(reference.Mode);
            const auto selected = reference.Mode == Element::Edge ? EdgeOrdinals(Mesh{meshes, f.Output(mesh)}, std::array{1u, 4u}) : std::vector{reference.Mode == Element::Vertex ? 1u : 0u};
            SelectOrdinals(f, mesh, selected, reference.Mode);
            const auto shape = f.Shaped(mesh);
            const auto check = [&] {
                const auto id = f.Output(mesh);
                const Mesh current{meshes, id};
                std::vector<uint32_t> vertices, faces;
                std::vector<std::array<uint32_t, 2>> edges;
                meshes.GetHiddenElements(id, Element::Vertex).ForEach([&](uint32_t v) { vertices.push_back(current.VertexOrdinal(he::VH{v})); });
                meshes.GetHiddenElements(id, Element::Face).ForEach([&](uint32_t face) { faces.push_back(current.FaceOrdinal(he::FH{face})); });
                meshes.GetHiddenElements(id, Element::Edge).ForEach([&](uint32_t e) {
                    const auto h = current.GetHalfedge(he::EH{e}, 0u);
                    std::array pair{current.VertexOrdinal(current.GetFromVertex(h)), current.VertexOrdinal(current.GetToVertex(h))};
                    std::ranges::sort(pair);
                    edges.push_back(pair);
                });
                std::ranges::sort(edges);
                expect(vertices == reference.Vertices && faces == reference.Faces && edges == reference.Edges);
                expect(f.Shaped(mesh) == shape);
            };
            f.Do(action::mesh::Hide{reference.Unselected});
            check();
            f.Do(action::selection::SelectAll{});
            check();
            const auto visible = Mesh{meshes, f.Output(mesh)}.ElementCount(reference.Mode) - meshes.GetHiddenElements(f.Output(mesh), reference.Mode).Count();
            expect(meshes.GetSelectedElements(f.Output(mesh), reference.Mode).Count() == visible);
            const auto hidden_node = f.P->History.Present.value();
            f.Do(action::mesh::Reveal{false});
            expect(meshes.GetSelectedElements(f.Output(mesh), reference.Mode).Count() == visible);
            for (const auto mode : {Element::Vertex, Element::Edge, Element::Face}) expect(meshes.GetHiddenElements(f.Output(mesh), mode).Count() == 0u);
            // Reveal can match the hide node's parent, which history adopts directly.
            f.Navigate(hidden_node);
            check();
            f.Do(action::mesh::Reveal{});
            expect(meshes.GetSelectedElements(f.Output(mesh), reference.Mode).Count() == Mesh{meshes, f.Output(mesh)}.ElementCount(reference.Mode));
        });
    }
}

// A click reaches the visible element behind a hidden one and never selects a hidden element.
void HiddenPicking(ProcedureFixture &f, Fixture &other) {
    for (const auto mode : {Element::Vertex, Element::Edge, Element::Face}) {
        Scenario(f, other, std::format("HiddenPick{}", DomainName(mode)), [mode](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            f.Render();
            const auto normal = Normalize(f.View().CameraPosition), right = Normalize(Cross(normal, vec3{0, 0, 1})), up = Cross(normal, right);
            std::vector<vec3> positions;
            std::vector<std::vector<uint32_t>> faces;
            std::vector<std::array<uint32_t, 2>> edges;
            // Two layers of one element each along the view direction, the nearer first.
            for (uint32_t layer = 0u; layer < 2u; ++layer) {
                const auto center = normal * (layer ? -.5f : .5f);
                if (mode == Element::Vertex) positions.push_back(center);
                else if (mode == Element::Edge) {
                    positions.insert(positions.end(), {center - right, center + right});
                    edges.push_back({2u * layer, 2u * layer + 1u});
                } else {
                    for (const auto offset : {-right - up, right - up, right + up, -right + up}) positions.push_back(center + offset);
                    faces.push_back({4u * layer, 4u * layer + 1u, 4u * layer + 2u, 4u * layer + 3u});
                }
            }
            const auto mesh = AddSourced(f, Sourced(std::move(positions), std::move(faces), std::move(edges))).first;
            f.EnterEdit(mode);
            SelectOrdinals(f, mesh, std::array{0u}, mode);
            f.Do(action::mesh::Hide{});
            f.Click(vec3{}, false);
            const auto id = f.Output(mesh);
            expect(meshes.GetSelectedElements(id, mode).Count() == 1u);
            const auto hidden = meshes.GetHiddenElements(id, mode);
            meshes.GetSelectedElements(id, mode).ForEach([&](uint32_t h) { expect(!hidden.Contains(h)); });
            f.Do(action::mesh::Hide{});
            f.Click(vec3{}, false);
            expect(meshes.GetSelectedElements(f.Output(mesh), mode).Count() == 0u);
            f.Do(action::mesh::Reveal{});
            expect(meshes.GetSelectedElements(f.Output(mesh), mode).Count() == 2u);
        });
    }
}

// A hidden face stays hidden through a subdivision of its neighbors, undo and redo, and a duplicate.
void HiddenFaceSurvivesEdits(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = AddSourced(f, Sourced({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}}, {{0, 1, 4, 3}, {1, 2, 5, 4}})).first;
    f.EnterEdit(Element::Face);
    SelectOrdinals(f, mesh, std::array{0u}, Element::Face);
    f.Do(action::mesh::Hide{});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 1u});
    const auto check = [&](uint32_t id) {
        const Mesh current{meshes, id};
        const auto hidden = meshes.GetHiddenElements(id, Element::Face);
        expect(current.FaceCount() == 5u && hidden.Count() == 1u);
        for (const auto face : current.faces()) {
            float x = 0.f;
            for (const auto v : current.fv_range(face)) x += current.GetPosition(v).x;
            expect(hidden.Contains(*face) == (x / float(current.GetValence(face)) < 1.f));
        }
        expect(meshes.GetHiddenElements(id, Element::Vertex).Count() == 2u && meshes.GetHiddenElements(id, Element::Edge).Count() == 3u);
    };
    check(f.Output(mesh));
    f.Undo();
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 2u);
    f.Redo();
    check(f.Output(mesh));
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::Duplicate{});
    for (const auto [e, evaluated] : f.R.view<const EvaluatedMesh>().each()) check(evaluated.StoreId);
}

// A selection in a record's last block survives deleting that block and returns with undo.
void SparseSelection(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    std::vector<vec3> positions;
    for (uint32_t i = 0u; i < 258u; ++i) positions.push_back({float(i), 0, 0});
    const auto mesh = AddSourced(f, Sourced(std::move(positions))).first;
    f.EnterEdit(Element::Vertex);
    SelectOrdinals(f, mesh, std::array{256u, 257u}, Element::Vertex);
    const auto check = [&](bool deleted) {
        const auto id = f.Output(mesh);
        expect(Mesh{meshes, id}.VertexCount() == (deleted ? 256u : 258u));
        expect(meshes.GetSelectedElements(id, Element::Vertex).Count() == (deleted ? 0u : 2u));
    };
    check(false);
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::Vertices});
    check(true);
    f.Undo();
    check(false);
    f.Redo();
    check(true);
}

// Adds a three-vertex point cloud in each world, selects both with every vertex selected, and returns the mesh entities.
std::array<state::Entity, 2> AddPair(ProcedureFixture &f, const std::array<Transform, 2> &worlds) {
    static const std::array<std::vector<vec3>, 2> positions{{{{-2, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {{0, -1, 0}, {2, 1, 0}, {0, 0, 2}}}};
    std::array<state::Entity, 2> meshes, instances;
    for (uint32_t m = 0u; m < 2u; ++m) std::tie(meshes[m], instances[m]) = AddMesh(f.R, CreateMesh(f.R, Sourced(positions[m])).StoreId, MeshInstanceCreateInfo{.Transform = worlds[m]});
    f.Commit("Add pair");
    f.Do(action::selection::Select{instances[0]});
    f.Do(action::selection::ToggleSelected{instances[1]});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    return meshes;
}
void ExpectPair(ProcedureFixture &f, const std::array<state::Entity, 2> &meshes, const std::array<std::array<vec3, 3>, 2> &expected) {
    for (uint32_t m = 0u; m < 2u; ++m) {
        const Mesh mesh{f.Store(), f.Output(meshes[m])};
        for (uint32_t i = 0u; i < 3u; ++i) {
            const auto p = mesh.GetPosition(mesh.VertexAt(i));
            expect(Near(p, expected[m][i])) << "mesh" << m << "vertex" << i << p.x << p.y << p.z;
        }
    }
}
// Two objects, one rotated and unevenly scaled, edited together about their shared world median, against Blender 5.2.2 with center_override at that median.
void TransformedPairs(ProcedureFixture &f, Fixture &other) {
    Scenario(f, other, "RadialPair", [](ProcedureFixture &f) {
        const auto meshes = AddPair(f, {Transform{.P = {2.f, 0.f, 0.f}, .R = numeric::AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0.f, 0.f, 1.f}), .S = {2.f, 1.f, 1.f}}, Transform{.P = {-1.f, 1.f, 0.f}, .S = {1.f, 3.f, 1.f}}});
        f.Do(action::mesh::ToSphere{.6f});
        ExpectPair(f, meshes, {{{{{-1.602138877f, 0.254631042f, 0.063657761f}, {1.016452074f, -0.023930311f, -0.005982578f}, {-0.127709180f, 0.489163578f, -0.510836482f}}}, {{{0.078688025f, -0.965901852f, 0.015737623f}, {2.049850702f, 1.191094637f, -0.049850762f}, {0.283103585f, -0.047183931f, 1.716896415f}}}}});
        f.Undo();
        f.Do(action::mesh::PushPull{.4f});
        ExpectPair(f, meshes, {{{{{-1.666109681f, 0.213689774f, 0.053422447f}, {0.778049350f, 0.322837293f, 0.080709331f}, {0.069631100f, 1.278524160f, 0.278524280f}}}, {{{0.360994071f, -0.843569219f, 0.072198816f}, {1.902101994f, 0.624724150f, 0.097898051f}, {0.280898780f, -0.046816461f, 1.719101191f}}}}});
    });
    // Blender 5.2.2 shear, active object 0, angle .6, the Z and X axes, about the shared median, in global and in local space.
    Scenario(f, other, "ShearPair", [](ProcedureFixture &f) {
        const auto meshes = AddPair(f, {Transform{.P = {2, 0, 0}, .R = numeric::AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}), .S = {2, 1, -1.f}}, Transform{.P = {-1, 1, 0}, .S = {-1.f, 3, .5f}}});
        f.Do(action::mesh::Shear{.Angle = .6f});
        ExpectPair(f, meshes, {{{{{-2, -2.850570202f, 0}, {1, 1.254251003f, 0}, {0, .885977149f, 0}}}, {{{-1.482296467f, -1, 0}, {4.622524738f, 1, 0}, {.570114136f, 0, 2}}}}});
        f.Undo();
        f.Do(action::mesh::Shear{.Angle = .6f, .Local = true});
        ExpectPair(f, meshes, {{{{{-2.684136868f, 0, 0}, {.315863132f, 0, 0}, {-.342068434f, 1, 0}}}, {{{0, -.771954298f, 0}, {2, 1.684136748f, 0}, {0, .228045583f, 2}}}}});
    });
}

// A separation's outputs by their vertices' x coordinates, with the attributes, materials and hidden faces each output keeps.
struct SeparateCase {
    std::string_view Name;
    action::mesh::SeparateMode Operation;
    Element Domain;
    uint32_t Vertices;
    std::vector<std::vector<uint32_t>> Faces;
    std::vector<std::array<uint32_t, 2>> Edges;
    std::vector<uint32_t> Selected;
    std::vector<std::vector<uint32_t>> Expected;
    uint32_t OutputEdges;
};
void Separations(ProcedureFixture &f, Fixture &other) {
    using Mode = action::mesh::SeparateMode;
    static const std::array<SeparateCase, 5> cases{{
        // Extract both a face with an attached wire and an independent wire from a mixed source, with the first face hidden.
        {"SeparateLooseParts", Mode::LooseParts, Element::Vertex, 10u, {{0, 1, 2}, {3, 4, 5}}, {{5, 6}, {7, 8}}, {}, {{0, 1, 2}, {3, 4, 5, 6}, {7, 8}, {9}}, 8u},
        {"SeparateMaterial", Mode::Material, Element::Face, 7u, {{0, 1, 2}, {2, 1, 3}, {4, 5, 6}}, {}, {}, {{0, 1, 2, 4, 5, 6}, {1, 2, 3}}, 9u},
        {"SeparatePoints", Mode::Selected, Element::Vertex, 6u, {}, {}, {1, 4}, {{0, 2, 3, 5}, {1, 4}}, 0u},
        {"SeparateWires", Mode::Selected, Element::Vertex, 6u, {}, {{0, 1}, {1, 2}, {3, 4}, {4, 5}}, {1, 2}, {{0, 1, 3, 4, 5}, {1, 2}}, 4u},
        {"SeparateFaces", Mode::Selected, Element::Face, 4u, {{0, 1, 2}, {2, 1, 3}}, {}, {1}, {{0, 1, 2}, {1, 2, 3}}, 6u},
    }};
    for (const auto &c : cases) {
        Scenario(f, other, std::string{c.Name}, [&c](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto &a = meshes.Arenas();
            MeshSource source{.Weld = false};
            source.Data.Edges = c.Edges;
            for (const auto &face : c.Faces) source.Data.AddFace(face);
            auto &uvs = source.Attrs.TexCoords0.emplace();
            auto &colors = source.Attrs.Colors0.emplace();
            source.Attrs.Colors0ComponentCount = 4u;
            for (uint32_t i = 0u; i < c.Vertices; ++i) {
                const vec3 p{float(i), float(i % 3u == 1u), 0.f};
                source.Data.Positions.push_back(p);
                uvs.push_back({p.x, p.y});
                colors.push_back({p.x / 10.f, p.y, 0.f, 1.f});
            }
            const auto material = r.Resources.get<GpuBuffers>().Materials.Append(PBRMaterial{});
            source.Primitives.MaterialIndices = {material, 0u, material};
            source.Primitives.ElementPrimitiveIndices.assign(c.Faces.empty() ? c.Vertices : c.Faces.size(), 0u);
            if (c.Faces.size() >= 2u) source.Primitives.ElementPrimitiveIndices[1] = 1u;
            if (c.Faces.size() == 3u) source.Primitives.ElementPrimitiveIndices[2] = 2u;
            const auto mesh = AddSourced(f, std::move(source)).first;
            if (c.Operation == Mode::LooseParts) {
                f.EnterEdit(Element::Face);
                SelectOrdinals(f, mesh, std::array{0u}, Element::Face);
                f.Do(action::mesh::Hide{});
            }
            f.EnterEdit(c.Domain);
            if (!c.Selected.empty()) SelectOrdinals(f, mesh, c.Selected, c.Domain);
            f.Do(action::mesh::Separate{c.Operation});
            std::vector<std::vector<uint32_t>> actual;
            uint32_t edges = 0u, faces = 0u, hidden = 0u;
            for (const auto [e, evaluated] : r.view<const EvaluatedMesh>().each()) {
                const Mesh output{meshes, evaluated.StoreId};
                auto &vertices = actual.emplace_back();
                for (const auto v : output.vertices()) vertices.push_back(uint32_t(output.GetPosition(v).x));
                std::ranges::sort(vertices);
                const auto &record = meshes.Get(evaluated.StoreId);
                const auto palette = a.PrimitiveMaterials.Get(record.PrimitiveMaterials);
                for (const auto face : output.faces()) {
                    const bool second = std::ranges::any_of(output.fv_range(face), [&](auto v) { return output.GetPosition(v).x == 3.f; });
                    expect(palette[a.FacePrimitives.Get(record.FacePrimitives, *face)] == (second ? 0u : material));
                    for (const auto h : output.fh_range(face)) {
                        const auto p = output.GetPosition(output.GetToVertex(h));
                        expect(a.CornerUvs[0].Get(record.CornerUvs[0], *h) == vec2{p.x, p.y});
                        expect(a.CornerColors.Get(record.CornerColors, *h) == vec4{p.x / 10.f, p.y, 0.f, 1.f});
                    }
                }
                if (c.Faces.empty())
                    for (const auto v : output.vertices()) {
                        const auto p = output.GetPosition(v);
                        expect(palette[a.VertexPrimitives.Get(record.VertexPrimitives, *v)] == material);
                        expect(a.VertexColors.Get(record.VertexColors, *v) == vec4{p.x / 10.f, p.y, 0.f, 1.f});
                    }
                edges += output.EdgeCount();
                faces += output.FaceCount();
                hidden += meshes.GetHiddenElements(evaluated.StoreId, Element::Face).Count();
            }
            std::ranges::sort(actual);
            expect(actual == c.Expected);
            expect(edges == c.OutputEdges && faces == c.Faces.size()) << "edges" << edges << "faces" << faces;
            expect(hidden == uint32_t(c.Operation == Mode::LooseParts));
        });
    }
}
} // namespace

void TestOperators(ProcedureFixture &f, Fixture &other) {
    ViewportOperators(f, other);
    RecordedOperators(f, other);
    EditVisibility(f, other);
    HiddenPicking(f, other);
    TransformedPairs(f, other);
    Separations(f, other);
    Scenario(f, other, "HiddenFaceSurvivesEdits", HiddenFaceSurvivesEdits);
    Scenario(f, other, "SparseSelection", SparseSelection);
}
