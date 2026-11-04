#include "MeshProperties.h"
#include "ProcedureFixture.h"
#include "TransformMath.h"
#include "procedure/Macros.h"
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

Mesh::HH NextOutgoing(const Mesh &mesh, Mesh::HH h) {
    const auto opposite = mesh.GetConnectivity().Opposite(h);
    return opposite ? mesh.GetConnectivity().Next(opposite) : Mesh::HH{};
}
test::Circulation OutgoingHalfedges(const Mesh &mesh, Mesh::VH v) {
    const auto &c = mesh.GetConnectivity();
    return {&mesh, NextOutgoing, v && c.HasVertexBlock(*v) ? c.Outgoing(v) : Mesh::HH{}};
}

std::vector<vec3> FacingEdgeCenters(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = test::Halfedge(mesh, e);
        if (Facing(mesh, test::Face(mesh, h), eye) && Facing(mesh, test::Face(mesh, test::Opposite(mesh, h)), eye)) centers.push_back(test::EdgeCenter(mesh, e));
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
    // Runs the operator on the clone at the clicked handles and returns its selected elements.
    GeometrySelection (*Kernel)(ProcedureFixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};

// Each viewport edit appends one node cooking like the kernel at the clicked elements, and a drag commits as one history node.
void ViewportOperators(ProcedureFixture &f, Fixture &other) {
    using D = MeshStore::ElementDomain;
    static const std::array<ViewportRow, 4> rows{{
        {"Extrude", Element::Face, 2u, KindOf<procedure::Extrude>, {0.f}, [](ProcedureFixture &f, float, bool) { f.Do(action::mesh::Extrude{}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::ExtrudeRegion, .Selection = clicked}); }},
        {"BevelDrag", Element::Edge, 1u, KindOf<action::mesh::Bevel>, {.05f, .1f, .2f, .3f}, [](ProcedureFixture &f, float width, bool first) {
             if (first) f.Stage(action::mesh::Bevel{.Width = width});
             else f.Stage(UpdateOf<&action::mesh::Bevel::Width>(action::OnSelected{}, width)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float width) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::BevelEdges, .Param0 = width, .Steps = 1u, .Selection = clicked}); }},
        {"InsetDrag", Element::Face, 1u, KindOf<action::mesh::Inset>, {.1f, .25f}, [](ProcedureFixture &f, float thickness, bool first) {
             if (first) f.Stage(action::mesh::Inset{.Thickness = thickness});
             else f.Stage(UpdateOf<&action::mesh::Inset::Thickness>(action::OnSelected{}, thickness)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float thickness) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::InsetRegion, .Param0 = thickness, .Flags = TopologyFlagEvenOffset, .Selection = clicked}); }},
        {"TransformDrag", Element::Vertex, 1u, KindOf<TransformElements>, {.5f, 1.f, 1.5f}, [](ProcedureFixture &f, float z, bool) { f.Stage(action::view::TransformElements{{.P = {0.f, 0.f, z}}}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float z) {
             const PositionOperationTarget target{.StoreId = clone, .Selection = clicked};
             expect(ExecuteGeometryPositions(f.R, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {0.f, 0.f, z}}}}).has_value());
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
                const Mesh cube{meshes, f.Output(mesh)};
                const auto eye = f.View().CameraPosition;
                std::vector<vec3> centers;
                if (row.Domain == Element::Vertex) {
                    for (const auto v : cube.vertices())
                        if (std::ranges::all_of(OutgoingHalfedges(cube, v), [&](he::HH h) { return Facing(cube, test::Face(cube, h), eye); })) centers.push_back(test::Position(cube, v));
                } else if (row.Domain == Element::Edge) {
                    centers = FacingEdgeCenters(cube, eye);
                } else {
                    centers = FacingFaceCenters(cube, eye);
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
// The record an operator leaves per an independent reference: positions by ordinal, a position set, faces as source-index loops, or a property of the result.
struct Oracle {
    std::vector<vec3> Positions{}, Points{};
    std::vector<std::vector<uint32_t>> Faces{};
    bool (*Holds)(const Mesh &, std::span<const std::vector<uint32_t>> loops){};
    bool (*Relates)(const Mesh &, const Mesh &source){};
    // Both leave the record as it was: Unchanged cooks an identity and Faulted passes its input through.
    bool Unchanged{}, Faulted{};
};
// A row selects its clicked Targets, else its Select ordinals in Domain with an edge as its two vertex ordinals, else every element.
// Targets takes the eye in mesh-local coordinates and returns mesh-local positions.
// The gesture runs under a rigid frame that places World and moves the view camera, and it places its world-space parameters by that frame.
struct OperatorRow {
    std::string_view Name;
    std::variant<PrimitiveParameters, MeshSource (*)()> Shape;
    Element Domain;
    uint16_t Kind;
    void (*Gesture)(ProcedureFixture &, const Transform &frame);
    std::vector<vec3> (*Targets)(const Mesh &, vec3 eye){};
    Transform World{};
    void (*Prepare)(ProcedureFixture &){};
    std::vector<uint32_t> Select{};
    Oracle Expected{};
};
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
std::vector<vec3> AdjacentFaceCenters(const Mesh &mesh, vec3) {
    const auto first = *mesh.faces().begin();
    for (const auto h : test::FaceHalfedges(mesh, first)) {
        if (const auto opposite = test::Opposite(mesh, h); opposite && test::Face(mesh, opposite)) return {test::Centroid(mesh, first), test::Centroid(mesh, test::Face(mesh, opposite))};
    }
    return {};
}
std::vector<vec3> FacingFaceCenter(const Mesh &mesh, vec3 eye) { return FacingFaceCenters(mesh, eye) | std::views::take(1) | std::ranges::to<std::vector>(); }
std::vector<vec3> FacingEdgeCenter(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(1) | std::ranges::to<std::vector>(); }
std::vector<vec3> CornerEdgeCenters(const Mesh &mesh, vec3 eye) { return FacingEdgeCenters(mesh, eye) | std::views::take(2) | std::ranges::to<std::vector>(); }

std::vector<vec3> FacingSide(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye)) continue;
        auto corners = test::FaceVertices(mesh, face) | std::views::take(2) | std::ranges::to<std::vector>();
        std::ranges::sort(corners);
        return {test::Position(mesh, corners[0]), test::Position(mesh, corners[1])};
    }
    return {};
}
std::vector<vec3> FacingDiagonal(const Mesh &mesh, vec3 eye) {
    for (const auto face : mesh.faces()) {
        if (!Facing(mesh, face, eye) || test::Valence(mesh, face) != 4u) continue;
        const auto corners = test::FaceVertices(mesh, face) | std::ranges::to<std::vector>();
        return {test::Position(mesh, corners[0]), test::Position(mesh, corners[2])};
    }
    return {};
}
std::vector<vec3> BoundaryEdgeCenters(const Mesh &mesh, vec3) {
    std::vector<vec3> centers;
    for (const auto e : mesh.edges()) {
        const auto h = test::Halfedge(mesh, e), opposite = test::Opposite(mesh, h);
        if (bool(test::Face(mesh, h)) != (opposite && test::Face(mesh, opposite))) centers.push_back(test::EdgeCenter(mesh, e));
    }
    return centers;
}
std::vector<vec3> OppositeEdgeCenters(const Mesh &mesh, vec3) {
    const auto center = test::EdgeCenter(mesh, *mesh.edges().begin());
    return {center, -center};
}

void ShadedSmooth(ProcedureFixture &f) { f.Do(action::object::ShadeSelected{procedure::ShadeMode::Smooth}); }
template<float Amount> void Randomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Randomize{.Amount = Amount, .Seed = 3u});
}
void Wires(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::OnlyFaces});
}
const Transform &ActiveWorld(const ProcedureFixture &f) { return *WorldTransformOf(f.R, FindActiveEntity(f.R)); }
void TriangulatedRandomized(ProcedureFixture &f) {
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::SplitFaces{});
    f.Do(action::mesh::Randomize{.Amount = .3f, .Seed = 3u});
}
// Deletes the first edge between two faces looking toward the eye with its faces, which opens a boundary loop.
void Opened(ProcedureFixture &f) {
    f.Render();
    const auto mesh = f.R.get<const Instance>(FindActiveEntity(f.R)).Entity;
    const auto &world = ActiveWorld(f);
    f.Click(TransformPoint(world, FacingEdgeCenter(Mesh{f.Store(), f.Output(mesh)}, InverseTransformPoint(world, f.View().CameraPosition)).front()), false);
    f.Do(action::mesh::Delete{.Mode = action::mesh::DeleteMode::Edges});
}

bool Near(vec3 actual, vec3 expected) { return Length(actual - expected) <= 2e-4f * std::max(1.f, Length(expected)); }
// Each face as a loop of indices into the pre-operator positions `source`, with InvalidOffset for a new vertex.
std::vector<std::vector<uint32_t>> Loops(const Mesh &mesh, std::span<const vec3> source) {
    std::vector<std::vector<uint32_t>> loops;
    for (const auto face : mesh.faces()) {
        auto &loop = loops.emplace_back();
        for (const auto v : test::FaceVertices(mesh, face)) {
            const auto p = test::Position(mesh, v);
            const auto found = std::ranges::find_if(source, [&](vec3 q) { return Near(p, q); });
            loop.push_back(found == source.end() ? InvalidOffset : uint32_t(found - source.begin()));
        }
    }
    return loops;
}
std::vector<std::vector<uint32_t>> Canonical(std::vector<std::vector<uint32_t>> loops) {
    for (auto &loop : loops) std::ranges::rotate(loop, std::ranges::min_element(loop));
    std::ranges::sort(loops);
    return loops;
}
std::vector<uint32_t> EdgeOrdinals(const Mesh &mesh, std::span<const uint32_t> pairs) {
    std::vector<uint32_t> ordinals;
    for (size_t i = 0u; i + 1u < pairs.size(); i += 2u) {
        uint32_t ordinal = 0u;
        for (const auto e : mesh.edges()) {
            const auto [from, to] = test::EdgeVertices(mesh, e);
            if (std::minmax(mesh.VertexOrdinal(from), mesh.VertexOrdinal(to)) == std::minmax(pairs[i], pairs[i + 1u])) {
                ordinals.push_back(ordinal);
                break;
            }
            ++ordinal;
        }
    }
    expect(ordinals.size() == pairs.size() / 2u);
    return ordinals;
}
std::vector<uint32_t> Ordinals(uint32_t first, uint32_t last) {
    std::vector<uint32_t> ordinals(last - first + 1u);
    std::iota(ordinals.begin(), ordinals.end(), first);
    return ordinals;
}
MeshSource Sourced(std::vector<vec3> positions, std::vector<std::vector<uint32_t>> faces = {}, std::vector<std::array<uint32_t, 2>> edges = {}) {
    MeshSource source{.Weld = false};
    source.Data.Positions = std::move(positions);
    for (const auto &face : faces) source.Data.AddFace(face);
    source.Data.Edges = std::move(edges);
    return source;
}
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
MeshSource Rings() {
    std::vector<vec3> positions;
    std::vector<std::vector<uint32_t>> faces;
    for (const float radius : {1.f, 2.f, 4.f}) positions.insert(positions.end(), {{radius, radius, 0}, {-radius, radius, 0}, {-radius, -radius, 0}, {radius, -radius, 0}});
    for (uint32_t ring = 0u; ring < 2u; ++ring)
        for (uint32_t i = 0u; i < 4u; ++i) faces.push_back({ring * 4u + i, ring * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + i});
    return Sourced(std::move(positions), std::move(faces));
}
MeshSource TurnedRings() {
    auto source = Rings();
    for (auto &position : source.Data.Positions) position = {-position.x, -position.y, position.z};
    return source;
}
// The rings with the loop cut between the inner two slid halfway outward.
std::vector<vec3> RingsSlid() {
    auto points = Rings().Data.Positions;
    points.insert(points.end(), {{1.75f, 1.75f, 0}, {-1.75f, 1.75f, 0}, {-1.75f, -1.75f, 0}, {1.75f, -1.75f, 0}});
    return points;
}
// Three separate triangles along +Y, each with a slide vertex, and an isolated vertex.
MeshSource ThreeTriangles() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 4, 0}, {4, 4, 0}, {-1, 6, 0}, {0, 8, 0}, {1, 8, 0}, {0, 10, 0}, {0, 12, 0}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}); }
// Three triangles sharing the edge from the origin to (2, 0, 0), one of them opening the fan into a nonmanifold edge.
MeshSource Fan3() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}, {{0, 1, 2}, {1, 0, 3}, {0, 2, 4}}); }
MeshSource NonmanifoldFan() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 2, 1}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}, {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}, {5, 6, 7}, {7, 6, 5}}); }
MeshSource NonmanifoldEdge() { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}}, {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}}); }
MeshSource Square() { return Sourced({{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, {{0, 1, 2, 3}}); }
MeshSource UnevenChain() { return Sourced({{0, 0, 0}, {.25f, 1, 0}, {2, 0, 0}, {3, 2, 0}, {5, 0, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}); }
MeshSource KinkedChain() { return Sourced({{0, 0, 0}, {1, 2, 0}, {4, 0, 0}, {9, 9, 9}}, {}, {{0, 1}, {1, 2}}); }
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
// The positions a wireframe of the unit quad leaves for each combination of thickness, offset, even insets, boundary ring and replacement.
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
// Whether the loops hold the Windings faces past its cubes, the fans and the lone triangle, each wound as in the shape or reversed.
bool WindingsLooseFaces(std::span<const std::vector<uint32_t>> loops, bool reversed) {
    const auto canonical = Canonical({loops.begin(), loops.end()});
    const auto source = Windings();
    for (uint32_t f = 12u; f < source.Data.FaceCount(); ++f) {
        std::vector<uint32_t> loop(std::from_range, source.Data.Face(f));
        if (reversed) std::ranges::reverse(loop);
        if (!std::ranges::binary_search(canonical, Canonical({loop})[0])) return false;
    }
    return true;
}
// Looks straight down the world Z axis turned by the frame, which the bend and warp references take as the viewport plane.
void LookDown(ProcedureFixture &f, const Transform &frame) {
    f.R.patch<ViewCameraState>(FirstViewport(f.R), [&](auto &view) { view.Active.Orientation = frame.R; });
}
const Transform BendWorld{.P = {3, -1, 2}, .R = numeric::AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}), .S = {-2, .5f, 1.5f}};
const Transform SlideWorld{.P = {2, -1, 3}, .R = numeric::AngleAxis(.6f, Normalize(vec3{1, 2, 3})), .S = {3, -2, .5f}};
const Transform VertexSlideWorld{.P = {2, -1, 3}, .R = numeric::AngleAxis(.4f, vec3{0, 0, 1}) * numeric::AngleAxis(-.2f, vec3{0, 1, 0}) * numeric::AngleAxis(.3f, vec3{1, 0, 0}), .S = {-3, .5f, 2}};
// A chain of four vertices whose world positions bend onto a quarter circle of radius two.
MeshSource BendChain() {
    std::vector<vec3> positions;
    for (const auto p : {vec3{-1, 0, 0}, vec3{1, 0, 1}, vec3{2, 0, 2}, vec3{3, 0, 3}}) positions.push_back(InverseTransformPoint(BendWorld, p));
    return Sourced(std::move(positions), {}, {{0, 1}, {1, 2}, {2, 3}});
}

std::vector<vec3> Replaced(MeshSource source, std::initializer_list<uint32_t> indices, std::initializer_list<vec3> values) {
    auto positions = source.Data.Positions;
    auto value = values.begin();
    for (const auto i : indices) positions[i] = *value++;
    return positions;
}
// The triangle ordinals of the corner cells of a triangulated grid of `size` cells a side, `corner` cells a side.
std::vector<uint32_t> CornerCells(uint32_t size, uint32_t corner) {
    std::vector<uint32_t> faces;
    for (uint32_t y = 0u; y < corner; ++y)
        for (uint32_t x = 0u; x < corner; ++x) faces.append_range(std::array{2u * (y * size + x), 2u * (y * size + x) + 1u});
    return faces;
}
// The Ctrl+R gesture: clicks the edge at the mesh-local midpoint, appends an unslid Loop Cut and Slide, slides by each factor toward `positive`, and commits.
void LoopSlide(ProcedureFixture &f, vec3 edge, vec3 positive, std::initializer_list<float> factors) {
    const auto values = [&](float factor, uint32_t reference) {
        return std::vector{Words(1u), Words(factor), Words(positive), Words(reference)};
    };
    const auto hit = f.Pick(TransformPoint(ActiveWorld(f), edge), Element::Edge);
    expect(fatal(hit.has_value()));
    std::vector<action::Action> cut;
    cut.push_back(action::MakeAction(action::selection::ApplyEditElementClick{false, hit}));
    cut.push_back(action::MakeAction(action::nodes::AppendMacro{procedure::LoopCutAndSlideMacro, values(0.f, InvalidOffset)}));
    f.Stage(std::move(cut));
    const Mesh mesh{f.Store(), f.Output(f.R.get<const Instance>(FindActiveEntity(f.R)).Entity)};
    const auto vertices = mesh.VertexValues();
    uint32_t reference = InvalidOffset;
    f.Store().GetSelectedElements(mesh.GetStoreId(), Element::Vertex).ForEach([&](uint32_t vertex) {
        if (Length(vertices[vertex].Position - edge) < 1e-4f) reference = vertex;
    });
    expect(fatal(reference != InvalidOffset));
    for (const auto factor : factors) f.Stage(action::nodes::AppendMacro{procedure::LoopCutAndSlideMacro, values(factor, reference)});
    f.CommitGesture();
}
void Activate(ProcedureFixture &f, uint32_t ordinal) {
    const auto mesh = f.R.get<const Instance>(FindActiveEntity(f.R)).Entity;
    f.R.emplace_or_replace<MeshActiveElement>(mesh, *Mesh{f.Store(), f.Output(mesh)}.VertexAt(ordinal));
    f.Commit("Set active vertex");
}
// A cube of half extent 2 centered at 50 on every axis, split across x by a ring of vertices .002 beyond its center, within the plane side tolerance there.
MeshSource FarSplitCube() {
    std::vector<vec3> positions;
    for (const float x : {48.f, 50.002f, 52.f}) {
        for (const float y : {48.f, 52.f}) {
            for (const float z : {48.f, 52.f}) positions.emplace_back(x, y, z);
        }
    }
    const auto v = [](uint32_t i, uint32_t j, uint32_t k) { return 4u * i + 2u * j + k; };
    std::vector<std::vector<uint32_t>> faces{{v(0, 0, 0), v(0, 0, 1), v(0, 1, 1), v(0, 1, 0)}, {v(2, 0, 0), v(2, 1, 0), v(2, 1, 1), v(2, 0, 1)}};
    for (uint32_t i = 0u; i < 2u; ++i) {
        faces.push_back({v(i, 0, 0), v(i + 1, 0, 0), v(i + 1, 0, 1), v(i, 0, 1)});
        faces.push_back({v(i, 1, 0), v(i, 1, 1), v(i + 1, 1, 1), v(i + 1, 1, 0)});
        faces.push_back({v(i, 0, 0), v(i, 1, 0), v(i + 1, 1, 0), v(i + 1, 0, 0)});
        faces.push_back({v(i, 0, 1), v(i + 1, 0, 1), v(i + 1, 1, 1), v(i, 1, 1)});
    }
    return Sourced(std::move(positions), std::move(faces));
}
// Eight vertices with a symmetric partner within .2 of each one's mirror across x = 0, where the last competes with the first for the second's partner.
MeshSource Symmetric() { return Sourced({{-1, 0, 0}, {1.08f, .04f, .02f}, {-2, 1, .2f}, {2.02f, 1.05f, .25f}, {.02f, 2, .3f}, {4, 2, 3}, {.1f, 4, .6f}, {-1.02f, 0, 0}}, {{0, 1, 4}, {2, 3, 5}}); }
// Four wires: two separate edges to slide along, a zero-length edge and a fork.
MeshSource SlideWires() { return Sourced({{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {4, 3, 0}, {7, 0, 0}, {7, 0, 0}, {9, 0, 0}, {0, 9, 0}}, {}, {{0, 1}, {2, 3}, {4, 5}, {4, 6}}); }
// A fan of triangles over the convex polygon, then a triangle and a quad outside the selection on the fan's first edge.
MeshSource Fanned(std::vector<vec3> polygon) {
    const auto n = uint32_t(polygon.size());
    std::vector<std::vector<uint32_t>> faces;
    for (uint32_t i = 1u; i + 1u < n; ++i) faces.push_back({0u, i, i + 1u});
    polygon.insert(polygon.end(), {{-7, 0, 0}, {-7, -3, 0}});
    faces.push_back({1u, 0u, n});
    faces.push_back({0u, n - 1u, n, n + 1u});
    return Sourced(std::move(polygon), std::move(faces));
}
// Whether the triangulated cube keeps its element counts and every face pair passes the empty-circle test in its quad's plane.
bool BeautifiedCube(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    return mesh.VertexCount() == 8u && mesh.EdgeCount() == 18u && mesh.FaceCount() == 12u && test::Delaunay(mesh, HandleList(mesh.faces()));
}
// Whether the eight-sided fan's six triangles pass the empty-circle test and its two unselected faces are as Fanned built them.
bool BeautifiedFan(const Mesh &mesh, std::span<const std::vector<uint32_t>> loops) {
    std::vector<uint32_t> fan;
    for (uint32_t i = 0u; const auto face : mesh.faces()) {
        if (std::ranges::all_of(loops[i++], [](uint32_t v) { return v < 8u; })) fan.push_back(*face);
    }
    const auto canonical = Canonical({loops.begin(), loops.end()});
    return fan.size() == 6u && loops.size() == 8u && std::ranges::binary_search(canonical, std::vector<uint32_t>{0, 8, 1}) && std::ranges::binary_search(canonical, std::vector<uint32_t>{0, 7, 8, 9}) && test::Delaunay(mesh, fan);
}
// Whether the cube of half extent 2 bisected at z = .5 keeps its part above the plane as a closed box of six faces.
bool CappedCube(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    const auto volumes = test::ClosedComponentVolumes(mesh);
    return mesh.VertexCount() == 8u && mesh.FaceCount() == 6u && volumes.size() == 1u && std::abs(volumes[0] - 24.0) < 1e-4;
}
// Whether the two by two grid solidified by .25 is one closed outward shell of 16 quads enclosing its area times the thickness.
bool SolidifiedGrid(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    const auto volumes = test::ClosedComponentVolumes(mesh);
    return mesh.FaceCount() == 16u && volumes.size() == 1u && std::ranges::all_of(volumes, [](double v) { return v > 0.0; }) && std::abs(test::EnclosedVolume(mesh).value_or(0.0) - 1.0) < 1e-4;
}
// Whether the unit cube with its top face solidified by .25 keeps its six faces and adds the top's copy and four rim quads.
bool SolidifiedCubeFace(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    std::vector<std::vector<Mesh::VH>> shell;
    for (const auto face : mesh.faces()) {
        std::vector<Mesh::VH> loop;
        for (const auto v : test::FaceVertices(mesh, face)) loop.push_back(v);
        if (std::ranges::all_of(loop, [&](Mesh::VH v) { return test::Position(mesh, v).z > .5f; })) shell.push_back(std::move(loop));
    }
    std::multiset<std::pair<uint32_t, uint32_t>> edges;
    double volume = 0.0;
    for (const auto &loop : shell) {
        for (size_t k = 0; k < loop.size(); ++k) edges.emplace(*loop[k], *loop[(k + 1) % loop.size()]);
        const numeric::dvec3 a{test::Position(mesh, loop[0])};
        for (size_t k = 1; k + 1 < loop.size(); ++k) volume += Dot(a, Cross(numeric::dvec3{test::Position(mesh, loop[k])}, numeric::dvec3{test::Position(mesh, loop[k + 1])})) / 6.0;
    }
    const bool closed = std::ranges::all_of(edges, [&](const auto &e) { return edges.count(e) == 1u && edges.count({e.second, e.first}) == 1u; });
    uint32_t copies = 0u;
    for (const auto v : mesh.vertices()) copies += test::Position(mesh, v).z == .75f;
    return mesh.VertexCount() == 12u && mesh.FaceCount() == 11u && shell.size() == 6u && closed && copies == 4u && std::abs(volume - 1.0) < 1e-4 && test::ClosedComponentVolumes(mesh) == std::vector{8.0};
}
// Whether the cube of half extent 2 bisected through the corners on the plane x + y = 0 keeps the closed prism in front of it, its top and bottom split along their diagonals.
bool CappedCorners(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    const auto volumes = test::ClosedComponentVolumes(mesh);
    return mesh.VertexCount() == 6u && mesh.FaceCount() == 5u && volumes.size() == 1u && std::abs(volumes[0] - 32.0) < 1e-4;
}
// Whether the mesh is one closed outward surface, mirror-symmetric across the plane at Center on the axis, keeping the source's faces in front of that plane.
template<uint32_t Axis, bool Negative, float Center = 0.f> bool Symmetrized(const Mesh &mesh, std::span<const std::vector<uint32_t>> loops) {
    std::vector<vec3> points;
    for (const auto v : mesh.vertices()) points.push_back(test::Position(mesh, v));
    const bool mirrored = std::ranges::all_of(points, [&](vec3 p) {
        auto reflected = p;
        reflected[Axis] = 2.f * Center - p[Axis];
        return std::ranges::any_of(points, [&](vec3 q) { return Length(q - reflected) <= 1e-5f; });
    });
    bool kept = false;
    for (uint32_t i = 0u; const auto face : mesh.faces()) {
        const auto &loop = loops[i++];
        if (!std::ranges::all_of(test::FaceVertices(mesh, face), [&](auto v) { return (Negative ? -1.f : 1.f) * (test::Position(mesh, v)[Axis] - Center) > 1.f; })) continue;
        if (std::ranges::contains(loop, InvalidOffset)) return false;
        kept = true;
    }
    for (const auto face : mesh.faces()) {
        for (const auto h : test::FaceHalfedges(mesh, face)) {
            if (!test::Opposite(mesh, h) || !test::Face(mesh, test::Opposite(mesh, h))) return false;
        }
    }
    const auto volumes = test::ClosedComponentVolumes(mesh);
    return mirrored && kept && volumes.size() == 1u && std::ranges::all_of(volumes, [](double v) { return v > 0.0; });
}
// Whether the cube of half extent 2 sectioned at z = .5 leaves one square of its width at that height facing up.
bool SectionedCube(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    if (mesh.VertexCount() != 4u || mesh.FaceCount() != 1u) return false;
    const auto face = *mesh.faces().begin();
    std::vector<vec3> points;
    for (const auto v : test::FaceVertices(mesh, face)) points.push_back(test::Position(mesh, v));
    const auto area = test::AreaVector(mesh, face);
    return std::ranges::all_of(points, [](vec3 p) { return std::abs(p.z - .5f) < 1e-5f && std::abs(std::abs(p.x) - 2.f) < 1e-5f && std::abs(std::abs(p.y) - 2.f) < 1e-5f; }) && std::abs(area.z - 16.f) < 1e-4f;
}
bool ClosedOutward(const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
    const auto volumes = test::ClosedComponentVolumes(mesh);
    return test::EnclosedVolume(mesh).has_value() && volumes.size() == 1u && volumes[0] > 0.0;
}
state::Entity SetUpRow(ProcedureFixture &f, const OperatorRow &row, const Transform &frame) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    if (frame != Transform{}) {
        r.patch<ViewCameraState>(FirstViewport(r), [&](auto &view) {
            view.Active.Target = TransformPoint(frame, view.Active.Target);
            view.Active.Orientation = frame.R * view.Active.Orientation;
        });
    }
    const auto world = ComposeLocalTransforms(frame, row.World);
    if (const auto *shape = std::get_if<PrimitiveParameters>(&row.Shape)) {
        f.Do(action::object::AddMeshPrimitive{*shape, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = world})});
    } else {
        f.AddSourced(std::get<MeshSource (*)()>(row.Shape)(), world);
    }
    const auto mesh = r.get<const Instance>(FindActiveEntity(r)).Entity;
    f.EnterEdit(row.Domain);
    if (row.Prepare) row.Prepare(f);
    if (row.Targets) {
        f.Render();
        const auto placed = ActiveWorld(f);
        const auto targets = row.Targets(Mesh{meshes, f.Output(mesh)}, InverseTransformPoint(placed, f.View().CameraPosition));
        expect(fatal(!targets.empty()));
        for (uint32_t i = 0u; i < targets.size(); ++i) f.Click(TransformPoint(placed, targets[i]), i > 0u);
    } else if (!row.Select.empty()) {
        const auto selected = row.Domain == Element::Edge ? EdgeOrdinals(Mesh{meshes, f.Output(mesh)}, row.Select) : row.Select;
        SelectOrdinals(f, mesh, selected, row.Domain);
    } else {
        f.Do(action::selection::SelectAll{});
    }
    return mesh;
}
// The record the mesh draws, with each vertex's world position by ordinal, each face as a loop of vertex ordinals and each edge as its two vertex ordinals.
struct WorldRecord {
    std::vector<vec3> Positions;
    std::vector<std::vector<uint32_t>> Faces;
    std::vector<std::array<uint32_t, 2>> Edges;
};
WorldRecord WorldRecordOf(ProcedureFixture &f, state::Entity mesh) {
    const Mesh drawn{f.Store(), f.Output(mesh)};
    const auto world = *WorldTransformOf(f.R, f.Object(mesh));
    WorldRecord record;
    for (const auto v : drawn.vertices()) record.Positions.push_back(TransformPoint(world, test::Position(drawn, v)));
    for (const auto face : drawn.faces()) {
        auto &loop = record.Faces.emplace_back();
        for (const auto v : test::FaceVertices(drawn, face)) loop.push_back(drawn.VertexOrdinal(v));
    }
    for (const auto e : drawn.edges()) {
        const auto [from, to] = test::EdgeVertices(drawn, e);
        record.Edges.push_back({drawn.VertexOrdinal(from), drawn.VertexOrdinal(to)});
    }
    return record;
}
// A turn about a skew axis and a translation, which the frame loop places each row's object and the view camera by.
const Transform Rigid{.P = {1.5f, -2.f, .75f}, .R = numeric::AngleAxis(1.1f, Normalize(vec3{-2.f, 1.f, 3.f}))};
// Each gesture draws the same world-space record on its object at the row's World and at that placement moved by Rigid, with the view camera moved by Rigid.
void FrameEquivalence(ProcedureFixture &f, std::span<const OperatorRow> rows) {
    for (const auto &row : rows) {
        boost::ut::test(std::format("{}Frame", row.Name)) = [&f, &row] {
            std::vector<WorldRecord> records;
            for (const auto &frame : {Transform{}, Rigid}) {
                const TestDir dir{std::format("/tmp/mesheditor-scratch/{}Frame{}", row.Name, records.size()).c_str()};
                f.Begin(dir.Path);
                const auto mesh = SetUpRow(f, row, frame);
                row.Gesture(f, frame);
                records.push_back(WorldRecordOf(f, mesh));
            }
            const auto &placed = records[0], &moved = records[1];
            expect(moved.Faces == placed.Faces && moved.Edges == placed.Edges);
            expect(fatal(moved.Positions.size() == placed.Positions.size()));
            for (uint32_t i = 0u; i < placed.Positions.size(); ++i) {
                const auto p = moved.Positions[i], expected = TransformPoint(Rigid, placed.Positions[i]);
                expect(Near(p, expected)) << "vertex" << i << p.x << p.y << p.z << "expected" << expected.x << expected.y << expected.z;
            }
        };
    }
}
// The node each operator appends cooks the record its Oracle describes at the same selection, and the operator draws the same world-space record under a rigid frame.
// Operators that edit in their mesh's world frame place the object at a rotated and unevenly scaled transform.
void RecordedOperators(ProcedureFixture &f, Fixture &other) {
    using namespace action::mesh;
    static const Transform Placed{.P = {.5f, .25f, -.5f}, .R = numeric::AngleAxis(.4f, Normalize(vec3{1.f, 2.f, 3.f})), .S = {1.f, 1.5f, .75f}};
    static const std::vector<OperatorRow> rows{
        {"LimitedDissolveFaces", primitive::Circle{.Radius = 5.f, .Segments = 8u, .Fill = primitive::FillType::Fan}, Element::Face, KindOf<Dissolve>, [](ProcedureFixture &f, const Transform &) { f.Do(Dissolve{.Mode = DissolveMode::Limited}); }, AdjacentFaceCenters},
        {"RelaxEdgeLoopsEven", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<RelaxEdgeLoops>, [](ProcedureFixture &f, const Transform &) { f.Do(RelaxEdgeLoops{}); }, CornerEdgeCenters},
        {"FlattenView", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Flatten>, [](ProcedureFixture &f, const Transform &frame) { f.R.patch<ViewCameraState>(FirstViewport(f.R), [&](auto &v) { v.Active.Orientation = frame.R * ViewCamera::OrientationFromAway(Normalize(vec3{1.f, 2.f, 2.f})); }); f.Do(action::mesh::Flatten{.Method = FlattenMethod::View}); }, {}, Placed},
        {"EdgeSlideEvenActive", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::EdgeSlide>, [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .5f, .Direction = frame.R * vec3{0.f, 1.f, 0.f}, .Even = true}); }, FacingSide, {}, Randomized<.3f>},
        // The cut through the ring of the grid's bottom left edge adds a loop at x = .5, and sliding it by half toward +X moves each new vertex halfway along its rail to x = .75.
        {.Name = "LoopCutGrid", .Shape = +[] { return Grid(2u); }, .Domain = Element::Edge, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::LoopCutMacro, {{1u, 0u, 0u, 0u}}}); }, .Select = {0, 1}, .Expected = {.Points = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {.5f, 0, 0}, {.5f, 1, 0}, {.5f, 2, 0}}}},
        {.Name = "LoopCutAndSlideGrid", .Shape = +[] { return Grid(2u); }, .Domain = Element::Edge, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { LoopSlide(f, {.5f, 2, 0}, {-1, 0, 0}, {.25f, .5f}); }, .Expected = {.Points = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}, {.25f, 0, 0}, {.25f, 1, 0}, {.25f, 2, 0}}}},
        // The rings' rails reverse across the loop, and the slide follows the drag at the clicked edge on the near corner, away from the cut's first new vertex on the far corner.
        {.Name = "LoopCutAndSlideRings", .Shape = TurnedRings, .Domain = Element::Edge, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { LoopSlide(f, {1.5f, 1.5f, 0}, {1, 1, 0}, {.25f, .5f}); }, .Expected = {.Points = RingsSlid()}},
        {.Name = "BisectFill", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::BisectFillMacro, {{0u, 0u, std::bit_cast<uint32_t>(.5f), 0u}, {0u, 0u, std::bit_cast<uint32_t>(1.f), 0u}}}); }, .Expected = {.Holds = CappedCube}},
        {.Name = "BisectFillCorners", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::BisectFillMacro, {{0u, 0u, 0u, 0u}, {std::bit_cast<uint32_t>(1.f), std::bit_cast<uint32_t>(1.f), 0u, 0u}}}); }, .Expected = {.Holds = CappedCorners}},
        // Symmetrize mirrors the randomized cube's kept side, in front along the plane's normal, onto the cleared one.
        {.Name = "Symmetrize", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::SymmetrizeMacro}); }, .Prepare = Randomized<.3f>, .Expected = {.Holds = Symmetrized<0u, false>}},
        // Symmetrize through the ring off the far cube's center snaps the ring onto the plane, so the seam welds at any coordinate scale.
        {.Name = "SymmetrizeFar", .Shape = FarSplitCube, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::SymmetrizeMacro, {{std::bit_cast<uint32_t>(50.f), std::bit_cast<uint32_t>(50.f), std::bit_cast<uint32_t>(50.f), 0u}, {std::bit_cast<uint32_t>(1.f), 0u, 0u, 0u}}}); }, .Expected = {.Holds = Symmetrized<0u, false, 50.f>}},
        {.Name = "SymmetrizeNegativeY", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::SymmetrizeMacro, {{0u, 0u, 0u, 0u}, {0u, std::bit_cast<uint32_t>(-1.f), 0u, 0u}}}); }, .Prepare = Randomized<.3f>, .Expected = {.Holds = Symmetrized<1u, true>}},
        {.Name = "CrossSection", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<Subnet>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::nodes::AppendMacro{procedure::CrossSectionMacro, {{0u, 0u, std::bit_cast<uint32_t>(.5f), 0u}}}); }, .Expected = {.Holds = SectionedCube}},
        {"Fill", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<Fill>, [](ProcedureFixture &f, const Transform &) { f.Do(Fill{}); }, FacingDiagonal},
        {"BridgeEdgeLoops", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<BridgeEdgeLoops>, [](ProcedureFixture &f, const Transform &) { f.Do(BridgeEdgeLoops{}); }, OppositeEdgeCenters, {}, Wires},
        {"GridFill", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Edge, KindOf<GridFill>, [](ProcedureFixture &f, const Transform &) { f.Do(GridFill{}); }, BoundaryEdgeCenters, {}, Opened},
        {"EdgeRotate", primitive::IcoSphere{.Radius = 3.f, .Subdivisions = 1u}, Element::Edge, KindOf<EdgeRotate>, [](ProcedureFixture &f, const Transform &) { f.Do(EdgeRotate{}); }, FacingEdgeCenter},
        // Beautify leaves a randomized triangulated cube Delaunay in each quad's own plane with its element counts, and a selected planar fan Delaunay beside its unselected faces.
        {.Name = "BeautifyFaces", .Shape = primitive::Cube{{2.f, 2.f, 2.f}}, .Domain = Element::Face, .Kind = KindOf<BeautifyFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(BeautifyFaces{}); }, .Prepare = TriangulatedRandomized, .Expected = {.Holds = BeautifiedCube}},
        {.Name = "BeautifyArea", .Shape = +[] { return Fanned({{-5, 0, 0}, {-4, -3, 0}, {-1, -4, 0}, {3, -3, 0}, {6, 0, 0}, {5, 4, 0}, {2, 6, 0}, {-2, 5, 0}}); }, .Domain = Element::Face, .Kind = KindOf<BeautifyFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(BeautifyFaces{}); }, .Select = {0, 1, 2, 3, 4, 5}, .Expected = {.Holds = BeautifiedFan}},
        {"RotateUVs", AttributedGrid, Element::Face, KindOf<CycleCornerAttribute>, [](ProcedureFixture &f, const Transform &) { f.Do(CycleCornerAttribute{}); }, AdjacentFaceCenters},
        {"ReverseUVs", AttributedGrid, Element::Face, KindOf<CycleCornerAttribute>, [](ProcedureFixture &f, const Transform &) { f.Do(CycleCornerAttribute{.Cycle = CornerCycle::Reverse}); }, AdjacentFaceCenters},
        {"RotateColors", AttributedGrid, Element::Face, KindOf<CycleCornerAttribute>, [](ProcedureFixture &f, const Transform &) { f.Do(CycleCornerAttribute{.Attribute = CornerAttribute::Colors, .Cycle = CornerCycle::Backward}); }, AdjacentFaceCenters},
        {"MarkSharpEdges", primitive::Cube{{1.f, 2.f, 3.f}}, Element::Edge, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::SetSelectedSharp{.Element = Element::Edge, .Sharp = true}); }, CornerEdgeCenters},
        {"MarkSharpVertices", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Vertex, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::SetSelectedSharp{.Element = Element::Vertex, .Sharp = true}); }, FacingSide},
        {"SmoothFace", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::MarkSharp>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::SetSelectedSharp{.Element = Element::Face, .Sharp = false}); }, FacingFaceCenter},
        {"Knife", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Knife>, [](ProcedureFixture &f, const Transform &frame) {
             f.Render();
             f.Do(action::mesh::Knife{f.ScreenFraction(TransformPoint(frame, Placed.P + vec3{-4.f, .3f, 0.f})) * f.View().ViewportSize, f.ScreenFraction(TransformPoint(frame, Placed.P + vec3{4.f, -.2f, 0.f})) * f.View().ViewportSize}); }, {}, Placed},
        {"ShadeSmooth", primitive::Cube{{2.f, 2.f, 2.f}}, Element::Face, KindOf<procedure::Shade>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::ShadeSelected{procedure::ShadeMode::Smooth}); }},
        {"ShadeFlat", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::Shade>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::ShadeSelected{procedure::ShadeMode::Flat}); }, {}, {}, ShadedSmooth},
        {"ShadeSmoothByAngle", primitive::UVSphere{.Radius = 2.f, .Slices = 8u, .Stacks = 4u}, Element::Face, KindOf<procedure::Shade>, [](ProcedureFixture &f, const Transform &) { f.Do(action::object::ShadeSelected{procedure::ShadeMode::ByAngle, .6f}); }},
        // Fits against independent least-squares, SciPy spline and Blender references, each leaving the unselected far point.
        {.Name = "Circularize", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 3, 0}, {-1, 2, 0}, Far}, {{0, 1, 2, 3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<Circularize>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Circularize{}); }, .Expected = {.Positions = {{-.4543723752f, .2665205426f, 0}, {1.74700349f, -.2120457217f, 0}, {2.882409607f, 1.733702031f, 0}, {1.382753314f, 3.41480654f, 0}, {-.6794913643f, 2.508038512f, 0}, Far}}},
        {.Name = "CircularizeOpen", .Shape = +[] { return Sourced({{2, 0, 0}, {1.2f, 1, 0}, {0, 2, 0}, {-1, 1.1f, 0}, {-2, 0, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<Circularize>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Circularize{.Method = CircleFit::Contract, .Regular = false}); }, .Expected = {.Positions = {{1.08173014f, .2805637001f, 0}, {1.074599406f, .9587836917f, 0}, {.001674496185f, 1.730462079f, 0}, {-1, 1.1f, 0}, {-1.065226088f, .2831478019f, 0}, Far}}},
        {.Name = "CurveBetweenSelected", .Shape = +[] { return Sourced({{0, 0, 0}, {.5f, 1, 0}, {1, 0, .2f}, {2, 1, .5f}, {3, -1, 0}, {4, 0, 0}, {5, 1, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}); }, .Domain = Element::Vertex, .Kind = KindOf<CurveBetweenSelected>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(CurveBetweenSelected{}); }, .Select = {1, 3, 5}, .Expected = {.Positions = {{0, 0, 0}, {.5f, 1, 0}, {1.203125f, 1.09375f, .34375f}, {2, 1, .5f}, {2.953125f, .59375f, .34375f}, {4, 0, 0}, {5, 1, 0}, Far}}},
        {.Name = "Flatten", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {2, 2, 1}, {0, 2, 0}, {10, 0, 0}, {10, 2, 0}, {11, 2, 2}, {10, 0, 2}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::Flatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Flatten{}); }, .Expected = {.Positions = {{.06480586011f, .06480586011f, -.2449150189f}, {1.941974115f, -.05802588532f, .2192920636f}, {2.051245911f, 2.051245911f, .8063308916f}, {-.05802588532f, 1.941974115f, .2192920636f}, {9.755084981f, .06480586011f, .06480586011f}, {10.21929206f, 1.941974115f, -.05802588532f}, {10.80633089f, 2.051245911f, 2.051245911f}, {10.21929206f, -.05802588532f, 1.941974115f}, Far}}},
        {.Name = "RelaxEdgeLoops", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {3, 0, 0}, {3, 2, 0}, {0, 2, 0}, Far}, {}, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}); }, .Domain = Element::Edge, .Kind = KindOf<RelaxEdgeLoops>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(RelaxEdgeLoops{.EvenSpacing = false}); }, .Expected = {.Positions = {{.257891029f, .292481363f, 0}, {.982795715f, -.248387098f, 0}, {2.63177109f, .308072478f, 0}, {2.78709674f, 1.73870969f, 0}, {.611708343f, 1.09812188f, 0}, Far}}},
        {.Name = "SpaceEvenlyLinear", .Shape = UnevenChain, .Domain = Element::Edge, .Kind = KindOf<SpaceEvenly>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SpaceEvenly{.Interpolation = EdgeLoopInterpolation::Linear}); }, .Expected = {.Positions = {{0, 0, 0}, {1.11557984f, .50538290f, 0}, {2.45127296f, .90254611f, 0}, {3.56619334f, 1.43380666f, 0}, {5, 0, 0}, Far}}},
        {.Name = "SpaceEvenlyCubic", .Shape = UnevenChain, .Domain = Element::Edge, .Kind = KindOf<SpaceEvenly>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SpaceEvenly{.Interpolation = EdgeLoopInterpolation::Cubic}); }, .Expected = {.Positions = {{0, 0, 0}, {1.05305076f, .56604123f, 0}, {2.54712796f, .72494555f, 0}, {3.43722606f, 2.06138992f, 0}, {5, 0, 0}, Far}}},
        {.Name = "SmoothVertices", .Shape = KinkedChain, .Domain = Element::Vertex, .Kind = KindOf<SmoothVertices>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SmoothVertices{.Factor = .5f, .Repeat = 2u, .X = false}); }, .Select = {1, 3}, .Expected = {.Positions = {{0, 0, 0}, {1, .5f, 0}, {4, 0, 0}, {9, 9, 9}}}},
        {.Name = "SmoothZero", .Shape = KinkedChain, .Domain = Element::Vertex, .Kind = KindOf<SmoothVertices>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SmoothVertices{.Factor = 0.f}); }, .Select = {1, 3}, .Expected = {.Unchanged = true}},
        {.Name = "MakePlanarFaces", .Shape = +[] { return Sourced({{-1, -1, 1}, {1, -1, -1}, {1, 1, 1}, {-1, 1, -1}, {3, 0, 2}, {4, 0, 2}, {3, 1, 2}}, {{0, 1, 2, 3}, {4, 5, 6}}); }, .Domain = Element::Face, .Kind = KindOf<MakePlanarFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(MakePlanarFaces{}); }, .Expected = {.Positions = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}, {3, 0, 2}, {4, 0, 2}, {3, 1, 2}}}},
        // bpy.ops.transform.shrink_fatten(value=.2), Blender 5.2.2 d13f752e3b9c.
        // The unselected third face changes vertex normals only, and every user of a nonmanifold edge contributes to its normal and shell factor.
        {.Name = "ShrinkFattenFaces", .Shape = Fan3, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Select = {0, 1}, .Expected = {.Positions = {{0, .141421363f, .141421363f}, {2, .141421363f, .141421363f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenFacesEven", .Shape = Fan3, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Select = {0, 1}, .Expected = {.Positions = {{0, .2f, .2f}, {2, .2f, .2f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenVertices", .Shape = Fan3, .Domain = Element::Vertex, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Select = {0, 1, 2}, .Expected = {.Positions = {{.033676531f, .121543162f, .155219704f}, {2, .141421363f, .141421363f}, {.100692429f, 1, .172803462f}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenVerticesEven", .Shape = Fan3, .Domain = Element::Vertex, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Select = {0, 1, 2}, .Expected = {.Positions = {{.043392085f, .156607896f, .2f}, {2, .2f, .2f}, {.116539836f, 1, .2f}, {0, 0, 1}, {-1, 2, 1}, {2, 2, 2}}}},
        {.Name = "ShrinkFattenNonmanifold", .Shape = NonmanifoldFan, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f}); }, .Expected = {.Positions = {{0, .039776370f, .196004689f}, {2, .033695214f, .197141156f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 1.91055727f, 1.17888546f}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}}},
        {.Name = "ShrinkFattenNonmanifoldEven", .Shape = NonmanifoldFan, .Domain = Element::Face, .Kind = KindOf<ShrinkFatten>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ShrinkFatten{.Distance = .2f, .Even = true}); }, .Expected = {.Positions = {{0, .093217827f, .459346354f}, {2, .086327799f, .505079567f}, {0, 1, .2f}, {0, .2f, 1}, {-1, 1.91055727f, 1.17888546f}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}}}},
        // Radial edits about the selection center, whose mean radius the 252 central vertices dilute, and the far point unselected.
        {.Name = "ToSphere", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::ToSphere{.25f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Positions = Scaled(Cross(), 4u, [](float radius) { return (radius * .75f + (6.f / 257.f) * .25f) / radius; })}},
        {.Name = "PushPull", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::PushPull>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::PushPull{-.4f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Positions = Scaled(Cross(), 4u, [](float radius) { return (radius + .4f) / radius; })}},
        {.Name = "ToSphereZero", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::ToSphere{0.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        {.Name = "ToSphereNegative", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::ToSphere>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::ToSphere{-1.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        {.Name = "PushPullZero", .Shape = Cross, .Domain = Element::Vertex, .Kind = KindOf<procedure::PushPull>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::PushPull{0.f}); }, .Select = Ordinals(0u, 256u), .Expected = {.Unchanged = true}},
        // Blender 5.2.2 edge_slide, positive side up, over strips, forked rails, a pentagon, a quad turn, a transformed nonplanar fork and a closed loop.
        {.Name = "EdgeSlide", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.125f, .5f, 0}, {2, .75f, 0}, {4.25f, .25f, 0}})}},
        {.Name = "EdgeSlideBack", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = -.25f, .Direction = frame.R * vec3{0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.05f, -.25f, 0}, {2, -.5f, 0}, {4.125f, -.25f, 0}})}},
        {.Name = "EdgeSlideEvenFlipped", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{0, 1, 0}, .Even = true, .Flipped = true}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.22286583f, .89146334f, 0}, {2, -.11564839f, 0}, {4.85387611f, .85387623f, 0}})}},
        {.Name = "EdgeSlideUnclamped", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = -.25f, .Direction = frame.R * vec3{0, 1, 0}, .Clamp = false}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{.125f, -.5f, 0}, {2, -.75f, 0}, {3.75f, -.25f, 0}})}},
        {.Name = "EdgeSlideEvenZero", .Shape = SlideQuads, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = 0.f, .Direction = frame.R * vec3{0, 1, 0}, .Even = true}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideQuads(), {3, 4, 5}, {{-.1304878f, .5219512f, 0}, {2, 1.49251878f, 0}, {4.17364359f, -.34728706f, 0}})}},
        {.Name = "EdgeSlideTriangles", .Shape = +[] { return SlideTriangles(); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{0, 1, 0}}); }, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideTriangles(), {3, 4, 5}, {{.5f, .75f, 0}, {2.375f, .5f, 0}, {4.25f, .25f, 0}})}},
        {.Name = "EdgeSlidePentagon", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {1.5f, 3, 0}, {0, 2, 0}}, {{0, 1, 2, 3, 4}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{0, 1, 0}}); }, .Select = {0, 1, 1, 2}, .Expected = {.Positions = {{0, .5f, 0}, {1.72951460f, .65300971f, 0}, {2.625f, 1.5f, 0}, {1.5f, 3, 0}, {0, 2, 0}}}},
        {.Name = "EdgeSlideQuadTurn", .Shape = +[] { return Sourced({{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {0, 2, 0}}, {{0, 1, 2, 3}}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{0, 1, 0}}); }, .Select = {0, 1, 1, 2}, .Expected = {.Positions = {{0, .5f, 0}, {1.5f, .5f, 0}, {2.25f, 2, 0}, {0, 2, 0}}}},
        {.Name = "EdgeSlideTransformed", .Shape = +[] { return SlideTriangles({.4f, -.2f, .5f, 0, 0, 0, .3f, .6f, -.3f}); }, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * SlideWorld.R * vec3{0, -1, 0}}); }, .World = SlideWorld, .Select = {3, 4, 4, 5}, .Expected = {.Positions = Replaced(SlideTriangles({.4f, -.2f, .5f, 0, 0, 0, .3f, .6f, -.3f}), {3, 4, 5}, {{.5f, .75f, .15f}, {2.375f, .5f, .0375f}, {4.25f, .25f, -.075f}})}},
        {.Name = "EdgeSlideNonmanifold", .Shape = NonmanifoldEdge, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::EdgeSlide{}); }, .Select = {0, 1}, .Expected = {.Unchanged = true}},
        {.Name = "EdgeSlideLoop", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{1, 1, 0}}); }, .Select = {4, 5, 5, 6, 6, 7, 7, 4}, .Expected = {.Positions = Replaced(Rings(), {4, 5, 6, 7}, {{2.5f, 2.5f, 0}, {-2.5f, 2.5f, 0}, {-2.5f, -2.5f, 0}, {2.5f, -2.5f, 0}})}},
        {.Name = "EdgeSlideLoopEven", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::EdgeSlide{.Factor = .25f, .Direction = frame.R * vec3{1, 1, 0}, .Even = true}); }, .Select = {4, 5, 5, 6, 6, 7, 7, 4}, .Expected = {.Positions = Replaced(Rings(), {4, 5, 6, 7}, {{2.875f, 2.875f, 0}, {-2.875f, 2.875f, 0}, {-2.875f, -2.875f, 0}, {2.875f, -2.875f, 0}})}},
        {.Name = "EdgeSlideBranch", .Shape = Rings, .Domain = Element::Edge, .Kind = KindOf<procedure::EdgeSlide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::EdgeSlide{}); }, .Select = {4, 5, 4, 7, 4, 8}, .Expected = {.Unchanged = true}},
        // Blender 5.2.2 vert_slide with an explicit world direction and correct_uv=false, the first selected vertex its reference.
        {.Name = "VertexSlide", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = frame.R * vec3{1, 0, 0}}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{.5f, 0, 0}, {1, 4, 0}, {.25f, 8, 0}})}},
        {.Name = "VertexSlideEvenFlipped", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = frame.R * vec3{1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{1.5f, 0, 0}, {3.5f, 4, 0}, {.5f, 8, 0}})}},
        {.Name = "VertexSlideEvenZero", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Factor = 0.f, .Direction = frame.R * vec3{1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{2, 0, 0}, {4, 4, 0}, {1, 8, 0}})}},
        {.Name = "VertexSlideUnclamped", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Factor = -.5f, .Direction = frame.R * vec3{1, 0, 0}, .Even = true, .Clamp = false}); }, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{-1, 0, 0}, {-1, 4, 0}, {-1, 8, 0}})}},
        {.Name = "VertexSlideTransformed", .Shape = ThreeTriangles, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = frame.R * vec3{-.3f, 1, .2f}, .Even = true}); }, .World = VertexSlideWorld, .Select = {0, 3, 6, 9}, .Expected = {.Positions = Replaced(ThreeTriangles(), {0, 3, 6}, {{0, .5f, 0}, {-.223606795f, 4.447213650f, 0}, {0, 8.5f, 0}})}},
        // The active vertex is the even slide's reference, both endpoints of one edge meet at its middle, and a zero-length edge or a fork leaves its vertex.
        {.Name = "VertexSlideActive", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { Activate(f, 2u); f.Do(action::mesh::VertexSlide{.Factor = .25f, .Direction = frame.R * vec3{1, 0, 0}, .Even = true}); }, .Select = {0, 2, 7}, .Expected = {.Positions = Replaced(SlideWires(), {0, 2}, {{1, 0, 0}, {1, 3, 0}})}},
        {.Name = "VertexSlideBothEnds", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Direction = frame.R * vec3{1, 0, 0}}); }, .Select = {0, 1}, .Expected = {.Positions = Replaced(SlideWires(), {0, 1}, {{1, 0, 0}, {1, 0, 0}})}},
        {.Name = "VertexSlideDegenerate", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Direction = frame.R * vec3{-1, 0, 0}, .Flipped = true}); }, .Select = {4, 7}, .Expected = {.Positions = SlideWires().Data.Positions}},
        {.Name = "VertexSlideDegenerateEven", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { f.Do(action::mesh::VertexSlide{.Direction = frame.R * vec3{-1, 0, 0}, .Even = true, .Flipped = true}); }, .Select = {4, 7}, .Expected = {.Positions = SlideWires().Data.Positions}},
        {.Name = "VertexSlideZeroDirection", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::VertexSlide{.Direction = {}}); }, .Select = {0, 2, 7}, .Expected = {.Unchanged = true}},
        {.Name = "VertexSlideNan", .Shape = SlideWires, .Domain = Element::Vertex, .Kind = KindOf<procedure::VertexSlide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::VertexSlide{.Factor = std::numeric_limits<float>::quiet_NaN()}); }, .Select = {0, 2, 7}, .Expected = {.Unchanged = true}},
        // Blender 5.2.2 snap to symmetry toward positive partners, unselected ones included, where the first canonical pair owns its partner and the competing last vertex stays.
        {.Name = "SnapSymmetry", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SnapSymmetry{.Threshold = .2f, .Factor = .25f, .Negative = false}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Positions = {{-1.06f, .03f, .015f}, {1.06f, .03f, .015f}, {-2.015f, 1.0375f, .2375f}, {2.015f, 1.0375f, .2375f}, {0, 2, .3f}, {4, 2, 3}, {.1f, 4, .6f}, {-1.02f, 0, 0}}}},
        {.Name = "SnapSymmetryNegative", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SnapSymmetry{.Threshold = .2f, .Factor = 0.f, .Negative = true, .Center = false}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Positions = Replaced(Symmetric(), {1, 3}, {{1, 0, 0}, {2, 1, .2f}})}},
        {.Name = "SnapSymmetryZeroThreshold", .Shape = Symmetric, .Domain = Element::Vertex, .Kind = KindOf<SnapSymmetry>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SnapSymmetry{.Threshold = 0.f}); }, .Select = {0, 2, 4, 5, 6, 7}, .Expected = {.Unchanged = true}},
        // A bend onto a quarter circle that stays put near zero angle, and a warp continuing beyond its bounds, with the world XY plane as the viewport plane.
        {.Name = "Bend", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Bend{.Angle = std::numbers::pi_v<float> / 2.f, .Radius = 2.f, .Center = frame.P}); }, .World = BendWorld, .Expected = {.Positions = [] {
                                                                                                                                                                                                 std::vector<vec3> positions;
                                                                                                                                                                                                 const float root2 = std::sqrt(2.f);
                                                                                                                                                                                                 for (const auto p : {vec3{-1, 0, 0}, vec3{root2, root2 - 2.f, 1}, vec3{2, -2, 2}, vec3{2, -3, 3}}) positions.push_back(InverseTransformPoint(BendWorld, p));
                                                                                                                                                                                                 return positions;
                                                                                                                                                                                             }()}},
        {.Name = "BendSmallAngle", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Bend{.Angle = 1e-6f, .Radius = 2.f, .Center = frame.P, .Clamp = false}); }, .World = BendWorld, .Expected = {.Positions = BendChain().Data.Positions}},
        {.Name = "BendZeroRadius", .Shape = BendChain, .Domain = Element::Vertex, .Kind = KindOf<procedure::Bend>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Bend{.Radius = 0.f}); }, .World = BendWorld, .Expected = {.Unchanged = true}},
        {.Name = "Warp", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Warp{.Angle = std::numbers::pi_v<float>, .Center = frame.P, .AutoRange = false, .Min = 2.f, .Max = -2.f}); }, .Expected = {.Positions = [] {
                                                                                                                                                                           std::vector<vec3> positions;
                                                                                                                                                                           for (uint32_t i = 0u; i < 265u; ++i) positions.push_back(std::array{vec3{-1, -1, 0}, vec3{-1, 0, 0}, vec3{0, 1, 0}, vec3{1, 0, 0}, vec3{1, -1, 0}}[i % 5u] + vec3{0, 0, float(i / 5u)});
                                                                                                                                                                           return positions;
                                                                                                                                                                       }()}},
        {.Name = "WarpAuto", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Warp{.Angle = std::numbers::pi_v<float>, .Center = frame.P, .AutoRange = true, .Min = 2.f, .Max = -2.f}); }, .Expected = {.Positions = [] {
                                                                                                                                                                               std::vector<vec3> positions;
                                                                                                                                                                               const float root3 = std::sqrt(3.f) * .5f;
                                                                                                                                                                               for (uint32_t i = 0u; i < 265u; ++i) positions.push_back(std::array{vec3{-1, 0, 0}, vec3{-root3, .5f, 0}, vec3{0, 1, 0}, vec3{root3, .5f, 0}, vec3{1, 0, 0}}[i % 5u] + vec3{0, 0, float(i / 5u)});
                                                                                                                                                                               return positions;
                                                                                                                                                                           }()}},
        {.Name = "WarpFlat", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Warp{.Angle = 0.f, .Center = frame.P}); }, .Expected = {.Positions = [] {
                                                                                                                                                                               std::vector<vec3> positions;
                                                                                                                                                                               for (uint32_t i = 0u; i < 265u; ++i) positions.push_back({0, 1, float(i / 5u)});
                                                                                                                                                                               return positions;
                                                                                                                                                                           }()}},
        {.Name = "WarpEmptyRange", .Shape = Columns, .Domain = Element::Vertex, .Kind = KindOf<procedure::Warp>, .Gesture = [](ProcedureFixture &f, const Transform &frame) { LookDown(f, frame); f.Do(action::mesh::Warp{.AutoRange = false, .Min = 1.f, .Max = 1.f}); }, .Expected = {.Unchanged = true}},
        // bmesh.ops.unsubdivide and connect_verts_concave partitions, Blender 5.2.2 d13f752e3b9c, as oriented loops of source vertices.
        {.Name = "UnsubdivideOnce", .Shape = +[] { return Grid(3u); }, .Domain = Element::Vertex, .Kind = KindOf<Unsubdivide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Unsubdivide{1u}); }, .Expected = {.Faces = {{9, 4, 1, 6}, {0, 1, 4}, {14, 9, 6, 11}, {9, 12, 4}, {9, 14, 12}, {1, 3, 6}, {11, 15, 14}, {3, 11, 6}}}},
        {.Name = "UnsubdivideTwice", .Shape = +[] { return Grid(3u); }, .Domain = Element::Vertex, .Kind = KindOf<Unsubdivide>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Unsubdivide{2u}); }, .Expected = {.Faces = {{9, 14, 4}, {0, 1, 4}, {14, 9, 11}, {9, 4, 1}, {11, 9, 1, 3}, {11, 15, 14}}}},
        {.Name = "SplitConcaveFaces", .Shape = +[] { return Sourced({{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {3, 4, 0}, {3, 1, 0}, {1, 1, 0}, {1, 4, 0}, {0, 4, 0}, {10, 0, 0}, {13, 0, 0}, {13, 1, 0}, {11, 1, 0}, {11, 3, 0}, {10, 3, 0}}, {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13}}); }, .Domain = Element::Face, .Kind = KindOf<SplitFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SplitFaces{SplitTarget::Convex}); }, .Select = {0}, .Expected = {.Faces = {{0, 5, 6, 7}, {0, 1, 4, 5}, {1, 2, 3, 4}, {8, 9, 10, 11, 12, 13}}}},
        {.Name = "ConnectVertices", .Shape = +[] { return Sourced({{2, 0, 0}, {1, 2, 0}, {-1, 2, 0}, {-2, 0, 0}, {-1, -2, 0}, {1, -2, 0}}, {{0, 1, 2, 3, 4, 5}}); }, .Domain = Element::Vertex, .Kind = KindOf<ConnectVertices>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ConnectVertices{}); }, .Select = {0, 2, 4}, .Expected = {.Faces = {{0, 1, 2}, {2, 3, 4}, {0, 4, 5}, {0, 2, 4}}}},
        // A hexagon folded ninety degrees splits along its crease, and a nonplanar polygon has no valence limit.
        {.Name = "SplitNonplanarFaces", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 1}, {1, 2, 1}, {0, 2, 0}, {-1, 2, 1}, {-1, 0, 1}, {3, 0, 0}, {4, 0, 0}, {4, 1, 0}, {3, 1, 0}, {2, 0, 0}, {2, 2, 2}, {3, 2, 1}}, {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9}, {2, 1, 10, 11}, {8, 12, 9}}); }, .Domain = Element::Face, .Kind = KindOf<SplitFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SplitFaces{SplitTarget::Planar}); }, .Select = {0, 1, 3}, .Expected = {.Faces = {{0, 1, 2, 3}, {0, 3, 4, 5}, {6, 7, 8, 9}, {2, 1, 10, 11}, {8, 12, 9}}}},
        {.Name = "SplitNonplanarPolygon", .Shape = Ridge, .Domain = Element::Face, .Kind = KindOf<SplitFaces>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(SplitFaces{SplitTarget::Planar}); }, .Expected = {.Faces = {Ordinals(0u, 33u), [] { auto loop = Ordinals(33u, 65u); loop.push_back(0u); return loop; }()}}},
        // A closed or nonmanifold input passes through simplification, and a half-ratio triangle grid ends at 36 triangles, as Blender 5.2.2 ends at 35 or 36.
        {.Name = "SimplifyClosed", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 1, 2}, {1, 0, 3}, {1, 3, 2}, {0, 2, 3}}); }, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Simplify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Simplify{.Ratio = 0.f}); }, .Expected = {.Faulted = true}},
        {.Name = "SimplifyNonmanifold", .Shape = NonmanifoldEdge, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Simplify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Simplify{.Ratio = 0.f}); }, .Expected = {.Faulted = true}},
        {.Name = "SimplifyGrid", .Shape = +[] { return Grid(6u, true); }, .Domain = Element::Vertex, .Kind = KindOf<action::mesh::Simplify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Simplify{.Ratio = .5f}); }, .Expected = {.Relates = [](const Mesh &mesh, const Mesh &source) { return test::SimplifiedWithin(mesh, source, 36u, 1e-5f); }}},
        // A corner of a grid of many clusters retires only some of them, so the drawn record repairs once from what every round changed.
        {.Name = "SimplifyGridCorner", .Shape = +[] { return Grid(16u, true); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Simplify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Simplify{.Ratio = .25f}); }, .Select = CornerCells(16u, 4u), .Expected = {.Relates = [](const Mesh &mesh, const Mesh &source) { return test::SimplifiedWithin(mesh, source, 512u - 32u + 8u, 1e-5f); }}},
        // A wireframe of the unit quad takes the analytic wire positions for each option, and a cube's wireframe closes.
        {.Name = "WireframeQuad", .Shape = Square, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Wireframe{.Thickness = .2f, .Offset = .5f}); }, .Expected = {.Points = WirePoints(.2f, .5f, true, true, true)}},
        {.Name = "WireframeQuadKept", .Shape = Square, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Wireframe{.Thickness = .2f, .Offset = -1.f, .Replace = false, .Boundary = false, .Even = false}); }, .Expected = {.Points = WirePoints(.2f, -1.f, false, false, false)}},
        {.Name = "WireframeQuadRelative", .Shape = Square, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Wireframe{.Thickness = .1f, .Even = false, .Relative = true}); }, .Expected = {.Points = WirePoints(.2f, 0.f, false, true, true)}},
        {.Name = "SolidifyGrid", .Shape = +[] { return Grid(2u); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Solidify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Solidify{.Thickness = .25f}); }, .Expected = {.Holds = SolidifiedGrid}},
        // The quad tilted to the unit normal (0, -.6, .8) has its copy .25 behind it.
        {.Name = "SolidifyTiltedQuad", .Shape = +[] { return Sourced({{-1, -1, 0}, {1, -1, 0}, {1, 1, 1.5f}, {-1, 1, 1.5f}}, {{0, 1, 2, 3}}); }, .Domain = Element::Face, .Kind = KindOf<action::mesh::Solidify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Solidify{.Thickness = .25f}); }, .Expected = {.Points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 1.5f}, {-1, 1, 1.5f}, {-1, -.85f, -.2f}, {1, -.85f, -.2f}, {1, 1.15f, 1.3f}, {-1, 1.15f, 1.3f}}}},
        {.Name = "SolidifyCubeFace", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = KindOf<action::mesh::Solidify>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Solidify{.Thickness = .25f}); }, .Select = {1}, .Expected = {.Holds = SolidifiedCubeFace}},
        {.Name = "WireframeCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = KindOf<action::mesh::Wireframe>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Wireframe{.Thickness = .2f}); }, .Expected = {.Holds = ClosedOutward}},
        // A revolve of an open grid and of a closed tetrahedron, whose UVs and colors turn with each step.
        {.Name = "RevolveGrid", .Shape = +[] { return Attributed(Grid(2u)); }, .Domain = Element::Face, .Kind = KindOf<procedure::Extrude>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Revolve{.Steps = 3u, .Angle = std::numbers::pi_v<float> * 1.5f, .Center = {.5f, .5f, 0.f}, .Offset = 6.f}); }},
        {.Name = "RevolveTetrahedron", .Shape = +[] { return Attributed(Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}})); }, .Domain = Element::Face, .Kind = KindOf<procedure::Extrude>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Revolve{.Steps = 3u, .Angle = std::numbers::pi_v<float> * 1.5f, .Center = {.5f, .5f, 0.f}, .Offset = 6.f}); }},
        // Recalculate Normals turns both cubes outward and leaves the fans and the lone triangle as wound, and Flip Normals after it turns every face inside out.
        {.Name = "RecalculateNormals", .Shape = Windings, .Domain = Element::Face, .Kind = KindOf<RecalculateNormals>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(RecalculateNormals{}); }, .Expected = {.Holds = [](const Mesh &mesh, std::span<const std::vector<uint32_t>> loops) {
                                                                                                                                                                                                                  const auto volumes = test::ClosedComponentVolumes(mesh);
                                                                                                                                                                                                                  return volumes.size() == 2u && std::ranges::all_of(volumes, [](double v) { return v > 0.0; }) && WindingsLooseFaces(loops, false);
                                                                                                                                                                                                              }}},
        {.Name = "RecalculateNormalsFlipped", .Shape = Windings, .Domain = Element::Face, .Kind = KindOf<FlipNormals>, .Gesture = [](ProcedureFixture &f, const Transform &) {
             f.Do(RecalculateNormals{});
             f.Do(FlipNormals{});
         },
         .Expected = {.Holds = [](const Mesh &mesh, std::span<const std::vector<uint32_t>> loops) {
             const auto volumes = test::ClosedComponentVolumes(mesh);
             return volumes.size() == 2u && std::ranges::all_of(volumes, [](double volume) { return volume < 0.0; }) && WindingsLooseFaces(loops, true);
         }}},
        {.Name = "PointHull", .Shape = +[] { return Sourced({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}); }, .Domain = Element::Vertex, .Kind = KindOf<ConvexHull>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(ConvexHull{}); }, .Expected = {.Holds = [](const Mesh &mesh, std::span<const std::vector<uint32_t>>) {
                                                                                                                                                                                                                                                        const auto volumes = test::ClosedComponentVolumes(mesh);
                                                                                                                                                                                                                                                        return mesh.VertexCount() == 4u && mesh.FaceCount() == 4u && volumes.size() == 1u && std::abs(volumes[0] - 1.0 / 6.0) < 1e-6;
                                                                                                                                                                                                                                                    }}},
        // A shear by NaN cooks an identity, while a shear along one axis, a separation that leaves the cube whole and face layer edits on absent layers append no node.
        {.Name = "ShearSameAxis", .Shape = primitive::Cube{}, .Domain = Element::Vertex, .Kind = NoKind, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Shear{.6f, ShearAxis::X, ShearAxis::X}); }},
        {.Name = "ShearNan", .Shape = primitive::Cube{}, .Domain = Element::Vertex, .Kind = KindOf<procedure::Shear>, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(action::mesh::Shear{std::numeric_limits<float>::quiet_NaN()}); }, .Expected = {.Unchanged = true}},
        {.Name = "SeparateLooseCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Separate{SeparateMode::LooseParts}); }},
        {.Name = "SeparateMaterialCube", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f, const Transform &) { f.Do(Separate{SeparateMode::Material}); }},
        {.Name = "AbsentFaceLayers", .Shape = primitive::Cube{}, .Domain = Element::Face, .Kind = NoKind, .Gesture = [](ProcedureFixture &f, const Transform &) {
             f.Do(CycleCornerAttribute{.Attribute = CornerAttribute::Colors});
             f.Do(CycleCornerAttribute{.Attribute = CornerAttribute::Colors, .Cycle = CornerCycle::Reverse});
             f.Do(CycleCornerAttribute{.UVSet = 3u});
             f.Do(CycleCornerAttribute{.Cycle = CornerCycle::Reverse, .UVSet = 3u});
         }},
    };
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
    // A boolean with another object appends a Boolean node in the active object's space fed by an Object Info node, which cuts a closed part from the cube.
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
        const auto cut = f.Shaped(mesh);
        const auto volume = OutputVolume(f, mesh);
        expect(volume > 0.0 && volume < 8.0 - 1e-3) << volume;
        for (const auto moved : {tool, active}) {
            const auto placed = r.get<const Transform>(moved).P;
            f.Do(UpdateOf<&Transform::P>(moved, placed + vec3{10.f, 0.f, 0.f}));
            expect(std::abs(OutputVolume(f, mesh) - 8.0) < 1e-4);
            f.Do(UpdateOf<&Transform::P>(moved, placed));
            expect(f.Shaped(mesh) == cut);
        }
        // A generated Mesh 2 is in this procedure's mesh space, so it stays in place relative to Mesh 1 wherever Space moves: a bar through the cube cuts a 1 x 1 x 2 tunnel.
        const auto bar = f.Append(mesh, KindOf<primitive::Cube>);
        f.Do(UpdateOf<&primitive::Cube::HalfExtents>(bar, vec3{.5f, .5f, 2.f}));
        f.Do(Link{mesh, bar, f.Out(bar), boolean, 1u});
        expect(!r.any_of<NodeError>(boolean) && std::abs(OutputVolume(f, mesh) - 6.0) < 1e-4);
        const auto cooks = Cooks(r);
        const auto placed = r.get<const Transform>(active).P;
        f.Do(UpdateOf<&Transform::P>(active, placed + vec3{10.f, 0.f, 0.f}));
        expect(Cooks(r) == cooks && std::abs(OutputVolume(f, mesh) - 6.0) < 1e-4);
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
        const auto cut = f.Shaped(mesh);
        const auto cooks = Cooks(r);
        f.Do(UpdateOf<&Transform::P>(parent, vec3{3.f, -2.f, 1.f}));
        f.Do(UpdateOf<&Transform::R>(parent, numeric::AngleAxis(.7f, Normalize(vec3{1.f, 0.f, 1.f}))));
        f.Do(UpdateOf<&Transform::S>(parent, vec3{1.5f, 1.5f, 1.5f}));
        expect(Cooks(r) == cooks && f.Shaped(mesh) == cut);
        f.Do(UpdateOf<&Transform::P>(tool, r.get<const Transform>(tool).P + vec3{10.f, 0.f, 0.f}));
        expect(Cooks(r) == cooks + 1u && std::abs(OutputVolume(f, mesh) - 8.0) < 1e-4);
    });
    // A pose on an object placing an operand re-cooks the Boolean, and the next pass, which only reads the posed world transform the first published, does not cook again.
    Scenario(f, other, "BooleanPose", [&](ProcedureFixture &f) {
        auto &r = f.R;
        const auto [active, mesh, tool] = add_cubes(f);
        f.Do(action::mesh::Boolean{BooleanOperation::Difference});
        const auto cut = f.Shaped(mesh);
        const auto cooks = Cooks(r);
        auto posed = r.get<const Transform>(tool);
        posed.P += vec3{10.f, 0.f, 0.f};
        r.emplace<PosedLocal>(tool, posed);
        f.Commit("Pose tool");
        expect(Cooks(r) == cooks + 1u && std::abs(OutputVolume(f, mesh) - 8.0) < 1e-4);
        f.P->Settle();
        expect(Cooks(r) == cooks + 1u);
        r.remove<PosedLocal>(tool);
        f.Commit("Release pose");
        expect(Cooks(r) == cooks + 2u && f.Shaped(mesh) == cut);
    });
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto mesh = SetUpRow(f, row, {});
            std::vector<vec3> source;
            for (const Mesh before{meshes, f.Output(mesh)}; const auto v : before.vertices()) source.push_back(test::Position(before, v));
            const auto shape = f.Shaped(mesh);
            const auto node_count = r.get<const CookPlan>(mesh).Nodes.size();
            const auto displayed = r.get<const GeometryProcedure>(mesh).DisplayNode;
            const auto present = f.P->History.Present.value();
            row.Gesture(f, {});
            const auto &procedure = r.get<const CookPlan>(mesh);
            if (row.Kind == NoKind) {
                expect(procedure.Nodes.size() == node_count && f.P->History.Present == present);
                expect(f.Shaped(mesh) == shape);
                return;
            }
            const auto node = r.get<const GeometryProcedure>(mesh).DisplayNode;
            expect(fatal(procedure.Nodes.size() > node_count && node != displayed));
            const auto kind = r.get<const ProcedureNode>(node).Kind;
            const auto &expected = row.Expected;
            expect(kind == row.Kind && r.any_of<NodeError>(node) == expected.Faulted);
            if (expected.Unchanged || expected.Faulted) {
                expect(f.Shaped(mesh) == shape);
                return;
            }
            const Mesh result{meshes, f.Output(mesh)};
            if (!expected.Positions.empty()) {
                expect(result.VertexCount() == expected.Positions.size());
                for (uint32_t i = 0u; i < std::min<size_t>(result.VertexCount(), expected.Positions.size()); ++i) {
                    const auto p = test::Position(result, result.VertexAt(i));
                    expect(Near(p, expected.Positions[i])) << "vertex" << i << p.x << p.y << p.z;
                }
            }
            if (!expected.Points.empty()) {
                std::vector<vec3> sorted;
                for (const auto v : result.vertices()) sorted.push_back(test::Position(result, v));
                auto points = expected.Points;
                const auto corner = [](vec3 p) { return Corner{p.x, p.y, p.z}; };
                std::ranges::sort(sorted, {}, corner);
                std::ranges::sort(points, {}, corner);
                expect(sorted.size() == points.size());
                for (uint32_t i = 0u; i < std::min(sorted.size(), points.size()); ++i) expect(Near(sorted[i], points[i])) << "point" << i << sorted[i].x << sorted[i].y << sorted[i].z;
            }
            if (!expected.Faces.empty()) expect(Canonical(Loops(result, source)) == Canonical(expected.Faces));
            if (expected.Holds) expect(expected.Holds(result, Loops(result, source)));
            if (expected.Relates) expect(f.WithSource(mesh, [&](uint32_t id) { return expected.Relates(Mesh{meshes, f.Output(mesh)}, Mesh{meshes, id}); }));
        });
    }
    FrameEquivalence(f, rows);
}

std::string_view DomainName(Element domain) { return domain == Element::Vertex ? "Vertices" : domain == Element::Edge ? "Edges" :
                                                                                                                        "Faces"; }

MeshSource TwoQuads() { return Sourced({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}}, {{0, 1, 4, 3}, {1, 2, 5, 4}}); }
// Blender 5.2.2 mesh.hide over the two quads, hiding the selection or the rest with vertex 1, face 0 or the edge from 1 to 4 selected.
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
            const auto mesh = f.AddSourced(TwoQuads()).first;
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
                    const auto [from, to] = test::EdgeVertices(current, he::EH{e});
                    std::array pair{current.VertexOrdinal(from), current.VertexOrdinal(to)};
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
            const auto mesh = f.AddSourced(Sourced(std::move(positions), std::move(faces), std::move(edges))).first;
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
    const auto mesh = f.AddSourced(Sourced({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}}, {{0, 1, 4, 3}, {1, 2, 5, 4}})).first;
    f.EnterEdit(Element::Face);
    SelectOrdinals(f, mesh, std::array{0u}, Element::Face);
    f.Do(action::mesh::Hide{});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 1u});
    const auto check = [&](uint32_t id) {
        const Mesh current{meshes, id};
        const auto hidden = meshes.GetHiddenElements(id, Element::Face);
        expect(current.FaceCount() == 5u && hidden.Count() == 1u);
        for (const auto face : current.faces()) expect(hidden.Contains(*face) == (test::Centroid(current, face).x < 1.f));
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
    const auto mesh = f.AddSourced(Sourced(std::move(positions))).first;
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
            const auto p = test::Position(mesh, mesh.VertexAt(i));
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
        // Extracts both a face with an attached wire and an independent wire from a mixed source, with the first face hidden.
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
            const auto mesh = f.AddSourced(std::move(source)).first;
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
                for (const auto v : output.vertices()) vertices.push_back(uint32_t(test::Position(output, v).x));
                std::ranges::sort(vertices);
                const auto &record = meshes.Get(evaluated.StoreId);
                const auto palette = a.PrimitiveMaterials.Get(record.PrimitiveMaterials);
                for (const auto face : output.faces()) {
                    const bool second = std::ranges::any_of(test::FaceVertices(output, face), [&](auto v) { return test::Position(output, v).x == 3.f; });
                    expect(palette[a.FacePrimitives.Get(record.FacePrimitives, *face)] == (second ? 0u : material));
                    for (const auto h : test::FaceHalfedges(output, face)) {
                        const auto p = test::Position(output, test::ToVertex(output, h));
                        expect(a.CornerUvs[0].Get(record.CornerUvs[0], *h) == vec2{p.x, p.y});
                        expect(a.CornerColors.Get(record.CornerColors, *h) == vec4{p.x / 10.f, p.y, 0.f, 1.f});
                    }
                }
                if (c.Faces.empty())
                    for (const auto v : output.vertices()) {
                        const auto p = test::Position(output, v);
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
