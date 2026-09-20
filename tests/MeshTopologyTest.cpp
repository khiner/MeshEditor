// Runs the edit-mode topology operators through the engine and checks the output meshes' invariants and history round trips.
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "project/Project.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MeshBuffers.h"
#include "scene/Entity.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportInteractionState.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using boost::ut::expect;

namespace {
// An engine over a fresh project directory named for the case, at a 64 by 64 view.
struct Fixture : Engine {
    TestDir Dir;
    explicit Fixture(const char *name) : Engine{false}, Dir{(std::string{"mesheditor-topology-"} + name).c_str()} {
        expect(P->Begin(Dir));
        R.Context.get<ViewportExtent>().Value = {64, 64};
        P->Settle();
    }
    // Adds one `shape` and enters edit mode over `element`.
    Fixture(const char *name, auto shape, Element element) : Fixture{name} {
        Do(action::object::AddMeshPrimitive{shape, std::make_unique<MeshInstanceCreateInfo>()});
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
    }
    // Every case ends by rendering its last output and auditing the project.
    ~Fixture() {
        Render();
        Audit();
    }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::printf("audit: %s\n", why.c_str());
        expect(valid);
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    template<typename A> int Do(A a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        return node;
    }
    void Render() {
        SubmitViewport(R, Viewport);
        WaitForRender(R);
    }
    Mesh ActiveMesh() const { return GetMesh(R, GetActiveMeshEntity(R)); }
};

struct Counts {
    uint32_t Vertices, Edges, Faces, Triangles;
    bool operator==(const Counts &) const = default;
};

Counts CountsOf(const Mesh &mesh) { return {mesh.VertexCount(), mesh.EdgeCount(), mesh.FaceCount(), mesh.TriangleIndexCount() / 3}; }

void ExpectCounts(const Mesh &mesh, Counts expected) {
    const auto actual = CountsOf(mesh);
    if (actual != expected) std::printf("counts %u %u %u %u against %u %u %u %u\n", actual.Vertices, actual.Edges, actual.Faces, actual.Triangles, expected.Vertices, expected.Edges, expected.Faces, expected.Triangles);
    expect(actual == expected);
}

// Every corner names a live vertex, every opposite pair is reciprocal, every face loop closes, and the tables agree on the triangle count.
void CheckInvariants(const MeshStore &meshes, const Mesh &mesh) {
    const auto &c = mesh.GetConnectivity();
    const auto corners = mesh.CornerVertices();
    for (uint32_t h = 0; h < corners.size(); ++h) {
        if (corners[h] < mesh.VertexCount()) continue;
        std::printf("corner %u of face %u names vertex %u of %u\n", h, *mesh.GetFace(he::HH{h}), corners[h], mesh.VertexCount());
        expect(false);
    }
    for (uint32_t h = 0; h < c.Opposites.size(); ++h) {
        if (const auto opposite = c.Opposites[h]) {
            expect(*c.Opposites[*opposite] == h);
            expect(*mesh.GetFromVertex(he::HH{h}) == *mesh.GetToVertex(opposite) && *mesh.GetToVertex(he::HH{h}) == *mesh.GetFromVertex(opposite));
        }
    }
    uint32_t triangles = 0, halfedges = 0;
    for (const auto fh : mesh.faces()) {
        const auto valence = mesh.GetValence(fh);
        expect(valence >= 3u);
        triangles += valence - 2;
        halfedges += valence;
    }
    expect(halfedges == c.Opposites.size());
    expect(triangles == mesh.TriangleIndexCount() / 3);
    const auto &record = meshes.Get(mesh.GetStoreId());
    expect(record.TriangleCount == triangles);
    const auto first_triangles = meshes.Arenas().FaceFirstTriangles.Get(record.FaceData);
    const auto face_ids = meshes.Arenas().TriangleFaceIds.Get(record.TriangleFaceIds);
    uint32_t t = 0;
    for (const auto fh : mesh.faces()) {
        expect(first_triangles[*fh] == t);
        for (uint32_t k = 0; k + 2 < mesh.GetValence(fh); ++k) expect(face_ids[t++] == *fh + 1);
    }
    expect(meshes.Arenas().EdgeSharpness.Get(record.EdgeSharpness).size() == mesh.EdgeCount());
    // Every halfedge names an edge whose first halfedge names it back, and the firsts number the edges.
    uint32_t firsts = 0;
    for (uint32_t h = 0; h < mesh.HalfEdgeCount(); ++h) {
        const auto edge = *c.HalfedgeToEdge[h];
        expect(edge < mesh.EdgeCount());
        const auto first = *c.Edges[edge];
        expect(first <= h && *c.HalfedgeToEdge[first] == edge);
        firsts += first == h;
    }
    expect(firsts == mesh.EdgeCount());
}

void CheckInvariants(const Fixture &f) { CheckInvariants(f.R.Context.get<const MeshStore>(), f.ActiveMesh()); }

uint32_t SelectedCount(const Fixture &f, Element element) {
    const auto &meshes = f.R.Context.get<const MeshStore>();
    const auto &summary = meshes.GetSelectionSummary(f.ActiveMesh().GetStoreId());
    return summary.Mode == element ? summary.SelectedCount : 0u;
}

// The number of set bits in the active mesh's selection over `element`, whichever mode is active.
uint32_t BitCount(const Fixture &f, Element element) {
    uint32_t n = 0;
    for (const auto word : f.R.Context.get<const MeshStore>().GetSelectionBits(f.ActiveMesh().GetStoreId(), element)) n += uint32_t(std::popcount(word));
    return n;
}

// The active mesh's counts and invariants.
void ExpectMesh(const Fixture &f, Counts expected) {
    ExpectCounts(f.ActiveMesh(), expected);
    CheckInvariants(f);
}

std::vector<vec3> Positions(const Mesh &mesh) {
    std::vector<vec3> positions;
    for (uint32_t v = 0; v < mesh.VertexCount(); ++v) positions.push_back(mesh.GetPosition(he::VH{v}));
    return positions;
}

std::unique_ptr<RenderView> View(const Fixture &f) { return std::make_unique<RenderView>(f.R.Context.get<const GpuBuffers>().FrameView); }

// Clicks the centre of the view and expects it to pick one `element`.
void Pick(Fixture &f, Element element) {
    f.Render();
    f.Do(action::selection::ApplyEditElementClick{{0.5f, 0.5f}, false, View(f)});
    expect(SelectedCount(f, element) == 1);
}

void SelectAll(Fixture &f, Element element) {
    f.Do(action::view::SetEditMode{.Mode = element});
    f.Do(action::selection::SelectAll{});
}

// Box-selects the view's full height from its left edge to the `right` pixel of its 64.
void BoxSelect(Fixture &f, uint32_t right) {
    f.Render();
    f.Do(action::selection::ApplyBoxSelect{{{0.f, 0.f}, {float(right) / 64.f, 63.f / 64.f}}, false, View(f)});
}

struct Region {
    uint32_t BoundaryEdges, BoundaryVertices, Vertices;
};

// The selected faces' region: the edges with exactly one selected adjacent face, the vertices on them, and every vertex of a selected face.
Region RegionOf(const Fixture &f) {
    const auto mesh = f.ActiveMesh();
    const auto faces = f.R.Context.get<const MeshStore>().GetSelectionBits(mesh.GetStoreId(), Element::Face);
    const auto selected = [&](he::FH fh) { return fh && (faces[*fh / 32] >> (*fh % 32)) & 1u; };
    uint32_t edges = 0;
    std::vector<uint8_t> on_boundary(mesh.VertexCount(), 0), in_region(mesh.VertexCount(), 0);
    for (const auto eh : mesh.edges()) {
        const auto h = mesh.GetHalfedge(eh, 0);
        const auto opposite = mesh.GetOppositeHalfedge(h);
        if (selected(mesh.GetFace(h)) == (opposite && selected(mesh.GetFace(opposite)))) continue;
        ++edges;
        on_boundary[*mesh.GetToVertex(h)] = on_boundary[*mesh.GetFromVertex(h)] = 1;
    }
    for (const auto fh : mesh.faces()) {
        if (!selected(fh)) continue;
        for (const auto vh : mesh.fv_range(fh)) in_region[*vh] = 1;
    }
    return {edges, uint32_t(std::ranges::count(on_boundary, 1)), uint32_t(std::ranges::count(in_region, 1))};
}

void TestDeleteSphereVertex() {
    Fixture f{"delete", primitive::UVSphere{}, Element::Vertex};
    const auto &meshes = f.R.Context.get<const MeshStore>();
    const auto before = CountsOf(f.ActiveMesh());
    const auto base = f.P->History.Present;
    Pick(f, Element::Vertex);
    const auto picked = meshes.GetSelectionSummary(f.ActiveMesh().GetStoreId()).ActiveHandle;
    const auto valence = uint32_t(std::ranges::distance(f.ActiveMesh().voh_range(he::VH{picked})));
    const auto selected = f.P->History.Present;

    f.Do(action::mesh::Delete{action::mesh::DeleteMode::Vertices});
    const auto after = CountsOf(f.ActiveMesh());
    expect(after.Vertices == before.Vertices - 1);
    expect(after.Faces == before.Faces - valence);
    CheckInvariants(f);
    expect(SelectedCount(f, Element::Vertex) == 0);
    f.Render();

    // Undo restores the source mesh and its selection, and redo restores the output.
    f.P->Navigate(selected);
    ExpectMesh(f, before);
    expect(SelectedCount(f, Element::Vertex) == 1);
    f.P->Redo();
    ExpectMesh(f, after);
    f.Render();
    f.Audit();
    f.P->Navigate(base);
    ExpectCounts(f.ActiveMesh(), before);
}

void TestDeleteAllFaces() {
    Fixture f{"delete-all", primitive::Cuboid{}, Element::Face};
    f.Do(action::selection::SelectAll{});
    expect(SelectedCount(f, Element::Face) == 6);
    // Every face selected derives every edge and vertex.
    expect(BitCount(f, Element::Edge) == 12);
    expect(BitCount(f, Element::Vertex) == 8);
    // The GPU wrote the draws' edge endpoints in edge order, matching the host walk.
    f.Render();
    {
        const auto mesh = f.ActiveMesh();
        std::vector<uint32_t> expected(mesh.EdgeCount() * 2);
        mesh.WriteEdgeIndices(expected);
        const auto &indices = f.R.get<const MeshBuffers>(GetActiveMeshEntity(f.R)).EdgeIndices;
        const auto written = f.R.Context.get<const GpuBuffers>().EdgeIndexBuffer.Get(indices);
        expect(std::ranges::equal(written, expected));
    }
    // Only Faces keeps every vertex as a loose point cloud, and Faces removes everything.
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces});
    ExpectCounts(f.ActiveMesh(), {8, 0, 0, 0});
    f.Render();
    f.P->Undo();
    ExpectMesh(f, {8, 12, 6, 12});
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::Faces});
    ExpectCounts(f.ActiveMesh(), {0, 0, 0, 0});
    f.Render();
    f.P->Undo();
    ExpectCounts(f.ActiveMesh(), {8, 12, 6, 12});
}

void TestMerge() {
    Fixture f{"merge", primitive::Cuboid{}, Element::Vertex};
    // A box over the left half of the view takes some of the cube's vertices.
    BoxSelect(f, 31);
    const auto selected = SelectedCount(f, Element::Vertex);
    expect(selected > 1 && selected < 8);
    const auto before = f.P->History.Present;
    f.Do(action::mesh::Merge{action::mesh::MergeMode::Center});
    expect(f.ActiveMesh().VertexCount() == 8 - selected + 1);
    CheckInvariants(f);
    expect(SelectedCount(f, Element::Vertex) == 1);
    f.Render();
    f.P->Navigate(before);
    expect(f.ActiveMesh().VertexCount() == 8);
    expect(SelectedCount(f, Element::Vertex) == selected);
    // Collapsing the same run of vertices merges them at their center and drops the faces that lose their area.
    f.Do(action::mesh::Merge{.Mode = action::mesh::MergeMode::Collapse});
    expect(f.ActiveMesh().VertexCount() == 8 - selected + 1);
    CheckInvariants(f);
    f.Render();
    f.P->Navigate(before);
    // Merging every vertex leaves one point and no faces.
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Merge{action::mesh::MergeMode::First});
    ExpectCounts(f.ActiveMesh(), {1, 0, 0, 0});
}

void TestExtrude() {
    Fixture f{"extrude", primitive::Cuboid{}, Element::Face};
    const auto &meshes = f.R.Context.get<const MeshStore>();
    BoxSelect(f, 40);
    const auto selected = SelectedCount(f, Element::Face);
    expect(selected > 0 && selected < 6);
    const auto [boundary_edges, boundary_vertices, region_vertices] = RegionOf(f);
    const auto partial = f.P->History.Present;

    // A region bordering unselected faces moves onto copied boundary vertices with a side quad per boundary edge.
    f.Do(action::mesh::Extrude{action::mesh::ExtrudeMode::Region});
    expect(f.ActiveMesh().VertexCount() == 8 + boundary_vertices);
    expect(f.ActiveMesh().FaceCount() == 6 + boundary_edges);
    expect(SelectedCount(f, Element::Face) == selected);
    CheckInvariants(f);
    f.Render();

    // A staged extrude and its placement drag commit as one node named for both.
    f.P->Navigate(partial);
    const auto nodes = f.P->History.Nodes.size();
    const auto frame = [&](auto a, action::Phase phase) {
        action::Emit(std::move(a), phase);
        f.P->Frame(action::Drain());
    };
    frame(action::mesh::Extrude{action::mesh::ExtrudeMode::Region}, action::Phase::Stage);
    frame(action::view::TransformElements{{.P = vec3{0.f, 0.f, 0.5f}}}, action::Phase::Stage);
    action::Commit();
    f.P->Frame(action::Drain());
    f.Audit();
    expect(f.P->History.Nodes.size() == nodes + 1);
    expect(f.P->History.Nodes[f.P->History.Present].Label == "Extrude, TransformElements");
    expect(f.ActiveMesh().FaceCount() == 6 + boundary_edges);
    CheckInvariants(f);
    f.P->Undo();
    expect(f.ActiveMesh().FaceCount() == 6);

    // Cancelling the placement drag discards the extrude with it and records nothing.
    f.P->Navigate(partial);
    frame(action::mesh::Extrude{action::mesh::ExtrudeMode::Region}, action::Phase::Stage);
    frame(action::view::TransformElements{{.P = vec3{0.f, 0.f, 0.5f}}}, action::Phase::Stage);
    action::Cancel();
    f.P->Frame(action::Drain());
    f.Audit();
    expect(f.P->History.Nodes.size() == nodes + 1);
    expect(f.P->History.Present == partial);
    expect(f.ActiveMesh().FaceCount() == 6);

    // Splitting the same region copies only the shared boundary vertices.
    f.P->Navigate(partial);
    f.Do(action::mesh::Split{});
    ExpectMesh(f, {8 + boundary_vertices, 12 + boundary_edges, 6, 12});

    // Separating moves the region into a new mesh and leaves the rest behind.
    f.P->Navigate(partial);
    const auto mesh_entities_before = f.R.view<const MeshHandle>().size();
    f.Do(action::mesh::Separate{});
    expect(f.R.view<const MeshHandle>().size() == mesh_entities_before + 1);
    expect(f.ActiveMesh().FaceCount() == 6 - selected);
    CheckInvariants(f);
    for (const auto e : f.R.view<const MeshHandle>()) {
        const auto mesh = GetMesh(f.R, e);
        if (mesh.GetStoreId() == f.ActiveMesh().GetStoreId()) continue;
        expect(mesh.FaceCount() == selected);
        CheckInvariants(meshes, mesh);
    }
    f.Render();

    // A selection bordering nothing duplicates instead, as does Duplicate itself.
    f.P->Navigate(partial);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Extrude{action::mesh::ExtrudeMode::Region});
    ExpectMesh(f, {16, 24, 12, 24});
    expect(SelectedCount(f, Element::Face) == 6);
    f.P->Undo();
    f.Do(action::mesh::Duplicate{});
    ExpectMesh(f, {16, 24, 12, 24});
    f.P->Undo();
    f.Do(action::mesh::Extrude{action::mesh::ExtrudeMode::FacesIndividual});
    ExpectMesh(f, {32, 60, 30, 60});
    expect(SelectedCount(f, Element::Face) == 6);
    f.Render();

    // Extruding edges adds a quad per selected edge on copies of their vertices.
    f.P->Navigate(partial);
    f.Do(action::view::SetEditMode{.Mode = Element::Edge});
    BoxSelect(f, 40);
    const auto edges = SelectedCount(f, Element::Edge);
    expect(edges > 0);
    const auto edge_vertices = BitCount(f, Element::Vertex);
    f.Do(action::mesh::Extrude{action::mesh::ExtrudeMode::Edges});
    expect(f.ActiveMesh().VertexCount() == 8 + edge_vertices);
    expect(f.ActiveMesh().FaceCount() == 6 + edges);
    expect(SelectedCount(f, Element::Edge) == edges);
    CheckInvariants(f);
}

void TestDissolve() {
    Fixture f{"dissolve", primitive::Cuboid{}, Element::Vertex};
    const auto base = f.P->History.Present;

    // Dissolving one cube edge joins its two faces and removes its endpoints, which have two edges left, so the neighbors lose a corner.
    f.Do(action::view::SetEditMode{.Mode = Element::Edge});
    Pick(f, Element::Edge);
    f.Do(action::mesh::Dissolve{action::mesh::DissolveMode::Edges});
    ExpectMesh(f, {6, 9, 5, 8});
    f.Render();

    // Dissolving one cube vertex joins its three faces into a hexagon.
    f.P->Navigate(base);
    Pick(f, Element::Vertex);
    f.Do(action::mesh::Dissolve{action::mesh::DissolveMode::Vertices});
    ExpectMesh(f, {7, 9, 4, 10});
    f.Render();
    // The GPU wrote the hexagon's and the quads' fan triangles in face order, matching the host walk.
    {
        const auto mesh = f.ActiveMesh();
        const auto &indices = f.R.get<const MeshBuffers>(GetActiveMeshEntity(f.R)).FaceIndices;
        const auto written = f.R.Context.get<const GpuBuffers>().FaceIndexBuffer.Get(indices);
        expect(std::ranges::equal(written, mesh.CreateTriangleIndices()));
    }

    // Dissolving a connected region of faces leaves one face bounded by the region's boundary.
    f.P->Navigate(base);
    f.Do(action::view::SetEditMode{.Mode = Element::Face});
    BoxSelect(f, 40);
    const auto selected = SelectedCount(f, Element::Face);
    expect(selected > 1 && selected < 6);
    const auto [boundary_edges, boundary_vertices, region_vertices] = RegionOf(f);
    f.Do(action::mesh::Dissolve{action::mesh::DissolveMode::Faces});
    // A vertex inside the region loses every edge and is removed.
    const auto interior = region_vertices - boundary_vertices;
    ExpectMesh(f, {8 - interior, 12 - (selected - 1) - interior, 6 - selected + 1, boundary_edges - 2 + 2 * (6 - selected)});
    expect(SelectedCount(f, Element::Face) == 1);
    f.Render();

    // A closed selection has no boundary to walk, so nothing joins.
    f.P->Navigate(base);
    SelectAll(f, Element::Face);
    f.Do(action::mesh::Dissolve{action::mesh::DissolveMode::Faces});
    ExpectMesh(f, {8, 12, 6, 12});
}

void TestSubdivide() {
    Fixture f{"subdivide", primitive::Cuboid{}, Element::Edge};
    const auto base = f.P->History.Present;

    // Every edge cut once fills each quad with a two by two grid.
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1});
    ExpectMesh(f, {26, 48, 24, 48});
    expect(SelectedCount(f, Element::Edge) == 48);
    f.Render();
    f.P->Undo();
    f.Do(action::mesh::Subdivide{2});
    ExpectMesh(f, {56, 108, 54, 108});

    // One cut edge leaves its two quads as pentagons.
    f.P->Navigate(base);
    Pick(f, Element::Edge);
    f.Do(action::mesh::Subdivide{1});
    ExpectMesh(f, {9, 13, 6, 14});

    // A quad with two cut edges splits along the chord between the cuts, so a box of edges cuts opposite and corner pairs.
    f.P->Navigate(base);
    BoxSelect(f, 40);
    const auto edges = SelectedCount(f, Element::Edge);
    expect(edges > 1);
    // A quad whose four edges are all selected also gains its grid's center vertex.
    const auto full_faces = BitCount(f, Element::Face);
    f.Do(action::mesh::Subdivide{1});
    expect(f.ActiveMesh().VertexCount() == 8 + edges + full_faces);
    CheckInvariants(f);
    f.Render();

    // A sphere mixes quads and pole triangles, each filled by its grid.
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = 8, .Stacks = 4}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto sphere = CountsOf(f.ActiveMesh());
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1});
    expect(f.ActiveMesh().FaceCount() == 4 * sphere.Faces);
    CheckInvariants(f);
}

void TestFaceOperators() {
    Fixture f{"faces", primitive::Cuboid{}, Element::Face};
    f.Do(action::selection::SelectAll{});
    const auto base = f.P->History.Present;

    // Triangulating every quad and joining the triangles back restores six quads.
    f.Do(action::mesh::Triangulate{});
    ExpectMesh(f, {8, 18, 12, 12});
    f.Do(action::mesh::TrisToQuads{});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();

    // Poking every face adds a center per face and fans it, and the centers lift along their face normals.
    f.P->Navigate(base);
    f.Do(action::mesh::Poke{0.5f});
    ExpectMesh(f, {14, 36, 24, 24});
    expect(std::ranges::any_of(Positions(f.ActiveMesh()), [](vec3 p) { return std::abs(p.x) > 1.4f; }));
    f.Render();

    // Flipping keeps every count and every pairing.
    f.P->Navigate(base);
    f.Do(action::mesh::FlipNormals{});
    ExpectMesh(f, {8, 12, 6, 12});

    // Splitting every edge leaves six separate quads.
    f.P->Navigate(base);
    SelectAll(f, Element::Edge);
    f.Do(action::mesh::EdgeSplit{});
    ExpectMesh(f, {24, 24, 6, 12});
    f.Render();

    // Inset individual keeps each face's ring and inner face, moved inward.
    f.P->Navigate(base);
    f.Do(action::mesh::Inset{.Thickness = 0.25f, .Individual = true});
    ExpectMesh(f, {32, 60, 30, 60});
    expect(std::ranges::count_if(Positions(f.ActiveMesh()), [](vec3 p) {
               return std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) < 0.99f || (std::abs(std::abs(p.x) - 0.75f) < 1e-3f || std::abs(std::abs(p.y) - 0.75f) < 1e-3f || std::abs(std::abs(p.z) - 0.75f) < 1e-3f);
           }) == 24);
    f.Render();
    // A negative thickness insets by zero, leaving the copies on their corners.
    f.P->Undo();
    f.Do(action::mesh::Inset{.Thickness = -0.25f, .Individual = true});
    ExpectMesh(f, {32, 60, 30, 60});
    expect(std::ranges::count_if(Positions(f.ActiveMesh()), [](vec3 p) { return std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) < 0.99f; }) == 0);

    // Inset of a partial region moves the boundary copies inward and leaves side quads like an extrude.
    f.P->Navigate(base);
    BoxSelect(f, 40);
    const auto [boundary_edges, boundary_vertices, region_vertices] = RegionOf(f);
    f.Do(action::mesh::Inset{.Thickness = 0.2f});
    expect(f.ActiveMesh().VertexCount() == 8 + boundary_vertices);
    expect(f.ActiveMesh().FaceCount() == 6 + boundary_edges);
    CheckInvariants(f);
    f.Render();

    // Filling the hole left by a deleted face restores the cube.
    f.P->Navigate(base);
    Pick(f, Element::Face);
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces});
    ExpectCounts(f.ActiveMesh(), {8, 12, 5, 10});
    SelectAll(f, Element::Edge);
    f.Do(action::mesh::Fill{});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();
    // A grid fill of the same hole with a span of one restores the cube's quad.
    f.P->Undo();
    f.Do(action::mesh::GridFill{1});
    ExpectMesh(f, {8, 12, 6, 12});
    // A span of two on a loop cut around the hole fills with a two by two grid on a center vertex.
    f.P->Undo();
    f.Do(action::mesh::Subdivide{1});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::GridFill{2});
    expect(f.ActiveMesh().FaceCount() == 24);
    CheckInvariants(f);
    f.Render();

    // A loop cut through a picked edge rings the cube with four new edges.
    f.P->Navigate(base);
    f.Do(action::view::SetEditMode{.Mode = Element::Edge});
    Pick(f, Element::Edge);
    f.Do(action::mesh::LoopCut{1});
    ExpectMesh(f, {12, 20, 10, 20});
    expect(SelectedCount(f, Element::Edge) == 4);
}

void TestMergeVariants() {
    Fixture f{"merge-variants", primitive::Cuboid{}, Element::Vertex};
    const auto base = f.P->History.Present;

    // Splitting every edge apart and merging by distance welds the cube back together.
    SelectAll(f, Element::Edge);
    f.Do(action::mesh::EdgeSplit{});
    ExpectCounts(f.ActiveMesh(), {24, 24, 6, 12});
    SelectAll(f, Element::Vertex);
    f.Do(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = 0.001f});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();

    // A limited dissolve over a subdivided cube removes every flat edge and every straight vertex, leaving the six faces.
    f.P->Navigate(base);
    SelectAll(f, Element::Edge);
    f.Do(action::mesh::Subdivide{1});
    ExpectCounts(f.ActiveMesh(), {26, 48, 24, 48});
    SelectAll(f, Element::Face);
    f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = 0.1f});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();

    // Degenerate dissolve collapses edges under the distance, which a tiny cube has everywhere.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Degenerate, .Distance = 10.f});
    ExpectCounts(f.ActiveMesh(), {1, 0, 0, 0});
}

void TestTransformOperators() {
    Fixture f{"transforms", primitive::Cuboid{}, Element::Face};
    const auto base = f.P->History.Present;

    // Spinning a picked face four times extrudes it four times, each step rotated further, so the last step's face has rotated a full radian about the center.
    Pick(f, Element::Face);
    f.Do(action::mesh::Spin{.Steps = 4, .Angle = 1.f, .Axis = {0.f, 1.f, 0.f}, .Center = {3.f, 0.f, 0.f}});
    ExpectMesh(f, {24, 44, 22, 44});
    expect(std::ranges::any_of(Positions(f.ActiveMesh()), [](vec3 p) { return std::abs(p.z) > 1.5f; }));
    f.Render();

    // Repeating an extrude moves each step by the offset.
    f.P->Navigate(base);
    Pick(f, Element::Face);
    f.Do(action::mesh::ExtrudeRepeat{.Steps = 3, .Offset = {0.f, 0.f, 0.5f}});
    ExpectMesh(f, {20, 36, 18, 36});

    // Solidifying the whole cube adds an inner flipped shell with no rim.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Solidify{0.2f});
    ExpectMesh(f, {16, 24, 12, 24});
    expect(std::ranges::count_if(Positions(f.ActiveMesh()), [](vec3 p) { return std::abs(p.x) < 0.9f; }) == 8);
    f.Render();

    // Bisecting along x = 0 rings the cube, and clearing the inner side halves it.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Bisect{.Point = {0.f, 0.f, 0.f}, .Normal = {1.f, 0.f, 0.f}});
    ExpectMesh(f, {12, 20, 10, 20});
    f.P->Undo();
    f.Do(action::mesh::Bisect{.Point = {0.f, 0.f, 0.f}, .Normal = {1.f, 0.f, 0.f}, .ClearInner = true});
    ExpectMesh(f, {8, 12, 5, 10});
    f.Render();

    // Symmetrizing across x mirrors the positive half back into a ringed cube.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Symmetrize{.Axis = action::mesh::SymmetrizeAxis::X});
    ExpectMesh(f, {12, 20, 10, 20});
    f.Render();

    // A knife stroke across the view cuts the edges it crosses.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Render();
    f.Do(action::mesh::Knife{{0.f, 32.f}, {63.f, 32.f}, View(f)});
    expect(f.ActiveMesh().FaceCount() > 6);
    CheckInvariants(f);
}

void TestListOperators() {
    Fixture f{"lists", primitive::Cuboid{}, Element::Face};
    f.Do(action::selection::SelectAll{});
    const auto base = f.P->History.Present;

    // Two bisects leave a four-quad tube with a ring at each end.
    f.Do(action::mesh::Bisect{.Point = {0.f, 0.f, 0.f}, .Normal = {1.f, 0.f, 0.f}, .ClearInner = true});
    f.Do(action::mesh::Bisect{.Point = {0.5f, 0.f, 0.f}, .Normal = {1.f, 0.f, 0.f}, .ClearOuter = true});
    ExpectCounts(f.ActiveMesh(), {8, 12, 4, 8});
    const auto tube = f.P->History.Present;
    SelectAll(f, Element::Edge);
    // Bridging the rings adds a quad strip between them.
    f.Do(action::mesh::BridgeEdgeLoops{});
    ExpectMesh(f, {8, 12, 8, 16});
    f.Render();
    // Filling the holes caps both rings.
    f.P->Navigate(tube);
    f.Do(action::mesh::FillHoles{4});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();

    // The hull of the cube's vertices adds twelve triangles over its faces, sharing the cube's edges and adding a diagonal per face.
    f.P->Navigate(base);
    SelectAll(f, Element::Vertex);
    f.Do(action::mesh::ConvexHull{});
    ExpectMesh(f, {8, 18, 18, 24});

    // Rotating a picked edge keeps the counts and every pairing.
    f.P->Navigate(base);
    f.Do(action::view::SetEditMode{.Mode = Element::Edge});
    Pick(f, Element::Edge);
    f.Do(action::mesh::EdgeRotate{});
    ExpectMesh(f, {8, 12, 6, 12});
    f.Render();

    // Ripping every edge tears the faces apart and selects the torn copies.
    f.P->Navigate(base);
    SelectAll(f, Element::Edge);
    f.Do(action::mesh::Rip{});
    ExpectMesh(f, {24, 24, 6, 12});
}

void TestBevel() {
    Fixture f{"bevel", primitive::Cuboid{}, Element::Edge};
    const auto base = f.P->History.Present;

    // Beveling one edge chamfers it and cuts the corner off the two end faces.
    Pick(f, Element::Edge);
    f.Do(action::mesh::Bevel{.Width = 0.2f});
    ExpectMesh(f, {10, 15, 7, 16});
    f.Render();

    // Beveling every edge gives the classic beveled cube, and segments round its strips.
    f.P->Navigate(base);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Bevel{.Width = 0.2f});
    ExpectMesh(f, {24, 48, 26, 44});
    f.Render();
    f.P->Undo();
    f.Do(action::mesh::Bevel{.Width = 0.2f, .Segments = 2});
    ExpectMesh(f, {48, 84, 38, 92});
    f.Render();

    // Beveling one vertex cuts its corner off into a triangle.
    f.P->Navigate(base);
    f.Do(action::view::SetEditMode{.Mode = Element::Vertex});
    Pick(f, Element::Vertex);
    f.Do(action::mesh::Bevel{.Width = 0.2f, .Vertices = true});
    ExpectMesh(f, {10, 15, 7, 16});
}

// Editing a committed operator re-runs its recorded actions with the edited values on the node's parent.
// The commit forks a node with descendants and replaces a leaf node in place.
void TestEditNode() {
    Fixture f{"edit-node", primitive::Cuboid{}, Element::Edge};
    f.Do(action::selection::SelectAll{});
    const auto base = f.P->History.Present;
    const auto node = f.Do(action::mesh::Bevel{.Width = 0.1f});
    const auto count = f.P->History.Nodes.size();
    f.Do(action::mesh::Subdivide{1});
    const auto child = f.P->History.Present;
    f.P->Navigate(node);

    const auto bevel = [&](int of) -> action::mesh::Bevel & {
        return std::get<action::mesh::Bevel>(std::get<action::mesh::Action>(f.P->DraftOf(of).RecordedActions[0].Action));
    };
    const auto restage = [&](float width, uint32_t segments) {
        bevel(node).Width = width;
        bevel(node).Segments = segments;
        f.P->RequestRestage();
        f.P->Frame(action::Drain());
    };
    restage(0.2f, 1);
    expect(f.P->History.Present == base);
    expect(f.P->Editing == node);
    restage(0.3f, 2);
    action::Commit();
    f.P->Frame(action::Drain());
    f.Audit();
    const auto forked = f.P->History.Present;
    expect(forked != node);
    expect(f.P->History.Nodes[forked].Parent == base);
    expect(f.P->History.Nodes[forked].Label == f.P->History.Nodes[node].Label);
    expect(f.P->History.Nodes.size() == count + 2);
    expect(f.P->History.Nodes[node].Children == std::vector{child});
    expect(!f.P->Editing);
    ExpectMesh(f, {48, 84, 38, 92});
    expect(bevel(forked).Width == 0.3f);
    // Running the final bevel once from the same base gives the same mesh.
    const auto staged_positions = Positions(f.ActiveMesh());
    f.P->Navigate(base);
    f.Do(action::mesh::Bevel{.Width = 0.3f, .Segments = 2});
    expect(Positions(f.ActiveMesh()) == staged_positions);

    f.P->Navigate(forked);
    bevel(forked).Width = 0.2f;
    f.P->RequestRestage();
    f.P->Frame(action::Drain());
    action::Commit();
    f.P->Frame(action::Drain());
    f.Audit();
    expect(f.P->History.Present == forked);
    expect(f.P->History.Nodes.size() == count + 2);
    expect(bevel(forked).Width == 0.2f);
    expect(bevel(forked).Segments == 2u);
    const auto replaced_counts = CountsOf(f.ActiveMesh());

    // Cancelling an edit returns to the node unchanged.
    bevel(forked).Width = 0.5f;
    f.P->RequestRestage();
    f.P->Frame(action::Drain());
    expect(f.P->History.Present == base);
    action::Cancel();
    f.P->Frame(action::Drain());
    expect(f.P->History.Present == forked);
    expect(!f.P->Editing);
    ExpectCounts(f.ActiveMesh(), replaced_counts);
    expect(bevel(forked).Width == 0.2f);
}

// A loaded mesh with authored normals carries its custom corner normals through an operator.
void TestCustomNormals() {
    Fixture f{"custom-normals"};
    f.Do(action::io::Load{std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/Duck/glTF/Duck.gltf"});
    const auto &meshes = f.R.Context.get<const MeshStore>();
    state::Entity instance_entity = state::Null;
    for (const auto [e, instance] : f.R.view<const Instance>().each()) {
        const auto *handle = f.R.try_get<const MeshHandle>(instance.Entity);
        if (handle && meshes.Get(handle->StoreId).CustomCornerMasks.Count > 0) instance_entity = e;
    }
    expect(instance_entity != state::Null);
    f.Do(action::selection::Select{instance_entity});
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    SelectAll(f, Element::Face);
    const auto before = CountsOf(f.ActiveMesh());
    const auto custom_before = meshes.Get(f.ActiveMesh().GetStoreId()).CustomCornerNormals.Count;
    expect(custom_before > 0);
    f.Do(action::mesh::Subdivide{});
    const auto after = CountsOf(f.ActiveMesh());
    expect(after.Faces > before.Faces);
    CheckInvariants(f);
    // Every original corner keeps its authored normal, and the cut corners derive theirs.
    expect(meshes.Get(f.ActiveMesh().GetStoreId()).CustomCornerNormals.Count == custom_before);
    f.Render();
    f.P->Undo();
    ExpectCounts(f.ActiveMesh(), before);
}
} // namespace

int main() {
    // Progress reaches the terminal even when a case crashes.
    setvbuf(stdout, nullptr, _IONBF, 0);
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    boost::ut::suite tests = [] {
        using namespace boost::ut;
        "delete a picked sphere vertex with its fan and round-trip the history"_test = TestDeleteSphereVertex;
        "delete every cube face"_test = TestDeleteAllFaces;
        "merge vertices at center, by collapse, and at first"_test = TestMerge;
        "extrude, split, separate, and duplicate a region"_test = TestExtrude;
        "dissolve an edge, a vertex, and a face region"_test = TestDissolve;
        "subdivide edges into grids, pentagons, and chords"_test = TestSubdivide;
        "triangulate, join, poke, flip, split edges, inset, fill, grid fill, and loop cut"_test = TestFaceOperators;
        "merge by distance, limited and degenerate dissolve"_test = TestMergeVariants;
        "spin, repeat, solidify, bisect, symmetrize, and knife"_test = TestTransformOperators;
        "bridge, fill holes, hull, edge rotate, and rip"_test = TestListOperators;
        "bevel edges with segments and bevel vertices"_test = TestBevel;
        "editing a node restages on its parent and replaces it"_test = TestEditNode;
        "custom corner normals carry through a subdivide"_test = TestCustomNormals;
    };
    return RunSuites();
}
