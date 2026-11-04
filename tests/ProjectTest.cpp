#include "Paths.h"
#include "ProjectFixture.h"
#include "RunSuites.h"
#include "procedure/Procedure.h"
#include "project/Assets.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionState.h"
#include "viewport/ViewCameraOps.h"

using boost::ut::expect;

namespace {
void WriteFile(const std::filesystem::path &path, std::string_view text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream{path, std::ios::binary} << text;
}

void WriteTriangle(const std::filesystem::path &path, float x) {
    const std::array<vec3, 3> positions{vec3{0, 0, 0}, vec3{x, 0, 0}, vec3{0, 1, 0}};
    WriteFile(path, {reinterpret_cast<const char *>(positions.data()), sizeof(positions)});
}

// Hiding and showing an instance keeps its slot, through undo and redo, and a duplicate of a hidden instance is hidden.
void RenderInstanceDerivation(Fixture &f) {
    auto &r = f.R;
    const auto entity = f.AddCube();
    const auto slot = [&] { return r.get<const RenderInstance>(entity).BufferIndex; };
    const auto placed = slot();
    f.Do(action::object::SetVisible{false});
    expect(slot() == placed);
    f.Do(action::object::SetVisible{true});
    expect(slot() == placed);
    f.Do(action::object::SetVisible{false});
    f.Undo();
    f.Redo();
    expect(slot() == placed);
    const auto duplicate = f.Add(action::object::Duplicate{});
    expect(r.all_of<Hidden>(duplicate));
    f.Do(action::object::SetVisible{true});
    const auto other_mesh = r.get<const Instance>(f.AddCube()).Entity;
    r.replace<Instance>(duplicate, other_mesh);
    f.Commit("Retarget duplicate");
}

void HiddenObjectEscapesPick(Fixture &f) {
    auto &r = f.R;
    const auto cube = f.AddCube();
    const auto viewport = FirstViewport(r);
    f.Image();
    const auto boxed = RunBoxSelect(r, viewport, {uvec2{24, 24}, uvec2{40, 40}});
    f.Do(action::selection::DeselectAll{});
    SyncViewportRenderResources(r, viewport, {96, 80});
    f.Do(action::selection::ApplyBoxSelect{{vec2{.375f}, vec2{.625f}}, false});
    expect(std::ranges::equal(r.view<const Selected>(), boxed));
    const auto pick = [&] { f.Do(action::selection::Pick{false, PickObject(r, viewport, {32, 32})}); };
    f.Do(action::selection::DeselectAll{});
    pick();
    expect(r.all_of<Selected>(cube));
    f.Do(action::object::SetVisible{false});
    f.Do(action::selection::DeselectAll{});
    pick();
    expect(r.view<const Selected>().empty());
}

void ExtrudedClosedRegionVisibility(Fixture &f) {
    auto &r = f.R;
    const auto cube = f.AddCube(), viewport = FirstViewport(r);
    f.Do(action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, false));
    const std::array directions{vec3{1, .2f, .3f}, vec3{-1, .2f, .3f}, vec3{.2f, 1, .3f}, vec3{.2f, -1, .3f}, vec3{.2f, .3f, 1}, vec3{.2f, .3f, -1}};
    const auto image = [&](vec3 direction) {
        r.patch<ViewCameraState>(viewport, [&](auto &v) { v.Active = ViewCamera{8.f * Normalize(direction), vec3{}, v.Active.Data}; });
        return f.Image();
    };
    const std::array modes{ViewportShadingMode::Solid, ViewportShadingMode::MaterialPreview, ViewportShadingMode::Rendered};
    std::array<std::array<std::vector<std::byte>, directions.size()>, modes.size()> before;
    for (size_t mode = 0; mode < modes.size(); ++mode) {
        f.Do(action::view::SetViewportShading{viewport, modes[mode]});
        for (size_t i = 0; i < directions.size(); ++i) before[mode][i] = image(directions[i]);
    }
    f.EnterEdit(Element::Face);
    f.Do(action::selection::SelectAll{});
    f.Stage(action::mesh::Extrude{});
    f.Stage(action::view::TransformElements{{.P = {50, 50, 50}}});
    f.CommitGesture();
    expect(GetMesh(r, cube).FaceCount() == 12u);
    for (size_t mode = 0; mode < modes.size(); ++mode) {
        f.Do(action::view::SetViewportShading{viewport, modes[mode]});
        f.Do(action::view::SetInteractionMode{InteractionMode::Object});
        for (size_t i = 0; i < directions.size(); ++i) expect(image(directions[i]) == before[mode][i]);
        f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
        for (const auto direction : directions) {
            image(direction);
            const auto hit = RunEditElementPick(r, viewport, Element::Face, {32, 32});
            expect(hit.has_value());
            if (!hit) continue;
            expect(Dot(test::Centroid(GetMesh(r, cube), he::FH{hit->Handle}), direction) > 0.f);
        }
    }
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
}

// Drawing and object/face picking agree on visibility across viewport and material culling policies.
void ViewportBackfacePolicy(Fixture &f) {
    auto &r = f.R;
    f.Add(action::object::AddMeshPrimitive{primitive::Plane{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto viewport = FirstViewport(r);
    f.Do(action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, false));
    f.Do(action::UpdateOf<&ViewportDisplay::XRayWireframe>(viewport, false));
    r.patch<ViewCameraState>(viewport, [](auto &v) { v.Active = ViewCamera{vec3{0, -5, .1f}, vec3{}, v.Active.Data}; });
    const auto cooks = Cooks(r);
    for (const auto mode : {ViewportShadingMode::Solid, ViewportShadingMode::Wireframe, ViewportShadingMode::MaterialPreview, ViewportShadingMode::Rendered}) {
        f.Do(action::view::SetViewportShading{viewport, mode});
        for (const bool cull : {false, true}) {
            f.Do(action::UpdateOf<&ViewportDisplay::BackfaceCulling>(viewport, cull));
            for (const auto double_sided : {0u, 1u}) {
                f.Do(action::object::UpdateMaterial<uint32_t>{0, offsetof(PBRMaterial, DoubleSided), double_sided});
                f.Do(action::view::SetInteractionMode{InteractionMode::Object});
                f.Do(action::object::SetVisible{false});
                const auto empty = f.Image();
                f.Do(action::object::SetVisible{true});
                const bool visible = mode == ViewportShadingMode::Wireframe || (mode == ViewportShadingMode::Solid ? !cull : double_sided != 0u);
                expect((f.Image() != empty) == visible);
                expect(PickObject(r, viewport, {32, 32}).has_value() == visible);
                f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
                f.Image();
                expect(RunEditElementPick(r, viewport, Element::Face, {32, 32}).has_value() == visible);
            }
        }
    }
    expect(Cooks(r) == cooks);
}

// Three empties chained A > B > C, with A and C selected and A active.
// The transform applies once to the selected grandchild, which the transform roots exclude.
void ObjectHierarchy(Fixture &f) {
    auto &r = f.R;
    const auto a = f.AddEmpty("A"), b = f.AddEmpty("B"), c = f.AddEmpty("C");
    f.Do(action::object::SetParent{b, a});
    f.Do(action::object::SetParent{c, b});
    f.Do(action::selection::Select{c});
    f.Do(action::selection::ExtendActive{a});
    expect(r.get<const TransformRoots>(state::Document).Roots == std::vector{a});
    f.Do(action::view::TransformSelection{Transform{.P = {1, 0, 0}}});
    expect(std::abs(WorldTransformOf(r, a)->P.x - 1.f) < 1e-4f);
    expect(std::abs(WorldTransformOf(r, c)->P.x - 1.f) < 1e-4f);
}
// A linked duplicate shares its source's mesh, and hiding one instance or deleting the other, each undone, renders the scene as before.
void InstanceLifecycleRendersBack(Fixture &f) {
    auto &r = f.R;
    const auto first = f.AddCube();
    f.Do(action::view::TransformSelection{Transform{.P = {-2, 0, 0}}});
    const auto linked = f.Add(action::object::DuplicateLinked{});
    f.Do(action::view::TransformSelection{Transform{.P = {2, 0, 0}}});
    expect(r.get<const Instance>(linked).Entity == r.get<const Instance>(first).Entity);
    const auto renders_back = [&](state::Entity e, auto a) {
        f.Do(action::selection::Select{e});
        const auto before = Render ? f.Image() : std::vector<std::byte>{};
        f.Do(std::move(a));
        f.Undo();
        if (Render) expect(f.Image() == before);
    };
    renders_back(first, action::object::SetVisible{false});
    renders_back(linked, action::object::Delete{});
}

// An unlinked duplicate copies its source's drawn record onto storage of its own, which hashes equal to its source's, and keeps its source's Source states.
void DuplicateDrawsAsItsSource(Fixture &f) {
    auto &r = f.R;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto render_owner = [&](state::Entity e) -> const MeshStore::Record & {
        return meshes.Get(*DrawnStoreId(r, r.get<const Instance>(e).Entity));
    };
    const auto duplicate = [&](state::Entity source) {
        f.Do(action::selection::Select{source});
        const auto copy = f.Add(action::object::Duplicate{});
        expect(render_owner(copy).StoreId != render_owner(source).StoreId);
        const auto states = [&](state::Entity instance) {
            std::vector<procedure::SourceState> found;
            for (const auto &node : r.get<const procedure::CookPlan>(r.get<const Instance>(instance).Entity).Nodes) {
                if (const auto *source = r.try_get<const procedure::SourceState>(node.Entity)) found.push_back(*source);
            }
            return found;
        };
        expect(states(copy) == states(source));
        expect(meshes.CanonicalHash(render_owner(copy).StoreId) == meshes.CanonicalHash(render_owner(source).StoreId));
        return copy;
    };
    const auto fragment = [&](state::Entity instance, std::span<const uint32_t> selected) {
        f.Do(action::selection::Select{instance});
        f.EnterEdit(Element::Vertex);
        SelectOrdinals(f, r.get<const Instance>(instance).Entity, selected, Element::Vertex);
        f.Do(action::mesh::Delete{action::mesh::DeleteMode::Vertices});
        f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    };
    MeshSource points;
    for (uint32_t i = 0u; i < 770u; ++i) points.Data.Positions.push_back({float(i) * .001f, float(i % 7u) * .1f, 0.f});
    const auto point_source = f.AddSourced(std::move(points)).second;
    std::vector<uint32_t> removed;
    for (uint32_t i = 0u; i < 256u; ++i) removed.push_back(i);
    removed.push_back(512u);
    fragment(point_source, removed);
    expect(render_owner(duplicate(point_source)).RenderTopologies == 4u);
    MeshSource lines;
    for (uint32_t i = 0u; i < 520u; ++i) {
        lines.Data.Positions.push_back({float(i) * .001f, float(i % 3u) * .1f, 0.f});
        if (i) lines.Data.Edges.push_back({i - 1u, i});
    }
    const auto line_source = f.AddSourced(std::move(lines)).second;
    fragment(line_source, std::array{255u, 300u});
    expect(render_owner(duplicate(line_source)).RenderTopologies == 2u);

    // Enough triangles to take a cluster hierarchy, with 128-face fans at the poles.
    const auto source = f.Add(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = 128u, .Stacks = 64u}, std::make_unique<MeshInstanceCreateInfo>()});
    expect(meshes.ClusterGroupCount(render_owner(source)) > 0u);
    f.EnterEdit(Element::Face);
    SelectOrdinals(f, r.get<const Instance>(source).Entity, std::array{2000u, 3000u}, Element::Face);
    f.Do(action::mesh::Subdivide{2u});
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::selection::SelectAll{});
    f.Do(action::object::SetVisible{false});
    f.Do(action::selection::Select{source});
    f.Do(action::object::SetVisible{true});
    const auto before = f.Image();
    const auto copy = duplicate(source);
    f.Do(action::selection::Select{source});
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    f.Do(action::selection::SelectAll{});
    f.Stage(action::view::TransformElements{{.R = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{1.f, 0.f, 0.f})}});
    f.CommitGesture();
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::SetVisible{false});
    f.Do(action::selection::Select{copy});
    if (Render) expect(f.Image() == before);
    f.Do(action::selection::Select{point_source});
    f.Do(action::selection::ExtendActive{line_source});
    f.Do(action::selection::ExtendActive{copy});
    const auto mesh_count = r.view<const procedure::GeometryProcedure>().size();
    f.Do(action::object::Duplicate{});
    expect(r.view<const procedure::GeometryProcedure>().size() == mesh_count + 3u);
}

// Mesh layouts follow scene load, linked skinned duplicates, history, edit-mode drags, separation and shading, which the per-step layout check verifies.
void IncrementalLayout(Fixture &f) {
    auto &r = f.R;
    f.Do(action::io::LoadGltf{RiggedSimple});
    expect(!r.view<const PosedLayout>().empty());
    const auto skinned = *r.view<const ArmatureModifier>().begin();
    f.Do(action::selection::Select{skinned});
    f.Add(action::object::DuplicateLinked{});
    expect(r.get<const BoundsRun>(r.get<const Instance>(skinned).Entity).Entries.Count == 2u);
    const auto cube = f.AddCube();
    f.Add(action::object::Duplicate{});
    f.Do(action::object::SetVisible{false});
    f.Do(action::object::Delete{});
    f.Undo();
    f.Do(action::selection::Select{cube});
    f.EnterEdit(Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto mesh_entity = r.get<const Instance>(cube).Entity;
    f.Stage(action::view::TransformElements{{.P = {0.f, 1.f, 0.f}}});
    expect(r.all_of<procedure::TransformElements>(r.get<const procedure::GeometryProcedure>(mesh_entity).Last));
    CheckLayout(f);
    f.CommitGesture();
    SelectOrdinals(f, mesh_entity, std::array{0u, 1u}, Element::Face);
    const auto mesh_count = r.view<const procedure::GeometryProcedure>().size();
    f.Do(action::mesh::Separate{});
    expect(r.view<const procedure::GeometryProcedure>().size() == mesh_count + 1u);
    expect(r.all_of<action::mesh::Delete>(r.get<const procedure::GeometryProcedure>(mesh_entity).Last));
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::ShadeSelected{procedure::ShadeMode::Smooth});
    f.Do(action::view::SetViewportShading{FirstViewport(f.R), ViewportShadingMode::Wireframe});
}

void TestProject() {
    Fixture f, other;
    const TestDir source{"/tmp/mesheditor-scratch/project-external-source"};
    const auto gltf = source.Path / "Triangle.gltf", buffer = source.Path / "Triangle.bin";
    WriteFile(gltf, R"({
  "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [2, 1, 0]}],
  "bufferViews": [{"buffer": 0, "byteLength": 36}], "buffers": [{"byteLength": 36, "uri": "Triangle.bin"}]
})");
    WriteTriangle(buffer, 1.f);
    Scenario(f, other, "external references", [&](Fixture &f) {
        const auto max_x = [&] {
            expect(f.R.view<const procedure::GeometryProcedure>().size() == 1u);
            const auto mesh = GetMesh(f.R, *f.R.view<const procedure::GeometryProcedure>().begin());
            float largest = 0;
            for (uint32_t i = 0; i < mesh.VertexCount(); ++i) largest = std::max(largest, test::Position(mesh, mesh.VertexAt(i)).x);
            return largest;
        };
        f.Do(action::io::LoadGltf{gltf});
        const auto &actions = f.P->DraftOf(*f.P->History.Present).RecordedActions;
        const auto &path = std::get<action::io::LoadGltf>(std::get<action::io::Action>(actions.front())).Path;
        expect(path == gltf && path.is_absolute());
        expect(max_x() == 1.f);
        expect(!std::filesystem::exists(f.P->History.Dir / project::Assets::DirectoryName));
        // Replay reads the buffer's current contents, so a changed buffer makes the replay differ and a missing one fails it, each leaving the document.
        const auto document = f.P->History.MaterializeLive();
        auto &errors = f.R.Resources.get<action::Errors>().Messages;
        for (const bool missing : {false, true}) {
            if (missing) std::filesystem::remove(buffer);
            else WriteTriangle(buffer, 2.f);
            expect(!f.P->Replay());
            expect(f.P->History.MaterializeLive() == document);
            expect(!errors.empty());
            errors.clear();
        }
        WriteTriangle(buffer, 1.f);
    });
    Scenario(f, other, "render instance derivation", RenderInstanceDerivation);
    Scenario(f, other, "hidden object escapes pick", HiddenObjectEscapesPick);
    if (Render) {
        Scenario(f, other, "extruded closed region visibility", ExtrudedClosedRegionVisibility);
        Scenario(f, other, "viewport backface policy", ViewportBackfacePolicy);
    }
    Scenario(f, other, "object hierarchy", ObjectHierarchy);
    Scenario(f, other, "instance lifecycle renders back", InstanceLifecycleRendersBack);
    Scenario(f, other, "duplicate draws as its source", DuplicateDrawsAsItsSource);
    Scenario(f, other, "incremental layout", IncrementalLayout);
}
} // namespace

void TestProcedures();
void TestHistory();

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) {
        if (std::string_view{argv[i]} == "--no-render") Render = false;
        else if (argv[i][0] == '-') {
            std::println(stderr, "Usage: {} [--no-render] [test name filter]", argv[0]);
            return 1;
        } else boost::ut::cfg<boost::ut::override> = {.filter = argv[i]};
    }

    if (!Render) std::println("Skipping rendered-image comparisons (--no-render).");
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    TestProject();
    TestProcedures();
    TestHistory();
    return RunSuites();
}
