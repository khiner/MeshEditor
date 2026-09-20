#include "project/Project.h"
#include "Compress.h"
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "project/Sessions.h"
#include "render/GpuBuffers.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/Viewport.h"
#include <fstream>

#include <cstdio>
#include <map>

using boost::ut::expect;

namespace {
bool Render = true;

struct Fixture : Engine {
    Fixture() : Engine{true} {}
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
    template<typename A> void Stage(A a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        Audit();
    }
    void Finish() {
        action::Commit();
        P->Frame(action::Drain());
        Audit();
    }
    std::vector<std::byte> Image() {
        SubmitViewport(R, Viewport);
        WaitForRender(R);
        const auto &image = R.Context.get<const RenderTargets>().Resources->FinalColorImage;
        return ReadbackImageRgba8(R.Context.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};

struct State {
    std::vector<std::byte> Persistent, Image;
    explicit State(Fixture &f)
        : Persistent(f.P->History.MaterializeLive()), Image(Render ? f.Image() : std::vector<std::byte>{}) {
        expect(f.P->History.MaterializeLive() == Persistent);
    }
    void Check(Fixture &f) const {
        expect(f.P->History.MaterializeLive() == Persistent);
        if (Render) {
            const auto rendered = f.Image();
            if (rendered != Image) std::printf("node %d image differs at byte %zu\n", f.P->History.Present, size_t(std::ranges::mismatch(Image, rendered).in1 - Image.begin()));
            expect(rendered == Image);
        }
        expect(f.P->History.MaterializeLive() == Persistent);
        f.Audit();
    }
};

void TestNativeStateHistory() {
    const TestDir dir{"mesheditor-native-state"};
    Fixture f;
    auto &p = *f.P;
    expect(p.Begin(dir));
    const auto entity = f.R.create();
    const std::string first(8192, 'a'), second(16384, 'b');
    f.R.emplace<Name>(entity, first);
    const auto before = p.History.Commit("native first", {});
    f.R.patch<Name>(entity, [&](auto &name) { name.Value = second; });
    const auto after = p.History.Commit("native second", {});
    expect(p.History.Stats().OwnedBytes >= first.size());
    const auto epoch = f.R.Epoch;
    p.Navigate(before);
    expect(f.R.Epoch != epoch && f.R.get<Name>(entity).Value == first);
    p.Navigate(after);
    expect(f.R.get<Name>(entity).Value == second);
    f.R.DocumentReadOnly = true;
    bool rejected = false;
    try {
        f.R.edit<Name>(entity).Value = "must not write";
    } catch (const std::logic_error &) { rejected = true; }
    f.R.DocumentReadOnly = false;
    expect(rejected && f.R.get<Name>(entity).Value == second);
    p.History.Evict(0);
    p.Navigate(before);
    expect(f.R.get<Name>(entity).Value == first);
    // The derived name index must describe the restored value, including same-ID replacement.
    const auto other = f.R.create();
    expect(EmplaceUniqueName(f.R, other, first).Value != first);
    f.R.destroy(other);
    p.Navigate(after);
    expect(f.R.get<Name>(entity).Value == second);
    f.Audit();
}

void TestPickingIdentity() {
    const TestDir dir{"mesheditor-picking-identity"};
    Fixture f;
    expect(f.P->Begin(dir));
    f.R.Context.get<ViewportExtent>().Value = {64, 64};
    f.P->Settle();
    f.Do(action::object::AddMeshPrimitive{primitive::UVSphere{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto first = FindActiveEntity(f.R);
    const auto check = [&](state::Entity entity) {
        f.Image();
        const auto box = RunBoxSelect(f.R, f.Viewport, {{0, 0}, {63, 63}});
        expect(std::ranges::find(box, entity) != box.end());
        const auto picked = RunObjectPick(f.R, {32, 32}, 32);
        expect(std::ranges::find(picked, entity) != picked.end());
    };
    check(first);
    // Grow the picking buffers without changing the visible object or its identity.
    for (int i = 0; i < 500; ++i) f.R.create();
    check(first);
    f.Do(action::object::Delete{});
    f.Do(action::object::AddMeshPrimitive{primitive::UVSphere{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto recycled = FindActiveEntity(f.R);
    expect(recycled != first && !f.R.valid(first));
    check(recycled);
}

void TestProject(const char *sample) {
    std::printf("history: %s\n", sample);
    const TestDir dir{"mesheditor-history"}, source{"mesheditor-history-source"}, moved{"mesheditor-history-moved"}, archive{"mesheditor-history-archive"};
    std::map<int, State> expected;
    {
        Fixture f;
        auto &p = *f.P;
        expect(p.Begin(dir));
        f.R.Context.get<ViewportExtent>().Value = {64, 64};
        f.P->Settle();
        const auto record = [&] {
            const auto node = p.History.Present;
            if (expected.contains(node)) expected.at(node).Check(f);
            else expected.emplace(node, State{f});
            return node;
        };
        record();
        f.Do(action::UpdateOf<&ViewportDisplay::ShowOverlays>(action::OnViewport{}, false));
        record();
        // Verify entity-generation restoration when component values match.
        f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
        const auto first = FindActiveEntity(f.R);
        record();
        f.Do(action::object::Delete{});
        record();
        f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
        const auto reused = FindActiveEntity(f.R);
        expect(first != reused && state::Index(first) == state::Index(reused));
        record();
        f.Do(action::object::Delete{});
        record();
        const bool mesh_edit = std::string_view{sample} == "Sphere";
        if (mesh_edit) {
            f.Do(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = 64, .Stacks = 32}, std::make_unique<MeshInstanceCreateInfo>()});
        } else {
            const auto input = std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models" / sample / "glTF";
            std::filesystem::copy(input, source.Path, std::filesystem::copy_options::recursive);
            f.Do(action::io::Load{source.Path / (std::string{sample} + ".gltf")});
        }
        const auto base = record();
        f.Do(action::selection::SelectAll{});
        record();
        f.Do(action::object::Duplicate{});
        record();
        f.Do(action::object::Delete{});
        const auto deleted = record();
        f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
        const auto continued = record();
        // Repeat an allocating action from restored state, including after eviction.
        p.Navigate(0);
        p.History.Evict(0);
        p.Navigate(deleted);
        expect(f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()}) == continued);
        record();
        p.Navigate(base);
        if (mesh_edit && Render) {
            f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
            record();
            // Test sparse and dense GPU writes.
            for (bool dense : {false, true}) {
                if (dense) f.Do(action::selection::SelectAll{});
                else {
                    f.Image();
                    f.Do(action::selection::ApplyEditElementClick{{32, 32}, false, std::make_unique<RenderView>(f.R.Context.get<const GpuBuffers>().FrameView)});
                    const auto &meshes = f.R.Context.get<const MeshStore>();
                    const auto id = f.R.get<const MeshHandle>(GetActiveMeshEntity(f.R)).StoreId;
                    expect(meshes.GetSelectionSummary(id).SelectedVertexCount == 1);
                }
                const auto selected = record();
                const auto count = p.History.Nodes.size();
                for (int i = 1; i <= 3; ++i) {
                    f.Stage(action::view::TransformElements{{.P = vec3{0.4f, 0.3f, 0.2f} * (float(i) / 3)}});
                    expect(p.History.Nodes.size() == count);
                }
                f.Finish();
                expect(p.History.Nodes.size() == count + 1);
                expect(p.History.MaterializeLive() != expected.at(selected).Persistent);
                if (dense) expect(f.Image() != expected.at(selected).Image);
                record();
                f.Stage(action::view::TransformElements{{.P = {2.f, 0.f, 0.f}}});
                p.CancelGesture();
                expect(!p.HasStaged());
                record();
            }
        } else {
            f.Do(action::timeline::SetFrame{15});
            record();
            f.Do(action::selection::SelectAll{});
            record();
            const auto count = p.History.Nodes.size();
            for (int i = 1; i <= 3; ++i) f.Stage(action::UpdateOf<&Transform::P>(action::OnSelectedDelta{}, vec3{float(i), 0, 0}));
            expect(p.History.Nodes.size() == count);
            f.Finish();
            expect(p.History.Nodes.size() == count + 1);
            record();
            f.Stage(action::UpdateOf<&Transform::P>(action::OnSelectedDelta{}, vec3{2, 0, 0}));
            p.CancelGesture();
            expect(!p.HasStaged());
            record();
        }
        p.Undo();
        expected.at(p.History.Present).Check(f);
        p.Redo();
        expected.at(p.History.Present).Check(f);
        expect(p.Save());
        for (bool cold : {false, true}) {
            for (const auto &[node, state] : expected) {
                if (cold) {
                    p.Navigate(0);
                    p.History.Evict(0);
                }
                p.Navigate(node);
                expect(p.History.Present == node);
                state.Check(f);
                const auto error = p.History.ValidateReplay(node);
                if (!error.empty()) std::printf("node %d replay: %s\n", node, error.c_str());
                expect(error.empty());
                state.Check(f);
            }
        }
        expect(p.Save());
    }
    // Verify that the archive includes external files for every branch.
    expect(Compress(dir.Path, archive.Path / "history.project"));
    std::filesystem::remove_all(source.Path);
    std::filesystem::remove_all(dir.Path);
    expect(Decompress(archive.Path / "history.project", moved.Path));
    Fixture f;
    expect(f.P->Open(moved));
    f.R.Context.get<ViewportExtent>().Value = {64, 64};
    f.P->Settle();
    expected.at(f.P->History.Present).Check(f);
    for (const auto &[node, state] : expected) {
        f.P->Navigate(node);
        state.Check(f);
        expect(f.P->Replay());
        state.Check(f);
    }
    const TestDir named{"mesheditor-named-project"}, copy{"mesheditor-copied-project"};
    const auto before_save = State{f};
    expect(f.P->SaveAs(named));
    expect(!std::filesystem::exists(moved.Path));
    before_save.Check(f);
    const auto saved = File::Read(f.P->SavedPath).value();
    f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
    expect(File::Read(f.P->SavedPath).value() == saved);
    const auto edited = State{f};
    std::filesystem::remove(copy.Path);
    expect(f.P->SaveAs(copy));
    edited.Check(f);
    expect(std::filesystem::exists(named.Path / "working/tree.log"));
    expect(File::Read(named.Path / "Saved.project").value() == saved);
    {
        Fixture other;
        expect(!other.P->Open(copy.Path / "working"));
        expect(!other.R.Context.get<action::Errors>().Messages.empty());
    }
    {
        const File::DirectoryLock lock{named.Path / "working"};
        expect(!f.P->SaveAs(named));
        expect(!f.R.Context.get<action::Errors>().Messages.empty());
        f.R.Context.get<action::Errors>().Messages.clear();
        expect(File::Read(named.Path / "Saved.project").value() == saved);
    }
    for (const auto &invalid : {archive.Path, copy.Path / "working/nested", copy.Path.parent_path()}) {
        expect(!f.P->SaveAs(invalid));
        expect(!f.R.Context.get<action::Errors>().Messages.empty());
        f.R.Context.get<action::Errors>().Messages.clear();
        edited.Check(f);
    }
    std::ofstream{named.Path / "working/obsolete"} << "old project";
    expect(f.P->SaveAs(named));
    expect(f.P->SavedPath == named.Path / "Saved.project");
    expect(!std::filesystem::exists(named.Path / "working/obsolete"));
    expect(std::filesystem::exists(copy.Path / "working/tree.log"));
    edited.Check(f);
    expect(f.P->SaveAs(named));
    const auto saved_node = f.P->History.Present;
    const auto redo_node = f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
    const auto redo_state = f.P->History.MaterializeLive();
    f.P->Undo();
    const std::vector<std::byte> saved_workspace{std::byte{1}, std::byte{2}};
    expect(f.P->SaveArchive(f.P->SavedPath, saved_workspace));
    f.P->Redo();
    const auto later_node = f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
    const auto later_state = f.P->History.MaterializeLive();
    f.P->Navigate(saved_node);
    const auto branch_node = f.Do(action::UpdateOf<&ViewportDisplay::ShowGrid>(action::OnViewport{}, false));
    const auto branch_state = f.P->History.MaterializeLive();
    const auto retained_nodes = f.P->History.Nodes.size();
    expect(f.P->Save());
    expect(f.P->New(dir));
    expect(f.P->Open(named.Path / "working", named.Path / "Saved.project"));
    expect(f.P->History.Present == saved_node && f.P->History.Nodes.size() == retained_nodes);
    expect(f.P->RestoredWorkspace == saved_workspace);
    expect(f.P->History.MaterializeLive() == edited.Persistent);
    f.P->History.Evict(0);
    for (const auto &[node, state] : {std::pair{redo_node, &redo_state}, {later_node, &later_state}, {branch_node, &branch_state}}) {
        f.P->Navigate(node);
        expect(f.P->History.MaterializeLive() == *state);
        expect(f.P->Replay());
        expect(f.P->History.MaterializeLive() == *state);
        f.Audit();
    }
    expect(f.P->RevertSaved());
    expect(f.P->History.Nodes.size() == retained_nodes);
    expect(f.P->RestoredWorkspace == saved_workspace);
    edited.Check(f);
    f.P->Redo();
    expect(f.P->History.MaterializeLive() == branch_state);
    f.P->Undo();
    edited.Check(f);
    expect(f.P->Replay());
    edited.Check(f);
    expect(f.P->SaveArchive(f.P->SavedPath, saved_workspace));
    const auto archive_bytes = File::Read(f.P->SavedPath).value();
    expect(bool(File::WriteAtomic(f.P->SavedPath, std::span{archive_bytes}.first(1))));
    expect(!f.P->RevertSaved());
    f.R.Context.get<action::Errors>().Messages.clear();
    edited.Check(f);
    expect(bool(File::WriteAtomic(f.P->SavedPath, archive_bytes)));
    f.P->Navigate(branch_node);
    expect(f.P->ClearHistory());
    expect(f.P->History.Nodes.size() == 2 && f.P->History.Present == 1);
    expect(!f.P->History.Nodes[0].Hot);
    expect(f.P->History.MaterializeLive() == branch_state);
    expect(File::Read(f.P->SavedPath).value() == archive_bytes);
    f.Do(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>()});
    const auto after_clear_edit = f.P->History.MaterializeLive();
    expect(f.P->New(dir));
    expect(f.P->Open(named.Path / "working", named.Path / "Saved.project"));
    expect(f.P->History.Present == 0 && f.P->History.Nodes.size() == 3);
    expect(f.P->RestoredWorkspace == saved_workspace);
    expect(f.P->History.MaterializeLive() == edited.Persistent);
    for (const auto &state : {branch_state, after_clear_edit}) {
        f.P->Redo();
        expect(f.P->History.MaterializeLive() == state);
        expect(f.P->Replay());
        expect(f.P->History.MaterializeLive() == state);
        f.Audit();
    }
    expect(f.P->RevertSaved());
    const auto before_clear = State{f};
    expect(f.P->ClearHistory());
    expect(f.P->History.Nodes.size() == 1);
    before_clear.Check(f);
    expect(f.P->Open(named.Path / "working", named.Path / "Saved.project"));
    expect(f.P->History.Present == 0 && f.P->History.Nodes.size() == 1);
    before_clear.Check(f);
    expect(f.P->New(dir));
    expect(!f.P->HasStaged() && f.P->History.Nodes.size() == 1);
    f.Audit();
}
} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && std::string_view{argv[1]} == "--no-render") Render = false;
    else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [--no-render]\n", argv[0]);
        return 1;
    }
    if (!Render) std::puts("Skipping rendered-image comparisons, GPU picking, and mesh-edit gestures (--no-render).");
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    {
        const TestDir sessions{"mesheditor-session-retention"};
        Paths::Init(MESHEDITOR_BUILD_DIR, sessions);
        std::vector<std::filesystem::path> directories;
        std::vector<File::DirectoryLock> locks;
        for (int i = 0; i < 12; ++i) {
            directories.push_back(project::ReserveRestoreSession());
            locks.emplace_back(directories.back());
            expect(bool(locks.back()));
            std::ofstream{directories.back() / "tree.log"} << i;
        }
        expect(project::ListRestoreSessions().empty());
        for (int i = 0; i < 12; ++i) {
            int owner = -1;
            std::ifstream{directories[i] / "tree.log"} >> owner;
            expect(owner == i);
        }
        locks.clear();
        expect(project::ListRestoreSessions().size() == 12);
        project::ReserveRestoreSession();
        expect(project::ListRestoreSessions().size() <= 5);
        Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    }
    TestNativeStateHistory();
    if (Render) TestPickingIdentity();
    for (const char *sample : {"Sphere", "SimpleSkin", "SimpleMorph", "BoxTextured"}) TestProject(sample);
    return RunSuites();
}
