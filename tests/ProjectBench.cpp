#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Build.h"
#include "editor/Engine.h"
#include "mesh/MeshStore.h"
#include "procedure/Procedure.h"
#include "project/Recording.h"
#include "scene/Entity.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"

using boost::ut::expect;

namespace {
double Ms(auto fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// An exception reports before engine teardown, since teardown waits for every GPU command and a command that never completes blocks later output.
bool RunBench(const char *name, auto &&bench) {
    const TestDir dir{name};
    Paths::SetProject(dir);
    Engine engine{true};
    try {
        auto &p = *engine.P;
        expect(p.New(dir, true));
        bench(engine.R, p);
        expect(p.Close());
        return true;
    } catch (const std::exception &e) {
        std::println(stderr, "{}", e.what());
        return false;
    }
}

constexpr uint64_t MiB = 1u << 20;
double Median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
// Times hot navigations alternating between the last two nodes, then cold navigations from the last node to each earlier one after evicting from the root.
std::pair<std::vector<double>, std::vector<double>> Navigations(store::History &history, uint32_t root, std::span<const uint32_t> nodes, auto &&navigate) {
    std::vector<double> hot, cold;
    const auto last = nodes.size() - 1u;
    for (size_t i = 0; i < nodes.size(); ++i) hot.push_back(Ms([&] { navigate(nodes[i % 2 ? last : last - 1u]); }));
    navigate(root);
    history.Evict(0);
    navigate(nodes[last]);
    for (size_t i = last; i-- > 0;) {
        expect(!history.Nodes[nodes[i]].Hot);
        cold.push_back(Ms([&] { navigate(nodes[i]); }));
        expect(history.Present == nodes[i]);
    }
    return {std::move(hot), std::move(cold)};
}
bool BenchScene(uint32_t slices, bool render) {
    return RunBench("mesheditor-project-scene-bench", [&](state::Registry &r, project::Project &p) {
        r.replace<ViewportPanel>(FirstViewport(r), ViewportPanel{.Extent = {128, 128}});
        p.Settle();
        project::Do(p, action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
        project::Do(p, action::MakeAction(action::view::SetInteractionMode{InteractionMode::Edit}));
        project::Do(p, action::MakeAction(action::selection::SelectAll{}));
        const auto finish_frame = [&] {
            if (render) {
                SubmitViewports(r);
                WaitForRender(r);
            }
        };
        finish_frame();
        const auto base = *p.History.Present;
        const auto mesh = GetMesh(r, GetActiveMeshEntity(r));
        const auto vertices = mesh.VertexCount();
        std::vector<uint32_t> nodes;
        std::vector<double> edits;
        for (int i = 0; i < 30; ++i) {
            edits.push_back(Ms([&] {
                action::Emit(action::view::TransformElements{{.P = {0.001f, 0.f, 0.f}}}, action::Phase::Stage);
                p.Frame();
                action::Commit();
                p.Frame();
                finish_frame();
            }));
            nodes.push_back(*p.History.Present);
            expect(nodes.back() != (i ? nodes[i - 1] : base));
        }
        expect(p.Save());
        const auto [hot, cold] = Navigations(p.History, base, nodes, [&](uint32_t node) {
            p.Navigate(node);
            finish_frame();
        });
        std::string why;
        expect(p.Audit(why));
        p.Navigate(nodes.back());
        expect(p.History.ValidateReplay().empty());
        expect(p.History.TakeIntegrityError().empty());
        std::println("{} vertices: dense drag + commit {:.3f} ms, hot {:.3f} ms, cold {:.3f} ms (medians, {} render)", vertices, Median(edits), Median(hot), Median(cold), render ? "including" : "excluding");
    });
}
// The history bytes a node append and a parameter change copy on a procedure of the node count.
bool BenchProcedureHistory(uint32_t node_count) {
    return RunBench("mesheditor-project-procedure-bench", [&](state::Registry &r, project::Project &p) {
        p.Settle();
        project::Do(p, action::MakeAction(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>()}));
        const auto mesh_entity = GetActiveMeshEntity(r);
        const auto add = [&] { project::Do(p, action::MakeAction(action::nodes::AddNode{mesh_entity, procedure::KindOf<procedure::FloatValue>, {}})); };
        for (uint32_t n = 1u; n < node_count; ++n) add();
        const auto owned = [&] { return p.History.Stats().OwnedBytes; };
        std::vector<double> appends, changes;
        for (int i = 0; i < 30; ++i) {
            const auto before = owned();
            add();
            appends.push_back(double(owned() - before));
        }
        const auto node = r.get<const procedure::GeometryProcedure>(mesh_entity).Last;
        for (int i = 0; i < 30; ++i) {
            const auto before = owned();
            project::Do(p, action::MakeAction(action::UpdateOf<&procedure::FloatValue::Value>(node, float(i + 1))));
            changes.push_back(double(owned() - before));
        }
        std::println("{}-node procedure: {:.0f} owned history bytes per node append, {:.0f} per parameter change (medians)", node_count, Median(appends), Median(changes));
    });
}
// The time of a gesture step that changes a parameter of the first node or of the last node of a chain of the kinds over a UV sphere, and of the gesture's commit.
// Each kind is SmoothVertices, ShrinkFatten or Poke.
bool BenchChainCook(std::string_view name, uint32_t slices, std::span<const uint16_t> kinds) {
    return RunBench("mesheditor-project-chain-bench", [&](state::Registry &r, project::Project &p) {
        using namespace action::mesh;
        p.Settle();
        project::Do(p, action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
        const auto mesh_entity = GetActiveMeshEntity(r);
        std::vector<state::Entity> nodes;
        for (const auto kind : kinds) {
            const auto display = r.get<const procedure::GeometryProcedure>(mesh_entity).DisplayNode;
            const auto result = *procedure::ResultSocket(r, display);
            project::Do(p, action::MakeAction(action::nodes::AddNode{mesh_entity, kind, {}}));
            nodes.push_back(r.get<const procedure::GeometryProcedure>(mesh_entity).Last);
            project::Do(p, action::MakeAction(action::nodes::Link{mesh_entity, display, result, nodes.back(), 0u}));
            project::Do(p, action::MakeAction(action::nodes::SetDisplay{mesh_entity, nodes.back()}));
        }
        const auto stage = [&](state::Entity node, float value) {
            const auto kind = r.get<const procedure::ProcedureNode>(node).Kind;
            if (kind == procedure::KindOf<SmoothVertices>) action::Emit(action::UpdateOf<&SmoothVertices::Factor>(node, value), action::Phase::Stage);
            else if (kind == procedure::KindOf<ShrinkFatten>) action::Emit(action::UpdateOf<&ShrinkFatten::Distance>(node, value), action::Phase::Stage);
            else action::Emit(action::UpdateOf<&Poke::Offset>(node, value), action::Phase::Stage);
            p.Frame();
        };
        // The medians of a gesture's steps and of its commit, over three gestures.
        const auto drag = [&](state::Entity node) {
            std::vector<double> steps, commits;
            for (int gesture = 0; gesture < 3; ++gesture) {
                for (int i = 0; i < 3; ++i) stage(node, .01f * float(gesture + i + 1));
                for (int i = 0; i < 8; ++i) steps.push_back(Ms([&] { stage(node, .05f + .01f * float(gesture + i)); }));
                commits.push_back(Ms([&] {
                    action::Commit();
                    p.Frame();
                }));
            }
            return std::pair{Median(steps), Median(commits)};
        };
        const auto last = drag(nodes.back()), first = drag(nodes.front());
        std::println("{} chain of {} nodes, {} faces: step on the first node {:.3f} ms and commit {:.3f} ms, step on the last node {:.3f} ms and commit {:.3f} ms (medians)", name, kinds.size(), GetMesh(r, mesh_entity).FaceCount(), first.first, first.second, last.first, last.second);
    });
}
} // namespace

int main() {
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    mtl::Context ctx;
    mtl::BindlessSet slots{ctx};
    mtl::BufferContext buffers{ctx, slots};
    for (const uint64_t mib : {1, 16, 64}) {
        const TestDir dir{"mesheditor-project-bench"};
        mtl::Buffer buffer{buffers, 0};
        buffer.SetUsedSize(mib * MiB);
        const auto initial = buffer.GetMutableRange(0, buffer.UsedSize);
        std::ranges::fill(initial, std::byte{});
        // Distinct nonzero pages make the timing scale with live data size.
        for (uint64_t page = 0; page < buffer.UsedSize / 4096; ++page) {
            const auto value = page + 1;
            std::memcpy(initial.data() + page * 4096, &value, sizeof(value));
        }
        store::History history;
        buffer.Track(history, "buffer");
        expect(history.Begin(dir));
        expect(history.Save());
        const auto before = history.LogBytes();
        std::vector<double> commits;
        for (uint64_t i = 0; i < 30; ++i) {
            const uint64_t value = 1000000 + i;
            commits.push_back(Ms([&] {
                buffer.Update(BytesOf(value));
                history.Commit("eight bytes", {});
            }));
        }
        expect(history.Save());
        const auto appended = history.LogBytes() - before;
        const auto nodes = std::views::iota(1u, 31u) | std::ranges::to<std::vector>();
        const auto [hot, cold] = Navigations(history, 0u, nodes, [&](uint32_t node) { history.Navigate(node); });
        std::string why;
        expect(history.Audit(why));
        expect(history.TakeIntegrityError().empty());
        std::println("{:2} MiB: eight-byte write + commit {:.4f} ms, hot {:.4f} ms, cold {:.4f} ms (medians), {:.0f} appended bytes/edit", mib, Median(commits), Median(hot), Median(cold), double(appended) / 30);
        history.Navigate(0);
        history.Evict(0);
        const auto stats = history.Stats();
        std::println("        after eviction: {} owned bytes, {:.3f} MiB shared nodes", stats.OwnedBytes, double(stats.SharedNodeBytes) / MiB);
        expect(history.Close());
    }
    for (const uint32_t node_count : {20u, 200u}) {
        if (!BenchProcedureHistory(node_count)) return 1;
    }
    {
        using procedure::KindOf;
        const auto smooth = KindOf<action::mesh::SmoothVertices>, fatten = KindOf<action::mesh::ShrinkFatten>, poke = KindOf<action::mesh::Poke>;
        const std::array positions{smooth, fatten, smooth, fatten, smooth, fatten, smooth, fatten};
        const std::array topology{poke, poke, poke};
        const std::array mixed{smooth, poke, fatten, smooth, poke, fatten};
        if (!BenchChainCook("position", 40u, positions) || !BenchChainCook("topology", 16u, topology) || !BenchChainCook("mixed", 16u, mixed)) return 1;
        if (!BenchChainCook("position", 700u, positions) || !BenchChainCook("topology", 128u, topology)) return 1;
    }
    for (const bool render : {false, true}) {
        if (!BenchScene(64, render) || !BenchScene(256, render)) return 1;
    }
    return RunSuites();
}
