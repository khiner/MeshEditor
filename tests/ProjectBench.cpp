#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "editor/Engine.h"
#include "mesh/MeshStore.h"
#include "metal/Buffer.h"
#include "project/BufferHistory.h"
#include "project/Project.h"
#include "scene/Entity.h"
#include "viewport/Viewport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

using boost::ut::expect;

namespace {
double Ms(auto fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

constexpr uint64_t MiB = 1u << 20;
double Median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
void BenchScene(uint32_t slices, bool render) {
    const TestDir dir{"mesheditor-project-scene-bench"};
    Paths::SetProject(dir);
    Engine engine{true};
    auto &r = engine.R;
    auto &p = *engine.P;
    const auto viewport = engine.Viewport;
    expect(p.Begin(dir));
    p.Do(action::MakeAction(action::view::SetExtent{{128, 128}}));
    p.Do(action::MakeAction(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = slices, .Stacks = slices / 2}, std::make_unique<MeshInstanceCreateInfo>()}));
    p.Do(action::MakeAction(action::view::SetInteractionMode{InteractionMode::Edit}));
    p.Do(action::MakeAction(action::selection::SelectAll{}));
    const auto finish_frame = [&] {
        if (render) {
            SubmitViewport(r, viewport);
            WaitForRender(r);
        }
    };
    finish_frame();
    const auto base = p.History.Present;
    const auto mesh = GetMesh(r, GetActiveMeshEntity(r));
    const auto vertices = mesh.VertexCount();
    std::vector<int> nodes;
    std::vector<double> edits, hot, cold;
    for (int i = 0; i < 30; ++i) {
        edits.push_back(Ms([&] {
            auto move = std::make_unique<PendingTransform>();
            move->Delta.P.x = 0.001f;
            action::EmitStaged(action::view::DragGizmoMeshEdit{std::move(move)});
            p.Frame(action::Drain());
            action::Commit();
            p.Frame(action::Drain());
            finish_frame();
        }));
        nodes.push_back(p.History.Present);
        expect(nodes.back() != (i ? nodes[i - 1] : base));
    }
    expect(p.Save());
    const auto navigate = [&](int node) {
        p.Navigate(node);
        finish_frame();
    };
    for (int i = 0; i < 30; ++i) hot.push_back(Ms([&] { navigate(nodes[i % 2 ? 29 : 28]); }));
    p.Navigate(base);
    p.History.Evict(0);
    p.Navigate(nodes.back());
    for (int i = 28; i >= 0; --i) {
        expect(!p.History.Nodes[nodes[i]].Hot);
        cold.push_back(Ms([&] { navigate(nodes[i]); }));
        expect(p.History.Present == nodes[i]);
    }
    std::string why;
    expect(p.Audit(why));
    expect(p.History.ValidateReplay(nodes.back()).empty());
    expect(p.History.TakeIntegrityError().empty());
    std::printf("%u vertices: dense drag + commit %.3f ms, hot %.3f ms, cold %.3f ms (medians, %s render)\n", vertices, Median(edits), Median(hot), Median(cold), render ? "including" : "excluding");
    expect(p.Close());
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
        // Initialize distinct nonzero pages to measure scaling with live data size.
        for (uint64_t page = 0; page < buffer.UsedSize / 4096; ++page) {
            const auto value = page + 1;
            std::memcpy(initial.data() + page * 4096, &value, sizeof(value));
        }
        store::History history;
        buffer.Track(history, "buffer");
        expect(history.Begin(dir));
        expect(history.Save());
        const auto before = history.LogBytes();
        std::vector<double> commits, hot, cold;
        for (uint64_t i = 0; i < 30; ++i) {
            const uint64_t value = 1000000 + i;
            commits.push_back(Ms([&] {
                buffer.Update(as_bytes(value));
                history.Commit("eight bytes", {});
            }));
        }
        expect(history.Save());
        const auto appended = history.LogBytes() - before;
        for (int i = 0; i < 30; ++i) {
            hot.push_back(Ms([&] { history.Navigate(i % 2 ? 30 : 29); }));
        }
        history.Navigate(0);
        history.Evict(0);
        history.Navigate(30);
        for (int node = 29; node > 0; --node) {
            expect(!history.Nodes[node].Hot);
            cold.push_back(Ms([&] { history.Navigate(node); }));
            expect(history.Present == node);
        }
        std::string why;
        expect(history.Audit(why));
        expect(history.TakeIntegrityError().empty());
        std::printf("%2llu MiB: eight-byte write + commit %.4f ms, hot %.4f ms, cold %.4f ms (medians), %.0f appended bytes/edit\n", (unsigned long long)mib, Median(commits), Median(hot), Median(cold), double(appended) / 30);
        history.Navigate(0);
        history.Evict(0);
        const auto stats = history.Stats();
        std::printf("        after eviction: %llu owned bytes, %.3f MiB retained, %.3f MiB peak pending writes\n", (unsigned long long)stats.OwnedBytes, double(stats.RetainedBytes()) / MiB, double(stats.PeakPendingBytes) / MiB);
        expect(history.Close());
    }
    for (const bool render : {false, true}) {
        BenchScene(64, render);
        BenchScene(256, render);
    }
    return RunSuites();
}
