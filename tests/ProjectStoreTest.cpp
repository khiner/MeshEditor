#include "project/store/History.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include "RunSuites.h"
#include "TestPaths.h"

#include <csignal>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sys/resource.h>

using boost::ut::expect;

namespace {
using namespace store;

using Model = std::map<uint64_t, std::vector<std::byte>>;

// Absent slots and variable-size values over a map, driving the trie directly.
struct MapOwner {
    Model Values;
    uint64_t Len{};
    LiveTrie Trie{3};

    uint64_t Length() const { return Len; }
    bool Present(uint64_t s) const { return Values.contains(s); }
    std::span<const std::byte> Read(uint64_t s) const { return Values.at(s); }
    std::span<const std::byte> Encode(const Blob &value) const { return value.View(); }

    void Write(uint64_t s) {
        if (Trie.Uncaptured(s)) Trie.Capture(s, Present(s) ? std::optional{CopyBlob(Read(s))} : std::nullopt);
        Trie.MarkDirty(s, 1);
    }
    void Settle() {
        for (const auto s : Trie.Dirty()) Trie.Rehash(s, Present(s) ? std::optional{Read(s)} : std::nullopt);
        Trie.ClearDirty();
    }
    bool Restore(const Version &v) {
        Settle();
        auto plan = Trie.PlanRestore(v);
        for (auto &c : plan.Changes) {
            const bool present = Present(c.Slot);
            if (c.Erase) {
                if (!present) {
                    c.Unchanged = true;
                    continue;
                }
                c.Old = CopyBlob(Read(c.Slot));
                c.WasPresent = true;
                Values.erase(c.Slot);
                continue;
            }
            if (present && c.MaybeEqual && Unchanged(Read(c.Slot), c.Incoming.View())) {
                c.Unchanged = true;
                continue;
            }
            if (present) {
                c.Old = CopyBlob(Read(c.Slot));
                c.WasPresent = true;
            }
            Values[c.Slot].assign(c.Incoming.Data, c.Incoming.Data + c.Incoming.Size);
            FreeBlob(c.Incoming);
        }
        Len = plan.Length;
        return Trie.CommitRestore(std::move(plan));
    }
    Model Materialize(const Version &v) {
        Model m;
        for (const auto &run : Trie.Materialize(v)) {
            if (run.Owned) m[run.Slot].assign(run.Owned->View().begin(), run.Owned->View().end());
            else {
                for (uint64_t s = run.Slot, end = run.Slot + run.Count; s < end; ++s)
                    if (Present(s)) m[s] = Values.at(s);
            }
        }
        return m;
    }
};

std::vector<std::byte> RandomBytes(std::mt19937 &rng, size_t n) {
    std::vector<std::byte> b(n);
    for (auto &x : b) x = std::byte(rng() & 0xff);
    return b;
}

// Reconstruct from live bytes to check cached hashes independently.
Hash128 RebuiltManifest(auto &owner) {
    std::map<uint64_t, Hash128> hashes;
    for (uint64_t s = 0, n = owner.Trie.SlotsFor(owner.Length()); s < n; ++s) {
        if (!owner.Present(s)) continue;
        const auto bytes = owner.Read(s);
        if (owner.Trie.PageBytes && IsZero(bytes)) continue;
        hashes[s] = HashBytes(bytes);
    }
    for (uint32_t level = 0; level < owner.Trie.Levels; ++level) {
        std::map<uint64_t, ManifestChildren> groups;
        for (const auto &[slot, hash] : hashes) groups[slot / Fanout][slot % Fanout] = hash;
        hashes.clear();
        for (const auto &[index, children] : groups) hashes[index] = ManifestRecord{level, children}.Hash();
    }
    return hashes.empty() ? Hash128{} : hashes.at(0);
}
Hash128 LiveManifest(auto &owner) {
    owner.Settle();
    return owner.Trie.ManifestRoot(owner.Length());
}
Version Pin(auto &owner) {
    owner.Settle();
    return owner.Trie.Pin(owner.Length());
}

void TestPoolTrieAgainstModel() {
    std::mt19937 rng{7};
    MapOwner live{.Len = 4096};
    auto &trie = live.Trie;
    expect(trie.Stats().Nodes == 1);
    expect(trie.Stats().AliasedNodes == 1);
    std::vector<std::pair<Version, Model>> pinned;
    pinned.push_back({Pin(live), live.Values});
    for (int step = 0; step < 400; ++step) {
        const int op = rng() % 10;
        if (op < 6) {
            // Concentrate writes to exercise shared leaves.
            const int n = 1 + rng() % 4;
            for (int i = 0; i < n; ++i) {
                const uint64_t slot = (rng() % 3 == 0) ? rng() % 4096 : rng() % 200;
                live.Write(slot);
                if (rng() % 5 == 0 && live.Values.contains(slot)) live.Values.erase(slot);
                else live.Values[slot] = RandomBytes(rng, rng() % 40);
            }
        } else if (op < 8) {
            pinned.push_back({Pin(live), live.Values});
        } else if (op == 8 && pinned.size() > 1) {
            const auto i = rng() % pinned.size();
            expect(live.Restore(pinned[i].first));
            expect(live.Values == pinned[i].second);
        } else if (pinned.size() > 2) {
            const auto i = 1 + rng() % (pinned.size() - 1);
            trie.Release(pinned[i].first);
            pinned.erase(pinned.begin() + i);
        }
        expect(LiveManifest(live) == RebuiltManifest(live));
        std::string why;
        std::vector<Version> versions;
        for (auto &[v, _] : pinned) versions.push_back(v);
        if (!trie.Check(why, versions)) {
            std::printf("invariant broken at step %d: %s\n", step, why.c_str());
            expect(false);
            return;
        }
        for (auto &[v, model] : pinned) expect(live.Materialize(v) == model);
    }
    for (auto &[v, model] : pinned) {
        expect(live.Restore(v));
        expect(live.Values == model);
    }
    for (auto &[v, _] : pinned) trie.Release(v);
    expect(trie.Stats().OwnedBytes == 0);
    expect(trie.Stats().OwnedSlots == 0);
}

void TestBufferAgainstModel() {
    std::mt19937 rng{11};
    Pages buf{64, 3}; // Use small pages to test edits across page boundaries.
    std::vector<std::pair<Version, std::vector<std::byte>>> pinned;
    auto image = [&] { return std::vector<std::byte>(buf.Data(), buf.Data() + buf.Length()); };
    buf.Resize(1000);
    const auto init = buf.Mutable(0, 1000);
    for (uint64_t i = 0; i < 1000; ++i) init[i] = std::byte(i);
    pinned.push_back({Pin(buf), image()});
    for (int step = 0; step < 300; ++step) {
        const int op = rng() % 8;
        if (op < 4 && buf.Length()) {
            const uint64_t off = rng() % buf.Length(), len = 1 + rng() % 200;
            const auto end = std::min<uint64_t>(off + len, buf.Length());
            const auto span = buf.Mutable(off, end - off);
            for (auto &b : span) b = std::byte(rng());
        } else if (op == 4) {
            const uint64_t len = rng() % 3000;
            buf.Resize(len);
            for (uint64_t i = 0; i < len; ++i)
                if (rng() % 7 == 0) buf.Mutable(i, 1)[0] = std::byte(rng());
        } else if (op == 5) {
            pinned.push_back({Pin(buf), image()});
        } else if (op == 6 && !pinned.empty()) {
            const auto i = rng() % pinned.size();
            expect(buf.Restore(pinned[i].first));
            expect(image() == pinned[i].second);
        } else if (pinned.size() > 1) {
            const auto i = rng() % pinned.size();
            buf.Trie.Release(pinned[i].first);
            pinned.erase(pinned.begin() + i);
        }
        expect(LiveManifest(buf) == RebuiltManifest(buf));
        std::string why;
        std::vector<Version> versions;
        for (auto &[v, _] : pinned) versions.push_back(v);
        if (!buf.Trie.Check(why, versions)) {
            std::printf("buffer invariant broken at step %d: %s\n", step, why.c_str());
            expect(false);
            return;
        }
    }
    for (auto &[v, img] : pinned) {
        expect(buf.Restore(v));
        expect(image() == img);
        buf.Trie.Release(v);
    }
    expect(buf.Trie.Stats().OwnedBytes == 0);
}

// Use a seed as the recorded action to reproduce buffer and record writes.
struct ToyApp {
    using Values = std::vector<std::vector<std::byte>>;
    Pages Buf{64, 3};
    Values Pool = Values(256);
    Records PoolRecords{Pool, 2};
    History H;

    int Replays{};

    ToyApp() {
        H.Track(Buf, "buf", 0);
        H.Track(PoolRecords, "pool", 1);
        H.Callbacks = {.Replay = [this](const std::vector<std::byte> &a) { Apply(a); ++Replays; }};
        Buf.Resize(512);
    }
    static std::vector<std::byte> Encode(uint32_t seed) {
        std::vector<std::byte> b(4);
        std::memcpy(b.data(), &seed, 4);
        return b;
    }
    void Apply(const std::vector<std::byte> &action) {
        uint32_t seed;
        std::memcpy(&seed, action.data(), 4);
        if (seed & 0x80000000u) {
            // Alternate constant fills to test inverse edits.
            Buf.Resize(std::max<uint64_t>(64, Buf.Length()));
            for (auto &b : Buf.Mutable(0, 64)) b = std::byte(seed & 0xff);
            return;
        }
        std::mt19937 rng{seed};
        if (seed % 3 == 0) Buf.Resize(std::array{0u, 65u, 65 * 64 + 7u, 8192u}[rng() % 4]);
        if (Buf.Length()) {
            const uint64_t off = rng() % Buf.Length(), len = std::min<uint64_t>(1 + rng() % 200, Buf.Length() - off);
            auto span = Buf.Mutable(off, len);
            for (auto &b : span) b = seed % 5 ? std::byte(rng()) : std::byte{};
        }
        const uint64_t slot = rng() % 256;
        PoolRecords.Write(slot, 1);
        if (rng() % 4 == 0) Pool[slot].clear();
        else Pool[slot] = RandomBytes(rng, rng() % 20);
    }
    void Step(uint32_t seed) {
        auto a = Encode(seed);
        Apply(a);
        H.Commit("step " + std::to_string(seed), std::move(a));
    }
    std::pair<std::vector<std::byte>, Values> State() const {
        return {std::vector<std::byte>(Buf.Data(), Buf.Data() + Buf.Length()), Pool};
    }
};

// Compare each node against full copies after cached restoration, cold restoration, reopening, and replay.
void TestHistoryAgainstModel() {
    const TestDir dir{"projectstore_model"};
    ToyApp app;
    expect(app.H.Begin(dir));
    std::map<int, decltype(app.State())> expected{{0, app.State()}};
    std::mt19937 rng{19};
    for (uint32_t seed = 1; seed <= 80; ++seed) {
        if (seed % 7 == 0) app.H.Navigate(rng() % app.H.Nodes.size());
        expect(app.State() == expected.at(app.H.Present));
        app.Step(seed);
        expected.emplace(app.H.Present, app.State());
        expect(app.State() == expected.at(app.H.Present));
        const auto &roots = app.H.Nodes[app.H.Present].Roots;
        expect(roots[0] == RebuiltManifest(app.Buf));
        expect(roots[1] == RebuiltManifest(app.PoolRecords));
    }
    expect(app.H.Save());
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 2) {
            expect(app.H.Close());
            expect(app.H.Open(dir));
            expect(app.State() == expected.at(app.H.Present));
        }
        for (const auto &[node, state] : expected) {
            if (pass) {
                app.H.Navigate(0);
                app.H.Evict(0);
                expect(app.H.Stats().HotNodes == 1);
                // Cold deltas must compare against actual live bytes, including uncommitted writes.
                app.Apply(ToyApp::Encode(1000));
            }
            const auto replays = app.Replays;
            app.H.Navigate(node);
            expect(app.Replays == replays);
            expect(app.H.Present == node);
            expect(app.State() == state);
            expect(app.H.ValidateReplay(node).empty());
            expect(app.State() == state);
            std::string why;
            expect(app.H.Audit(why));
            expect(app.H.TakeIntegrityError().empty());
        }
    }
    app.Apply(ToyApp::Encode(999));
    app.H.Revert();
    expect(app.State() == expected.at(app.H.Present));
    // Verify that the independent audit detects an uncaptured write.
    app.Buf.Resize(64);
    app.Buf.Settle();
    app.Buf.Storage[0] ^= std::byte{1};
    std::string why;
    expect(!app.H.Audit(why));
    app.Buf.Storage[0] ^= std::byte{1};
    expect(app.H.Audit(why));
}

void TestCommitIdentity() {
    const TestDir dir{"projectstore_identity"};
    ToyApp app;
    app.H.Begin(dir);
    constexpr uint32_t Fill1 = 0x80000000u | 1, Fill2 = 0x80000000u | 2;
    app.Step(Fill1);
    const int a = app.H.Present;
    app.Step(Fill2);
    const int b = app.H.Present;
    expect(a != b);
    const auto count = app.H.Nodes.size();
    app.Step(Fill2);
    expect(app.H.Present == b);
    expect(app.H.Nodes.size() == count);
    app.Step(Fill1);
    expect(app.H.Present == a);
    expect(app.H.Nodes.size() == count);
    app.Step(Fill2);
    expect(app.H.Present == b);
    expect(app.H.Nodes.size() == count);
    app.H.Navigate(a);
    app.H.Evict(0);
    expect(!app.H.Nodes[b].Hot);
    const int replays_before = app.Replays;
    app.Step(Fill2);
    expect(app.H.Present == b);
    expect(app.H.Nodes[b].Hot.has_value());
    expect(app.H.Nodes.size() == count);
    expect(app.Replays == replays_before);
    expect(app.H.Save());
    expect(app.H.Close());
    expect(app.H.Open(dir));
    expect(!app.H.Nodes[a].Hot);
    app.Step(Fill1);
    expect(app.H.Present == a);
    expect(app.H.Nodes.size() == count);
    expect(app.Replays == replays_before);
    app.H.Navigate(b);
    app.H.Navigate(a);
    std::string why;
    expect(app.H.Audit(why));
    if (!why.empty()) std::printf("  %s\n", why.c_str());
    app.H.Close();
}

void TestColdLoadFailure() {
    const TestDir dir{"projectstore_cold_failure"};
    Pages a{64}, b{64};
    History h;
    h.Track(a, "a", 0);
    h.Track(b, "b", 0);
    a.Resize(64);
    b.Resize(64);
    expect(h.Begin(dir));
    a.Mutable(0, 1)[0] = std::byte{11};
    b.Mutable(0, 1)[0] = std::byte{22};
    h.Commit("both", {});
    expect(h.Save());
    h.Navigate(0);
    h.Evict(0);
    const auto before = h.MaterializeLive();
    // Corrupt the second track's leaf and verify that loading changes neither track.
    const auto write_leaf = [&](char value) {
        std::fstream out{dir.Path / "leaves.log", std::ios::binary | std::ios::in | std::ios::out};
        out.seekp(2 * (sizeof(Hash128) + sizeof(uint32_t)) + 64);
        out.write(&value, 1);
    };
    write_leaf(42);
    h.Navigate(1);
    expect(h.Present == 0);
    expect(h.MaterializeLive() == before);
    expect(!h.TakeIntegrityError().empty());
    std::string why;
    expect(h.Audit(why));
    write_leaf(22);
    h.Navigate(1);
    expect(h.Present == 1);
    expect(a.Data()[0] == std::byte{11});
    expect(b.Data()[0] == std::byte{22});
    expect(h.Audit(why));
}

void TestFormatMismatch() {
    const TestDir dir{"projectstore_format"};
    {
        Pages a{64}, b{64};
        History h;
        h.Track(a, "A", 0);
        h.Track(b, "B", 0);
        a.Resize(64);
        b.Resize(64);
        a.Mutable(0, 1)[0] = std::byte{11};
        b.Mutable(0, 1)[0] = std::byte{22};
        expect(h.Begin(dir));
    }
    const auto read = [&](const char *name) {
        std::ifstream in{dir.Path / name, std::ios::binary};
        return std::vector<char>{std::istreambuf_iterator<char>{in}, {}};
    };
    const auto tree = read("tree.log"), leaves = read("leaves.log"), nodes = read("nodes.log");
    for (int mismatch = 0; mismatch < 4; ++mismatch) {
        Pages a{mismatch == 2 ? 128u : 64u}, b{64};
        History h;
        if (mismatch == 0) {
            h.Track(b, "B", 0);
            h.Track(a, "A", 0);
        } else {
            h.Track(a, "A", mismatch == 1 ? 1 : 0);
            h.Track(b, "B", 0);
        }
        h.SchemaRevision = mismatch == 3 ? 1 : 0;
        expect(!h.Open(dir));
        expect(!h.TakeIntegrityError().empty());
        expect(a.Length() == 0 && b.Length() == 0);
        expect(read("tree.log") == tree);
        expect(read("leaves.log") == leaves);
        expect(read("nodes.log") == nodes);
    }
}

void TestLogOpenFailure() {
    const TestDir unwritable{"projectstore_log_open_failure"}, current{"projectstore_current"}, failed_write{"projectstore_failed_root"};
    std::filesystem::create_directories(unwritable.Path / "leaves.log");
    Pages buffer;
    History h;
    h.Track(buffer, "buffer", 0);
    expect(!h.Begin(unwritable));
    expect(!h.TakeIntegrityError().empty());
    buffer.Resize(4096);
    buffer.Mutable(0, 1)[0] = std::byte{11};
    expect(h.Begin(current));
    const auto before = h.MaterializeLive();
    expect(!h.Begin(unwritable));
    expect(h.Present == 0 && h.MaterializeLive() == before);
    expect(!h.TakeIntegrityError().empty());

    // Fail after the new streams open, while the candidate root is being written.
    rlimit original{};
    expect(getrlimit(RLIMIT_FSIZE, &original) == 0);
    const auto handler = std::signal(SIGXFSZ, SIG_IGN);
    const rlimit no_writes{0, original.rlim_max};
    expect(setrlimit(RLIMIT_FSIZE, &no_writes) == 0);
    const bool created = h.Begin(failed_write);
    expect(setrlimit(RLIMIT_FSIZE, &original) == 0);
    std::signal(SIGXFSZ, handler);
    expect(!created);
    expect(h.Present == 0 && h.MaterializeLive() == before);
    expect(!h.TakeIntegrityError().empty());
    expect(h.Save());
    expect(h.Close());
    expect(h.Open(current));
    expect(h.Present == 0 && h.MaterializeLive() == before);
}

// Truncate records used only by the final committed step.
void TestTornTailRecovery() {
    const TestDir dir{"projectstore_torn"};
    std::map<int, std::pair<std::vector<std::byte>, ToyApp::Values>> expected;
    int last_node = -1;
    {
        ToyApp app;
        app.H.Begin(dir);
        expected[0] = app.State();
        for (uint32_t s = 1; s <= 8; ++s) {
            app.Step(s);
            expected[app.H.Present] = app.State();
        }
        // Use unique data to make the final leaf record exclusive to this node.
        app.Step(0x80000000u | 42);
        last_node = app.H.Present;
        expected[last_node] = app.State();
        app.H.Save();
        app.H.Close();
    }
    const auto leaves_path = dir.Path / "leaves.log", nodes_path = dir.Path / "nodes.log", tree_path = dir.Path / "tree.log";
    // Discard appended garbage while preserving the tree.
    for (const auto &p : {leaves_path, nodes_path, tree_path}) {
        std::ofstream out{p, std::ios::binary | std::ios::app};
        out.write("garbage!garbage!", 16);
    }
    {
        ToyApp app;
        expect(app.H.Open(dir));
        expect(app.H.Nodes.size() == expected.size());
        for (const auto &[node, state] : expected) {
            app.H.Navigate(node);
            expect(app.State() == state);
        }
        expect(app.H.TakeIntegrityError().empty());
        app.H.Save();
        app.H.Close();
    }
    // Discard the incomplete leaf record and its node while preserving earlier nodes.
    std::filesystem::resize_file(leaves_path, std::filesystem::file_size(leaves_path) - 1);
    {
        ToyApp app;
        expect(app.H.Open(dir));
        expect(app.H.Nodes.size() == expected.size() - 1);
        const auto lost = expected.at(last_node);
        expected.erase(last_node);
        for (const auto &[node, state] : expected) {
            app.H.Navigate(node);
            expect(app.State() == state);
        }
        expect(app.H.TakeIntegrityError().empty());
        // Rewrite missing leaves when repeating the discarded step.
        app.Step(0x80000000u | 42);
        app.H.Navigate(0);
        app.H.Evict(0);
        app.H.Navigate(last_node);
        expect(app.H.Present == last_node);
        expect(app.State() == lost);
        expect(app.H.TakeIntegrityError().empty());
        expect(app.H.Save());
        expect(app.H.Close());
        expect(app.H.Open(dir));
        expect(app.H.Present == last_node);
        expect(app.State() == lost);
    }
}

void TestClearHistory() {
    for (const bool retain_saved : {false, true}) {
        const TestDir dir{"projectstore_clear"};
        ToyApp app;
        expect(app.H.Begin(dir));
        app.Step(1);
        const auto &saved_node = app.H.Nodes[app.H.Present];
        const HistoryPosition saved{-1, saved_node.Stamps, saved_node.Roots};
        const auto saved_state = app.State();
        const int baseline = retain_saved ? 1 : 0;
        if (retain_saved) app.H.Navigate(0);
        app.Step(2);
        expect(app.H.Save());
        const auto before_clear = std::filesystem::file_size(dir.Path / "tree.log");
        const auto state = app.State();
        const auto content_bytes = app.H.LogBytes();
        expect(app.H.Clear(retain_saved ? &saved : nullptr));
        expect(app.H.Nodes.size() == size_t(baseline + 1) && app.H.Present == baseline);
        expect(app.State() == state);
        expect(app.H.LogBytes() == content_bytes);
        expect(app.H.Stats().OwnedBytes == 0);
        if (retain_saved) expect(!app.H.Nodes[0].Hot);
        expect(app.H.Close());
        const auto complete_clear = std::filesystem::file_size(dir.Path / "tree.log");
        expect(complete_clear > before_clear);
        // An incomplete replacement root preserves the previous tree on reopen.
        std::filesystem::resize_file(dir.Path / "tree.log", complete_clear - 1);
        expect(app.H.Open(dir));
        expect(app.H.Nodes.size() == 3);
        expect(app.State() == state);
        if (retain_saved) {
            expect(app.H.Clear());
            expect(app.H.FindPosition(saved) == -1);
        }
        expect(app.H.Clear(retain_saved ? &saved : nullptr));
        app.Step(3);
        expect(app.H.Save());
        expect(app.H.Close());
        expect(app.H.Open(dir));
        expect(app.H.Nodes.size() == size_t(baseline + 2));
        expect(app.H.Nodes[baseline + 1].Label == "step 3");
        const auto leaf = app.H.MaterializeLive();
        const auto replays = app.Replays;
        expect(app.H.Replay(baseline + 1).empty());
        expect(app.Replays == replays + 1);
        expect(app.H.MaterializeLive() == leaf);
        app.H.Undo();
        expect(app.State() == state);
        if (retain_saved) {
            expect(app.H.FindPosition(saved) == 0);
            app.H.Undo();
            expect(app.State() == saved_state);
            expect(app.H.Replay(0).empty());
            app.H.Redo();
            app.H.Evict(0);
            expect(!app.H.Nodes[0].Hot);
            const auto replay_count = app.Replays;
            expect(app.H.Replay(baseline).empty());
            expect(app.H.ValidateReplay(baseline).empty());
            expect(app.Replays == replay_count);
            expect(app.State() == state);
        }
        app.H.Callbacks = {.Replay = [&](const auto &) { app.Apply(ToyApp::Encode(99)); }};
        expect(!app.H.Replay(baseline + 1).empty());
        expect(app.H.Present == baseline);
        expect(app.State() == state);
        app.H.Callbacks = {.Replay = [&](const auto &) {
            app.Apply(ToyApp::Encode(100));
            throw std::runtime_error("bad command");
        }};
        expect(app.H.Replay(baseline + 1) == "bad command");
        expect(app.H.Present == baseline);
        expect(app.State() == state);
    }
}
} // namespace

int main() {
    TestPoolTrieAgainstModel();
    TestBufferAgainstModel();
    TestHistoryAgainstModel();
    TestCommitIdentity();
    TestColdLoadFailure();
    TestFormatMismatch();
    TestLogOpenFailure();
    TestTornTailRecovery();
    TestClearHistory();
    return RunSuites();
}
