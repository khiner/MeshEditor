#include "project/store/History.h"
#include "project/store/Pages.h"
#include "project/store/Records.h"

#include "TestPaths.h"
#include <boost/ut.hpp>

#include <csignal>
#include <map>
#include <print>
#include <random>
#include <ranges>

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

// The manifest root rebuilt from the live bytes, independent of the cached hashes.
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
// Whether the live manifest matches one rebuilt from the live bytes and the trie holds its invariants over the pinned versions.
bool Consistent(auto &owner, const auto &pinned, int step) {
    expect(LiveManifest(owner) == RebuiltManifest(owner));
    std::string why;
    if (owner.Trie.Check(why, pinned | std::views::keys | std::ranges::to<std::vector>())) return true;
    std::println("invariant broken at step {}: {}", step, why);
    expect(false);
    return false;
}

void TestRecordPagesAcrossResize() {
    std::vector<std::optional<std::string>> values(35);
    for (uint32_t i = 0; i < values.size(); ++i) values[i] = std::string(100u + i, char('a' + i % 26u));
    Records records{values, 3};
    records.Trie.CollectChanged = true;
    std::vector<std::pair<Version, decltype(values)>> versions;
    versions.emplace_back(Pin(records), values);
    records.Write(31, 34);
    values.resize(65);
    for (uint32_t i = 31; i < values.size(); ++i) values[i] = std::to_string(i);
    versions.emplace_back(Pin(records), values);
    records.Write(15, 42);
    for (uint32_t i = 15; i < 57u; ++i) values[i].reset();
    versions.emplace_back(Pin(records), values);
    records.Write(0, values.size());
    values.resize(1);
    versions.emplace_back(Pin(records), values);
    for (uint32_t round = 0; round < 3u; ++round) {
        for (auto &[version, expected] : versions) {
            expect(records.Restore(version));
            expect(values == expected);
            expect(LiveManifest(records) == RebuiltManifest(records));
            records.TakeChanged();
        }
    }
    for (auto &[version, _] : versions) records.Trie.Release(version);
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
            // Most writes land in the first 200 slots, which pinned versions share.
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
        if (!Consistent(live, pinned, step)) return;
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

// A slot captured alone skips recapture until a pin or a restore shares the present again.
void TestCaptureEpoch() {
    MapOwner live{.Len = 64};
    auto &trie = live.Trie;
    struct Fetch {
        MapOwner &Live;
        uint32_t Count{};
    } fetch{live};
    const auto write = [&](uint64_t slot, uint8_t value) {
        trie.Capture(slot, 1, &fetch, [](void *owner, uint64_t s) -> std::optional<Blob> {
            auto &f = *static_cast<Fetch *>(owner);
            ++f.Count;
            return f.Live.Present(s) ? std::optional{CopyBlob(f.Live.Read(s))} : std::nullopt;
        });
        trie.MarkDirty(slot, 1);
        live.Values[slot] = {std::byte{value}};
    };
    const auto check = [&](const std::vector<Version> &versions) {
        live.Settle();
        std::vector<std::pair<uint64_t, Hash128>> hashes;
        for (const auto &[slot, bytes] : live.Values) hashes.emplace_back(slot, HashBytes(bytes));
        std::string why;
        expect(trie.Check(why, versions));
        expect(trie.CheckHashes(why, hashes));
    };
    auto base = Pin(live);
    write(5, 1);
    write(5, 2);
    expect(fetch.Count == 1u);
    auto pinned = Pin(live);
    write(5, 3);
    expect(fetch.Count == 2u);
    check({base, pinned});
    expect(live.Restore(pinned));
    write(5, 4);
    expect(fetch.Count == 3u);
    check({base, pinned});
    expect(live.Materialize(base).empty());
    expect(live.Materialize(pinned) == Model{{5, {std::byte{2}}}});
    expect(live.Restore(base));
    expect(live.Values.empty());
    expect(live.Restore(pinned));
    expect(live.Values == Model{{5, {std::byte{2}}}});
    trie.Release(base);
    trie.Release(pinned);
}

void TestBufferAgainstModel() {
    std::mt19937 rng{11};
    Pages buf{64, 3}; // Small pages put edits across page boundaries.
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
        if (!Consistent(buf, pinned, step)) return;
    }
    for (auto &[v, img] : pinned) {
        expect(buf.Restore(v));
        expect(image() == img);
        buf.Trie.Release(v);
    }
    expect(buf.Trie.Stats().OwnedBytes == 0);
}

// An application whose recorded action is a seed that reproduces its buffer and record writes.
struct ToyApp {
    using Values = std::vector<std::vector<std::byte>>;
    Pages Buf{64, 3};
    Values Pool = Values(256);
    Records PoolRecords{Pool, 2};
    History H;

    int Replays{};

    void Reopen(const std::filesystem::path &dir) {
        expect(H.Save());
        expect(H.Close());
        expect(H.Open(dir));
    }
    ToyApp() {
        H.Track(Buf, "buf", 0);
        H.Track(PoolRecords, "pool", 1);
        H.Callbacks = {
            .Replay = [this](store::History &, std::span<const std::byte> a, std::span<const uint32_t>) -> std::expected<void, std::string> {
                if (!a.empty()) Apply(a);
                ++Replays;
                return {}; },
            .Reset = [this](const std::filesystem::path &) { Reset(512); },
        };
        Buf.Resize(512);
    }
    void Reset(uint64_t length) {
        Buf.Resize(0);
        Buf.Resize(length);
        PoolRecords.Write(0, Pool.size());
        std::ranges::fill(Pool, std::vector<std::byte>{});
    }
    static std::vector<std::byte> Encode(uint32_t seed) {
        std::vector<std::byte> b(4);
        std::memcpy(b.data(), &seed, 4);
        return b;
    }
    void Apply(std::span<const std::byte> action) {
        uint32_t seed;
        std::memcpy(&seed, action.data(), 4);
        if (seed & 0x80000000u) {
            // A constant fill can return the buffer to an earlier state, which exercises inverse edits.
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

void ExpectAudit(History &h) {
    std::string why;
    expect(h.Audit(why)) << why;
}

// Each node matches its full copy after cached restoration, cold restoration, reopening and replay.
// A commit adds a node exactly when no node adjacent to the present holds its state.
void TestHistoryAgainstModel() {
    const TestDir dir{"projectstore_model"};
    ToyApp app;
    expect(app.H.Begin(dir));
    std::map<uint32_t, decltype(app.State())> expected{{0, app.State()}};
    std::mt19937 rng{19};
    for (uint32_t seed = 1; seed <= 80; ++seed) {
        if (seed % 7 == 0) app.H.Navigate(rng() % app.H.Nodes.size());
        expect(app.State() == expected.at(*app.H.Present));
        const auto adjacent = [&] {
            const auto &present = app.H.Nodes[*app.H.Present];
            auto nodes = present.Children;
            nodes.push_back(*app.H.Present);
            if (present.Parent) nodes.push_back(*present.Parent);
            return nodes;
        }();
        const auto count = app.H.Nodes.size();
        app.Step(rng() % 3 == 0 ? 0x80000000u | (rng() % 3u) : seed);
        const auto state = app.State();
        const bool held = std::ranges::any_of(adjacent, [&](uint32_t node) { return expected.at(node) == state; });
        expect((app.H.Nodes.size() == count + 1u) == !held) << seed;
        expected.emplace(*app.H.Present, state);
        expect(app.State() == expected.at(*app.H.Present));
        const auto &roots = app.H.Nodes[*app.H.Present].Roots;
        expect(roots[0] == RebuiltManifest(app.Buf));
        expect(roots[1] == RebuiltManifest(app.PoolRecords));
    }
    expect(app.H.Save());
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 2) {
            expect(app.H.Close());
            expect(app.H.Open(dir));
            expect(app.State() == expected.at(*app.H.Present));
        }
        for (const auto &[node, state] : expected) {
            if (pass) {
                app.H.Navigate(0);
                app.H.Evict(0);
                expect(app.H.Stats().HotNodes == 1);
                // Cold deltas compare against the live bytes, including this uncommitted write.
                app.Apply(ToyApp::Encode(1000));
            }
            const auto replays = app.Replays;
            app.H.Navigate(node);
            expect(app.Replays == replays);
            expect(app.H.Present == node);
            expect(app.State() == state);
            expect(app.H.ValidateReplay().empty());
            expect(app.State() == state);
            ExpectAudit(app.H);
            expect(app.H.TakeIntegrityError().empty());
        }
    }
    app.Apply(ToyApp::Encode(999));
    app.H.Revert();
    expect(app.State() == expected.at(*app.H.Present));
    app.Buf.Resize(64);
    app.Buf.Settle();
    app.Buf.Storage[0] ^= std::byte{1};
    std::string why;
    expect(!app.H.Audit(why));
    app.Buf.Storage[0] ^= std::byte{1};
    expect(app.H.Audit(why));
}

// A commit whose state an adjacent cold node holds moves to that node without replaying, before and after reopening.
void TestCommitIdentity() {
    const TestDir dir{"projectstore_identity"};
    ToyApp app;
    app.H.Begin(dir);
    constexpr uint32_t Fill1 = 0x80000000u | 1, Fill2 = 0x80000000u | 2;
    app.Step(Fill1);
    const auto a = *app.H.Present;
    app.Step(Fill2);
    const auto b = *app.H.Present;
    const auto count = app.H.Nodes.size();
    app.H.Navigate(a);
    app.H.Evict(0);
    expect(!app.H.Nodes[b].Hot);
    const int replays_before = app.Replays;
    app.Step(Fill2);
    expect(app.H.Present == b);
    expect(app.H.Nodes[b].Hot.has_value());
    expect(app.H.Nodes.size() == count);
    expect(app.Replays == replays_before);
    app.Reopen(dir);
    expect(!app.H.Nodes[a].Hot);
    app.Step(Fill1);
    expect(app.H.Present == a);
    expect(app.H.Nodes.size() == count);
    expect(app.Replays == replays_before);
    app.H.Navigate(b);
    app.H.Navigate(a);
    ExpectAudit(app.H);
    app.H.Close();
}

// A replaced node keeps its identity and drops its descendants, in memory and after reopening.
void TestReplace() {
    const TestDir dir{"projectstore_replace"};
    ToyApp app;
    app.H.Begin(dir);
    app.Step(1);
    const auto a = *app.H.Present;
    app.Step(2);
    const auto b = *app.H.Present;
    app.Step(3);
    const auto c = *app.H.Present;
    app.Step(4);
    const auto d = *app.H.Present;
    app.H.Navigate(a);
    const auto a_state = app.State();
    app.Apply(ToyApp::Encode(5));
    const auto replaced = app.State();
    expect(app.H.Replace(b, ToyApp::Encode(5)) == b);
    expect(app.H.Present == b);
    expect(app.H.Nodes.size() == 5u);
    expect(app.H.Nodes[b].Children.empty());
    expect(!app.H.CanRedo());
    expect(app.H.Nodes[a].Children == std::vector{b});
    expect(!app.H.Nodes[c].Hot && !app.H.Nodes[d].Hot);
    app.H.Navigate(a);
    expect(app.State() == a_state);
    app.H.Navigate(b);
    expect(app.State() == replaced);
    expect(app.H.Replay().empty());
    expect(app.State() == replaced);
    app.Reopen(dir);
    expect(app.H.Present == b);
    expect(app.H.Nodes[b].Children.empty());
    expect(!app.H.CanRedo());
    app.H.Navigate(a);
    app.H.Navigate(b);
    expect(app.State() == replaced);
    ExpectAudit(app.H);
}

// A Begin that fails to open its logs or to write the new root leaves the open history in place.
void TestFailedBegin() {
    const TestDir current{"projectstore_current"}, unwritable{"projectstore_unwritable"}, failed_write{"projectstore_failed_root"};
    std::filesystem::create_directories(unwritable.Path / "leaves.log");
    Pages buffer;
    History h;
    h.Track(buffer, "buffer", 0);
    buffer.Resize(4096);
    buffer.Mutable(0, 1)[0] = std::byte{11};
    expect(h.Begin(current));
    const auto before = h.MaterializeLive();
    expect(!h.Begin(unwritable));
    expect(h.Present == 0 && h.MaterializeLive() == before);
    expect(!h.TakeIntegrityError().empty());
    // The streams open and the candidate root fails to write.
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

void AppendGarbage(const std::filesystem::path &log) { std::ofstream{log, std::ios::binary | std::ios::app}.write("garbage!garbage!", 16); }
void DropLastByte(const std::filesystem::path &log) { std::filesystem::resize_file(log, std::filesystem::file_size(log) - 1); }
void FlipLastByte(const std::filesystem::path &log) {
    std::fstream file{log, std::ios::binary | std::ios::in | std::ios::out};
    file.seekg(-1, std::ios::end);
    const auto value = char(file.get() ^ 1);
    file.seekp(-1, std::ios::end);
    file.put(value);
}
void ReplaceWithDirectory(const std::filesystem::path &log) {
    std::filesystem::remove(log);
    std::filesystem::create_directory(log);
}

// A damage to one log of a saved history, and the count of trailing nodes it destroys.
// A cold damage applies after Open to the records a cold load reads, and applying it again repairs them.
// A damage that keeps Open from reading the history leaves the open history in place.
struct Damage {
    std::string_view Log;
    void (*Apply)(const std::filesystem::path &);
    uint32_t Lost{};
    bool Cold{false}, Opens{true};
};

// Every node a damage leaves intact navigates to its recorded state, and repeating the last step restores its state across a reopen.
void TestDamagedStores() {
    constexpr std::array<Damage, 8> Damages{{
        {"leaves.log", AppendGarbage},
        {"nodes.log", AppendGarbage},
        {"tree.log", AppendGarbage},
        {"leaves.log", DropLastByte, 1u},
        {"nodes.log", DropLastByte, 1u},
        // The last record is the saved navigation, so every node survives.
        {"tree.log", DropLastByte},
        // The last node's leaf fails to load while the pool track it shares with its parent loads, and neither track changes.
        {"leaves.log", FlipLastByte, 0u, true},
        {"leaves.log", ReplaceWithDirectory, 0u, false, false},
    }};
    constexpr uint32_t UniqueStep = 0x80000000u | 42;
    const TestDir root{"projectstore_damage"};
    const auto saved = root.Path / "saved";
    std::map<uint32_t, std::pair<std::vector<std::byte>, ToyApp::Values>> expected;
    uint32_t parent{}, last{};
    {
        ToyApp app;
        expect(app.H.Begin(saved));
        expected[0] = app.State();
        for (uint32_t s = 1; s <= 8; ++s) {
            app.Step(s);
            expected[*app.H.Present] = app.State();
        }
        parent = *app.H.Present;
        // Unique data makes the final leaf and manifest records exclusive to the last node.
        app.Step(UniqueStep);
        last = *app.H.Present;
        expected[last] = app.State();
        // Saving at the root leaves the last node cold after Open.
        app.H.Navigate(0);
        expect(app.H.Save());
        expect(app.H.Close());
    }
    ToyApp app;
    for (uint32_t i = 0; i < Damages.size(); ++i) {
        const auto &damage = Damages[i];
        const auto dir = root.Path / std::to_string(i);
        std::filesystem::copy(saved, dir, std::filesystem::copy_options::recursive);
        if (!damage.Cold) damage.Apply(dir / damage.Log);
        const auto present = app.H.Present;
        const auto state = app.State();
        if (!damage.Opens) {
            expect(!app.H.Open(dir)) << "damage" << i;
            expect(app.H.Present == present && app.State() == state) << "damage" << i;
            expect(!app.H.TakeIntegrityError().empty()) << "damage" << i;
            continue;
        }
        expect(boost::ut::fatal(app.H.Open(dir))) << "damage" << i;
        expect(app.H.Nodes.size() == expected.size() - damage.Lost) << "damage" << i << "nodes" << app.H.Nodes.size();
        expect(app.H.TakeIntegrityError().empty()) << "damage" << i;
        if (damage.Cold) damage.Apply(dir / damage.Log);
        for (const auto &[node, node_state] : expected | std::views::reverse) {
            if (node >= app.H.Nodes.size()) continue;
            const auto from = app.H.Present;
            const auto from_state = app.State();
            app.H.Navigate(node);
            if (damage.Cold && node == last) {
                expect(app.H.Present == from && app.State() == from_state) << "damage" << i;
                expect(!app.H.TakeIntegrityError().empty()) << "damage" << i;
            } else {
                expect(app.H.Present == node && app.State() == node_state) << "damage" << i << "node" << node;
            }
        }
        std::string why;
        expect(app.H.Audit(why)) << "damage" << i << why;
        if (damage.Cold) damage.Apply(dir / damage.Log);
        // Repeating the last step rewrites the records the damage lost.
        app.H.Navigate(parent);
        app.Step(UniqueStep);
        expect(app.H.Present == last && app.H.Nodes.size() == expected.size()) << "damage" << i;
        app.H.Navigate(0);
        app.H.Evict(0);
        app.H.Navigate(last);
        expect(app.H.Present == last && app.State() == expected.at(last)) << "damage" << i;
        app.Reopen(dir);
        expect(app.H.Present == last && app.State() == expected.at(last)) << "damage" << i;
        expect(app.H.TakeIntegrityError().empty()) << "damage" << i;
    }
}

void TestClearHistory() {
    const TestDir dir{"/tmp/mesheditor-scratch/projectstore-clear"};
    ToyApp app;
    expect(app.H.Begin(dir));
    app.Step(1);
    const auto &saved_node = app.H.Nodes[*app.H.Present];
    const HistoryPosition saved{std::nullopt, saved_node.Stamps, saved_node.Roots};
    const auto saved_state = app.State();
    app.Step(2);
    const auto baseline = app.State();
    expect(app.H.Clear(&saved));
    expect(app.H.Nodes.size() == 2u && app.H.Present == 1);
    expect(app.State() == baseline);
    app.Step(3);
    const auto continued = app.State();
    app.Reopen(dir);
    expect(app.H.Nodes.size() == 3u && app.H.Present == 2);
    expect(app.State() == continued);
    app.H.Undo();
    expect(app.State() == baseline);
    app.H.Undo();
    expect(app.State() == saved_state);
    app.H.Redo();
    expect(app.State() == baseline);
    app.H.Redo();
    expect(app.State() == continued);
}
} // namespace

const boost::ut::suite ProjectStore = [] {
    boost::ut::test("record pages across resize") = TestRecordPagesAcrossResize;
    boost::ut::test("pool trie against a model") = TestPoolTrieAgainstModel;
    boost::ut::test("capture epoch") = TestCaptureEpoch;
    boost::ut::test("buffer against a model") = TestBufferAgainstModel;
    boost::ut::test("history against a model") = TestHistoryAgainstModel;
    boost::ut::test("commit identity") = TestCommitIdentity;
    boost::ut::test("replace") = TestReplace;
    boost::ut::test("failed begin") = TestFailedBegin;
    boost::ut::test("damaged stores keep every node whose records survive") = TestDamagedStores;
    boost::ut::test("clear history") = TestClearHistory;
};
