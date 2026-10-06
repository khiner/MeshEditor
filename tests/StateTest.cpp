#include "RangeAllocator.h"
#include "RunSuites.h"
#include "scene/Entity.h"
#include "state/Scene.h"

#include <map>
#include <random>
#include <set>
#include <type_traits>

using namespace boost::ut;

int main() {
    static_assert(std::is_same_v<decltype(std::declval<state::Scene &>().get<Name>(state::Null)), const Name &>);
    "owned tables agree with an independent sparse model"_test = [] {
        state::Scene r;
        std::mt19937 rng{173};
        std::map<state::Entity, std::pair<std::string, bool>> model;
        std::vector<state::Entity> dead;
        for (unsigned step = 0; step < 5000; ++step) {
            if (model.empty() || rng() % 4 == 0) {
                const auto e = r.create();
                const auto name = std::to_string(rng());
                model.emplace(e, std::pair{name, false});
                r.emplace<Name>(e, name);
            } else {
                auto it = std::next(model.begin(), rng() % model.size());
                const auto e = it->first;
                switch (rng() % 4) {
                    case 0:
                        dead.push_back(e);
                        r.destroy(e);
                        model.erase(it);
                        break;
                    case 1:
                        it->second.first = std::string(256 + rng() % 512, char('a' + rng() % 26));
                        r.edit<Name>(e).Value = it->second.first;
                        break;
                    case 2:
                        it->second.second = true;
                        r.emplace_or_replace<Selected>(e);
                        break;
                    case 3:
                        it->second.second = false;
                        r.remove<Selected>(e);
                        break;
                }
            }
            if (step % 50) continue;
            std::set<state::Entity> seen;
            for (auto [e, name] : r.view<const Name>().each()) {
                expect(seen.insert(e).second && model.contains(e));
                expect(name.Value == model.at(e).first);
            }
            expect(seen.size() == model.size());
            for (const auto &[e, value] : model) {
                expect(r.valid(e));
                expect(r.all_of<Selected>(e) == value.second);
                expect(r.view<const Name>(state::Exclude<Selected>).contains(e) == !value.second);
            }
            for (const auto e : dead) expect(!r.valid(e) && !r.try_get<Name>(e));
        }
        // Removing the current record while traversing must visit each record once.
        size_t removed = 0;
        for (const auto e : r.view<const Name>()) {
            r.destroy(e);
            ++removed;
        }
        expect(removed == model.size() && r.view<const Name>().empty());
    };
    "generation exhaustion retires a slot"_test = [] {
        state::Scene r;
        const auto first = r.create();
        auto current = first;
        for (uint32_t i = 0; i < 0xfffu; ++i) {
            expect(state::Generation(current) == i);
            r.destroy(current);
            current = r.create();
            expect(!r.valid(first));
        }
        expect(state::Index(current) != state::Index(first));
        std::set<state::Entity> allocated{current};
        for (unsigned i = 0; i < 70; ++i) expect(allocated.insert(r.create()).second);
    };
    "batch destruction captures page runs and preserves sparse survivors"_test = [] {
        state::Scene r;
        std::vector<state::Entity> entities, targets;
        for (uint32_t i = 0; i < 160u; ++i) {
            const auto e = r.create();
            entities.push_back(e);
            r.emplace<Name>(e, std::to_string(i));
            r.emplace<Selected>(e);
        }
        // Reused slots exercise generation matching as well as page masks.
        const auto stale = entities[64];
        r.destroy(stale);
        entities[64] = r.create();
        r.emplace<Name>(entities[64], "64");
        r.emplace<Selected>(entities[64]);
        r.ClearChanges();
        struct Capture {
            std::vector<std::string> Values;
            uint32_t Calls{};
        };
        Capture captured;
        std::vector<std::string> notified;
        r.HistoryOwner = &captured;
        r.Capture = [](state::Scene &r, state::TypeId type, std::span<const state::PageMask> pages) {
            if (type != state::Type<Name>()) return;
            auto &capture = *static_cast<Capture *>(r.HistoryOwner);
            ++capture.Calls;
            auto &values = capture.Values;
            for (const auto [page, mask] : pages)
                for (auto bits = mask; bits; bits &= bits - 1u) values.push_back(r.get<Name>(r.EntityAt(page * state::Table::PageCount + std::countr_zero(bits))).Value);
        };
        state::DirtySet destroyed;
        destroyed.bind(r);
        destroyed.on<Name>(state::On::Destroy);
        r.on_destroy<Name, [](state::Scene &r, state::Entity e) {
            // Dependent component removals during a callback must not be erased twice.
            expect(!r.get<Name>(e).Value.empty());
            r.remove<Selected>(e);
        }>();
        for (uint32_t i = 0; i < entities.size(); ++i) {
            if (i < 96u || i % 3u == 0u) {
                targets.push_back(entities[i]);
                notified.push_back(std::to_string(i));
            }
        }
        std::ranges::reverse(targets);
        targets.push_back(targets.front());
        r.destroy(targets);
        expect(captured.Values == notified);
        expect(captured.Calls == 1u);
        expect(destroyed.size() == notified.size());
        expect(r.Living.size() == entities.size() - notified.size());
        for (uint32_t i = 0; i < entities.size(); ++i) {
            const bool removed = i < 96u || i % 3u == 0u;
            expect(r.valid(entities[i]) == !removed);
            expect(r.all_of<Name, Selected>(entities[i]) == !removed);
            if (!removed) expect(r.get<Name>(entities[i]).Value == std::to_string(i));
        }
        r.Capture = nullptr;
        const auto reused = r.create();
        expect(std::ranges::find(entities, reused) == entities.end());
        r.emplace<Name>(reused, "survivor");
        r.RemoveComponents(stale);
        expect(r.get<Name>(reused).Value == "survivor");
    };
    "clearing a component captures each page once and notifies every holder"_test = [] {
        state::Scene r;
        std::vector<state::Entity> holders;
        for (uint32_t i = 0; i < 200u; ++i) {
            const auto e = r.create();
            if (i % 7u == 0u || (i >= 64u && i < 96u)) continue;
            r.emplace<Selected>(e);
            holders.push_back(e);
        }
        std::map<uint32_t, uint32_t> captured;
        r.HistoryOwner = &captured;
        r.Capture = [](state::Scene &r, state::TypeId type, std::span<const state::PageMask> pages) {
            if (type != state::Type<Selected>()) return;
            for (const auto [page, mask] : pages) {
                expect(r.storage<Selected>().mask(page) == mask);
                ++(*static_cast<std::map<uint32_t, uint32_t> *>(r.HistoryOwner))[page];
            }
        };
        state::DirtySet destroyed;
        destroyed.bind(r);
        destroyed.on<Selected>(state::On::Destroy);
        r.clear<Selected>();
        r.Capture = nullptr;
        std::map<uint32_t, uint32_t> expected;
        for (const auto e : holders) expected[state::Index(e) / state::Table::PageCount] = 1u;
        const bool each_page_once = captured == expected;
        expect(each_page_once);
        expect(destroyed.size() == holders.size());
        for (const auto e : holders) expect(destroyed.contains(e) && !r.all_of<Selected>(e));
        expect(r.view<const Selected>().empty());
    };
    "capture precedes mutation and dirty lifetime respects generation and reset"_test = [] {
        state::Scene r;
        std::vector<std::string> old;
        r.HistoryOwner = &old;
        r.Capture = [](state::Scene &r, state::TypeId type, std::span<const state::PageMask> pages) {
            if (type == state::Type<Name>()) {
                for (const auto [page, mask] : pages)
                    for (auto bits = mask; bits; bits &= bits - 1u) {
                        const auto e = r.EntityAt(page * state::Table::PageCount + std::countr_zero(bits));
                        static_cast<std::vector<std::string> *>(r.HistoryOwner)->push_back(r.all_of<Name>(e) ? r.get<Name>(e).Value : "absent");
                    }
            }
        };
        auto &managed = reactive(r, state::Change::Selected);
        managed.on<Name>(state::On::Create | state::On::Destroy);
        state::DirtySet removed;
        removed.bind(r);
        removed.on<Name>(state::On::Destroy);
        const auto e = r.create();
        r.emplace<Name>(e, "first");
        r.patch<Name>(e, [](auto &name) { name.Value = "second"; });
        r.destroy(e);
        expect(old == std::vector<std::string>{"absent", "first", "second"});
        expect(managed.empty() && removed.contains(e));
        const auto next = r.create();
        r.emplace<Name>(next, "reused");
        expect(managed.contains(next) && !managed.contains(e));
        r.destroy(next);
        expect(removed.contains(next) && !removed.contains(e));
        r.ResetEntities();
        expect(removed.empty() && managed.empty());
        expect(state::Integral(r.create()) == 0u);
    };
    "bulk range retirement preserves unrelated allocations and address ordering"_test = [] {
        RangeAllocator batch, scalar;
        std::mt19937 random{934};
        std::vector<Range> retired;
        for (uint32_t i = 0u; i < 512u; ++i) {
            const auto count = 1000000u + random() % 6000000u;
            const auto range = batch.Allocate(count);
            const auto other = scalar.Allocate(count);
            expect(range.Offset == other.Offset && range.Count == other.Count);
            if (i % 3u) retired.push_back(range);
        }
        std::shuffle(retired.begin(), retired.end(), random);
        for (const auto range : retired) scalar.Free(range);
        batch.Free(retired);
        for (uint32_t i = 0u; i < 256u; ++i) {
            const auto count = 1u + random() % 1000000u;
            const auto actual = batch.Allocate(count), expected = scalar.Allocate(count);
            expect(actual.Offset == expected.Offset && actual.Count == expected.Count);
        }
        RangeAllocator all;
        std::vector<Range> ranges;
        for (uint32_t i = 0u; i < 512u; ++i) ranges.push_back(all.Allocate(3000000u + random() % 4000000u));
        const auto end = all.HighWaterMark();
        std::shuffle(ranges.begin(), ranges.end(), random);
        all.Free(std::move(ranges));
        const auto restored = all.Allocate(end);
        expect(restored.Offset == 0u && restored.Count == end);
    };
    return RunSuites();
}
