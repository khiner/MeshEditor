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
    "capture precedes mutation and dirty lifetime respects generation and reset"_test = [] {
        state::Scene r;
        std::vector<std::string> old;
        r.HistoryOwner = &old;
        r.Capture = [](state::Scene &r, state::TypeId type, state::Entity e) {
            if (type == state::Type<Name>()) static_cast<std::vector<std::string> *>(r.HistoryOwner)->push_back(r.all_of<Name>(e) ? r.get<Name>(e).Value : "absent");
        };
        auto &managed = r.changes(state::Type<state::DirtySet>());
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
    return RunSuites();
}
