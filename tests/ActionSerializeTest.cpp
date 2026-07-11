#include "action/LogSerialize.h"

#include <boost/ut.hpp>

#include <sstream>

using namespace action;

namespace {
// One default-constructed Action per (domain, leaf) alternative, constructed in place since Action is move-only (unique_ptr payloads).
std::vector<Action> AllDefaultActions() {
    std::vector<Action> all;
    [&]<size_t... Ds>(std::index_sequence<Ds...>) {
        ([&] {
            constexpr size_t D = Ds;
            using DV = std::variant_alternative_t<D, Action>;
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (all.emplace_back(std::in_place_index<D>, DV{std::in_place_index<Is>}), ...);
            }(std::make_index_sequence<std::variant_size_v<DV>>{});
        }(),
         ...);
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
    return all;
}

// The (domain, leaf) alternative indices identifying the action's type.
std::pair<size_t, size_t> IndexOf(const Action &a) {
    return {a.index(), std::visit([](const auto &dv) { return dv.index(); }, a)};
}

template<class> constexpr bool IsUniquePtr = false;
template<class T, class D> constexpr bool IsUniquePtr<std::unique_ptr<T, D>> = true;

// Fill null owning pointers (zpp::bits can't serialize null) so every alternative round-trips.
void EnsureSerializable(Action &a) {
    std::visit(
        [](auto &dv) {
            std::visit(
                [](auto &alt) {
                    zpp::bits::visit_members(alt, [](auto &...members) {
                        ([](auto &m) {
                            using M = std::decay_t<decltype(m)>;
                            if constexpr (IsUniquePtr<M>) {
                                if (!m) m = std::make_unique<typename M::element_type>();
                            }
                        }(members),
                         ...);
                    });
                },
                dv
            );
        },
        a
    );
}

std::stringstream MakeStream() { return std::stringstream{std::ios::in | std::ios::out | std::ios::binary}; }
} // namespace

int main(int argc, const char **argv) {
    using namespace boost::ut;

    // Optional test-name filter, e.g. `MeshEditorActionSerializeTest "every action*"`.
    if (argc > 1) cfg<override> = {.filter = argv[1]};

    "every action round-trips through the log"_test = [] {
        for (auto &a : AllDefaultActions()) {
            EnsureSerializable(a);

            auto out = MakeStream();
            SerializeAction(a, out);
            const auto encoded = out.str();

            std::vector<Action> decoded;
            StreamActions(out, [&](Action &&d) { decoded.emplace_back(std::move(d)); });
            expect(decoded.size() == 1u);
            if (decoded.size() != 1u) continue;
            expect(IndexOf(decoded[0]) == IndexOf(a));

            // Re-encoding the decoded action reproduces the same bytes.
            auto reout = MakeStream();
            SerializeAction(decoded[0], reout);
            const bool stable = reout.str() == encoded;
            expect(stable);
        }
    };

    "back-to-back records read back in order"_test = [] {
        auto out = MakeStream();
        std::vector<std::pair<size_t, size_t>> written;
        for (auto &a : AllDefaultActions()) {
            EnsureSerializable(a);
            SerializeAction(a, out);
            written.emplace_back(IndexOf(a));
        }
        std::vector<std::pair<size_t, size_t>> read;
        StreamActions(out, [&](Action &&d) { read.emplace_back(IndexOf(d)); });
        const bool same = read == written;
        expect(same);
    };
}
