#include "action/LogSerialize.h"

#include <boost/ut.hpp>

#include <sstream>

using namespace action;

namespace {
// A merged Action holding alternative `idx`, default-constructed.
// (Can't use CreateVariantByIndex - it builds a constexpr table and copies, but Action is move-only bc of unique_ptrs)
Action DefaultAt(size_t idx) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) -> Action {
        Action a;
        ((Is == idx ? (void)a.emplace<Is>() : void()), ...);
        return a;
    }(std::make_index_sequence<std::variant_size_v<Action>>{});
}

template<class> constexpr bool IsUniquePtr = false;
template<class T, class D> constexpr bool IsUniquePtr<std::unique_ptr<T, D>> = true;

// Fill null owning pointers (zpp::bits can't serialize null) so every alternative round-trips.
void EnsureSerializable(Action &a) {
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
        for (size_t i = 0; i < std::variant_size_v<Action>; ++i) {
            auto a = DefaultAt(i);
            EnsureSerializable(a);

            auto out = MakeStream();
            SerializeAction(a, out);
            const auto encoded = out.str();

            std::vector<Action> decoded;
            StreamActions(out, [&](Action &&d) { decoded.emplace_back(std::move(d)); });
            expect(decoded.size() == 1u);
            if (decoded.size() != 1u) continue;
            expect(decoded[0].index() == i);

            // Re-encoding the decoded action reproduces the same bytes.
            auto reout = MakeStream();
            SerializeAction(decoded[0], reout);
            const bool stable = reout.str() == encoded;
            expect(stable);
        }
    };

    "back-to-back records read back in order"_test = [] {
        auto out = MakeStream();
        std::vector<size_t> written;
        for (size_t i = 0; i < std::variant_size_v<Action>; ++i) {
            auto a = DefaultAt(i);
            EnsureSerializable(a);
            SerializeAction(a, out);
            written.emplace_back(i);
        }
        std::vector<size_t> read;
        StreamActions(out, [&](Action &&d) { read.emplace_back(d.index()); });
        const bool same = read == written;
        expect(same);
    };
}
