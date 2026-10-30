#include "action/LogSerialize.h"

#include <boost/ut.hpp>

#include <zpp_bits.h>

#include <filesystem>
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

std::pair<size_t, size_t> IndexOf(const Action &a) {
    return {a.index(), std::visit([](const auto &dv) { return dv.index(); }, a)};
}

template<class> constexpr bool IsUniquePtr = false;
template<class T, class D> constexpr bool IsUniquePtr<std::unique_ptr<T, D>> = true;
template<class> constexpr bool IsVector = false;
template<class T, class A> constexpr bool IsVector<std::vector<T, A>> = true;
template<class> constexpr bool IsArray = false;
template<class T, size_t N> constexpr bool IsArray<std::array<T, N>> = true;
template<class> constexpr bool IsOptional = false;
template<class T> constexpr bool IsOptional<std::optional<T>> = true;
template<class> constexpr bool IsVariant = false;
template<class... Ts> constexpr bool IsVariant<std::variant<Ts...>> = true;
template<class T> concept Components = requires(T v) { v.x; v.y; };

// Writes a distinct value into every member of a payload, through containers, optionals, owning pointers and variants, so a round trip that drops or defaults a member shows.
// A variant takes its last alternative, and a null owning pointer takes a filled value, since zpp::bits serializes neither a null pointer nor an empty choice.
struct Filler {
    uint32_t Next{1};
    void operator()(auto &value) {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::same_as<T, bool>) value = true;
        else if constexpr (std::is_enum_v<T>) value = T(std::underlying_type_t<T>(1));
        else if constexpr (std::is_arithmetic_v<T>) value = T(Next++);
        else if constexpr (std::same_as<T, std::string> || std::same_as<T, std::filesystem::path>) value = std::to_string(Next++);
        else if constexpr (IsVector<T>) {
            value.resize(2);
            for (auto &element : value) (*this)(element);
        } else if constexpr (IsArray<T>) {
            for (auto &element : value) (*this)(element);
        } else if constexpr (IsOptional<T>) {
            value.emplace();
            (*this)(*value);
        } else if constexpr (IsUniquePtr<T>) {
            if (!value) value = std::make_unique<typename T::element_type>();
            (*this)(*value);
        } else if constexpr (IsVariant<T>) {
            value.template emplace<std::variant_size_v<T> - 1>();
            std::visit([&](auto &alternative) { (*this)(alternative); }, value);
        } else if constexpr (Components<T>) {
            value.x = Next++;
            value.y = Next++;
            if constexpr (requires { value.z; }) value.z = Next++;
            if constexpr (requires { value.w; }) value.w = Next++;
        } else if constexpr (std::is_empty_v<T>) {
        } else {
            zpp::bits::visit_members(value, [&](auto &...members) { ((*this)(members), ...); });
        }
    }
};

// Member-wise equality over the same structure.
struct Comparer {
    template<size_t I = 0> bool Variant(const auto &a, const auto &b) const {
        using T = std::remove_cvref_t<decltype(a)>;
        if constexpr (I < std::variant_size_v<T>) {
            if (a.index() == I) return b.index() == I && (*this)(std::get<I>(a), std::get<I>(b));
            return Variant<I + 1>(a, b);
        } else return false;
    }
    bool operator()(const auto &a, const auto &b) const {
        using T = std::remove_cvref_t<decltype(a)>;
        if constexpr (std::is_enum_v<T> || std::is_arithmetic_v<T> || std::same_as<T, std::string> || std::same_as<T, std::filesystem::path>) return a == b;
        else if constexpr (IsVector<T> || IsArray<T>) return a.size() == b.size() && std::ranges::equal(a, b, *this);
        else if constexpr (IsOptional<T>) return a.has_value() == b.has_value() && (!a || (*this)(*a, *b));
        else if constexpr (IsUniquePtr<T>) return bool(a) == bool(b) && (!a || (*this)(*a, *b));
        else if constexpr (IsVariant<T>) return Variant(a, b);
        else if constexpr (Components<T>) {
            bool equal = a.x == b.x && a.y == b.y;
            if constexpr (requires { a.z; }) equal = equal && a.z == b.z;
            if constexpr (requires { a.w; }) equal = equal && a.w == b.w;
            return equal;
        } else if constexpr (std::is_empty_v<T>) return true;
        else return zpp::bits::visit_members(a, [&](const auto &...ma) { return zpp::bits::visit_members(b, [&](const auto &...mb) { return ((*this)(ma, mb) && ...); }); });
    }
};

// Fills the action's leaf payload without changing which action it is.
void Fill(Action &a) {
    Filler filler;
    std::visit([&](auto &domain) { std::visit([&](auto &leaf) { filler(leaf); }, domain); }, a);
}
bool Equal(const Action &a, const Action &b) {
    return IndexOf(a) == IndexOf(b) && std::visit([&](const auto &domain) { return std::visit([&](const auto &leaf) {
                                                                                 using L = std::remove_cvref_t<decltype(leaf)>;
                                                                                 return Comparer{}(leaf, std::get<L>(std::get<std::remove_cvref_t<decltype(domain)>>(b))); }, domain); }, a);
}

std::stringstream MakeStream() { return std::stringstream{std::ios::in | std::ios::out | std::ios::binary}; }
} // namespace

const boost::ut::suite ActionSerialize = [] {
    using namespace boost::ut;

    "every action round-trips through the log with every member filled"_test = [] {
        for (auto &a : AllDefaultActions()) {
            Fill(a);
            auto out = MakeStream();
            SerializeAction(a, out);
            const auto encoded = out.str();
            std::vector<Action> decoded;
            StreamActions(out, [&](Action &&d) { decoded.emplace_back(std::move(d)); });
            expect(decoded.size() == 1u) << IndexOf(a).first << IndexOf(a).second;
            if (decoded.size() != 1u) continue;
            expect(Equal(decoded[0], a)) << IndexOf(a).first << IndexOf(a).second;
            auto again = MakeStream();
            SerializeAction(decoded[0], again);
            expect(again.str() == encoded);
        }
    };

    "a stream reads back in order up to a limit and stops at a truncated record"_test = [] {
        auto out = MakeStream();
        std::vector<std::pair<size_t, size_t>> written;
        size_t last_record = 0;
        for (auto &a : AllDefaultActions()) {
            Fill(a);
            last_record = out.str().size();
            SerializeAction(a, out);
            written.emplace_back(IndexOf(a));
        }
        const auto all = out.str();
        const auto read = [&](std::string bytes, uint64_t limit = std::numeric_limits<uint64_t>::max()) {
            auto in = MakeStream();
            in << bytes;
            std::vector<std::pair<size_t, size_t>> indices;
            StreamActions(in, [&](Action &&d) { indices.emplace_back(IndexOf(d)); }, limit);
            return indices;
        };
        expect(read(all, 0).empty());
        expect(bool(read(all, 1) == std::vector{written.front()}));
        expect(bool(read(all) == written));
        for (size_t length = last_record; length < all.size(); ++length) expect(read(all.substr(0, length)).size() == written.size() - 1u) << length;
    };
};
