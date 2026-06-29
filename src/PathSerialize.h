#pragma once

#include <zpp_bits.h>

#include <filesystem>
#include <string>

// zpp::bits can't reflect std::filesystem::path, so it gets two ADL hooks (in std::filesystem so ADL finds them):
// a members<1> count to stop zpp structured-binding its private members, plus a serialize through its native string.
namespace std::filesystem {
auto serialize(const path &) -> zpp::bits::members<1>;
constexpr auto serialize(auto &archive, const path &p) { return archive(p.string()); }
constexpr auto serialize(auto &archive, path &p) {
    if constexpr (std::remove_cvref_t<decltype(archive)>::kind() == zpp::bits::kind::out) {
        return archive(p.string());
    } else {
        std::string s;
        const auto result = archive(s);
        p = path{s};
        return result;
    }
}
} // namespace std::filesystem
