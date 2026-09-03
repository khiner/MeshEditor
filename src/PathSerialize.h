#pragma once

#include <zpp_bits.h>

#include <filesystem>
#include <string>

// zpp::bits requires ADL hooks because std::filesystem::path is not reflectable.
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
