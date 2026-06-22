#pragma once

#include "Path.h"

#include <zpp_bits.h>

#include <filesystem>
#include <string>

// zpp::bits can't reflect std::filesystem::path, so (de)serialize the Path component via its native string.
// Include only where Path is serialized (its snapshot registration), to keep zpp out of Path.h's many users.
constexpr auto serialize(auto &archive, const Path &p) { return archive(p.Value.string()); }
constexpr auto serialize(auto &archive, Path &p) {
    if constexpr (std::remove_cvref_t<decltype(archive)>::kind() == zpp::bits::kind::out) {
        return archive(p.Value.string());
    } else {
        std::string s;
        const auto result = archive(s);
        p.Value = std::filesystem::path{s};
        return result;
    }
}
