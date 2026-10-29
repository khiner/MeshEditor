#pragma once

#include "Field.h"
#include "state/Entity.h"
#include <variant>

namespace state {
// Visits entity references in native component records and their nested values.
void ForEachEntity(auto &value, auto &&visit) {
    using T = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::same_as<T, Entity>) visit(value);
    else if constexpr (requires { std::variant_size<T>::value; }) std::visit([&](auto &v) { ForEachEntity(v, visit); }, value);
    else if constexpr (requires { value.has_value(); }) {
        if (value) ForEachEntity(*value, visit);
    } else if constexpr (requires { value.data(); value.size(); }) {
        for (auto &v : value) ForEachEntity(v, visit);
    } else if constexpr (field::Walkable<T>) field::ForEach(value, [&](auto &v, auto) { ForEachEntity(v, visit); });
}
} // namespace state
