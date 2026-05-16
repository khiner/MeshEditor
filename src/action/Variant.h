#pragma once

#include "entt_fwd.h"

#include <variant>

namespace action::detail {
template<typename>
struct VariantFrom;
template<typename... Ts>
struct VariantFrom<entt::type_list<Ts...>> {
    using type = std::variant<Ts...>;
};

template<typename L>
using VariantFromT = typename VariantFrom<L>::type;
} // namespace action::detail
