#pragma once

#include "entt_fwd.h"

#include <entt/core/type_info.hpp>

namespace action {
struct SetTag {
    entt::entity Entity;
    entt::id_type TagType;
    bool Present;
};

template<typename Tag>
constexpr SetTag SetTagOf(entt::entity e, bool present) {
    return {e, entt::type_hash<Tag>::value(), present};
}
} // namespace action
