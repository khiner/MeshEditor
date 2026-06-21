#pragma once

#include "numeric/vec2.h"

#include <entt/entity/fwd.hpp>

// Physical pixel render extent for the viewport
uvec2 RenderExtentPx(const entt::registry &);
