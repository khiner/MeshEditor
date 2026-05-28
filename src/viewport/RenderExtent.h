#pragma once

#include "numeric/vec2.h"

// Physical pixel render extent for a viewport's logical size, scaled by the current DPI framebuffer scale.
uvec2 RenderExtentPx(uvec2 logical_extent);
