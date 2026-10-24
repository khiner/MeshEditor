#pragma once

#include "numeric/uvec2.h"
#include "state/Entity.h"

using numeric::uvec2;

// The viewport's extent in render target pixels.
uvec2 RenderExtentPx(const state::Registry &, state::Entity viewport);
// The extent's width over its height, and one for an empty extent.
float AspectRatio(uvec2);
