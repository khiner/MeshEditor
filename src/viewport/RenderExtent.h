#pragma once

#include "numeric/uvec2.h"

using numeric::uvec2;

struct FrameState;
struct Viewport;

// The viewport's extent in render target pixels.
uvec2 RenderExtentPx(const Viewport &, const FrameState &);
