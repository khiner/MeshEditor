#pragma once

#include "gpu/SlotOffset.h"

// One pass of a separable filter along rows or along columns, one thread per output texel.
// Output texel (x, y) of an OutWidth by OutHeight image weighs the input texels around input texel (x + Offset, y) along its row, or (x, y + Offset) along its column, clamped to the input's InWidth by InHeight texels.
// Texels take Stride floats each, row by row from the top.
// A box filter weighs each texel by its overlap with a box Width texels wide, and a Gaussian one by a Gaussian whose deviation is a sixth of Width, normalized to sum to one.
struct ImageFilterPushConstants {
    SlotOffset Input DEFAULT(), Output DEFAULT();
    uint32_t InWidth DEFAULT(), InHeight DEFAULT(), OutWidth DEFAULT(), OutHeight DEFAULT(), Offset DEFAULT(), Stride DEFAULT(), Gaussian DEFAULT();
    float Width DEFAULT();
};
