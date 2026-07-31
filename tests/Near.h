#pragma once

#include <algorithm>
#include <cmath>

// Closeness relative to `b`, with an absolute floor of one, so a quantity passing through zero is compared absolutely.
inline bool Near(double a, double b, double tolerance = 1e-6) { return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b)); }
