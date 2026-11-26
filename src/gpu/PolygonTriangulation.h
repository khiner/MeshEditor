#pragma once

#include "gpu/Types.h"
#ifndef __METAL_VERSION__
#include "numeric/VectorOps.h"

#include <cfloat>
#endif

#ifdef __METAL_VERSION__
using PolygonPoint = float2;
using PolygonVector = float3;
#else
using PolygonPoint = vec2;
using PolygonVector = vec3;
#endif

inline float PolygonDot(PolygonPoint a, PolygonPoint b) { return a.x * b.x + a.y * b.y; }
inline float PolygonDet(PolygonPoint a, PolygonPoint b) { return a.x * b.y - a.y * b.x; }
inline float PolygonAbs(float value) { return value < 0.f ? -value : value; }
inline float PolygonMin(float a, float b) { return a < b ? a : b; }
inline float PolygonMax(float a, float b) { return a > b ? a : b; }
inline float PolygonSqrt(float value) {
#ifdef __METAL_VERSION__
    return metal::sqrt(value);
#else
    return __builtin_sqrtf(value);
#endif
}
inline PolygonPoint PolygonUnit(PolygonPoint v) {
    const float length = PolygonSqrt(PolygonDot(v, v));
    return length > 0.f ? PolygonPoint{v.x / length, v.y / length} : PolygonPoint{0.f, 0.f};
}
inline PolygonVector PolygonCross(PolygonVector a, PolygonVector b) { return PolygonVector{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float PolygonDot3(PolygonVector a, PolygonVector b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Manifold's CCW: the turn's sign at p0 from p1 to p2, or zero when the triangle's height over its longer side from p0 is within half of `tolerance`.
inline int PolygonCCW(PolygonPoint p0, PolygonPoint p1, PolygonPoint p2, float tolerance) {
    const PolygonPoint v1 = p1 - p0, v2 = p2 - p0;
    const float area = PolygonDet(v1, v2);
    const float base2 = PolygonMax(PolygonDot(v1, v1), PolygonDot(v2, v2));
    if (area * area * 4.f <= base2 * tolerance * tolerance) return 0;
    return area > 0.f ? 1 : -1;
}

struct PolygonCrossing {
    float X;
    bool Valid;
};

// The corners each lane of a one-loop scan reads ahead of its tests.
GPU_CONSTANT uint32_t PolygonScanBlock = 8u;

// Corner `i` of `n`, or the last corner past the end, which a block reads and does not test.
inline uint32_t PolygonCornerIn(uint32_t i, uint32_t n) { return i < n ? i : n - 1u; }

// One lane clipping a polygon alone.
struct PolygonLane {
    uint32_t Lane{0u}, Count{1u};
    float Max(float value) const { return value; }
    uint32_t Least(float, uint32_t, uint32_t corner) const { return corner; }
};

// Triangulates a polygon's loops as manifold's EarClip does, on projected corner points the caller writes with the loops' links.
// Corner i's point is values[2i] and values[2i + 1], its next and previous corners are links[i] and links[capacity + i], and loop l's first corner is loops[l].
// The caller owns 4 * capacity - n words of `links`, 3 * capacity floats of `values` and 2 * loop_count words of `loops`.
// `capacity` covers the n corners and two more per hole.
// Each triangle is emitted counterclockwise with its corners, where a copy stands for the corner it copies, and its index.
// The lanes of `lanes` run the same clip and split the scans of a one-loop polygon, each lane writing every value the clip writes.
template<typename Links, typename Values, typename Loops, typename Emit, typename Lanes = PolygonLane>
inline void TriangulateLoops(uint32_t n, uint32_t capacity, uint32_t loop_count, Links links, Values values, Loops loops, float epsilon, Emit emit, Lanes lanes = {}) {
    const auto point = [&](uint32_t i) { return PolygonPoint{values[2u * i], values[2u * i + 1u]}; };
    // Each corner links to its live neighbors, and a clipped corner keeps the links it had when it was clipped.
    // The third run holds the clip stack of the degenerate passes and then each queued ear's insertion order, and the fourth each copy's corner.
    const auto next = [&](uint32_t i) { return uint32_t(links[i]); };
    const auto prev = [&](uint32_t i) { return uint32_t(links[capacity + i]); };
    const auto link = [&](uint32_t l, uint32_t r) {
        links[l] = r;
        links[capacity + r] = l;
    };
    const auto corner = [&](uint32_t i) { return i < n ? i : uint32_t(links[3u * capacity + i - n]); };
    const auto clipped = [&](uint32_t i) { return prev(next(i)) != i; };
    // One loop's live corners are all on the cycle being clipped, so its scans run over the corner array without a chain of dependent link reads.
    const bool one_loop = loop_count == 1u;
    const auto is_short = [&](uint32_t i) {
        const PolygonPoint edge = point(next(i)) - point(i);
        return PolygonDot(edge, edge) * 4.f < epsilon * epsilon;
    };
    uint32_t emitted = 0u;
    const auto clip = [&](uint32_t ear) {
        const uint32_t l = prev(ear), r = next(ear);
        link(l, r);
        const uint32_t a = corner(l), b = corner(ear), c = corner(r);
        if (a != b && b != c && c != a) emit(uvec3{a, b, c}, emitted++);
    };
    const auto clip_degenerate = [&](uint32_t first) {
        uint32_t top = 0u;
        links[2u * capacity + top++] = first;
        while (top) {
            const uint32_t ear = links[2u * capacity + --top];
            const uint32_t l = prev(ear), r = next(ear);
            if (clipped(ear) || l == r) continue;
            const PolygonPoint pl = point(l), pe = point(ear), pr = point(r);
            if (!is_short(ear) && !(PolygonCCW(pl, pe, pr, epsilon) == 0 && PolygonDot(pl - pe, pr - pe) > 0.f)) continue;
            clip(ear);
            links[2u * capacity + top++] = r;
            links[2u * capacity + top++] = l;
        }
    };
    // Visits each live corner once from `first`, and returns a live corner, or capacity when two corners remain.
    const auto loop = [&](uint32_t first, auto visit) {
        uint32_t i = first;
        do {
            if (clipped(i)) {
                first = prev(next(i));
                if (!clipped(first)) {
                    i = first;
                    if (next(i) == prev(i)) return capacity;
                    visit(i);
                }
            } else {
                if (next(i) == prev(i)) return capacity;
                visit(i);
            }
            i = next(i);
        } while (i != first);
        return i;
    };
    // Whether corner `self` lies inside the edge from `tail` to its next corner, walking past corners within epsilon until the side is certain.
    const auto inside_edge = [&](uint32_t self, uint32_t tail, bool to_left) {
        const float p2 = epsilon * epsilon;
        const uint32_t stop = to_left ? next(self) : prev(self);
        const auto step = [&](uint32_t i) { return to_left ? prev(i) : next(i); };
        uint32_t next_l = next(prev(self)), next_r = next(tail), center = tail, last = center;
        while (next_l != next_r && tail != next_r && next_l != stop) {
            const PolygonPoint edge_l = point(next_l) - point(center);
            const float l2 = PolygonDot(edge_l, edge_l);
            if (l2 <= p2) {
                next_l = step(next_l);
                continue;
            }
            const PolygonPoint edge_r = point(next_r) - point(center);
            const float r2 = PolygonDot(edge_r, edge_r);
            if (r2 <= p2) {
                next_r = next(next_r);
                continue;
            }
            const PolygonPoint across = point(next_r) - point(next_l);
            if (PolygonDot(across, across) <= p2) {
                last = center;
                center = next_l;
                next_l = step(next_l);
                if (next_l == next_r) break;
                next_r = next(next_r);
                continue;
            }
            int convexity = PolygonCCW(point(next_l), point(center), point(next_r), epsilon);
            if (center != last) convexity += PolygonCCW(point(last), point(center), point(next_l), epsilon) + PolygonCCW(point(next_r), point(center), point(last), epsilon);
            if (convexity != 0) return convexity > 0;
            if (l2 < r2) {
                center = next_l;
                next_l = step(next_l);
            } else {
                center = next_r;
                next_r = next(next_r);
            }
            last = center;
        }
        return true;
    };
    const auto reflex = [&](uint32_t i) { return !inside_edge(prev(i), next(prev(i)), true); };

    for (uint32_t i = 0u; i < n; ++i) clip_degenerate(i);
    uint32_t copies = n;

    // Each loop's state after its start is empty, simple, outer, a hole, joined, or a hole no outer loop holds.
    // A hole's state also records whether its start lies on its box's top or bottom within epsilon.
    const uint32_t empty = 0u, simple = 1u, outer = 2u, hole = 3u, joined = 4u, unjoined = 5u;
    const auto kind = [&](uint32_t l) { return uint32_t(loops[loop_count + l]) & 15u; };
    // A loop starts at its rightmost reflex corner, and is a hole when it has one and its area is below minus epsilon times its box's longer side.
    for (uint32_t l = 0u; l < loop_count; ++l) {
        const uint32_t first = loops[l];
        const PolygonPoint origin = point(first);
        uint32_t start = first;
        bool any_reflex = false;
        float right_x = -FLT_MAX, area = 0.f, compensation = 0.f;
        PolygonPoint low{FLT_MAX, FLT_MAX}, high{-FLT_MAX, -FLT_MAX};
        if (loop(first, [&](uint32_t i) {
                const PolygonPoint p = point(i);
                low = PolygonPoint{PolygonMin(low.x, p.x), PolygonMin(low.y, p.y)};
                high = PolygonPoint{PolygonMax(high.x, p.x), PolygonMax(high.y, p.y)};
                // Kahan summation
                const float term = PolygonDet(p - origin, point(next(i)) - origin), sum = area + term;
                compensation += (area - sum) + term;
                area = sum;
                if ((!any_reflex || p.x > right_x) && reflex(i)) {
                    any_reflex = true;
                    right_x = p.x;
                    start = i;
                }
            }) == capacity) {
            loops[loop_count + l] = empty;
            continue;
        }
        area += compensation;
        const float min_area = epsilon * PolygonMax(high.x - low.x, high.y - low.y);
        loops[l] = start;
        if (any_reflex && area < -min_area) {
            const float y = point(start).y;
            const int on_top = y >= high.y - epsilon ? 1 : y <= low.y + epsilon ? -1 :
                                                                                  0;
            loops[loop_count + l] = hole | uint32_t(on_top + 1) << 4u;
        } else {
            loops[loop_count + l] = area > min_area ? outer : simple;
        }
    }
    // The loop of a kind whose start lies furthest right, the first of equals, or loop_count when none is left.
    const auto rightmost = [&](uint32_t of_kind) {
        uint32_t best = loop_count;
        for (uint32_t l = 0u; l < loop_count; ++l) {
            if (kind(l) == of_kind && (best == loop_count || point(loops[l]).x > point(loops[best]).x)) best = l;
        }
        return best;
    };
    // The x at which the edge from `edge` to its next corner crosses the height of `s` upward, within epsilon.
    const auto crossing = [&](uint32_t edge, PolygonPoint s, int on_top) {
        const PolygonPoint p = point(edge), r = point(next(edge));
        if (PolygonAbs(p.y - s.y) <= epsilon) {
            if (r.y <= s.y + epsilon || on_top == 1) return PolygonCrossing{0.f, false};
            return PolygonCrossing{p.x, true};
        }
        if (p.y < s.y - epsilon) {
            if (r.y > s.y + epsilon) return PolygonCrossing{p.x + (s.y - p.y) * (r.x - p.x) / (r.y - p.y), true};
            if (r.y < s.y - epsilon || on_top == -1) return PolygonCrossing{0.f, false};
            return PolygonCrossing{r.x, true};
        }
        return PolygonCrossing{0.f, false};
    };
    const auto each_outer = [&](auto visit) {
        for (uint32_t l = 0u; l < loop_count; ++l) {
            if (kind(l) == outer) loop(loops[l], visit);
        }
    };
    // Holes join rightmost first, each from its start to the nearest outer edge it sees rightward along its height, or to a reflex corner closer to it.
    for (uint32_t l = rightmost(hole); l != loop_count; l = rightmost(hole)) {
        const uint32_t start = loops[l];
        const int on_top = int(uint32_t(loops[loop_count + l]) >> 4u) - 1;
        const PolygonPoint s = point(start);
        uint32_t edge = capacity;
        each_outer([&](uint32_t e) {
            const PolygonCrossing x = crossing(e, s, on_top);
            if (x.Valid && inside_edge(start, e, true) &&
                (edge == capacity || PolygonCCW(PolygonPoint{x.X, s.y}, point(edge), point(next(edge)), epsilon) == 1 ||
                 (point(edge).y < point(e).y ? inside_edge(e, edge, false) : !inside_edge(edge, e, false)))) edge = e;
        });
        if (edge == capacity) {
            loops[loop_count + l] = unjoined;
            continue;
        }
        loops[loop_count + l] = joined;
        const PolygonPoint pe = point(edge), pr = point(next(edge));
        uint32_t connector = pe.x < s.x ? next(edge) : pr.x < s.x ? edge :
            pr.y - s.y > s.y - pe.y                               ? edge :
                                                                    next(edge);
        if (PolygonAbs(point(connector).y - s.y) > epsilon) {
            const float above = point(connector).y > s.y ? 1.f : -1.f;
            each_outer([&](uint32_t v) {
                const PolygonPoint p = point(v), c = point(connector);
                const float inside = above * float(PolygonCCW(s, p, c, epsilon));
                if (p.x > s.x - epsilon && p.y * above > s.y * above - epsilon &&
                    (inside > 0.f || (inside == 0.f && p.x < c.x && p.y * above < c.y * above)) &&
                    inside_edge(v, edge, true) && reflex(v)) connector = v;
            });
        }
        const auto copy = [&](uint32_t original) {
            const uint32_t i = copies++;
            values[2u * i] = values[2u * original];
            values[2u * i + 1u] = values[2u * original + 1u];
            links[3u * capacity + i - n] = corner(original);
            return i;
        };
        const uint32_t start_copy = copy(start), connector_copy = copy(connector);
        const uint32_t start_next = next(start), connector_prev = prev(connector);
        link(start_copy, start_next);
        link(connector_prev, connector_copy);
        link(start, connector);
        link(connector_copy, start_copy);
        clip_degenerate(start);
        clip_degenerate(start_copy);
        clip_degenerate(connector);
        clip_degenerate(connector_copy);
    }

    // An ear's cost is below -epsilon when it is valid.
    const auto ear_cost = [&](uint32_t i) {
        const uint32_t l = prev(i), r = next(i);
        const PolygonPoint pl = point(l), pi = point(i), pr = point(r);
        const PolygonPoint left_direction = PolygonUnit(pi - pl), right_direction = PolygonUnit(pr - pi);
        float total = PolygonDot(left_direction, right_direction) - 1.f - epsilon;
        if (PolygonCCW(pi, pl, pr, epsilon) == 0) return total;
        const PolygonPoint open = pl - pr, center = (pl + pr) * .5f;
        const float open2 = PolygonDot(open, open), scale = 4.f / open2, radius = PolygonSqrt(open2) / 2.f;
        const PolygonPoint side = PolygonUnit(open);
        const PolygonPoint low{PolygonMin(center.x - radius, pi.x) - epsilon, PolygonMin(center.y - radius, pi.y) - epsilon};
        const PolygonPoint high{PolygonMax(center.x + radius, pi.x) + epsilon, PolygonMax(center.y + radius, pi.y) + epsilon};
        const auto boxed = [&](PolygonPoint q) { return !(q.x < low.x || q.y < low.y || q.x > high.x || q.y > high.y); };
        const auto add = [&](uint32_t t, PolygonPoint q) {
            const auto signed_distance = [&](PolygonPoint unit) {
                const float d = PolygonDet(unit, q - pi);
                if (PolygonAbs(d) < epsilon) {
                    const float d_right = PolygonDet(unit, point(next(t)) - pi);
                    if (PolygonAbs(d_right) > epsilon) return d_right;
                    const float d_left = PolygonDet(unit, point(prev(t)) - pi);
                    if (PolygonAbs(d_left) > epsilon) return d_left;
                }
                return d;
            };
            float cost = PolygonMin(PolygonMin(signed_distance(right_direction), signed_distance(left_direction)), PolygonDet(side, q - pr));
            if (cost < -epsilon) {
                const PolygonPoint offset = q - center;
                cost = -epsilon - scale * PolygonDot(offset, offset);
            }
            total = PolygonMax(total, cost);
        };
        if (one_loop) {
            // Each block's points load before any is tested, so the reads overlap.
            for (uint32_t first = lanes.Lane; first < n; first += PolygonScanBlock * lanes.Count) {
                PolygonPoint block[PolygonScanBlock];
                for (uint32_t k = 0u; k < PolygonScanBlock; ++k) block[k] = point(PolygonCornerIn(first + k * lanes.Count, n));
                for (uint32_t k = 0u; k < PolygonScanBlock && first + k * lanes.Count < n; ++k) {
                    const uint32_t t = first + k * lanes.Count;
                    if (t != i && t != l && t != r && boxed(block[k]) && !clipped(t)) add(t, block[k]);
                }
            }
            return lanes.Max(total);
        }
        const uint32_t ci = corner(i), cl = corner(l), cr = corner(r);
        for (uint32_t t = next(r); t != l && t != i; t = next(t)) {
            const uint32_t ct = corner(t);
            if (ct == ci || ct == cl || ct == cr) continue;
            const PolygonPoint q = point(t);
            if (boxed(q)) add(t, q);
        }
        return total;
    };
    // Queued ears pop by cost, and equal costs pop in insertion order.
    uint32_t sequence = 0u;
    const auto process = [&](uint32_t i) {
        links[2u * capacity + i] = InvalidOffset;
        if (is_short(i)) {
            values[2u * capacity + i] = -FLT_MAX;
            links[2u * capacity + i] = sequence++;
        } else if (PolygonCCW(point(prev(i)), point(i), point(next(i)), 2.f * epsilon) >= 0) {
            values[2u * capacity + i] = ear_cost(i);
            links[2u * capacity + i] = sequence++;
        }
    };
    const auto triangulate = [&](uint32_t start) {
        // A clipped corner of the one loop holds no insertion order, so the queue scan reads only live ears.
        if (one_loop) {
            for (uint32_t c = 0u; c < n; ++c) links[2u * capacity + c] = InvalidOffset;
        }
        int remaining = -2;
        uint32_t i = loop(start, [&](uint32_t c) {
            process(c);
            ++remaining;
        });
        if (i == capacity) return;
        while (remaining > 0) {
            uint32_t ear = capacity;
            if (one_loop) {
                float best_cost = 0.f;
                uint32_t best_order = InvalidOffset;
                for (uint32_t first = lanes.Lane; first < n; first += PolygonScanBlock * lanes.Count) {
                    uint32_t orders[PolygonScanBlock];
                    float costs[PolygonScanBlock];
                    for (uint32_t k = 0u; k < PolygonScanBlock; ++k) {
                        const uint32_t c = PolygonCornerIn(first + k * lanes.Count, n);
                        orders[k] = links[2u * capacity + c];
                        costs[k] = values[2u * capacity + c];
                    }
                    for (uint32_t k = 0u; k < PolygonScanBlock && first + k * lanes.Count < n; ++k) {
                        if (orders[k] != InvalidOffset && (ear == capacity || costs[k] < best_cost || (costs[k] == best_cost && orders[k] < best_order))) {
                            ear = first + k * lanes.Count;
                            best_cost = costs[k];
                            best_order = orders[k];
                        }
                    }
                }
                ear = lanes.Least(best_cost, best_order, ear);
            } else {
                uint32_t c = i;
                do {
                    const uint32_t order = links[2u * capacity + c];
                    if (order != InvalidOffset &&
                        (ear == capacity || values[2u * capacity + c] < values[2u * capacity + ear] || (values[2u * capacity + c] == values[2u * capacity + ear] && order < links[2u * capacity + ear]))) ear = c;
                    c = next(c);
                } while (c != i);
            }
            // Without a queued ear the loop is invalid, and the corner after the last clip clips to keep the result manifold.
            if (ear != capacity) {
                i = ear;
                links[2u * capacity + i] = InvalidOffset;
            }
            clip(i);
            --remaining;
            process(prev(i));
            process(next(i));
            i = next(i);
        }
    };
    for (uint32_t l = 0u; l < loop_count; ++l) {
        if (kind(l) == simple || kind(l) == outer) triangulate(loops[l]);
    }
    for (uint32_t l = rightmost(unjoined); l != loop_count; l = rightmost(unjoined)) {
        loops[loop_count + l] = empty;
        triangulate(loops[l]);
    }
}

// Words of `links`, and again floats of `values`, that TriangulatePolygon uses per corner.
GPU_CONSTANT uint32_t PolygonTriangulationWords = 3u;
// The largest corner count a polygon triangulates with in thread memory, which needs no scratch.
GPU_CONSTANT uint32_t PolygonThreadCorners = 8u;

// Writes the projection of a polygon's corners onto the plane through its first corner normal to its vector area to `values`, and returns its epsilon.
// The epsilon is RelativePrecision times the corners' largest coordinate magnitude.
template<typename Position, typename Values>
inline float ProjectPolygon(uint32_t n, Position position, Values values) {
    const PolygonVector origin = PolygonVector(position(0u));
    PolygonVector area{0.f, 0.f, 0.f}, previous{0.f, 0.f, 0.f};
    float magnitude = PolygonMax(PolygonMax(PolygonAbs(origin.x), PolygonAbs(origin.y)), PolygonAbs(origin.z));
    for (uint32_t i = 1u; i < n; ++i) {
        const PolygonVector p = PolygonVector(position(i));
        magnitude = PolygonMax(magnitude, PolygonMax(PolygonMax(PolygonAbs(p.x), PolygonAbs(p.y)), PolygonAbs(p.z)));
        const PolygonVector relative = p - origin;
        area += PolygonCross(previous, relative);
        previous = relative;
    }
    // A right-handed frame about the vector area keeps the loop counterclockwise.
    const float area_length = PolygonSqrt(PolygonDot3(area, area));
    const PolygonVector normal = area_length > 0.f ? area / area_length : PolygonVector{0.f, 0.f, 1.f};
    const PolygonVector axis = PolygonAbs(normal.x) < .5f ? PolygonVector{1.f, 0.f, 0.f} : PolygonVector{0.f, 1.f, 0.f};
    const PolygonVector u_direction = PolygonCross(axis, normal);
    const PolygonVector u = u_direction / PolygonSqrt(PolygonDot3(u_direction, u_direction)), v = PolygonCross(normal, u);
    for (uint32_t i = 0u; i < n; ++i) {
        const PolygonVector relative = PolygonVector(position(i)) - origin;
        values[2u * i] = PolygonDot3(relative, u);
        values[2u * i + 1u] = PolygonDot3(relative, v);
    }
    return RelativePrecision * magnitude;
}

// Emits the triangles of a projected convex polygon, and returns whether it emitted them.
// A corner is real when its next corner lies more than epsilon left of its previous edge's line, which makes its next edge longer than epsilon.
// A corner is straight when it is not real, lies within epsilon of the line through its neighbors, turns by less than a right angle and its next edge is longer than epsilon.
// A sliver is a triangle whose every corner lies within epsilon of the line through the other two.
// A polygon of real corners takes the zigzag fan of its corners.
// A polygon of real and straight corners takes the zigzag fan of its corners that are real both ways, whose previous corner also lies more than epsilon left of their next edge's line.
// Every other corner lies on the edge between the two such corners around it, and that edge's corners fan from the opposite corner of its zigzag triangle.
// When a second or third edge of that triangle holds corners, they fan from the corner of the first fan beside them.
// So the triangles use every corner, and they tile the polygon when each turns counterclockwise.
// The zigzag starts where its first and last triangles hold the fewest such edges, since their fans join nearby corners.
// It emits nothing when fewer than three corners are real both ways or a triangle is clockwise or a sliver.
// The caller owns as many words of `links` as the polygon has corners.
template<typename Links, typename Values, typename Emit>
inline bool TriangulateConvexPolygon(uint32_t n, Links links, Values values, float epsilon, Emit emit) {
    const auto point = [&](uint32_t i) { return PolygonPoint{values[2u * i], values[2u * i + 1u]}; };
    const auto next = [&](uint32_t i) { return i + 1u < n ? i + 1u : 0u; };
    const auto prev = [&](uint32_t i) { return i ? i - 1u : n - 1u; };
    // Calls `visit(i, real, straight, turn, edge)` with each corner in order until it returns false, and returns whether it never did.
    // The turn is the determinant of the corner's previous and next edges.
    const auto corners = [&](auto visit) {
        const PolygonPoint first_edge = point(0u) - point(n - 1u);
        PolygonPoint last_edge = first_edge;
        for (uint32_t i = 0u; i < n; ++i) {
            const PolygonPoint edge = i + 1u < n ? point(i + 1u) - point(i) : first_edge, chord = last_edge + edge;
            const float turn = PolygonDet(last_edge, edge);
            const bool real = turn > epsilon * PolygonSqrt(PolygonDot(last_edge, last_edge));
            const bool straight = !real && turn >= -epsilon * PolygonSqrt(PolygonDot(chord, chord)) && PolygonDot(last_edge, edge) > 0.f && PolygonDot(edge, edge) > epsilon * epsilon;
            if (!visit(i, real, straight, turn, edge)) return false;
            last_edge = edge;
        }
        return true;
    };
    uint32_t real_count = 0u;
    if (!corners([&](uint32_t, bool real, bool straight, float, PolygonPoint) {
            real_count += real ? 1u : 0u;
            return real || straight;
        })) return false;
    // Calls `visit(i, j, k)` with the zigzag fan of `count` corners.
    const auto zigzag = [](uint32_t count, auto visit) {
        uint32_t low = 0u, high = count - 1u;
        bool right = true;
        while (low + 1u < high) {
            const uint32_t middle = right ? low + 1u : high - 1u;
            visit(low, middle, high);
            if (right) low = middle;
            else high = middle;
            right = !right;
        }
    };
    if (real_count == n) {
        uint32_t t = 0u;
        zigzag(n, [&](uint32_t i, uint32_t j, uint32_t k) { emit(uvec3{i, j, k}, t++); });
        return true;
    }
    real_count = 0u;
    corners([&](uint32_t i, bool real, bool, float turn, PolygonPoint edge) {
        if (real && turn > epsilon * PolygonSqrt(PolygonDot(edge, edge))) links[real_count++] = i;
        return true;
    });
    if (real_count < 3u) return false;
    // The number of edges holding corners among the two at zigzag corner `r`.
    const auto held = [&](uint32_t r) { return uint32_t(next(links[r]) != links[r + 1u < real_count ? r + 1u : 0u]); };
    const auto held_at = [&](uint32_t r) { return held(r ? r - 1u : real_count - 1u) + held(r); };
    uint32_t start = 0u, fewest = 5u;
    for (uint32_t r = 0u; r < real_count && fewest; ++r) {
        const uint32_t at_ends = held_at(r) + held_at((r + real_count / 2u) % real_count);
        if (at_ends < fewest) start = r, fewest = at_ends;
    }
    // Calls `visit(a, b, c)` with each triangle in emission order.
    const auto triangulate = [&](auto visit) {
        zigzag(real_count, [&](uint32_t i, uint32_t j, uint32_t k) {
            const uint32_t abc[3] = {links[(start + i) % real_count], links[(start + j) % real_count], links[(start + k) % real_count]};
            // An edge between consecutive zigzag corners holds the corners after its first.
            const bool holds[3] = {j == i + 1u && next(abc[0]) != abc[1], k == j + 1u && next(abc[1]) != abc[2], i == 0u && k == real_count - 1u && next(abc[2]) != abc[0]};
            if (!holds[0] && !holds[1] && !holds[2]) {
                visit(abc[0], abc[1], abc[2]);
                return;
            }
            const uint32_t rotation = holds[0] ? 0u : 2u - uint32_t(holds[1]);
            const uint32_t x = abc[rotation], y = abc[(rotation + 1u) % 3u], z = abc[(rotation + 2u) % 3u];
            const uint32_t first = next(x), last = prev(y);
            if (holds[(rotation + 2u) % 3u]) {
                for (uint32_t s = z; s != x; s = next(s)) visit(first, s, next(s));
            } else {
                visit(x, first, z);
            }
            for (uint32_t s = first; s != last; s = next(s)) visit(s, next(s), z);
            if (holds[(rotation + 1u) % 3u]) {
                for (uint32_t s = y; s != z; s = next(s)) visit(last, s, next(s));
            } else {
                visit(last, y, z);
            }
        });
    };
    bool valid = true;
    triangulate([&](uint32_t a, uint32_t b, uint32_t c) {
        const PolygonPoint ab = point(b) - point(a), ac = point(c) - point(a), bc = point(c) - point(b);
        const float area = PolygonDet(ab, ac), shortest = PolygonMin(PolygonMin(PolygonDot(ab, ab), PolygonDot(ac, ac)), PolygonDot(bc, bc));
        valid = valid && area > 0.f && area * area > epsilon * epsilon * shortest;
    });
    if (!valid) return false;
    uint32_t t = 0u;
    triangulate([&](uint32_t a, uint32_t b, uint32_t c) { emit(uvec3{a, b, c}, t++); });
    return true;
}

// Clips the ears of a projected polygon as manifold's EarClip does, with `lanes` running the clip together.
template<typename Links, typename Values, typename Emit, typename Lanes = PolygonLane>
inline void ClipPolygon(uint32_t n, Links links, Values values, float epsilon, Emit emit, Lanes lanes = {}) {
    for (uint32_t i = 0u; i < n; ++i) {
        links[i] = (i + 1u) % n;
        links[n + i] = (i + n - 1u) % n;
    }
    uint32_t loops[2] = {0u, 0u};
    TriangulateLoops(n, n, 1u, links, values, loops, epsilon, emit, lanes);
}

// Triangulates a polygon of `n` corners as manifold::Triangulate does, on its projection, except for a convex polygon with a corner within epsilon of straight.
// The caller owns PolygonTriangulationWords * n words of `links` and as many floats of `values`, in host memory, thread memory, or GPU scratch.
// Each triangle is emitted counterclockwise in the loop's winding with its index.
template<typename Position, typename Links, typename Values, typename Emit>
inline void TriangulatePolygon(uint32_t n, Position position, Links links, Values values, Emit emit) {
    if (n < 3u) return;
    if (n == 3u) {
        emit(uvec3{0u, 1u, 2u}, 0u);
        return;
    }
    const float epsilon = ProjectPolygon(n, position, values);
    if (!TriangulateConvexPolygon(n, links, values, epsilon, emit)) ClipPolygon(n, links, values, epsilon, emit);
}

#ifdef __METAL_VERSION__
// The lanes of a simdgroup clipping one polygon together, each scanning every Count-th corner.
struct PolygonSimdgroup {
    uint32_t Lane, Count;
    float Max(float value) const { return simd_max(value); }
    // The corner of the least cost, and of the least insertion order among equal costs, over the lanes' candidates.
    uint32_t Least(float cost, uint32_t order, uint32_t corner) const {
        for (uint32_t offset = 1u; offset < Count; offset <<= 1u) {
            const float other_cost = simd_shuffle_xor(cost, ushort(offset));
            const uint32_t other_order = simd_shuffle_xor(order, ushort(offset)), other_corner = simd_shuffle_xor(corner, ushort(offset));
            if (other_order != InvalidOffset && (order == InvalidOffset || other_cost < cost || (other_cost == cost && other_order < order))) {
                cost = other_cost;
                order = other_order;
                corner = other_corner;
            }
        }
        return corner;
    }
};

// The scan steps each lane of a shared clip spends per corner on the clip's serial work, where one lane clipping n corners alone takes about n squared steps.
constant uint32_t PolygonSharedCornerSteps = 12u;

// Triangulates the polygon of each lane where `active` as TriangulatePolygon does.
// `polygon(index, visit)` returns `visit(n, position, links, values, emit)` for polygon `index`, with device scratch of PolygonTriangulationWords words of `links` and as many floats of `values` per corner.
// Polygons up to PolygonThreadCorners corners triangulate in thread memory, and larger convex ones in their scratch.
// The whole simdgroup clips the remaining polygons one at a time when the estimated steps end sooner than each lane clipping its own, and each of its lanes writes every value the clip writes.
// Every lane of the simdgroup calls it.
template<typename Polygon>
inline void TriangulatePolygons(bool active, uint index, uint lane, uint count, Polygon polygon) {
    uint n = 0u;
    const bool deferred = active && polygon(index, [&](uint corners, auto position, device uint *links, device float *values, auto emit) {
        n = corners;
        if (corners > PolygonThreadCorners) return !TriangulateConvexPolygon(corners, links, values, ProjectPolygon(corners, position, values), emit);
        uint local_links[PolygonTriangulationWords * PolygonThreadCorners];
        float local_values[PolygonTriangulationWords * PolygonThreadCorners];
        TriangulatePolygon(corners, position, local_links, local_values, emit);
        return false;
    });
    const auto clip = [&](uint k, auto lanes) {
        polygon(k, [&](uint corners, auto position, device uint *links, device float *values, auto emit) {
            ClipPolygon(corners, links, values, ProjectPolygon(corners, position, values), emit, lanes);
            return false;
        });
    };
    const float own = deferred ? float(n) * float(n) : 0.f;
    const float shared = deferred ? own / float(count) + float(PolygonSharedCornerSteps * n) : 0.f;
    if (simd_sum(shared) >= simd_max(own)) {
        if (deferred) clip(index, PolygonLane{});
        return;
    }
    for (auto mask = static_cast<simd_vote::vote_t>(simd_ballot(deferred)); mask; mask &= mask - 1u) clip(simd_shuffle(index, ushort(ctz(mask))), PolygonSimdgroup{lane, count});
}
#endif
