#include "mesh/Partition.h"
#include "numeric/VectorMath.h"
#include <cmath>

namespace {
using ivec4 = std::array<int32_t, 4>;
using bvec4 = std::array<bool, 4>;

vec4 Lerp(vec4 a, vec4 b, float t) { return a * (1.f - t) + b * t; }

struct Builder {
    std::vector<vec4> Weights;
    std::vector<std::array<uint32_t, 3>> Triangles;

    void Triangle(int32_t a, int32_t b, int32_t c) { Triangles.push_back({uint32_t(a), uint32_t(b), uint32_t(c)}); }
    // Side 0 has added points while sides 1 and 2 do not, and the fan spreads from corner 2.
    void Fan(std::array<int32_t, 3> corners, int32_t added, int32_t edge_offset) {
        int32_t last = corners[0];
        for (int32_t i = 0; i < added; ++i) {
            const int32_t next = edge_offset + i;
            Triangle(last, next, corners[2]);
            last = next;
        }
        Triangle(last, corners[1], corners[2]);
    }
    void Quad(ivec4 corners, ivec4 edge_offsets, ivec4 added, bvec4 forward) {
        const auto edge_vertex = [&](int32_t edge, int32_t index) { return edge_offsets[edge] + (forward[edge] ? 1 : -1) * index; };
        int32_t corner = -1, last = 3, max_edge = -1;
        for (const int32_t i : {0, 1, 2, 3}) {
            if (corner == -1 && added[i] == 0 && added[last] == 0) corner = i;
            if (added[i] > 0) max_edge = max_edge == -1 ? i : -2;
            last = i;
        }
        if (corner >= 0) {
            if (max_edge >= 0) {
                const ivec4 edge{max_edge % 4, (1 + max_edge) % 4, (2 + max_edge) % 4, (3 + max_edge) % 4};
                const int32_t middle = added[max_edge] / 2;
                Triangle(corners[edge[2]], corners[edge[3]], edge_vertex(max_edge, middle));
                int32_t previous = corners[edge[0]];
                for (int32_t i = 0; i <= middle; ++i) {
                    const int32_t next = edge_vertex(max_edge, i);
                    Triangle(corners[edge[3]], previous, next);
                    previous = next;
                }
                previous = corners[edge[1]];
                for (int32_t i = added[max_edge] - 1; i >= middle; --i) {
                    const int32_t next = edge_vertex(max_edge, i);
                    Triangle(corners[edge[2]], next, previous);
                    previous = next;
                }
            } else {
                int32_t side_vertex = corners[0];
                for (const int32_t j : {1, 2}) {
                    const int32_t side = (corner + j) % 4;
                    if (j == 2 && added[side] > 0) Triangle(corners[side], edge_vertex(side, 0), side_vertex);
                    else side_vertex = corners[side];
                    for (int32_t i = 0; i < added[side]; ++i) {
                        const int32_t next = edge_vertex(side, i);
                        Triangle(corners[corner], side_vertex, next);
                        side_vertex = next;
                    }
                    if (j == 2 || added[side] == 0) Triangle(corners[corner], side_vertex, corners[(corner + j + 1) % 4]);
                }
            }
            return;
        }
        const int32_t partitions = 1 + std::min(added[1], added[3]);
        ivec4 new_corners{corners[1], -1, -1, corners[0]};
        ivec4 new_offsets{edge_offsets[1], -1, edge_vertex(3, added[3] + 1), edge_offsets[0]};
        ivec4 new_added{0, -1, 0, added[0]};
        bvec4 new_forward{forward[1], true, forward[3], forward[0]};
        for (int32_t i = 1; i < partitions; ++i) {
            const int32_t corner_offset1 = (added[1] * i) / partitions, corner_offset3 = added[3] - 1 - (added[3] * i) / partitions;
            const int32_t next_offset1 = edge_vertex(1, corner_offset1 + 1), next_offset3 = edge_vertex(3, corner_offset3 + 1);
            const auto points = int32_t(std::round(std::lerp(double(added[0]), double(added[2]), double(i) / partitions)));
            new_corners[1] = edge_vertex(1, corner_offset1);
            new_corners[2] = edge_vertex(3, corner_offset3);
            new_added[0] = std::abs(next_offset1 - new_offsets[0]) - 1;
            new_added[1] = points;
            new_added[2] = std::abs(next_offset3 - new_offsets[2]) - 1;
            new_offsets[1] = int32_t(Weights.size());
            new_offsets[2] = next_offset3;
            for (int32_t j = 0; j < points; ++j) Weights.push_back(Lerp(Weights[new_corners[1]], Weights[new_corners[2]], float((j + 1.0) / (points + 1.0))));
            Quad(new_corners, new_offsets, new_added, new_forward);
            new_corners[0] = new_corners[1];
            new_corners[3] = new_corners[2];
            new_added[3] = new_added[1];
            new_offsets[0] = next_offset1;
            new_offsets[3] = new_offsets[1] + new_added[1] - 1;
            new_forward[3] = false;
        }
        new_corners[1] = corners[2];
        new_corners[2] = corners[3];
        new_offsets[1] = edge_offsets[2];
        new_added[0] = added[1] - std::abs(new_offsets[0] - edge_offsets[1]);
        new_added[1] = added[2];
        new_added[2] = std::abs(new_offsets[2] - edge_offsets[3]) - 1;
        new_offsets[2] = edge_offsets[3];
        new_forward[1] = forward[2];
        Quad(new_corners, new_offsets, new_added, new_forward);
    }
};
} // namespace

PartitionTable Partition(std::array<uint32_t, 4> n) {
    Builder b;
    const uint32_t corners = n[3] ? 4u : 3u;
    for (uint32_t i = 0u; i < corners; ++i) b.Weights.push_back({i == 0u, i == 1u, i == 2u, i == 3u});
    for (uint32_t i = 0u; i < corners; ++i) {
        const auto next = b.Weights[(i + 1u) % corners];
        for (uint32_t j = 1u; j < n[i]; ++j) b.Weights.push_back(Lerp(b.Weights[i], next, float(j) / float(n[i])));
    }
    if (n[3] > 0u) {
        ivec4 offsets{4, 0, 0, 0};
        for (const int32_t i : {1, 2, 3}) offsets[i] = offsets[i - 1] + int32_t(n[i - 1]) - 1;
        b.Quad({0, 1, 2, 3}, offsets, {int32_t(n[0]) - 1, int32_t(n[1]) - 1, int32_t(n[2]) - 1, int32_t(n[3]) - 1}, {true, true, true, true});
    } else {
        const std::array<int32_t, 3> offsets{3, 3 + int32_t(n[0]) - 1, 3 + int32_t(n[0]) - 1 + int32_t(n[1]) - 1};
        const double f = double(n[2]) * n[2] + double(n[0]) * n[0];
        if (n[1] == 1u) {
            if (n[0] == 1u) b.Triangle(0, 1, 2);
            else b.Fan({0, 1, 2}, int32_t(n[0]) - 1, offsets[0]);
        } else if (double(n[1]) * n[1] > f - std::sqrt(2.0) * n[0] * n[2]) {
            b.Triangle(offsets[1] - 1, 1, offsets[1]);
            b.Quad({offsets[1] - 1, offsets[1], 2, 0}, {-1, offsets[1] + 1, offsets[2], offsets[0]}, {0, int32_t(n[1]) - 2, int32_t(n[2]) - 1, int32_t(n[0]) - 2}, {true, true, true, true});
        } else {
            // An obtuse triangle splits into two acute parts along a height from corner 2 onto side 0.
            const int32_t ns = std::min(int32_t(n[0]) - 2, int32_t(std::round((f - double(n[1]) * n[1]) / (2.0 * n[0]))));
            const int32_t nh = std::max(1, int32_t(std::round(std::sqrt(double(n[2]) * n[2] - double(ns) * ns))));
            const int32_t h_offset = int32_t(b.Weights.size());
            const auto middle = b.Weights[offsets[0] + ns - 1];
            for (int32_t j = 1; j < nh; ++j) b.Weights.push_back(Lerp(b.Weights[2], middle, float(j) / float(nh)));
            b.Triangle(offsets[1] - 1, 1, offsets[1]);
            b.Quad({offsets[1] - 1, offsets[1], 2, offsets[0] + ns - 1}, {-1, offsets[1] + 1, h_offset, offsets[0] + ns}, {0, int32_t(n[1]) - 2, nh - 1, int32_t(n[0]) - ns - 2}, {true, true, true, true});
            if (n[2] == 1u) b.Fan({0, offsets[0] + ns - 1, 2}, ns - 1, offsets[0]);
            else if (ns == 1) {
                b.Triangle(h_offset, 2, offsets[2]);
                b.Quad({h_offset, offsets[2], 0, offsets[0]}, {-1, offsets[2] + 1, -1, h_offset + nh - 2}, {0, int32_t(n[2]) - 2, ns - 1, nh - 2}, {true, true, true, false});
            } else {
                b.Triangle(h_offset - 1, 0, offsets[0]);
                b.Quad({h_offset - 1, offsets[0], offsets[0] + ns - 1, 2}, {-1, offsets[0] + 1, h_offset + nh - 2, offsets[2]}, {0, ns - 2, nh - 1, int32_t(n[2]) - 2}, {true, true, false, true});
            }
        }
    }
    return {n, std::move(b.Weights), std::move(b.Triangles)};
}
