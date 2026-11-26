#include "mesh/Primitives.h"
#include "numeric/VectorMath.h"
#include <numbers>
#include <ranges>
#include <unordered_map>
using std::views::transform, std::ranges::iota_view, std::ranges::to;

namespace primitive {
namespace {
// The layouts give texture coordinates with V up, and the store holds them with V down.
vec2 StoredUv(float u, float v) { return {u, 1.f - v}; }

// The grid offsets of a grid cell's corners in loop order.
constexpr std::array<std::array<uint32_t, 2>, 4> CellCorners{{{0u, 0u}, {1u, 0u}, {1u, 1u}, {0u, 1u}}};

MeshSource Generate(const Plane &p, bool texcoords) {
    const auto [half, n] = p;
    std::vector<vec3> vertices;
    vertices.reserve((n.x + 1) * (n.y + 1));
    for (uint32_t j = 0; j <= n.y; ++j) {
        for (uint32_t i = 0; i <= n.x; ++i) vertices.emplace_back(half.x * (2.f * i / n.x - 1.f), 0, half.y * (1.f - 2.f * j / n.y));
    }
    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(n.x * n.y);
    std::vector<vec2> uvs;
    uvs.reserve(texcoords ? 4u * n.x * n.y : 0u);
    // The grid spans the unit square, with U along X and V toward -Z.
    for (uint32_t j = 0; j < n.y; ++j) {
        for (uint32_t i = 0; i < n.x; ++i) {
            const uint32_t v = j * (n.x + 1) + i;
            faces.push_back({v, v + 1, v + n.x + 2, v + n.x + 1});
            if (!texcoords) continue;
            for (const auto [a, b] : CellCorners) uvs.push_back(StoredUv(float(i + a) / float(n.x), float(j + b) / float(n.y)));
        }
    }
    return {.Data = {std::move(vertices), faces}, .CornerTexCoords0 = std::move(uvs)};
}

// Fills the ring with a face or a fan around a new center vertex, whose corners take the ring's texture coordinates and the center's where `ring_uvs` holds them.
void Close(MeshSource &source, FillType fill, std::span<const uint32_t> ring, std::span<const vec2> ring_uvs, vec3 center, vec2 center_uv) {
    auto &data = source.Data;
    if (fill == FillType::Face) {
        data.AddFace(ring);
        source.CornerTexCoords0.append_range(ring_uvs);
    }
    if (fill == FillType::Fan) {
        const auto c = uint32_t(data.Positions.size());
        data.Positions.push_back(center);
        for (uint32_t i = 0; i < ring.size(); ++i) {
            const auto next = (i + 1) % ring.size();
            data.AddFace(std::array{ring[i], ring[next], c});
            if (!ring_uvs.empty()) source.CornerTexCoords0.append_range(std::array{ring_uvs[i], ring_uvs[next], center_uv});
        }
    }
}

MeshSource Generate(const Circle &p, bool texcoords) {
    const auto [radius, n, fill] = p;
    MeshSource source{.Data = MeshData{iota_view{0u, n} | transform([radius, n](uint32_t i) { return vec3{radius * __cospi(2.f * i / n), 0, -radius * __sinpi(2.f * i / n)}; }) | to<std::vector>()}};
    // The disc projects from above onto the unit square.
    const auto uvs = iota_view{0u, texcoords ? n : 0u} | transform([n](uint32_t i) { return StoredUv(.5f + .5f * __cospi(2.f * i / n), .5f + .5f * __sinpi(2.f * i / n)); }) | to<std::vector>();
    Close(source, fill, iota_view{0u, n} | to<std::vector>(), uvs, {0, 0, 0}, StoredUv(.5f, .5f));
    if (fill == FillType::None) source.Data.Edges = iota_view{0u, n} | transform([n](uint32_t i) { return std::array{i, (i + 1) % n}; }) | to<std::vector>();
    return source;
}

// A face of the cube layout, a cross of quarter squares: the axes its U and V run along on the cube, each in its positive or negative direction, and its lower left texture coordinates.
struct CubeFaceLayout {
    uint32_t U;
    bool UPositive;
    uint32_t V;
    bool VPositive;
    vec2 Origin;
};
// The faces by the axis they face and their side, negative first.
constexpr std::array<CubeFaceLayout, 6> CubeLayout{{
    {1u, true, 2u, false, {.375f, 0.f}},
    {1u, true, 2u, true, {.375f, .5f}},
    {0u, true, 2u, true, {.125f, .5f}},
    {0u, false, 2u, true, {.625f, .5f}},
    {1u, true, 0u, true, {.375f, .25f}},
    {1u, true, 0u, false, {.375f, .75f}},
}};
// The texture coordinates of the point at grid index `g` of a cube whose faces are n by n grids, on the face facing the axis on the side.
vec2 CubeFaceUv(uint32_t axis, bool positive, std::array<uint32_t, 3> g, uint32_t n) {
    const auto &face = CubeLayout[2u * axis + uint32_t(positive)];
    const auto steps = [&](uint32_t a, bool forward) { return float(forward ? g[a] : n - g[a]); };
    return StoredUv(face.Origin.x + .25f * steps(face.U, face.UPositive) / float(n), face.Origin.y + .25f * steps(face.V, face.VPositive) / float(n));
}

MeshSource Generate(const Cube &p, bool texcoords) {
    const auto x = p.HalfExtents.x, y = p.HalfExtents.y, z = p.HalfExtents.z;
    // Each vertex as its grid index on each axis, zero on the negative side.
    constexpr std::array<std::array<uint32_t, 3>, 8> Corners{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
    // Each face as its vertices and the axis and side it faces.
    struct Face {
        std::array<uint32_t, 4> Vertices;
        uint32_t Axis;
        bool Positive;
    };
    constexpr std::array<Face, 6> Faces{{
        {{0, 3, 2, 1}, 2, false},
        {{4, 5, 6, 7}, 2, true},
        {{0, 1, 5, 4}, 1, false},
        {{3, 7, 6, 2}, 1, true},
        {{0, 4, 7, 3}, 0, false},
        {{1, 2, 6, 5}, 0, true},
    }};
    MeshSource source;
    for (const auto g : Corners) source.Data.Positions.push_back({g[0] ? x : -x, g[1] ? y : -y, g[2] ? z : -z});
    for (const auto &face : Faces) {
        source.Data.AddFace(face.Vertices);
        if (!texcoords) continue;
        for (const auto v : face.Vertices) source.CornerTexCoords0.push_back(CubeFaceUv(face.Axis, face.Positive, Corners[v], 1u));
    }
    return source;
}

// The net of the icosahedron, eleven columns wide over rows of equilateral triangles, as the column and row of each face's corners.
constexpr std::array<std::array<std::array<uint8_t, 2>, 3>, 20> IcoNet{{
    {{{1, 3}, {0, 2}, {2, 2}}},
    {{{3, 3}, {2, 2}, {4, 2}}},
    {{{5, 3}, {4, 2}, {6, 2}}},
    {{{7, 3}, {6, 2}, {8, 2}}},
    {{{9, 3}, {8, 2}, {10, 2}}},
    {{{4, 2}, {2, 2}, {3, 1}}},
    {{{2, 2}, {0, 2}, {1, 1}}},
    {{{10, 2}, {8, 2}, {9, 1}}},
    {{{8, 2}, {6, 2}, {7, 1}}},
    {{{6, 2}, {4, 2}, {5, 1}}},
    {{{2, 0}, {3, 1}, {1, 1}}},
    {{{10, 0}, {11, 1}, {9, 1}}},
    {{{8, 0}, {9, 1}, {7, 1}}},
    {{{6, 0}, {7, 1}, {5, 1}}},
    {{{4, 0}, {5, 1}, {3, 1}}},
    {{{1, 1}, {3, 1}, {2, 2}}},
    {{{9, 1}, {11, 1}, {10, 2}}},
    {{{7, 1}, {9, 1}, {8, 2}}},
    {{{5, 1}, {7, 1}, {6, 2}}},
    {{{3, 1}, {5, 1}, {4, 2}}},
}};

MeshSource Generate(const IcoSphere &p, bool texcoords) {
    const auto [radius, recursion_level] = p;
    static const float t = std::numbers::phi_v<float>;
    // clang-format off
    std::vector<vec3> vertices{
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
    };
    for (auto &v : vertices) v = Normalize(v);

    std::vector<std::vector<uint32_t>> indices{
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
        {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
        {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
    };
    // clang-format on

    std::unordered_map<uint32_t, uint32_t> cache;
    const auto AddMidVertex = [&](uint32_t p1, uint32_t p2) -> uint32_t {
        const uint32_t key = (std::min(p1, p2) << 16) + std::max(p1, p2);
        if (const auto found = cache.find(key); found != cache.end()) return found->second;

        vertices.emplace_back(Normalize((vertices[p1] + vertices[p2]) / 2.f));

        const uint32_t i = vertices.size() - 1;
        cache[key] = i;
        return i;
    };

    // Each face's corners' texture coordinates with V up, which each subdivision interpolates linearly.
    const auto net = [](std::array<uint8_t, 2> cell) { return vec2{float(cell[0]) / 11.f, float(cell[1]) * std::numbers::sqrt3_v<float> / 11.f}; };
    std::vector<std::array<vec2, 3>> uvs;
    if (texcoords) {
        uvs.reserve(indices.size());
        for (const auto &face : IcoNet) uvs.push_back({net(face[0]), net(face[1]), net(face[2])});
    }
    for (uint32_t r = 0; r < recursion_level; ++r) {
        std::vector<std::vector<uint32_t>> new_indices;
        new_indices.reserve(indices.size() * 4);
        std::vector<std::array<vec2, 3>> new_uvs;
        new_uvs.reserve(uvs.size() * 4);
        for (uint32_t f = 0; f < indices.size(); ++f) {
            const auto &tri = indices[f];
            const uint32_t a = tri[0], b = tri[1], c = tri[2];
            const uint32_t ab = AddMidVertex(a, b), bc = AddMidVertex(b, c), ca = AddMidVertex(c, a);
            new_indices.insert(new_indices.end(), {{a, ab, ca}, {b, bc, ab}, {c, ca, bc}, {ab, bc, ca}});
            if (!texcoords) continue;
            const auto [ua, ub, uc] = uvs[f];
            const auto uab = (ua + ub) * .5f, ubc = (ub + uc) * .5f, uca = (uc + ua) * .5f;
            new_uvs.insert(new_uvs.end(), {{ua, uab, uca}, {ub, ubc, uab}, {uc, uca, ubc}, {uab, ubc, uca}});
        }
        indices = std::move(new_indices);
        uvs = std::move(new_uvs);
    }

    for (auto &v : vertices) v *= radius;

    std::vector<vec2> corner_uvs;
    corner_uvs.reserve(3u * uvs.size());
    for (const auto &face : uvs) {
        for (const auto uv : face) corner_uvs.push_back(StoredUv(uv.x, uv.y));
    }
    return {.Data = {std::move(vertices), std::move(indices)}, .CornerTexCoords0 = std::move(corner_uvs)};
}

MeshSource Generate(const UVSphere &p, bool texcoords) {
    const auto [radius, n_slices, n_stacks] = p;
    std::vector<vec3> vertices;
    vertices.reserve(2 + n_slices * (n_stacks - 2));
    vertices.emplace_back(0, radius, 0);
    for (uint32_t i = 1; i < n_stacks; ++i) {
        const float pp = float(i) / float(n_stacks);
        for (uint32_t j = 0; j < n_slices; ++j) {
            const float tt = 2.f * j / n_slices;
            vertices.emplace_back(vec3{__sinpi(pp) * __cospi(tt), __cospi(pp), __sinpi(pp) * __sinpi(tt)} * radius);
        }
    }
    vertices.emplace_back(0, -radius, 0);

    // U runs east around the axis from the slice nearest -X, and V runs from the bottom pole to the top.
    // The slices between i and i + 1 span column (seam - 1 - i) mod n, and a pole's corner takes its column's middle.
    const uint32_t seam = n_slices / 2u;
    const auto column = [&](uint32_t i) { return float((seam + n_slices - 1u - i) % n_slices); };
    const auto uv = [&](float u_steps, uint32_t stack) { return StoredUv(u_steps / float(n_slices), 1.f - float(stack) / float(n_stacks)); };
    std::vector<std::vector<uint32_t>> indices;
    indices.reserve(2 * n_slices + n_slices * (n_stacks - 2));
    std::vector<vec2> uvs;
    uvs.reserve(texcoords ? 6u * n_slices + 4u * n_slices * (n_stacks - 2) : 0u);
    for (uint32_t i = 0; i < n_slices; ++i) {
        indices.push_back({0, 1 + (i + 1) % n_slices, 1 + i});
        if (texcoords) uvs.append_range(std::array{uv(column(i) + .5f, 0u), uv(column(i), 1u), uv(column(i) + 1.f, 1u)});
    }
    for (uint32_t j = 0; j < n_stacks - 2; ++j) {
        const uint32_t j0 = 1 + j * n_slices, j1 = 1 + (j + 1) * n_slices;
        for (uint32_t i = 0; i < n_slices; ++i) {
            indices.push_back({
                j0 + i,
                j0 + (i + 1) % n_slices,
                j1 + (i + 1) % n_slices,
                j1 + i,
            });
            if (texcoords) uvs.append_range(std::array{uv(column(i) + 1.f, j + 1u), uv(column(i), j + 1u), uv(column(i), j + 2u), uv(column(i) + 1.f, j + 2u)});
        }
    }
    const uint32_t bottom_i = vertices.size() - 1;
    for (uint32_t i = 0; i < n_slices; ++i) {
        indices.push_back({
            bottom_i,
            1 + (n_stacks - 2) * n_slices + i,
            1 + (n_stacks - 2) * n_slices + (i + 1) % n_slices,
        });
        if (texcoords) uvs.append_range(std::array{uv(column(i) + .5f, n_stacks), uv(column(i) + 1.f, n_stacks - 1u), uv(column(i), n_stacks - 1u)});
    }

    return {.Data = {std::move(vertices), std::move(indices)}, .CornerTexCoords0 = std::move(uvs)};
}

MeshSource Generate(const QuadSphere &p, bool texcoords) {
    const auto [radius, n, method] = p;
    // Each axis warps independently by the tangent of its angle from the face center, scaled so a face edge lands on the neighbor's face.
    // The even-area angle, about 49.77 degrees, minimizes area distortion between quads.
    const float angle = method == QuadSphereMethod::EvenAngle ? std::numbers::pi_v<float> / 4 : 0.8687f;
    const auto warp = [angle, n, scale = 1.f / std::tan(angle)](uint32_t i) { return std::tan((2.f * i / n - 1.f) * angle) * scale; };

    std::vector<vec3> vertices;
    std::unordered_map<uint32_t, uint32_t> index;
    const auto vertex = [&](std::array<uint32_t, 3> g) {
        const auto [it, inserted] = index.try_emplace((g[0] * (n + 1) + g[1]) * (n + 1) + g[2], uint32_t(vertices.size()));
        if (inserted) vertices.push_back(Normalize(vec3{warp(g[0]), warp(g[1]), warp(g[2])}) * radius);
        return it->second;
    };
    std::vector<std::vector<uint32_t>> faces;
    std::vector<vec2> uvs;
    uvs.reserve(texcoords ? 24u * n * n : 0u);
    // Each cube face takes its square of the cube layout.
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const uint32_t b = (axis + 1) % 3, c = (axis + 2) % 3;
        for (const uint32_t side : {n, 0u}) {
            const auto grid = [&](uint32_t bi, uint32_t cj) {
                std::array<uint32_t, 3> g;
                g[axis] = side;
                g[b] = bi;
                g[c] = cj;
                return g;
            };
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t j = 0; j < n; ++j) {
                    const std::array corners{grid(i, j), grid(i + 1, j), grid(i + 1, j + 1), grid(i, j + 1)};
                    std::vector quad{vertex(corners[0]), vertex(corners[1]), vertex(corners[2]), vertex(corners[3])};
                    if (side == 0) std::ranges::reverse(quad);
                    faces.push_back(std::move(quad));
                    if (!texcoords) continue;
                    std::array quad_uvs{CubeFaceUv(axis, side == n, corners[0], n), CubeFaceUv(axis, side == n, corners[1], n), CubeFaceUv(axis, side == n, corners[2], n), CubeFaceUv(axis, side == n, corners[3], n)};
                    if (side == 0) std::ranges::reverse(quad_uvs);
                    uvs.append_range(quad_uvs);
                }
            }
        }
    }
    return {.Data = {std::move(vertices), faces}, .CornerTexCoords0 = std::move(uvs)};
}

MeshSource Generate(const Torus &p, bool texcoords) {
    const auto [major_radius, minor_radius, n_major, n_minor] = p;
    std::vector<vec3> vertices;
    vertices.reserve(n_major * n_minor);
    for (uint32_t i = 0; i < n_major; ++i) {
        const float t = 2.f * i / n_major;
        for (uint32_t j = 0; j < n_minor; ++j) {
            const float pp = 2.f * j / n_minor;
            const float r = major_radius + minor_radius * __cospi(pp);
            vertices.emplace_back(r * __sinpi(t), minor_radius * __sinpi(pp), r * __cospi(t));
        }
    }

    std::vector<std::vector<uint32_t>> indices;
    indices.reserve(n_major * n_minor);
    std::vector<vec2> uvs;
    uvs.reserve(texcoords ? 4u * n_major * n_minor : 0u);
    // The layout is a grid over the two angles, with U around the ring and V around the tube.
    // Each angle's first cell starts at the first cell boundary at or past the middle of the texture.
    const auto start = [](uint32_t n) { return (n + 1u) / 2u; };
    for (uint32_t i = 0; i < n_major; ++i) {
        for (uint32_t j = 0; j < n_minor; ++j) {
            indices.push_back({
                i * n_minor + j,
                ((i + 1) % n_major) * n_minor + j,
                ((i + 1) % n_major) * n_minor + (j + 1) % n_minor,
                i * n_minor + (j + 1) % n_minor,
            });
            if (!texcoords) continue;
            const uint32_t u = (start(n_major) + i) % n_major, v = (start(n_minor) + j) % n_minor;
            for (const auto [a, b] : CellCorners) uvs.push_back(StoredUv(float(u + a) / float(n_major), float(v + b) / float(n_minor)));
        }
    }

    return {.Data = {std::move(vertices), std::move(indices)}, .CornerTexCoords0 = std::move(uvs)};
}

MeshSource Generate(const Cylinder &p, bool texcoords) {
    const auto [radius, height, top, slices, cap] = p;
    const float top_radius = top.value_or(radius);
    const bool tip = top_radius <= 0.f;
    MeshSource source;
    auto &data = source.Data;
    const auto ring = [&](float r, float y) {
        for (uint32_t i = 0; i < slices; ++i) data.Positions.emplace_back(r * __cospi(2.f * i / slices), y, r * __sinpi(2.f * i / slices));
    };
    ring(radius, -height / 2);
    if (tip) data.Positions.emplace_back(0, height / 2, 0);
    else ring(top_radius, height / 2);

    // The side unrolls over the top half and each cap is a disc in the bottom half, or the side unrolls over the whole square without caps.
    // A cone's side projects onto the top disc from above.
    const bool capped = cap != FillType::None;
    const float side_bottom = capped ? .5f : 0.f, disc_radius = capped ? .24f : .5f;
    const vec2 top_center{capped ? .25f : .5f, capped ? .25f : .5f}, bottom_center{capped ? .75f : .5f, top_center.y};
    const auto disc = [&](vec2 center, uint32_t i) { return StoredUv(center.x + disc_radius * __cospi(2.f * i / slices), center.y - disc_radius * __sinpi(2.f * i / slices)); };
    const auto discs = [&](vec2 center, auto &&indices) { return texcoords ? indices | transform([&](uint32_t i) { return disc(center, i); }) | to<std::vector>() : std::vector<vec2>{}; };

    const uint32_t ring_top = slices;
    Close(source, cap, iota_view{0u, slices} | to<std::vector>(), discs(bottom_center, iota_view{0u, slices}), {0, -height / 2, 0}, StoredUv(bottom_center.x, bottom_center.y));
    for (uint32_t i = 0; i < slices; ++i) {
        const uint32_t j = (i + 1) % slices;
        if (tip) {
            data.AddFace(std::array{i, ring_top, j});
            if (texcoords) source.CornerTexCoords0.append_range(std::array{disc(top_center, i), StoredUv(top_center.x, top_center.y), disc(top_center, j)});
        } else {
            data.AddFace(std::array{i, ring_top + i, ring_top + j, j});
            const float u0 = 1.f - float(i) / float(slices), u1 = 1.f - float(i + 1u) / float(slices);
            if (texcoords) source.CornerTexCoords0.append_range(std::array{StoredUv(u0, side_bottom), StoredUv(u0, 1.f), StoredUv(u1, 1.f), StoredUv(u1, side_bottom)});
        }
    }
    // The top loop runs in reverse to face outward.
    if (!tip) Close(source, cap, iota_view{0u, slices} | transform([ring_top](uint32_t i) { return ring_top + i; }) | std::views::reverse | to<std::vector>(), discs(top_center, iota_view{0u, slices} | std::views::reverse), {0, height / 2, 0}, StoredUv(top_center.x, top_center.y));
    return source;
}
} // namespace

MeshSource CreateMesh(const PrimitiveParameters &shape, bool texcoords) {
    auto source = std::visit([texcoords](const auto &s) { return Generate(s, texcoords); }, shape);
    source.FlatShaded = true;
    return source;
}

BoneOctahedronData BoneOctahedron(float length) {
    const float d = length * 0.1f;
    const std::array<vec3, 6> unique{
        vec3{0, 0, 0},
        vec3{d, d, d},
        vec3{d, d, -d},
        vec3{-d, d, -d},
        vec3{-d, d, d},
        vec3{0, length, 0},
    };

    // Match Blender's bone_octahedral_solid_tris.
    // clang-format off
    constexpr std::array<std::array<uint32_t, 3>, 8> tris{{
        {2, 1, 0}, {3, 2, 0}, {4, 3, 0}, {1, 4, 0},
        {5, 1, 2}, {5, 2, 3}, {5, 3, 4}, {5, 4, 1},
    }};
    // clang-format on

    // Match Blender's bone_octahedral_solid_normals.
    constexpr float Rsqrt2 = 0.7071068f;
    constexpr std::array<vec3, 8> face_normals{{
        {Rsqrt2, -Rsqrt2, 0},
        {0, -Rsqrt2, -Rsqrt2},
        {-Rsqrt2, -Rsqrt2, 0},
        {0, -Rsqrt2, Rsqrt2},
        {0.9940f, 0.1100f, 0},
        {0, 0.1100f, -0.9940f},
        {-0.9940f, 0.1100f, 0},
        {0, 0.1100f, 0.9940f},
    }};

    std::vector<vec3> positions;
    positions.reserve(30);
    for (const auto &p : unique) positions.push_back(p);
    std::vector<vec3> normals(6, vec3{0});
    normals.reserve(30);

    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(8);
    for (uint32_t f = 0; f < 8; ++f) {
        const uint32_t base = 6 + f * 3;
        for (uint32_t v = 0; v < 3; ++v) {
            positions.push_back(unique[tris[f][v]]);
            normals.push_back(face_normals[f]);
        }
        faces.push_back({base, base + 1, base + 2});
    }

    return {
        .Mesh{std::move(positions), std::move(faces)},
        .Attrs{.Normals = std::move(normals)},
        // Four adjacency indices per edge preserve outward-facing silhouette normals.
        // clang-format off
        .AdjacencyIndices{
            0, 1, 2, 5,
            0, 2, 3, 5,
            0, 3, 4, 5,
            0, 4, 1, 5,
            2, 0, 1, 4,
            3, 0, 2, 1,
            4, 0, 3, 2,
            1, 0, 4, 3,
            4, 5, 1, 2,
            1, 5, 2, 3,
            2, 5, 3, 4,
            3, 5, 4, 1,
        },
        // clang-format on
    };
}

BoneSphereData BoneSphereDisc(float radius, uint32_t segments) {
    std::vector<vec3> positions;
    positions.reserve(segments + 1);
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = 2.f * std::numbers::pi_v<float> * float(i) / float(segments);
        positions.emplace_back(radius * std::cos(angle), radius * std::sin(angle), 0);
    }
    positions.emplace_back(0, 0, 0);

    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(segments);
    std::vector<uint32_t> outline;
    outline.reserve(segments * 2);
    for (uint32_t i = 0; i < segments; ++i) {
        faces.push_back({i, (i + 1) % segments, segments});
        outline.push_back(i);
        outline.push_back((i + 1) % segments);
    }

    return {
        .Mesh{std::move(positions), std::move(faces)},
        .OutlineIndices = std::move(outline),
    };
}
} // namespace primitive
