#include "ExtrasMesh.h"

#include "gpu/PunctualLight.h"

#include <algorithm>
#include <numbers>

namespace {
constexpr float Pi{std::numbers::pi_v<float>};
constexpr uint32_t SpotConeSegments{32};
float AngleFromCos(float cos_theta) { return std::acos(std::clamp(cos_theta, -1.f, 1.f)); }
} // namespace

uint32_t ExtrasWireframe::AddVertex(vec3 pos, uint8_t vclass) {
    const uint32_t i = Data.Positions.size();
    Data.Positions.emplace_back(pos);
    VertexClasses.emplace_back(vclass);
    return i;
}

void ExtrasWireframe::AddEdge(uint32_t a, uint32_t b) { Data.Edges.push_back({a, b}); }

void ExtrasWireframe::AddCircle(float radius, uint32_t segments, float z, uint8_t vclass, uint32_t edge_stride) {
    const uint32_t base = Data.Positions.size();
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = float(i) * 2.f * Pi / float(segments);
        AddVertex({radius * std::cos(angle), radius * std::sin(angle), z}, vclass);
    }
    for (uint32_t i = 0; i < segments; i += edge_stride) AddEdge(base + i, base + (i + 1) % segments);
}

void ExtrasWireframe::AddDiamond(float radius, uint8_t vclass, vec3 axis1, vec3 axis2, vec3 center) {
    const uint32_t base = Data.Positions.size();
    for (auto a : {axis1, axis2, -axis1, -axis2}) AddVertex(center + radius * a, vclass);
    for (uint32_t i = 0; i < 4; ++i) AddEdge(base + i, base + (i + 1) % 4);
}

MeshData BuildCameraFrustumMesh(const Camera &camera) {
    float display_near{0.01f}, display_far{5.f};
    float near_half_w{0.f}, near_half_h{0.f}, far_half_w{0.f}, far_half_h{0.f};
    if (const auto *perspective = std::get_if<Perspective>(&camera)) {
        // Clamp far plane for display so wireframe doesn't extend to infinity.
        display_near = perspective->NearClip;
        display_far = std::min(perspective->FarClip.value_or(5.f), 5.f);

        const float aspect = AspectRatio(camera);
        near_half_h = display_near * std::tan(perspective->FieldOfViewRad * 0.5f);
        near_half_w = near_half_h * aspect;
        far_half_h = display_far * std::tan(perspective->FieldOfViewRad * 0.5f);
        far_half_w = far_half_h * aspect;
    } else if (const auto *orthographic = std::get_if<Orthographic>(&camera)) {
        display_near = orthographic->NearClip;
        display_far = std::min(orthographic->FarClip, 5.f);
        near_half_w = far_half_w = orthographic->Mag.x;
        near_half_h = far_half_h = orthographic->Mag.y;
    }

    std::vector<vec3> positions{
        {-near_half_w, -near_half_h, -display_near},
        {near_half_w, -near_half_h, -display_near},
        {near_half_w, near_half_h, -display_near},
        {-near_half_w, near_half_h, -display_near},
        {-far_half_w, -far_half_h, -display_far},
        {far_half_w, -far_half_h, -display_far},
        {far_half_w, far_half_h, -display_far},
        {-far_half_w, far_half_h, -display_far},
        {-far_half_w, far_half_h, -display_far},
        {far_half_w, far_half_h, -display_far},
        {0.f, far_half_h + far_half_h * 0.3f, -display_far},
    };

    // clang-format off
    std::vector<std::array<uint32_t, 2>> edges{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
        {8, 10}, {10, 9},
    };
    // clang-format on

    return {.Positions = std::move(positions), .Edges = std::move(edges)};
}

ExtrasWireframe BuildLightMesh(const PunctualLight &light) {
    ExtrasWireframe wf;

    const auto add_range_circle = [&](float range) {
        if (range > 0.f) wf.AddCircle(range, 32, 0.f, VClassBillboard);
    };

    if (light.Type == PunctualLightType::Point) {
        add_range_circle(light.Range);
    } else if (light.Type == PunctualLightType::Directional) {
        constexpr uint32_t ray_count = 8;
        constexpr float d0s = 14.f, d0e = 16.f, d1s = 18.f, d1e = 20.f;
        for (uint32_t i = 0; i < ray_count; ++i) {
            const float angle = float(i) * 2.f * Pi / float(ray_count);
            const float dx = std::cos(angle), dy = std::sin(angle);
            const auto a = wf.AddVertex({d0s * dx, d0s * dy, 0.f}, VClassScreenspace);
            const auto b = wf.AddVertex({d0e * dx, d0e * dy, 0.f}, VClassScreenspace);
            const auto c = wf.AddVertex({d1s * dx, d1s * dy, 0.f}, VClassScreenspace);
            const auto d = wf.AddVertex({d1e * dx, d1e * dy, 0.f}, VClassScreenspace);
            wf.AddEdge(a, b);
            wf.AddEdge(c, d);
        }
    } else if (light.Type == PunctualLightType::Spot) {
        constexpr float depth = 2.f;
        const float outer_angle = std::min(AngleFromCos(light.OuterConeCos), glm::radians(89.f));
        const float inner_angle = std::min(AngleFromCos(light.InnerConeCos), outer_angle);
        const float outer_radius = depth * std::tan(outer_angle);
        const float inner_radius = depth * std::tan(inner_angle);
        add_range_circle(light.Range);
        wf.AddCircle(outer_radius, SpotConeSegments, -depth, VClassNone);
        if (inner_radius > 0.f) wf.AddCircle(inner_radius, SpotConeSegments, -depth, VClassNone);

        for (uint32_t i = 0; i < SpotConeSegments; ++i) {
            const float angle = float(i) * 2.f * Pi / float(SpotConeSegments);
            const auto ai = wf.AddVertex({0.f, 0.f, 0.f}, VClassNone);
            const auto bi = wf.AddVertex({outer_radius * std::cos(angle), outer_radius * std::sin(angle), -depth}, VClassSpotCone);
            wf.AddEdge(ai, bi);
        }
    }

    wf.AddDiamond(2.7f, VClassScreenspace, {1, 0, 0}, {0, 1, 0});
    wf.AddCircle(9.f, 16, 0.f, VClassScreenspace, 2);
    wf.AddCircle(9.f * 1.33f, 20, 0.f, VClassScreenspace, 2);

    const auto top = wf.AddVertex({0.f, 1.f, 0.f}, VClassGroundPoint);
    const auto bot = wf.AddVertex({0.f, 0.f, 0.f}, VClassGroundPoint);
    wf.AddEdge(top, bot);
    wf.AddDiamond(3.f, VClassGroundPoint, {1, 0, 0}, {0, 0, 1});

    return wf;
}
