#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "ModelGizmo.h"
#include "numeric/mat3.h"
#include "numeric/quat.h"
#include "numeric/vec4.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include "AxisColors.h" // Must be after imgui.h

#include <algorithm>
#include <format>
#include <optional>
#include <span>

namespace {
// Subset of (the externally visible) `Type` without `Universal`.
enum class TransformType : uint8_t {
    Translate,
    Rotate,
    Scale,
};

enum class InteractionOp : uint8_t {
    AxisX,
    AxisY,
    AxisZ,
    YZ,
    ZX,
    XY,
    Screen,
    Trackball, // Rotate only
};

// Hovered/active transform/op.
struct Interaction {
    TransformType Transform;
    InteractionOp Op;

    bool operator==(const Interaction &) const = default;
};

namespace state {
// Context captured when mouse is pressed on a hovered Interaction.
struct StartContext {
    mat4 M; // Model matrix
    vec2 MousePx;
    ray MouseRayWs;
    mat3 ViewInv;
    float WorldPerNdc; // World units per (signed) NDC at the gizmo origin (sampled along screen-x)
};

struct Context {
    mat4 MVP;

    ImRect ScreenRect{{0, 0}, {0, 0}};
    ImVec2 MousePx{0, 0};

    // World units per (signed) NDC unit at the gizmo origin (sampled along screen-x).
    // Use to convert a dimensionless NDC span to a world-space length (at the gizmo origin).
    float WorldPerNdc;

    std::optional<Interaction> Interaction; // If `Start` is present, active interaction. Otherwise, hovered interaction.
    std::optional<StartContext> Start; // Captured at mouse press on hovered Interaction.
    // Transform diffs from start
    vec3 Scale;
    float RotationAngle;
    vec2 RotationYawPitch;
};

struct Style {
    float SizeUv{0.08}; // Size of the gizmo as a ratio of screen width

    // `Size` vars are relative to gizmo size. To convert to screen-relative, multiply by SizeUv.

    // `AxisHandle`s are the lines for translate/scale
    float TranslationArrowSize{0.18}, TranslationArrowRadSize{0.3f * TranslationArrowSize};
    float TranslationArrowPosSizeUniversal{1 + TranslationArrowSize}; // Translation arrows in Universal mode are the only thing "outside" the gizmo
    float AxisHandleSize{1.f - TranslationArrowSize}; // Tip is exactly at the gizmo size
    float UniversalAxisHandleSize{AxisHandleSize - TranslationArrowSize}; // For scale handles in Universal mode
    float PlaneQuadSize{0.12}; // Translate/scale plane quads
    float CenterCircleRadSize{0.06}; // Radius of circle at the center of the translate/scale gizmo
    float CubeHalfExtentSize{0.75f * CenterCircleRadSize}; // Half extent of scale cube handles
    float InnerCircleRadSize{0.18}; // Radius of the inner selection circle at the center for translate/scale selection
    float OuterCircleRadSize{1.0}; // Outer circle is exactly the size of the gizmo
    float RotationCircleSize{AxisHandleSize}; // Rotation axes and trackball circles
    // Axes/planes fade from opaque to transparent between these ranges
    float AxisOpaqueRadSize{2.5f * InnerCircleRadSize}, AxisTransparentRadSize{InnerCircleRadSize};
    float PlaneOpaqueAngleRad{0.4}, PlaneTransparentAngleRad{0.2}; // facing camera -> opaque; more edge-on -> transparent

    float LineWidth{2}; // Used for axis handle/guide and inner/outer circle lines
    float RotationLineWidth{2.5}; // Thickness of rotation gizmo lines
};

struct Color {
    ImU32 TranslationLine{IM_COL32(170, 170, 170, 170)};
    ImU32 ScaleLine{IM_COL32(64, 64, 64, 255)};
    ImU32 StartGhost{IM_COL32(150, 150, 150, 160)};
    ImU32 RotationActiveFill{IM_COL32(255, 255, 255, 64)};
    ImU32 RotationTrackballHoverFill{IM_COL32(255, 255, 255, 15)};
    ImU32 Text{IM_COL32(255, 255, 255, 255)}, TextShadow{IM_COL32(0, 0, 0, 255)};
};
} // namespace state

state::Context g;
constexpr state::Style Style;
constexpr state::Color Color;
} // namespace

namespace ModelGizmo {
bool IsUsing() { return g.Start.has_value(); }

std::string_view ToString() {
    using enum InteractionOp;

    if (!g.Interaction) return "";

    using enum TransformType;

    const auto [type, axis] = *g.Interaction;
    if (type == Translate && axis == AxisX) return "TranslateX";
    if (type == Translate && axis == AxisY) return "TranslateY";
    if (type == Translate && axis == AxisZ) return "TranslateZ";
    if (type == Translate && axis == Screen) return "TranslateScreen";
    if (type == Translate && axis == YZ) return "TranslateYZ";
    if (type == Translate && axis == ZX) return "TranslateZX";
    if (type == Translate && axis == XY) return "TranslateXY";
    if (type == Rotate && axis == AxisX) return "RotateX";
    if (type == Rotate && axis == AxisY) return "RotateY";
    if (type == Rotate && axis == AxisZ) return "RotateZ";
    if (type == Rotate && axis == Screen) return "RotateScreen";
    if (type == Scale && axis == AxisX) return "ScaleX";
    if (type == Scale && axis == AxisY) return "ScaleY";
    if (type == Scale && axis == AxisZ) return "ScaleZ";
    if (type == Scale && axis == Screen) return "ScaleXYZ";
    return "";
}
} // namespace ModelGizmo

namespace {

using enum InteractionOp;

constexpr vec4 Right(const mat4 &m) { return {m[0]}; }
constexpr vec4 Dir(const mat4 &m) { return {m[2]}; }
constexpr vec3 Pos(const mat4 &m) { return {m[3]}; } // Assume affine matrix, with w = 1
constexpr vec3 GetScale(const mat4 &m) { return {glm::length(m[0]), glm::length(m[1]), glm::length(m[2])}; }
// Get rotation & translation from a model matrix.
mat4 GetRT(const mat4 &m) { return {glm::normalize(m[0]), glm::normalize(m[1]), glm::normalize(m[2]), m[3]}; }

constexpr float Length2(vec2 v) { return v.x * v.x + v.y * v.y; }

// Assumes no scaling or shearing, only rotation and translation
constexpr mat4 InverseRigid(const mat4 &m) {
    const auto r = glm::transpose(mat3{m});
    return {{r[0], 0}, {r[1], 0}, {r[2], 0}, {-r * vec3{m[3]}, 1}};
}

constexpr InteractionOp AxisOp(uint32_t axis_i) {
    if (axis_i == 0) return AxisX;
    if (axis_i == 1) return AxisY;
    if (axis_i == 2) return AxisZ;

    assert(false);
    return AxisX;
}

constexpr InteractionOp TranslatePlanes[]{InteractionOp::YZ, InteractionOp::ZX, InteractionOp::XY}; // In axis order

constexpr std::optional<uint32_t> TranslatePlaneIndex(InteractionOp plane) {
    if (plane == YZ) return 0;
    if (plane == ZX) return 1;
    if (plane == XY) return 2;
    return {};
}
constexpr uint32_t AxisIndex(InteractionOp op) {
    if (op == AxisX) return 0;
    if (op == AxisY) return 1;
    if (op == AxisZ) return 2;
    if (auto i = TranslatePlaneIndex(op)) return *i;

    assert(false);
    return -1;
}

constexpr std::optional<std::pair<InteractionOp, InteractionOp>> PlaneAxes(InteractionOp plane) {
    if (plane == YZ) return std::pair{AxisY, AxisZ};
    if (plane == ZX) return std::pair{AxisZ, AxisX};
    if (plane == XY) return std::pair{AxisX, AxisY};
    return {};
}

constexpr std::pair<uint32_t, uint32_t> PerpendicularAxes(uint32_t axis_i) { return {(axis_i + 1) % 3, (axis_i + 2) % 3}; }

constexpr vec4 BuildPlane(vec3 p, const vec4 &p_normal) { return {vec3{p_normal}, glm::dot(p_normal, vec4{p, 1})}; }

// Homogeneous clip space to signed NDC in [-1,+1]
constexpr vec3 CsToNdc(vec4 cs) { return {fabsf(cs.w) > FLT_EPSILON ? vec3{cs} / cs.w : vec3{cs}}; }
// NDC (signed) to UV [0,1] (top-left origin)
constexpr vec2 NdcToUv(vec3 ndc) { return {ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f}; }
// UV to pixels in window rect
constexpr ImVec2 UvToPx(vec2 uv) { return g.ScreenRect.Min + ImVec2{uv.x, uv.y} * g.ScreenRect.GetSize(); }
constexpr vec4 WsToCs(vec3 ws, const mat4 &vp) { return vp * vec4{ws, 1}; }
constexpr vec3 WsToNdc(vec3 ws, const mat4 &vp) { return CsToNdc(WsToCs(ws, vp)); }
constexpr vec2 WsToUv(vec3 ws, const mat4 &vp) { return NdcToUv(WsToNdc(ws, vp)); }
constexpr ImVec2 WsToPx(vec3 ws, const mat4 &vp) { return UvToPx(WsToUv(ws, vp)); }
// `size` is ratio of gizmo width
constexpr float SizeToPx(float size = 1.f) { return (g.ScreenRect.Max.x - g.ScreenRect.Min.x) * Style.SizeUv * size; }

constexpr float IntersectPlane(const ray &r, vec4 plane) {
    const float num = glm::dot(vec3{plane}, r.o) - plane.w;
    const float den = glm::dot(vec3{plane}, r.d);
    return fabsf(den) < FLT_EPSILON ? -1 : -num / den; // if normal is orthogonal to vector, can't intersect
}

constexpr float Snap(float v, float snap) {
    if (snap <= FLT_EPSILON) return v;

    static constexpr float SnapTension{0.5};
    const float modulo = fmodf(v, snap);
    const float modulo_ratio = fabsf(modulo) / snap;
    if (modulo_ratio < SnapTension) return v - modulo;
    if (modulo_ratio > 1 - SnapTension) return v - modulo + snap * (v < 0 ? -1 : 1);
    return v;
}
constexpr vec3 Snap(vec3 v, vec3 snap) { return {Snap(v[0], snap[0]), Snap(v[1], snap[1]), Snap(v[2], snap[2])}; }

constexpr char AxisLabels[]{"XYZ"};
constexpr std::string AxisLabel(uint32_t i, float v) { return i >= 0 && i < 3 ? std::format("{}: {:.3f}", AxisLabels[i], v) : ""; }
constexpr std::string AxisLabel(uint32_t i, vec3 v) { return AxisLabel(i, v[i]); }
constexpr std::string AxisLabel(InteractionOp a, vec3 v) { return AxisLabel(AxisIndex(a), v); }

// If Rotate, v[0] holds rotation angle (rad), or v[0]/v[1] yaw/pitch (rad) for Trackball.
constexpr std::string ValueLabel(Interaction i, vec3 v) {
    using enum TransformType;
    switch (i.Transform) {
        case Scale: // fallthrough
        case Translate: {
            switch (i.Op) {
                case AxisX:
                case AxisY:
                case AxisZ:
                    return AxisLabel(AxisIndex(i.Op), v);
                case YZ: return std::format("{} {}", AxisLabel(AxisY, v), AxisLabel(AxisZ, v));
                case ZX: return std::format("{} {}", AxisLabel(AxisZ, v), AxisLabel(AxisX, v));
                case XY: return std::format("{} {}", AxisLabel(AxisX, v), AxisLabel(AxisY, v));
                case Trackball: // passthrough (shouldn't happen)
                case Screen: return i.Transform == Scale ?
                    std::format("XYZ: {:.3f}", v.x) :
                    std::format("{} {} {}", AxisLabel(AxisX, v), AxisLabel(AxisY, v), AxisLabel(AxisZ, v));
            }
        }
        case Rotate: {
            if (i.Op == InteractionOp::Trackball) {
                const auto [yaw, pitch] = std::pair{v[0], v[1]};
                return std::format("Trackball: {:.2f}°, {:.2f}°", yaw * 180 / M_PI, pitch * 180 / M_PI);
            }
            const auto rad = v[0];
            if (i.Op == InteractionOp::Screen) return std::format("Screen: {:.2f}°", rad * 180 / M_PI);
            return AxisLabel(AxisIndex(i.Op), rad);
        }
    }
}

using ModelGizmo::Mode;

struct Model {
    Model(const mat4 &m, Mode mode)
        : RT{GetRT(m)}, M{mode == Mode::Local ? RT : glm::translate(I4, Pos(RT))}, Mode{mode} {}
    const mat4 RT; // Model matrix rotation + translation
    const mat4 M; // Gizmo model matrix
    const mat4 Inv{InverseRigid(M)}; // Inverse of Gizmo model matrix

    Mode Mode;
};

vec4 GetPlaneNormal(const Interaction &interaction, const mat4 &m, Mode mode, const ray &cam_ray) {
    using enum TransformType;
    if (interaction.Op == Screen || interaction.Op == Trackball) return -vec4{cam_ray.d, 0};
    if (auto plane_index = TranslatePlaneIndex(interaction.Op)) return m[*plane_index];

    const auto i = AxisIndex(interaction.Op);
    if (interaction.Transform == Scale) return m[(i + 1) % 3];
    if (interaction.Transform == Rotate) return mode == Mode::Local ? m[i] : vec4{I3[i], 0};

    const auto n = vec3{m[i]};
    const auto v = glm::normalize(Pos(m) - cam_ray.o);
    return vec4{v - n * glm::dot(n, v), 0};
};

mat4 Transform(const mat4 &m, const Model &model, Interaction interaction, const ray &mouse_ray, std::optional<vec3> snap) {
    using enum TransformType;

    assert(g.Start);

    const auto [transform, op] = interaction;
    const auto o_ws = Pos(model.M);
    const auto o_start_ws = Pos(g.Start->M);
    const auto plane_start = BuildPlane(o_start_ws, GetPlaneNormal(interaction, GetRT(g.Start->M), model.Mode, g.Start->MouseRayWs));
    const auto mouse_plane_intersect_ws = mouse_ray(IntersectPlane(mouse_ray, plane_start));
    const auto mouse_plane_intersect_start_ws = g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, plane_start));
    if (transform == Translate) {
        auto delta = (mouse_plane_intersect_ws - o_ws) - (mouse_plane_intersect_start_ws - o_start_ws);
        if (op == AxisX || op == AxisY || op == AxisZ) {
            const auto &plane = model.M[AxisIndex(op)];
            delta = plane * glm::dot(plane, vec4{delta, 0});
        }
        if (snap) {
            const vec4 d{o_ws + delta - o_start_ws, 0};
            const vec3 delta_cumulative = model.Mode == Mode::Local || op == Screen ? m * vec4{Snap(model.Inv * d, *snap), 0} : Snap(d, *snap);
            delta = o_start_ws + delta_cumulative - o_ws;
        }
        return glm::translate(I4, delta) * m;
    }
    if (transform == Scale) {
        // All scaling is based on mouse distance to origin
        const auto scale = glm::distance(mouse_plane_intersect_ws, o_start_ws) / glm::distance(mouse_plane_intersect_start_ws, o_start_ws);
        if (op == AxisX || op == AxisY || op == AxisZ) {
            g.Scale[AxisIndex(op)] = scale;
        } else if (auto plane_axes = PlaneAxes(op)) {
            g.Scale[AxisIndex(plane_axes->first)] = scale;
            g.Scale[AxisIndex(plane_axes->second)] = scale;
        } else {
            g.Scale = vec3{scale};
        }

        g.Scale = glm::max(snap ? Snap(g.Scale, *snap) : g.Scale, 0.001f);
        return model.RT * glm::scale(I4, g.Scale * GetScale(g.Start->M));
    }
    // Rotation
    if (op == Trackball) {
        const auto &m = g.Start->M;
        const auto delta_px = std::bit_cast<vec2>(g.MousePx) - g.Start->MousePx;
        const auto delta = delta_px / SizeToPx(Style.RotationCircleSize); // normalized drag in screen space
        if (Length2(delta) < 1e-12f) return m;

        const float angle = glm::length(delta);
        const auto axis_ws = glm::normalize(g.Start->ViewInv * vec3{delta.y, delta.x, 0}); // (dx→yaw, dy→pitch)
        const auto rot_delta = glm::mat4_cast(glm::angleAxis(angle, axis_ws));
        g.RotationYawPitch = delta;
        // Apply rotation, preserving translation
        auto res = rot_delta * mat4{mat3{m}};
        res[3] = vec4{Pos(m), 1};
        return res;
    }

    // Compute angle on plane relative to the rotation origin
    const auto rotation_origin = glm::normalize(mouse_plane_intersect_start_ws - o_start_ws);
    const auto perp = glm::cross(rotation_origin, vec3{plane_start});
    const auto pos_local = glm::normalize(mouse_plane_intersect_ws - o_ws);
    float rotation_angle = acosf(glm::clamp(glm::dot(pos_local, rotation_origin), -1.f, 1.f)) * -glm::sign(glm::dot(pos_local, perp));
    if (snap) rotation_angle = Snap(rotation_angle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{model.Inv} * vec3{plane_start}); // Assumes affine model
    const mat4 rot_delta{glm::rotate(I4, rotation_angle - g.RotationAngle, rot_axis_local)};
    g.RotationAngle = rotation_angle;
    if (model.Mode == Mode::Local) return model.RT * rot_delta * glm::scale(I4, GetScale(m));

    // Apply rotation, preserving translation
    auto res = rot_delta * mat4{mat3{m}};
    res[3] = vec4{Pos(m), 1};
    return res;
}

ImVec2 PointOnSegment(ImVec2 p, ImVec2 s1, ImVec2 s2) {
    const auto vec = std::bit_cast<vec2>(s2 - s1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, std::bit_cast<vec2>(p - s1));
    if (t <= 0) return s1;
    if (t * t > Length2(vec)) return s2;
    return s1 + std::bit_cast<ImVec2>(v) * t;
}

constexpr float AxisAlphaForDistSqPx(float dist_sq_px) {
    const float d_min = SizeToPx(Style.AxisTransparentRadSize);
    const float d_max = SizeToPx(Style.AxisOpaqueRadSize);
    return std::clamp((dist_sq_px - d_min * d_min) / (d_max * d_max - d_min * d_min), 0.f, 1.f);
}

float PlaneAlpha(uint32_t axis_i, const mat4 &m, const ray cam_ray) {
    const vec3 n_ws = glm::normalize(vec3{m[axis_i]});
    const vec3 v_ws = glm::normalize(Pos(m) - cam_ray.o);
    const float c = fabsf(glm::dot(n_ws, v_ws)); // [0=edge-on, 1=face-on]
    // Apply sin for comparison with c
    const float opaque = sinf(Style.PlaneOpaqueAngleRad);
    const float transparent = sinf(Style.PlaneTransparentAngleRad);
    return std::clamp((c - transparent) / (opaque - transparent), 0.f, 1.f);
}

std::optional<Interaction> FindHoveredInteraction(const Model &model, ModelGizmo::Type type, ImVec2 mouse_px, const ray &mouse_ray, const mat4 &view, const mat4 &vp, const ray &cam_ray) {
    using ModelGizmo::Type;

    static constexpr float SelectDist{8};

    const auto center = WsToPx(vec3{0}, g.MVP);
    const auto mouse_r_sq = ImLengthSqr(mouse_px - center);
    const auto inner_circle_rad_px = SizeToPx(Style.InnerCircleRadSize);
    if ((type == Type::Translate || type == Type::Universal) && mouse_r_sq <= inner_circle_rad_px * inner_circle_rad_px) {
        return Interaction{TransformType::Translate, Screen};
    }

    if (type != Type::Rotate) {
        const auto o_ws = Pos(model.M);
        const auto screen_min_px = g.ScreenRect.Min, mouse_rel_px{mouse_px - screen_min_px};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto dir = model.M[i];
            const auto dir_ndc = dir * g.WorldPerNdc;
            const auto start = WsToPx(vec4{o_ws, 1} + dir_ndc * Style.InnerCircleRadSize, vp) - screen_min_px;
            const auto end_size = type == Type::Universal ? Style.UniversalAxisHandleSize : Style.AxisHandleSize;
            const auto end = WsToPx(vec4{o_ws, 1} + dir_ndc * end_size, vp) - screen_min_px;
            if (ImLengthSqr(PointOnSegment(mouse_rel_px, start, end) - mouse_rel_px) < SelectDist * SelectDist) {
                return Interaction{type == Type::Translate ? TransformType::Translate : TransformType::Scale, AxisOp(i)};
            }

            if (type == Type::Universal) {
                const auto arrow_center_size = Style.TranslationArrowPosSizeUniversal + Style.TranslationArrowSize * 0.5f;
                const auto translate_pos = WsToPx(vec4{o_ws, 1} + dir_ndc * arrow_center_size, vp) - screen_min_px;
                const auto half_arrow_px = SizeToPx(Style.TranslationArrowSize) * 0.5f;
                if (ImLengthSqr(translate_pos - mouse_rel_px) < half_arrow_px * half_arrow_px) {
                    return Interaction{TransformType::Translate, AxisOp(i)};
                }
            }

            if (type == Type::Universal || PlaneAlpha(i, model.M, cam_ray) == 0) continue;

            const auto pos_plane = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(o_ws, dir)));
            const auto [ui, vi] = PerpendicularAxes(i);
            const auto plane_x_world = vec3{model.M[ui]};
            const auto plane_y_world = vec3{model.M[vi]};
            const auto delta_world = (pos_plane - o_ws) / g.WorldPerNdc;
            const float dx = glm::dot(delta_world, plane_x_world);
            const float dy = glm::dot(delta_world, plane_y_world);
            const float PlaneQuadUVMin = 0.5f - Style.PlaneQuadSize * 0.5f;
            const float PlaneQuadUVMax = 0.5f + Style.PlaneQuadSize * 0.5f;
            if (dx >= PlaneQuadUVMin && dx <= PlaneQuadUVMax && dy >= PlaneQuadUVMin && dy <= PlaneQuadUVMax) {
                return Interaction{type == Type::Scale ? TransformType::Scale : TransformType::Translate, TranslatePlanes[i]};
            }
        }
    }
    if (type == Type::Rotate || type == Type::Universal) {
        const auto rotation_radius = SizeToPx(Style.OuterCircleRadSize);
        const auto inner_rad = rotation_radius - SelectDist / 2, outer_rad = rotation_radius + SelectDist / 2;
        if (mouse_r_sq >= inner_rad * inner_rad && mouse_r_sq < outer_rad * outer_rad) {
            return Interaction{TransformType::Rotate, Screen};
        }

        const auto o_ws = Pos(model.M);
        const auto mv_pos = view * vec4{o_ws, 1};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto intersect_pos_world = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(o_ws, model.M[i])));
            const auto intersect_pos = view * vec4{vec3{intersect_pos_world}, 1};
            if (fabsf(mv_pos.z) - fabsf(intersect_pos.z) < -FLT_EPSILON) continue;

            const auto circle_pos_world = model.Inv * vec4{glm::normalize(intersect_pos_world - o_ws), 0};
            const auto circle_pos = WsToPx(circle_pos_world * Style.RotationCircleSize * g.WorldPerNdc, g.MVP);
            if (ImLengthSqr(circle_pos - mouse_px) < SelectDist * SelectDist) {
                return Interaction{TransformType::Rotate, AxisOp(i)};
            }
        }
        if (const auto circle_rad_px = SizeToPx(Style.RotationCircleSize);
            mouse_r_sq < circle_rad_px * circle_rad_px) {
            return Interaction{TransformType::Rotate, Trackball};
        }
    }
    if (type == Type::Scale) {
        if (const auto outer_circle_rad_px = SizeToPx(Style.OuterCircleRadSize);
            mouse_r_sq >= inner_circle_rad_px * inner_circle_rad_px &&
            mouse_r_sq < outer_circle_rad_px * outer_circle_rad_px) {
            return Interaction{TransformType::Scale, Screen};
        }
    }
    return std::nullopt;
}

void Label(std::string_view label, ImVec2 pos) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.AddText(pos + ImVec2{15, 15}, Color.TextShadow, label.data());
    dl.AddText(pos + ImVec2{14, 14}, Color.Text, label.data());
}

// Fast approximation of an ellipse by stepping a 2D rotation matrix instead of using sin/cos.
// For a half circle, pass `step_mult = 0.5`.
void FastEllipse(std::span<ImVec2> out, ImVec2 o, ImVec2 u, ImVec2 v, bool clockwise = true, float step_mult = 1.f) {
    const uint32_t count = out.size();
    const float d = (clockwise ? -2.f : 2.f) * step_mult * M_PI / float(count - 1);
    const float cos_d = cosf(d), sin_d = sinf(d);
    const glm::mat2 rot{cos_d, -sin_d, sin_d, cos_d};
    vec2 cs{1, 0}; // (cos0, sin0)
    for (uint32_t i = 0; i < count; ++i) {
        out[i] = o + u * cs.x + v * cs.y;
        cs = rot * cs;
    }
}

constexpr ImU32 SelectionColor(ImU32 color, bool selected) {
    return selected ? color : colors::MultAlpha(color, 0.8f);
}

// Clip ray `p + t*d` to rect `r` using Liang–Barsky algorithm.
// Returns line endpoints, or empty if no rect intersection.
std::optional<std::pair<ImVec2, ImVec2>> ClipRayToRect(const ImRect &r, ImVec2 p, ImVec2 d) {
    static constexpr float eps = 1e-6f;
    if (ImLengthSqr(d) <= eps) return {};

    // Check if parallel and outside (shouldn't happen in practice for axis guide lines)
    const auto pc = ImClamp(p, r.Min, r.Max);
    if ((fabsf(d.x) < eps && pc.x != p.x) || (fabsf(d.y) < eps && pc.y != p.y)) return {};

    const ImVec2 d_inv{1.f / d.x, 1.f / d.y};
    const auto t0 = (r.Min - p) * d_inv;
    const auto t1 = (r.Max - p) * d_inv;
    const auto tmin = ImMin(t0, t1), tmax = ImMax(t0, t1);
    float t_enter = std::max(tmin.x, tmin.y);
    float t_exit = std::min(tmax.x, tmax.y);
    if (t_enter > t_exit) return {}; // No intersection

    return {{p + d * t_enter, p + d * t_exit}};
}

void Render(const Model &model, ModelGizmo::Type type, const mat4 &vp, const ray &cam_ray) {
    using ModelGizmo::Type;
    using enum TransformType;

    const auto o_px = WsToPx(vec3{0}, g.MVP);

    if (g.Start && g.Interaction->Op == InteractionOp::Trackball) {
        Label(ValueLabel(*g.Interaction, vec3{g.RotationYawPitch, 0}), o_px);
        return;
    }

    auto &dl = *ImGui::GetWindowDrawList();

    // Full-screen axis guide lines during axis/plane interactions
    if (g.Start && g.Interaction->Op != InteractionOp::Screen) {
        const auto o_ws = Pos(g.Start->M);
        const auto DrawAxisGuideLine = [&](InteractionOp op) {
            const auto axis_i = AxisIndex(op);
            const auto axis_ws = vec3{g.Start->M[axis_i]};
            const auto p0 = WsToPx(o_ws, vp);
            const auto p1 = WsToPx(o_ws + axis_ws * g.WorldPerNdc, vp);
            if (const auto clipped = ClipRayToRect(g.ScreenRect, p0, p1 - p0)) {
                dl.AddLine(clipped->first, clipped->second, colors::Lighten(colors::Axes[axis_i], 0.25f), Style.LineWidth);
            }
        };

        if (const auto plane_axes = PlaneAxes(g.Interaction->Op)) {
            DrawAxisGuideLine(plane_axes->first);
            DrawAxisGuideLine(plane_axes->second);
        } else {
            DrawAxisGuideLine(g.Interaction->Op);
        }
    }

    // Center filled circle
    if (g.Start && g.Interaction->Transform != Rotate && g.Interaction->Op != Screen) {
        const auto axis_i = AxisIndex(g.Interaction->Op);
        const auto color = SelectionColor(colors::Axes[axis_i], true);
        dl.AddCircleFilled(o_px, SizeToPx(Style.CenterCircleRadSize), color);
        dl.AddCircleFilled(WsToPx(Pos(g.Start->M), vp), SizeToPx(Style.CenterCircleRadSize), Color.StartGhost);
    }
    // Ghost inner circle
    if (g.Start && g.Interaction->Op == Screen && g.Interaction->Transform != Rotate) {
        const auto center = g.Interaction->Transform == Translate ? WsToPx(Pos(g.Start->M), vp) : o_px;
        dl.AddCircle(center, SizeToPx(Style.InnerCircleRadSize), Color.StartGhost, 0, Style.LineWidth);
    }
    // Inner circle
    if ((!g.Start && type != Type::Rotate) ||
        (g.Start && g.Interaction->Transform != Rotate && g.Interaction->Op == Screen)) {
        const auto color = SelectionColor(IM_COL32_WHITE, g.Interaction && g.Interaction->Op == Screen);
        const auto scale = g.Start && g.Interaction->Transform == Scale ? g.Scale[0] : 1.f;
        dl.AddCircle(o_px, SizeToPx(scale * Style.InnerCircleRadSize), color, 0, Style.LineWidth);
    }
    // Outer circle
    if (type != Type::Translate && (!g.Start || g.Interaction == Interaction{Rotate, Screen})) {
        dl.AddCircle(
            o_px,
            SizeToPx(Style.OuterCircleRadSize),
            SelectionColor(IM_COL32_WHITE, g.Interaction && g.Interaction->Op == Screen),
            0,
            Style.LineWidth
        );
    }

    if (type != Type::Rotate) {
        enum class HandleType {
            Arrow, // Arrow cone silhouette (triangle + half-ellipse)
            Cube, // Cube silhouette
        };
        const auto DrawAxisHandle = [&](HandleType handle_type, bool is_active, bool ghost, uint32_t axis_i, float size, bool draw_line) {
            const auto &m = ghost ? g.Start->M : model.M;
            const auto o_ws = Pos(m);
            const auto axis_dir_ws = glm::normalize(vec3{m[axis_i]});

            const auto w2s = ghost ? g.Start->WorldPerNdc : g.WorldPerNdc;
            const auto end_ws = o_ws + w2s * axis_dir_ws * size;
            const auto end_px = WsToPx(end_ws, vp);
            const auto color = ghost ? Color.StartGhost :
                                       SelectionColor(colors::WithAlpha(colors::Axes[axis_i], AxisAlphaForDistSqPx(ImLengthSqr(end_px - o_px))), is_active);
            if (draw_line) {
                const float line_begin_size = g.Start && is_active ? Style.CenterCircleRadSize : Style.InnerCircleRadSize;
                dl.AddLine(WsToPx(o_ws + w2s * axis_dir_ws * line_begin_size, vp), end_px, color, Style.LineWidth);
            }

            if (handle_type == HandleType::Arrow) {
                // Build a single cone silhouette polygon: triangle + outer half of ellipse

                // Endpoints/basis
                const auto u_ws = glm::normalize((cam_ray.o - end_ws) - glm::dot(cam_ray.o - end_ws, axis_dir_ws) * axis_dir_ws);
                const auto v_ws = glm::cross(axis_dir_ws, u_ws);
                const auto p_tip = WsToPx(end_ws + w2s * axis_dir_ws * Style.TranslationArrowSize, vp);
                const auto p_b1 = WsToPx(end_ws + w2s * v_ws * Style.TranslationArrowRadSize, vp);
                const auto p_b2 = WsToPx(end_ws - w2s * v_ws * Style.TranslationArrowRadSize, vp);
                const auto p_u = WsToPx(end_ws + w2s * u_ws * Style.TranslationArrowRadSize, vp);

                // Ellipse frame
                const auto c = (p_b1 + p_b2) * 0.5f;
                const auto b = p_b2 - c; // base semi-axis vector (c -> p_b2)
                const float b_len2 = ImLengthSqr(b);
                if (b_len2 <= 1e-12f) return;

                // Unit outward normal to the base (perp(b)), flipped to point away from tip
                const float inv_b_len = 1.f / std::sqrt(b_len2);
                auto n_hat = ImVec2{-b.y, b.x} * inv_b_len;
                const auto tip_dir = p_tip - c;
                if (tip_dir.x * n_hat.x + tip_dir.y * n_hat.y > 0) n_hat = -n_hat;

                // Minor radius
                const auto u = p_u - c;
                const float r2 = ImLengthSqr(u);
                if (r2 <= 1e-12f) return;

                // Half-ellipse
                static constexpr uint32_t n{16 + 1}; // ellipse + tip
                ImVec2 poly[n];
                poly[0] = p_tip;
                FastEllipse(std::span<ImVec2>{poly}.subspan(1, n - 1), c, -b, n_hat * std::sqrt(r2), true, 0.5f);

                // Winding already CW
                dl.AddConvexPolyFilled(poly, n, color);
            } else if (handle_type == HandleType::Cube) {
                const auto [ui, vi] = PerpendicularAxes(axis_i);
                const auto u_ws = glm::normalize(vec3{m[ui]});
                const auto v_ws = glm::normalize(vec3{m[vi]});
                const float half_ws = w2s * Style.CubeHalfExtentSize;
                const vec3 A = axis_dir_ws * half_ws, U = u_ws * half_ws, V = v_ws * half_ws;
                const vec3 C = end_ws + A; // inner (−A) face touches the endpoint

                static constexpr uint8_t NumCorners{8};
                vec3 P[NumCorners]; // (bits: x=U, y=V, z=A)
                for (uint8_t i = 0; i < NumCorners; ++i) {
                    P[i] = C + ((i & 1) ? U : -U) + ((i & 2) ? V : -V) + ((i & 4) ? A : -A);
                }

                uint8_t deg[NumCorners]{0};
                uint8_t adj[NumCorners][3];
                const auto link = [&adj, &deg](uint8_t a, uint8_t b) {
                    adj[a][deg[a]++] = b;
                    adj[b][deg[b]++] = a;
                };

                const vec3 view_dir = glm::normalize(cam_ray.o - C);
                const bool sU = glm::dot(u_ws, view_dir) < 0;
                const bool sV = glm::dot(v_ws, view_dir) < 0;
                const bool sA = glm::dot(axis_dir_ws, view_dir) < 0;
                for (uint8_t i = 0; i < NumCorners; ++i) {
                    const bool bU = i & 1, bV = i & 2, bA = i & 4;
                    int j = i ^ 1;
                    if (i < j && ((bV ^ bA) ^ (sV ^ sA))) link(i, j); // along U
                    j = i ^ 2;
                    if (i < j && ((bU ^ bA) ^ (sU ^ sA))) link(i, j); // along V
                    j = i ^ 4;
                    if (i < j && ((bU ^ bV) ^ (sU ^ sV))) link(i, j); // along A
                }

                const uint8_t start = std::ranges::find_if_not(deg, [](auto d) { return d == 0; }) - std::begin(deg);
                if (start == NumCorners) return;

                // Walk the polygon loop
                uint8_t loop_idx[NumCorners], n = 0;
                uint8_t cur = start;
                std::optional<uint8_t> prev;
                do {
                    loop_idx[n++] = cur;
                    std::optional<uint8_t> next;
                    for (uint8_t k = 0; k < deg[cur]; ++k) {
                        if (auto nb = adj[cur][k]; !prev || nb != *prev) {
                            next = nb;
                            break;
                        }
                    }
                    if (!next) return; // shouldn't happen - safety
                    prev = cur;
                    cur = *next;
                } while (cur != start && n < NumCorners);
                if (n < 3) return;

                static ImVec2 hull[NumCorners];
                for (uint8_t i = 0; i < n; ++i) hull[i] = WsToPx(P[loop_idx[i]], vp);

                // CW winding for outward AA in ImGui
                float area2{0};
                for (uint8_t i = 0, j = n - 1; i < n; j = i++) area2 += hull[j].x * hull[i].y - hull[i].x * hull[j].y;
                if (area2 < 0) std::reverse(hull, hull + n);

                dl.AddConvexPolyFilled(hull, n, color);
            }
        };

        for (uint32_t i = 0; i < 3; ++i) {
            if (type != Type::Scale) { // Always draw translation handles in translate mode
                const bool is_active = g.Interaction == Interaction{Translate, AxisOp(i)};
                const float size = type == Type::Universal ? Style.TranslationArrowPosSizeUniversal : Style.AxisHandleSize;
                DrawAxisHandle(HandleType::Arrow, is_active, false, i, size, type != Type::Universal || g.Start);
                if (g.Start && is_active) DrawAxisHandle(HandleType::Arrow, is_active, true, i, size, true);
            }
            if (const bool scale_active = g.Interaction == Interaction{Scale, AxisOp(i)};
                type != Type::Translate && (!g.Start || scale_active)) {
                const float size = type == Type::Universal ? Style.UniversalAxisHandleSize : Style.AxisHandleSize;
                DrawAxisHandle(HandleType::Cube, scale_active, false, i, size * (g.Start ? g.Scale[i] : 1.0f), true);
                if (g.Start) DrawAxisHandle(HandleType::Cube, scale_active, true, i, size, true);
            }
            if (type != Type::Universal && (!g.Start || g.Interaction->Op == TranslatePlanes[i])) {
                const auto [ui, vi] = PerpendicularAxes(i);

                const auto screen_pos = [&](vec2 s, bool ghost) {
                    const auto &m = ghost ? GetRT(g.Start->M) : model.RT;
                    const auto w2s = ghost ? g.Start->WorldPerNdc : g.WorldPerNdc;
                    const auto mult = g.Start && !ghost && type == Type::Scale ? g.Scale[AxisIndex(PlaneAxes(g.Interaction->Op)->first)] : 1.f;
                    const auto uv = s * Style.PlaneQuadSize * 0.5f + 0.5f * mult;
                    return WsToPx(w2s * (I3[ui] * uv.x + I3[vi] * uv.y), vp * m);
                };
                const auto p1{screen_pos({-1, -1}, false)}, p2{screen_pos({-1, 1}, false)}, p3{screen_pos({1, 1}, false)}, p4{screen_pos({1, -1}, false)};
                const bool is_selected = g.Interaction && g.Interaction->Op == TranslatePlanes[i];
                const auto plane_alpha = PlaneAlpha(i, model.RT, cam_ray);
                dl.AddQuad(p1, p2, p3, p4, colors::MultAlpha(SelectionColor(colors::Axes[i], is_selected), plane_alpha), 1.f);
                dl.AddQuadFilled(p1, p2, p3, p4, colors::MultAlpha(SelectionColor(colors::WithAlpha(colors::Axes[i], 0.5f), is_selected), plane_alpha));
                if (g.Start) {
                    const auto p1{screen_pos({-1, -1}, true)}, p2{screen_pos({-1, 1}, true)}, p3{screen_pos({1, 1}, true)}, p4{screen_pos({1, -1}, true)};
                    dl.AddQuad(p1, p2, p3, p4, colors::Lighten(Color.StartGhost, 0.5f), 1.f);
                    dl.AddQuadFilled(p1, p2, p3, p4, Color.StartGhost);
                }
            }
        }

        if (g.Start) {
            Label(ValueLabel(*g.Interaction, g.Interaction->Transform == Translate ? Pos(model.M) - Pos(g.Start->M) : g.Scale), o_px);
        }
    }
    if (type == Type::Rotate || type == Type::Universal) {
        const auto o_ws = Pos(model.M);
        static constexpr uint32_t HalfCircleSegmentCount{128}, FullCircleSegmentCount{HalfCircleSegmentCount * 2 + 1};
        static ImVec2 CirclePositions[FullCircleSegmentCount];
        if (g.Start && g.Interaction->Transform == Rotate) {
            {
                const auto o_start_ws = Pos(g.Start->M);
                const auto plane_start = BuildPlane(o_start_ws, GetPlaneNormal(*g.Interaction, GetRT(g.Start->M), model.Mode, g.Start->MouseRayWs));
                const auto u = glm::normalize(g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, plane_start)) - o_ws);
                const auto v = glm::cross(vec3{plane_start}, u);
                const float r = g.WorldPerNdc * (g.Interaction->Op == Screen ? Style.OuterCircleRadSize : Style.RotationCircleSize);
                const auto u_px = WsToPx(o_ws + u * r, vp) - o_px;
                const auto v_px = WsToPx(o_ws + v * r, vp) - o_px;
                FastEllipse(CirclePositions, o_px, u_px, v_px, g.RotationAngle >= 0);
            }
            const uint32_t angle_i = float(FullCircleSegmentCount - 1) * fabsf(g.RotationAngle) / (2 * M_PI);
            const auto angle_circle_pos = CirclePositions[angle_i + 1]; // save
            CirclePositions[angle_i + 1] = o_px;
            dl.AddConvexPolyFilled(CirclePositions, angle_i + 2, Color.RotationActiveFill);

            CirclePositions[angle_i + 1] = angle_circle_pos; // restore
            const auto color = g.Interaction->Op == Screen ? IM_COL32_WHITE : colors::Axes[AxisIndex(g.Interaction->Op)];
            dl.AddPolyline(CirclePositions, FullCircleSegmentCount, color, false, Style.RotationLineWidth);
            dl.AddLine(o_px, CirclePositions[0], color, Style.RotationLineWidth / 2);
            dl.AddLine(o_px, CirclePositions[angle_i], color, Style.RotationLineWidth);
            Label(ValueLabel(*g.Interaction, vec3{g.RotationAngle}), CirclePositions[1]);
        } else if (!g.Start) {
            // Half-circles facing the camera
            const float r = g.WorldPerNdc * Style.RotationCircleSize;
            const vec3 cam_to_model = mat3{model.Inv} * glm::normalize(o_ws - cam_ray.o);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float angle_start = M_PI_2 + atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]);
                const vec4 axis_start{cosf(angle_start), sinf(angle_start), 0.f, 0.f};
                const vec4 axis_offset{-axis_start.y, axis_start.x, 0.f, 0.f};
                const auto [ui, vi] = PerpendicularAxes(axis);
                const vec3 u_local{axis_start[axis], axis_start[ui], axis_start[vi]};
                const vec3 v_local{axis_offset[axis], axis_offset[ui], axis_offset[vi]};
                const auto u_px = WsToPx(u_local * r, g.MVP) - o_px;
                const auto v_px = WsToPx(v_local * r, g.MVP) - o_px;
                FastEllipse(std::span{CirclePositions}.first(HalfCircleSegmentCount + 1), o_px, u_px, v_px, true, 0.5f);
                const auto color = SelectionColor(colors::Axes[2 - axis], g.Interaction == Interaction{Rotate, AxisOp(2 - axis)});
                dl.AddPolyline(CirclePositions, HalfCircleSegmentCount + 1, color, false, Style.RotationLineWidth);
            }
            if (g.Interaction && g.Interaction->Op == Trackball) {
                dl.AddCircleFilled(o_px, SizeToPx(Style.RotationCircleSize), Color.RotationTrackballHoverFill);
            }
        }
    }
    // Dashed center->mouse guide line + double arrow cursor
    if (g.Start && g.Interaction->Transform != Translate) {
        // Same as ImDrawList::AddLine but w/o half-px offset
        static const auto AddLine = [](ImDrawList &dl, ImVec2 p1, ImVec2 p2, ImU32 col, float thickness) {
            dl.PathLineTo(p1);
            dl.PathLineTo(p2);
            dl.PathStroke(col, 0, thickness);
        };

        static const auto DrawDashedLine = [](ImDrawList &dl, ImVec2 a, ImVec2 b, ImU32 color) {
            static constexpr float Thickness{1}, DashLen{4}, GapLen{3};

            const auto dir = b - a;
            const float len = sqrtf(ImLengthSqr(dir));
            if (len <= 1e-3f) return;

            const auto dir_unit = dir / len;
            float t{0};
            while (t < len) {
                AddLine(dl, a + dir_unit * t, a + dir_unit * (t + std::min(DashLen, len - t)), color, Thickness);
                t += DashLen + GapLen;
            }
        };

        static constexpr auto LineColor{IM_COL32(255, 255, 255, 255)}, ShadowLineColor{IM_COL32(90, 90, 90, 200)};
        static constexpr ImVec2 ShadowOffset{1.5, 1.5};
        DrawDashedLine(dl, o_px + ShadowOffset, g.MousePx + ShadowOffset, ShadowLineColor);
        DrawDashedLine(dl, o_px, g.MousePx, LineColor);

        static const auto DrawCursorArrow = [](ImDrawList &dl, ImVec2 base, ImVec2 dir, ImVec2 lat) {
            static constexpr float CursorThickness{2}, ShaftLength{22}, HeadLength{7}, HeadWidth{5};

            const auto head = base + dir * (ShaftLength * 0.5f - HeadLength);
            const ImVec2 points[]{
                // tip -> +head -> +base -> -base -> -head
                base + dir * ShaftLength * 0.5f,
                head + lat * HeadWidth,
                base + lat * CursorThickness,
                base - lat * CursorThickness,
                head - lat * HeadWidth,
            };

            static constexpr auto FillColor{IM_COL32(255, 255, 255, 255)}, OutlineColor{IM_COL32(0, 0, 0, 255)};
            dl.AddConvexPolyFilled(points, 5, FillColor);
            dl.AddPolyline(points, 5, OutlineColor, true, 1.f);
        };

        // Two arrows pointing in opposite directions, either along or perpendicular to `line_dir`
        static const auto DrawCursorDoubleArrow = [](ImDrawList &dl, ImVec2 center, ImVec2 line_dir, bool perpendicular = true) {
            static constexpr float CenterGap{10};

            const float len2 = ImLengthSqr(line_dir);
            if (len2 <= 1e-6f) return;

            const auto line_unit = line_dir / sqrtf(len2);
            const ImVec2 perp_unit{-line_unit.y, line_unit.x};
            const auto dir = perpendicular ? perp_unit : line_unit;
            const auto lat = perpendicular ? line_unit : perp_unit;
            const auto gap = dir * CenterGap * 0.5f;
            DrawCursorArrow(dl, center + gap, dir, lat);
            DrawCursorArrow(dl, center - gap, -dir, lat);
        };

        ImGui::SetMouseCursor(ImGuiMouseCursor_None); // Custom cursor
        DrawCursorDoubleArrow(dl, g.MousePx, g.MousePx - o_px, g.Interaction->Transform == Rotate);
    }
}
} // namespace

namespace ModelGizmo {
bool Draw(Mode mode, Type type, vec2 pos, vec2 size, vec2 mouse_px, ray mouse_ray_ws, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap) {
    g.ScreenRect = {std::bit_cast<ImVec2>(pos), std::bit_cast<ImVec2>(pos + size)};
    g.MousePx = std::bit_cast<ImVec2>(mouse_px);

    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (type == Type::Scale || type == Type::Universal) mode = Mode::Local;

    const Model model{m, mode};
    const mat4 vp = proj * view;
    g.MVP = vp * model.M;
    // Behind-camera cull
    if (!g.Start && g.MVP[3].z < 0.001f) return false;

    const auto view_inv = InverseRigid(view);
    const ray cam_ray{Pos(view_inv), Dir(view_inv)};
    // Compute world units per NDC at model position, sampling along camera-right projected to screen at the model origin.
    // 2xNDC spans screen width.
    g.WorldPerNdc = 2 * Style.SizeUv /
        glm::length(CsToNdc(vp * vec4{Pos(model.RT) + vec3{Right(view_inv)}, 1}) - CsToNdc(vp * vec4{Pos(model.RT), 1}));
    if (g.Start && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        g.Start = {};
        g.Interaction = {};
    }

    if (g.Start) {
        assert(g.Interaction);
        m = Transform(m, model, *g.Interaction, mouse_ray_ws, snap);
    } else if (ImGui::IsWindowHovered()) {
        if (g.Interaction = FindHoveredInteraction(model, type, std::bit_cast<ImVec2>(mouse_px), mouse_ray_ws, view, vp, cam_ray);
            g.Interaction && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g.Start = state::StartContext{
                .M = m,
                .MousePx = mouse_px,
                .MouseRayWs = mouse_ray_ws,
                .ViewInv = mat3{view_inv},
                .WorldPerNdc = g.WorldPerNdc,
            };
            g.Scale = {1, 1, 1};
            g.RotationAngle = 0;
            g.RotationYawPitch = {0, 0};
        }
    }

    Render(model, type, vp, cam_ray);
    return bool(g.Start);
}
} // namespace ModelGizmo
