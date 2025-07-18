#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "ModelGizmo.h"

#include "numeric/mat3.h"
#include "numeric/ray.h"
#include "numeric/vec4.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include <format>
#include <optional>
#include <span>
#include <vector>

namespace {
using namespace ModelGizmo;
using enum Op;

constexpr auto OpVal = [](auto op) { return static_cast<std::underlying_type_t<Op>>(op); };
constexpr Op operator&(Op a, Op b) { return Op(OpVal(a) & OpVal(b)); }
constexpr Op operator|(Op a, Op b) { return Op(OpVal(a) | OpVal(b)); }
constexpr Op operator<<(Op op, uint32_t shift) { return Op(OpVal(op) << shift); }
constexpr bool HasAnyOp(Op a, Op b) { return (a & b) != Op::NoOp; }

namespace state {
struct Context {
    mat4 MVP;
    mat4 MVPLocal; // Full MVP model, whereas MVP might only be translation in case of World space edition
    ImRect ScreenRect{{0, 0}, {0, 0}};

    // World-space distance that projects to Style.SizeNDC, computed as:
    // Style.SizeNDC / (NDC-length of a world-space unit vector along the camera’s right direction, projected at the model’s origin)
    float WorldToSizeNDC;

    float RotationAngle; // Relative to the start rotation
    vec3 PosStart;
    vec3 Scale, ScaleStart;
    vec2 MousePosStart;
    ray MouseRayStart;
    vec4 InteractionPlane;

    Op Op{NoOp};
    bool Using{false};
};

// `Scale` members are relative to the gizmo size. To convert to screen-relative, multiply them by SizeNDC.
struct Style {
    // Size of the gizmo in NDC coordinates, relative to the screen size.
    // todo Actually, it's currently the size of an axis handle, but I want to change it to be the size of the whole gizmo,
    // and make all other scales <= 1.
    float SizeNDC{0.15};
    float AxisLengthVisibilityMinNDC{0.03}; // Minimum length of the axis line to be visible

    // `AxisHandle`s are the lines for translate/scale
    float TranslationArrowScale{0.18}, TranslationArrowRadScale{TranslationArrowScale * 0.3f};
    float AxisHandleScale{1.f - TranslationArrowScale}; // Tip is exacly at the gizmo size
    float AxisHandleLineWidth{2}; // in pixels
    // Radius and length of the arrow at the end of translation axes
    float TranslationArrowUniversalPosScale{1 + TranslationArrowScale}; // Translation arrows in Universal mode are the only thing "outside" the gizmo.
    float CircleRadScale{0.03}; // Radius of circle at the end of scale lines and the center of the translate/scale gizmo
    float UniversalScaleCircleRad{20}; // Pixels (todo: scale)
    float UniversalScaleCircleWidth{3}; // Pixels
    float RotationScreenCircleScale{1}; // Screen rotation circle is larger than other rotation circles, exactly the size of the gizmo.
    float RotationAxesCircleScale{AxisHandleScale}; // Rotation axes circles are smaller than the screen circle, equal to the the translation arrow base.
    float RotationLineWidth{2}; // Base thickness of lines for rotation gizmo, in pixels
    float PlaneCenterAxisScale{0.4}, PlaneSizeAxisScale{0.1};
};

struct Color {
    ImU32 Selection{IM_COL32(255, 128, 16, 138)};
    ImU32 TranslationLine{IM_COL32(170, 170, 170, 170)};
    ImU32 ScaleLine{IM_COL32(64, 64, 64, 255)};
    ImU32 RotationBorderActive{IM_COL32(255, 128, 16, 255)};
    ImU32 RotationFillActive{IM_COL32(255, 128, 16, 128)};
    ImU32 Text{IM_COL32(255, 255, 255, 255)}, TextShadow{IM_COL32(0, 0, 0, 255)};

    ImU32 Directions[3]{IM_COL32(255, 54, 83, 255), IM_COL32(138, 219, 0, 255), IM_COL32(44, 143, 255, 255)};
    ImU32 Planes[3]{IM_COL32(255, 54, 83, 100), IM_COL32(138, 219, 0, 100), IM_COL32(44, 143, 255, 100)};
};
} // namespace state

state::Context g;
constexpr state::Style Style;
constexpr state::Color Color;
} // namespace

namespace ModelGizmo {
bool IsActive() { return g.Using; }
Op CurrentOp() { return g.Op; }

std::string_view ToString(Op op) {
    if (op == NoOp) return "";
    if (op == (Translate | AxisX)) return "TranslateX";
    if (op == (Translate | AxisY)) return "TranslateY";
    if (op == (Translate | AxisZ)) return "TranslateZ";
    if (op == TranslateScreen) return "TranslateScreen";
    if (op == TranslateYZ) return "TranslateYZ";
    if (op == TranslateZX) return "TranslateZX";
    if (op == TranslateXY) return "TranslateXY";
    if (op == (Rotate | AxisX)) return "RotateX";
    if (op == (Rotate | AxisY)) return "RotateY";
    if (op == (Rotate | AxisZ)) return "RotateZ";
    if (op == RotateScreen) return "RotateScreen";
    if (op == (Scale | AxisX)) return "ScaleX";
    if (op == (Scale | AxisY)) return "ScaleY";
    if (op == (Scale | AxisZ)) return "ScaleZ";
    if (op == ScaleXYZ) return "ScaleXYZ";
    return "";
}
} // namespace ModelGizmo

namespace {
constexpr vec4 Right(const mat4 &m) { return {m[0]}; }
constexpr vec4 Dir(const mat4 &m) { return {m[2]}; }
constexpr vec3 Pos(const mat4 &m) { return {m[3]}; } // Assume affine matrix, with w = 1

// Assumes no scaling or shearing, only rotation and translation
constexpr mat4 InverseRigid(const mat4 &m) {
    const auto r = glm::transpose(mat3{m});
    return {{r[0], 0}, {r[1], 0}, {r[2], 0}, {-r * vec3{m[3]}, 1}};
}

constexpr Op AxisOp(uint32_t axis_i) { return AxisX << axis_i; }
constexpr uint32_t AxisIndex(Op op, Op type) {
    const auto axis_only = Op(uint32_t(op) - uint32_t(type));
    if (axis_only == AxisX) return 0;
    if (axis_only == AxisY) return 1;
    if (axis_only == AxisZ) return 2;
    assert(false);
    return -1;
}

constexpr vec3 Axes[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
constexpr Op TranslatePlanes[]{TranslateYZ, TranslateZX, TranslateXY}; // In axis order

constexpr std::optional<uint32_t> TranslatePlaneIndex(Op op) {
    if (op == TranslateYZ) return 0;
    if (op == TranslateZX) return 1;
    if (op == TranslateXY) return 2;
    return {};
}
constexpr std::pair<vec3, vec3> DirPlaneXY(uint32_t axis_i) { return {Axes[(axis_i + 1) % 3], Axes[(axis_i + 2) % 3]}; }

// Assumes p_normal is normalized
constexpr vec4 BuildPlane(vec3 p, const vec4 &p_normal) { return {vec3{p_normal}, glm::dot(p_normal, vec4{p, 1})}; }

constexpr float IntersectPlane(const ray &r, vec4 plane) {
    const float num = glm::dot(vec3{plane}, r.o) - plane.w;
    const float den = glm::dot(vec3{plane}, r.d);
    return fabsf(den) < FLT_EPSILON ? -1 : -num / den; // if normal is orthogonal to vector, can't intersect
}

// Camera _right_ vector is used to calculate WorldToSizeNDC,
// so screen _width_ is used to convert back to pixels.
constexpr float ScaleToPixels(float scale = 1.f) { return (g.ScreenRect.Max.x - g.ScreenRect.Min.x) * Style.SizeNDC * scale; }

// Homogeneous clip space to NDC
constexpr vec2 ToNDC(vec4 v) { return {fabsf(v.w) > FLT_EPSILON ? v / v.w : v}; }
constexpr float Length2(vec2 v) { return v.x * v.x + v.y * v.y; }

constexpr bool IsAxisVisible(uint32_t axis_i) {
    const auto dir = Axes[axis_i] * g.WorldToSizeNDC;
    const auto length2_ndc = Length2(ToNDC(g.MVP * vec4{dir, 1}) - ToNDC(g.MVP * vec4{0, 0, 0, 1}));
    return length2_ndc >= Style.AxisLengthVisibilityMinNDC * Style.AxisLengthVisibilityMinNDC;
}
constexpr bool IsPlaneVisible(vec3 dir_x, vec3 dir_y) {
    static constexpr auto ToScreenNDC = [](vec3 v) { return ToNDC(g.MVP * vec4{v, 1}); };
    static constexpr float ParallelogramAreaLimit{0.0025};
    const auto o = ToScreenNDC(vec3{0});
    const auto pa = ToScreenNDC(dir_x * g.WorldToSizeNDC) - o;
    const auto pb = ToScreenNDC(dir_y * g.WorldToSizeNDC) - o;
    return fabsf(pa.x * pb.y - pa.y * pb.x) > ParallelogramAreaLimit; // abs cross product
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

namespace Format {
static constexpr char AxisLabels[]{"XYZ"};
constexpr std::string Axis(uint32_t i, float v) { return i >= 0 && i < 3 ? std::format("{}: {:.3f}", AxisLabels[i], v) : ""; }
constexpr std::string Axis(uint32_t i, vec3 v) { return Axis(i, v[i]); }
constexpr std::string Translation(Op op, vec3 v) {
    if (op == TranslateScreen) return std::format("{} {} {}", Axis(0, v[0]), Axis(1, v[1]), Axis(2, v[2]));
    if (op == TranslateYZ) return std::format("{} {}", Axis(1, v[1]), Axis(2, v[2]));
    if (op == TranslateZX) return std::format("{} {}", Axis(2, v[2]), Axis(0, v[0]));
    if (op == TranslateXY) return std::format("{} {}", Axis(0, v[0]), Axis(1, v[1]));
    return Axis(AxisIndex(op, Op::Translate), v);
}
constexpr std::string Scale(Op op, vec3 v) { return op == ScaleXYZ ? std::format("XYZ: {:.3f}", v.x) : Axis(AxisIndex(op, Op::Scale), v); }
constexpr std::string Rotation(Op op, float rad) {
    if (!HasAnyOp(op, Rotate)) return "";

    const auto deg_rad = std::format("{:.3f} deg {:.3f} rad", rad * 180 / M_PI, rad);
    if (op == RotateScreen) return std::format("Screen: {}", deg_rad);

    const auto axis_i = AxisIndex(op, Rotate);
    return axis_i >= 0 && axis_i < 3 ? std::format("{}: {}", AxisLabels[axis_i], deg_rad) : "";
}
} // namespace Format

struct Model {
    Model(const mat4 &m, Mode mode)
        : RT{glm::normalize(m[0]), glm::normalize(m[1]), glm::normalize(m[2]), m[3]},
          M{mode == Local ? RT : glm::translate(mat4{1}, Pos(RT))} {}
    const mat4 RT; // Model matrix rotation + translation
    const mat4 M; // Gizmo model matrix (Local or World space).
    const mat4 Inv{InverseRigid(M)}; // Inverse of Gizmo model matrix
};

mat4 Transform(const mat4 &m, const Model &model, Mode mode, Op type, vec2 mouse_pos, const ray &mouse_ray, std::optional<vec3> snap) {
    const auto p = Pos(model.M);
    if (HasAnyOp(type, Translate)) {
        auto delta = mouse_ray(fabsf(IntersectPlane(mouse_ray, g.InteractionPlane))) -
            (g.MouseRayStart(IntersectPlane(g.MouseRayStart, g.InteractionPlane)) - g.PosStart) - p;
        // Single axis constraint
        if (type == (Translate | AxisX) || type == (Translate | AxisY) || type == (Translate | AxisZ)) {
            const auto axis_i = AxisIndex(type, Translate);
            delta = model.M[axis_i] * glm::dot(model.M[axis_i], vec4{delta, 0});
        }
        if (snap) {
            const vec4 d{p + delta - g.PosStart, 0};
            const vec3 delta_cumulative = mode == Local || type == TranslateScreen ? m * vec4{Snap(model.Inv * d, *snap), 0} : Snap(d, *snap);
            delta = g.PosStart + delta_cumulative - p;
        }
        return glm::translate(mat4{1}, delta) * m;
    }
    if (HasAnyOp(type, Scale)) {
        if (type == ScaleXYZ) {
            g.Scale = vec3{1 + (mouse_pos.x - g.MousePosStart.x) * 0.01f};
        } else { // Single axis constraint
            const auto axis_i = AxisIndex(type, Scale);
            const vec3 axis_value{model.RT[axis_i]};
            const auto relative_origin = g.MouseRayStart(IntersectPlane(g.MouseRayStart, g.InteractionPlane)) - g.PosStart;
            const auto p_ortho = Pos(model.RT);
            const auto base = relative_origin / g.WorldToSizeNDC - p_ortho;
            const auto delta = axis_value * glm::dot(axis_value, mouse_ray(IntersectPlane(mouse_ray, g.InteractionPlane)) - relative_origin - p_ortho);
            g.Scale[axis_i] = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
        }

        if (snap) g.Scale = Snap(g.Scale, *snap);
        for (uint32_t i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        return model.RT * glm::scale(mat4{1}, g.Scale * g.ScaleStart);
    }

    // Rotation: Compute angle on plane relative to the rotation origin
    const auto rotation_origin = glm::normalize(g.MouseRayStart(IntersectPlane(g.MouseRayStart, g.InteractionPlane)) - g.PosStart);
    const auto perp = glm::cross(rotation_origin, vec3{g.InteractionPlane});
    const auto pos_local = glm::normalize(mouse_ray(IntersectPlane(mouse_ray, g.InteractionPlane)) - p);
    float rotation_angle = acosf(glm::clamp(glm::dot(pos_local, rotation_origin), -1.f, 1.f)) * -glm::sign(glm::dot(pos_local, perp));
    if (snap) rotation_angle = Snap(rotation_angle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{model.Inv} * g.InteractionPlane); // Assumes affine model
    const mat4 rot_delta{glm::rotate(mat4{1}, rotation_angle - g.RotationAngle, rot_axis_local)};
    g.RotationAngle = rotation_angle;

    if (mode == Local) {
        const vec3 model_scale{glm::length(m[0]), glm::length(m[1]), glm::length(m[2])};
        return model.RT * rot_delta * glm::scale(mat4{1}, model_scale);
    }

    // Apply rotation, preserving translation
    auto res = rot_delta * mat4{mat3{m}};
    res[3] = vec4{Pos(m), 1};
    return res;
}

constexpr ImVec2 WorldToScreen(vec3 world, const mat4 &view_proj) {
    const vec2 uv = ToNDC(view_proj * vec4{world, 1}) * 0.5f + 0.5f; // [0,1]
    // Flip y (ImGui’s origin is top-left), and scale to gizmo rect.
    return g.ScreenRect.Min + ImVec2{uv.x, 1.f - uv.y} * g.ScreenRect.GetSize();
}

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 s1, ImVec2 s2) {
    const auto vec = std::bit_cast<vec2>(s2 - s1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, std::bit_cast<vec2>(p - s1));
    if (t <= 0) return s1;
    if (t * t > Length2(vec)) return s2;
    return s1 + std::bit_cast<ImVec2>(v) * t;
}

Op FindHoveredOp(const Model &model, Op op, ImVec2 mouse_pos, const ray &mouse_ray, const mat4 &view, const mat4 &view_proj) {
    const auto center = WorldToScreen(vec3{0}, g.MVP);
    // Op selection check order is important because of universal mode.
    // Universal = Scale | Translate | Rotate
    const auto is_translate = HasAnyOp(op, Translate), is_scale = HasAnyOp(op, Scale), is_rotate = HasAnyOp(op, Rotate);
    if (is_translate || is_scale) {
        const float dist_sq = ImLengthSqr(mouse_pos - center);
        const auto circle_rad_pixels = ScaleToPixels(Style.CircleRadScale);
        if (dist_sq <= circle_rad_pixels * circle_rad_pixels) return is_translate || op == Universal ? TranslateScreen : ScaleXYZ;

        if (op == Universal) {
            static constexpr float SelectDist = 8;
            const auto inner_rad = Style.UniversalScaleCircleRad - SelectDist / 2;
            const auto outer_rad = Style.UniversalScaleCircleRad + SelectDist / 2;
            if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return ScaleXYZ;
        }

        const auto half_arrow_px = ScaleToPixels(Style.TranslationArrowScale) * 0.5f;
        const auto screen_pos = g.ScreenRect.Min, mouse_pos_rel{mouse_pos - screen_pos};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto dir = model.M * vec4{Axes[i], 0};
            const auto p = Pos(model.M);
            const auto start = WorldToScreen(vec4{p, 1} + dir * g.WorldToSizeNDC * Style.AxisHandleScale * 0.1f, view_proj) - screen_pos;
            const auto end = WorldToScreen(vec4{p, 1} + dir * g.WorldToSizeNDC * (Style.AxisHandleScale + Style.TranslationArrowScale), view_proj) - screen_pos;
            if (is_scale && ImLengthSqr(end - mouse_pos_rel) <= half_arrow_px * half_arrow_px) return Scale | AxisOp(i);
            if (ImLengthSqr(PointOnSegment(mouse_pos_rel, start, end) - mouse_pos_rel) < half_arrow_px * half_arrow_px) {
                return (is_translate && op != Universal ? Translate : Scale) | AxisOp(i);
            }
            if (!is_translate) continue;

            if (op == Universal) {
                const auto arrow_center_scale = Style.TranslationArrowUniversalPosScale + Style.TranslationArrowScale * 0.5f;
                const auto translate_pos = WorldToScreen(vec4{p, 1} + dir * g.WorldToSizeNDC * arrow_center_scale, view_proj) - screen_pos;
                if (ImLengthSqr(translate_pos - mouse_pos_rel) < half_arrow_px * half_arrow_px) return Translate | AxisOp(i);
            }

            const auto [dir_x, dir_y] = DirPlaneXY(i);
            if (!IsPlaneVisible(dir_x, dir_y)) continue;

            const auto p_world = Pos(model.M);
            const auto pos_plane = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(p_world, dir)));
            const auto plane_x_world = vec3{model.M * vec4{dir_x, 0}};
            const auto plane_y_world = vec3{model.M * vec4{dir_y, 0}};
            const auto delta_world = (pos_plane - p_world) / g.WorldToSizeNDC;
            const float dx = glm::dot(delta_world, plane_x_world);
            const float dy = glm::dot(delta_world, plane_y_world);
            const float PlaneQuadUVMin = Style.PlaneCenterAxisScale - Style.PlaneSizeAxisScale * 0.5f;
            const float PlaneQuadUVMax = Style.PlaneCenterAxisScale + Style.PlaneSizeAxisScale * 0.5f;
            if (dx >= PlaneQuadUVMin && dx <= PlaneQuadUVMax && dy >= PlaneQuadUVMin && dy <= PlaneQuadUVMax) return TranslatePlanes[i];
        }
    }
    if (is_rotate) {
        static constexpr float SelectDist = 8;
        const auto dist_sq = ImLengthSqr(mouse_pos - center);
        const auto rotation_radius = 0.5f * ScaleToPixels(Style.RotationScreenCircleScale);
        const auto inner_rad = rotation_radius - SelectDist / 2, outer_rad = rotation_radius + SelectDist / 2;
        if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return RotateScreen;

        const auto p = Pos(model.M);
        const auto mv_pos = view * vec4{p, 1};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto intersect_pos_world = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(p, model.M[i])));
            const auto intersect_pos = view * vec4{vec3{intersect_pos_world}, 1};
            if (fabsf(mv_pos.z) - fabsf(intersect_pos.z) < -FLT_EPSILON) continue;

            const auto circle_pos_world = model.Inv * vec4{glm::normalize(intersect_pos_world - p), 0};
            const auto circle_pos = WorldToScreen(circle_pos_world * Style.RotationAxesCircleScale * g.WorldToSizeNDC, g.MVP);
            if (ImLengthSqr(circle_pos - mouse_pos) < SelectDist * SelectDist) return Rotate | AxisOp(i);
        }
    }
    return NoOp;
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

void Render(const Model &model, Op op, Op type, const mat4 &view_proj, vec3 cam_origin) {
    const auto CircleRadPixels = ScaleToPixels(Style.CircleRadScale);
    auto &dl = *ImGui::GetWindowDrawList();
    const auto center = WorldToScreen(vec3{0}, g.MVP);
    const auto center_ws = Pos(model.M);
    if (HasAnyOp(op, Translate)) {
        const float arrow_len_ws = Style.TranslationArrowScale * g.WorldToSizeNDC;
        const float arrow_rad_ws = Style.TranslationArrowRadScale * g.WorldToSizeNDC;
        const auto cam_dir_ws = glm::normalize(cam_origin - center_ws);
        for (uint32_t i = 0; i < 3; ++i) {
            const bool using_type = type == (Translate | AxisOp(i));
            if ((!g.Using || using_type) && IsAxisVisible(i)) {
                const auto base = WorldToScreen(Axes[i] * g.WorldToSizeNDC * Style.AxisHandleScale * 0.1f, g.MVP);
                const auto end = WorldToScreen(Axes[i] * g.WorldToSizeNDC * Style.AxisHandleScale, g.MVP);
                const auto color = using_type ? Color.Selection : Color.Directions[i];
                // Extend line a bit into the middle of the arrow, to avoid gaps between the the axis line and arrow base.
                if (op != Universal) dl.AddLine(base, end, color, Style.AxisHandleLineWidth + arrow_len_ws * 0.5f);
                // Draw translation arrows as cone silhouettes.

                // Billboard triangle facing the camera, with the middle of the triangle at the end of the axis line.
                const auto axis_dir_ws = vec3{model.RT[i]};
                const auto u_ws = glm::normalize(cam_dir_ws - glm::dot(cam_dir_ws, axis_dir_ws) * axis_dir_ws);
                const auto v_ws = glm::cross(axis_dir_ws, u_ws);

                const float scale = op == Universal ? Style.TranslationArrowUniversalPosScale : Style.AxisHandleScale;
                const auto base_ws = center_ws + axis_dir_ws * (g.WorldToSizeNDC * scale);
                const auto p_tip = WorldToScreen(base_ws + axis_dir_ws * arrow_len_ws, view_proj);
                const auto p_b1 = WorldToScreen(base_ws + v_ws * arrow_rad_ws, view_proj);
                const auto p_b2 = WorldToScreen(base_ws - v_ws * arrow_rad_ws, view_proj);
                dl.AddTriangleFilled(p_tip, p_b1, p_b2, color);

                // Ellipse at the base of the triangle.
                static constexpr uint32_t EllipsePointCount{16};
                static ImVec2 ellipse_pts[EllipsePointCount];

                const auto ellipse_base = (p_b1 + p_b2) * 0.5f;
                const auto p_u = WorldToScreen(base_ws + u_ws * arrow_rad_ws, view_proj);
                FastEllipse(std::span{ellipse_pts}, ellipse_base, p_u - ellipse_base, p_b1 - ellipse_base);
                dl.AddConvexPolyFilled(ellipse_pts, EllipsePointCount, color);
            }
            if (!g.Using || type == TranslatePlanes[i]) {
                const auto [dir_x, dir_y] = DirPlaneXY(i);
                if (!IsPlaneVisible(dir_x, dir_y)) continue;

                const auto screen_pos = [&](float sx, float sy) {
                    const auto uv = vec2{sx, sy} * Style.PlaneSizeAxisScale * 0.5f + Style.PlaneCenterAxisScale;
                    return WorldToScreen((dir_x * uv.x + dir_y * uv.y) * g.WorldToSizeNDC, g.MVP);
                };
                const auto p1{screen_pos(-1, -1)}, p2{screen_pos(-1, 1)}, p3{screen_pos(1, 1)}, p4{screen_pos(1, -1)};
                dl.AddQuad(p1, p2, p3, p4, Color.Directions[i], 1.f);
                dl.AddQuadFilled(p1, p2, p3, p4, (type == TranslatePlanes[i]) ? Color.Selection : Color.Planes[i]);
            }
        }
        if (!g.Using || type == TranslateScreen) {
            dl.AddCircleFilled(center, CircleRadPixels, type == TranslateScreen ? Color.Selection : IM_COL32_WHITE);
        }
        if (g.Using && HasAnyOp(type, Translate)) {
            const auto translation_line_color = Color.TranslationLine;
            const auto source_pos = WorldToScreen(g.PosStart, view_proj);
            dl.AddCircle(source_pos, 6.f, translation_line_color);
            dl.AddCircle(center, 6.f, translation_line_color);
            const auto dif = std::bit_cast<ImVec2>(glm::normalize(std::bit_cast<vec2>(center - source_pos)) * 5.f);
            dl.AddLine(source_pos + dif, center - dif, translation_line_color, 2.f);
            Label(Format::Translation(type, center_ws - g.PosStart), center);
        }
    }
    if (HasAnyOp(op, Scale)) {
        for (uint32_t i = 0; i < 3; ++i) {
            const bool using_type = type == (Scale | AxisOp(i));
            if ((!g.Using || using_type) && IsAxisVisible(i)) {
                const auto color = type == (Scale | AxisOp(i)) ? Color.Selection : Color.Directions[i];
                const auto base = WorldToScreen(Axes[i] * g.WorldToSizeNDC * Style.AxisHandleScale * .1f, g.MVP);
                const auto end = WorldToScreen(Axes[i] * g.WorldToSizeNDC * Style.AxisHandleScale, g.MVP);
                dl.AddCircleFilled(end, CircleRadPixels, color);
                dl.AddLine(base, end, color, Style.AxisHandleLineWidth);
            }
        }
        if (!g.Using || HasAnyOp(type, Scale)) {
            const auto color = g.Using || type == ScaleXYZ ? Color.Selection : IM_COL32_WHITE;
            if (op == Universal && (!g.Using || type == ScaleXYZ)) {
                dl.AddCircle(center, Style.UniversalScaleCircleRad, color, 0, Style.UniversalScaleCircleWidth);
            } else {
                dl.AddCircleFilled(center, CircleRadPixels, color, 0);
            }
            if (g.Using) Label(Format::Scale(type, g.Scale), center);
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr uint32_t HalfCircleSegmentCount{128}, FullCircleSegmentCount{HalfCircleSegmentCount * 2 + 1};
        static ImVec2 CirclePositions[FullCircleSegmentCount];
        if (g.Using && HasAnyOp(type, Rotate)) {
            {
                const auto u = glm::normalize(g.MouseRayStart(IntersectPlane(g.MouseRayStart, g.InteractionPlane)) - center_ws);
                const auto v = glm::cross(vec3{g.InteractionPlane}, u);
                const float r = g.WorldToSizeNDC * (type == RotateScreen ? Style.RotationScreenCircleScale : Style.RotationAxesCircleScale);
                const auto u_screen = WorldToScreen(center_ws + u * r, view_proj) - center;
                const auto v_screen = WorldToScreen(center_ws + v * r, view_proj) - center;
                FastEllipse(std::span{CirclePositions}, center, u_screen, v_screen, g.RotationAngle >= 0);
            }
            dl.AddPolyline(CirclePositions, FullCircleSegmentCount, Color.Selection, false, Style.RotationLineWidth);
            const uint32_t angle_i = float(FullCircleSegmentCount - 1) * fabsf(g.RotationAngle) / (2 * M_PI);
            CirclePositions[angle_i + 1] = center;
            dl.AddConvexPolyFilled(CirclePositions, angle_i + 2, Color.RotationFillActive);
            dl.AddLine(center, CirclePositions[0], Color.RotationBorderActive, Style.RotationLineWidth / 2);
            dl.AddLine(center, CirclePositions[angle_i], Color.RotationBorderActive, Style.RotationLineWidth);
            Label(Format::Rotation(type, g.RotationAngle), CirclePositions[1]);
        } else if (!g.Using) {
            // Half-circles facing the camera
            const float r = g.WorldToSizeNDC * Style.RotationAxesCircleScale;
            const vec3 cam_to_model = mat3{model.Inv} * glm::normalize(center_ws - cam_origin);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float angle_start = M_PI_2 + atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]);
                const vec4 axis_start{cosf(angle_start), sinf(angle_start), 0.f, 0.f};
                const vec4 axis_offset{-axis_start.y, axis_start.x, 0.f, 0.f};
                const vec3 u_local{axis_start[axis], axis_start[(axis + 1) % 3], axis_start[(axis + 2) % 3]};
                const vec3 v_local{axis_offset[axis], axis_offset[(axis + 1) % 3], axis_offset[(axis + 2) % 3]};
                const auto u_screen = WorldToScreen(u_local * r, g.MVP) - center;
                const auto v_screen = WorldToScreen(v_local * r, g.MVP) - center;
                FastEllipse(std::span{CirclePositions}.first(HalfCircleSegmentCount + 1), center, u_screen, v_screen, true, 0.5f);
                const auto color = (type == (Rotate | AxisOp(2 - axis))) ? Color.Selection : Color.Directions[2 - axis];
                dl.AddPolyline(CirclePositions, HalfCircleSegmentCount + 1, color, false, Style.RotationLineWidth);
            }
            // Screen rotation circle
            dl.AddCircle(
                center,
                0.5f * ScaleToPixels(Style.RotationScreenCircleScale),
                (type == RotateScreen) ? Color.Selection : IM_COL32_WHITE,
                FullCircleSegmentCount,
                Style.RotationLineWidth * 1.5f
            );
        }
    }
}
} // namespace

namespace ModelGizmo {
bool Draw(Mode mode, Op op, vec2 pos, vec2 size, vec2 mouse_pos, ray mouse_ray, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap) {
    g.ScreenRect = {std::bit_cast<ImVec2>(pos), std::bit_cast<ImVec2>(pos + size)};
    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (HasAnyOp(op, Scale)) mode = Local;

    const Model model{m, mode};
    const mat4 view_proj = proj * view;
    g.MVP = view_proj * model.M;
    g.MVPLocal = view_proj * model.RT;
    // Behind‐camera cull
    if (!g.Using && g.MVP[3].z < 0.001f) return false;

    const auto view_inv = InverseRigid(view);
    const ray camera_ray{Pos(view_inv), Dir(view_inv)};

    // Compute scale from camera right vector projected onto screen at model position.
    g.WorldToSizeNDC = Style.SizeNDC / glm::length(ToNDC(view_proj * vec4{Pos(model.RT) + vec3{Right(view_inv)}, 1}) - ToNDC(view_proj * vec4{Pos(model.RT), 1}));
    const bool commit = g.Using && !ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (commit) g.Using = false;

    if (g.Using) {
        m = Transform(m, model, mode, g.Op, mouse_pos, mouse_ray, snap);
    } else if (ImGui::IsWindowHovered()) {
        if (g.Op = FindHoveredOp(model, op, std::bit_cast<ImVec2>(mouse_pos), mouse_ray, view, view_proj);
            g.Op != NoOp && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g.Using = true;
            g.PosStart = Pos(model.RT);
            g.MousePosStart = mouse_pos;
            g.MouseRayStart = mouse_ray;
            g.Scale = {1, 1, 1};
            g.RotationAngle = 0;
            if (HasAnyOp(g.Op, Scale)) g.ScaleStart = {glm::length(m[0]), glm::length(m[1]), glm::length(m[2])};

            const auto GetPlaneNormal = [&](Op op) {
                if (g.Op == ScaleXYZ || g.Op == RotateScreen || g.Op == TranslateScreen) return -vec4{camera_ray.d, 0};
                if (HasAnyOp(g.Op, Scale)) return model.M[(AxisIndex(g.Op, Scale) + 1) % 3];
                if (HasAnyOp(g.Op, Rotate)) return mode == Local ? model.M[AxisIndex(g.Op, Rotate)] : vec4{Axes[AxisIndex(g.Op, Rotate)], 0};
                if (auto plane_index = TranslatePlaneIndex(op)) return model.M[*plane_index];

                const auto n = vec3{model.M[AxisIndex(op, Translate)]};
                const auto v = glm::normalize(g.PosStart - camera_ray.o);
                return vec4{v - n * glm::dot(n, v), 0};
            };
            g.InteractionPlane = BuildPlane(g.PosStart, GetPlaneNormal(g.Op));
        }
    }
    Render(model, op, g.Op, view_proj, camera_ray.o);
    return g.Using || commit;
}
} // namespace ModelGizmo
