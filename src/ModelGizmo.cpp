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
    vec2 Pos{0, 0}, Size{0, 0};

    vec4 InteractionPlane;
    vec3 MatrixOrigin;

    float ScreenFactor;
    float RotationAngle; // Relative to the start rotation

    vec3 Scale, ScaleOrigin;
    vec2 MousePosOrigin;
    ray MouseRayOrigin;

    Op Op{NoOp};
    bool Using{false};
};

struct Style {
    float SizeClipSpace{0.1};
    float LineWidth{3}; // Thickness of lines for translate/scale gizmo
    float HatchedAxisLineWidth{6}; // Thickness of hatched axis lines
    float LineArrowSize{6}; // Size of arrow at the end of translation lines
    float CircleRad{6}; // Radius of circle at the end of scale lines and the center of the translate/scale gizmo
    float RotationDisplayScale{1.2}; // Scale a bit so translate axes don't touch when in universal.
    float RotationLineWidth{2}; // Base thickness of lines for rotation gizmo
};

struct Color {
    ImU32 Selection{IM_COL32(255, 128, 16, 138)};
    ImU32 TranslationLine{IM_COL32(170, 170, 170, 170)};
    ImU32 ScaleLine{IM_COL32(64, 64, 64, 255)};
    ImU32 RotationBorderActive{IM_COL32(255, 128, 16, 255)};
    ImU32 RotationFillActive{IM_COL32(255, 128, 16, 128)};
    ImU32 HatchedAxisLines{IM_COL32(0, 0, 0, 128)};
    ImU32 Text{IM_COL32(255, 255, 255, 255)}, TextShadow{IM_COL32(0, 0, 0, 255)};

    ImU32 Directions[3]{IM_COL32(255, 54, 83, 255), IM_COL32(138, 219, 0, 255), IM_COL32(44, 143, 255, 255)};
    ImU32 Planes[3]{IM_COL32(154, 57, 71, 255), IM_COL32(98, 138, 34, 255), IM_COL32(52, 100, 154, 255)};
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

constexpr Op TranslatePlanes[]{TranslateYZ, TranslateZX, TranslateXY}; // In axis order

constexpr std::optional<uint32_t> TranslatePlaneIndex(Op op) {
    if (op == TranslateYZ) return 0;
    if (op == TranslateZX) return 1;
    if (op == TranslateXY) return 2;
    return {};
}

constexpr float IntersectRayPlane(const ray &r, vec4 plane) {
    const float num = glm::dot(vec3{plane}, r.o) - plane.w;
    const float den = glm::dot(vec3{plane}, r.d);
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -num / den;
}

constexpr vec4 BuildPlane(vec3 p, const vec4 &p_normal) {
    const auto normal = glm::normalize(p_normal);
    return {vec3{normal}, glm::dot(normal, vec4{p, 1})};
}

constexpr float Length2(vec2 v) { return v.x * v.x + v.y * v.y; }
constexpr float Length2(vec3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// Homogeneous clip space to NDC
constexpr vec3 ToNDC(vec4 v) { return {fabsf(v.w) > FLT_EPSILON ? v / v.w : v}; }

constexpr float LengthClipSpaceSq(vec3 v, bool local = false) {
    const auto &mvp = local ? g.MVPLocal : g.MVP;
    return Length2(ToNDC(mvp * vec4{v, 1}) - ToNDC(mvp * vec4{0, 0, 0, 1}));
}

constexpr bool IsDirNeg(vec3 dir, bool local = false) {
    return LengthClipSpaceSq(dir, local) + FLT_EPSILON < LengthClipSpaceSq(-dir, local);
}

constexpr mat3 DirUnary{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
constexpr vec3 DirAxis(uint32_t axis_i, bool local = false) {
    return DirUnary[axis_i] * (IsDirNeg(DirUnary[axis_i], local) ? -1.f : 1.f);
}
constexpr std::pair<vec3, vec3> DirPlaneXY(uint32_t axis_i, bool local = false) {
    return {
        DirUnary[(axis_i + 1) % 3] * (IsDirNeg(DirUnary[(axis_i + 1) % 3], local) ? -1.f : 1.f),
        DirUnary[(axis_i + 2) % 3] * (IsDirNeg(DirUnary[(axis_i + 2) % 3], local) ? -1.f : 1.f)
    };
}

constexpr bool IsAxisVisible(vec3 dir, bool local = false) {
    static constexpr float AxisLimit{0.02};
    return LengthClipSpaceSq(dir * g.ScreenFactor, local) > AxisLimit * AxisLimit;
}
constexpr bool IsPlaneVisible(vec3 dir_x, vec3 dir_y) {
    static constexpr auto ToScreenNDC = [](vec3 v) { return vec2{ToNDC(g.MVP * vec4{v, 1})}; };
    static constexpr float ParallelogramAreaLimit{0.0025};
    const auto o = ToScreenNDC(vec3{0});
    const auto pa = ToScreenNDC(dir_x * g.ScreenFactor) - o;
    const auto pb = ToScreenNDC(dir_y * g.ScreenFactor) - o;
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
    if (op == TranslateScreen) return std::format("{} {} {}", Axis(0, v.x), Axis(1, v.y), Axis(2, v.z));
    if (op == TranslateYZ) return std::format("{} {}", Axis(1, v.y), Axis(2, v.z));
    if (op == TranslateZX) return std::format("{} {}", Axis(2, v.z), Axis(0, v.x));
    if (op == TranslateXY) return std::format("{} {}", Axis(0, v.x), Axis(1, v.y));
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
    const mat4 &M;
    const mat4 &Ortho;
    const mat4 &Inv;
};

mat4 Transform(const mat4 &m, Model model, Mode mode, Op type, vec2 mouse_pos, const ray &mouse_ray, std::optional<vec3> snap) {
    const auto p = Pos(m);
    if (HasAnyOp(type, Translate)) {
        auto delta = mouse_ray(fabsf(IntersectRayPlane(mouse_ray, g.InteractionPlane))) -
            (g.MouseRayOrigin(IntersectRayPlane(g.MouseRayOrigin, g.InteractionPlane)) - g.MatrixOrigin) - p;
        // Single axis constraint
        if (type == (Translate | AxisX) || type == (Translate | AxisY) || type == (Translate | AxisZ)) {
            const auto axis_i = AxisIndex(type, Translate);
            delta = m[axis_i] * glm::dot(m[axis_i], vec4{delta, 0});
        }
        if (snap) {
            const vec4 d{p + delta - g.MatrixOrigin, 0};
            const vec3 delta_cumulative = mode == Local || type == TranslateScreen ? model.M * vec4{Snap(model.Inv * d, *snap), 0} : Snap(d, *snap);
            delta = g.MatrixOrigin + delta_cumulative - p;
        }
        return glm::translate(mat4{1}, delta) * model.M;
    }
    if (HasAnyOp(type, Scale)) {
        if (type == ScaleXYZ) {
            g.Scale = vec3{1 + (mouse_pos.x - g.MousePosOrigin.x) * 0.01f};
        } else { // Single axis constraint
            const auto axis_i = AxisIndex(type, Scale);
            const vec3 axis_value{model.Ortho[axis_i]};
            const auto relative_origin = g.MouseRayOrigin(IntersectRayPlane(g.MouseRayOrigin, g.InteractionPlane)) - g.MatrixOrigin;
            const auto p_ortho = Pos(model.Ortho);
            const auto base = relative_origin / g.ScreenFactor - p_ortho;
            const auto delta = axis_value * glm::dot(axis_value, mouse_ray(IntersectRayPlane(mouse_ray, g.InteractionPlane)) - relative_origin - p_ortho);
            g.Scale[axis_i] = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
        }

        if (snap) g.Scale = Snap(g.Scale, *snap);
        for (uint32_t i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        return model.Ortho * glm::scale(mat4{1}, g.Scale * g.ScaleOrigin);
    }

    // Rotation: Compute angle on plane relative to the rotation origin
    const auto rotation_origin = glm::normalize(g.MouseRayOrigin(IntersectRayPlane(g.MouseRayOrigin, g.InteractionPlane)) - g.MatrixOrigin);
    const auto perp = glm::normalize(glm::cross(rotation_origin, vec3{g.InteractionPlane}));
    const auto pos_local = glm::normalize(mouse_ray(IntersectRayPlane(mouse_ray, g.InteractionPlane)) - p);
    float rotation_angle = acosf(glm::clamp(glm::dot(pos_local, rotation_origin), -1.f, 1.f)) * -glm::sign(glm::dot(pos_local, perp));
    if (snap) rotation_angle = Snap(rotation_angle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{model.Inv} * g.InteractionPlane); // Assumes affine model
    const mat4 rot_delta{glm::rotate(mat4{1}, rotation_angle - g.RotationAngle, rot_axis_local)};
    g.RotationAngle = rotation_angle;

    if (mode == Local) {
        const vec3 model_scale{glm::length(model.M[0]), glm::length(model.M[1]), glm::length(model.M[2])};
        return model.Ortho * rot_delta * glm::scale(mat4{1}, model_scale);
    }

    // Apply rotation, preserving translation
    auto res = rot_delta * mat4{mat3{model.M}};
    res[3] = vec4{Pos(model.M), 1};
    return res;
}

constexpr ImVec2 WorldToScreen(vec3 world, const mat4 &m) {
    auto trans = vec2{m * vec4{world, 1}} * (0.5f / glm::dot(glm::transpose(m)[3], vec4{world, 1})) + 0.5f;
    trans.y = 1 - trans.y;
    return std::bit_cast<ImVec2>(g.Pos + trans * g.Size);
}

constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 s1, ImVec2 s2) {
    const auto vec = std::bit_cast<vec2>(s2 - s1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, std::bit_cast<vec2>(p - s1));
    if (t <= 0) return s1;
    if (t * t > Length2(vec)) return s2;
    return s1 + std::bit_cast<ImVec2>(v) * t;
}

Op FindHoveredOp(Model model, Op op, ImVec2 mouse_pos, const ray &mouse_ray, const mat4 &view, const mat4 &view_proj) {
    static constexpr auto SelectDistSq = Style.CircleRad * Style.CircleRad;

    const auto center = WorldToScreen(vec3{0}, g.MVP);
    // Op selection check order is important because of universal mode.
    // Universal = Scale | Translate | Rotate
    if (HasAnyOp(op, Scale)) {
        if (op != Universal) {
            if (ImLengthSqr(mouse_pos - center) <= Style.CircleRad * Style.CircleRad) return ScaleXYZ;

            for (uint32_t i = 0; i < 3; ++i) {
                const auto dir = model.Ortho * vec4{DirAxis(i, true), 0};
                const auto p = Pos(model.Ortho);
                const auto pos_plane = WorldToScreen(mouse_ray(IntersectRayPlane(mouse_ray, BuildPlane(p, dir))), view_proj);
                const auto start = WorldToScreen(vec4{p, 1} + dir * g.ScreenFactor * 0.1f, view_proj);
                const auto end = WorldToScreen(vec4{p, 1} + dir * g.ScreenFactor, view_proj);
                if (ImLengthSqr(PointOnSegment(pos_plane, start, end) - pos_plane) < SelectDistSq) return Scale | AxisOp(i);
            }
        } else { // Universal
            if (float dist_sq = ImLengthSqr(mouse_pos - center); dist_sq >= 17 * 17 && dist_sq < 23 * 23) {
                return ScaleXYZ;
            }
            for (uint32_t i = 0; i < 3; ++i) {
                if (auto dir = DirAxis(i, true); IsAxisVisible(dir, true)) {
                    const auto end = WorldToScreen(dir * g.ScreenFactor, g.MVPLocal);
                    if (ImLengthSqr(end - mouse_pos) < SelectDistSq) return Scale | AxisOp(i);
                }
            }
        }
    }
    if (HasAnyOp(op, Translate)) {
        if (ImLengthSqr(mouse_pos - center) <= Style.CircleRad * Style.CircleRad) return TranslateScreen;

        const auto pos = std::bit_cast<ImVec2>(g.Pos), pos_screen{mouse_pos - pos};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto dir = model.M * vec4{DirAxis(i), 0};
            const auto start = WorldToScreen(vec4{Pos(model.M), 1} + dir * g.ScreenFactor * 0.1f, view_proj) - pos;
            const auto end = WorldToScreen(vec4{Pos(model.M), 1} + dir * g.ScreenFactor, view_proj) - pos;
            if (ImLengthSqr(PointOnSegment(pos_screen, start, end) - pos_screen) < SelectDistSq) return Translate | AxisOp(i);

            const auto [dir_plane_x, dir_plane_y] = DirPlaneXY(i);
            if (IsPlaneVisible(model.M * vec4{dir_plane_x, 0}, model.M * vec4{dir_plane_y, 0})) {
                const auto p_world = Pos(model.M);
                const auto pos_plane = mouse_ray(IntersectRayPlane(mouse_ray, BuildPlane(p_world, dir)));
                const auto plane_x_world = vec3{model.M * vec4{dir_plane_x, 0}};
                const auto plane_y_world = vec3{model.M * vec4{dir_plane_y, 0}};
                const auto delta_world = (pos_plane - p_world) / g.ScreenFactor;

                const float dx = glm::dot(delta_world, plane_x_world);
                const float dy = glm::dot(delta_world, plane_y_world);
                if (dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3]) return TranslatePlanes[i];
            }
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr float SelectDist = 8;
        const auto dist_sq = ImLengthSqr(mouse_pos - center);
        const auto rotation_radius = (g.Size.y / 2) * Style.SizeClipSpace * Style.RotationDisplayScale * 1.3f;
        const auto inner_rad = rotation_radius - SelectDist / 2, outer_rad = rotation_radius + SelectDist / 2;
        if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return RotateScreen;

        const auto p = Pos(model.M);
        const auto mv_pos = view * vec4{p, 1};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto intersect_pos_world = mouse_ray(IntersectRayPlane(mouse_ray, BuildPlane(p, model.M[i])));
            const auto intersect_pos = view * vec4{vec3{intersect_pos_world}, 1};
            if (fabsf(mv_pos.z) - fabsf(intersect_pos.z) < -FLT_EPSILON) continue;

            const auto circle_pos_world = model.Inv * vec4{glm::normalize(intersect_pos_world - p), 0};
            const auto circle_pos = WorldToScreen(circle_pos_world * Style.RotationDisplayScale * g.ScreenFactor, g.MVP);
            if (ImLengthSqr(circle_pos - mouse_pos) < SelectDist * SelectDist) return Rotate | AxisOp(i);
        }
    }
    return NoOp;
}

void Render(const mat4 &m, Op op, Op type, const mat4 &view_proj, vec3 cam_to_model) {
    static const auto DrawHatchedAxis = [](vec3 axis) {
        if (Style.HatchedAxisLineWidth <= 0) return;

        for (uint32_t i = 1; i < 10; ++i) {
            const auto base = WorldToScreen(axis * 0.05f * float(i * 2) * g.ScreenFactor, g.MVP);
            const auto end = WorldToScreen(axis * 0.05f * float(i * 2 + 1) * g.ScreenFactor, g.MVP);
            ImGui::GetWindowDrawList()->AddLine(base, end, Color.HatchedAxisLines, Style.HatchedAxisLineWidth);
        }
    };

    auto &dl = *ImGui::GetWindowDrawList();
    const auto p = Pos(m);
    const auto origin = WorldToScreen(p, view_proj);
    if (HasAnyOp(op, Translate)) {
        for (uint32_t i = 0; i < 3; ++i) {
            const bool is_neg = IsDirNeg(DirUnary[i]);
            const auto dir = DirUnary[i] * (is_neg ? -1.f : 1.f);
            const bool below_axis_limit = IsAxisVisible(dir);
            const bool using_type = type == (Translate | AxisOp(i));
            if ((!g.Using || using_type) && below_axis_limit) {
                const auto base = WorldToScreen(dir * g.ScreenFactor * 0.1f, g.MVP);
                const auto end = WorldToScreen(dir * g.ScreenFactor, g.MVP);
                const auto color = using_type ? Color.Selection : Color.Directions[i];
                dl.AddLine(base, end, color, Style.LineWidth);
                // In universal mode, draw scale circles instead of translate arrows.
                // (Show arrow when using though.)
                if (op != Universal || g.Using) {
                    const auto dir = (origin - end) * Style.LineArrowSize / sqrtf(ImLengthSqr(origin - end));
                    const ImVec2 orth_dir{dir.y, -dir.x};
                    dl.AddTriangleFilled(end - dir, end + dir + orth_dir, end + dir - orth_dir, color);
                }
                if (is_neg) DrawHatchedAxis(dir);
            }
            if (!g.Using || type == TranslatePlanes[i]) {
                const auto [dir_plane_x, dir_plane_y] = DirPlaneXY(i);
                if (IsPlaneVisible(dir_plane_x, dir_plane_y)) {
                    static ImVec2 quad_pts_screen[4];
                    for (uint32_t j = 0; j < 4; ++j) {
                        const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                        quad_pts_screen[j] = WorldToScreen(corner_pos_world, g.MVP);
                    }
                    dl.AddPolyline(quad_pts_screen, 4, Color.Directions[i], true, 1.0f);
                    dl.AddConvexPolyFilled(quad_pts_screen, 4, type == TranslatePlanes[i] ? Color.Selection : Color.Planes[i]);
                }
            }
        }
        if (!g.Using || type == TranslateScreen) {
            const auto color = type == TranslateScreen ? Color.Selection : IM_COL32_WHITE;
            dl.AddCircleFilled(WorldToScreen(vec3{0}, g.MVP), Style.CircleRad, color, 32);
        }
        if (g.Using && HasAnyOp(type, Translate)) {
            const auto translation_line_color = Color.TranslationLine;
            const auto source_pos_screen = WorldToScreen(g.MatrixOrigin, view_proj);
            const auto dif = std::bit_cast<ImVec2>(vec2{glm::normalize(vec4{origin.x - source_pos_screen.x, origin.y - source_pos_screen.y, 0, 0}) * 5.f});
            dl.AddCircle(source_pos_screen, 6.f, translation_line_color);
            dl.AddCircle(origin, 6.f, translation_line_color);
            dl.AddLine(source_pos_screen + dif, origin - dif, translation_line_color, 2.f);

            const auto delta_info = p - g.MatrixOrigin;
            const auto formatted = Format::Translation(type, delta_info);
            dl.AddText(origin + ImVec2{15, 15}, Color.TextShadow, formatted.data());
            dl.AddText(origin + ImVec2{14, 14}, Color.Text, formatted.data());
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr uint32_t HalfCircleSegmentCount{128}, FullCircleSegmentCount{HalfCircleSegmentCount * 2 + 1};
        static ImVec2 CirclePositions[FullCircleSegmentCount];
        if (g.Using) {
            const auto u = glm::normalize(g.MouseRayOrigin(IntersectRayPlane(g.MouseRayOrigin, g.InteractionPlane)) - p);
            const auto v = glm::cross(glm::normalize(vec3{g.InteractionPlane}), u);
            const auto sign = g.RotationAngle < 0 ? -1.f : 1.f;
            for (uint32_t i = 0; i < FullCircleSegmentCount; ++i) {
                const float ng = sign * 2 * M_PI * float(i) / float(FullCircleSegmentCount - 1);
                const vec3 pos = cosf(ng) * u + sinf(ng) * v;
                CirclePositions[i] = WorldToScreen(p + pos * g.ScreenFactor * Style.RotationDisplayScale * (type == RotateScreen ? 1.2f : 1.f), view_proj);
            }

            dl.AddPolyline(CirclePositions, FullCircleSegmentCount, Color.Selection, false, Style.RotationLineWidth);
            const uint32_t angle_i = float(FullCircleSegmentCount - 1) * fabsf(g.RotationAngle) / (2 * M_PI);
            if (angle_i > 1) {
                CirclePositions[angle_i + 1] = origin;
                dl.AddConvexPolyFilled(CirclePositions, angle_i + 2, Color.RotationFillActive);
                dl.AddLine(origin, CirclePositions[0], Color.RotationBorderActive, Style.RotationLineWidth / 2);
            }
            dl.AddLine(origin, CirclePositions[angle_i], Color.RotationBorderActive, Style.RotationLineWidth);
            {
                const auto formatted = Format::Rotation(type, g.RotationAngle);
                const auto dest_pos = CirclePositions[1];
                dl.AddText(dest_pos + ImVec2{15, 15}, Color.TextShadow, formatted.data());
                dl.AddText(dest_pos + ImVec2{14, 14}, Color.Text, formatted.data());
            }
        } else {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const auto point_count = HalfCircleSegmentCount + 1;
                const float angle_start = M_PI_2 + atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]);
                for (uint32_t i = 0; i < point_count; ++i) {
                    const float ng = angle_start + M_PI * float(i) / float(point_count - 1);
                    const vec4 axis_pos{cosf(ng), sinf(ng), 0, 0};
                    const vec3 pos{axis_pos[axis], axis_pos[(axis + 1) % 3], axis_pos[(axis + 2) % 3]};
                    CirclePositions[i] = WorldToScreen(pos * g.ScreenFactor * Style.RotationDisplayScale, g.MVP);
                }
                const auto color = type == (Rotate | AxisOp(2 - axis)) ? Color.Selection : Color.Directions[2 - axis];
                dl.AddPolyline(CirclePositions, point_count, color, false, Style.RotationLineWidth);
            }
            dl.AddCircle(
                origin,
                (g.Size.y / 2) * Style.SizeClipSpace * Style.RotationDisplayScale * 1.3f,
                type == RotateScreen ? Color.Selection : IM_COL32_WHITE, FullCircleSegmentCount, Style.RotationLineWidth * 1.5f
            );
        }
    }
    if (HasAnyOp(op, Scale)) {
        if (!g.Using) {
            for (uint32_t i = 0; i < 3; ++i) {
                const bool is_neg = IsDirNeg(DirUnary[i], true);
                const auto dir = DirUnary[i] * (is_neg ? -1.f : 1.f);
                if (IsAxisVisible(dir, true)) {
                    const auto color = type == (Scale | AxisOp(i)) ? Color.Selection : Color.Directions[i];
                    if (op != Universal) {
                        const auto base = WorldToScreen(dir * g.ScreenFactor * 0.1f, g.MVP);
                        if (g.Using) {
                            const auto center = WorldToScreen(dir * g.ScreenFactor, g.MVP);
                            dl.AddLine(base, center, Color.ScaleLine, Style.LineWidth);
                            dl.AddCircleFilled(center, Style.CircleRad, Color.ScaleLine);
                        }
                        const auto end = WorldToScreen(dir * g.ScreenFactor, g.MVP);
                        dl.AddLine(base, end, color, Style.LineWidth);
                        dl.AddCircleFilled(end, Style.CircleRad, color);
                        if (is_neg) DrawHatchedAxis(dir);
                    } else {
                        const auto end = WorldToScreen(dir * g.ScreenFactor, g.MVPLocal);
                        dl.AddCircleFilled(end, Style.CircleRad, color);
                    }
                }
            }
        }
        if (!g.Using || HasAnyOp(type, Scale)) {
            const auto circle_color = g.Using || type == ScaleXYZ ? Color.Selection : IM_COL32_WHITE;
            const auto circle_pos = WorldToScreen(vec3{0}, g.MVP);
            if (op != Universal) dl.AddCircleFilled(circle_pos, Style.CircleRad, circle_color, 32);
            else dl.AddCircle(circle_pos, 20.f, circle_color, 32, Style.CircleRad);
            if (g.Using) {
                const auto formatted = Format::Scale(type, g.Scale);
                dl.AddText(origin + ImVec2{15, 15}, Color.TextShadow, formatted.data());
                dl.AddText(origin + ImVec2{14, 14}, Color.Text, formatted.data());
            }
        }
    }
}
} // namespace

namespace ModelGizmo {
bool Draw(Mode mode, Op op, vec2 pos, vec2 size, vec2 mouse_pos, ray mouse_ray, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap) {
    g.Pos = pos;
    g.Size = size;
    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (HasAnyOp(op, Scale)) mode = Local;

    const auto p = Pos(m);
    const mat4 m_ortho{glm::normalize(m[0]), glm::normalize(m[1]), glm::normalize(m[2]), m[3]};
    const auto m_ = mode == Local ? m_ortho : glm::translate(mat4{1}, p);
    const mat3 m_inv = InverseRigid(m_);
    const mat4 view_proj = proj * view;
    g.MVP = view_proj * m_;
    g.MVPLocal = view_proj * m_ortho;

    // Behind‐camera cull
    if (!g.Using && (g.MVP * vec4{vec3{0}, 1}).z < 0.001f) return false;

    const auto view_inv = InverseRigid(view);
    const ray camera_ray{Pos(view_inv), Dir(view_inv)};

    // Compute scale from camera right vector projected onto screen at m pos
    g.ScreenFactor = Style.SizeClipSpace / sqrtf(LengthClipSpaceSq(m_inv * vec4{vec3{Right(view_inv)}, 0}));

    const bool commit = g.Using && !ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (commit) g.Using = false;

    if (g.Using) {
        m = Transform(m_, {m, m_ortho, m_inv}, mode, g.Op, mouse_pos, mouse_ray, snap);
    } else if (ImGui::IsWindowHovered()) {
        if (g.Op = FindHoveredOp({m_, m_ortho, m_inv}, op, std::bit_cast<ImVec2>(mouse_pos), mouse_ray, view, view_proj);
            g.Op != NoOp && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g.Using = true;
            g.MatrixOrigin = p;
            g.MousePosOrigin = mouse_pos;
            g.MouseRayOrigin = mouse_ray;
            g.Scale = {1, 1, 1};
            g.RotationAngle = 0;
            if (HasAnyOp(g.Op, Scale)) g.ScaleOrigin = {glm::length(m[0]), glm::length(m[1]), glm::length(m[2])};
            const auto GetPlaneNormal = [&](Op op) {
                if (g.Op == ScaleXYZ || g.Op == RotateScreen || g.Op == TranslateScreen) return -vec4{camera_ray.d, 0};
                if (HasAnyOp(g.Op, Scale)) return m_[(AxisIndex(g.Op, Scale) + 1) % 3];
                if (HasAnyOp(g.Op, Rotate)) return mode == Local ? m_[AxisIndex(g.Op, Rotate)] : vec4{DirUnary[AxisIndex(g.Op, Rotate)], 0};
                if (auto plane_index = TranslatePlaneIndex(op)) return m_[*plane_index];

                const auto n = glm::normalize(vec3{m_[AxisIndex(op, Translate)]});
                const auto v = glm::normalize(g.MatrixOrigin - camera_ray.o);
                return vec4{v - n * glm::dot(n, v), 0};
            };
            g.InteractionPlane = BuildPlane(g.MatrixOrigin, GetPlaneNormal(g.Op));
        }
    }
    Render(m_, op, g.Op, view_proj, mat3{m_inv} * glm::normalize(p - camera_ray.o));
    return g.Using || commit;
}
} // namespace ModelGizmo
