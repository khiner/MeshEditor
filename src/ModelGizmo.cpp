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

    float ScreenFactor;
    vec3 RelativeOrigin;
    vec4 TranslationPlane;
    vec3 TranslationPlaneOrigin;
    vec3 MatrixOrigin;

    vec3 RotationVectorSource;
    float RotationAngle, RotationRadius;

    vec3 Scale, ScaleOrigin;
    vec2 StartMousePos;
    vec2 Pos{0, 0}, Size{0, 0};

    Op Op{NoOp};
    bool Using{false};
};

struct Style {
    float LineWidth{3}; // Thickness of lines for translate/scale gizmo
    float HatchedAxisLineWidth{6}; // Thickness of hatched axis lines
    float LineArrowSize{6}; // Size of arrow at the end of translation lines
    float CircleRad{6}; // Radius of circle at the end of scale lines and the center of the translate/scale gizmo
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
constexpr vec4 Up(const mat4 &m) { return {m[1]}; }
constexpr vec4 Dir(const mat4 &m) { return {m[2]}; }
constexpr vec3 Pos(const mat4 &m) { return {m[3]}; } // Assume affine matrix, with w = 1

constexpr vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
constexpr ImVec2 ToImVec(vec2 v) { return {v.x, v.y}; }

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

constexpr float ComputeAngleOnPlane(const mat4 &m, const ray &r) {
    const auto perp = glm::normalize(glm::cross(g.RotationVectorSource, vec3{g.TranslationPlane}));
    const auto pos_local = glm::normalize(r(IntersectRayPlane(r, g.TranslationPlane)) - Pos(m));
    const float acos_angle = glm::clamp(glm::dot(vec3{pos_local}, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (glm::dot(pos_local, perp) < 0 ? 1 : -1);
}

constexpr vec4 BuildPlane(vec3 p, const vec4 &p_normal) {
    const auto normal = glm::normalize(p_normal);
    return {vec3{normal}, glm::dot(normal, vec4{p, 1})};
}

constexpr float Length2(vec2 v) { return v.x * v.x + v.y * v.y; }
constexpr float Length2(vec3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 s1, ImVec2 s2) {
    const auto vec = ToGlm(s2 - s1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, ToGlm(p - s1));
    if (t <= 0) return s1;
    if (t * t > Length2(vec)) return s2;
    return s1 + ToImVec(v) * t;
}

// Homogeneous clip space to NDC
constexpr vec3 ToNDC(vec4 v) { return {fabsf(v.w) > FLT_EPSILON ? v / v.w : v}; }

constexpr float Length2ClipSpace(vec3 v, bool local = false) {
    const auto &mvp = local ? g.MVPLocal : g.MVP;
    return Length2(ToNDC(mvp * vec4{v, 1}) - ToNDC(mvp * vec4{0, 0, 0, 1}));
}

constexpr bool IsDirNeg(vec3 dir, bool local = false) {
    return Length2ClipSpace(dir, local) + FLT_EPSILON < Length2ClipSpace(-dir, local);
}

constexpr mat3 DirUnary{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
constexpr vec3 DirAxis(uint32_t axis_i, bool local = false) {
    return DirUnary[axis_i] * (IsDirNeg(DirUnary[axis_i], local) ? -1.f : 1.f);
}
constexpr vec3 DirPlaneX(uint32_t axis_i, bool local = false) {
    return DirUnary[(axis_i + 1) % 3] * (IsDirNeg(DirUnary[(axis_i + 1) % 3], local) ? -1.f : 1.f);
}
constexpr vec3 DirPlaneY(uint32_t axis_i, bool local = false) {
    return DirUnary[(axis_i + 2) % 3] * (IsDirNeg(DirUnary[(axis_i + 2) % 3], local) ? -1.f : 1.f);
}

constexpr bool IsAxisVisible(vec3 dir_axis, bool local = false) {
    static constexpr float AxisLimit{0.02};
    return Length2ClipSpace(dir_axis * g.ScreenFactor, local) > AxisLimit * AxisLimit;
}
constexpr bool IsPlaneVisible(vec3 dir_plane_x, vec3 dir_plane_y) {
    static constexpr auto ToScreenNDC = [](vec3 v) { return vec2{ToNDC(g.MVP * vec4{v, 1})}; };
    static constexpr float ParallelogramAreaLimit{0.0025};
    const auto o = ToScreenNDC(vec3{0});
    const auto pa = ToScreenNDC(dir_plane_x * g.ScreenFactor) - o;
    const auto pb = ToScreenNDC(dir_plane_y * g.ScreenFactor) - o;
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
    if (HasAnyOp(type, Translate)) {
        const float len_signed = IntersectRayPlane(mouse_ray, g.TranslationPlane);
        const float len = fabsf(len_signed); // near plane
        const auto new_origin = mouse_ray(len) - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(m);

        // 1 axis constraint
        if (type == (Translate | AxisX) || type == (Translate | AxisY) || type == (Translate | AxisZ)) {
            const auto axis_i = AxisIndex(type, Translate);
            delta = m[axis_i] * glm::dot(m[axis_i], vec4{delta, 0});
        }

        if (snap) {
            auto delta_cumulative = Pos(m) + delta - vec3{g.MatrixOrigin};
            if (mode == Local || type == TranslateScreen) {
                auto ms_norm = model.M;
                ms_norm[0] = glm::normalize(model.M[0]);
                ms_norm[1] = glm::normalize(model.M[1]);
                ms_norm[2] = glm::normalize(model.M[2]);
                delta_cumulative = glm::inverse(ms_norm) * vec4{delta_cumulative, 0};
                delta_cumulative = Snap(vec4{delta_cumulative, 0}, *snap);
                delta_cumulative = ms_norm * vec4{delta_cumulative, 0};
            } else {
                delta_cumulative = Snap(vec4{delta_cumulative, 0}, *snap);
            }
            delta = g.MatrixOrigin + delta_cumulative - Pos(m);
        }

        return glm::translate(mat4{1}, vec3{delta}) * model.M;
    }
    if (HasAnyOp(type, Scale)) {
        const float len = IntersectRayPlane(mouse_ray, g.TranslationPlane);
        const auto new_origin = mouse_ray(len) - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(model.Ortho);
        // 1 axis constraint
        if (type != ScaleXYZ) {
            const auto axis_i = AxisIndex(type, Scale);
            const auto axis_value = vec3{model.Ortho[axis_i]};
            const float length_on_axis = glm::dot(axis_value, delta);
            delta = axis_value * length_on_axis;

            const auto base = g.TranslationPlaneOrigin - Pos(model.Ortho);
            const float ratio = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
            g.Scale[axis_i] = std::max(ratio, 0.001f);
        } else {
            const float scale_delta = (mouse_pos.x - g.StartMousePos.x) * 0.01f;
            g.Scale = vec3{std::max(1.f + scale_delta, 0.001f)};
        }

        if (snap) g.Scale = Snap(g.Scale, *snap);

        // no 0 allowed
        for (uint32_t i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        return model.Ortho * glm::scale(mat4{1}, g.Scale * g.ScaleOrigin);
    }

    assert(HasAnyOp(type, Rotate));
    float rotation_angle = ComputeAngleOnPlane(m, mouse_ray);
    if (snap) rotation_angle = Snap(rotation_angle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{model.Inv} * g.TranslationPlane); // Assumes affine model
    const mat4 delta_rot{glm::rotate(mat4{1}, rotation_angle - g.RotationAngle, rot_axis_local)};
    g.RotationAngle = rotation_angle;

    if (mode == Local) {
        const vec3 model_scale{glm::length(model.M[0]), glm::length(model.M[1]), glm::length(model.M[2])};
        return model.Ortho * delta_rot * glm::scale(mat4{1}, model_scale);
    }

    auto res = model.M;
    res[3] = {0, 0, 0, 1};
    res = delta_rot * res;
    res[3] = vec4{Pos(model.M), 1};
    return res;
}

constexpr ImVec2 WorldToPos(vec3 pos_world, const mat4 &m) {
    auto trans = vec2{m * vec4{pos_world, 1}} * (0.5f / glm::dot(glm::transpose(m)[3], vec4{pos_world, 1})) + 0.5f;
    trans.y = 1 - trans.y;
    return ToImVec(g.Pos + trans * g.Size);
}

constexpr float RotationDisplayScale{1.2}; // Scale a bit so translate axes don't touch when in universal.
constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

Op FindHoveredOp(Model model, Op op, ImVec2 mouse_pos, const ray &mouse_ray, const mat4 &view, const mat4 &view_proj) {
    static constexpr auto SelectDistSq = Style.CircleRad * Style.CircleRad;

    const auto center = WorldToPos(vec3{0}, g.MVP);
    // Op selection check order is important because of universal mode.
    // Universal = Scale | Translate | Rotate
    if (HasAnyOp(op, Scale)) {
        if (op != Universal) {
            if (ImLengthSqr(mouse_pos - center) <= Style.CircleRad * Style.CircleRad) return ScaleXYZ;

            for (uint32_t i = 0; i < 3; ++i) {
                const auto dir_axis = model.Ortho * vec4{DirAxis(i, true), 0};
                const auto pos_plane = WorldToPos(mouse_ray(IntersectRayPlane(mouse_ray, BuildPlane(Pos(model.Ortho), dir_axis))), view_proj);
                const auto axis_start = WorldToPos(vec4{Pos(model.Ortho), 1} + dir_axis * g.ScreenFactor * 0.1f, view_proj);
                const auto axis_end = WorldToPos(vec4{Pos(model.Ortho), 1} + dir_axis * g.ScreenFactor, view_proj);
                const auto closest_on_axis = PointOnSegment(pos_plane, axis_start, axis_end);
                if (ImLengthSqr(closest_on_axis - pos_plane) < SelectDistSq) return Scale | AxisOp(i);
            }
        } else { // Universal
            if (float dist_sq = ImLengthSqr(mouse_pos - center); dist_sq >= 17 * 17 && dist_sq < 23 * 23) {
                return ScaleXYZ;
            }
            for (uint32_t i = 0; i < 3; ++i) {
                if (auto dir_axis = DirAxis(i, true); IsAxisVisible(dir_axis, true)) {
                    const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVPLocal);
                    if (ImLengthSqr(end - mouse_pos) < SelectDistSq) return Scale | AxisOp(i);
                }
            }
        }
    }
    if (HasAnyOp(op, Translate)) {
        if (ImLengthSqr(mouse_pos - center) <= Style.CircleRad * Style.CircleRad) return TranslateScreen;

        const auto pos = ToImVec(g.Pos), pos_screen{mouse_pos - pos};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto dir_axis = model.M * vec4{DirAxis(i), 0};
            const auto dir_plane_x = model.M * vec4{DirPlaneX(i), 0};
            const auto dir_plane_y = model.M * vec4{DirPlaneY(i), 0};
            const auto axis_start = WorldToPos(vec4{Pos(model.M), 1} + dir_axis * g.ScreenFactor * 0.1f, view_proj) - pos;
            const auto axis_end = WorldToPos(vec4{Pos(model.M), 1} + dir_axis * g.ScreenFactor, view_proj) - pos;
            const auto closest_on_axis = PointOnSegment(pos_screen, axis_start, axis_end);
            if (ImLengthSqr(closest_on_axis - pos_screen) < SelectDistSq) return Translate | AxisOp(i);

            if (IsPlaneVisible(dir_plane_x, dir_plane_y)) {
                const auto pos_plane = mouse_ray(IntersectRayPlane(mouse_ray, BuildPlane(Pos(model.M), dir_axis)));
                const float dx = glm::dot(vec3{dir_plane_x}, vec3{pos_plane - Pos(model.M)} / g.ScreenFactor);
                const float dy = glm::dot(vec3{dir_plane_y}, vec3{pos_plane - Pos(model.M)} / g.ScreenFactor);
                if (dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3]) return TranslatePlanes[i];
            }
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr float SelectDist = 8;
        const auto dist_sq = ImLengthSqr(mouse_pos - center);
        const auto inner_rad = g.RotationRadius - SelectDist / 2, outer_rad = g.RotationRadius + SelectDist / 2;
        if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return RotateScreen;

        const auto mv_pos = view * vec4{Pos(model.M), 1};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto pickup_plane = BuildPlane(Pos(model.M), model.M[i]);
            const auto intersect_world_pos = mouse_ray(IntersectRayPlane(mouse_ray, pickup_plane));
            const auto intersect_view_pos = view * vec4{vec3{intersect_world_pos}, 1};
            if (ImAbs(mv_pos.z) - ImAbs(intersect_view_pos.z) < -FLT_EPSILON) continue;

            const auto circle_pos = model.Inv * vec4{glm::normalize(intersect_world_pos - Pos(model.M)), 0};
            const auto circle_pos_screen = WorldToPos(circle_pos * RotationDisplayScale * g.ScreenFactor, g.MVP);
            if (ImLengthSqr(circle_pos_screen - mouse_pos) < SelectDist * SelectDist) return Rotate | AxisOp(i);
        }
    }
    return NoOp;
}

void Render(const mat4 &m, const mat4 &m_inv, Op op, Op type, const mat4 &view_proj, const mat4 &view_inv, const ray &camera_ray) {
    static const auto DrawHatchedAxis = [](vec3 axis) {
        if (Style.HatchedAxisLineWidth <= 0) return;

        for (uint32_t i = 1; i < 10; ++i) {
            const auto base = WorldToPos(axis * 0.05f * float(i * 2) * g.ScreenFactor, g.MVP);
            const auto end = WorldToPos(axis * 0.05f * float(i * 2 + 1) * g.ScreenFactor, g.MVP);
            ImGui::GetWindowDrawList()->AddLine(base, end, Color.HatchedAxisLines, Style.HatchedAxisLineWidth);
        }
    };

    static constexpr bool orthographic{false};
    const bool universal = op == Universal;
    auto &dl = *ImGui::GetWindowDrawList();
    const auto origin = WorldToPos(Pos(m), view_proj);
    if (HasAnyOp(op, Translate)) {
        for (uint32_t i = 0; i < 3; ++i) {
            const bool is_neg = IsDirNeg(DirUnary[i]);
            const auto dir_axis = DirUnary[i] * (is_neg ? -1.f : 1.f);
            const bool below_axis_limit = IsAxisVisible(dir_axis);
            const bool using_type = type == (Translate | AxisOp(i));
            if ((!g.Using || using_type) && below_axis_limit) {
                const auto base = WorldToPos(dir_axis * g.ScreenFactor * 0.1f, g.MVP);
                const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                const auto color = using_type ? Color.Selection : Color.Directions[i];
                dl.AddLine(base, end, color, Style.LineWidth);
                // In universal mode, draw scale circles instead of translate arrows.
                // (Show arrow when using though.)
                if (!universal || g.Using) {
                    const auto dir = (origin - end) * Style.LineArrowSize / sqrtf(ImLengthSqr(origin - end));
                    const ImVec2 orth_dir{dir.y, -dir.x};
                    dl.AddTriangleFilled(end - dir, end + dir + orth_dir, end + dir - orth_dir, color);
                }
                if (is_neg) DrawHatchedAxis(dir_axis);
            }
            if (!g.Using || type == TranslatePlanes[i]) {
                const auto dir_plane_x = DirPlaneX(i), dir_plane_y = DirPlaneY(i);
                if (IsPlaneVisible(dir_plane_x, dir_plane_y)) {
                    ImVec2 quad_pts_screen[4];
                    for (uint32_t j = 0; j < 4; ++j) {
                        const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                        quad_pts_screen[j] = WorldToPos(corner_pos_world, g.MVP);
                    }
                    dl.AddPolyline(quad_pts_screen, 4, Color.Directions[i], true, 1.0f);
                    const auto color = type == TranslatePlanes[i] ? Color.Selection : Color.Planes[i];
                    dl.AddConvexPolyFilled(quad_pts_screen, 4, color);
                }
            }
        }
        if (!g.Using || type == TranslateScreen) {
            const auto color = type == TranslateScreen ? Color.Selection : IM_COL32_WHITE;
            dl.AddCircleFilled(WorldToPos(vec3{0}, g.MVP), Style.CircleRad, color, 32);
        }
        if (g.Using && HasAnyOp(type, Translate)) {
            const auto translation_line_color = Color.TranslationLine;
            const auto source_pos_screen = WorldToPos(g.MatrixOrigin, view_proj);
            const auto dif = ToImVec(glm::normalize(vec4{origin.x - source_pos_screen.x, origin.y - source_pos_screen.y, 0, 0}) * 5.f);
            dl.AddCircle(source_pos_screen, 6.f, translation_line_color);
            dl.AddCircle(origin, 6.f, translation_line_color);
            dl.AddLine(source_pos_screen + dif, origin - dif, translation_line_color, 2.f);

            const auto delta_info = Pos(m) - g.MatrixOrigin;
            const auto formatted = Format::Translation(type, delta_info);
            dl.AddText(origin + ImVec2{15, 15}, Color.TextShadow, formatted.data());
            dl.AddText(origin + ImVec2{14, 14}, Color.Text, formatted.data());
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr uint32_t HalfCircleSegmentCount{64};
        static constexpr float RotateRadiusScale{0.06};
        g.RotationRadius = RotateRadiusScale * g.Size.y;

        // Assumes affine model
        const auto cam_to_model = mat3{m_inv} * vec3{orthographic ? -Dir(view_inv) : glm::normalize(Pos(m) - camera_ray.o)};
        static ImVec2 CirclePositions[HalfCircleSegmentCount * 2 + 1];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const bool is_type = type == (Rotate | AxisOp(2 - axis));
            const auto circle_mul = g.Using && is_type ? 2 : 1;
            const auto point_count = circle_mul * HalfCircleSegmentCount + 1;
            const auto angle_start = atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]) + M_PI_2;
            for (uint32_t i = 0; i < point_count; ++i) {
                const float ng = angle_start + float(circle_mul) * M_PI * (float(i) / float(point_count - 1));
                const vec4 axis_pos{cosf(ng), sinf(ng), 0, 0};
                const auto pos = vec4{axis_pos[axis], axis_pos[(axis + 1) % 3], axis_pos[(axis + 2) % 3], 0} * g.ScreenFactor * RotationDisplayScale;
                CirclePositions[i] = WorldToPos(pos, g.MVP);
            }
            if (!g.Using || is_type) {
                const auto color = is_type ? Color.Selection : Color.Directions[2 - axis];
                dl.AddPolyline(CirclePositions, point_count, color, false, Style.RotationLineWidth);
            }
            if (float radius_axis_sq = ImLengthSqr(origin - CirclePositions[0]);
                radius_axis_sq > g.RotationRadius * g.RotationRadius) {
                g.RotationRadius = sqrtf(radius_axis_sq);
            }
        }
        if (!g.Using || type == RotateScreen) {
            const auto color = type == RotateScreen ? Color.Selection : IM_COL32_WHITE;
            dl.AddCircle(origin, g.RotationRadius, color, 64, Style.RotationLineWidth * 1.5f);
        }
        if (g.Using && HasAnyOp(type, Rotate)) {
            static ImVec2 CirclePositions[HalfCircleSegmentCount + 1];
            CirclePositions[0] = origin;
            for (uint32_t i = 1; i < HalfCircleSegmentCount + 1; ++i) {
                const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
                const mat4 rotate{glm::rotate(mat4{1}, ng, vec3{g.TranslationPlane})};
                const auto pos = vec3{rotate * vec4{g.RotationVectorSource, 1}} * g.ScreenFactor * RotationDisplayScale;
                CirclePositions[i] = WorldToPos(pos + Pos(m), view_proj);
            }
            dl.AddConvexPolyFilled(CirclePositions, HalfCircleSegmentCount + 1, Color.RotationFillActive);
            dl.AddPolyline(CirclePositions, HalfCircleSegmentCount + 1, Color.RotationBorderActive, true, Style.RotationLineWidth);

            const auto formatted = Format::Rotation(type, g.RotationAngle);
            const auto dest_pos = CirclePositions[1];
            dl.AddText(dest_pos + ImVec2{15, 15}, Color.TextShadow, formatted.data());
            dl.AddText(dest_pos + ImVec2{14, 14}, Color.Text, formatted.data());
        }
    }
    if (HasAnyOp(op, Scale)) {
        if (!g.Using) {
            for (uint32_t i = 0; i < 3; ++i) {
                const vec4 dir{DirUnary[i], 0};
                const bool is_neg = IsDirNeg(dir, true);
                const auto dir_axis = dir * (is_neg ? -1.f : 1.f);
                if (IsAxisVisible(dir_axis, true)) {
                    const auto color = type == (Scale | AxisOp(i)) ? Color.Selection : Color.Directions[i];
                    if (!universal) {
                        const auto base = WorldToPos(dir_axis * g.ScreenFactor * 0.1f, g.MVP);
                        if (g.Using) {
                            const auto center = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                            dl.AddLine(base, center, Color.ScaleLine, Style.LineWidth);
                            dl.AddCircleFilled(center, Style.CircleRad, Color.ScaleLine);
                        }
                        const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                        dl.AddLine(base, end, color, Style.LineWidth);
                        dl.AddCircleFilled(end, Style.CircleRad, color);
                        if (is_neg) DrawHatchedAxis(dir_axis);
                    } else {
                        const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVPLocal);
                        dl.AddCircleFilled(end, Style.CircleRad, color);
                    }
                }
            }
        }
        if (!g.Using || HasAnyOp(type, Scale)) {
            const auto circle_color = g.Using || type == ScaleXYZ ? Color.Selection : IM_COL32_WHITE;
            const auto circle_pos = WorldToPos(vec3{0}, g.MVP);
            if (!universal) dl.AddCircleFilled(circle_pos, Style.CircleRad, circle_color, 32);
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
bool Draw(Mode mode, Op op, vec2 pos, vec2 size, vec2 mouse_pos, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap) {
    g.Pos = pos;
    g.Size = size;

    mat4 m_ortho; // orthonormalized model
    m_ortho[0] = glm::normalize(m[0]);
    m_ortho[1] = glm::normalize(m[1]);
    m_ortho[2] = glm::normalize(m[2]);
    m_ortho[3] = m[3];

    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (HasAnyOp(op, Scale)) mode = Local;

    const auto m_ = mode == Local ? m_ortho : glm::translate(mat4{1}, Pos(m));
    const mat4 view_proj = proj * view;
    g.MVP = view_proj * m_;
    g.MVPLocal = view_proj * m_ortho;

    // Check if behind camera
    const auto pos_cam_space = g.MVP * vec4{vec3{0}, 1};
    static constexpr bool orthographic{false};
    if (!orthographic && pos_cam_space.z < 0.001 && !g.Using) return false;

    const auto m_inv = glm::inverse(m_);
    const mat4 view_inv = glm::inverse(view);
    const ray camera_ray{Dir(view_inv), Pos(view_inv)};

    // Compute scale from camera right vector projected onto screen at m pos
    static constexpr float GizmoSizeClipSpace{0.1};
    g.ScreenFactor = GizmoSizeClipSpace / sqrtf(Length2ClipSpace(m_inv * vec4{vec3{Right(view_inv)}, 0}));

    // Compute mouse ray
    const auto view_proj_inv = glm::inverse(proj * view);
    const auto mouse_pos_rel = mouse_pos - g.Pos;
    const float mox = (mouse_pos_rel.x / g.Size.x) * 2 - 1;
    const float moy = (1 - mouse_pos_rel.y / g.Size.y) * 2 - 1;
    const vec4 near_pos{proj * vec4{0, 0, 1, 1}};
    const vec4 far_pos{proj * vec4{0, 0, 2, 1}};
    const bool reversed = near_pos.z / near_pos.w > far_pos.z / far_pos.w;
    const float z_near = reversed ? 1 - FLT_EPSILON : 0;
    const float z_far = reversed ? 0 : 1 - FLT_EPSILON;
    const auto ray_o = ToNDC(view_proj_inv * vec4{mox, moy, z_near, 1});
    const ray mouse_ray{ray_o, glm::normalize(ToNDC(view_proj_inv * vec4{mox, moy, z_far, 1}) - ray_o)};

    const bool commit = g.Using && !ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (commit) g.Using = false;

    if (g.Using) {
        m = Transform(m_, {m, m_ortho, m_inv}, mode, g.Op, mouse_pos, mouse_ray, snap);
    } else if (ImGui::IsWindowHovered()) {
        if (g.Op = FindHoveredOp({m_, m_ortho, m_inv}, op, ToImVec(mouse_pos), mouse_ray, view, view_proj);
            g.Op != NoOp && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            g.Using = true;
            const auto p = Pos(m_);
            if (HasAnyOp(g.Op, Scale)) {
                g.TranslationPlane = BuildPlane(p, g.Op == ScaleXYZ ? -vec4{camera_ray.d, 0} : m_[(AxisIndex(g.Op, Scale) + 1) % 3]);
                g.TranslationPlaneOrigin = mouse_ray(IntersectRayPlane(mouse_ray, g.TranslationPlane));
                g.Scale = {1, 1, 1};
                g.MatrixOrigin = p;
                g.RelativeOrigin = (g.TranslationPlaneOrigin - p) / g.ScreenFactor;
                g.ScaleOrigin = {glm::length(Right(m)), glm::length(Up(m)), glm::length(Dir(m))};
                g.StartMousePos = mouse_pos;
            }
            if (HasAnyOp(g.Op, Translate)) {
                const auto GetTranslationPlane = [&](Op op) {
                    if (op == TranslateScreen) return -vec4{camera_ray.d, 0};
                    if (auto plane_index = TranslatePlaneIndex(op)) return m_[*plane_index];

                    const auto plane = m_[AxisIndex(op, Translate)];
                    const auto cam_to_model = glm::normalize(p - camera_ray.o);
                    return glm::normalize(vec4{glm::cross(vec3{plane}, glm::cross(vec3{plane}, cam_to_model)), 0});
                };

                g.TranslationPlane = BuildPlane(p, GetTranslationPlane(g.Op));
                g.TranslationPlaneOrigin = mouse_ray(IntersectRayPlane(mouse_ray, g.TranslationPlane));
                g.MatrixOrigin = p;
                g.RelativeOrigin = (g.TranslationPlaneOrigin - p) / g.ScreenFactor;
            }
            if (HasAnyOp(g.Op, Rotate)) {
                const auto translation_plane = g.Op == RotateScreen ?
                    -vec4{camera_ray.d, 0} :
                    mode == Local ? m_[AxisIndex(g.Op, Rotate)] :
                                    vec4{DirUnary[AxisIndex(g.Op, Rotate)], 0};
                g.TranslationPlane = BuildPlane(Pos(g.Op == RotateScreen || mode == Local ? m_ : m), translation_plane);
                g.RotationVectorSource = glm::normalize(mouse_ray(IntersectRayPlane(mouse_ray, g.TranslationPlane)) - p);
                g.RotationAngle = ComputeAngleOnPlane(m_, mouse_ray);
            }
        }
    }

    Render(m_, m_inv, op, g.Op, view_proj, view_inv, camera_ray);
    return g.Using || commit;
}
} // namespace ModelGizmo
