#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "ModelGizmo.h"

#include "numeric/mat3.h"
#include "numeric/vec4.h"

#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include <format>
#include <optional>
#include <unordered_map>
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
    mat4 View, Proj, ViewProj, Model;
    mat4 ModelLocal; // orthonormalized model
    mat4 ModelInverse;
    mat4 ModelSource;
    mat4 MVP;
    mat4 MVPLocal; // Full MVP model, whereas MVP might only be translation in case of World space edition

    vec4 ModelScaleOrigin;
    vec3 CameraEye, CameraDir;
    vec4 RayOrigin, RayDir;

    float RadiusSquareCenter;
    ImVec2 ScreenSquareCenter, ScreenSquareMin, ScreenSquareMax;

    float ScreenFactor;
    vec4 RelativeOrigin;

    bool Using{false};

    vec4 TranslationPlane;
    vec3 TranslationPlaneOrigin;
    vec4 MatrixOrigin;

    vec4 RotationVectorSource;
    float RotationAngle, RotationAngleOrigin;

    vec4 Scale;
    vec3 ScaleOrigin;
    float SaveMousePosX;

    // save axis factor when using gizmo
    bool BelowAxisLimit[3];
    bool BelowPlaneLimit[3];
    float AxisFactor[3];

    vec2 Pos{0, 0}, Size{0, 0};
    bool IsOrthographic{false};

    Op HoverOp{NoOp}, CurrentOp{NoOp};
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
state::Style Style;
state::Color Color;
} // namespace

namespace ModelGizmo {
Op HoverOp() { return g.HoverOp; }
Op UsingOp() { return g.Using ? g.CurrentOp : NoOp; }

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
constexpr ImVec2 WorldToPos(vec3 pos_world, const mat4 &m) {
    auto trans = vec2{m * vec4{pos_world, 1}} * (0.5f / glm::dot(glm::transpose(m)[3], vec4{pos_world, 1})) + 0.5f;
    trans.y = 1 - trans.y;
    trans = g.Pos + trans * g.Size;
    return {trans.x, trans.y};
}

constexpr float LengthClipSpace(vec3 v, bool local_coords = false) {
    const auto &mvp = local_coords ? g.MVPLocal : g.MVP;
    auto start = mvp * vec4{0, 0, 0, 1};
    if (fabsf(start.w) > FLT_EPSILON) start /= start.w;

    auto end = mvp * vec4{v, 1};
    if (fabsf(end.w) > FLT_EPSILON) end /= end.w;

    return glm::length(end - start);
}

vec4 Right(const mat4 &m) { return {m[0]}; }
vec4 Up(const mat4 &m) { return {m[1]}; }
vec4 Dir(const mat4 &m) { return {m[2]}; }
vec4 Pos(const mat4 &m) { return {m[3]}; }
void SetPos(mat4 &m, const vec4 &pos) { m[3] = pos; }

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
ImVec2 ToImVec(vec2 v) { return {v.x, v.y}; }

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

const mat3 DirUnary{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

void ComputeTripodAxis(uint32_t axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool local_coords = false) {
    if (g.Using) {
        // Use stored factors so the gizmo doesn't flip when translating.
        dir_axis *= g.AxisFactor[axis_i];
        dir_plane_x *= g.AxisFactor[(axis_i + 1) % 3];
        dir_plane_y *= g.AxisFactor[(axis_i + 2) % 3];
        return;
    }

    dir_axis = {DirUnary[axis_i], 0};
    dir_plane_x = {DirUnary[(axis_i + 1) % 3], 0};
    dir_plane_y = {DirUnary[(axis_i + 2) % 3], 0};

    const float len_dir = LengthClipSpace(dir_axis, local_coords);
    const float len_dir_minus = LengthClipSpace(-dir_axis, local_coords);
    const float len_dir_plane_x = LengthClipSpace(dir_plane_x, local_coords);
    const float len_dir_plane_x_minus = LengthClipSpace(-dir_plane_x, local_coords);
    const float len_dir_plane_y = LengthClipSpace(dir_plane_y, local_coords);
    const float len_dir_plane_y_minus = LengthClipSpace(-dir_plane_y, local_coords);
    // Flip gizmo axis for better visibility.
    const float mul_axis = len_dir < len_dir_minus && fabsf(len_dir - len_dir_minus) > FLT_EPSILON ? -1 : 1;
    const float mul_axis_x = len_dir_plane_x < len_dir_plane_x_minus && fabsf(len_dir_plane_x - len_dir_plane_x_minus) > FLT_EPSILON ? -1 : 1;
    const float mul_axis_y = len_dir_plane_y < len_dir_plane_y_minus && fabsf(len_dir_plane_y - len_dir_plane_y_minus) > FLT_EPSILON ? -1 : 1;

    dir_axis *= mul_axis;
    dir_plane_x *= mul_axis_x;
    dir_plane_y *= mul_axis_y;
    // Cache
    g.AxisFactor[axis_i] = mul_axis;
    g.AxisFactor[(axis_i + 1) % 3] = mul_axis_x;
    g.AxisFactor[(axis_i + 2) % 3] = mul_axis_y;
}

void ComputeTripodAxisAndVisibility(uint32_t axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool &below_axis_limit, bool &below_plane_limit, bool local_coords = false) {
    ComputeTripodAxis(axis_i, dir_axis, dir_plane_x, dir_plane_y, local_coords);
    if (g.Using) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        below_axis_limit = g.BelowAxisLimit[axis_i];
        below_plane_limit = g.BelowPlaneLimit[axis_i];
    } else {
        static constexpr float AxisLimit{0.02};
        below_axis_limit = LengthClipSpace(dir_axis * g.ScreenFactor, local_coords) > AxisLimit;
        g.BelowAxisLimit[axis_i] = below_axis_limit; // Cache

        static constexpr auto ToNDC = [](vec4 v) {
            v = g.MVP * vec4{vec3{v}, 1};
            if (fabsf(v.w) > FLT_EPSILON) v /= v.w;
            return vec2{v};
        };
        // Parallelogram area
        static constexpr float ParallelogramAreaLimit{0.0025};
        const auto o = ToNDC(vec4{0});
        const auto pa = ToNDC(dir_plane_x * g.ScreenFactor) - o;
        const auto pb = ToNDC(dir_plane_y * g.ScreenFactor) - o;
        below_plane_limit = fabsf(pa.x * pb.y - pa.y * pb.x) > ParallelogramAreaLimit; // abs cross product
        g.BelowPlaneLimit[axis_i] = below_plane_limit; // Cache
    }
}

constexpr float IntersectRayPlane(vec3 origin, vec3 dir, vec4 plan) {
    const float num = glm::dot(vec3{plan}, origin) - plan.w;
    const float den = glm::dot(vec3{plan}, dir);
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -num / den;
}

constexpr float ComputeAngleOnPlan() {
    const vec4 perp{glm::normalize(vec4{glm::cross(vec3{g.RotationVectorSource}, vec3{g.TranslationPlane}), 0})};
    const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
    const auto pos_local = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
    const float acos_angle = glm::clamp(glm::dot(pos_local, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (glm::dot(pos_local, perp) < 0 ? 1 : -1);
}

void DrawHatchedAxis(vec3 axis) {
    if (Style.HatchedAxisLineWidth <= 0) return;

    for (uint32_t i = 1; i < 10; i++) {
        const auto base = WorldToPos(axis * 0.05f * float(i * 2) * g.ScreenFactor, g.MVP);
        const auto end = WorldToPos(axis * 0.05f * float(i * 2 + 1) * g.ScreenFactor, g.MVP);
        ImGui::GetWindowDrawList()->AddLine(base, end, Color.HatchedAxisLines, Style.HatchedAxisLineWidth);
    }
}

constexpr vec4 BuildPlane(const vec4 &p, const vec4 &p_normal) {
    const auto normal = glm::normalize(p_normal);
    return {vec3{normal}, glm::dot(normal, p)};
}

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 vert_p1, ImVec2 vert_p2) {
    const auto vec = ToGlm(vert_p2 - vert_p1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, ToGlm(p - vert_p1));
    if (t < 0) return vert_p1;
    if (t > glm::length(vec)) return vert_p2;
    return vert_p1 + ImVec2{v.x, v.y} * t;
}

constexpr float SelectDistSq = 12 * 12;

constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

Op GetTranslateOp(Op op) {
    if (g.Using || !HasAnyOp(op, Translate)) return NoOp;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
        mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y) {
        return TranslateScreen;
    }

    const auto pos = ToImVec(g.Pos), pos_screen{mouse_pos - pos};
    for (uint32_t i = 0; i < 3; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);

        dir_axis = g.Model * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.Model * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.Model * vec4{vec3{dir_plane_y}, 0};

        const auto axis_start_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor * 0.1f, g.ViewProj) - pos;
        const auto axis_end_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor, g.ViewProj) - pos;
        const auto closest_on_axis = PointOnSegment(pos_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_screen) < SelectDistSq) return Translate | AxisOp(i);

        const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlane(Pos(g.Model), dir_axis));
        const auto pos_plane = g.RayOrigin + g.RayDir * len;
        const float dx = glm::dot(vec3{dir_plane_x}, vec3{pos_plane - Pos(g.Model)} / g.ScreenFactor);
        const float dy = glm::dot(vec3{dir_plane_y}, vec3{pos_plane - Pos(g.Model)} / g.ScreenFactor);
        if (below_plane_limit && dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3]) {
            return TranslatePlanes[i];
        }
    }
    return NoOp;
}

// Scale a bit so translate axes don't touch when in universal.
constexpr float RotationDisplayScale{1.2};

Op GetRotateOp(Op op) {
    if (!HasAnyOp(op, Rotate)) return NoOp;

    static constexpr float SelectDist = 8;
    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (HasAnyOp(op, RotateScreen)) {
        const auto dist_sq = ImLengthSqr(mouse_pos - g.ScreenSquareCenter);
        const auto inner_rad = g.RadiusSquareCenter - SelectDist / 2, outer_rad = g.RadiusSquareCenter + SelectDist / 2;
        if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return RotateScreen;
    }

    const auto mv_pos = g.View * vec4{vec3{Pos(g.Model)}, 1};
    for (uint32_t i = 0; i < 3; ++i) {
        const auto pickup_plane = BuildPlane(Pos(g.Model), g.Model[i]);
        const auto len = IntersectRayPlane(g.RayOrigin, g.RayDir, pickup_plane);
        const auto intersect_world_pos = g.RayOrigin + g.RayDir * len;
        const auto intersect_view_pos = g.View * vec4{vec3{intersect_world_pos}, 1};
        if (ImAbs(mv_pos.z) - ImAbs(intersect_view_pos.z) < -FLT_EPSILON) continue;

        const auto circle_pos = g.ModelInverse * vec4{vec3{glm::normalize(intersect_world_pos - Pos(g.Model))}, 0};
        const auto circle_pos_screen = WorldToPos(circle_pos * RotationDisplayScale * g.ScreenFactor, g.MVP);
        if (ImLengthSqr(circle_pos_screen - mouse_pos) < SelectDist * SelectDist) return Rotate | AxisOp(i);
    }

    return NoOp;
}

Op GetScaleOp(Op op) {
    const bool universal = op == Universal;
    if (!HasAnyOp(op, Scale) && !universal) return NoOp;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (!universal && mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
        mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y) {
        return ScaleXYZ;
    }

    if (!universal) {
        for (uint32_t i = 0; i < 3; ++i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            ComputeTripodAxis(i, dir_axis, dir_plane_x, dir_plane_y, true);
            dir_axis = g.ModelLocal * vec4{vec3{dir_axis}, 0};
            dir_plane_x = g.ModelLocal * vec4{vec3{dir_plane_x}, 0};
            dir_plane_y = g.ModelLocal * vec4{vec3{dir_plane_y}, 0};

            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlane(Pos(g.ModelLocal), dir_axis));
            const bool is_axis = op == (Translate | AxisOp(i));
            const auto pos_plane = g.RayOrigin + g.RayDir * len;
            const auto pos_plan_screen = WorldToPos(pos_plane, g.ViewProj);
            const auto axis_start_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * (is_axis ? 1.0f : 0.1f), g.ViewProj);
            const auto axis_end_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * (is_axis ? 1.4f : 1.0f), g.ViewProj);
            const auto closest_on_axis = PointOnSegment(pos_plan_screen, axis_start_screen, axis_end_screen);
            if (ImLengthSqr(closest_on_axis - pos_plan_screen) < SelectDistSq) return Scale | AxisOp(i);
        }
        return NoOp;
    }

    // Universal
    if (float dist_sq = ImLengthSqr(mouse_pos - g.ScreenSquareCenter);
        dist_sq >= 17 * 17 && dist_sq < 23 * 23) {
        return ScaleXYZ;
    }

    for (uint32_t i = 0; i < 3; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
        if (below_axis_limit) {
            const float marker_scale = op == (Translate | AxisOp(i)) ? 1.4f : 1.0f;
            const auto end = WorldToPos(dir_axis * marker_scale * g.ScreenFactor, g.MVPLocal);
            if (ImLengthSqr(end - mouse_pos) < SelectDistSq) return Scale | AxisOp(i);
        }
    }
    return NoOp;
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
constexpr vec4 Snap(vec4 v, vec3 snap) { return {Snap(v[0], snap[0]), Snap(v[1], snap[1]), Snap(v[2], snap[2]), v[3]}; }

Op HandleTranslation(mat4 &m, Op op, bool local, std::optional<vec3> snap = std::nullopt) {
    if (!HasAnyOp(op, Translate)) return NoOp;

    if (g.Using && HasAnyOp(g.CurrentOp, Translate)) {
        const float len_signed = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
        const float len = fabsf(len_signed); // near plan
        const auto new_pos = g.RayOrigin + g.RayDir * len;
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.Model);

        // 1 axis constraint
        if (g.CurrentOp == (Translate | AxisX) || g.CurrentOp == (Translate | AxisY) || g.CurrentOp == (Translate | AxisZ)) {
            const auto axis_i = AxisIndex(g.CurrentOp, Translate);
            delta = g.Model[axis_i] * glm::dot(g.Model[axis_i], delta);
        }

        if (snap) {
            auto delta_cumulative = Pos(g.Model) + delta - g.MatrixOrigin;
            if (local || g.CurrentOp == TranslateScreen) {
                auto model_source = g.ModelSource;
                model_source[0] = glm::normalize(model_source[0]);
                model_source[1] = glm::normalize(model_source[1]);
                model_source[2] = glm::normalize(model_source[2]);
                delta_cumulative = glm::inverse(model_source) * vec4{vec3{delta_cumulative}, 0};
                delta_cumulative = Snap(delta_cumulative, *snap);
                delta_cumulative = model_source * vec4{vec3{delta_cumulative}, 0};
            } else {
                delta_cumulative = Snap(delta_cumulative, *snap);
            }
            delta = g.MatrixOrigin + delta_cumulative - Pos(g.Model);
        }

        m = glm::translate(mat4{1}, vec3{delta}) * g.ModelSource;
        if (!ImGui::GetIO().MouseDown[0]) g.Using = false;
        return g.CurrentOp;
    }

    // Find new translate op
    const auto type = GetTranslateOp(op);
    if (type == NoOp) return NoOp;

    if (ImGui::IsMouseClicked(0)) {
        g.Using = true;
        g.CurrentOp = type;
        static constexpr auto GetTranslationPlane = [](Op op) {
            if (op == TranslateScreen) return -vec4{g.CameraDir, 0};
            if (auto plane_index = TranslatePlaneIndex(op)) return g.Model[*plane_index];

            const auto plane = g.Model[AxisIndex(op, Translate)];
            const auto cam_to_model = glm::normalize(vec3{Pos(g.Model)} - g.CameraEye);
            return glm::normalize(vec4{glm::cross(vec3{plane}, glm::cross(vec3{plane}, cam_to_model)), 0});
        };

        g.TranslationPlane = BuildPlane(Pos(g.Model), GetTranslationPlane(type));
        g.TranslationPlaneOrigin = g.RayOrigin + g.RayDir * IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
        g.MatrixOrigin = Pos(g.Model);
        g.RelativeOrigin = (vec4{g.TranslationPlaneOrigin, 1} - Pos(g.Model)) / g.ScreenFactor;
    }
    return type;
}

Op HandleScale(mat4 &m, Op op, std::optional<vec3> snap = std::nullopt) {
    if (!HasAnyOp(op, Scale)) return NoOp;

    if (!g.Using) {
        // Find new scale op
        const auto type = GetScaleOp(op);
        if (type == NoOp) return NoOp;

        if (ImGui::IsMouseClicked(0)) {
            g.Using = true;
            g.CurrentOp = type;
            const auto translation_plane = type == ScaleXYZ ?
                -vec4{g.CameraDir, 0} :
                g.Model[(AxisIndex(type, Scale) + 1) % 3];
            g.TranslationPlane = BuildPlane(Pos(g.Model), translation_plane);
            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
            g.TranslationPlaneOrigin = g.RayOrigin + g.RayDir * len;
            g.Scale = {1, 1, 1, 0};
            g.MatrixOrigin = Pos(g.Model);
            g.RelativeOrigin = (vec4{g.TranslationPlaneOrigin, 1} - Pos(g.Model)) / g.ScreenFactor;
            g.ScaleOrigin = {glm::length(Right(g.ModelSource)), glm::length(Up(g.ModelSource)), glm::length(Dir(g.ModelSource))};
            g.SaveMousePosX = ImGui::GetIO().MousePos.x;
        }
        return type;
    }
    if (!g.Using || !HasAnyOp(g.CurrentOp, Scale)) return NoOp;

    const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
    const auto new_pos = g.RayOrigin + g.RayDir * len;
    const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
    auto delta = new_origin - Pos(g.ModelLocal);
    // 1 axis constraint
    if (g.CurrentOp != ScaleXYZ) {
        const auto axis_i = AxisIndex(g.CurrentOp, Scale);
        const auto &axis_value = g.ModelLocal[axis_i];
        const float length_on_axis = glm::dot(axis_value, delta);
        delta = axis_value * length_on_axis;

        vec4 base = vec4{g.TranslationPlaneOrigin, 0} - Pos(g.ModelLocal);
        const float ratio = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
        g.Scale[axis_i] = std::max(ratio, 0.001f);
    } else {
        const float scale_delta = (ImGui::GetIO().MousePos.x - g.SaveMousePosX) * 0.01f;
        g.Scale = vec4{std::max(1.f + scale_delta, 0.001f)};
    }

    if (snap) g.Scale = Snap(g.Scale, *snap);

    // no 0 allowed
    for (uint32_t i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

    m = g.ModelLocal * glm::scale(mat4{1}, {g.Scale * vec4{g.ScaleOrigin, 0}});

    if (!ImGui::GetIO().MouseDown[0]) {
        g.Using = false;
        g.Scale = vec4{1, 1, 1, 0};
    }
    return g.CurrentOp;
}

Op HandleRotation(mat4 &m, Op op, bool local, std::optional<vec3> snap = std::nullopt) {
    if (!HasAnyOp(op, Rotate)) return NoOp;

    if (!g.Using) {
        const auto type = GetRotateOp(op);
        if (type == NoOp) return NoOp;

        if (ImGui::IsMouseClicked(0)) {
            g.Using = true;
            g.CurrentOp = type;
            if (local || type == RotateScreen) {
                const auto translation_plane = type == RotateScreen ? -vec4{g.CameraDir, 0} : g.Model[AxisIndex(type, Rotate)];
                g.TranslationPlane = BuildPlane(Pos(g.Model), translation_plane);
            } else {
                g.TranslationPlane = BuildPlane(Pos(g.ModelSource), {DirUnary[AxisIndex(type, Rotate)], 0});
            }
            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
            g.RotationVectorSource = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
            g.RotationAngleOrigin = ComputeAngleOnPlan();
        }
        return type;
    }
    if (!g.Using || !HasAnyOp(g.CurrentOp, Rotate)) return NoOp;

    g.RotationAngle = ComputeAngleOnPlan();
    if (snap) g.RotationAngle = Snap(g.RotationAngle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{g.ModelInverse} * g.TranslationPlane); // Assumes affine model
    const mat4 delta_rot{glm::rotate(mat4{1}, g.RotationAngle - g.RotationAngleOrigin, rot_axis_local)};
    if (g.RotationAngle != g.RotationAngleOrigin) g.RotationAngleOrigin = g.RotationAngle;

    if (local) {
        m = g.ModelLocal * delta_rot * glm::scale(mat4{1}, vec3{g.ModelScaleOrigin});
    } else {
        auto res = g.ModelSource;
        SetPos(res, vec4{0});
        m = delta_rot * res;
        SetPos(m, Pos(g.ModelSource));
    }

    if (!ImGui::GetIO().MouseDown[0]) g.Using = false;
    return g.CurrentOp;
}

namespace Format {
static constexpr char AxisLabels[] = "XYZ";
std::string Axis(uint32_t axis_i, float v) { return axis_i >= 0 && axis_i < 3 ? std::format("{}: {:.3f}", AxisLabels[axis_i], v) : ""; }
std::string Axis(uint32_t axis_i, vec3 v) { return Axis(axis_i, v[axis_i]); }
std::string Translation(Op op, vec3 v) {
    if (op == TranslateScreen) return std::format("{} {} {}", Axis(0, v.x), Axis(1, v.y), Axis(2, v.z));
    if (op == TranslateYZ) return std::format("{} {}", Axis(1, v.y), Axis(2, v.z));
    if (op == TranslateZX) return std::format("{} {}", Axis(2, v.z), Axis(0, v.x));
    if (op == TranslateXY) return std::format("{} {}", Axis(0, v.x), Axis(1, v.y));
    return Axis(AxisIndex(op, Op::Translate), v);
}
std::string Scale(Op op, vec3 v) { return op == ScaleXYZ ? std::format("XYZ: {:.3f}", v.x) : Axis(AxisIndex(op, Op::Scale), v); }
std::string Rotation(Op op, float rad) {
    if (!HasAnyOp(op, Rotate)) return "";
    const auto deg_rad = std::format("{:.3f} deg {:.3f} rad", rad * 180 / M_PI, rad);
    if (op == RotateScreen) return std::format("Screen: {}", deg_rad);

    const auto axis_i = AxisIndex(op, Rotate);
    return axis_i >= 0 && axis_i < 3 ? std::format("{}: {}", AxisLabels[axis_i], deg_rad) : "";
}
} // namespace Format
} // namespace

namespace ModelGizmo {
bool Draw(vec2 pos, vec2 size, const mat4 &view, const mat4 &proj, Op op, Mode mode, mat4 &m, std::optional<vec3> snap) {
    g.Pos = pos;
    g.Size = size;
    g.View = view;
    g.Proj = proj;

    auto &model_local = g.ModelLocal;
    model_local[0] = glm::normalize(m[0]);
    model_local[1] = glm::normalize(m[1]);
    model_local[2] = glm::normalize(m[2]);
    model_local[3] = m[3];

    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (HasAnyOp(op, Scale)) mode = Local;
    g.Model = mode == Local ? g.ModelLocal : glm::translate(mat4{1}, vec3{Pos(m)});
    g.ModelSource = m;
    g.ModelScaleOrigin = vec4{glm::length(Right(g.ModelSource)), glm::length(Up(g.ModelSource)), glm::length(Dir(g.ModelSource)), 0};

    g.ModelInverse = glm::inverse(g.Model);
    g.ViewProj = g.Proj * g.View;
    g.MVP = g.ViewProj * g.Model;
    g.MVPLocal = g.ViewProj * g.ModelLocal;

    const mat4 view_inv{glm::inverse(g.View)};
    g.CameraDir = vec3{Dir(view_inv)};
    g.CameraEye = vec3{Pos(view_inv)};

    // Compute scale from camera right vector projected onto screen at m pos
    static constexpr float GizmoSizeClipSpace{0.1};
    g.ScreenFactor = GizmoSizeClipSpace / LengthClipSpace(g.ModelInverse * vec4{vec3{Right(view_inv)}, 0});

    g.ScreenSquareCenter = WorldToPos(vec3{0}, g.MVP);
    g.ScreenSquareMin = g.ScreenSquareCenter - ImVec2{10, 10};
    g.ScreenSquareMax = g.ScreenSquareCenter + ImVec2{10, 10};

    // Compute camera ray
    const auto view_proj_inv = glm::inverse(g.Proj * g.View);
    const auto mouse_delta = ImGui::GetIO().MousePos - ToImVec(g.Pos);
    const float mox = (mouse_delta.x / g.Size.x) * 2 - 1;
    const float moy = (1 - mouse_delta.y / g.Size.y) * 2 - 1;
    const vec4 near_pos{g.Proj * vec4{0, 0, 1, 1}};
    const vec4 far_pos{g.Proj * vec4{0, 0, 2, 1}};
    const bool reversed = near_pos.z / near_pos.w > far_pos.z / far_pos.w;
    const float z_near = reversed ? 1 - FLT_EPSILON : 0;
    const float z_far = reversed ? 0 : 1 - FLT_EPSILON;
    g.RayOrigin = view_proj_inv * vec4{mox, moy, z_near, 1};
    g.RayOrigin /= g.RayOrigin.w;
    vec4 ray_end{view_proj_inv * vec4{mox, moy, z_far, 1}};
    ray_end /= ray_end.w;
    g.RayDir = glm::normalize(ray_end - g.RayOrigin);

    // behind camera
    const auto pos_cam_space = g.MVP * vec4{vec3{0}, 1};
    if (!g.IsOrthographic && pos_cam_space.z < 0.001 && !g.Using) return false;

    const bool window_hovered = ImGui::IsWindowHovered();
    // Order is important because of universal selection.
    auto type{NoOp};
    if (window_hovered) type = HandleScale(m, op, snap);
    // HandleTranslation has side effects, so call it even if not hovered.
    if (type == NoOp) type = HandleTranslation(m, op, mode == Local, snap);
    if (type == NoOp && window_hovered) type = HandleRotation(m, op, mode == Local, snap);
    g.HoverOp = type;

    // Draw
    auto *dl = ImGui::GetWindowDrawList();

    const bool universal = op == Universal;
    if (HasAnyOp(op, Translate)) {
        const auto origin = WorldToPos(Pos(g.Model), g.ViewProj);
        for (uint32_t i = 0; i < 3; ++i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit = false, below_plane_limit = false;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
            const bool using_type = type == (Translate | AxisOp(i));
            if ((!g.Using || using_type) && below_axis_limit) {
                // draw axis
                const auto base = WorldToPos(dir_axis * g.ScreenFactor * 0.1f, g.MVP);
                const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                const auto color = using_type ? Color.Selection : Color.Directions[i];
                dl->AddLine(base, end, color, Style.LineWidth);
                if (!g.Using && !universal) { // In universal mode, draw scale circles instead of translate arrows.
                    const auto dir = (origin - end) * Style.LineArrowSize / sqrtf(ImLengthSqr(origin - end));
                    const ImVec2 orth_dir{dir.y, -dir.x};
                    dl->AddTriangleFilled(end - dir, end + dir + orth_dir, end + dir - orth_dir, color);
                }
                if (g.AxisFactor[i] < 0) DrawHatchedAxis(dir_axis);
            }
            if ((!g.Using || type == TranslatePlanes[i]) && below_plane_limit) {
                // draw plane
                ImVec2 quad_pts_screen[4];
                for (uint32_t j = 0; j < 4; ++j) {
                    const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                    quad_pts_screen[j] = WorldToPos(corner_pos_world, g.MVP);
                }
                dl->AddPolyline(quad_pts_screen, 4, Color.Directions[i], true, 1.0f);
                const auto color = type == TranslateScreen || type == TranslatePlanes[i] ? Color.Selection : Color.Planes[i];
                dl->AddConvexPolyFilled(quad_pts_screen, 4, color);
            }
        }
        if (!g.Using || type == TranslateScreen) {
            const auto color = type == TranslateScreen ? Color.Selection : IM_COL32_WHITE;
            dl->AddCircleFilled(g.ScreenSquareCenter, Style.CircleRad, color, 32);
        }
        if (g.Using && HasAnyOp(type, Translate)) {
            const auto translation_line_color = Color.TranslationLine;
            const auto source_pos_screen = WorldToPos(g.MatrixOrigin, g.ViewProj);
            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            const auto dif = ToImVec(glm::normalize(vec4{dest_pos.x - source_pos_screen.x, dest_pos.y - source_pos_screen.y, 0, 0}) * 5.f);
            dl->AddCircle(source_pos_screen, 6.f, translation_line_color);
            dl->AddCircle(dest_pos, 6.f, translation_line_color);
            dl->AddLine(source_pos_screen + dif, dest_pos - dif, translation_line_color, 2.f);

            const auto delta_info = Pos(g.Model) - g.MatrixOrigin;
            const auto formatted = Format::Translation(type, delta_info);
            dl->AddText(dest_pos + ImVec2{15, 15}, Color.TextShadow, formatted.data());
            dl->AddText(dest_pos + ImVec2{14, 14}, Color.Text, formatted.data());
        }
    }
    if (HasAnyOp(op, Rotate)) {
        static constexpr uint32_t HalfCircleSegmentCount{64};
        static constexpr float ScreenRotateSize{0.06};
        g.RadiusSquareCenter = ScreenRotateSize * g.Size.y;

        // Assumes affine model
        const auto cam_to_model = mat3{g.ModelInverse} * vec3{g.IsOrthographic ? -Dir(glm::inverse(g.View)) : glm::normalize(vec3{Pos(g.Model)} - g.CameraEye)};
        static ImVec2 CirclePositions[HalfCircleSegmentCount * 2 + 1];
        for (uint32_t axis = 0; axis < 3; axis++) {
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
                dl->AddPolyline(CirclePositions, point_count, color, false, Style.RotationLineWidth);
            }
            if (float radius_axis_sq = ImLengthSqr(WorldToPos(Pos(g.Model), g.ViewProj) - CirclePositions[0]);
                radius_axis_sq > g.RadiusSquareCenter * g.RadiusSquareCenter) {
                g.RadiusSquareCenter = sqrtf(radius_axis_sq);
            }
        }
        if (!g.Using || type == RotateScreen) {
            const auto color = type == RotateScreen ? Color.Selection : IM_COL32_WHITE;
            dl->AddCircle(WorldToPos(Pos(g.Model), g.ViewProj), g.RadiusSquareCenter, color, 64, Style.RotationLineWidth * 1.5f);
        }
        if (g.Using && HasAnyOp(type, Rotate)) {
            static ImVec2 CirclePositions[HalfCircleSegmentCount + 1];
            CirclePositions[0] = WorldToPos(Pos(g.Model), g.ViewProj);
            for (uint32_t i = 1; i < HalfCircleSegmentCount + 1; ++i) {
                const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
                const mat4 rotate{glm::rotate(mat4{1}, ng, vec3{g.TranslationPlane})};
                const auto pos = rotate * vec4{vec3{g.RotationVectorSource}, 1} * g.ScreenFactor * RotationDisplayScale;
                CirclePositions[i] = WorldToPos(pos + Pos(g.Model), g.ViewProj);
            }
            dl->AddConvexPolyFilled(CirclePositions, HalfCircleSegmentCount + 1, Color.RotationFillActive);
            dl->AddPolyline(CirclePositions, HalfCircleSegmentCount + 1, Color.RotationBorderActive, true, Style.RotationLineWidth);

            const auto formatted = Format::Rotation(type, g.RotationAngle);
            const auto dest_pos = CirclePositions[1];
            dl->AddText(dest_pos + ImVec2{15, 15}, Color.TextShadow, formatted.data());
            dl->AddText(dest_pos + ImVec2{14, 14}, Color.Text, formatted.data());
        }
    }
    if (HasAnyOp(op, Scale)) {
        if (!g.Using) {
            for (uint32_t i = 0; i < 3; ++i) {
                vec4 dir_plane_x, dir_plane_y, dir_axis;
                bool below_axis_limit, below_plane_limit;
                ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
                if (below_axis_limit) {
                    const auto color = type == (Scale | AxisOp(i)) ? Color.Selection : Color.Directions[i];
                    if (!universal) {
                        const auto base = WorldToPos(dir_axis * g.ScreenFactor * 0.1f, g.MVP);
                        if (g.Using) {
                            const auto center = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                            dl->AddLine(base, center, Color.ScaleLine, Style.LineWidth);
                            dl->AddCircleFilled(center, Style.CircleRad, Color.ScaleLine);
                        }
                        const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                        dl->AddLine(base, end, color, Style.LineWidth);
                        dl->AddCircleFilled(end, Style.CircleRad, color);
                        if (g.AxisFactor[i] < 0) DrawHatchedAxis(dir_axis);
                    } else {
                        const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVPLocal);
                        dl->AddCircleFilled(end, Style.CircleRad, color);
                    }
                }
            }
        }
        if (!g.Using || HasAnyOp(type, Scale)) {
            const auto circle_color = g.Using || type == ScaleXYZ ? Color.Selection : IM_COL32_WHITE;
            if (!universal) dl->AddCircleFilled(g.ScreenSquareCenter, Style.CircleRad, circle_color, 32);
            else dl->AddCircle(g.ScreenSquareCenter, 20.f, circle_color, 32, Style.CircleRad);
            if (g.Using) {
                const auto formatted = Format::Scale(type, g.Scale);
                const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
                dl->AddText(dest_pos + ImVec2{15, 15}, Color.TextShadow, formatted.data());
                dl->AddText(dest_pos + ImVec2{14, 14}, Color.Text, formatted.data());
            }
        }
    }
    return type != NoOp;
}
} // namespace ModelGizmo
