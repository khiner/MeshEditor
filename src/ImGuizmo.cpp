#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "numeric/mat3.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include <vector>

namespace {
enum MoveType {
    MT_None = 0,
    MT_TranslateX,
    MT_TranslateY,
    MT_TranslateZ,
    MT_TranslateYZ,
    MT_TranslateZX,
    MT_TranslateXY,
    MT_TranslateScreen,
    MT_RotateX,
    MT_RotateY,
    MT_RotateZ,
    MT_RotateScreen,
    MT_ScaleX,
    MT_ScaleY,
    MT_ScaleZ,
    MT_ScaleXYZ
};

constexpr bool IsTranslate(MoveType type) { return type >= MT_TranslateX && type <= MT_TranslateScreen; }
constexpr bool IsRotate(MoveType type) { return type >= MT_RotateX && type <= MT_RotateScreen; }
constexpr bool IsScale(MoveType type) { return type >= MT_ScaleX && type <= MT_ScaleXYZ; }

using namespace ImGuizmo;
using enum Op;

constexpr auto OpVal = [](auto op) { return static_cast<std::underlying_type_t<Op>>(op); };
constexpr Op operator&(Op a, Op b) { return Op(OpVal(a) & OpVal(b)); }
constexpr Op operator|(Op a, Op b) { return Op(OpVal(a) | OpVal(b)); }
constexpr Op operator<<(Op op, unsigned int shift) { return Op(OpVal(op) << shift); }
constexpr bool HasAnyOp(Op a, Op b) { return (a & b) != Op::NoOp; }
constexpr bool HasAllOps(Op a, Op b) { return (a & b) == b; }

enum Color {
    DirectionX,
    DirectionY,
    DirectionZ,
    PlaneX,
    PlaneY,
    PlaneZ,
    Selection,
    Inactive,
    TranslationLine,
    ScaleLine,
    RotationBorderActive,
    RotationFillActive,
    HatchedAxisLines,
    Text,
    TextShadow,
    COUNT
};

struct Style {
    Style() {
        Colors[DirectionX] = {.666, 0, 0, 1};
        Colors[DirectionY] = {0, .666, 0, 1};
        Colors[DirectionZ] = {0, 0, .666, 1};
        Colors[PlaneX] = {.666, 0, 0, .38};
        Colors[PlaneY] = {0, .666, 0, .38};
        Colors[PlaneZ] = {0, 0, .666, .38};
        Colors[Selection] = {1, .5, .062, .541};
        Colors[Inactive] = {.6, .6, .6, .6};
        Colors[TranslationLine] = {.666, .666, .666, .666};
        Colors[ScaleLine] = {.25, .25, .25, 1};
        Colors[RotationBorderActive] = {1, .5, .062, 1};
        Colors[RotationFillActive] = {1, .5, .062, .5};
        Colors[HatchedAxisLines] = {0, 0, 0, .5};
        Colors[Text] = {1, 1, 1, 1};
        Colors[TextShadow] = {0, 0, 0, 1};
    }

    float TranslationLineThickness{3}; // Thickness of lines for translation gizmo
    float TranslationLineArrowSize{6}; // Size of arrow at the end of lines for translation gizmo
    float RotationLineThickness{2}; // Thickness of lines for rotation gizmo
    float RotationOuterLineThickness{3}; // Thickness of line surrounding the rotation gizmo
    float ScaleLineThickness{3}; // Thickness of lines for scale gizmo
    float ScaleLineCircleSize{6}; // Size of circle at the end of lines for scale gizmo
    float HatchedAxisLineThickness{6}; // Thickness of hatched axis lines
    float CenterCircleSize{6}; // Size of circle at the center of the translate/scale gizmo

    ImVec4 Colors[Color::COUNT];
};

struct Context {
    Style Style;

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
    bool Reversed{false}; // reversed proj m

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
    float AxisLimit{0.0025}, PlaneLimit{0.02};

    vec2 Pos{0, 0}, Size{0, 0};
    float GizmoSizeClipSpace{0.1};
    bool IsOrthographic{false};

    MoveType CurrentMoveType{MT_None};

    int ActualID{-1}, EditingID{-1};
};

Context g;

MoveType GetTranslateType(Op);
MoveType GetRotateType(Op);
MoveType GetScaleType(Op);
} // namespace

namespace ImGuizmo {
bool IsUsing() { return g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID); }

bool IsOver(Op op) {
    return IsUsing() ||
        (HasAnyOp(op, Translate) && GetTranslateType(op)) ||
        (HasAnyOp(op, Rotate) && GetRotateType(op)) ||
        (HasAnyOp(op, Scale) && GetScaleType(op));
}
} // namespace ImGuizmo

namespace {
constexpr ImVec2 WorldToPos(vec3 pos_world, const mat4 &m) {
    auto trans = vec2{m * vec4{pos_world, 1}} * (0.5f / glm::dot(glm::transpose(m)[3], vec4{pos_world, 1})) + 0.5f;
    trans.y = 1 - trans.y;
    trans = g.Pos + trans * g.Size;
    return {trans.x, trans.y};
}

constexpr float GetSegmentLengthClipSpace(vec3 end, bool local_coords = false) {
    static constexpr auto start = vec3{0};
    const auto &mvp = local_coords ? g.MVPLocal : g.MVP;
    auto segment_start = mvp * vec4{start, 1};
    // check for axis aligned with camera direction
    if (fabsf(segment_start.w) > FLT_EPSILON) segment_start /= segment_start.w;

    auto segment_end = mvp * vec4{end, 1};
    // check for axis aligned with camera direction
    if (fabsf(segment_end.w) > FLT_EPSILON) segment_end /= segment_end.w;

    auto axis_clip = segment_end - segment_start;
    const auto aspect_ratio = g.Size.x / g.Size.y;
    if (aspect_ratio < 1.0) axis_clip.x *= aspect_ratio;
    else axis_clip.y /= aspect_ratio;
    return sqrtf(axis_clip.x * axis_clip.x + axis_clip.y * axis_clip.y);
}

vec4 Right(const mat4 &m) { return {m[0]}; }
vec4 Up(const mat4 &m) { return {m[1]}; }
vec4 Dir(const mat4 &m) { return {m[2]}; }
vec4 Pos(const mat4 &m) { return {m[3]}; }
void SetPos(mat4 &m, const vec4 &pos) { m[3] = pos; }

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
ImVec2 ToImVec(vec2 v) { return {v.x, v.y}; }

constexpr ImU32 GetColorU32(int idx) {
    IM_ASSERT(idx < Color::COUNT);
    return ImGui::ColorConvertFloat4ToU32(g.Style.Colors[idx]);
}

constexpr void ComputeColors(ImU32 *colors, MoveType type, Op op) {
    const auto selection_color = GetColorU32(Selection);
    switch (op) {
        case Translate:
            colors[0] = (type == MT_TranslateScreen) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; ++i) {
                colors[i + 1] = type == MoveType(MT_TranslateX + i) ? selection_color : GetColorU32(DirectionX + i);
                colors[i + 4] = type == MoveType(MT_TranslateYZ + i) ? selection_color : GetColorU32(PlaneX + i);
                colors[i + 4] = type == MT_TranslateScreen ? selection_color : colors[i + 4];
            }
            break;
        case Rotate:
            colors[0] = (type == MT_RotateScreen) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; ++i) {
                colors[i + 1] = type == int(MT_RotateX + i) ? selection_color : GetColorU32(DirectionX + i);
            }
            break;
        case Scale:
        case ScaleU:
            colors[0] = type == MT_ScaleXYZ ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; ++i) {
                colors[i + 1] = type == int(MT_ScaleX + i) ? selection_color : GetColorU32(DirectionX + i);
            }
            break;
        // note: this internal function is only called with three possible values for op
        default:
            break;
    }
}

const mat3 DirUnary{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

void ComputeTripodAxis(int axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool local_coords = false) {
    if (IsUsing()) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        dir_axis *= g.AxisFactor[axis_i];
        dir_plane_x *= g.AxisFactor[(axis_i + 1) % 3];
        dir_plane_y *= g.AxisFactor[(axis_i + 2) % 3];
        return;
    }

    dir_axis = {DirUnary[axis_i], 0};
    dir_plane_x = {DirUnary[(axis_i + 1) % 3], 0};
    dir_plane_y = {DirUnary[(axis_i + 2) % 3], 0};

    const float len_dir = GetSegmentLengthClipSpace(dir_axis, local_coords);
    const float len_dir_minus = GetSegmentLengthClipSpace(-dir_axis, local_coords);
    const float len_dir_plane_x = GetSegmentLengthClipSpace(dir_plane_x, local_coords);
    const float len_dir_plane_x_minus = GetSegmentLengthClipSpace(-dir_plane_x, local_coords);
    const float len_dir_plane_y = GetSegmentLengthClipSpace(dir_plane_y, local_coords);
    const float len_dir_plane_y_minus = GetSegmentLengthClipSpace(-dir_plane_y, local_coords);

    // For readability, flip gizmo axis for better visibility
    // When false, they always stay along the positive world/local axis
    static constexpr bool AllowFlip = true;
    const float mul_axis = AllowFlip && len_dir < len_dir_minus && fabsf(len_dir - len_dir_minus) > FLT_EPSILON ? -1 : 1;
    const float mul_axis_x = AllowFlip && len_dir_plane_x < len_dir_plane_x_minus && fabsf(len_dir_plane_x - len_dir_plane_x_minus) > FLT_EPSILON ? -1 : 1;
    const float mul_axis_y = AllowFlip && len_dir_plane_y < len_dir_plane_y_minus && fabsf(len_dir_plane_y - len_dir_plane_y_minus) > FLT_EPSILON ? -1 : 1;
    dir_axis *= mul_axis;
    dir_plane_x *= mul_axis_x;
    dir_plane_y *= mul_axis_y;
    // Cache
    g.AxisFactor[axis_i] = mul_axis;
    g.AxisFactor[(axis_i + 1) % 3] = mul_axis_x;
    g.AxisFactor[(axis_i + 2) % 3] = mul_axis_y;
}

void ComputeTripodVisibility(int axis_i, float axis_length_clip, vec4 dir_plane_x, vec4 dir_plane_y, bool &below_axis_limit, bool &below_plane_limit) {
    if (IsUsing()) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        below_axis_limit = g.BelowAxisLimit[axis_i];
        below_plane_limit = g.BelowPlaneLimit[axis_i];
        return;
    }

    // Parallelogram area
    vec4 pa = dir_plane_x * g.ScreenFactor, pb = dir_plane_y * g.ScreenFactor;
    pa = g.MVP * vec4{vec3{pa}, 1};
    if (fabsf(pa.w) > FLT_EPSILON) pa /= pa.w;
    pb = g.MVP * vec4{vec3{pb}, 1};
    if (fabsf(pb.w) > FLT_EPSILON) pb /= pb.w;

    const auto aspect_ratio = g.Size.x / g.Size.y;
    pa.y /= aspect_ratio;
    pb.y /= aspect_ratio;

    const float para_surf = sqrtf(pa.x * pa.x + pa.y * pa.y) * fabsf(glm::dot(glm::normalize(vec3{-pa.y, pa.x, 0}), vec3{pb}));

    below_plane_limit = para_surf > g.AxisLimit;
    below_axis_limit = axis_length_clip > g.PlaneLimit;
    // Cache
    g.BelowAxisLimit[axis_i] = below_axis_limit;
    g.BelowPlaneLimit[axis_i] = below_plane_limit;
}

void ComputeTripodAxisAndVisibility(int axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool &below_axis_limit, bool &below_plane_limit, bool local_coords = false) {
    ComputeTripodAxis(axis_i, dir_axis, dir_plane_x, dir_plane_y, local_coords);
    const float axis_length_clip = GetSegmentLengthClipSpace(dir_axis * g.ScreenFactor, local_coords);
    ComputeTripodVisibility(axis_i, axis_length_clip, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
}

constexpr void ComputeSnap(float *value, float snap) {
    if (snap <= FLT_EPSILON) return;

    static constexpr float SnapTension{0.5};
    const float modulo = fmodf(*value, snap);
    const float modulo_ratio = fabsf(modulo) / snap;
    if (modulo_ratio < SnapTension) *value -= modulo;
    else if (modulo_ratio > 1 - SnapTension) *value = *value - modulo + snap * (*value < 0 ? -1 : 1);
}
constexpr void ComputeSnap(vec4 &value, const float *snap) {
    for (int i = 0; i < 3; ++i) ComputeSnap(&value[i], snap[i]);
}

constexpr float IntersectRayPlane(vec3 origin, vec3 dir, vec4 plan) {
    const float num = glm::dot(vec3{plan}, origin) - plan.w;
    const float den = glm::dot(vec3{plan}, dir);
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -num / den;
}

constexpr float ComputeAngleOnPlan() {
    vec4 perp{glm::normalize(vec4{glm::cross(vec3{g.RotationVectorSource}, vec3{g.TranslationPlane}), 0})};
    const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
    const auto pos_local = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
    const float acos_angle = glm::clamp(glm::dot(pos_local, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (glm::dot(pos_local, perp) < 0 ? 1.f : -1.f);
}

void DrawHatchedAxis(vec3 axis) {
    if (g.Style.HatchedAxisLineThickness <= 0) return;

    for (int i = 1; i < 10; i++) {
        const auto base = WorldToPos(axis * 0.05f * float(i * 2) * g.ScreenFactor, g.MVP);
        const auto end = WorldToPos(axis * 0.05f * float(i * 2 + 1) * g.ScreenFactor, g.MVP);
        ImGui::GetWindowDrawList()->AddLine(base, end, GetColorU32(HatchedAxisLines), g.Style.HatchedAxisLineThickness);
    }
}

constexpr vec4 BuildPlane(const vec4 &p_point1, const vec4 &p_normal) {
    const auto normal = glm::normalize(p_normal);
    return {vec3{normal}, glm::dot(normal, p_point1)};
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

MoveType GetTranslateType(Op op) {
    if (g.Using || !HasAnyOp(op, Translate)) return MT_None;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
        mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y) {
        return MT_TranslateScreen;
    }

    const auto pos = ToImVec(g.Pos), pos_screen{mouse_pos - pos};
    for (int i = 0; i < 3; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        dir_axis = g.Model * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.Model * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.Model * vec4{vec3{dir_plane_y}, 0};

        const auto axis_start_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor * 0.1f, g.ViewProj) - pos;
        const auto axis_end_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor, g.ViewProj) - pos;
        const auto closest_on_axis = PointOnSegment(pos_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_screen) < SelectDistSq) return MoveType(MT_TranslateX + i);

        const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlane(Pos(g.Model), dir_axis));
        const auto pos_plane = g.RayOrigin + g.RayDir * len;
        const float dx = glm::dot(vec3{dir_plane_x}, vec3{pos_plane - Pos(g.Model)} / g.ScreenFactor);
        const float dy = glm::dot(vec3{dir_plane_y}, vec3{pos_plane - Pos(g.Model)} / g.ScreenFactor);
        if (below_plane_limit && dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3]) {
            return MoveType(MT_TranslateYZ + i);
        }
    }
    return MT_None;
}

// Scale a bit so translate axes don't touch when in universal.
constexpr float RotationDisplayScale{1.2};

MoveType GetRotateType(Op op) {
    constexpr static float SelectDist = 8;
    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (HasAnyOp(op, RotateScreen)) {
        const auto dist_sq = ImLengthSqr(mouse_pos - g.ScreenSquareCenter);
        const auto inner_rad = g.RadiusSquareCenter - SelectDist / 2, outer_rad = g.RadiusSquareCenter + SelectDist / 2;
        if (dist_sq >= inner_rad * inner_rad && dist_sq < outer_rad * outer_rad) return MT_RotateScreen;
    }

    if (!HasAnyOp(op, Rotate)) return MT_None;

    const auto mv_pos = g.View * vec4{vec3{Pos(g.Model)}, 1};
    for (int i = 0; i < 3; ++i) {
        const auto pickup_plane = BuildPlane(Pos(g.Model), g.Model[i]);
        const auto len = IntersectRayPlane(g.RayOrigin, g.RayDir, pickup_plane);
        const auto intersect_world_pos = g.RayOrigin + g.RayDir * len;
        const auto intersect_view_pos = g.View * vec4{vec3{intersect_world_pos}, 1};
        if (ImAbs(mv_pos.z) - ImAbs(intersect_view_pos.z) < -FLT_EPSILON) continue;

        const auto circle_pos = g.ModelInverse * vec4{vec3{glm::normalize(intersect_world_pos - Pos(g.Model))}, 0};
        const auto circle_pos_screen = WorldToPos(circle_pos * RotationDisplayScale * g.ScreenFactor, g.MVP);
        if (ImLengthSqr(circle_pos_screen - mouse_pos) < SelectDist * SelectDist) return MoveType(MT_RotateX + i);
    }

    return MT_None;
}

MoveType GetScaleType(Op op) {
    if (g.Using && !HasAnyOp(op, Scale) && !HasAnyOp(op, ScaleU)) return MT_None;

    if (HasAnyOp(op, Scale)) {
        for (int i = 0; i < 3; ++i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            ComputeTripodAxis(i, dir_axis, dir_plane_x, dir_plane_y, true);
            dir_axis = g.ModelLocal * vec4{vec3{dir_axis}, 0};
            dir_plane_x = g.ModelLocal * vec4{vec3{dir_plane_x}, 0};
            dir_plane_y = g.ModelLocal * vec4{vec3{dir_plane_y}, 0};

            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlane(Pos(g.ModelLocal), dir_axis));
            const bool is_axis = HasAllOps(op, Translate | (AxisX << i));
            const auto pos_plane = g.RayOrigin + g.RayDir * len;
            const auto pos_plan_screen = WorldToPos(pos_plane, g.ViewProj);
            const auto axis_start_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * (is_axis ? 1.0f : 0.1f), g.ViewProj);
            const auto axis_end_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * (is_axis ? 1.4f : 1.0f), g.ViewProj);
            const auto closest_on_axis = PointOnSegment(pos_plan_screen, axis_start_screen, axis_end_screen);
            if (ImLengthSqr(closest_on_axis - pos_plan_screen) < SelectDistSq) return MoveType(MT_ScaleX + i);
        }
        return MT_None;
    }
    if (!HasAnyOp(op, ScaleU)) return MT_None;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    if (float dist_sq = ImLengthSqr(mouse_pos - g.ScreenSquareCenter);
        dist_sq >= 17 * 17 && dist_sq < 23 * 23) return MT_ScaleXYZ;

    // universal
    for (int i = 0; i < 3; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
        if (below_axis_limit) {
            const float marker_scale = HasAllOps(op, Translate | (AxisX << i)) ? 1.4f : 1.0f;
            const auto end = WorldToPos(dir_axis * marker_scale * g.ScreenFactor, g.MVPLocal);
            if (ImLengthSqr(end - mouse_pos) < SelectDistSq) return MoveType(MT_ScaleX + i);
        }
    }
    return MT_None;
}

bool CanActivate() { return ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive(); }

MoveType HandleTranslation(mat4 &m, Op op, const float *snap, bool local) {
    if (!HasAnyOp(op, Translate)) return MT_None;

    if (IsUsing() && IsTranslate(g.CurrentMoveType)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len_signed = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
        const float len = fabsf(len_signed); // near plan
        const auto new_pos = g.RayOrigin + g.RayDir * len;
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.Model);

        // 1 axis constraint
        if (g.CurrentMoveType >= MT_TranslateX && g.CurrentMoveType <= MT_TranslateZ) {
            const int axis_i = g.CurrentMoveType - MT_TranslateX;
            delta = g.Model[axis_i] * glm::dot(g.Model[axis_i], delta);
        }

        if (snap) {
            auto delta_cumulative = Pos(g.Model) + delta - g.MatrixOrigin;
            if (local || g.CurrentMoveType == MT_TranslateScreen) {
                auto model_source = g.ModelSource;
                model_source[0] = glm::normalize(model_source[0]);
                model_source[1] = glm::normalize(model_source[1]);
                model_source[2] = glm::normalize(model_source[2]);
                delta_cumulative = glm::inverse(model_source) * vec4{vec3{delta_cumulative}, 0};
                ComputeSnap(delta_cumulative, snap);
                delta_cumulative = model_source * vec4{vec3{delta_cumulative}, 0};
            } else {
                ComputeSnap(delta_cumulative, snap);
            }
            delta = g.MatrixOrigin + delta_cumulative - Pos(g.Model);
        }

        m = glm::translate(mat4{1}, vec3{delta}) * g.ModelSource;
        if (!ImGui::GetIO().MouseDown[0]) g.Using = false;
        return g.CurrentMoveType;
    }
    // find new possible way to move
    const auto type = GetTranslateType(op);
    if (type == MT_None) return MT_None;

    ImGui::SetNextFrameWantCaptureMouse(true);
    if (CanActivate()) {
        g.Using = true;
        g.EditingID = g.ActualID;
        g.CurrentMoveType = type;
        vec4 move_plane_normal[]{g.Model[0], g.Model[1], g.Model[2], g.Model[0], g.Model[1], g.Model[2], -vec4{g.CameraDir, 0}};
        const auto cam_to_model = glm::normalize(vec3{Pos(g.Model)} - g.CameraEye);
        for (unsigned int i = 0; i < 3; ++i) {
            move_plane_normal[i] = glm::normalize(
                vec4{glm::cross(vec3{move_plane_normal[i]}, glm::cross(vec3{move_plane_normal[i]}, cam_to_model)), 0}
            );
        }
        // pickup plane
        g.TranslationPlane = BuildPlane(Pos(g.Model), move_plane_normal[type - MT_TranslateX]);
        g.TranslationPlaneOrigin = g.RayOrigin + g.RayDir * IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
        g.MatrixOrigin = Pos(g.Model);
        g.RelativeOrigin = (vec4{g.TranslationPlaneOrigin, 1} - Pos(g.Model)) / g.ScreenFactor;
    }
    return type;
}

MoveType HandleScale(mat4 &m, Op op, const float *snap) {
    if (!HasAnyOp(op, Scale) && !HasAnyOp(op, ScaleU)) return MT_None;

    if (!g.Using) {
        // find new possible way to scale
        const auto type = GetScaleType(op);
        if (type == MT_None) return MT_None;

        ImGui::SetNextFrameWantCaptureMouse(true);
        if (CanActivate()) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentMoveType = type;
            const vec4 move_plane_normal[]{g.Model[1], g.Model[2], g.Model[0], g.Model[2], g.Model[1], g.Model[0], -vec4{g.CameraDir, 0}};
            g.TranslationPlane = BuildPlane(Pos(g.Model), move_plane_normal[type - MT_ScaleX]);
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
    if ((g.ActualID != -1 && g.ActualID != g.EditingID) || !IsScale(g.CurrentMoveType)) return MT_None;

    ImGui::SetNextFrameWantCaptureMouse(true);
    const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
    const auto new_pos = g.RayOrigin + g.RayDir * len;
    const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
    auto delta = new_origin - Pos(g.ModelLocal);
    // 1 axis constraint
    if (g.CurrentMoveType >= MT_ScaleX && g.CurrentMoveType <= MT_ScaleZ) {
        int axis_i = g.CurrentMoveType - MT_ScaleX;
        const vec4 &axis_value = *(vec4 *)&g.ModelLocal[axis_i];
        const float length_on_axis = glm::dot(axis_value, delta);
        delta = axis_value * length_on_axis;

        vec4 base = vec4{g.TranslationPlaneOrigin, 0} - Pos(g.ModelLocal);
        const float ratio = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
        g.Scale[axis_i] = std::max(ratio, 0.001f);
    } else {
        const float scale_delta = (ImGui::GetIO().MousePos.x - g.SaveMousePosX) * 0.01f;
        g.Scale = vec4{std::max(1.f + scale_delta, 0.001f)};
    }

    if (snap) {
        const float scale_snap[]{snap[0], snap[0], snap[0]};
        ComputeSnap(g.Scale, scale_snap);
    }

    // no 0 allowed
    for (int i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

    m = g.ModelLocal * glm::scale(mat4{1}, {g.Scale * vec4{g.ScaleOrigin, 0}});

    if (!ImGui::GetIO().MouseDown[0]) {
        g.Using = false;
        g.Scale = vec4{1, 1, 1, 0};
    }
    return g.CurrentMoveType;
}

MoveType HandleRotation(mat4 &m, Op op, const float *snap, bool local) {
    if (!HasAnyOp(op, Rotate)) return MT_None;

    if (!g.Using) {
        const auto type = GetRotateType(op);
        if (type == MT_None) return MT_None;

        ImGui::SetNextFrameWantCaptureMouse(true);
        if (CanActivate()) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentMoveType = type;
            const vec4 rotate_plan_normal[]{Right(g.Model), Up(g.Model), Dir(g.Model), -vec4{g.CameraDir, 0}};
            g.TranslationPlane = local || type == MT_RotateScreen ?
                BuildPlane(Pos(g.Model), rotate_plan_normal[type - MT_RotateX]) :
                BuildPlane(Pos(g.ModelSource), {DirUnary[type - MT_RotateX], 0});

            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlane);
            g.RotationVectorSource = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
            g.RotationAngleOrigin = ComputeAngleOnPlan();
        }
        return type;
    }
    if ((g.ActualID != -1 && g.ActualID != g.EditingID) || !IsRotate(g.CurrentMoveType)) return MT_None;

    ImGui::SetNextFrameWantCaptureMouse(true);
    g.RotationAngle = ComputeAngleOnPlan();
    if (snap) ComputeSnap(&g.RotationAngle, snap[0] * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{g.ModelInverse} * g.TranslationPlane); // Assumes affine model
    const mat4 delta_rot{glm::rotate(mat4{1}, g.RotationAngle - g.RotationAngleOrigin, rot_axis_local)};
    if (g.RotationAngle != g.RotationAngleOrigin) g.RotationAngleOrigin = g.RotationAngle;

    const mat4 scale_origin{glm::scale(mat4{1}, vec3{g.ModelScaleOrigin})};
    if (local) {
        m = g.ModelLocal * delta_rot * scale_origin;
    } else {
        auto res = g.ModelSource;
        SetPos(res, vec4{0});
        m = delta_rot * res;
        SetPos(m, Pos(g.ModelSource));
    }

    if (!ImGui::GetIO().MouseDown[0]) {
        g.Using = false;
        g.EditingID = -1;
    }
    return g.CurrentMoveType;
}
} // namespace

namespace ImGuizmo {
bool Manipulate(vec2 pos, vec2 size, const mat4 &view, const mat4 &proj, Op op, Mode mode, mat4 &m, const float *snap) {
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

    // proj reverse
    const vec4 near_pos{g.Proj * vec4{0, 0, 1, 1}};
    const vec4 far_pos{g.Proj * vec4{0, 0, 2, 1}};
    g.Reversed = near_pos.z / near_pos.w > far_pos.z / far_pos.w;

    // Compute scale from camera right vector projected on screen at m pos
    const auto right_point = g.ViewProj * vec4{vec3{Right(view_inv)}, 1};
    g.ScreenFactor = g.GizmoSizeClipSpace / (right_point.x / right_point.w - Pos(g.MVP).x / Pos(g.MVP).w);

    const float right_len = GetSegmentLengthClipSpace(g.ModelInverse * vec4{vec3{Right(view_inv)}, 0});
    g.ScreenFactor = g.GizmoSizeClipSpace / right_len;

    g.ScreenSquareCenter = WorldToPos(vec3{0}, g.MVP);
    g.ScreenSquareMin = g.ScreenSquareCenter - ImVec2{10, 10};
    g.ScreenSquareMax = g.ScreenSquareCenter + ImVec2{10, 10};

    // Compute camera ray
    const auto view_proj_inv = glm::inverse(g.Proj * g.View);
    const auto mouse_delta = ImGui::GetIO().MousePos - ToImVec(g.Pos);
    const float mox = (mouse_delta.x / g.Size.x) * 2 - 1;
    const float moy = (1 - mouse_delta.y / g.Size.y) * 2 - 1;
    const float z_near = g.Reversed ? 1 - FLT_EPSILON : 0;
    const float z_far = g.Reversed ? 0 : 1 - FLT_EPSILON;

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
    auto type{MT_None};
    if (window_hovered) type = HandleScale(m, op, snap);
    // HandleTranslation has side effects, so call it even if not hovered.
    if (type == MT_None) type = HandleTranslation(m, op, snap, mode == Local);
    if (type == MT_None && window_hovered) type = HandleRotation(m, op, snap, mode == Local);

    // Draw
    auto *draw_list = ImGui::GetWindowDrawList();
    if (!draw_list) return false;

    static constexpr int TranslationInfoIndex[]{0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2};
    static constexpr const char *ScaleInfoMask[]{"X : %5.2f", "Y : %5.2f", "Z : %5.2f", "XYZ : %5.2f"};
    ImU32 colors[7];
    if (HasAnyOp(op, Translate)) {
        ComputeColors(colors, type, Translate);

        const auto origin = WorldToPos(Pos(g.Model), g.ViewProj);
        for (int i = 0; i < 3; ++i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit = false, below_plane_limit = false;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
            if ((!g.Using || type == MT_TranslateX + i) && below_axis_limit) {
                // draw axis
                const auto base = WorldToPos(dir_axis * 0.1f * g.ScreenFactor, g.MVP);
                const auto end = WorldToPos(dir_axis * g.ScreenFactor, g.MVP);
                draw_list->AddLine(base, end, colors[i + 1], g.Style.TranslationLineThickness);

                // Arrow head begin
                auto dir = origin - end;
                dir /= sqrtf(ImLengthSqr(dir)); // Normalize
                dir *= g.Style.TranslationLineArrowSize;

                const ImVec2 orthogonal_dir{dir.y, -dir.x};
                const auto a = end + dir;
                draw_list->AddTriangleFilled(end - dir, a + orthogonal_dir, a - orthogonal_dir, colors[i + 1]);
                if (g.AxisFactor[i] < 0) DrawHatchedAxis(dir_axis);
            }
            if ((!g.Using || type == MT_TranslateYZ + i) && below_plane_limit) {
                // draw plane
                ImVec2 quad_pts_screen[4];
                for (int j = 0; j < 4; ++j) {
                    const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                    quad_pts_screen[j] = WorldToPos(corner_pos_world, g.MVP);
                }
                draw_list->AddPolyline(quad_pts_screen, 4, GetColorU32(DirectionX + i), true, 1.0f);
                draw_list->AddConvexPolyFilled(quad_pts_screen, 4, colors[i + 4]);
            }
        }

        draw_list->AddCircleFilled(g.ScreenSquareCenter, g.Style.CenterCircleSize, colors[0], 32);

        if (IsUsing() && IsTranslate(type)) {
            const auto translation_line_color = GetColorU32(TranslationLine);
            const auto source_pos_screen = WorldToPos(g.MatrixOrigin, g.ViewProj);
            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            const auto dif = ToImVec(glm::normalize(vec4{dest_pos.x - source_pos_screen.x, dest_pos.y - source_pos_screen.y, 0, 0}) * 5.f);
            draw_list->AddCircle(source_pos_screen, 6.f, translation_line_color);
            draw_list->AddCircle(dest_pos, 6.f, translation_line_color);
            draw_list->AddLine(source_pos_screen + dif, dest_pos - dif, translation_line_color, 2.f);

            static constexpr const char *TranslationInfoMask[]{
                "X : %5.3f", "Y : %5.3f", "Z : %5.3f", "Y : %5.3f Z : %5.3f", "X : %5.3f Z : %5.3f", "X : %5.3f Y : %5.3f", "X : %5.3f Y : %5.3f Z : %5.3f"
            };
            const auto delta_info = Pos(g.Model) - g.MatrixOrigin;
            const int info_i = (type - MT_TranslateX) * 3;

            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), TranslationInfoMask[type - MT_TranslateX], delta_info[TranslationInfoIndex[info_i]], delta_info[TranslationInfoIndex[info_i + 1]], delta_info[TranslationInfoIndex[info_i + 2]]);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }
    if (HasAnyOp(op, Rotate)) {
        ComputeColors(colors, type, Rotate);

        static constexpr int HalfCircleSegmentCount{64};
        static constexpr float ScreenRotateSize{0.06};
        g.RadiusSquareCenter = ScreenRotateSize * g.Size.y;

        // Assumes affine model
        const auto cam_to_model = mat3{g.ModelInverse} * vec3{g.IsOrthographic ? -Dir(glm::inverse(g.View)) : glm::normalize(vec3{Pos(g.Model)} - g.CameraEye)};
        for (int axis = 0; axis < 3; axis++) {
            const bool using_axis = g.Using && type == MT_RotateZ - axis;
            const int circle_mul = HasAnyOp(op, RotateScreen) && !using_axis ? 1 : 2;
            const int point_count = circle_mul * HalfCircleSegmentCount + 1;
            std::vector<ImVec2> circle_pos(point_count);
            float angle_start = atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]) + M_PI_2;
            for (int i = 0; i < point_count; ++i) {
                const float ng = angle_start + float(circle_mul) * M_PI * (float(i) / float(point_count - 1));
                const vec4 axis_pos{cosf(ng), sinf(ng), 0, 0};
                const auto pos = vec4{axis_pos[axis], axis_pos[(axis + 1) % 3], axis_pos[(axis + 2) % 3], 0} * g.ScreenFactor * RotationDisplayScale;
                circle_pos[i] = WorldToPos(pos, g.MVP);
            }
            if (!g.Using || type == MT_RotateZ - axis) {
                draw_list->AddPolyline(circle_pos.data(), point_count, colors[3 - axis], false, g.Style.RotationLineThickness);
            }
            if (float radius_axis_sq = ImLengthSqr(WorldToPos(Pos(g.Model), g.ViewProj) - circle_pos[0]);
                radius_axis_sq > g.RadiusSquareCenter * g.RadiusSquareCenter) {
                g.RadiusSquareCenter = sqrtf(radius_axis_sq);
            }
        }
        if (HasAnyOp(op, RotateScreen) && (!g.Using || type == MT_RotateScreen)) {
            draw_list->AddCircle(WorldToPos(Pos(g.Model), g.ViewProj), g.RadiusSquareCenter, colors[0], 64, g.Style.RotationOuterLineThickness);
        }

        if (IsUsing() && IsRotate(type)) {
            ImVec2 circle_pos[HalfCircleSegmentCount + 1];
            circle_pos[0] = WorldToPos(Pos(g.Model), g.ViewProj);
            for (unsigned int i = 1; i < HalfCircleSegmentCount + 1; ++i) {
                const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
                const mat4 rotate{glm::rotate(mat4{1}, ng, vec3{g.TranslationPlane})};
                const auto pos = rotate * vec4{vec3{g.RotationVectorSource}, 1} * g.ScreenFactor * RotationDisplayScale;
                circle_pos[i] = WorldToPos(pos + Pos(g.Model), g.ViewProj);
            }
            draw_list->AddConvexPolyFilled(circle_pos, HalfCircleSegmentCount + 1, GetColorU32(RotationFillActive));
            draw_list->AddPolyline(circle_pos, HalfCircleSegmentCount + 1, GetColorU32(RotationBorderActive), true, g.Style.RotationLineThickness);

            static constexpr const char *RotationInfoMask[]{"X : %5.2f deg %5.2f rad", "Y : %5.2f deg %5.2f rad", "Z : %5.2f deg %5.2f rad", "Screen : %5.2f deg %5.2f rad"};
            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), RotationInfoMask[type - MT_RotateX], (g.RotationAngle / M_PI) * 180.f, g.RotationAngle);

            const auto dest_pos = circle_pos[1];
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }

    if (HasAnyOp(op, Scale)) {
        ComputeColors(colors, type, Scale);

        vec4 scale_display{1};
        if (IsUsing()) scale_display = g.Scale;

        for (int i = 0; i < 3; ++i) {
            if (!g.Using || type == MT_ScaleX + i) {
                vec4 dir_plane_x, dir_plane_y, dir_axis;
                bool below_axis_limit, below_plane_limit;
                ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
                if (below_axis_limit) {
                    // draw axis
                    bool has_translate_on_axis = HasAllOps(op, Translate | (AxisX << i));
                    float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                    const auto base = WorldToPos(dir_axis * 0.1f * g.ScreenFactor, g.MVP);
                    const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVP);
                    if (IsUsing()) {
                        const auto line_color = GetColorU32(ScaleLine);
                        const auto center = WorldToPos(dir_axis * marker_scale * g.ScreenFactor, g.MVP);
                        draw_list->AddLine(base, center, line_color, g.Style.ScaleLineThickness);
                        draw_list->AddCircleFilled(center, g.Style.ScaleLineCircleSize, line_color);
                    }

                    if (g.Using || !has_translate_on_axis) draw_list->AddLine(base, end, colors[i + 1], g.Style.ScaleLineThickness);
                    draw_list->AddCircleFilled(end, g.Style.ScaleLineCircleSize, colors[i + 1]);
                    if (g.AxisFactor[i] < 0) DrawHatchedAxis(dir_axis * scale_display[i]);
                }
            }
        }

        draw_list->AddCircleFilled(g.ScreenSquareCenter, g.Style.CenterCircleSize, colors[0], 32);

        if (IsUsing() && IsScale(type)) {
            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_ScaleX], scale_display[TranslationInfoIndex[(type - MT_ScaleX) * 3]]);

            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }
    if (HasAnyOp(op, ScaleU)) {
        ComputeColors(colors, type, ScaleU);

        const auto scale_display = IsUsing() ? g.Scale : vec4{1};
        for (int i = 0; i < 3; ++i) {
            if (!g.Using || type == MT_ScaleX + i) {
                vec4 dir_plane_x, dir_plane_y, dir_axis;
                bool below_axis_limit, below_plane_limit;
                ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
                if (below_axis_limit) {
                    const bool has_translate_on_axis = HasAllOps(op, Translate | (AxisX << i));
                    const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                    const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVPLocal);
                    draw_list->AddCircleFilled(end, 12.f, colors[i + 1]);
                }
            }
        }
        draw_list->AddCircle(g.ScreenSquareCenter, 20.f, colors[0], 32, g.Style.CenterCircleSize);

        if (IsUsing() && IsScale(type)) {
            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_ScaleX], scale_display[TranslationInfoIndex[(type - MT_ScaleX) * 3]]);

            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }

    return type != MT_None;
}
} // namespace ImGuizmo
