#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>
#include <vector>

namespace {
enum MoveType {
    MT_None = 0,
    MT_MoveX,
    MT_MoveY,
    MT_MoveZ,
    MT_MoveYZ,
    MT_MoveZX,
    MT_MoveXY,
    MT_MoveScreen,
    MT_RotateX,
    MT_RotateY,
    MT_RotateZ,
    MT_RotateScreen,
    MT_ScaleX,
    MT_ScaleY,
    MT_ScaleZ,
    MT_ScaleXYZ
};

constexpr bool IsTranslate(MoveType type) { return type >= MT_MoveX && type <= MT_MoveScreen; }
constexpr bool IsRotate(MoveType type) { return type >= MT_RotateX && type <= MT_RotateScreen; }
constexpr bool IsScale(MoveType type) { return type >= MT_ScaleX && type <= MT_ScaleXYZ; }

using namespace ImGuizmo;
using enum Operation;

constexpr auto OpVal = [](auto op) { return static_cast<std::underlying_type_t<Operation>>(op); };
constexpr Operation operator&(Operation a, Operation b) { return static_cast<Operation>(OpVal(a) & OpVal(b)); }
constexpr Operation operator|(Operation a, Operation b) { return static_cast<Operation>(OpVal(a) | OpVal(b)); }
constexpr Operation operator<<(Operation op, unsigned int shift) { return static_cast<Operation>(OpVal(op) << shift); }
constexpr Operation operator>>(Operation op, unsigned int shift) { return static_cast<Operation>(OpVal(op) >> shift); }
constexpr bool HasAnyOp(Operation a, Operation b) { return (a & b) != Operation::NoOperation; }
constexpr bool HasAllOps(Operation a, Operation b) { return (a & b) == b; }

// Matches MT_MOVE_AB order
constexpr Operation TranslatePlans[]{TranslateY | TranslateZ, TranslateX | TranslateZ, TranslateX | TranslateY};

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
    Mode Mode;

    mat4 View, Proj, ViewProj, Model;
    mat4 ModelLocal; // orthonormalized model
    mat4 ModelInverse;
    mat4 ModelSource;
    mat4 MVP;
    mat4 MVPLocal; // MVP with full model m whereas MVP's model m might only be translation in case of World space edition

    vec4 ModelScaleOrigin;
    vec4 CameraEye, CameraDir;
    vec4 RayOrigin, RayDir;

    float RadiusSquareCenter;
    ImVec2 ScreenSquareCenter, ScreenSquareMin, ScreenSquareMax;

    float ScreenFactor;
    vec4 RelativeOrigin;

    bool Using{false};
    bool MouseOver{false};
    bool Reversed{false}; // reversed proj m

    vec4 TranslationPlan, TranslationPlanOrigin, TranslationPrevDelta;
    vec4 MatrixOrigin;

    vec4 RotationVectorSource;
    float RotationAngle, RotationAngleOrigin;

    vec4 Scale, ScaleOrigin, ScalePrev;
    float SaveMousePosX;

    // save axis factor when using gizmo
    bool BelowAxisLimit[3];
    bool BelowPlaneLimit[3];
    float AxisFactor[3];
    float AxisLimit{0.0025}, PlaneLimit{0.02};

    vec2 Pos{0, 0};
    vec2 Size{0, 0};
    float GizmoSizeClipSpace{0.1};
    bool IsOrthographic{false};

    Operation Op{NoOperation};
    MoveType CurrentMoveType{MT_None};

    int ActualID{-1}, EditingID{-1};
};

Context g;

MoveType GetMoveType(Operation);
MoveType GetRotateType(Operation);
MoveType GetScaleType(Operation);
} // namespace

namespace ImGuizmo {
bool IsUsing() { return g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID); }

bool IsOver() {
    return IsUsing() || (HasAnyOp(g.Op, Translate) && GetMoveType(g.Op)) ||
        (HasAnyOp(g.Op, Rotate) && GetRotateType(g.Op)) ||
        (HasAnyOp(g.Op, Scale) && GetScaleType(g.Op));
}
bool IsOver(Operation op) {
    if (IsUsing()) return true;
    if (HasAnyOp(op, Scale) && GetScaleType(op)) return true;
    if (HasAnyOp(op, Rotate) && GetRotateType(op)) return true;
    if (HasAnyOp(op, Translate) && GetMoveType(op)) return true;
    return false;
}
void SetRect(vec2 pos, vec2 size) {
    g.Pos = pos;
    g.Size = size;
}
} // namespace ImGuizmo

namespace {
constexpr ImVec2 WorldToPos(vec3 pos_world, const mat4 &m) {
    auto trans = vec2{m * vec4{pos_world, 1}} * (0.5f / glm::dot(glm::transpose(m)[3], vec4{pos_world, 1})) + 0.5f;
    trans.y = 1.f - trans.y;
    trans = g.Pos + trans * g.Size;
    return {trans.x, trans.y};
}

constexpr float GetSegmentLengthClipSpace(const vec4 &start, const vec4 &end, const bool local_coords = false) {
    const auto &mvp = local_coords ? g.MVPLocal : g.MVP;
    auto segment_start = mvp * vec4{vec3{start}, 1};
    // check for axis aligned with camera direction
    if (fabsf(segment_start.w) > FLT_EPSILON) segment_start /= segment_start.w;

    auto segment_end = mvp * vec4{vec3{end}, 1};
    // check for axis aligned with camera direction
    if (fabsf(segment_end.w) > FLT_EPSILON) segment_end /= segment_end.w;

    auto clip_space_axis = segment_end - segment_start;
    const auto aspect_ratio = g.Size.x / g.Size.y;
    if (aspect_ratio < 1.0) clip_space_axis.x *= aspect_ratio;
    else clip_space_axis.y /= aspect_ratio;
    return sqrtf(clip_space_axis.x * clip_space_axis.x + clip_space_axis.y * clip_space_axis.y);
}

constexpr float GetParallelogram(const vec4 &p0, const vec4 &pa, const vec4 &pb) {
    vec4 pts[]{p0, pa, pb};
    for (uint32_t i = 0; i < 3; ++i) {
        pts[i] = g.MVP * vec4{vec3{pts[i]}, 1};
        // check for axis aligned with camera direction
        if (fabsf(pts[i].w) > FLT_EPSILON) pts[i] /= pts[i].w;
    }
    const auto aspect_ratio = g.Size.x / g.Size.y;
    auto seg_a = pts[1] - pts[0];
    seg_a.y /= aspect_ratio;

    auto seg_b = pts[2] - pts[0];
    seg_b.y /= aspect_ratio;

    const auto seg_a_ortho = glm::normalize(vec4{-seg_a.y, seg_a.x, 0, 0});
    return sqrtf(seg_a.x * seg_a.x + seg_a.y * seg_a.y) * fabsf(glm::dot(vec3{seg_a_ortho}, vec3{seg_b}));
}

vec4 Right(const mat4 &m) { return {m[0]}; }
vec4 Up(const mat4 &m) { return {m[1]}; }
vec4 Dir(const mat4 &m) { return {m[2]}; }
vec4 Pos(const mat4 &m) { return {m[3]}; }
void SetPos(mat4 &m, const vec4 &pos) { m[3] = pos; }

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }
ImVec2 ToImVec(vec2 v) { return {v.x, v.y}; }

void ComputeContext(const mat4 &view, const mat4 &proj, mat4 &m, Mode mode) {
    g.Mode = mode;
    g.View = view;
    g.Proj = proj;
    g.MouseOver = ImGui::IsWindowHovered();

    auto &model_local = g.ModelLocal;
    model_local[0] = glm::normalize(m[0]);
    model_local[1] = glm::normalize(m[1]);
    model_local[2] = glm::normalize(m[2]);
    model_local[3] = m[3];

    g.Model = mode == Local ? g.ModelLocal : glm::translate(mat4{1}, vec3{Pos(m)});
    g.ModelSource = m;
    g.ModelScaleOrigin = vec4{glm::length(Right(g.ModelSource)), glm::length(Up(g.ModelSource)), glm::length(Dir(g.ModelSource)), 0};

    g.ModelInverse = glm::inverse(g.Model);
    g.ViewProj = g.Proj * g.View;
    g.MVP = g.ViewProj * g.Model;
    g.MVPLocal = g.ViewProj * g.ModelLocal;

    const mat4 view_inv{glm::inverse(g.View)};
    g.CameraDir = Dir(view_inv);
    g.CameraEye = Pos(view_inv);

    // proj reverse
    const vec4 near_pos{g.Proj * vec4{0, 0, 1, 1}};
    const vec4 far_pos{g.Proj * vec4{0, 0, 2, 1}};
    g.Reversed = near_pos.z / near_pos.w > far_pos.z / far_pos.w;

    // compute scale from the size of camera right vector projected on screen at the m pos
    const auto right_point = g.ViewProj * vec4{vec3{Right(view_inv)}, 1};
    g.ScreenFactor = g.GizmoSizeClipSpace / (right_point.x / right_point.w - Pos(g.MVP).x / Pos(g.MVP).w);

    const float right_len = GetSegmentLengthClipSpace(vec4{0}, g.ModelInverse * vec4{vec3{Right(view_inv)}, 0});
    g.ScreenFactor = g.GizmoSizeClipSpace / right_len;

    g.ScreenSquareCenter = WorldToPos(vec3{0}, g.MVP);
    g.ScreenSquareMin = g.ScreenSquareCenter - ImVec2{10, 10};
    g.ScreenSquareMax = g.ScreenSquareCenter + ImVec2{10, 10};

    // Compute camera ray
    const auto view_proj_inv = glm::inverse(g.Proj * g.View);
    const auto mouse_delta = ImGui::GetIO().MousePos - ToImVec(g.Pos);
    const float mox = (mouse_delta.x / g.Size.x) * 2 - 1;
    const float moy = (1 - (mouse_delta.y / g.Size.y)) * 2 - 1;
    const float z_near = g.Reversed ? 1 - FLT_EPSILON : 0;
    const float z_far = g.Reversed ? 0 : 1 - FLT_EPSILON;
    g.RayOrigin = view_proj_inv * vec4{mox, moy, z_near, 1};
    g.RayOrigin /= g.RayOrigin.w;

    vec4 ray_end{view_proj_inv * vec4{mox, moy, z_far, 1}};
    ray_end /= ray_end.w;
    g.RayDir = glm::normalize(ray_end - g.RayOrigin);
}

constexpr ImU32 GetColorU32(int idx) {
    IM_ASSERT(idx < Color::COUNT);
    return ImGui::ColorConvertFloat4ToU32(g.Style.Colors[idx]);
}

constexpr void ComputeColors(ImU32 *colors, MoveType type, Operation op) {
    const auto selection_color = GetColorU32(Selection);
    switch (op) {
        case Translate:
            colors[0] = (type == MT_MoveScreen) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; ++i) {
                colors[i + 1] = type == MoveType(MT_MoveX + i) ? selection_color : GetColorU32(DirectionX + i);
                colors[i + 4] = type == MoveType(MT_MoveYZ + i) ? selection_color : GetColorU32(PlaneX + i);
                colors[i + 4] = type == MT_MoveScreen ? selection_color : colors[i + 4];
            }
            break;
        case Rotate:
            colors[0] = (type == MT_RotateScreen) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; ++i) {
                colors[i + 1] = type == int(MT_RotateX + i) ? selection_color : GetColorU32(DirectionX + i);
            }
            break;
        case ScaleU:
        case Scale:
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

const vec3 DirUnary[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

static constexpr vec4 Origin{0};

void ComputeTripodAxis(const int axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, const bool local_coords = false) {
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        dir_axis *= g.AxisFactor[axis_i];
        dir_plane_x *= g.AxisFactor[(axis_i + 1) % 3];
        dir_plane_y *= g.AxisFactor[(axis_i + 2) % 3];
        return;
    }

    dir_axis = {DirUnary[axis_i], 0};
    dir_plane_x = {DirUnary[(axis_i + 1) % 3], 0};
    dir_plane_y = {DirUnary[(axis_i + 2) % 3], 0};

    const float len_dir = GetSegmentLengthClipSpace(Origin, dir_axis, local_coords);
    const float len_dir_minus = GetSegmentLengthClipSpace(Origin, -dir_axis, local_coords);
    const float len_dir_plane_x = GetSegmentLengthClipSpace(Origin, dir_plane_x, local_coords);
    const float len_dir_plane_x_minus = GetSegmentLengthClipSpace(Origin, -dir_plane_x, local_coords);
    const float len_dir_plane_y = GetSegmentLengthClipSpace(Origin, dir_plane_y, local_coords);
    const float len_dir_plane_y_minus = GetSegmentLengthClipSpace(Origin, -dir_plane_y, local_coords);

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

void ComputeTripodVisibility(const int axis_i, float axis_length_clip_space, vec4 dir_plane_x, vec4 dir_plane_y, bool &below_axis_limit, bool &below_plane_limit) {
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        below_axis_limit = g.BelowAxisLimit[axis_i];
        below_plane_limit = g.BelowPlaneLimit[axis_i];
        return;
    }

    const float para_surf = GetParallelogram(Origin, dir_plane_x * g.ScreenFactor, dir_plane_y * g.ScreenFactor);
    below_plane_limit = para_surf > g.AxisLimit;
    below_axis_limit = axis_length_clip_space > g.PlaneLimit;
    // Cache
    g.BelowAxisLimit[axis_i] = below_axis_limit;
    g.BelowPlaneLimit[axis_i] = below_plane_limit;
}

void ComputeTripodAxisAndVisibility(const int axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool &below_axis_limit, bool &below_plane_limit, const bool local_coords = false) {
    ComputeTripodAxis(axis_i, dir_axis, dir_plane_x, dir_plane_y, local_coords);
    const float axis_length_clip_space = GetSegmentLengthClipSpace(Origin, dir_axis * g.ScreenFactor, local_coords);
    ComputeTripodVisibility(axis_i, axis_length_clip_space, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
}

constexpr void ComputeSnap(float *value, float snap) {
    if (snap <= FLT_EPSILON) return;

    static constexpr float SnapTension{0.5};
    const float modulo = fmodf(*value, snap);
    const float modulo_ratio = fabsf(modulo) / snap;
    if (modulo_ratio < SnapTension) *value -= modulo;
    else if (modulo_ratio > (1.f - SnapTension)) *value = *value - modulo + snap * (*value < 0 ? -1 : 1);
}
constexpr void ComputeSnap(vec4 &value, const float *snap) {
    for (int i = 0; i < 3; ++i) ComputeSnap(&value[i], snap[i]);
}

constexpr float IntersectRayPlane(const vec4 &origin, const vec4 &dir, const vec4 &plan) {
    const float num = glm::dot(vec3{plan}, vec3{origin}) - plan.w;
    const float den = glm::dot(vec3{plan}, vec3{dir});
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -(num / den);
}

constexpr float ComputeAngleOnPlan() {
    vec4 perp{glm::normalize(vec4{glm::cross(vec3{g.RotationVectorSource}, vec3{g.TranslationPlan}), 0})};
    const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
    const auto pos_local = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
    const float acos_angle = std::clamp(glm::dot(pos_local, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (glm::dot(pos_local, perp) < 0 ? 1.f : -1.f);
}

void DrawHatchedAxis(vec3 axis) {
    if (g.Style.HatchedAxisLineThickness <= 0.0f) return;

    for (int j = 1; j < 10; j++) {
        const auto base = WorldToPos(axis * 0.05f * float(j * 2) * g.ScreenFactor, g.MVP);
        const auto end = WorldToPos(axis * 0.05f * float(j * 2 + 1) * g.ScreenFactor, g.MVP);
        ImGui::GetWindowDrawList()->AddLine(base, end, GetColorU32(HatchedAxisLines), g.Style.HatchedAxisLineThickness);
    }
}

constexpr vec4 BuildPlan(const vec4 &p_point1, const vec4 &p_normal) {
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

MoveType GetScaleType(Operation op) {
    if (g.Using) return MT_None;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    auto type = mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
            mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y &&
            HasAllOps(op, Scale) ?
        MT_ScaleXYZ :
        MT_None;

    // compute
    for (int i = 0; i < 3 && type == MT_None; ++i) {
        if (!HasAnyOp(op, static_cast<Operation>(ScaleX << i))) continue;
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        ComputeTripodAxis(i, dir_axis, dir_plane_x, dir_plane_y, true);
        dir_axis = g.ModelLocal * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.ModelLocal * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.ModelLocal * vec4{vec3{dir_plane_y}, 0};

        const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlan(Pos(g.ModelLocal), dir_axis));
        const float start_offset = HasAllOps(op, Operation(TranslateX << i)) ? 1.0f : 0.1f;
        const float end_offset = HasAllOps(op, Operation(TranslateX << i)) ? 1.4f : 1.0f;
        const auto pos_plan = g.RayOrigin + g.RayDir * len;
        const auto pos_plan_screen = WorldToPos(pos_plan, g.ViewProj);
        const auto axis_start_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * start_offset, g.ViewProj);
        const auto axis_end_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * end_offset, g.ViewProj);
        const auto closest_on_axis = PointOnSegment(pos_plan_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_plan_screen) < SelectDistSq) type = MoveType(MT_ScaleX + i);
    }

    // universal
    const vec4 delta_screen{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0};
    if (float dist = glm::length(delta_screen); dist >= 17.0f && dist < 23.0f && HasAllOps(op, ScaleU)) type = MT_ScaleXYZ;

    for (int i = 0; i < 3 && type == MT_None; ++i) {
        if (!HasAnyOp(op, Operation(ScaleXU << i))) continue;

        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

        // draw axis
        if (below_axis_limit) {
            const bool has_translate_on_axis = HasAllOps(op, Operation(TranslateX << i));
            const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
            const auto end = WorldToPos((dir_axis * marker_scale) * g.ScreenFactor, g.MVPLocal);
            if (ImLengthSqr(end - mouse_pos) < SelectDistSq) type = MoveType(MT_ScaleX + i);
        }
    }
    return type;
}

// Scale a bit so translate axes don't touch when in universal.
constexpr float RotationDisplayScale{1.2};

MoveType GetRotateType(Operation op) {
    if (g.Using) return MT_None;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    const vec4 delta_screen{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0};
    const auto dist = glm::length(delta_screen);
    auto type = HasAnyOp(op, RotateScreen) && dist >= (g.RadiusSquareCenter - 4) && dist < (g.RadiusSquareCenter + 4) ?
        MT_RotateScreen :
        MT_None;

    const auto model_view_pos = g.View * vec4{vec3{Pos(g.Model)}, 1};
    for (int i = 0; i < 3 && type == MT_None; ++i) {
        if (!HasAnyOp(op, Operation(RotateX << i))) continue;

        const auto pickup_plan = BuildPlan(Pos(g.Model), g.Model[i]);
        const auto len = IntersectRayPlane(g.RayOrigin, g.RayDir, pickup_plan);
        const auto intersect_world_pos = g.RayOrigin + g.RayDir * len;
        const auto intersect_view_pos = g.View * vec4{vec3{intersect_world_pos}, 1};
        if (ImAbs(model_view_pos.z) - ImAbs(intersect_view_pos.z) < -FLT_EPSILON) continue;

        auto ideal_pos_circle = g.ModelInverse * vec4{vec3{glm::normalize(intersect_world_pos - Pos(g.Model))}, 0};
        const auto ideal_circle_pos_screen = WorldToPos(ideal_pos_circle * RotationDisplayScale * g.ScreenFactor, g.MVP);
        const auto distance_screen = ideal_circle_pos_screen - mouse_pos;
        if (glm::length(vec2(distance_screen.x, distance_screen.y)) < 8) type = MoveType(MT_RotateX + i);
    }

    return type;
}

constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

MoveType GetMoveType(Operation op) {
    if (g.Using || !g.MouseOver || !HasAnyOp(op, Translate)) return MT_None;

    auto &io = ImGui::GetIO();
    auto type{MT_None};
    if (io.MousePos.x >= g.ScreenSquareMin.x && io.MousePos.x <= g.ScreenSquareMax.x &&
        io.MousePos.y >= g.ScreenSquareMin.y && io.MousePos.y <= g.ScreenSquareMax.y &&
        HasAllOps(op, Translate)) {
        type = MT_MoveScreen;
    }

    const auto pos = ToImVec(g.Pos);
    const ImVec2 pos_screen{io.MousePos - pos};
    for (int i = 0; i < 3 && type == MT_None; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        dir_axis = g.Model * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.Model * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.Model * vec4{vec3{dir_plane_y}, 0};

        const auto axis_start_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor * 0.1f, g.ViewProj) - pos;
        const auto axis_end_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor, g.ViewProj) - pos;
        const auto closest_on_axis = PointOnSegment(pos_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_screen) < SelectDistSq && HasAnyOp(op, Operation(TranslateX << i))) {
            type = MoveType(MT_MoveX + i);
        }

        const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, BuildPlan(Pos(g.Model), dir_axis));
        const auto pos_plan = g.RayOrigin + g.RayDir * len;
        const float dx = glm::dot(vec3{dir_plane_x}, vec3{pos_plan - Pos(g.Model)} / g.ScreenFactor);
        const float dy = glm::dot(vec3{dir_plane_y}, vec3{pos_plan - Pos(g.Model)} / g.ScreenFactor);
        if (below_plane_limit && dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3] && HasAllOps(op, TranslatePlans[i])) {
            type = MoveType(MT_MoveYZ + i);
        }
    }
    return type;
}

bool CanActivate() { return ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive(); }

bool HandleTranslation(mat4 &m, Operation op, MoveType &type, const float *snap) {
    if (type != MT_None || !HasAnyOp(op, Translate)) return false;

    const bool apply_rot_locally = g.Mode == Local || type == MT_MoveScreen;
    bool modified = false;

    // move
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsTranslate(g.CurrentMoveType)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len_signed = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
        const float len = fabsf(len_signed); // near plan
        const auto new_pos = g.RayOrigin + g.RayDir * len;

        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.Model);

        // 1 axis constraint
        if (g.CurrentMoveType >= MT_MoveX && g.CurrentMoveType <= MT_MoveZ) {
            const int axis_i = g.CurrentMoveType - MT_MoveX;
            const auto &axis_value = *(vec4 *)&g.Model[axis_i];
            const auto length_on_axis = glm::dot(axis_value, delta);
            delta = axis_value * length_on_axis;
        }

        // snap
        if (snap) {
            auto delta_cumulative = Pos(g.Model) + delta - g.MatrixOrigin;
            if (apply_rot_locally) {
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

        if (delta != g.TranslationPrevDelta) {
            g.TranslationPrevDelta = delta;
            modified = true;
        }

        m = glm::translate(mat4{1}, vec3{delta}) * g.ModelSource;

        if (!ImGui::GetIO().MouseDown[0]) g.Using = false;

        type = g.CurrentMoveType;
    } else {
        // find new possible way to move
        type = GetMoveType(op);
        if (type != MT_None) {
            ImGui::SetNextFrameWantCaptureMouse(true);
            if (CanActivate()) {
                g.Using = true;
                g.EditingID = g.ActualID;
                g.CurrentMoveType = type;
                vec4 move_plan_normal[]{g.Model[0], g.Model[1], g.Model[2], g.Model[0], g.Model[1], g.Model[2], -g.CameraDir};
                const auto cam_to_model = glm::normalize(Pos(g.Model) - g.CameraEye);
                for (unsigned int i = 0; i < 3; ++i) {
                    move_plan_normal[i] = glm::normalize(
                        vec4{glm::cross(vec3{move_plan_normal[i]}, glm::cross(vec3{move_plan_normal[i]}, vec3{cam_to_model})), 0}
                    );
                }
                // pickup plan
                g.TranslationPlan = BuildPlan(Pos(g.Model), move_plan_normal[type - MT_MoveX]);
                const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
                g.TranslationPlanOrigin = g.RayOrigin + g.RayDir * len;
                g.MatrixOrigin = Pos(g.Model);

                g.RelativeOrigin = (g.TranslationPlanOrigin - Pos(g.Model)) * (1.f / g.ScreenFactor);
            }
        }
    }
    return modified;
}

bool HandleScale(mat4 &m, Operation op, MoveType &type, const float *snap) {
    if (type != MT_None || !g.MouseOver || (!HasAnyOp(op, Scale) && !HasAnyOp(op, ScaleU))) return false;

    bool modified = false;
    if (!g.Using) {
        // find new possible way to scale
        type = GetScaleType(op);
        if (type != MT_None) ImGui::SetNextFrameWantCaptureMouse(true);
        if (CanActivate() && type != MT_None) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentMoveType = type;
            const vec4 move_plan_normal[]{g.Model[1], g.Model[2], g.Model[0], g.Model[2], g.Model[1], g.Model[0], -g.CameraDir};
            g.TranslationPlan = BuildPlan(Pos(g.Model), move_plan_normal[type - MT_ScaleX]);
            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
            g.TranslationPlanOrigin = g.RayOrigin + g.RayDir * len;
            g.MatrixOrigin = Pos(g.Model);
            g.Scale = {1, 1, 1, 0};
            g.RelativeOrigin = (g.TranslationPlanOrigin - Pos(g.Model)) * (1.f / g.ScreenFactor);
            g.ScaleOrigin = {glm::length(Right(g.ModelSource)), glm::length(Up(g.ModelSource)), glm::length(Dir(g.ModelSource)), 0};
            g.SaveMousePosX = ImGui::GetIO().MousePos.x;
        }
    }
    // scale
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScale(g.CurrentMoveType)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
        const auto new_pos = g.RayOrigin + g.RayDir * len;
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.ModelLocal);
        // 1 axis constraint
        if (g.CurrentMoveType >= MT_ScaleX && g.CurrentMoveType <= MT_ScaleZ) {
            int axis_i = g.CurrentMoveType - MT_ScaleX;
            const vec4 &axis_value = *(vec4 *)&g.ModelLocal[axis_i];
            const float length_on_axis = glm::dot(axis_value, delta);
            delta = axis_value * length_on_axis;

            vec4 base = g.TranslationPlanOrigin - Pos(g.ModelLocal);
            const float ratio = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
            g.Scale[axis_i] = std::max(ratio, 0.001f);
        } else {
            const float scale_delta = (ImGui::GetIO().MousePos.x - g.SaveMousePosX) * 0.01f;
            g.Scale = vec4{std::max(1.f + scale_delta, 0.001f)};
        }

        // snap
        if (snap) {
            const float scale_snap[]{snap[0], snap[0], snap[0]};
            ComputeSnap(g.Scale, scale_snap);
        }

        // no 0 allowed
        for (int i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        if (g.ScalePrev != g.Scale) {
            g.ScalePrev = g.Scale;
            modified = true;
        }

        m = g.ModelLocal * glm::scale(mat4{1}, {g.Scale * g.ScaleOrigin});

        if (!ImGui::GetIO().MouseDown[0]) {
            g.Using = false;
            g.Scale = vec4{1, 1, 1, 0};
        }

        type = g.CurrentMoveType;
    }
    return modified;
}

bool HandleRotation(mat4 &m, Operation op, MoveType &type, const float *snap) {
    if (!HasAnyOp(op, Rotate) || type != MT_None || !g.MouseOver) return false;

    bool apply_rot_locally{g.Mode == Local};
    bool modified{false};
    if (!g.Using) {
        type = GetRotateType(op);

        if (type != MT_None) ImGui::SetNextFrameWantCaptureMouse(true);
        if (type == MT_RotateScreen) apply_rot_locally = true;

        if (CanActivate() && type != MT_None) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentMoveType = type;
            const vec4 rotate_plan_normal[]{Right(g.Model), Up(g.Model), Dir(g.Model), -g.CameraDir};
            g.TranslationPlan = apply_rot_locally ?
                BuildPlan(Pos(g.Model), rotate_plan_normal[type - MT_RotateX]) :
                BuildPlan(Pos(g.ModelSource), {DirUnary[type - MT_RotateX], 0});

            const float len = IntersectRayPlane(g.RayOrigin, g.RayDir, g.TranslationPlan);
            g.RotationVectorSource = glm::normalize(g.RayOrigin + g.RayDir * len - Pos(g.Model));
            g.RotationAngleOrigin = ComputeAngleOnPlan();
        }
    }

    // rotation
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotate(g.CurrentMoveType)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        g.RotationAngle = ComputeAngleOnPlan();
        if (snap) ComputeSnap(&g.RotationAngle, snap[0] * M_PI / 180.f);

        vec4 rot_axis_local_space = glm::normalize(g.ModelInverse * vec4{vec3(g.TranslationPlan), 0});
        const mat4 delta_rot{glm::rotate(mat4{1}, g.RotationAngle - g.RotationAngleOrigin, vec3{rot_axis_local_space})};
        if (g.RotationAngle != g.RotationAngleOrigin) {
            g.RotationAngleOrigin = g.RotationAngle;
            modified = true;
        }

        const mat4 scale_origin{glm::scale(mat4{1}, vec3{g.ModelScaleOrigin})};
        if (apply_rot_locally) {
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
        type = g.CurrentMoveType;
    }
    return modified;
}
} // namespace

namespace ImGuizmo {
bool Manipulate(const mat4 &view, const mat4 &proj, Operation op, Mode mode, mat4 &m, const float *snap) {
    // Scale is always local or m will be skewed when applying world scale or oriented m
    ComputeContext(view, proj, m, HasAllOps(op, Scale) ? Local : mode);

    // behind camera
    const auto pos_cam_space = g.MVP * vec4{vec3{0}, 1};
    if (!g.IsOrthographic && pos_cam_space.z < 0.001 && !g.Using) return false;

    auto type{MT_None};
    const bool manipulated = HandleTranslation(m, op, type, snap) ||
        HandleScale(m, op, type, snap) ||
        HandleRotation(m, op, type, snap);

    g.Op = op;

    // Draw
    auto *draw_list = ImGui::GetWindowDrawList();
    if (!draw_list) return false;

    ImU32 colors[7];
    if (HasAnyOp(op, Rotate)) {
        ComputeColors(colors, type, Rotate);

        vec4 cam_to_model = g.IsOrthographic ? -Dir(glm::inverse(g.View)) : glm::normalize(Pos(g.Model) - g.CameraEye);
        cam_to_model = g.ModelInverse * vec4{vec3{cam_to_model}, 0};

        static constexpr int HalfCircleSegmentCount{64};
        static constexpr float ScreenRotateSize{0.06};
        g.RadiusSquareCenter = ScreenRotateSize * g.Size.y;

        bool hasRSC = HasAnyOp(op, RotateScreen);
        for (int axis = 0; axis < 3; axis++) {
            if (!HasAnyOp(op, Operation(RotateZ >> axis))) continue;

            const bool using_axis = g.Using && type == MT_RotateZ - axis;
            const int circle_mul = hasRSC && !using_axis ? 1 : 2;
            const int point_count = circle_mul * HalfCircleSegmentCount + 1;
            std::vector<ImVec2> circle_pos(point_count);
            float angle_start = atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]) + M_PI_2;
            for (int i = 0; i < circle_mul * HalfCircleSegmentCount + 1; ++i) {
                const float ng = angle_start + float(circle_mul) * M_PI * (float(i) / float(circle_mul * HalfCircleSegmentCount));
                const vec4 axis_pos{cosf(ng), sinf(ng), 0, 0};
                const auto pos = vec4{axis_pos[axis], axis_pos[(axis + 1) % 3], axis_pos[(axis + 2) % 3], 0} * g.ScreenFactor * RotationDisplayScale;
                circle_pos[i] = WorldToPos(pos, g.MVP);
            }
            if (!g.Using || using_axis) {
                draw_list->AddPolyline(circle_pos.data(), circle_mul * HalfCircleSegmentCount + 1, colors[3 - axis], false, g.Style.RotationLineThickness);
            }
            if (float radius_axis = sqrtf((ImLengthSqr(WorldToPos(Pos(g.Model), g.ViewProj) - circle_pos[0])));
                radius_axis > g.RadiusSquareCenter) {
                g.RadiusSquareCenter = radius_axis;
            }
        }
        if (hasRSC && (!g.Using || type == MT_RotateScreen)) {
            draw_list->AddCircle(WorldToPos(Pos(g.Model), g.ViewProj), g.RadiusSquareCenter, colors[0], 64, g.Style.RotationOuterLineThickness);
        }

        if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotate(type)) {
            ImVec2 circle_pos[HalfCircleSegmentCount + 1];
            circle_pos[0] = WorldToPos(Pos(g.Model), g.ViewProj);
            for (unsigned int i = 1; i < HalfCircleSegmentCount + 1; ++i) {
                const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
                const mat4 rotate{glm::rotate(mat4{1}, ng, vec3{g.TranslationPlan})};
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

    static constexpr int TranslationInfoIndex[]{0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2};
    static constexpr const char *ScaleInfoMask[]{"X : %5.2f", "Y : %5.2f", "Z : %5.2f", "XYZ : %5.2f"};
    if (HasAnyOp(op, Translate)) {
        ComputeColors(colors, type, Translate);

        const auto origin = WorldToPos(Pos(g.Model), g.ViewProj);
        bool below_axis_limit = false, below_plane_limit = false;
        for (int i = 0; i < 3; ++i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
            if (!g.Using || (g.Using && type == MT_MoveX + i)) {
                // draw axis
                if (below_axis_limit && HasAnyOp(op, Operation(TranslateX << i))) {
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
            }
            // draw plane
            if (!g.Using || (g.Using && type == MT_MoveYZ + i)) {
                if (below_plane_limit && HasAllOps(op, TranslatePlans[i])) {
                    ImVec2 quad_pts_screen[4];
                    for (int j = 0; j < 4; ++j) {
                        const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                        quad_pts_screen[j] = WorldToPos(corner_pos_world, g.MVP);
                    }
                    draw_list->AddPolyline(quad_pts_screen, 4, GetColorU32(DirectionX + i), true, 1.0f);
                    draw_list->AddConvexPolyFilled(quad_pts_screen, 4, colors[i + 4]);
                }
            }
        }

        draw_list->AddCircleFilled(g.ScreenSquareCenter, g.Style.CenterCircleSize, colors[0], 32);

        if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsTranslate(type)) {
            const auto translation_line_color = GetColorU32(TranslationLine);
            const auto source_pos_screen = WorldToPos(g.MatrixOrigin, g.ViewProj);
            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            const auto dif = glm::normalize(vec4{dest_pos.x - source_pos_screen.x, dest_pos.y - source_pos_screen.y, 0, 0}) * 5.f;
            draw_list->AddCircle(source_pos_screen, 6.f, translation_line_color);
            draw_list->AddCircle(dest_pos, 6.f, translation_line_color);
            draw_list->AddLine({source_pos_screen.x + dif.x, source_pos_screen.y + dif.y}, {dest_pos.x - dif.x, dest_pos.y - dif.y}, translation_line_color, 2.f);

            const auto delta_info = Pos(g.Model) - g.MatrixOrigin;
            const int component_info_i = (type - MT_MoveX) * 3;

            static constexpr const char *TranslationInfoMask[]{"X : %5.3f", "Y : %5.3f", "Z : %5.3f", "Y : %5.3f Z : %5.3f", "X : %5.3f Z : %5.3f", "X : %5.3f Y : %5.3f", "X : %5.3f Y : %5.3f Z : %5.3f"};

            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), TranslationInfoMask[type - MT_MoveX], delta_info[TranslationInfoIndex[component_info_i]], delta_info[TranslationInfoIndex[component_info_i + 1]], delta_info[TranslationInfoIndex[component_info_i + 2]]);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }
    if (HasAnyOp(op, Scale)) {
        ComputeColors(colors, type, Scale);

        vec4 scale_display{1};
        if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) scale_display = g.Scale;

        for (int i = 0; i < 3; ++i) {
            if (!HasAnyOp(op, Operation(ScaleX << i))) continue;

            if (!g.Using || type == MT_ScaleX + i) {
                vec4 dir_plane_x, dir_plane_y, dir_axis;
                bool below_axis_limit, below_plane_limit;
                ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

                // draw axis
                if (below_axis_limit) {
                    bool has_translate_on_axis = HasAllOps(op, Operation(TranslateX << i));
                    float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                    const auto base = WorldToPos(dir_axis * 0.1f * g.ScreenFactor, g.MVP);
                    const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVP);
                    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
                        const auto line_color = GetColorU32(ScaleLine);
                        const auto center = WorldToPos(dir_axis * marker_scale * g.ScreenFactor, g.MVP);
                        draw_list->AddLine(base, center, line_color, g.Style.ScaleLineThickness);
                        draw_list->AddCircleFilled(center, g.Style.ScaleLineCircleSize, line_color);
                    }

                    if (!has_translate_on_axis || g.Using) draw_list->AddLine(base, end, colors[i + 1], g.Style.ScaleLineThickness);
                    draw_list->AddCircleFilled(end, g.Style.ScaleLineCircleSize, colors[i + 1]);
                    if (g.AxisFactor[i] < 0) DrawHatchedAxis(dir_axis * scale_display[i]);
                }
            }
        }

        draw_list->AddCircleFilled(g.ScreenSquareCenter, g.Style.CenterCircleSize, colors[0], 32);

        if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScale(type)) {
            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_ScaleX], scale_display[TranslationInfoIndex[(type - MT_ScaleX) * 3]]);

            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }
    if (HasAnyOp(op, ScaleU)) {
        ComputeColors(colors, type, ScaleU);

        const auto scale_display = g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) ? g.Scale : vec4{1};
        for (int i = 0; i < 3; ++i) {
            if (!HasAnyOp(op, Operation(ScaleXU << i))) continue;

            if (!g.Using || type == MT_ScaleX + i) {
                vec4 dir_plane_x, dir_plane_y, dir_axis;
                bool below_axis_limit, below_plane_limit;
                ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
                if (below_axis_limit) {
                    const bool has_translate_on_axis = HasAllOps(op, Operation(TranslateX << i));
                    const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                    const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVPLocal);
                    draw_list->AddCircleFilled(end, 12.f, colors[i + 1]);
                }
            }
        }

        draw_list->AddCircle(g.ScreenSquareCenter, 20.f, colors[0], 32, g.Style.CenterCircleSize);

        if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScale(type)) {
            char tmps[512];
            ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_ScaleX], scale_display[TranslationInfoIndex[(type - MT_ScaleX) * 3]]);

            const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
            draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TextShadow), tmps);
            draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(Text), tmps);
        }
    }

    return manipulated;
}
} // namespace ImGuizmo
