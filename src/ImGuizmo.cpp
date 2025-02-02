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
    MT_NONE,
    MT_MOVE_X,
    MT_MOVE_Y,
    MT_MOVE_Z,
    MT_MOVE_YZ,
    MT_MOVE_ZX,
    MT_MOVE_XY,
    MT_MOVE_SCREEN,
    MT_ROTATE_X,
    MT_ROTATE_Y,
    MT_ROTATE_Z,
    MT_ROTATE_SCREEN,
    MT_SCALE_X,
    MT_SCALE_Y,
    MT_SCALE_Z,
    MT_SCALE_XYZ
};

constexpr bool IsTranslateType(int type) { return type >= MT_MOVE_X && type <= MT_MOVE_SCREEN; }
constexpr bool IsRotateType(int type) { return type >= MT_ROTATE_X && type <= MT_ROTATE_SCREEN; }
constexpr bool IsScaleType(int type) { return type >= MT_SCALE_X && type <= MT_SCALE_XYZ; }

using namespace ImGuizmo;

constexpr bool Intersects(Operation lhs, Operation rhs) { return (lhs & rhs) != 0; }
constexpr bool Contains(Operation lhs, Operation rhs) { return (lhs & rhs) == rhs; }
constexpr Operation operator|(Operation lhs, Operation rhs) { return static_cast<Operation>(static_cast<int>(lhs) | static_cast<int>(rhs)); }

// Matches MT_MOVE_AB order
constexpr Operation TranslatePlans[]{TRANSLATE_Y | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Y};

struct Style {
    Style() {
        // initialize default colors
        Colors[DIRECTION_X] = {.666, 0, 0, 1};
        Colors[DIRECTION_Y] = {0, .666, 0, 1};
        Colors[DIRECTION_Z] = {0, 0, .666, 1};
        Colors[PLANE_X] = {.666, 0, 0, .38};
        Colors[PLANE_Y] = {0, .666, 0, .38};
        Colors[PLANE_Z] = {0, 0, .666, .38};
        Colors[SELECTION] = {1, .5, .062, .541};
        Colors[INACTIVE] = {.6, .6, .6, .6};
        Colors[TRANSLATION_LINE] = {.666, .666, .666, .666};
        Colors[SCALE_LINE] = {.25, .25, .25, 1};
        Colors[ROTATION_USING_BORDER] = {1, .5, .062, 1};
        Colors[ROTATION_USING_FILL] = {1, .5, .062, .5};
        Colors[HATCHED_AXIS_LINES] = {0, 0, 0, .5};
        Colors[TEXT] = {1, 1, 1, 1};
        Colors[TEXT_SHADOW] = {0, 0, 0, 1};
    }

    float TranslationLineThickness{3}; // Thickness of lines for translation gizmo
    float TranslationLineArrowSize{6}; // Size of arrow at the end of lines for translation gizmo
    float RotationLineThickness{2}; // Thickness of lines for rotation gizmo
    float RotationOuterLineThickness{3}; // Thickness of line surrounding the rotation gizmo
    float ScaleLineThickness{3}; // Thickness of lines for scale gizmo
    float ScaleLineCircleSize{6}; // Size of circle at the end of lines for scale gizmo
    float HatchedAxisLineThickness{6}; // Thickness of hatched axis lines
    float CenterCircleSize{6}; // Size of circle at the center of the translate/scale gizmo

    ImVec4 Colors[COLOR::COUNT];
};

struct Context {
    Style Style;
    MODE Mode;

    mat4 View, Proj, ViewProj, Model;
    mat4 ModelLocal; // orthonormalized model
    mat4 ModelInverse;
    mat4 ModelSource;
    mat4 MVP;
    mat4 MVPLocal; // MVP with full model m whereas MVP's model m might only be translation in case of World space edition

    vec4 ModelScaleOrigin;
    vec4 CameraEye, CameraDir;
    vec4 RayOrigin, RayVector;

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

    float X{0}, Y{0};
    float Width{0}, Height{0};
    float XMax{0}, YMax{0};
    float DisplayRatio{1};
    float GizmoSizeClipSpace{0.1};
    bool IsOrthographic{false};

    Operation Op = Operation(-1);
    int CurrentOp;

    int ActualID{-1}, EditingID{-1};
};

Context g;

int GetMoveType(Operation, vec4 *hit_proportion);
int GetRotateType(Operation);
int GetScaleType(Operation);
} // namespace

namespace ImGuizmo {
bool IsUsing() { return g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID); }

bool IsOver() {
    return (Intersects(g.Op, TRANSLATE) && GetMoveType(g.Op, NULL) != MT_NONE) ||
        (Intersects(g.Op, ROTATE) && GetRotateType(g.Op) != MT_NONE) ||
        (Intersects(g.Op, SCALE) && GetScaleType(g.Op) != MT_NONE) || IsUsing();
}
bool IsOver(Operation op) {
    if (IsUsing()) return true;
    if (Intersects(op, SCALE) && GetScaleType(op) != MT_NONE) return true;
    if (Intersects(op, ROTATE) && GetRotateType(op) != MT_NONE) return true;
    if (Intersects(op, TRANSLATE) && GetMoveType(op, NULL) != MT_NONE) return true;
    return false;
}
void SetRect(float x, float y, float width, float height) {
    g.X = x;
    g.Y = y;
    g.Width = width;
    g.Height = height;
    g.XMax = g.X + g.Width;
    g.YMax = g.Y + g.XMax;
    g.DisplayRatio = width / height;
}
} // namespace ImGuizmo

namespace {
bool IsHoveringWindow() {
    auto &g = *ImGui::GetCurrentContext();
    auto *window = ImGui::FindWindowByName(ImGui::GetWindowDrawList()->_OwnerName);
    if (g.HoveredWindow == window) return true; // Mouse hovering drawlist window
    if (g.HoveredWindow != nullptr) return false; // Another window is hovered
    // Hovering drawlist window rect, while no other window is hovered (for _NoInputs windows)
    if (ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max, false)) return true;
    return false;
}

constexpr ImU32 GetColorU32(int idx) {
    IM_ASSERT(idx < COLOR::COUNT);
    return ImGui::ColorConvertFloat4ToU32(g.Style.Colors[idx]);
}

constexpr ImVec2 WorldToPos(const vec4 &pos_world, const mat4 &m, ImVec2 pos = ImVec2(g.X, g.Y), ImVec2 size = ImVec2(g.Width, g.Height)) {
    vec4 trans{m * vec4{vec3{pos_world}, 1}};
    trans *= (0.5f / trans.w);
    trans += vec4{0.5, 0.5, 0, 0};
    trans.y = 1.f - trans.y;
    trans.x *= size.x;
    trans.y *= size.y;
    trans.x += pos.x;
    trans.y += pos.y;
    return {trans.x, trans.y};
}

void ComputeCameraRay(vec4 &ray_origin, vec4 &ray_dir, ImVec2 pos = {g.X, g.Y}, ImVec2 size = {g.Width, g.Height}) {
    const mat4 view_proj_inv{glm::inverse(g.Proj * g.View)};
    const auto mouse_delta = ImGui::GetIO().MousePos - pos;
    const float mox = (mouse_delta.x / size.x) * 2 - 1;
    const float moy = (1 - (mouse_delta.y / size.y)) * 2 - 1;
    const float z_near = g.Reversed ? (1 - FLT_EPSILON) : 0;
    const float z_far = g.Reversed ? 0 : (1 - FLT_EPSILON);
    ray_origin = view_proj_inv * vec4{mox, moy, z_near, 1};
    ray_origin /= ray_origin.w;

    vec4 ray_end{view_proj_inv * vec4{mox, moy, z_far, 1}};
    ray_end /= ray_end.w;
    ray_dir = glm::normalize(ray_end - ray_origin);
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
    if (g.DisplayRatio < 1.0) clip_space_axis.x *= g.DisplayRatio;
    else clip_space_axis.y /= g.DisplayRatio;
    return sqrtf(clip_space_axis.x * clip_space_axis.x + clip_space_axis.y * clip_space_axis.y);
}

constexpr float GetParallelogram(const vec4 &p0, const vec4 &pa, const vec4 &pb) {
    vec4 pts[]{p0, pa, pb};
    for (uint32_t i = 0; i < 3; i++) {
        pts[i] = g.MVP * vec4{vec3{pts[i]}, 1};
        // check for axis aligned with camera direction
        if (fabsf(pts[i].w) > FLT_EPSILON) pts[i] /= pts[i].w;
    }
    auto seg_a = pts[1] - pts[0];
    seg_a.y /= g.DisplayRatio;

    auto seg_b = pts[2] - pts[0];
    seg_b.y /= g.DisplayRatio;

    const auto seg_a_ortho = glm::normalize(vec4{-seg_a.y, seg_a.x, 0, 0});
    return sqrtf(seg_a.x * seg_a.x + seg_a.y * seg_a.y) * fabsf(glm::dot(vec3{seg_a_ortho}, vec3{seg_b}));
}

vec2 ToGlm(ImVec2 v) { return {v.x, v.y}; }

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 vert_p1, ImVec2 vert_p2) {
    const auto vec = ToGlm(vert_p2 - vert_p1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, ToGlm(p - vert_p1));
    if (t < 0.f) return vert_p1;
    if (t > glm::length(vec)) return vert_p2;
    return vert_p1 + ImVec2{v.x, v.y} * t;
}

vec4 Right(const mat4 &m) { return {m[0]}; }
vec4 Up(const mat4 &m) { return {m[1]}; }
vec4 Dir(const mat4 &m) { return {m[2]}; }
vec4 Pos(const mat4 &m) { return {m[3]}; }
void SetPos(mat4 &m, const vec4 &pos) { m[3] = pos; }

void ComputeContext(const mat4 &view, const mat4 &proj, mat4 &m, MODE mode) {
    g.Mode = mode;
    g.View = view;
    g.Proj = proj;
    g.MouseOver = IsHoveringWindow();

    auto &model_local = g.ModelLocal;
    model_local[0] = glm::normalize(m[0]);
    model_local[1] = glm::normalize(m[1]);
    model_local[2] = glm::normalize(m[2]);
    model_local[3] = m[3];

    if (mode == LOCAL) g.Model = g.ModelLocal;
    else g.Model = glm::translate(mat4{1}, vec3{Pos(m)});
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

    g.ScreenSquareCenter = WorldToPos(vec4{0}, g.MVP);
    g.ScreenSquareMin = g.ScreenSquareCenter - ImVec2{10, 10};
    g.ScreenSquareMax = g.ScreenSquareCenter + ImVec2{10, 10};

    ComputeCameraRay(g.RayOrigin, g.RayVector);
}

constexpr void ComputeColors(ImU32 *colors, int type, Operation op) {
    const auto selection_color = GetColorU32(SELECTION);
    switch (op) {
        case TRANSLATE:
            colors[0] = (type == MT_MOVE_SCREEN) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == int(MT_MOVE_X + i)) ? selection_color : GetColorU32(DIRECTION_X + i);
                colors[i + 4] = (type == int(MT_MOVE_YZ + i)) ? selection_color : GetColorU32(PLANE_X + i);
                colors[i + 4] = (type == MT_MOVE_SCREEN) ? selection_color : colors[i + 4];
            }
            break;
        case ROTATE:
            colors[0] = (type == MT_ROTATE_SCREEN) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == int(MT_ROTATE_X + i)) ? selection_color : GetColorU32(DIRECTION_X + i);
            }
            break;
        case SCALEU:
        case SCALE:
            colors[0] = (type == MT_SCALE_XYZ) ? selection_color : IM_COL32_WHITE;
            for (int i = 0; i < 3; i++) {
                colors[i + 1] = (type == int(MT_SCALE_X + i)) ? selection_color : GetColorU32(DIRECTION_X + i);
            }
            break;
        // note: this internal function is only called with three possible values for op
        default:
            break;
    }
}

const vec3 DirUnary[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

void ComputeTripodAxisAndVisibility(const int axis_i, vec4 &dir_axis, vec4 &dir_plane_x, vec4 &dir_plane_y, bool &below_axis_limit, bool &below_plane_limit, const bool local_coords = false) {
    dir_axis = {DirUnary[axis_i], 0};
    dir_plane_x = {DirUnary[(axis_i + 1) % 3], 0};
    dir_plane_y = {DirUnary[(axis_i + 2) % 3], 0};

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        below_axis_limit = g.BelowAxisLimit[axis_i];
        below_plane_limit = g.BelowPlaneLimit[axis_i];

        dir_axis *= g.AxisFactor[axis_i];
        dir_plane_x *= g.AxisFactor[(axis_i + 1) % 3];
        dir_plane_y *= g.AxisFactor[(axis_i + 2) % 3];
    } else {
        const vec4 origin{0};
        const float len_dir = GetSegmentLengthClipSpace(origin, dir_axis, local_coords);
        const float len_dir_minus = GetSegmentLengthClipSpace(origin, -dir_axis, local_coords);
        const float len_dir_plane_x = GetSegmentLengthClipSpace(origin, dir_plane_x, local_coords);
        const float len_dir_plane_x_minus = GetSegmentLengthClipSpace(origin, -dir_plane_x, local_coords);
        const float len_dir_plane_y = GetSegmentLengthClipSpace(origin, dir_plane_y, local_coords);
        const float len_dir_plane_y_minus = GetSegmentLengthClipSpace(origin, -dir_plane_y, local_coords);

        // For readability, flip gizmo axis for better visibility
        // When false, they always stay along the positive world/local axis
        static constexpr bool AllowFlip = true;
        const float mul_axis = AllowFlip && len_dir < len_dir_minus && fabsf(len_dir - len_dir_minus) > FLT_EPSILON ? -1.f : 1.f;
        const float mul_axis_x = AllowFlip && len_dir_plane_x < len_dir_plane_x_minus && fabsf(len_dir_plane_x - len_dir_plane_x_minus) > FLT_EPSILON ? -1.f : 1.f;
        const float mul_axis_y = AllowFlip && len_dir_plane_y < len_dir_plane_y_minus && fabsf(len_dir_plane_y - len_dir_plane_y_minus) > FLT_EPSILON ? -1.f : 1.f;
        dir_axis *= mul_axis;
        dir_plane_x *= mul_axis_x;
        dir_plane_y *= mul_axis_y;

        const float axis_length_clip_space = GetSegmentLengthClipSpace(origin, dir_axis * g.ScreenFactor, local_coords);
        const float para_surf = GetParallelogram(origin, dir_plane_x * g.ScreenFactor, dir_plane_y * g.ScreenFactor);
        below_plane_limit = (para_surf > g.AxisLimit);
        below_axis_limit = (axis_length_clip_space > g.PlaneLimit);

        // Store values
        g.AxisFactor[axis_i] = mul_axis;
        g.AxisFactor[(axis_i + 1) % 3] = mul_axis_x;
        g.AxisFactor[(axis_i + 2) % 3] = mul_axis_y;
        g.BelowAxisLimit[axis_i] = below_axis_limit;
        g.BelowPlaneLimit[axis_i] = below_plane_limit;
    }
}

constexpr void ComputeSnap(float *value, float snap) {
    if (snap <= FLT_EPSILON) return;

    static constexpr float SnapTension{0.5};
    const float modulo = fmodf(*value, snap);
    const float modulo_ratio = fabsf(modulo) / snap;
    if (modulo_ratio < SnapTension) *value -= modulo;
    else if (modulo_ratio > (1.f - SnapTension)) *value = *value - modulo + snap * ((*value < 0.f) ? -1.f : 1.f);
}
constexpr void ComputeSnap(vec4 &value, const float *snap) {
    for (int i = 0; i < 3; i++) ComputeSnap(&value[i], snap[i]);
}

constexpr float IntersectRayPlane(const vec4 &origin, const vec4 &dir, const vec4 &plan) {
    const float num = glm::dot(vec3{plan}, vec3{origin}) - plan.w;
    const float den = glm::dot(vec3{plan}, vec3{dir});
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -(num / den);
}

constexpr float ComputeAngleOnPlan() {
    vec4 perp{glm::normalize(vec4{glm::cross(vec3{g.RotationVectorSource}, vec3{g.TranslationPlan}), 0})};

    const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
    const auto pos_local = glm::normalize(g.RayOrigin + g.RayVector * len - Pos(g.Model));
    float acos_angle = std::clamp(glm::dot(pos_local, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (glm::dot(pos_local, perp) < 0.f ? 1.f : -1.f);
}

// Scale a bit so translate axes don't touch when in universal.
constexpr float RotationDisplayScale{1.2};

void DrawRotationGizmo(Operation op, int type) {
    if (!Intersects(op, ROTATE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, ROTATE);

    vec4 cam_to_model = g.IsOrthographic ? -Dir(glm::inverse(g.View)) : glm::normalize(Pos(g.Model) - g.CameraEye);
    cam_to_model = g.ModelInverse * vec4{vec3{cam_to_model}, 0};

    static constexpr int HalfCircleSegmentCount{64};
    static constexpr float ScreenRotateSize{0.06};
    g.RadiusSquareCenter = ScreenRotateSize * g.Height;

    auto *draw_list = ImGui::GetWindowDrawList();
    bool hasRSC = Intersects(op, ROTATE_SCREEN);
    for (int axis = 0; axis < 3; axis++) {
        if (!Intersects(op, static_cast<Operation>(ROTATE_Z >> axis))) continue;

        const bool using_axis = g.Using && type == MT_ROTATE_Z - axis;
        const int circle_mul = hasRSC && !using_axis ? 1 : 2;
        const int point_count = circle_mul * HalfCircleSegmentCount + 1;
        std::vector<ImVec2> circle_pos(point_count);
        float angle_start = atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]) + M_PI_2;
        for (int i = 0; i < circle_mul * HalfCircleSegmentCount + 1; i++) {
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
    if (hasRSC && (!g.Using || type == MT_ROTATE_SCREEN)) {
        draw_list->AddCircle(WorldToPos(Pos(g.Model), g.ViewProj), g.RadiusSquareCenter, colors[0], 64, g.Style.RotationOuterLineThickness);
    }

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotateType(type)) {
        ImVec2 circle_pos[HalfCircleSegmentCount + 1];
        circle_pos[0] = WorldToPos(Pos(g.Model), g.ViewProj);
        for (unsigned int i = 1; i < HalfCircleSegmentCount + 1; i++) {
            const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
            const mat4 rotate{glm::rotate(mat4{1}, ng, vec3{g.TranslationPlan})};
            const auto pos = rotate * vec4{vec3{g.RotationVectorSource}, 1} * g.ScreenFactor * RotationDisplayScale;
            circle_pos[i] = WorldToPos(pos + Pos(g.Model), g.ViewProj);
        }
        draw_list->AddConvexPolyFilled(circle_pos, HalfCircleSegmentCount + 1, GetColorU32(ROTATION_USING_FILL));
        draw_list->AddPolyline(circle_pos, HalfCircleSegmentCount + 1, GetColorU32(ROTATION_USING_BORDER), true, g.Style.RotationLineThickness);

        static constexpr const char *RotationInfoMask[]{"X : %5.2f deg %5.2f rad", "Y : %5.2f deg %5.2f rad", "Z : %5.2f deg %5.2f rad", "Screen : %5.2f deg %5.2f rad"};
        char tmps[512];
        ImFormatString(tmps, sizeof(tmps), RotationInfoMask[type - MT_ROTATE_X], (g.RotationAngle / M_PI) * 180.f, g.RotationAngle);

        const auto dest_pos = circle_pos[1];
        draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(TEXT), tmps);
    }
}

void DrawHatchedAxis(const vec4 &axis) {
    if (g.Style.HatchedAxisLineThickness <= 0.0f) return;

    for (int j = 1; j < 10; j++) {
        const auto base = WorldToPos(axis * 0.05f * float(j * 2) * g.ScreenFactor, g.MVP);
        const auto end = WorldToPos(axis * 0.05f * float(j * 2 + 1) * g.ScreenFactor, g.MVP);
        ImGui::GetWindowDrawList()->AddLine(base, end, GetColorU32(HATCHED_AXIS_LINES), g.Style.HatchedAxisLineThickness);
    }
}

constexpr int TranslationInfoIndex[]{0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2};
constexpr const char *ScaleInfoMask[]{"X : %5.2f", "Y : %5.2f", "Z : %5.2f", "XYZ : %5.2f"};

void DrawScaleGizmo(Operation op, int type) {
    if (!Intersects(op, SCALE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, SCALE);

    auto *draw_list = ImGui::GetWindowDrawList();
    vec4 scale_display{1};
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) scale_display = g.Scale;

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<Operation>(SCALE_X << i))) continue;

        if (!g.Using || type == MT_SCALE_X + i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit, below_plane_limit;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

            // draw axis
            if (below_axis_limit) {
                bool has_translate_on_axis = Contains(op, static_cast<Operation>(TRANSLATE_X << i));
                float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                const auto base = WorldToPos(dir_axis * 0.1f * g.ScreenFactor, g.MVP);
                const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVP);
                if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
                    const auto line_color = GetColorU32(SCALE_LINE);
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

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScaleType(type)) {
        const int component_info_i = (type - MT_SCALE_X) * 3;
        char tmps[512];
        ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_SCALE_X], scale_display[TranslationInfoIndex[component_info_i]]);

        const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
        draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(TEXT), tmps);
    }
}

void DrawScaleUniveralGizmo(Operation op, int type) {
    if (!Intersects(op, SCALEU)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, SCALEU);

    auto *draw_list = ImGui::GetWindowDrawList();
    vec4 scale_display{1};
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) scale_display = g.Scale;

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<Operation>(SCALE_XU << i))) continue;

        if (!g.Using || type == MT_SCALE_X + i) {
            vec4 dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit, below_plane_limit;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

            // draw axis
            if (below_axis_limit) {
                const bool has_translate_on_axis = Contains(op, static_cast<Operation>(TRANSLATE_X << i));
                const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVPLocal);
                draw_list->AddCircleFilled(end, 12.f, colors[i + 1]);
            }
        }
    }

    draw_list->AddCircle(g.ScreenSquareCenter, 20.f, colors[0], 32, g.Style.CenterCircleSize);

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScaleType(type)) {
        const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
        char tmps[512];
        const int component_info_i = (type - MT_SCALE_X) * 3;
        ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_SCALE_X], scale_display[TranslationInfoIndex[component_info_i]]);
        draw_list->AddText(ImVec2(dest_pos.x + 15, dest_pos.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(ImVec2(dest_pos.x + 14, dest_pos.y + 14), GetColorU32(TEXT), tmps);
    }
}

constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

void DrawTranslationGizmo(Operation op, int type) {
    auto *draw_list = ImGui::GetWindowDrawList();
    if (!draw_list || !Intersects(op, TRANSLATE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, TRANSLATE);

    const auto origin = WorldToPos(Pos(g.Model), g.ViewProj);
    bool below_axis_limit = false, below_plane_limit = false;
    for (int i = 0; i < 3; ++i) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        if (!g.Using || (g.Using && type == MT_MOVE_X + i)) {
            // draw axis
            if (below_axis_limit && Intersects(op, static_cast<Operation>(TRANSLATE_X << i))) {
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
        if (!g.Using || (g.Using && type == MT_MOVE_YZ + i)) {
            if (below_plane_limit && Contains(op, TranslatePlans[i])) {
                ImVec2 quad_pts_screen[4];
                for (int j = 0; j < 4; ++j) {
                    const auto corner_pos_world = (dir_plane_x * QuadUV[j * 2] + dir_plane_y * QuadUV[j * 2 + 1]) * g.ScreenFactor;
                    quad_pts_screen[j] = WorldToPos(corner_pos_world, g.MVP);
                }
                draw_list->AddPolyline(quad_pts_screen, 4, GetColorU32(DIRECTION_X + i), true, 1.0f);
                draw_list->AddConvexPolyFilled(quad_pts_screen, 4, colors[i + 4]);
            }
        }
    }

    draw_list->AddCircleFilled(g.ScreenSquareCenter, g.Style.CenterCircleSize, colors[0], 32);

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsTranslateType(type)) {
        const auto translation_line_color = GetColorU32(TRANSLATION_LINE);
        const auto source_pos_screen = WorldToPos(g.MatrixOrigin, g.ViewProj);
        const auto dest_pos = WorldToPos(Pos(g.Model), g.ViewProj);
        const auto dif = glm::normalize(vec4{dest_pos.x - source_pos_screen.x, dest_pos.y - source_pos_screen.y, 0, 0}) * 5.f;
        draw_list->AddCircle(source_pos_screen, 6.f, translation_line_color);
        draw_list->AddCircle(dest_pos, 6.f, translation_line_color);
        draw_list->AddLine({source_pos_screen.x + dif.x, source_pos_screen.y + dif.y}, {dest_pos.x - dif.x, dest_pos.y - dif.y}, translation_line_color, 2.f);

        const auto delta_info = Pos(g.Model) - g.MatrixOrigin;
        const int component_info_i = (type - MT_MOVE_X) * 3;
        static constexpr const char *TranslationInfoMask[]{"X : %5.3f", "Y : %5.3f", "Z : %5.3f", "Y : %5.3f Z : %5.3f", "X : %5.3f Z : %5.3f", "X : %5.3f Y : %5.3f", "X : %5.3f Y : %5.3f Z : %5.3f"};

        char tmps[512];
        ImFormatString(tmps, sizeof(tmps), TranslationInfoMask[type - MT_MOVE_X], delta_info[TranslationInfoIndex[component_info_i]], delta_info[TranslationInfoIndex[component_info_i + 1]], delta_info[TranslationInfoIndex[component_info_i + 2]]);
        draw_list->AddText(ImVec2(dest_pos.x + 15, dest_pos.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(ImVec2(dest_pos.x + 14, dest_pos.y + 14), GetColorU32(TEXT), tmps);
    }
}

constexpr vec4 BuildPlan(const vec4 &p_point1, const vec4 &p_normal) {
    const auto normal = glm::normalize(p_normal);
    return {vec3{normal}, glm::dot(normal, p_point1)};
}

constexpr float SelectDistSq = 12 * 12;

int GetScaleType(Operation op) {
    if (g.Using) return MT_NONE;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    int type = mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
            mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y &&
            Contains(op, SCALE) ?
        MT_SCALE_XYZ :
        MT_NONE;

    // compute
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<Operation>(SCALE_X << i))) continue;
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
        dir_axis = g.ModelLocal * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.ModelLocal * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.ModelLocal * vec4{vec3{dir_plane_y}, 0};

        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, BuildPlan(Pos(g.ModelLocal), dir_axis));
        const float start_offset = Contains(op, static_cast<Operation>(TRANSLATE_X << i)) ? 1.0f : 0.1f;
        const float end_offset = Contains(op, static_cast<Operation>(TRANSLATE_X << i)) ? 1.4f : 1.0f;
        const auto pos_plan = g.RayOrigin + g.RayVector * len;
        const auto pos_plan_screen = WorldToPos(pos_plan, g.ViewProj);
        const auto axis_start_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * start_offset, g.ViewProj);
        const auto axis_end_screen = WorldToPos(Pos(g.ModelLocal) + dir_axis * g.ScreenFactor * end_offset, g.ViewProj);
        const auto closest_on_axis = PointOnSegment(pos_plan_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_plan_screen) < SelectDistSq) type = MT_SCALE_X + i; // pixel size
    }

    // universal
    const vec4 delta_screen{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0};
    if (float dist = glm::length(delta_screen); dist >= 17.0f && dist < 23.0f && Contains(op, SCALEU)) type = MT_SCALE_XYZ;

    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<Operation>(SCALE_XU << i))) continue;

        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

        // draw axis
        if (below_axis_limit) {
            const bool has_translate_on_axis = Contains(op, static_cast<Operation>(TRANSLATE_X << i));
            const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
            const auto end = WorldToPos((dir_axis * marker_scale) * g.ScreenFactor, g.MVPLocal);
            if (ImLengthSqr(end - mouse_pos) < SelectDistSq) type = MT_SCALE_X + i;
        }
    }
    return type;
}

int GetRotateType(Operation op) {
    if (g.Using) return MT_NONE;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    const vec4 delta_screen{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0};
    const auto dist = glm::length(delta_screen);
    int type = Intersects(op, ROTATE_SCREEN) && dist >= (g.RadiusSquareCenter - 4.0f) && dist < (g.RadiusSquareCenter + 4.0f) ?
        MT_ROTATE_SCREEN :
        MT_NONE;

    const vec4 plan_normals[]{Right(g.Model), Up(g.Model), Dir(g.Model)};
    const auto model_view_pos = g.View * vec4{vec3{Pos(g.Model)}, 1};
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<Operation>(ROTATE_X << i))) continue;

        const auto pickup_plan = BuildPlan(Pos(g.Model), plan_normals[i]);
        const auto len = IntersectRayPlane(g.RayOrigin, g.RayVector, pickup_plan);
        const auto intersect_world_pos = g.RayOrigin + g.RayVector * len;
        const auto intersect_view_pos = g.View * vec4{vec3{intersect_world_pos}, 1};
        if (ImAbs(model_view_pos.z) - ImAbs(intersect_view_pos.z) < -FLT_EPSILON) continue;

        auto ideal_pos_circle = g.ModelInverse * vec4{vec3{glm::normalize(intersect_world_pos - Pos(g.Model))}, 0};
        const auto ideal_circle_pos_screen = WorldToPos(ideal_pos_circle * RotationDisplayScale * g.ScreenFactor, g.MVP);
        const auto distance_screen = ideal_circle_pos_screen - mouse_pos;
        if (glm::length(vec2(distance_screen.x, distance_screen.y)) < 8) type = MT_ROTATE_X + i; // pixel size
    }

    return type;
}

int GetMoveType(Operation op, vec4 *hit_proportion = nullptr) {
    if (g.Using || !g.MouseOver || !Intersects(op, TRANSLATE)) return MT_NONE;

    auto &io = ImGui::GetIO();
    int type = MT_NONE;
    if (io.MousePos.x >= g.ScreenSquareMin.x && io.MousePos.x <= g.ScreenSquareMax.x &&
        io.MousePos.y >= g.ScreenSquareMin.y && io.MousePos.y <= g.ScreenSquareMax.y &&
        Contains(op, TRANSLATE)) {
        type = MT_MOVE_SCREEN;
    }

    const ImVec2 pos_screen{io.MousePos - ImVec2{g.X, g.Y}};
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        vec4 dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        dir_axis = g.Model * vec4{vec3{dir_axis}, 0};
        dir_plane_x = g.Model * vec4{vec3{dir_plane_x}, 0};
        dir_plane_y = g.Model * vec4{vec3{dir_plane_y}, 0};

        const auto axis_start_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor * 0.1f, g.ViewProj) - ImVec2{g.X, g.Y};
        const auto axis_end_screen = WorldToPos(Pos(g.Model) + dir_axis * g.ScreenFactor, g.ViewProj) - ImVec2{g.X, g.Y};
        const auto closest_on_axis = PointOnSegment(pos_screen, axis_start_screen, axis_end_screen);
        if (ImLengthSqr(closest_on_axis - pos_screen) < SelectDistSq && Intersects(op, static_cast<Operation>(TRANSLATE_X << i))) type = MT_MOVE_X + i;

        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, BuildPlan(Pos(g.Model), dir_axis));
        const auto pos_plan = g.RayOrigin + g.RayVector * len;
        const float dx = glm::dot(vec3{dir_plane_x}, vec3{pos_plan - Pos(g.Model)} / g.ScreenFactor);
        const float dy = glm::dot(vec3{dir_plane_y}, vec3{pos_plan - Pos(g.Model)} / g.ScreenFactor);
        if (below_plane_limit && dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3] && Contains(op, TranslatePlans[i])) {
            type = MT_MOVE_YZ + i;
        }

        if (hit_proportion != nullptr) *hit_proportion = {dx, dy, 0, 0};
    }
    return type;
}

bool CanActivate() { return ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive(); }

bool HandleTranslation(mat4 &m, Operation op, int &type, const float *snap) {
    if (!Intersects(op, TRANSLATE) || type != MT_NONE) return false;

    const bool apply_rot_locally = g.Mode == LOCAL || type == MT_MOVE_SCREEN;
    bool modified = false;

    // move
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsTranslateType(g.CurrentOp)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len_signed = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
        const float len = fabsf(len_signed); // near plan
        const auto new_pos = g.RayOrigin + g.RayVector * len;

        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.Model);

        // 1 axis constraint
        if (g.CurrentOp >= MT_MOVE_X && g.CurrentOp <= MT_MOVE_Z) {
            const int axis_i = g.CurrentOp - MT_MOVE_X;
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

        type = g.CurrentOp;
    } else {
        // find new possible way to move
        type = GetMoveType(op);
        if (type != MT_NONE) {
            ImGui::SetNextFrameWantCaptureMouse(true);
            if (CanActivate()) {
                g.Using = true;
                g.EditingID = g.ActualID;
                g.CurrentOp = type;
                vec4 move_plan_normal[]{Right(g.Model), Up(g.Model), Dir(g.Model), Right(g.Model), Up(g.Model), Dir(g.Model), -g.CameraDir};
                const auto cam_to_model = glm::normalize(Pos(g.Model) - g.CameraEye);
                for (unsigned int i = 0; i < 3; i++) {
                    move_plan_normal[i] = glm::normalize(
                        vec4{glm::cross(vec3{move_plan_normal[i]}, glm::cross(vec3(move_plan_normal[i]), vec3{cam_to_model})), 0}
                    );
                }
                // pickup plan
                g.TranslationPlan = BuildPlan(Pos(g.Model), move_plan_normal[type - MT_MOVE_X]);
                const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
                g.TranslationPlanOrigin = g.RayOrigin + g.RayVector * len;
                g.MatrixOrigin = Pos(g.Model);

                g.RelativeOrigin = (g.TranslationPlanOrigin - Pos(g.Model)) * (1.f / g.ScreenFactor);
            }
        }
    }
    return modified;
}

bool HandleScale(mat4 &m, Operation op, int &type, const float *snap) {
    if (type != MT_NONE || !g.MouseOver || (!Intersects(op, SCALE) && !Intersects(op, SCALEU))) return false;

    bool modified = false;
    if (!g.Using) {
        // find new possible way to scale
        type = GetScaleType(op);
        if (type != MT_NONE) ImGui::SetNextFrameWantCaptureMouse(true);
        if (CanActivate() && type != MT_NONE) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentOp = type;
            const vec4 move_plan_normal[]{Up(g.Model), Dir(g.Model), Right(g.Model), Dir(g.Model), Up(g.Model), Right(g.Model), -g.CameraDir};
            g.TranslationPlan = BuildPlan(Pos(g.Model), move_plan_normal[type - MT_SCALE_X]);
            const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
            g.TranslationPlanOrigin = g.RayOrigin + g.RayVector * len;
            g.MatrixOrigin = Pos(g.Model);
            g.Scale = {1, 1, 1, 0};
            g.RelativeOrigin = (g.TranslationPlanOrigin - Pos(g.Model)) * (1.f / g.ScreenFactor);
            g.ScaleOrigin = {glm::length(Right(g.ModelSource)), glm::length(Up(g.ModelSource)), glm::length(Dir(g.ModelSource)), 0};
            g.SaveMousePosX = ImGui::GetIO().MousePos.x;
        }
    }
    // scale
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScaleType(g.CurrentOp)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
        const auto new_pos = g.RayOrigin + g.RayVector * len;
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - Pos(g.ModelLocal);
        // 1 axis constraint
        if (g.CurrentOp >= MT_SCALE_X && g.CurrentOp <= MT_SCALE_Z) {
            int axis_i = g.CurrentOp - MT_SCALE_X;
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
        for (int i = 0; i < 3; i++) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        if (g.ScalePrev != g.Scale) {
            g.ScalePrev = g.Scale;
            modified = true;
        }

        m = g.ModelLocal * glm::scale(mat4{1}, {g.Scale * g.ScaleOrigin});

        if (!ImGui::GetIO().MouseDown[0]) {
            g.Using = false;
            g.Scale = vec4{1, 1, 1, 0};
        }

        type = g.CurrentOp;
    }
    return modified;
}

bool HandleRotation(mat4 &m, Operation op, int &type, const float *snap) {
    if (!Intersects(op, ROTATE) || type != MT_NONE || !g.MouseOver) return false;

    bool apply_rot_locally{g.Mode == LOCAL};
    bool modified{false};
    if (!g.Using) {
        type = GetRotateType(op);

        if (type != MT_NONE) ImGui::SetNextFrameWantCaptureMouse(true);
        if (type == MT_ROTATE_SCREEN) apply_rot_locally = true;

        if (CanActivate() && type != MT_NONE) {
            g.Using = true;
            g.EditingID = g.ActualID;
            g.CurrentOp = type;
            const vec4 rotate_plan_normal[]{Right(g.Model), Up(g.Model), Dir(g.Model), -g.CameraDir};
            g.TranslationPlan = apply_rot_locally ?
                BuildPlan(Pos(g.Model), rotate_plan_normal[type - MT_ROTATE_X]) :
                BuildPlan(Pos(g.ModelSource), {DirUnary[type - MT_ROTATE_X], 0});

            const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
            g.RotationVectorSource = glm::normalize(g.RayOrigin + g.RayVector * len - Pos(g.Model));
            g.RotationAngleOrigin = ComputeAngleOnPlan();
        }
    }

    // rotation
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotateType(g.CurrentOp)) {
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
            SetPos(res, {vec4{0}});
            m = delta_rot * res;
            SetPos(m, Pos(g.ModelSource));
        }

        if (!ImGui::GetIO().MouseDown[0]) {
            g.Using = false;
            g.EditingID = -1;
        }
        type = g.CurrentOp;
    }
    return modified;
}
} // namespace

namespace ImGuizmo {
bool Manipulate(const mat4 &view, const mat4 &proj, Operation op, MODE mode, mat4 &m, const float *snap) {
    // Scale is always local or m will be skewed when applying world scale or oriented m
    ComputeContext(view, proj, m, (op & SCALE) ? LOCAL : mode);

    // behind camera
    const auto pos_cam_space = g.MVP * vec4{vec3{0}, 1};
    if (!g.IsOrthographic && pos_cam_space.z < 0.001 && !g.Using) return false;

    int type = MT_NONE;
    const bool manipulated = HandleTranslation(m, op, type, snap) ||
        HandleScale(m, op, type, snap) ||
        HandleRotation(m, op, type, snap);

    g.Op = op;
    DrawRotationGizmo(op, type);
    DrawTranslationGizmo(op, type);
    DrawScaleGizmo(op, type);
    DrawScaleUniveralGizmo(op, type);
    return manipulated;
}
} // namespace ImGuizmo
