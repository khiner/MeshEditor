#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "numeric/mat4.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "ImGuizmo.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <glm/gtx/matrix_decompose.hpp>

#include <algorithm>
#include <vector>

namespace {
struct matrix_t;
struct vec_t {
    vec4 _v;

    vec_t() = default;
    vec_t(const vec4 &v) : _v(v) {}

    void Set(float v) { _v = vec4{v}; }
    void Set(float _x, float _y, float _z = 0.f, float _w = 0.f) { _v = vec4{_x, _y, _z, _w}; }

    vec_t &operator-=(const vec_t &v) {
        _v -= v._v;
        return *this;
    }
    vec_t &operator+=(const vec_t &v) {
        _v += v._v;
        return *this;
    }
    vec_t &operator*=(const vec_t &v) {
        _v *= v._v;
        return *this;
    }
    vec_t &operator*=(float v) {
        _v *= v;
        return *this;
    }

    vec_t operator*(float f) const { return {_v * f}; }
    vec_t operator-() const { return {-_v}; }
    vec_t operator-(const vec_t &v) const { return {_v - v._v}; }
    vec_t operator+(const vec_t &v) const { return {_v + v._v}; }
    vec_t operator*(const vec_t &v) const { return {_v * v._v}; }

    const vec_t &operator+() const { return *this; }
    float Length() const { return glm::length(_v); };
    float LengthSq() const { return glm::dot(_v, _v); };
    vec_t Normalize(const vec_t &v) {
        _v = glm::normalize(v._v);
        return *this;
    }

    void Cross(const vec_t &v) { _v = vec4(glm::cross(vec3(_v), vec3(v._v)), 0); }
    void Cross(const vec_t &v1, const vec_t &v2) { _v = vec4(glm::cross(vec3(v1._v), vec3(v2._v)), 0); }

    float Dot(const vec_t &v) const { return glm::dot(_v, v._v); }
    float Dot3(const vec_t &v) const { return glm::dot(vec3(_v), vec3(v._v)); }

    void Transform(const matrix_t &);
    void Transform(const vec_t &s, const matrix_t &);

    void TransformVector(const matrix_t &);
    void TransformPoint(const matrix_t &);

    float &operator[](size_t index) { return _v[index]; }
    const float &operator[](size_t index) const { return _v[index]; }
    bool operator!=(const vec_t &other) const { return _v != other._v; }
};

constexpr vec_t MakeVect(float _x, float _y, float _z = 0.f, float _w = 0.f) { return vec_t({_x, _y, _z, _w}); }
constexpr vec_t MakeVect(ImVec2 v) { return vec_t({v.x, v.y, 0, 0}); }

constexpr vec_t Normalized(const vec_t &v) { return {glm::normalize(v._v)}; }
constexpr vec_t Cross(const vec_t &v1, const vec_t &v2) { return {vec4(glm::cross(vec3(v1._v), vec3(v2._v)), 0)}; }
constexpr float Dot(const vec_t &v1, const vec_t &v2) { return glm::dot(vec3(v1._v), vec3(v2._v)); }

constexpr vec_t BuildPlan(const vec_t &p_point1, const vec_t &p_normal) {
    const auto normal = Normalized(p_normal);
    return {vec4(vec3(normal._v), normal.Dot(p_point1))};
}

struct matrix_t {
public:
    union {
        struct
        {
            vec_t right, up, dir, pos;
        } v;
        mat4 m4;
    };

    void Translation(float _x, float _y, float _z) { this->Translation(MakeVect(_x, _y, _z)); }

    void Translation(const vec_t &vt) {
        v.right.Set(1.f, 0.f, 0.f, 0.f);
        v.up.Set(0.f, 1.f, 0.f, 0.f);
        v.dir.Set(0.f, 0.f, 1.f, 0.f);
        v.pos.Set(vt._v.x, vt._v.y, vt._v.z, 1.f);
    }

    void Scale(float _x, float _y, float _z) { m4 = glm::scale(mat4{1}, vec3{_x, _y, _z}); }
    void Scale(const vec_t &s) { m4 = glm::scale(mat4{1}, vec3{s._v}); }

    matrix_t operator*(const matrix_t &m) const {
        matrix_t tmp{};
        tmp.m4 = m.m4 * m4;
        return tmp;
    }

    void Multiply(const matrix_t &m) { m4 *= m.m4; }
    void Multiply(const matrix_t &m1, const matrix_t &m2) { m4 = m2.m4 * m1.m4; }
    void Inverse(const matrix_t &m) { m4 = glm::inverse(m.m4); }
    void SetToIdentity() { m4 = mat4{1}; }
    void Transpose() { m4 = glm::transpose(m4); }
    void RotationAxis(const vec_t &axis, float angle) { m4 = glm::rotate(mat4{1}, angle, vec3{axis._v}); }
    void OrthoNormalize() {
        m4[0] = glm::normalize(m4[0]);
        m4[1] = glm::normalize(m4[1]);
        m4[2] = glm::normalize(m4[2]);
    }
};

void vec_t::Transform(const matrix_t &m) { _v = m.m4 * _v; }
void vec_t::Transform(const vec_t &s, const matrix_t &m) {
    *this = s;
    Transform(m);
}
void vec_t::TransformPoint(const matrix_t &m) { _v = m.m4 * vec4(vec3(_v), 1); }
void vec_t::TransformVector(const matrix_t &m) { _v = m.m4 * vec4(vec3(_v), 0); }

enum MOVETYPE {
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

constexpr bool Intersects(OPERATION lhs, OPERATION rhs) { return (lhs & rhs) != 0; }
constexpr bool Contains(OPERATION lhs, OPERATION rhs) { return (lhs & rhs) == rhs; }
constexpr OPERATION operator|(OPERATION lhs, OPERATION rhs) { return static_cast<OPERATION>(static_cast<int>(lhs) | static_cast<int>(rhs)); }

// Matches MT_MOVE_AB order
constexpr OPERATION TranslatePlans[]{TRANSLATE_Y | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Z, TRANSLATE_X | TRANSLATE_Y};

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

    matrix_t View;
    matrix_t Proj;
    matrix_t Model;
    matrix_t ModelLocal; // orthonormalized model
    matrix_t ModelInverse;
    matrix_t ModelSource;
    matrix_t MVP;
    matrix_t MVPLocal; // MVP with full model matrix whereas MVP's model matrix might only be translation in case of World space edition
    matrix_t ViewProj;

    vec_t ModelScaleOrigin;
    vec_t CameraEye, CameraDir;
    vec_t RayOrigin, RayVector;

    float RadiusSquareCenter;
    ImVec2 ScreenSquareCenter, ScreenSquareMin, ScreenSquareMax;

    float ScreenFactor;
    vec_t RelativeOrigin;

    bool Using{false};
    bool MouseOver{false};
    bool Reversed{false}; // reversed projection matrix

    vec_t TranslationPlan, TranslationPlanOrigin, TranslationPrevDelta;
    vec_t MatrixOrigin;

    vec_t RotationVectorSource;
    float RotationAngle, RotationAngleOrigin;

    vec_t Scale, ScaleOrigin, ScalePrev;
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

    OPERATION Op = OPERATION(-1);
    int CurrentOp;

    int ActualID{-1}, EditingID{-1};
};

Context g;

int GetMoveType(OPERATION, vec_t *hit_proportion);
int GetRotateType(OPERATION);
int GetScaleType(OPERATION);
} // namespace

namespace ImGuizmo {
bool IsUsing() { return g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID); }

bool IsOver() {
    return (Intersects(g.Op, TRANSLATE) && GetMoveType(g.Op, NULL) != MT_NONE) ||
        (Intersects(g.Op, ROTATE) && GetRotateType(g.Op) != MT_NONE) ||
        (Intersects(g.Op, SCALE) && GetScaleType(g.Op) != MT_NONE) || IsUsing();
}
bool IsOver(OPERATION op) {
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

constexpr ImVec2 WorldToPos(const vec_t &pos_world, const matrix_t &mat, ImVec2 pos = ImVec2(g.X, g.Y), ImVec2 size = ImVec2(g.Width, g.Height)) {
    vec_t trans = pos_world;
    trans.TransformPoint(mat);
    trans *= 0.5f / trans._v.w;
    trans += MakeVect(0.5, 0.5);
    trans._v.y = 1.f - trans._v.y;
    trans._v.x *= size.x;
    trans._v.y *= size.y;
    trans._v.x += pos.x;
    trans._v.y += pos.y;
    return {trans._v.x, trans._v.y};
}

void ComputeCameraRay(vec_t &ray_origin, vec_t &ray_dir, ImVec2 pos = {g.X, g.Y}, ImVec2 size = {g.Width, g.Height}) {
    matrix_t view_proj_inv{};
    view_proj_inv.Inverse(g.View * g.Proj);

    const auto mouse_delta = ImGui::GetIO().MousePos - pos;
    const float mox = (mouse_delta.x / size.x) * 2 - 1;
    const float moy = (1 - (mouse_delta.y / size.y)) * 2 - 1;
    const float z_near = g.Reversed ? (1 - FLT_EPSILON) : 0;
    const float z_far = g.Reversed ? 0 : (1 - FLT_EPSILON);
    ray_origin.Transform(MakeVect(mox, moy, z_near, 1), view_proj_inv);
    ray_origin *= 1.f / ray_origin._v.w;

    vec_t ray_end;
    ray_end.Transform(MakeVect(mox, moy, z_far, 1), view_proj_inv);
    ray_end *= 1.f / ray_end._v.w;
    ray_dir = Normalized(ray_end - ray_origin);
}

constexpr float GetSegmentLengthClipSpace(const vec_t &start, const vec_t &end, const bool local_coords = false) {
    auto segment_start = start;
    const auto &mvp = local_coords ? g.MVPLocal : g.MVP;
    segment_start.TransformPoint(mvp);
    // check for axis aligned with camera direction
    if (fabsf(segment_start._v.w) > FLT_EPSILON) segment_start *= 1.f / segment_start._v.w;

    auto segment_end = end;
    segment_end.TransformPoint(mvp);
    // check for axis aligned with camera direction
    if (fabsf(segment_end._v.w) > FLT_EPSILON) segment_end *= 1.f / segment_end._v.w;

    auto clip_space_axis = segment_end - segment_start;
    if (g.DisplayRatio < 1.0) clip_space_axis._v.x *= g.DisplayRatio;
    else clip_space_axis._v.y /= g.DisplayRatio;
    return sqrtf(clip_space_axis._v.x * clip_space_axis._v.x + clip_space_axis._v.y * clip_space_axis._v.y);
}

constexpr float GetParallelogram(const vec_t &p0, const vec_t &pa, const vec_t &pb) {
    vec_t pts[]{p0, pa, pb};
    for (uint32_t i = 0; i < 3; i++) {
        pts[i].TransformPoint(g.MVP);
        // check for axis aligned with camera direction
        if (fabsf(pts[i]._v.w) > FLT_EPSILON) pts[i] *= 1.f / pts[i]._v.w;
    }
    auto seg_a = pts[1] - pts[0];
    seg_a._v.y /= g.DisplayRatio;

    auto seg_b = pts[2] - pts[0];
    seg_b._v.y /= g.DisplayRatio;

    const auto seg_a_ortho = Normalized(MakeVect(-seg_a._v.y, seg_a._v.x));
    return sqrtf(seg_a._v.x * seg_a._v.x + seg_a._v.y * seg_a._v.y) * fabsf(seg_a_ortho.Dot3(seg_b));
}

constexpr vec_t PointOnSegment(const vec_t &p, const vec_t &vert_p1, const vec_t &vert_p2) {
    vec_t v;
    v.Normalize(vert_p2 - vert_p1);
    const float t = v.Dot3(p - vert_p1);
    if (t < 0.f) return vert_p1;
    if (t > (vert_p2 - vert_p1).Length()) return vert_p2;
    return vert_p1 + v * t;
}

void ComputeContext(const float *view, const float *projection, float *matrix, MODE mode) {
    g.Mode = mode;
    g.View = *(matrix_t *)view;
    g.Proj = *(matrix_t *)projection;
    g.MouseOver = IsHoveringWindow();

    g.ModelLocal = *(matrix_t *)matrix;
    g.ModelLocal.OrthoNormalize();

    if (mode == LOCAL) g.Model = g.ModelLocal;
    else g.Model.Translation(((matrix_t *)matrix)->v.pos);
    g.ModelSource = *(matrix_t *)matrix;
    g.ModelScaleOrigin.Set(g.ModelSource.v.right.Length(), g.ModelSource.v.up.Length(), g.ModelSource.v.dir.Length());

    g.ModelInverse.Inverse(g.Model);
    g.ViewProj = g.View * g.Proj;
    g.MVP = g.Model * g.ViewProj;
    g.MVPLocal = g.ModelLocal * g.ViewProj;

    matrix_t view_inv{};
    view_inv.Inverse(g.View);
    g.CameraDir = view_inv.v.dir;
    g.CameraEye = view_inv.v.pos;

    // projection reverse
    vec_t near_pos, far_pos;
    near_pos.Transform(MakeVect(0, 0, 1.f, 1.f), g.Proj);
    far_pos.Transform(MakeVect(0, 0, 2.f, 1.f), g.Proj);
    g.Reversed = near_pos._v.z / near_pos._v.w > far_pos._v.z / far_pos._v.w;

    // compute scale from the size of camera right vector projected on screen at the matrix pos
    auto right_point = view_inv.v.right;
    right_point.TransformPoint(g.ViewProj);
    g.ScreenFactor = g.GizmoSizeClipSpace / (right_point._v.x / right_point._v.w - g.MVP.v.pos._v.x / g.MVP.v.pos._v.w);

    auto right_view_inv = view_inv.v.right;
    right_view_inv.TransformVector(g.ModelInverse);
    const float right_len = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f), right_view_inv);
    g.ScreenFactor = g.GizmoSizeClipSpace / right_len;

    g.ScreenSquareCenter = WorldToPos(MakeVect(0.f, 0.f), g.MVP);
    g.ScreenSquareMin = g.ScreenSquareCenter - ImVec2{10, 10};
    g.ScreenSquareMax = g.ScreenSquareCenter + ImVec2{10, 10};

    ComputeCameraRay(g.RayOrigin, g.RayVector);
}

constexpr void ComputeColors(ImU32 *colors, int type, OPERATION op) {
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

const vec_t DirUnary[]{MakeVect(1, 0, 0), MakeVect(0, 1, 0), MakeVect(0, 0, 1)};

void ComputeTripodAxisAndVisibility(const int axis_i, vec_t &dir_axis, vec_t &dir_plane_x, vec_t &dir_plane_y, bool &below_axis_limit, bool &below_plane_limit, const bool local_coords = false) {
    dir_axis = DirUnary[axis_i];
    dir_plane_x = DirUnary[(axis_i + 1) % 3];
    dir_plane_y = DirUnary[(axis_i + 2) % 3];

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) {
        // When using, use stored factors so the gizmo doesn't flip when we translate
        below_axis_limit = g.BelowAxisLimit[axis_i];
        below_plane_limit = g.BelowPlaneLimit[axis_i];

        dir_axis *= g.AxisFactor[axis_i];
        dir_plane_x *= g.AxisFactor[(axis_i + 1) % 3];
        dir_plane_y *= g.AxisFactor[(axis_i + 2) % 3];
    } else {
        const float len_dir = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), dir_axis, local_coords);
        const float len_dir_minus = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), -dir_axis, local_coords);
        const float len_dir_plane_x = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), dir_plane_x, local_coords);
        const float len_dir_plane_x_minus = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), -dir_plane_x, local_coords);
        const float len_dir_plane_y = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), dir_plane_y, local_coords);
        const float len_dir_plane_y_minus = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), -dir_plane_y, local_coords);

        // For readability, flip gizmo axis for better visibility
        // When false, they always stay along the positive world/local axis
        static constexpr bool AllowFlip = true;
        const float mul_axis = AllowFlip && len_dir < len_dir_minus && fabsf(len_dir - len_dir_minus) > FLT_EPSILON ? -1.f : 1.f;
        const float mul_axis_x = AllowFlip && len_dir_plane_x < len_dir_plane_x_minus && fabsf(len_dir_plane_x - len_dir_plane_x_minus) > FLT_EPSILON ? -1.f : 1.f;
        const float mul_axis_y = AllowFlip && len_dir_plane_y < len_dir_plane_y_minus && fabsf(len_dir_plane_y - len_dir_plane_y_minus) > FLT_EPSILON ? -1.f : 1.f;
        dir_axis *= mul_axis;
        dir_plane_x *= mul_axis_x;
        dir_plane_y *= mul_axis_y;

        const float axis_length_clip_space = GetSegmentLengthClipSpace(MakeVect(0.f, 0.f, 0.f), dir_axis * g.ScreenFactor, local_coords);
        const float para_surf = GetParallelogram(MakeVect(0.f, 0.f, 0.f), dir_plane_x * g.ScreenFactor, dir_plane_y * g.ScreenFactor);
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
constexpr void ComputeSnap(vec_t &value, const float *snap) {
    for (int i = 0; i < 3; i++) ComputeSnap(&value[i], snap[i]);
}

constexpr float IntersectRayPlane(const vec_t &rOrigin, const vec_t &rVector, const vec_t &plan) {
    const float num = plan.Dot3(rOrigin) - plan._v.w;
    const float den = plan.Dot3(rVector);
    // if normal is orthogonal to vector, can't intersect
    return fabsf(den) < FLT_EPSILON ? -1 : -(num / den);
}

constexpr float ComputeAngleOnPlan() {
    vec_t perp;
    perp.Cross(g.RotationVectorSource, g.TranslationPlan);
    perp = Normalized(perp);

    const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
    const auto pos_local = Normalized(g.RayOrigin + g.RayVector * len - g.Model.v.pos);
    float acos_angle = std::clamp(Dot(pos_local, g.RotationVectorSource), -1.f, 1.f);
    return acosf(acos_angle) * (Dot(pos_local, perp) < 0.f ? 1.f : -1.f);
}

// Scale a bit so translate axes don't touch when in universal.
constexpr float RotationDisplayScale{1.2};

void DrawRotationGizmo(OPERATION op, int type) {
    if (!Intersects(op, ROTATE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, ROTATE);

    vec_t cam_to_model;
    if (g.IsOrthographic) {
        matrix_t view_inv{};
        view_inv.Inverse(*(matrix_t *)&g.View);
        cam_to_model = -view_inv.v.dir;
    } else {
        cam_to_model = Normalized(g.Model.v.pos - g.CameraEye);
    }

    cam_to_model.TransformVector(g.ModelInverse);

    static constexpr int HalfCircleSegmentCount{64};
    static constexpr float ScreenRotateSize{0.06};
    g.RadiusSquareCenter = ScreenRotateSize * g.Height;

    auto *draw_list = ImGui::GetWindowDrawList();
    bool hasRSC = Intersects(op, ROTATE_SCREEN);
    for (int axis = 0; axis < 3; axis++) {
        if (!Intersects(op, static_cast<OPERATION>(ROTATE_Z >> axis))) continue;

        const bool using_axis = g.Using && type == MT_ROTATE_Z - axis;
        const int circle_mul = hasRSC && !using_axis ? 1 : 2;
        const int point_count = circle_mul * HalfCircleSegmentCount + 1;
        std::vector<ImVec2> circle_pos(point_count);
        float angle_start = atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]) + M_PI_2;
        for (int i = 0; i < circle_mul * HalfCircleSegmentCount + 1; i++) {
            const float ng = angle_start + float(circle_mul) * M_PI * (float(i) / float(circle_mul * HalfCircleSegmentCount));
            const auto axis_pos = MakeVect(cosf(ng), sinf(ng), 0.f);
            const auto pos = MakeVect(axis_pos[axis], axis_pos[(axis + 1) % 3], axis_pos[(axis + 2) % 3]) * g.ScreenFactor * RotationDisplayScale;
            circle_pos[i] = WorldToPos(pos, g.MVP);
        }
        if (!g.Using || using_axis) {
            draw_list->AddPolyline(circle_pos.data(), circle_mul * HalfCircleSegmentCount + 1, colors[3 - axis], false, g.Style.RotationLineThickness);
        }
        if (float radius_axis = sqrtf((ImLengthSqr(WorldToPos(g.Model.v.pos, g.ViewProj) - circle_pos[0])));
            radius_axis > g.RadiusSquareCenter) {
            g.RadiusSquareCenter = radius_axis;
        }
    }
    if (hasRSC && (!g.Using || type == MT_ROTATE_SCREEN)) {
        draw_list->AddCircle(WorldToPos(g.Model.v.pos, g.ViewProj), g.RadiusSquareCenter, colors[0], 64, g.Style.RotationOuterLineThickness);
    }

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotateType(type)) {
        ImVec2 circle_pos[HalfCircleSegmentCount + 1];
        circle_pos[0] = WorldToPos(g.Model.v.pos, g.ViewProj);
        for (unsigned int i = 1; i < HalfCircleSegmentCount + 1; i++) {
            const float ng = g.RotationAngle * (float(i - 1) / float(HalfCircleSegmentCount - 1));
            matrix_t rotate{};
            rotate.RotationAxis(g.TranslationPlan, ng);
            vec_t pos = g.RotationVectorSource;
            pos.TransformPoint(rotate);
            pos *= g.ScreenFactor * RotationDisplayScale;
            circle_pos[i] = WorldToPos(pos + g.Model.v.pos, g.ViewProj);
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

void DrawHatchedAxis(const vec_t &axis) {
    if (g.Style.HatchedAxisLineThickness <= 0.0f) return;

    for (int j = 1; j < 10; j++) {
        const auto base = WorldToPos(axis * 0.05f * float(j * 2) * g.ScreenFactor, g.MVP);
        const auto end = WorldToPos(axis * 0.05f * float(j * 2 + 1) * g.ScreenFactor, g.MVP);
        ImGui::GetWindowDrawList()->AddLine(base, end, GetColorU32(HATCHED_AXIS_LINES), g.Style.HatchedAxisLineThickness);
    }
}

constexpr int TranslationInfoIndex[]{0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2};
constexpr const char *ScaleInfoMask[]{"X : %5.2f", "Y : %5.2f", "Z : %5.2f", "XYZ : %5.2f"};

void DrawScaleGizmo(OPERATION op, int type) {
    if (!Intersects(op, SCALE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, SCALE);

    auto *draw_list = ImGui::GetWindowDrawList();
    vec_t scale_display{{1, 1, 1, 1}};
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) scale_display = g.Scale;

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_X << i))) continue;

        if (!g.Using || type == MT_SCALE_X + i) {
            vec_t dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit, below_plane_limit;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

            // draw axis
            if (below_axis_limit) {
                bool has_translate_on_axis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
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

        const auto dest_pos = WorldToPos(g.Model.v.pos, g.ViewProj);
        draw_list->AddText(dest_pos + ImVec2{15, 15}, GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(dest_pos + ImVec2{14, 14}, GetColorU32(TEXT), tmps);
    }
}

void DrawScaleUniveralGizmo(OPERATION op, int type) {
    if (!Intersects(op, SCALEU)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, SCALEU);

    auto *draw_list = ImGui::GetWindowDrawList();
    vec_t scale_display{{1, 1, 1, 1}};
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID)) scale_display = g.Scale;

    for (int i = 0; i < 3; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_XU << i))) continue;

        if (!g.Using || type == MT_SCALE_X + i) {
            vec_t dir_plane_x, dir_plane_y, dir_axis;
            bool below_axis_limit, below_plane_limit;
            ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

            // draw axis
            if (below_axis_limit) {
                const bool has_translate_on_axis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
                const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
                const auto end = WorldToPos((dir_axis * marker_scale * scale_display[i]) * g.ScreenFactor, g.MVPLocal);
                draw_list->AddCircleFilled(end, 12.f, colors[i + 1]);
            }
        }
    }

    draw_list->AddCircle(g.ScreenSquareCenter, 20.f, colors[0], 32, g.Style.CenterCircleSize);

    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScaleType(type)) {
        const auto dest_pos = WorldToPos(g.Model.v.pos, g.ViewProj);
        char tmps[512];
        const int component_info_i = (type - MT_SCALE_X) * 3;
        ImFormatString(tmps, sizeof(tmps), ScaleInfoMask[type - MT_SCALE_X], scale_display[TranslationInfoIndex[component_info_i]]);
        draw_list->AddText(ImVec2(dest_pos.x + 15, dest_pos.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(ImVec2(dest_pos.x + 14, dest_pos.y + 14), GetColorU32(TEXT), tmps);
    }
}

constexpr float QuadMin{0.5}, QuadMax{0.8};
constexpr float QuadUV[]{QuadMin, QuadMin, QuadMin, QuadMax, QuadMax, QuadMax, QuadMax, QuadMin};

void DrawTranslationGizmo(OPERATION op, int type) {
    auto *draw_list = ImGui::GetWindowDrawList();
    if (!draw_list || !Intersects(op, TRANSLATE)) return;

    ImU32 colors[7];
    ComputeColors(colors, type, TRANSLATE);

    const auto origin = WorldToPos(g.Model.v.pos, g.ViewProj);
    bool below_axis_limit = false, below_plane_limit = false;
    for (int i = 0; i < 3; ++i) {
        vec_t dir_plane_x, dir_plane_y, dir_axis;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        if (!g.Using || (g.Using && type == MT_MOVE_X + i)) {
            // draw axis
            if (below_axis_limit && Intersects(op, static_cast<OPERATION>(TRANSLATE_X << i))) {
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
        const auto dest_pos = WorldToPos(g.Model.v.pos, g.ViewProj);
        const auto dif = Normalized({{dest_pos.x - source_pos_screen.x, dest_pos.y - source_pos_screen.y, 0, 0}}) * 5;
        draw_list->AddCircle(source_pos_screen, 6.f, translation_line_color);
        draw_list->AddCircle(dest_pos, 6.f, translation_line_color);
        draw_list->AddLine({source_pos_screen.x + dif._v.x, source_pos_screen.y + dif._v.y}, {dest_pos.x - dif._v.x, dest_pos.y - dif._v.y}, translation_line_color, 2.f);

        const auto delta_info = g.Model.v.pos - g.MatrixOrigin;
        const int component_info_i = (type - MT_MOVE_X) * 3;
        static constexpr const char *TranslationInfoMask[]{"X : %5.3f", "Y : %5.3f", "Z : %5.3f", "Y : %5.3f Z : %5.3f", "X : %5.3f Z : %5.3f", "X : %5.3f Y : %5.3f", "X : %5.3f Y : %5.3f Z : %5.3f"};

        char tmps[512];
        ImFormatString(tmps, sizeof(tmps), TranslationInfoMask[type - MT_MOVE_X], delta_info[TranslationInfoIndex[component_info_i]], delta_info[TranslationInfoIndex[component_info_i + 1]], delta_info[TranslationInfoIndex[component_info_i + 2]]);
        draw_list->AddText(ImVec2(dest_pos.x + 15, dest_pos.y + 15), GetColorU32(TEXT_SHADOW), tmps);
        draw_list->AddText(ImVec2(dest_pos.x + 14, dest_pos.y + 14), GetColorU32(TEXT), tmps);
    }
}

int GetScaleType(OPERATION op) {
    if (g.Using) return MT_NONE;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    int type = mouse_pos.x >= g.ScreenSquareMin.x && mouse_pos.x <= g.ScreenSquareMax.x &&
            mouse_pos.y >= g.ScreenSquareMin.y && mouse_pos.y <= g.ScreenSquareMax.y &&
            Contains(op, SCALE) ?
        MT_SCALE_XYZ :
        MT_NONE;

    // compute
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_X << i))) continue;
        vec_t dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);
        dir_axis.TransformVector(g.ModelLocal);
        dir_plane_x.TransformVector(g.ModelLocal);
        dir_plane_y.TransformVector(g.ModelLocal);

        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, BuildPlan(g.ModelLocal.v.pos, dir_axis));
        const float start_offset = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i)) ? 1.0f : 0.1f;
        const float end_offset = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i)) ? 1.4f : 1.0f;
        const auto pos_plan = g.RayOrigin + g.RayVector * len;
        const auto pos_plan_screen = WorldToPos(pos_plan, g.ViewProj);
        const auto axis_start_screen = WorldToPos(g.ModelLocal.v.pos + dir_axis * g.ScreenFactor * start_offset, g.ViewProj);
        const auto axis_end_screen = WorldToPos(g.ModelLocal.v.pos + dir_axis * g.ScreenFactor * end_offset, g.ViewProj);
        const auto closest_on_axis = PointOnSegment(MakeVect(pos_plan_screen), MakeVect(axis_start_screen), MakeVect(axis_end_screen));
        if ((closest_on_axis - MakeVect(pos_plan_screen)).Length() < 12.f) type = MT_SCALE_X + i; // pixel size
    }

    // universal
    const vec_t delta_screen{{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0}};
    if (float dist = delta_screen.Length(); dist >= 17.0f && dist < 23.0f && Contains(op, SCALEU)) type = MT_SCALE_XYZ;

    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(SCALE_XU << i))) continue;

        vec_t dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit, true);

        // draw axis
        if (below_axis_limit) {
            const bool has_translate_on_axis = Contains(op, static_cast<OPERATION>(TRANSLATE_X << i));
            const float marker_scale = has_translate_on_axis ? 1.4f : 1.0f;
            const auto end = WorldToPos((dir_axis * marker_scale) * g.ScreenFactor, g.MVPLocal);
            if (float distance = sqrtf(ImLengthSqr(end - mouse_pos)); distance < 12.f) type = MT_SCALE_X + i;
        }
    }
    return type;
}

int GetRotateType(OPERATION op) {
    if (g.Using) return MT_NONE;

    const auto mouse_pos = ImGui::GetIO().MousePos;
    const vec_t delta_screen{{mouse_pos.x - g.ScreenSquareCenter.x, mouse_pos.y - g.ScreenSquareCenter.y, 0, 0}};
    const auto dist = delta_screen.Length();
    int type = Intersects(op, ROTATE_SCREEN) && dist >= (g.RadiusSquareCenter - 4.0f) && dist < (g.RadiusSquareCenter + 4.0f) ?
        MT_ROTATE_SCREEN :
        MT_NONE;

    const vec_t plan_normals[]{g.Model.v.right, g.Model.v.up, g.Model.v.dir};
    vec_t model_view_pos = g.Model.v.pos;
    model_view_pos.TransformPoint(g.View);
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        if (!Intersects(op, static_cast<OPERATION>(ROTATE_X << i))) continue;

        const auto pickup_plan = BuildPlan(g.Model.v.pos, plan_normals[i]);
        const auto len = IntersectRayPlane(g.RayOrigin, g.RayVector, pickup_plan);
        const auto intersect_world_pos = g.RayOrigin + g.RayVector * len;
        vec_t intersect_view_pos = intersect_world_pos;
        intersect_view_pos.TransformPoint(g.View);
        if (ImAbs(model_view_pos._v.z) - ImAbs(intersect_view_pos._v.z) < -FLT_EPSILON) continue;

        const auto pos_local = intersect_world_pos - g.Model.v.pos;
        auto ideal_pos_circle = Normalized(pos_local);
        ideal_pos_circle.TransformVector(g.ModelInverse);
        const auto ideal_circle_pos_screen = WorldToPos(ideal_pos_circle * RotationDisplayScale * g.ScreenFactor, g.MVP);
        const auto distance_screen = ideal_circle_pos_screen - mouse_pos;
        if (MakeVect(distance_screen).Length() < 8.f) type = MT_ROTATE_X + i; // pixel size
    }

    return type;
}

int GetMoveType(OPERATION op, vec_t *hit_proportion = nullptr) {
    if (g.Using || !g.MouseOver || !Intersects(op, TRANSLATE)) return MT_NONE;

    auto &io = ImGui::GetIO();
    int type = MT_NONE;
    if (io.MousePos.x >= g.ScreenSquareMin.x && io.MousePos.x <= g.ScreenSquareMax.x &&
        io.MousePos.y >= g.ScreenSquareMin.y && io.MousePos.y <= g.ScreenSquareMax.y &&
        Contains(op, TRANSLATE)) {
        type = MT_MOVE_SCREEN;
    }

    const auto pos_screen = MakeVect(io.MousePos - ImVec2(g.X, g.Y));
    for (int i = 0; i < 3 && type == MT_NONE; i++) {
        vec_t dir_plane_x, dir_plane_y, dir_axis;
        bool below_axis_limit, below_plane_limit;
        ComputeTripodAxisAndVisibility(i, dir_axis, dir_plane_x, dir_plane_y, below_axis_limit, below_plane_limit);
        dir_axis.TransformVector(g.Model);
        dir_plane_x.TransformVector(g.Model);
        dir_plane_y.TransformVector(g.Model);

        const auto axis_start_screen = WorldToPos(g.Model.v.pos + dir_axis * g.ScreenFactor * 0.1f, g.ViewProj) - ImVec2(g.X, g.Y);
        const auto axis_end_screen = WorldToPos(g.Model.v.pos + dir_axis * g.ScreenFactor, g.ViewProj) - ImVec2(g.X, g.Y);
        const vec_t closest_on_axis = PointOnSegment(pos_screen, MakeVect(axis_start_screen), MakeVect(axis_end_screen));
        if ((closest_on_axis - pos_screen).Length() < 12.f && Intersects(op, static_cast<OPERATION>(TRANSLATE_X << i))) type = MT_MOVE_X + i;

        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, BuildPlan(g.Model.v.pos, dir_axis));
        const auto pos_plan = g.RayOrigin + g.RayVector * len;
        const float dx = dir_plane_x.Dot3((pos_plan - g.Model.v.pos) * (1.f / g.ScreenFactor));
        const float dy = dir_plane_y.Dot3((pos_plan - g.Model.v.pos) * (1.f / g.ScreenFactor));
        if (below_plane_limit && dx >= QuadUV[0] && dx <= QuadUV[4] && dy >= QuadUV[1] && dy <= QuadUV[3] && Contains(op, TranslatePlans[i])) {
            type = MT_MOVE_YZ + i;
        }

        if (hit_proportion != nullptr) *hit_proportion = MakeVect(dx, dy, 0.f);
    }
    return type;
}

bool CanActivate() { return ImGui::IsMouseClicked(0) && !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive(); }

bool HandleTranslation(float *matrix, float *delta_matrix, OPERATION op, int &type, const float *snap) {
    if (!Intersects(op, TRANSLATE) || type != MT_NONE) return false;

    const bool apply_rot_locally = g.Mode == LOCAL || type == MT_MOVE_SCREEN;
    bool modified = false;

    // move
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsTranslateType(g.CurrentOp)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len_signed = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
        const float len = fabsf(len_signed); // near plan
        const auto new_pos = g.RayOrigin + g.RayVector * len;

        // compute delta
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - g.Model.v.pos;

        // 1 axis constraint
        if (g.CurrentOp >= MT_MOVE_X && g.CurrentOp <= MT_MOVE_Z) {
            const int axis_i = g.CurrentOp - MT_MOVE_X;
            const auto &axis_value = *(vec_t *)&g.Model.m4[axis_i];
            const auto length_on_axis = Dot(axis_value, delta);
            delta = axis_value * length_on_axis;
        }

        // snap
        if (snap) {
            auto delta_cumulative = g.Model.v.pos + delta - g.MatrixOrigin;
            if (apply_rot_locally) {
                auto model_source = g.ModelSource;
                model_source.OrthoNormalize();
                matrix_t model_source_inv{};
                model_source_inv.Inverse(model_source);
                delta_cumulative.TransformVector(model_source_inv);
                ComputeSnap(delta_cumulative, snap);
                delta_cumulative.TransformVector(model_source);
            } else {
                ComputeSnap(delta_cumulative, snap);
            }
            delta = g.MatrixOrigin + delta_cumulative - g.Model.v.pos;
        }

        if (delta != g.TranslationPrevDelta) {
            g.TranslationPrevDelta = delta;
            modified = true;
        }

        // compute matrix & delta
        matrix_t delta_matrix_translation{};
        delta_matrix_translation.Translation(delta);
        if (delta_matrix) memcpy(delta_matrix, &delta_matrix_translation.m4[0][0], sizeof(mat4));

        *(matrix_t *)matrix = g.ModelSource * delta_matrix_translation;

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
                vec_t move_plan_normal[]{g.Model.v.right, g.Model.v.up, g.Model.v.dir, g.Model.v.right, g.Model.v.up, g.Model.v.dir, -g.CameraDir};
                const auto cam_to_model = Normalized(g.Model.v.pos - g.CameraEye);
                for (unsigned int i = 0; i < 3; i++) {
                    const auto ortho = Cross(move_plan_normal[i], cam_to_model);
                    move_plan_normal[i].Cross(ortho);
                    move_plan_normal[i] = Normalized(move_plan_normal[i]);
                }
                // pickup plan
                g.TranslationPlan = BuildPlan(g.Model.v.pos, move_plan_normal[type - MT_MOVE_X]);
                const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
                g.TranslationPlanOrigin = g.RayOrigin + g.RayVector * len;
                g.MatrixOrigin = g.Model.v.pos;

                g.RelativeOrigin = (g.TranslationPlanOrigin - g.Model.v.pos) * (1.f / g.ScreenFactor);
            }
        }
    }
    return modified;
}

bool HandleScale(float *matrix, float *delta_matrix, OPERATION op, int &type, const float *snap) {
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
            const vec_t move_plan_normal[]{g.Model.v.up, g.Model.v.dir, g.Model.v.right, g.Model.v.dir, g.Model.v.up, g.Model.v.right, -g.CameraDir};
            g.TranslationPlan = BuildPlan(g.Model.v.pos, move_plan_normal[type - MT_SCALE_X]);
            const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
            g.TranslationPlanOrigin = g.RayOrigin + g.RayVector * len;
            g.MatrixOrigin = g.Model.v.pos;
            g.Scale.Set(1.f, 1.f, 1.f);
            g.RelativeOrigin = (g.TranslationPlanOrigin - g.Model.v.pos) * (1.f / g.ScreenFactor);
            g.ScaleOrigin = MakeVect(g.ModelSource.v.right.Length(), g.ModelSource.v.up.Length(), g.ModelSource.v.dir.Length());
            g.SaveMousePosX = ImGui::GetIO().MousePos.x;
        }
    }
    // scale
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsScaleType(g.CurrentOp)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
        const auto new_pos = g.RayOrigin + g.RayVector * len;
        const auto new_origin = new_pos - g.RelativeOrigin * g.ScreenFactor;
        auto delta = new_origin - g.ModelLocal.v.pos;
        // 1 axis constraint
        if (g.CurrentOp >= MT_SCALE_X && g.CurrentOp <= MT_SCALE_Z) {
            int axis_i = g.CurrentOp - MT_SCALE_X;
            const vec_t &axis_value = *(vec_t *)&g.ModelLocal.m4[axis_i];
            const float length_on_axis = Dot(axis_value, delta);
            delta = axis_value * length_on_axis;

            vec_t base = g.TranslationPlanOrigin - g.ModelLocal.v.pos;
            const float ratio = Dot(axis_value, base + delta) / Dot(axis_value, base);
            g.Scale[axis_i] = std::max(ratio, 0.001f);
        } else {
            const float scale_delta = (ImGui::GetIO().MousePos.x - g.SaveMousePosX) * 0.01f;
            g.Scale.Set(std::max(1.f + scale_delta, 0.001f));
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

        // compute matrix & delta
        matrix_t delta_mat_scale{};
        delta_mat_scale.Scale(g.Scale * g.ScaleOrigin);

        matrix_t res = delta_mat_scale * g.ModelLocal;
        *(matrix_t *)matrix = res;

        if (delta_matrix) {
            vec_t original_scale_divider{{vec3(1.f / g.ModelScaleOrigin._v), 0}};
            delta_mat_scale.Scale(g.Scale * g.ScaleOrigin * original_scale_divider);
            memcpy(delta_matrix, &delta_mat_scale.m4[0][0], sizeof(mat4));
        }

        if (!ImGui::GetIO().MouseDown[0]) {
            g.Using = false;
            g.Scale.Set(1.f, 1.f, 1.f);
        }

        type = g.CurrentOp;
    }
    return modified;
}

bool HandleRotation(float *matrix, float *delta_matrix, OPERATION op, int &type, const float *snap) {
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
            const vec_t rotate_plan_normal[]{g.Model.v.right, g.Model.v.up, g.Model.v.dir, -g.CameraDir};
            g.TranslationPlan = apply_rot_locally ?
                BuildPlan(g.Model.v.pos, rotate_plan_normal[type - MT_ROTATE_X]) :
                BuildPlan(g.ModelSource.v.pos, DirUnary[type - MT_ROTATE_X]);

            const float len = IntersectRayPlane(g.RayOrigin, g.RayVector, g.TranslationPlan);
            const auto pos_local = g.RayOrigin + g.RayVector * len - g.Model.v.pos;
            g.RotationVectorSource = Normalized(pos_local);
            g.RotationAngleOrigin = ComputeAngleOnPlan();
        }
    }

    // rotation
    if (g.Using && (g.ActualID == -1 || g.ActualID == g.EditingID) && IsRotateType(g.CurrentOp)) {
        ImGui::SetNextFrameWantCaptureMouse(true);
        g.RotationAngle = ComputeAngleOnPlan();
        if (snap) ComputeSnap(&g.RotationAngle, snap[0] * M_PI / 180.f);

        vec_t rot_axis_local_space{{vec3(g.TranslationPlan._v), 0.f}};
        rot_axis_local_space.TransformVector(g.ModelInverse);
        rot_axis_local_space = Normalized(rot_axis_local_space);

        matrix_t delta_rot{};
        delta_rot.RotationAxis(rot_axis_local_space, g.RotationAngle - g.RotationAngleOrigin);
        if (g.RotationAngle != g.RotationAngleOrigin) {
            g.RotationAngleOrigin = g.RotationAngle;
            modified = true;
        }

        matrix_t scale_origin{};
        scale_origin.Scale(g.ModelScaleOrigin);

        if (apply_rot_locally) {
            *(matrix_t *)matrix = scale_origin * delta_rot * g.ModelLocal;
        } else {
            auto res = g.ModelSource;
            res.v.pos.Set(0.f);

            *(matrix_t *)matrix = res * delta_rot;
            ((matrix_t *)matrix)->v.pos = g.ModelSource.v.pos;
        }

        if (delta_matrix) *(matrix_t *)delta_matrix = g.ModelInverse * delta_rot * g.Model;

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
bool Manipulate(const float *view, const float *projection, OPERATION op, MODE mode, float *matrix, float *delta_matrix, const float *snap) {
    // Scale is always local or matrix will be skewed when applying world scale or oriented matrix
    ComputeContext(view, projection, matrix, (op & SCALE) ? LOCAL : mode);

    if (delta_matrix) ((matrix_t *)delta_matrix)->SetToIdentity();

    // behind camera
    vec_t pos_cam_space{};
    pos_cam_space.TransformPoint(g.MVP);
    if (!g.IsOrthographic && pos_cam_space._v.z < 0.001 && !g.Using) return false;

    int type = MT_NONE;
    const bool manipulated = HandleTranslation(matrix, delta_matrix, op, type, snap) ||
        HandleScale(matrix, delta_matrix, op, type, snap) ||
        HandleRotation(matrix, delta_matrix, op, type, snap);

    g.Op = op;
    DrawRotationGizmo(op, type);
    DrawTranslationGizmo(op, type);
    DrawScaleGizmo(op, type);
    DrawScaleUniveralGizmo(op, type);
    return manipulated;
}
} // namespace ImGuizmo
