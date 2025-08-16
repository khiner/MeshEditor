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

#include "AxisColors.h" // Must be after imgui.h

#include <algorithm>
#include <format>
#include <optional>
#include <span>

namespace {
enum class InteractionAxis : uint8_t {
    AxisX = 1 << 0,
    AxisY = 1 << 1,
    AxisZ = 1 << 2,
    Screen,
    YZ,
    ZX,
    XY,
};

// `TransformType` is a subset of (the externally visible) `Type` without `Universal`.
enum class TransformType : uint8_t {
    Translate,
    Rotate,
    Scale,
};

using namespace ModelGizmo;

TransformType ToTransformType(Type type) {
    switch (type) {
        case Type::Translate: return TransformType::Translate;
        case Type::Rotate: return TransformType::Rotate;
        case Type::Scale: return TransformType::Scale;
        case Type::Universal: {
            assert(false); // Universal type cannot be converted to a single TransformType
            return TransformType::Scale;
        }
    }
}

using enum InteractionAxis;

// Specific hovered/active transform/axis.
struct Interaction {
    TransformType Transform;
    InteractionAxis Axis;

    bool operator==(const Interaction &other) const {
        return other.Transform == Transform && other.Axis == Axis;
    }
};

namespace state {
// Context captured when mouse is pressed on a hovered transform.
struct StartContext {
    mat4 M; // Model matrix
    vec4 PlaneWs; // Plane for the pressed transform
    vec2 MousePx;
    ray MouseRayWs;
    float WorldToSizeNdc;
};

struct Context {
    mat4 MVP;

    ImRect ScreenRect{{0, 0}, {0, 0}};

    // World-space distance that projects to Style.SizeNdc, computed as:
    // Style.SizeNdc / (NDC-length of a world-space unit vector along the camera’s right direction, projected at the model’s origin)
    float WorldToSizeNdc;

    std::optional<Interaction> Interaction; // If `Start` is present, active interaction. Otherwise, hovered interaction.
    std::optional<StartContext> Start; // Captured at mouse press on hovered Interaction.
    vec3 Scale; // Scale factor since start
    float RotationAngle; // Relative to start rotation
};

// `Scale` members are relative to the gizmo size. To convert to screen-relative, multiply them by SizeNdc.
struct Style {
    // Size of the gizmo in NDC coordinates, relative to the screen size.
    // todo Actually, it's currently the size of an axis handle, but I want to change it to be the size of the whole gizmo,
    // and make all other scales <= 1.
    float SizeNdc{0.15};

    // `AxisHandle`s are the lines for translate/scale
    float TranslationArrowScale{0.18}, TranslationArrowRadScale{TranslationArrowScale * 0.3f};
    float AxisHandleScale{1.f - TranslationArrowScale}; // Tip is exacly at the gizmo size
    float UniversalAxisHandleScale{AxisHandleScale - TranslationArrowScale}; // For scale handles in Universal mode
    float AxisHandleLineWidth{2};
    // Radius and length of the arrow at the end of translation axes
    float TranslationArrowUniversalPosScale{1 + TranslationArrowScale}; // Translation arrows in Universal mode are the only thing "outside" the gizmo
    float PlaneSizeAxisScale{0.13}; // Translation plane quads
    float CircleRadScale{0.03}; // Radius of circle at the end of scale lines and the center of the translate/scale gizmo
    float CubeHalfExtentScale{1.45f * CircleRadScale}; // Half extent of scale cube handles
    float InnerCircleRadScale{0.1}; // Radius of the inner selection circle at the center for translate/scale selection
    float OuterCircleRadScale{0.5}; // Outer circle is exactly the size of the gizmo
    float CircleLineWidth{2}; // Thickness of inner & outer circle
    float RotationAxesCircleScale{AxisHandleScale}; // Rotation axes circles are smaller than the screen circle, equal to the the translation arrow base
    float RotationLineWidth{2.5}; // Thickness of rotation gizmo lines

    float AxisInvisibleRadScale{InnerCircleRadScale}; // Axes gradually fade into invisibility at this distance from center
    float AxisOpaqueRadScale{2 * InnerCircleRadScale}; // Axes are fully opaque at this distance from center
};

struct Color {
    ImU32 TranslationLine{IM_COL32(170, 170, 170, 170)};
    ImU32 ScaleLine{IM_COL32(64, 64, 64, 255)};
    ImU32 StartGhost{IM_COL32(160, 160, 160, 160)};
    ImU32 RotationFillActive{IM_COL32(255, 255, 255, 64)};
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
constexpr vec4 Right(const mat4 &m) { return {m[0]}; }
constexpr vec4 Dir(const mat4 &m) { return {m[2]}; }
constexpr vec3 Pos(const mat4 &m) { return {m[3]}; } // Assume affine matrix, with w = 1

constexpr float Length2(vec2 v) { return v.x * v.x + v.y * v.y; }

// Assumes no scaling or shearing, only rotation and translation
constexpr mat4 InverseRigid(const mat4 &m) {
    const auto r = glm::transpose(mat3{m});
    return {{r[0], 0}, {r[1], 0}, {r[2], 0}, {-r * vec3{m[3]}, 1}};
}

constexpr InteractionAxis AxisOp(uint32_t axis_i) { return InteractionAxis(uint32_t(AxisX) << axis_i); }
constexpr uint32_t AxisIndex(InteractionAxis axis) {
    if (axis == AxisX) return 0;
    if (axis == AxisY) return 1;
    if (axis == AxisZ) return 2;
    assert(false);
    return -1;
}

constexpr vec3 Axes[]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
constexpr InteractionAxis TranslatePlanes[]{InteractionAxis::YZ, InteractionAxis::ZX, InteractionAxis::XY}; // In axis order

constexpr std::optional<uint32_t> TranslatePlaneIndex(InteractionAxis plane) {
    if (plane == InteractionAxis::YZ) return 0;
    if (plane == InteractionAxis::ZX) return 1;
    if (plane == InteractionAxis::XY) return 2;
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

// Camera _right_ vector is used to calculate WorldToSizeNdc,
// so screen _width_ is used to convert back to pixels.
constexpr float ScaleToPx(float scale = 1.f) { return (g.ScreenRect.Max.x - g.ScreenRect.Min.x) * Style.SizeNdc * scale; }
constexpr ImVec2 NdcToPx(vec2 ndc) { return g.ScreenRect.Min + ImVec2{ndc.x, 1.f - ndc.y} * g.ScreenRect.GetSize(); }
// Homogeneous clip space to NDC [0, 1]
constexpr vec2 ClipToNdc(vec4 v) { return {fabsf(v.w) > FLT_EPSILON ? v / v.w : v}; }
constexpr vec2 WorldToNdc(vec3 ws, const mat4 &view_proj) { return ClipToNdc(view_proj * vec4{ws, 1}) * 0.5f + 0.5f; }
constexpr ImVec2 WorldToPx(vec3 ws, const mat4 &view_proj) { return NdcToPx(WorldToNdc(ws, view_proj)); }

constexpr bool IsPlaneVisible(vec3 dir_x, vec3 dir_y) {
    static constexpr auto ToScreenNdc = [](vec3 v) { return ClipToNdc(g.MVP * vec4{v, 1}); };
    static constexpr float ParallelogramAreaLimit{0.0025};
    const auto o = ToScreenNdc(vec3{0});
    const auto pa = ToScreenNdc(dir_x * g.WorldToSizeNdc) - o;
    const auto pb = ToScreenNdc(dir_y * g.WorldToSizeNdc) - o;
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
constexpr char AxisLabels[]{"XYZ"};
constexpr std::string Axis(uint32_t i, float v) { return i >= 0 && i < 3 ? std::format("{}: {:.3f}", AxisLabels[i], v) : ""; }
constexpr std::string Axis(uint32_t i, vec3 v) { return Axis(i, v[i]); }
constexpr std::string Translation(InteractionAxis axis, vec3 v) {
    if (axis == InteractionAxis::Screen) return std::format("{} {} {}", Axis(0, v[0]), Axis(1, v[1]), Axis(2, v[2]));
    if (axis == InteractionAxis::YZ) return std::format("{} {}", Axis(1, v[1]), Axis(2, v[2]));
    if (axis == InteractionAxis::ZX) return std::format("{} {}", Axis(2, v[2]), Axis(0, v[0]));
    if (axis == InteractionAxis::XY) return std::format("{} {}", Axis(0, v[0]), Axis(1, v[1]));
    return Axis(AxisIndex(axis), v);
}
constexpr std::string Scale(InteractionAxis axis, vec3 v) {
    return axis == InteractionAxis::Screen ? std::format("XYZ: {:.3f}", v.x) : Axis(AxisIndex(axis), v);
}
constexpr std::string Rotation(InteractionAxis axis, float rad) {
    const auto deg_rad = std::format("{:.3f} deg {:.3f} rad", rad * 180 / M_PI, rad);
    if (axis == InteractionAxis::Screen) return std::format("Screen: {}", deg_rad);

    const auto axis_i = AxisIndex(axis);
    return axis_i >= 0 && axis_i < 3 ? std::format("{}: {}", AxisLabels[axis_i], deg_rad) : "";
}
} // namespace Format

struct Model {
    Model(const mat4 &m, Mode mode)
        : RT{glm::normalize(m[0]), glm::normalize(m[1]), glm::normalize(m[2]), m[3]},
          M{mode == Mode::Local ? RT : glm::translate(mat4{1}, Pos(RT))} {}
    const mat4 RT; // Model matrix rotation + translation
    const mat4 M; // Gizmo model matrix (Local or World space).
    const mat4 Inv{InverseRigid(M)}; // Inverse of Gizmo model matrix
};

mat4 Transform(const mat4 &m, const Model &model, Mode mode, Interaction interaction, vec2 mouse_pos, const ray &mouse_ray, std::optional<vec3> snap) {
    using enum TransformType;

    assert(g.Start);

    const auto transform = interaction.Transform;
    const auto axis = interaction.Axis;
    const auto p = Pos(model.M);
    const auto p_start_ws = Pos(g.Start->M);
    const auto &plane = g.Start->PlaneWs;
    if (transform == Translate) {
        auto delta = mouse_ray(fabsf(IntersectPlane(mouse_ray, plane))) -
            (g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, plane)) - p_start_ws) - p;
        // Single axis constraint
        if (axis == AxisX || axis == AxisY || axis == AxisZ) {
            const auto axis_i = AxisIndex(axis);
            delta = model.M[axis_i] * glm::dot(model.M[axis_i], vec4{delta, 0});
        }
        if (snap) {
            const vec4 d{p + delta - p_start_ws, 0};
            const vec3 delta_cumulative = mode == Mode::Local || axis == Screen ? m * vec4{Snap(model.Inv * d, *snap), 0} : Snap(d, *snap);
            delta = p_start_ws + delta_cumulative - p;
        }
        return glm::translate(mat4{1}, delta) * m;
    }
    if (transform == Scale) {
        if (axis == Screen) {
            g.Scale = vec3{1 + (mouse_pos.x - g.Start->MousePx.x) * 0.01f};
        } else { // Single axis constraint
            const auto axis_i = AxisIndex(axis);
            const vec3 axis_value{model.RT[axis_i]};
            const auto relative_origin = g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, plane)) - p_start_ws;
            const auto p_ortho = Pos(model.RT);
            const auto base = relative_origin / g.WorldToSizeNdc - p_ortho;
            const auto delta = axis_value * glm::dot(axis_value, mouse_ray(IntersectPlane(mouse_ray, plane)) - relative_origin - p_ortho);
            g.Scale[axis_i] = glm::dot(axis_value, base + delta) / glm::dot(axis_value, base);
        }

        if (snap) g.Scale = Snap(g.Scale, *snap);
        for (uint32_t i = 0; i < 3; ++i) g.Scale[i] = std::max(g.Scale[i], 0.001f);

        const auto &m_start = g.Start->M;
        const vec3 scale_start{glm::length(m_start[0]), glm::length(m_start[1]), glm::length(m_start[2])};
        return model.RT * glm::scale(mat4{1}, g.Scale * scale_start);
    }

    // Rotation: Compute angle on plane relative to the rotation origin
    const auto rotation_origin = glm::normalize(g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, plane)) - p_start_ws);
    const auto perp = glm::cross(rotation_origin, vec3{plane});
    const auto pos_local = glm::normalize(mouse_ray(IntersectPlane(mouse_ray, plane)) - p);
    float rotation_angle = acosf(glm::clamp(glm::dot(pos_local, rotation_origin), -1.f, 1.f)) * -glm::sign(glm::dot(pos_local, perp));
    if (snap) rotation_angle = Snap(rotation_angle, snap->x * M_PI / 180.f);

    const vec3 rot_axis_local = glm::normalize(glm::mat3{model.Inv} * vec3{plane}); // Assumes affine model
    const mat4 rot_delta{glm::rotate(mat4{1}, rotation_angle - g.RotationAngle, rot_axis_local)};
    g.RotationAngle = rotation_angle;

    if (mode == Mode::Local) {
        const vec3 model_scale{glm::length(m[0]), glm::length(m[1]), glm::length(m[2])};
        return model.RT * rot_delta * glm::scale(mat4{1}, model_scale);
    }

    // Apply rotation, preserving translation
    auto res = rot_delta * mat4{mat3{m}};
    res[3] = vec4{Pos(m), 1};
    return res;
}

constexpr ImVec2 PointOnSegment(ImVec2 p, ImVec2 s1, ImVec2 s2) {
    const auto vec = std::bit_cast<vec2>(s2 - s1);
    const auto v = glm::normalize(vec);
    const float t = glm::dot(v, std::bit_cast<vec2>(p - s1));
    if (t <= 0) return s1;
    if (t * t > Length2(vec)) return s2;
    return s1 + std::bit_cast<ImVec2>(v) * t;
}

std::optional<Interaction> FindHoveredInteraction(const Model &model, Type type, ImVec2 mouse_pos, const ray &mouse_ray, const mat4 &view, const mat4 &view_proj) {
    using enum Type;

    const auto center = WorldToPx(vec3{0}, g.MVP);
    const auto mouse_r_sq = ImLengthSqr(mouse_pos - center);
    if (type == Rotate || type == Universal) {
        static constexpr float SelectDist = 8;
        const auto rotation_radius = ScaleToPx(Style.OuterCircleRadScale);
        const auto inner_rad = rotation_radius - SelectDist / 2, outer_rad = rotation_radius + SelectDist / 2;
        if (mouse_r_sq >= inner_rad * inner_rad && mouse_r_sq < outer_rad * outer_rad) {
            return Interaction{ToTransformType(Rotate), Screen};
        }

        const auto p = Pos(model.M);
        const auto mv_pos = view * vec4{p, 1};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto intersect_pos_world = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(p, model.M[i])));
            const auto intersect_pos = view * vec4{vec3{intersect_pos_world}, 1};
            if (fabsf(mv_pos.z) - fabsf(intersect_pos.z) < -FLT_EPSILON) continue;

            const auto circle_pos_world = model.Inv * vec4{glm::normalize(intersect_pos_world - p), 0};
            const auto circle_pos = WorldToPx(circle_pos_world * Style.RotationAxesCircleScale * g.WorldToSizeNdc, g.MVP);
            if (ImLengthSqr(circle_pos - mouse_pos) < SelectDist * SelectDist) {
                return Interaction{ToTransformType(Rotate), AxisOp(i)};
            }
        }
    }
    if (type == Translate || type == Scale || type == Universal) {
        const auto inner_circle_rad_px = ScaleToPx(Style.InnerCircleRadScale);
        if ((type == Translate || type == Universal) && mouse_r_sq <= inner_circle_rad_px * inner_circle_rad_px) {
            return Interaction{ToTransformType(Translate), Screen};
        }

        const auto half_arrow_px = ScaleToPx(Style.TranslationArrowScale) * 0.5f;
        const auto screen_pos = g.ScreenRect.Min, mouse_pos_rel{mouse_pos - screen_pos};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto dir = model.M * vec4{Axes[i], 0};
            const auto p = Pos(model.M);
            const auto scale = type == Universal ? Style.UniversalAxisHandleScale : Style.AxisHandleScale;
            const auto start = WorldToPx(vec4{p, 1} + dir * g.WorldToSizeNdc * scale * Style.InnerCircleRadScale, view_proj) - screen_pos;
            const auto end = WorldToPx(vec4{p, 1} + dir * g.WorldToSizeNdc * (scale + Style.TranslationArrowScale), view_proj) - screen_pos;
            if (ImLengthSqr(PointOnSegment(mouse_pos_rel, start, end) - mouse_pos_rel) < half_arrow_px * half_arrow_px) {
                return Interaction{ToTransformType(type == Translate ? Translate : Scale), AxisOp(i)};
            }
            if (type == Scale) continue;

            if (type == Universal) {
                const auto arrow_center_scale = Style.TranslationArrowUniversalPosScale + Style.TranslationArrowScale * 0.5f;
                const auto translate_pos = WorldToPx(vec4{p, 1} + dir * g.WorldToSizeNdc * arrow_center_scale, view_proj) - screen_pos;
                if (ImLengthSqr(translate_pos - mouse_pos_rel) < half_arrow_px * half_arrow_px) {
                    return Interaction{ToTransformType(Translate), AxisOp(i)};
                }
            }

            const auto [dir_x, dir_y] = DirPlaneXY(i);
            if (!IsPlaneVisible(dir_x, dir_y)) continue;

            const auto p_world = Pos(model.M);
            const auto pos_plane = mouse_ray(IntersectPlane(mouse_ray, BuildPlane(p_world, dir)));
            const auto plane_x_world = vec3{model.M * vec4{dir_x, 0}};
            const auto plane_y_world = vec3{model.M * vec4{dir_y, 0}};
            const auto delta_world = (pos_plane - p_world) / g.WorldToSizeNdc;
            const float dx = glm::dot(delta_world, plane_x_world);
            const float dy = glm::dot(delta_world, plane_y_world);
            const float PlaneQuadUVMin = 0.5f - Style.PlaneSizeAxisScale * 0.5f;
            const float PlaneQuadUVMax = 0.5f + Style.PlaneSizeAxisScale * 0.5f;
            if (dx >= PlaneQuadUVMin && dx <= PlaneQuadUVMax && dy >= PlaneQuadUVMin && dy <= PlaneQuadUVMax) {
                return Interaction{ToTransformType(Translate), TranslatePlanes[i]};
            }
        }
        if (type != Translate) {
            const auto outer_circle_rad_px = ScaleToPx(Style.OuterCircleRadScale);
            if (mouse_r_sq >= inner_circle_rad_px * inner_circle_rad_px &&
                mouse_r_sq < outer_circle_rad_px * outer_circle_rad_px) {
                return Interaction{ToTransformType(Scale), Screen};
            }
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

constexpr float AxisAlphaForDistPxSq(float dist_px_sq) {
    const float min_dist = ScaleToPx(Style.AxisInvisibleRadScale);
    if (dist_px_sq <= min_dist * min_dist) return 0;

    const float max_dist = ScaleToPx(Style.AxisOpaqueRadScale);
    if (dist_px_sq >= max_dist * max_dist) return 1;

    return (sqrt(dist_px_sq) - min_dist) / (max_dist - min_dist);
}

constexpr ImU32 SelectionColor(ImU32 color, bool selected) {
    return selected ? color : colors::MultAlpha(color, 0.8f);
}

void Render(const Model &model, Type type, const mat4 &view_proj, vec3 cam_origin) {
    using enum TransformType;

    auto &dl = *ImGui::GetWindowDrawList();
    const auto p_px = WorldToPx(vec3{0}, g.MVP);
    const auto p_ws = Pos(model.M);
    // Ghost circle
    if (g.Start && g.Interaction->Axis == Screen && (g.Interaction->Transform == Translate || g.Interaction->Transform == Scale)) {
        const auto center = g.Interaction->Transform == Translate ? WorldToPx(Pos(g.Start->M), view_proj) : p_px;
        dl.AddCircle(center, ScaleToPx(Style.InnerCircleRadScale), Color.StartGhost, 0, Style.CircleLineWidth);
    }
    // Inner circle
    if ((!g.Start && type != Type::Rotate) ||
        (g.Start && ((g.Interaction->Transform == Translate || g.Interaction->Transform == Scale) && g.Interaction->Axis == Screen))) {
        const auto color = SelectionColor(IM_COL32_WHITE, g.Interaction && g.Interaction->Axis == Screen);
        const auto scale = g.Start && g.Interaction->Transform == Scale ? g.Scale[0] : 1.f;
        dl.AddCircle(p_px, ScaleToPx(scale * Style.InnerCircleRadScale), color, 0, Style.CircleLineWidth);
    }
    // Outer circle
    if (type != Type::Translate && (!g.Start || g.Interaction == Interaction{Rotate, Screen})) {
        dl.AddCircle(
            p_px,
            ScaleToPx(Style.OuterCircleRadScale),
            SelectionColor(IM_COL32_WHITE, g.Interaction && g.Interaction->Axis == Screen),
            0,
            Style.CircleLineWidth
        );
    }

    if (type == Type::Translate || type == Type::Universal) {
        // Circle + line + cone silhouette
        const auto DrawTranslateHandle = [&](bool is_active, bool ghost, uint32_t axis_i, float end_scale, bool draw_center_circle, bool draw_line) {
            const auto &m = ghost ? g.Start->M : model.M;
            const auto o_ws = Pos(m);
            const auto o_px = WorldToPx(o_ws, view_proj);
            const auto w2s = ghost ? g.Start->WorldToSizeNdc : g.WorldToSizeNdc;
            const auto axis_dir_ws = glm::normalize(vec3{m[axis_i]});
            const auto arrow_base_ws = o_ws + axis_dir_ws * w2s * end_scale;
            const auto arrow_base_px = WorldToPx(arrow_base_ws, view_proj);
            const auto axis_alpha = AxisAlphaForDistPxSq(ImLengthSqr(arrow_base_px - o_px));
            const auto color = ghost ? Color.StartGhost : SelectionColor(colors::WithAlpha(colors::Axes[axis_i], axis_alpha), is_active);
            if (draw_line) {
                const float line_base_scale = g.Start ? Style.CircleRadScale : Style.InnerCircleRadScale;
                const auto line_start_ws = o_ws + axis_dir_ws * w2s * line_base_scale;
                dl.AddLine(WorldToPx(line_start_ws, view_proj), arrow_base_px, color, Style.AxisHandleLineWidth);
            }
            if (draw_center_circle) dl.AddCircleFilled(o_px, ScaleToPx(Style.CircleRadScale), color);

            // Arrow cone silhouette (triangle + ellipse)

            const auto u_ws = glm::normalize((cam_origin - arrow_base_ws) - glm::dot(cam_origin - arrow_base_ws, axis_dir_ws) * axis_dir_ws);
            const auto v_ws = glm::cross(axis_dir_ws, u_ws);
            const auto p_tip = WorldToPx(arrow_base_ws + axis_dir_ws * Style.TranslationArrowScale * w2s, view_proj);
            const auto p_b1 = WorldToPx(arrow_base_ws + v_ws * Style.TranslationArrowRadScale * w2s, view_proj);
            const auto p_b2 = WorldToPx(arrow_base_ws - v_ws * Style.TranslationArrowRadScale * w2s, view_proj);
            dl.AddTriangleFilled(p_tip, p_b1, p_b2, color);

            static constexpr uint32_t EllipsePointCount{16};
            static ImVec2 EllipsePoints[EllipsePointCount];
            const auto c = (p_b1 + p_b2) * 0.5f;
            const auto p_u = WorldToPx(arrow_base_ws + u_ws * (Style.TranslationArrowRadScale * w2s), view_proj);
            FastEllipse(EllipsePoints, c, p_u - c, p_b1 - c);
            dl.AddConvexPolyFilled(EllipsePoints, EllipsePointCount, color);
        };

        const float end_scale = type == Type::Universal ? Style.TranslationArrowUniversalPosScale : Style.AxisHandleScale;
        for (uint32_t i = 0; i < 3; ++i) {
            const bool is_active = g.Interaction == Interaction{Translate, AxisOp(i)};
            if (!g.Start || is_active) {
                DrawTranslateHandle(is_active, false, i, end_scale, bool(g.Start), type != Type::Universal || g.Start);
                // Start-ghost
                if (g.Start) DrawTranslateHandle(is_active, true, i, end_scale, true, true);
            }
        }
        if (g.Start && g.Interaction->Transform == Translate) {
            Label(Format::Translation(g.Interaction->Axis, p_ws - Pos(g.Start->M)), p_px);
        }
    }

    if (type == Type::Scale || type == Type::Universal) {
        auto &dl = *ImGui::GetWindowDrawList();

        // Draws the scale axis, line, and cube silhouette at its tip.
        const auto DrawScaleHandle = [&](bool is_active, bool ghost, uint32_t axis_i, float handle_scale, bool draw_center_circle) {
            // Choose state (current vs start snapshot)
            const auto &m = ghost ? g.Start->M : model.M;
            const auto w2s = ghost ? g.Start->WorldToSizeNdc : g.WorldToSizeNdc;
            const auto o_ws = Pos(m);
            const auto o_px = WorldToPx(o_ws, view_proj);
            // Stable axis basis from the chosen matrix
            const auto axis_dir_ws = glm::normalize(vec3{m[axis_i]});
            const auto u_ws = glm::normalize(vec3{m[(axis_i + 1) % 3]});
            const auto v_ws = glm::normalize(vec3{m[(axis_i + 2) % 3]});
            // End tip and (optional) line start
            const auto end_tip_ws = o_ws + axis_dir_ws * (w2s * handle_scale);
            const auto end_px = WorldToPx(end_tip_ws, view_proj);
            const auto color = ghost ? Color.StartGhost : SelectionColor(colors::WithAlpha(colors::Axes[axis_i], AxisAlphaForDistPxSq(ImLengthSqr(end_px - p_px))), is_active);
            // Line
            const float line_base_scale = g.Start ? Style.CircleRadScale : Style.InnerCircleRadScale;
            const auto line_start_ws = o_ws + axis_dir_ws * w2s * line_base_scale;
            dl.AddLine(WorldToPx(line_start_ws, view_proj), end_px, color, Style.AxisHandleLineWidth);

            if (draw_center_circle) dl.AddCircleFilled(o_px, ScaleToPx(Style.CircleRadScale), color);

            const float half_ws = Style.CubeHalfExtentScale * w2s;
            const vec3 A = axis_dir_ws * half_ws, U = u_ws * half_ws, V = v_ws * half_ws;
            const vec3 C = end_tip_ws + A; // inner (−A) face touches the endpoint

            static constexpr uint8_t NumCorners{8};
            vec3 P[NumCorners]; // (bits: x=U, y=V, z=A)
            for (uint8_t i = 0; i < NumCorners; ++i) {
                P[i] = C + ((i & 1) ? U : -U) + ((i & 2) ? V : -V) + ((i & 4) ? A : -A);
            }

            // Build adjacency for the visible silhouette (single cycle)
            uint8_t deg[NumCorners]{0};
            uint8_t adj[NumCorners][3];
            const auto link = [&adj, &deg](uint8_t a, uint8_t b) {
                adj[a][deg[a]++] = b;
                adj[b][deg[b]++] = a;
            };

            const vec3 view_dir = glm::normalize(cam_origin - C);
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
            if (start == NumCorners) return; // fully backfacing/degenerate

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

            // Project to pixels and fill
            static ImVec2 hull[NumCorners];
            for (uint8_t i = 0; i < n; ++i) hull[i] = WorldToPx(P[loop_idx[i]], view_proj);

            // Ensure CW winding for outward AA in ImGui
            float area2{0};
            for (uint8_t i = 0, j = n - 1; i < n; j = i++) area2 += hull[j].x * hull[i].y - hull[i].x * hull[j].y;
            if (area2 < 0) std::reverse(hull, hull + n);

            dl.AddConvexPolyFilled(hull, n, color);
        };

        const float base_scale = type == Type::Universal ? Style.UniversalAxisHandleScale : Style.AxisHandleScale;
        for (uint32_t i = 0; i < 3; ++i) {
            const bool is_active = g.Interaction == Interaction{Scale, AxisOp(i)};
            if (!g.Start || is_active) {
                const float handle_scale = base_scale * (g.Start ? g.Scale[i] : 1.0f);
                DrawScaleHandle(is_active, false, i, handle_scale, bool(g.Start));
                // Start-ghost
                if (g.Start) DrawScaleHandle(is_active, true, i, base_scale, true);
            }
        }
        if (g.Start && g.Interaction->Transform == Scale) {
            Label(Format::Scale(g.Interaction->Axis, g.Scale), p_px);
        }
    }
    if (type == Type::Rotate || type == Type::Universal) {
        static constexpr uint32_t HalfCircleSegmentCount{128}, FullCircleSegmentCount{HalfCircleSegmentCount * 2 + 1};
        static ImVec2 CirclePositions[FullCircleSegmentCount];
        if (g.Start && g.Interaction->Transform == Rotate) {
            {
                const auto u = glm::normalize(g.Start->MouseRayWs(IntersectPlane(g.Start->MouseRayWs, g.Start->PlaneWs)) - p_ws);
                const auto v = glm::cross(vec3{g.Start->PlaneWs}, u);
                const float r = g.WorldToSizeNdc * (g.Interaction->Axis == Screen ? (2 * Style.OuterCircleRadScale) : Style.RotationAxesCircleScale);
                const auto u_screen = WorldToPx(p_ws + u * r, view_proj) - p_px;
                const auto v_screen = WorldToPx(p_ws + v * r, view_proj) - p_px;
                FastEllipse(CirclePositions, p_px, u_screen, v_screen, g.RotationAngle >= 0);
            }
            const uint32_t angle_i = float(FullCircleSegmentCount - 1) * fabsf(g.RotationAngle) / (2 * M_PI);
            const auto angle_circle_pos = CirclePositions[angle_i + 1]; // save
            CirclePositions[angle_i + 1] = p_px;
            dl.AddConvexPolyFilled(CirclePositions, angle_i + 2, Color.RotationFillActive);

            CirclePositions[angle_i + 1] = angle_circle_pos; // restore
            const auto color = g.Interaction->Axis == Screen ? IM_COL32_WHITE : colors::Axes[AxisIndex(g.Interaction->Axis)];
            dl.AddPolyline(CirclePositions, FullCircleSegmentCount, color, false, Style.RotationLineWidth);
            dl.AddLine(p_px, CirclePositions[0], color, Style.RotationLineWidth / 2);
            dl.AddLine(p_px, CirclePositions[angle_i], color, Style.RotationLineWidth);
            Label(Format::Rotation(g.Interaction->Axis, g.RotationAngle), CirclePositions[1]);
        } else if (!g.Start) {
            // Half-circles facing the camera
            const float r = g.WorldToSizeNdc * Style.RotationAxesCircleScale;
            const vec3 cam_to_model = mat3{model.Inv} * glm::normalize(p_ws - cam_origin);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float angle_start = M_PI_2 + atan2f(cam_to_model[(4 - axis) % 3], cam_to_model[(3 - axis) % 3]);
                const vec4 axis_start{cosf(angle_start), sinf(angle_start), 0.f, 0.f};
                const vec4 axis_offset{-axis_start.y, axis_start.x, 0.f, 0.f};
                const vec3 u_local{axis_start[axis], axis_start[(axis + 1) % 3], axis_start[(axis + 2) % 3]};
                const vec3 v_local{axis_offset[axis], axis_offset[(axis + 1) % 3], axis_offset[(axis + 2) % 3]};
                const auto u_screen = WorldToPx(u_local * r, g.MVP) - p_px;
                const auto v_screen = WorldToPx(v_local * r, g.MVP) - p_px;
                FastEllipse(std::span{CirclePositions}.first(HalfCircleSegmentCount + 1), p_px, u_screen, v_screen, true, 0.5f);
                const auto color = SelectionColor(colors::Axes[2 - axis], g.Interaction == Interaction{Rotate, AxisOp(2 - axis)});
                dl.AddPolyline(CirclePositions, HalfCircleSegmentCount + 1, color, false, Style.RotationLineWidth);
            }
        }
    }
}
} // namespace

namespace ModelGizmo {
bool Draw(Mode mode, Type type, vec2 pos, vec2 size, vec2 mouse_px, ray mouse_ray_ws, mat4 &m, const mat4 &view, const mat4 &proj, std::optional<vec3> snap) {
    g.ScreenRect = {std::bit_cast<ImVec2>(pos), std::bit_cast<ImVec2>(pos + size)};
    // Scale is always local or m will be skewed when applying world scale or rotated m
    if (type == Type::Scale || type == Type::Universal) mode = Mode::Local;

    const Model model{m, mode};
    const mat4 view_proj = proj * view;
    g.MVP = view_proj * model.M;
    // Behind‐camera cull
    if (!g.Start && g.MVP[3].z < 0.001f) return false;

    const auto view_inv = InverseRigid(view);
    const ray camera_ray{Pos(view_inv), Dir(view_inv)};

    // Compute scale from camera right vector projected onto screen at model position.
    g.WorldToSizeNdc = Style.SizeNdc / glm::length(ClipToNdc(view_proj * vec4{Pos(model.RT) + vec3{Right(view_inv)}, 1}) - ClipToNdc(view_proj * vec4{Pos(model.RT), 1}));
    const bool commit = g.Start && !ImGui::IsMouseDown(ImGuiMouseButton_Left);

    if (g.Start) {
        assert(g.Interaction);
        m = Transform(m, model, mode, *g.Interaction, mouse_px, mouse_ray_ws, snap);
        if (commit) g.Start = {};
    } else {
        g.Interaction = std::nullopt;
        if (ImGui::IsWindowHovered()) {
            if (g.Interaction = FindHoveredInteraction(model, type, std::bit_cast<ImVec2>(mouse_px), mouse_ray_ws, view, view_proj);
                g.Interaction && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const auto GetPlaneNormal = [&camera_ray, &model, mode](const Interaction &i) -> vec4 {
                    using enum TransformType;
                    if (i.Axis == Screen) return -vec4{camera_ray.d, 0};
                    if (auto plane_index = TranslatePlaneIndex(i.Axis)) return model.M[*plane_index];

                    const auto index = AxisIndex(i.Axis);
                    if (i.Transform == Scale) return model.M[(index + 1) % 3];
                    if (i.Transform == Rotate) return mode == Mode::Local ? model.M[index] : vec4{Axes[index], 0};

                    const auto n = vec3{model.M[index]};
                    const auto v = glm::normalize(Pos(model.RT) - camera_ray.o);
                    return vec4{v - n * glm::dot(n, v), 0};
                };
                g.Start = state::StartContext{
                    .M = m,
                    .PlaneWs = BuildPlane(Pos(model.RT), GetPlaneNormal(*g.Interaction)),
                    .MousePx = mouse_px,
                    .MouseRayWs = mouse_ray_ws,
                    .WorldToSizeNdc = g.WorldToSizeNdc
                };
                g.Scale = {1, 1, 1};
                g.RotationAngle = 0;
            }
        }
    }

    Render(model, type, view_proj, camera_ray.o);
    return g.Start || commit;
}
} // namespace ModelGizmo
