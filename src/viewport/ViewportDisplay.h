#pragma once

#include "Field.h"
#include "gpu/DebugChannel.h"
#include "gpu/InteractionMode.h"

#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

#include <algorithm>
#include <string>

enum class ViewportShadingMode : uint8_t {
    Wireframe,
    Solid,
    MaterialPreview,
    Rendered,
};

enum class AnisotropicFilterLevel : uint8_t {
    Off,
    X2,
    X4,
    X8,
    X16
};

// Levels are consecutive powers of two from Off->1 through X16->16.
constexpr float ToMaxAnisotropy(AnisotropicFilterLevel level) { return float(1u << unsigned(level)); }

enum class MotionBlurMethod : uint8_t {
    Fast,
    FullSampling,
};

struct MotionBlur {
    float Shutter{0.5f};
    uint8_t Steps{16};
    MotionBlurMethod Method{MotionBlurMethod::Fast};
};
template<> inline constexpr field::SpecTable<MotionBlur> Specs<MotionBlur>{
    {&MotionBlur::Shutter, {.Min = 0, .Max = 2, .Unit = FieldUnit::Frames, .Label = "Shutter", .Help = "Time the shutter stays open"}},
    {&MotionBlur::Steps, {.Min = 1, .Max = 64, .Label = "Samples", .Help = "Number of renders across the shutter with Full Sampling. More samples reduce stepping."}},
    {&MotionBlur::Method, {.Help = "Fast blurs one render with motion vectors and approximates changing visibility, reflections, and lighting. Full Sampling renders across the shutter at higher cost."}},
};

struct ViewportOverlays {
    bool Grid{true}, Extras{true}, Bones{true}, Origins{true}, OutlineSelected{true};
};
template<> inline constexpr field::SpecTable<ViewportOverlays> Specs<ViewportOverlays>{
    {&ViewportOverlays::Grid, {.Help = "Ground grid and world axes"}},
    {&ViewportOverlays::Extras, {.Help = "Camera, light, empty, and collider wires"}},
    {&ViewportOverlays::Bones, {.Help = "Armature bones"}},
    {&ViewportOverlays::Origins, {.Help = "Origin points of selected and active objects"}},
    {&ViewportOverlays::OutlineSelected, {.Help = "Outline around selected objects"}},
};

// Overlays on selected objects.
struct SelectionOverlays {
    bool VertexNormals{false}, FaceNormals{false}, BoundingBoxes{false}, TetWireframe{false};
};
template<> inline constexpr field::SpecTable<SelectionOverlays> Specs<SelectionOverlays>{
    {&SelectionOverlays::VertexNormals, {.Help = "Normal line at each vertex of selected meshes"}},
    {&SelectionOverlays::FaceNormals, {.Help = "Normal line at each face center of selected meshes"}},
    {&SelectionOverlays::BoundingBoxes, {.Help = "Bounding box of each selected mesh"}},
    {&SelectionOverlays::TetWireframe, {.Help = "Edges of the tetrahedral mesh of each selected object that has one"}},
};

// Changes require command-buffer recording.
struct ViewportDisplay {
    ViewportShadingMode ViewportShading{ViewportShadingMode::Solid};
    ViewportShadingMode FillMode{ViewportShadingMode::Solid};
    bool ShowOverlays{true};
    ViewportOverlays Overlays;
    SelectionOverlays Selection;
    // Screen-space error budget for the cluster LOD cut, in pixels.
    // Zero renders original geometry alone.
    float LodErrorPixels{1.f};
    DebugChannel DebugChannel{DebugChannel::None};
    AnisotropicFilterLevel AnisotropicFilter{AnisotropicFilterLevel::X16};
    std::optional<MotionBlur> MotionBlur;
    bool XRaySolid{false}, XRayWireframe{true};
    float XRayAlpha{0.5f}, XRayAlphaWireframe{0.f};
};
template<> inline constexpr field::SpecTable<ViewportDisplay> Specs<ViewportDisplay>{
    {&ViewportDisplay::ViewportShading, {.Help = "Wireframe and Solid shade with the workspace lights, Material Preview with a studio environment, and Rendered with the scene lights"}},
    {&ViewportDisplay::FillMode, {.Help = "Shading restored when wireframe is toggled off"}},
    {&ViewportDisplay::ShowOverlays, {.Help = "Draw the viewport and selection overlays"}},
    {&ViewportDisplay::Overlays, {.Help = "Overlays drawn in the viewport"}},
    {&ViewportDisplay::Selection, {.Label = "Selection overlays", .Help = "Overlays drawn on selected objects"}},
    {&ViewportDisplay::LodErrorPixels, {.Unit = FieldUnit::Pixels, .Label = "LOD error", .Help = "Largest screen-space level-of-detail error. Zero renders the original geometry."}},
    {&ViewportDisplay::DebugChannel, {.Help = "Material or geometry channel shown in place of the shaded color"}},
    {&ViewportDisplay::AnisotropicFilter, {.Label = "Anisotropic filtering", .Help = "Highest anisotropic filtering level. Higher levels sharpen textures viewed at grazing angles."}},
    {&ViewportDisplay::MotionBlur, {.Help = "Blur of moving objects across the shutter interval. Absent disables it."}},
    {&ViewportDisplay::XRaySolid, {.Label = "X-ray solid", .Help = "See through surfaces in Solid shading"}},
    {&ViewportDisplay::XRayWireframe, {.Label = "X-ray wireframe", .Help = "See through surfaces in Wireframe shading"}},
    {&ViewportDisplay::XRayAlpha, {.Min = 0, .Max = 1, .Label = "X-ray opacity", .Help = "Surface opacity with X-ray in Solid shading"}},
    {&ViewportDisplay::XRayAlphaWireframe, {.Min = 0, .Max = 1, .Label = "X-ray wireframe opacity", .Help = "Surface opacity with X-ray in Wireframe shading"}},
};

// Wireframe and Solid shade with the workspace lights and support X-ray display.
constexpr bool WorkbenchShading(ViewportShadingMode mode) { return mode == ViewportShadingMode::Wireframe || mode == ViewportShadingMode::Solid; }
// The X-ray flag of the current shading family.
// Edit-mode element selection follows it directly.
constexpr bool XRayFlag(const ViewportDisplay &d) { return d.ViewportShading == ViewportShadingMode::Wireframe ? d.XRayWireframe : d.XRaySolid; }
constexpr float XRayOpacity(const ViewportDisplay &d) { return d.ViewportShading == ViewportShadingMode::Wireframe ? d.XRayAlphaWireframe : d.XRayAlpha; }
// X-ray display: the flag is set, the opacity is below one, and the shading is workbench.
constexpr bool XRayActive(const ViewportDisplay &d) { return WorkbenchShading(d.ViewportShading) && XRayFlag(d) && XRayOpacity(d) < 1.f; }
// Strength of overlays behind the scene surface.
// Zero leaves them to the depth test.
// X-ray display fades them by its opacity, and Edit mode with the flag draws back wires at half strength, as in Blender.
constexpr float OverlayBehindOpacity(const ViewportDisplay &d, InteractionMode mode) {
    if (XRayActive(d)) return 1.f - XRayOpacity(d);
    return mode == InteractionMode::Edit && XRayFlag(d) ? 0.5f : 0.f;
}

constexpr MotionBlur EffectiveMotionBlur(const ViewportDisplay &d) { return d.MotionBlur.value_or(MotionBlur{}); }
constexpr uint32_t MotionBlurSteps(const ViewportDisplay &d) { return std::clamp(uint32_t(EffectiveMotionBlur(d).Steps), 1u, 64u); }

struct PBRViewportLighting {
    bool UseSceneLights, UseSceneWorld;
    float EnvIntensity, EnvRotationDegrees;
    float BackgroundBlur{0.5f}, WorldOpacity{0.f};
    // Sample a mipmapped scene framebuffer at the refracted exit point for transmission.
    bool RealTransmission{true};
    // Exposure in EV stops.
    // Scales linear color by 2^EV before tone mapping.
    float ExposureEV{0.f};
};
// Exposure stays within 10 EV of zero so 2^EV stays finite.
template<> inline constexpr field::SpecTable<PBRViewportLighting> Specs<PBRViewportLighting>{
    {&PBRViewportLighting::UseSceneLights, {.Label = "Scene lights", .Help = "Light with the scene's punctual lights"}},
    {&PBRViewportLighting::UseSceneWorld, {.Label = "Scene world", .Help = "Light with the scene's image-based light instead of a studio environment"}},
    {&PBRViewportLighting::EnvIntensity, {.Min = 0, .Max = 2, .Label = "Intensity", .Help = "Multiplier on the studio environment's lighting"}},
    {&PBRViewportLighting::EnvRotationDegrees, {.Min = -180, .Max = 180, .Unit = FieldUnit::Degrees, .Label = "Rotation", .Help = "Rotation of the studio environment about the vertical axis"}},
    {&PBRViewportLighting::BackgroundBlur, {.Min = 0, .Max = 1, .Label = "Blur", .Help = "Blur of the background environment"}},
    {&PBRViewportLighting::WorldOpacity, {.Min = 0, .Max = 1, .Label = "World opacity", .Help = "Opacity of the background environment"}},
    {&PBRViewportLighting::RealTransmission, {.Label = "Real transmission", .Help = "Sample transmission from a pre-rendered scene image instead of the image-based light"}},
    {&PBRViewportLighting::ExposureEV, {.Min = -10, .Max = 10, .Unit = FieldUnit::Stops, .Label = "Exposure", .Help = "Scales linear color by 2^EV before tone mapping"}},
};

struct MaterialPreviewLighting {
    PBRViewportLighting Value;
};
struct RenderedLighting {
    PBRViewportLighting Value;
};

// The active studio HDRI environment, by source name so it stays stable across runs (unlike the directory-scan index).
struct StudioEnvironment {
    std::string Name;
};

const PBRViewportLighting &GetActivePbrLighting(const state::Registry &, state::Entity viewport, ViewportShadingMode);

struct ViewportExtent {
    uvec2 Value{};
};
