#pragma once

#include "Camera.h"
#include "gpu/SceneViewUBO.h"
#include "viewport/CameraView.h"

// Camera inputs retained for the rendered frame, including selection queries.
struct RenderView {
    CameraView Camera;
    vec2 Extent;

    bool operator==(const RenderView &) const = default;

    void ApplyTo(SceneViewUBO &view) const {
        const float aspect = Extent.x == 0 || Extent.y == 0 ? 1.f : Extent.x / Extent.y;
        const auto proj = Camera.Projection(aspect);
        const auto camera_view = Camera.View();
        view.ViewProj = proj * camera_view;
        view.ViewRotation = mat3(camera_view);
        view.CameraPosition = Camera.Position();
        view.CameraNear = Camera.NearClip();
        view.CameraFar = Camera.FarClip();
        // Positive scales with perspective depth; negative is an absolute orthographic pixel size.
        view.ScreenPixelScale = ScreenPixelScale(Camera.Data, std::max(Extent.y, 1.f));
        view.ViewportSize = Extent;
        // Polygon offset matching Blender's GPU_polygon_offset_calc.
        view.NdcOffsetFactor = std::holds_alternative<Perspective>(Camera.Data) ? proj[3][2] * -0.00125f : 0.000005f * std::max(std::abs(1.f / proj[0][0]), std::abs(1.f / proj[1][1]));
    }
};
