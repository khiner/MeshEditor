#pragma once

#include "gpu/SceneViewUBO.h"
#include "viewport/CameraView.h"
#include "viewport/ViewportDisplay.h"

// The view and display inputs retained for the rendered frame, including selection queries.
struct RenderView {
    CameraView Camera;
    vec2 Extent;
    ViewportDisplay Display;

    bool operator==(const RenderView &) const = default;

    void ApplyTo(SceneViewUBO &) const;
};
