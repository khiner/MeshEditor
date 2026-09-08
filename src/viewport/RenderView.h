#pragma once

#include "gpu/SceneViewUBO.h"
#include "viewport/CameraView.h"

// Camera inputs retained for the rendered frame, including selection queries.
struct RenderView {
    CameraView Camera;
    vec2 Extent;

    bool operator==(const RenderView &) const = default;

    void ApplyTo(SceneViewUBO &) const;
};
