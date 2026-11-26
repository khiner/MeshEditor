#pragma once

#include "gpu/MaterialPattern.h"
#include "gpu/MaterialPreview.h"
#include "gpu/SceneViewUBO.h"
#include "metal/Image.h"

#include <memory>
#include <optional>

namespace state {
struct Registry;
} // namespace state

// The display image of a material preview, which lives while a holder keeps it.
struct PreviewImage {
    mtl::Texture Display;
};
// A program a preview evaluates at the sphere's points in place of the material's lighting: its record, whose words are its outputs, and how its display shows them.
struct PreviewValue {
    MaterialPattern Program;
    PreviewDisplay Display;
};
// The view of the preview sphere from a fixed camera at the extent, lit by the default studio environment alone, with the slots of `base`.
SceneViewUBO PreviewView(state::Registry &, SceneViewUBO base, mtl::Extent2D);
// Renders the preview sphere seen by `view` at the extent and waits for it: the material record lit by the lit function with its pattern's computed inputs, or the value's program shown at each point.
// The sphere is opaque, so a transmissive material shades without its transmission.
std::shared_ptr<const PreviewImage> RenderMaterialPreview(state::Registry &, SceneViewUBO view, mtl::Extent2D, uint32_t material, const std::optional<PreviewValue> &);
