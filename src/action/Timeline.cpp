#include "action/Timeline.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "gltf/SourceAssets.h"
#include "render/LightComponents.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

namespace action::timeline {
void Apply(entt::registry &r, entt::entity viewport, const Action &action) {
    const auto enter_presentation = [&] {
        // Present the scene as authored: Rendered mode when the scene defines its own lighting
        // (punctual lights or IBL), else material preview showing the default world at full opacity.
        const auto *source_assets = r.try_get<const gltf::SourceAssets>(viewport);
        const bool explicit_ibl = source_assets && source_assets->ImageBasedLight.has_value();
        const bool authored_lighting = explicit_ibl || !r.storage<LightIndex>().empty();
        const auto mode = authored_lighting ? ViewportShadingMode::Rendered : ViewportShadingMode::MaterialPreview;
        r.patch<ViewportDisplay>(viewport, [&](auto &s) { s.ViewportShading = s.FillMode = mode; s.ShowOverlays = false; });
        if (!authored_lighting && r.all_of<MaterialPreviewLighting>(viewport)) {
            r.patch<MaterialPreviewLighting>(viewport, [](auto &l) { l.WorldOpacity = 1.f; });
        }
        if (authored_lighting && r.all_of<RenderedLighting>(viewport)) {
            r.patch<RenderedLighting>(viewport, [&](auto &l) {
                if (explicit_ibl) {
                    l.BackgroundBlur = 0.f;
                } else {
                    l.UseSceneWorld = false;
                    l.WorldOpacity = 1.f;
                }
            });
        }
    };
    std::visit(
        overloaded{
            [&](EnterPresentation) { enter_presentation(); },
            [&](StartPresentation) {
                enter_presentation();
                r.patch<TimelinePlayback>(viewport, [](auto &p) { p.Playing = true; });
            },
            [&](const TogglePlay &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.Playing = !p.Playing; p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetFrame &a) {
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = a.Frame; });
                r.get<PlaybackFrame>(viewport).Value = a.Frame;
            },
            [&](const SetStartFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.StartFrame = a.Frame; }); },
            [&](const SetEndFrame &a) { r.patch<TimelineRange>(viewport, [&](auto &range) { range.EndFrame = a.Frame; }); },
            [&](JumpToStart) { JumpToStartFrame(r, viewport); },
            [&](JumpToEnd) {
                const auto frame = r.get<const TimelineRange>(viewport).EndFrame;
                r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = frame; });
                r.get<PlaybackFrame>(viewport).Value = frame;
            },
            [&](const SetView &a) { r.replace<AnimationTimelineView>(viewport, AnimationTimelineView{a.PixelsPerFrame, a.ViewCenterFrame}); },
        },
        action
    );
}
} // namespace action::timeline
