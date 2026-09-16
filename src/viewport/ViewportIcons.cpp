#include "viewport/ViewportIcons.h"

#include "Paths.h"
#include "render/SvgResource.h"

#include "state/Scene.h"

void LoadViewportIcons(state::Scene &r) {
    const auto dir = Paths::Res() / "svg";
    const auto &ctx = r.ctx().get<const mtl::Context>();
    r.ctx().emplace<ViewportIcons>(
        ViewportIcons{
            .Transform = {
                .Select = LoadSvg(ctx, dir / "select.svg"),
                .SelectBox = LoadSvg(ctx, dir / "select_box.svg"),
                .Move = LoadSvg(ctx, dir / "move.svg"),
                .Rotate = LoadSvg(ctx, dir / "rotate.svg"),
                .Scale = LoadSvg(ctx, dir / "scale.svg"),
                .Universal = LoadSvg(ctx, dir / "transform.svg"),
            },
            .Shading = {
                .Wireframe = LoadSvg(ctx, dir / "shading_wire.svg"),
                .Solid = LoadSvg(ctx, dir / "shading_solid.svg"),
                .MaterialPreview = LoadSvg(ctx, dir / "shading_texture.svg"),
                .Rendered = LoadSvg(ctx, dir / "shading_rendered.svg"),
            },
            .Overlay = LoadSvg(ctx, dir / "overlay.svg"),
            .XRay = LoadSvg(ctx, dir / "xray.svg"),
            .Anim = {
                .Play = LoadSvg(ctx, dir / "play.svg"),
                .PlayReverse = LoadSvg(ctx, dir / "play_reverse.svg"),
                .Pause = LoadSvg(ctx, dir / "pause.svg"),
                .JumpStart = LoadSvg(ctx, dir / "jump_start.svg"),
                .JumpEnd = LoadSvg(ctx, dir / "jump_end.svg"),
                .FramePrev = LoadSvg(ctx, dir / "frame_prev.svg"),
                .FrameNext = LoadSvg(ctx, dir / "frame_next.svg"),
                .PrevKeyframe = LoadSvg(ctx, dir / "prev_keyframe.svg"),
                .NextKeyframe = LoadSvg(ctx, dir / "next_keyframe.svg"),
            },
        }
    );
}
