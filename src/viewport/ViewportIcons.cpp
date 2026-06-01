#include "viewport/ViewportIcons.h"

#include "Paths.h"
#include "render/SvgUpload.h"

#include <entt/entity/registry.hpp>

void LoadViewportIcons(entt::registry &r) {
    const auto dir = Paths::Res() / "svg";
    SvgUploadBatch batch{r};
    r.ctx().emplace<ViewportIcons>(
        ViewportIcons{
            .Transform = {
                .Select = LoadSvg(batch, dir / "select.svg"),
                .SelectBox = LoadSvg(batch, dir / "select_box.svg"),
                .Move = LoadSvg(batch, dir / "move.svg"),
                .Rotate = LoadSvg(batch, dir / "rotate.svg"),
                .Scale = LoadSvg(batch, dir / "scale.svg"),
                .Universal = LoadSvg(batch, dir / "transform.svg"),
            },
            .Shading = {
                .Wireframe = LoadSvg(batch, dir / "shading_wire.svg"),
                .Solid = LoadSvg(batch, dir / "shading_solid.svg"),
                .MaterialPreview = LoadSvg(batch, dir / "shading_texture.svg"),
                .Rendered = LoadSvg(batch, dir / "shading_rendered.svg"),
            },
            .Overlay = LoadSvg(batch, dir / "overlay.svg"),
            .Anim = {
                .Play = LoadSvg(batch, dir / "play.svg"),
                .Pause = LoadSvg(batch, dir / "pause.svg"),
                .JumpStart = LoadSvg(batch, dir / "jump_start.svg"),
                .JumpEnd = LoadSvg(batch, dir / "jump_end.svg"),
            },
        }
    );
    SubmitSvgUpload(batch);
}
