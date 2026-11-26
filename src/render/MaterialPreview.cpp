#include "render/MaterialPreview.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "mesh/Primitives.h"
#include "numeric/QuaternionMath.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/MaterialPatterns.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "state/Registry.h"
#include "viewport/RenderView.h"
#include "viewport/Viewport.h"

namespace Format = mtl::Format;

namespace {
// The corners of the triangles of the UV sphere primitive of 32 slices and 16 stacks, each face split along its first corner's diagonals, with the primitive's texture coordinates.
// Each vertex's normal weights the unit normals of its faces by the corner angles, as a smooth mesh's vertex normals do.
// Each corner's tangent runs along increasing U, and its bitangent along increasing V.
std::vector<PreviewCorner> SphereCorners() {
    const auto sphere = primitive::CreateMesh(primitive::UVSphere{.Radius = 1.f, .Slices = 32u, .Stacks = 16u}, true);
    const auto &mesh = sphere.Data;
    std::vector<vec3> normals(mesh.Positions.size(), vec3{0});
    for (uint32_t f = 0; f < mesh.FaceCount(); ++f) {
        const auto face = mesh.Face(f);
        const auto at = [&](size_t c) { return mesh.Positions[face[c % face.size()]]; };
        vec3 n{0};
        for (size_t c = 2; c < face.size(); ++c) n += numeric::Cross(at(c - 1u) - at(0u), at(c) - at(0u));
        n = numeric::Normalize(n);
        for (size_t c = 0; c < face.size(); ++c) {
            const auto e_prev = at(c + face.size() - 1u) - at(c), e_next = at(c + 1u) - at(c);
            normals[face[c]] += n * std::acos(std::clamp(numeric::Dot(e_prev, e_next) / (numeric::Length(e_prev) * numeric::Length(e_next)), -1.f, 1.f));
        }
    }
    std::vector<PreviewCorner> corners;
    for (uint32_t f = 0; f < mesh.FaceCount(); ++f) {
        const auto face = mesh.Face(f);
        const auto corner = [&](size_t c) {
            const auto uv = sphere.CornerTexCoords0[mesh.FaceStart(f) + c];
            return PreviewCorner{.Position = mesh.Positions[face[c]], .Normal = numeric::Normalize(normals[face[c]]), .Tangent = vec3{__sinpi(2.f * uv.x), 0, __cospi(2.f * uv.x)}, .TangentSign = -1.f, .Uv = uv};
        };
        for (size_t c = 2; c < face.size(); ++c) corners.append_range(std::array{corner(0u), corner(c - 1u), corner(c)});
    }
    return corners;
}

MaterialPatterns::PreviewPasses &PreviewPipelines(state::Registry &r) {
    auto &patterns = r.Resources.get<MaterialPatterns>();
    if (!patterns.Previews) {
        auto &libraries = r.Resources.get<mtl::LibraryCache>();
        const auto corners = SphereCorners();
        mtl::Buffer corner_buffer{r.Resources.get<GpuBuffers>().Ctx, std::span{corners}.size_bytes()};
        corner_buffer.Update(std::as_bytes(std::span{corners}));
        patterns.Previews.emplace(
            mtl::MakeRenderPipeline(libraries, {"MaterialPreview.metal", "PreviewSphereVertex"}, mtl::FunctionRef{"MaterialPreview.metal", "PreviewSphereFragment"}, {{Format::Uint}, Format::Depth}, {mtl::NoBlend}, mtl::DepthState{}),
            mtl::ComputePipeline{libraries, {"MaterialPreview.metal", "MaterialPreviewDisplay"}}, std::move(corner_buffer), uint32_t(corners.size())
        );
    }
    return *patterns.Previews;
}
const mtl::RenderPipeline &LitPipeline(state::Registry &r, MaterialPatterns::PreviewPasses &passes, PbrFeatureMask features) {
    const auto it = std::ranges::find(passes.Lit, features, &std::pair<PbrFeatureMask, mtl::RenderPipeline>::first);
    if (it != passes.Lit.end()) return it->second;
    return passes.Lit.emplace_back(features, MakePbrPreviewPipeline(r.Resources.get<mtl::LibraryCache>(), features)).second;
}
// The lobes the preview of the material record shades: those its values make live, without transmission, and those its graph's computed inputs make live.
PbrFeatureMask PreviewFeatures(const state::Registry &r, uint32_t material) {
    const auto features = PbrFeaturesOf(r.Resources.get<const GpuBuffers>().Materials.GetSpan<PBRMaterial>(), std::array{material}) & ~FeatureBit(PbrFeature::Transmission);
    const auto patterns = PatternRecords(r.Resources.get<const MaterialPatterns>());
    return features | (material < patterns.size() && patterns[material].Program.Index != InvalidOffset ? PatternFeatures(patterns[material].Inputs) : PbrFeatureMask{});
}
} // namespace

SceneViewUBO PreviewView(state::Registry &r, SceneViewUBO view, mtl::Extent2D extent) {
    const RenderView render_view{.Camera = {.Data = Perspective{.FieldOfViewRad = .7f, .FarClip = 100.f, .NearClip = .1f}, .Target = {}, .Distance = 3.6f, .Orientation = AngleAxis(-.3f, vec3{1, 0, 0})}, .Extent = vec2{float(extent.Width), float(extent.Height)}};
    render_view.ApplyTo(view);
    const auto lighting = MaterialPreviewLighting{}.Value;
    view.Ibl = StudioIbl(r, StudioEnvironment{}.Name);
    view.EnvIntensity = lighting.EnvIntensity;
    view.Exposure = std::exp2(lighting.ExposureEV);
    view.EnvRotation = mat3{1.f};
    view.BackgroundBlur = lighting.BackgroundBlur;
    view.WorldOpacity = lighting.WorldOpacity;
    view.UseSceneLightsRender = 0u;
    view.LightCount = 0u;
    view.InteractionMode = InteractionMode::Object;
    view.ShowOverlays = 0u;
    view.DebugChannel = DebugChannel::None;
    view.UseRealTransmission = 0u;
    view.RayActive = 0u;
    return view;
}

std::shared_ptr<const PreviewImage> RenderMaterialPreview(state::Registry &r, SceneViewUBO view, mtl::Extent2D extent, uint32_t material, const std::optional<PreviewValue> &value) {
    const auto &ctx = r.Resources.get<const mtl::Context>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &slots = r.Resources.get<const mtl::BindlessSet>();
    const auto &patterns = r.Resources.get<const MaterialPatterns>();
    auto &passes = PreviewPipelines(r);
    // The program the preview evaluates: the value's, or the pattern of a material whose graph computes inputs.
    const auto records = PatternRecords(patterns);
    const auto pattern = value ? std::optional{value->Program} : material < records.size() && records[material].Program.Index != InvalidOffset ? std::optional{records[material]} : std::nullopt;
    const auto triangles = mtl::CreateTexture2D(ctx, Format::Uint, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    const auto depth = mtl::CreateTexture2D(ctx, Format::Depth, extent, MTL::TextureUsageRenderTarget);
    const auto radiance = value ? mtl::Texture{} : mtl::CreateTexture2D(ctx, Format::HdrColor, extent, MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    auto image = std::make_shared<PreviewImage>(mtl::CreateTexture2D(ctx, Format::Color, extent, MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead));
    const auto words = pattern ? std::optional<mtl::Buffer>{std::in_place, buffers.Ctx, (PatternWordsHeader + uint64_t(extent.Width) * extent.Height * pattern->Words) * sizeof(uint32_t), SlotType::Buffer, mtl::BufferLifetime::Workspace} : std::nullopt;
    view.PatternSlot = patterns.Records ? patterns.Records->Slot : InvalidSlot;
    view.PatternWordSlot = words ? words->Slot : InvalidSlot;
    view.PatternStride = pattern ? pattern->Words : 0u;
    view.PatternWidth = extent.Width;
    const MaterialPreviewPushConstants pc{.Material = material, .Display = value ? value->Display : PreviewDisplay::Lit};
    const auto *viewport = r.view<const Viewport>().empty() ? nullptr : &r.get<const Viewport>(*r.view<const Viewport>().begin());
    auto command = ctx.Begin();
    {
        mtl::PassChain chain{command};
        const auto bind = [&] {
            command.SetBuffer(slots.Table(), 0, BufferIndex_Bindless);
            command.SetBytes(view, BufferIndex_SceneView);
            command.SetBuffer(*buffers.ViewportThemeUBO, 0, BufferIndex_ViewportTheme);
            if (viewport) command.SetBuffer(*viewport->WorkspaceLightsUBO, 0, BufferIndex_WorkspaceLights);
            command.SetTexture(*buffers.EeveeLut, 15);
            command.SetBuffer(*passes.Corners, 0, BufferIndex_PreviewCorners);
        };
        const std::array ids{mtl::ClearColor(*triangles, MTL::ClearColor{double(InvalidOffset), 0, 0, 0})};
        const auto sphere = mtl::MakePassDescriptor(ids, mtl::ClearDepth(*depth));
        auto render = chain.BeginRender(sphere.get(), "MaterialPreviewSphere");
        bind();
        passes.Sphere.Bind(render);
        render->setViewport({0.0, 0.0, double(extent.Width), double(extent.Height), 0.0, 1.0});
        render->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(passes.CornerCount));
        if (pattern) {
            if (!passes.Driver) passes.Driver = &mtl::AddDriver(r.Resources.get<mtl::ProgramRegistry>(), mtl::ComputePipeline{r.Resources.get<mtl::LibraryCache>(), {.Path = "SurfacePattern.metal", .Name = "SurfacePatternPreviewDriver", .BinaryLinking = true}});
            const auto compute = chain.BeginCompute("MaterialPreviewPattern", MTL::StageFragment);
            bind();
            command.SetBytes(*pattern, BufferIndex_PushConstants);
            command.SetTexture(*triangles, 0);
            mtl::Bind(compute, *passes.Driver, BufferIndex_Programs);
            compute->dispatchThreads({extent.Width, extent.Height, 1}, ThreadgroupSize::Tile8);
        }
        if (!value) {
            const std::array colors{mtl::ClearColor(*radiance)};
            const auto lit = mtl::MakePassDescriptor(colors);
            auto shade = chain.BeginRender(lit.get(), "MaterialPreviewLit", {{MTL::StageFragment | MTL::StageDispatch, MTL::StageFragment}});
            bind();
            command.SetFragmentBytes(pc, BufferIndex_PushConstants);
            command.SetTexture(*triangles, 0);
            LitPipeline(r, passes, PreviewFeatures(r, material)).Bind(shade);
            shade->setViewport({0.0, 0.0, double(extent.Width), double(extent.Height), 0.0, 1.0});
            shade->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
        }
        const auto display = chain.BeginCompute("MaterialPreviewDisplay", MTL::StageFragment | MTL::StageDispatch);
        bind();
        command.SetBytes(pc, BufferIndex_PushConstants);
        command.SetTexture(*triangles, 0);
        if (!value) command.SetTexture(*radiance, 1);
        command.SetTexture(*image->Display, 2);
        passes.Display.Bind(display);
        display->dispatchThreads({extent.Width, extent.Height, 1}, ThreadgroupSize::Tile8);
    }
    (void)ctx.Wait(ctx.Commit(std::move(command)));
    return image;
}
