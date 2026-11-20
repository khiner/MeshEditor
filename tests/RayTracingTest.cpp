#include "File.h"
#include "ProjectFixture.h"
#include "animation/Fields.h"
#include "gltf/GltfScene.h"
#include "image/ImageDecode.h"
#include "image/ImageEncode.h"
#include "render/MaterialComponents.h"
#include "render/ViewportSubmission.h"

#include <cmath>

namespace {
namespace fs = std::filesystem;

constexpr mtl::Extent2D Preview{512, 320};
const fs::path Source{MESHEDITOR_SOURCE_DIR};
const auto References = Source / "tests/fixtures/ray-tracing/reference";

enum class FixtureScene { Studio,
                          Vase,
                          Glass,
                          MixedGlass,
                          Lights64,
                          Blocker,
                          Emitter,
                          Skin };
constexpr auto GlassPath = "res/examples/RayTracingGlass/RayTracingGlass.gltf", AreaPath = "tests/fixtures/RayTracingArea.gltf";
constexpr std::pair<FixtureScene, const char *> Scenes[]{
    {FixtureScene::Studio, "res/examples/RayTracing/RayTracing.gltf"},
    {FixtureScene::Vase, "res/examples/RayTracingVase/RayTracingVase.gltf"},
    {FixtureScene::MixedGlass, GlassPath},
    {FixtureScene::Skin, "tests/fixtures/ray-tracing/MovingReflection.gltf"},
};
// The native documents of script/BenchmarkRayTracing, by case name.
constexpr std::tuple<FixtureScene, const char *, const char *> BenchmarkScenes[]{
    {FixtureScene::Glass, GlassPath, "glass"},
    {FixtureScene::MixedGlass, GlassPath, "mixed"},
    {FixtureScene::Lights64, AreaPath, "lights-64"},
    {FixtureScene::Blocker, AreaPath, "blocker"},
    {FixtureScene::Emitter, AreaPath, "emitter"},
};

state::Entity Node(const Fixture &f, uint32_t index) {
    for (const auto [entity, node] : f.R.view<const GltfNode>().each())
        if (node.Index == index) return entity;
    throw std::runtime_error(std::format("Missing glTF node {}", index));
}

void Load(Fixture &f, FixtureScene scene, const char *path, mtl::Extent2D extent) {
    f.Do(action::io::LoadGltf{Source / path});
    const auto viewport = FirstViewport(f.R);
    f.R.replace<ViewportPanel>(viewport, ViewportPanel{.Extent = {extent.Width, extent.Height}});
    for (const auto [entity, node] : f.R.view<const GltfNode>().each())
        if (node.CameraName == "Camera") f.Do(action::view::SetLookThroughCamera{viewport, entity});
    f.Do(action::view::SetViewportShading{viewport, scene == FixtureScene::Vase || scene == FixtureScene::Studio ? ViewportShadingMode::MaterialPreview : ViewportShadingMode::Rendered});
    f.Do(action::UpdateOf<&ViewportDisplay::ShowOverlays>(viewport, false));
    f.Do(action::UpdateOf<&ViewportDisplay::RayTracing, &RayTracingSettings::Enabled>(viewport, true));
    if (scene == FixtureScene::Studio) {
        // Behind the camera, the blue object is visible only in the studio mirror.
        f.Do(action::UpdateOf<&Transform::P>(Node(f, 2), vec3{2, 2, 11}));
    } else if (scene == FixtureScene::Glass || scene == FixtureScene::MixedGlass) {
        const auto light = *f.R.view<const SceneLight>().begin();
        f.Do(action::object::SetLightType{LightType::Rectangle, light});
        f.Do(action::UpdateOf<&SceneLight::Size>(light, vec2{2.5f, 1.5f}));
        f.Do(action::UpdateOf<&SceneLight::Intensity>(light, 500.f));
        f.Do(action::UpdateOf<&ViewportDisplay::RayTracing, &RayTracingSettings::ShadowSamples>(viewport, 4u));
        if (scene == FixtureScene::MixedGlass) {
            const auto materials = f.R.view<const Material>();
            const auto glass = std::ranges::find_if(materials, [&](auto e) { return f.R.get<const Material>(e).Name == "Clear glass"; });
            if (glass == materials.end()) throw std::runtime_error("Missing clear glass sphere material");
            f.Do(action::object::UpdateMaterial<float>{*glass, offsetof(PBRMaterial, Transmission) + offsetof(Transmission, Factor), 0.f});
            f.Do(action::object::UpdateMaterial<MaterialAlphaMode>{*glass, offsetof(PBRMaterial, AlphaMode), MaterialAlphaMode::Blend});
            f.Do(action::object::UpdateMaterial<vec4>{*glass, offsetof(PBRMaterial, BaseColorFactor), {.6f, .9f, 1.f, .4f}});
        }
    } else if (scene == FixtureScene::Lights64 || scene == FixtureScene::Blocker || scene == FixtureScene::Emitter) {
        // One square area light, with the studio environment off.
        const auto light = *f.R.view<const SceneLight>().begin();
        f.Do(action::object::SetLightType{LightType::Rectangle, light});
        f.Do(action::UpdateOf<&SceneLight::Size>(light, vec2{2.f, 2.f}));
        f.Do(action::UpdateOf<&RenderedLighting::Value, &PBRViewportLighting::EnvIntensity>(viewport, 0.f));
        if (scene == FixtureScene::Lights64) {
            // 64 coincident copies split the light's power.
            f.Do(action::UpdateOf<&SceneLight::Intensity>(light, 100.f / 64));
            f.Do(action::selection::Select{light});
            for (int i = 1; i < 64; ++i) f.Do(action::object::Duplicate{});
        } else {
            // The blocker or the emitter moves 2.4 units along x over frames 1 to 49.
            const auto target = scene == FixtureScene::Blocker ? Node(f, 1) : light;
            const auto p = f.R.get<const Transform>(target).P;
            f.Do(action::animation::AddAnimation{scene == FixtureScene::Blocker ? "Blocker translation" : "Emitter translation"});
            f.Do(action::UpdateOf<&Transform::P>(target, p - vec3{1.2f, 0, 0}));
            f.Do(action::animation::InsertKey{{target, animation::Target<&Transform::P>()}});
            f.Do(action::timeline::SetFrame{49});
            f.Do(action::UpdateOf<&PosedLocal::Value, &Transform::P>(target, p + vec3{1.2f, 0, 0}));
            f.Do(action::animation::InsertKey{{target, animation::Target<&Transform::P>()}});
            f.Do(action::timeline::SetFrame{1});
        }
    } else if (scene == FixtureScene::Skin) {
        f.Do(action::animation::SelectAnimation{0});
        f.Do(action::timeline::SetFrame{1});
    }
}

const Viewport &Draw(Fixture &f, unsigned frames) {
    const auto viewport = FirstViewport(f.R);
    f.P->Settle();
    for (unsigned i = 0; i < frames; ++i) {
        f.R.edit<Viewport>(viewport).RenderRequested = true;
        SubmitViewports(f.R);
        if (!f.R.Resources.get<const ViewportRenderResources>().RenderPending) throw std::runtime_error("No viewport frame submitted");
        WaitForRender(f.R);
    }
    const auto &view = f.R.get<const Viewport>(viewport);
    if (!RayTracingActive(view)) throw std::runtime_error("Hardware ray tracing inactive: " + view.RayTracingStatus);
    return view;
}

void SaveImage(const fs::path &path, const ViewportImageRgba8 &image) {
    fs::create_directories(path.parent_path());
    const auto png = EncodeImagePngRgba8(image.Pixels, image.Width, image.Height, path.string());
    if (!png) throw std::runtime_error(png.error());
    if (const auto saved = File::WriteAtomic(path, *png); !saved) throw std::runtime_error(saved.error());
}

void CheckImage(Fixture &f, std::string_view name, unsigned frames = 64) {
    Draw(f, frames);
    auto capture = ReadbackViewportImage(f.R, FirstViewport(f.R));
    if (!capture) throw std::runtime_error(capture.error());
    auto &image = *capture;
    const auto &actual = image.Pixels;
    const auto filename = std::format("{}.png", name);
    const auto output = fs::path{MESHEDITOR_BUILD_DIR} / "ray-tracing-failures";
    const auto expected = File::Read(References / filename).and_then([&](const auto &bytes) { return DecodeImageRgba8(bytes, filename); });
    if (!expected || expected->Width != image.Width || expected->Height != image.Height || actual.size() != expected->Pixels.size()) {
        SaveImage(output / filename, image);
        boost::ut::expect(false) << "Missing or invalid reference" << filename << "capture" << output.string();
        return;
    }
    std::vector<std::byte> diff(actual.size());
    size_t changed{};
    for (size_t i = 0; i < actual.size(); i += 4) {
        int peak{};
        for (size_t c = 0; c < 4; ++c) {
            const int delta = std::abs(int(expected->Pixels[i + c]) - int(actual[i + c]));
            peak = std::max(peak, delta);
            diff[i + c] = std::byte(std::min(255, delta * 8));
        }
        diff[i + 3] = std::byte{255};
        changed += peak > 3;
    }
    if (changed > image.Width * image.Height / 2000) {
        SaveImage(output / filename, image);
        image.Pixels = std::move(diff);
        SaveImage(output / std::format("{}-diff.png", name), image);
        boost::ut::expect(false) << filename << changed << "pixels differ by more than 3/255; images in" << output.string();
    }
}

void CheckStability(Fixture &f, const char *path, mtl::Extent2D extent) {
    const auto &ctx = f.R.Resources.get<const mtl::Context>();
    const size_t channels = size_t(extent.Width) * extent.Height * 4;
    mtl::ResidentBuffer readback{ctx, channels * sizeof(_Float16)};
    std::vector<_Float16> first;
    for (unsigned frame = 0; frame < 32; ++frame) {
        const auto &image = Draw(f, 1).Targets.Screen->SceneColorImage;
        if (image.Extent != extent || !mtl::CopyTextureRegion(ctx, image, 0, 0, extent, *readback, extent.Width * 4 * sizeof(_Float16)))
            throw std::runtime_error(std::format("Cannot read HDR frame for {}", path));
        const std::span pixels{reinterpret_cast<const _Float16 *>(readback.Contents().data()), channels};
        if (first.empty()) first.assign(pixels.begin(), pixels.end());
        for (size_t i = 0; i < channels; ++i) {
            const float value = pixels[i], reference = first[i];
            if (!std::isfinite(value) || std::abs(value - reference) > .002f * (1.f + std::abs(reference))) {
                throw std::runtime_error(std::format("{} frame {}, pixel ({}, {}), channel {}: {} vs {}", path, frame, i / 4 % extent.Width, i / 4 / extent.Width, i % 4, value, reference));
            }
        }
    }
}
} // namespace

void TestRayTracing(Fixture &f) {
    boost::ut::test("ray tracing viewport images") = [&] {
        for (const auto &[scene, path] : Scenes) {
            const TestDir dir{"ray-viewport"};
            f.Begin(dir);
            Load(f, scene, path, Preview);
            if (scene == FixtureScene::Studio) {
                CheckImage(f, "studio");
                f.Do(action::UpdateOf<&ViewportDisplay::RayTracing, &RayTracingSettings::Diffuse>(FirstViewport(f.R), DiffuseLighting::RayTraced));
                CheckImage(f, "studio-diffuse");
            } else if (scene == FixtureScene::Vase) {
                CheckImage(f, "vase");
            } else if (scene == FixtureScene::MixedGlass) {
                CheckImage(f, "glass");
                f.Do(action::UpdateOf<&Transform::P>(Node(f, 1), vec3{.8f, 2.f, 0.f}));
                CheckImage(f, "glass-moving", 1);
                CheckImage(f, "glass-settled");
            } else {
                CheckImage(f, "skin");
                f.Do(action::timeline::SetFrame{13});
                CheckImage(f, "skin-posed");
            }
            boost::ut::expect(f.P->Close());
        }
    };
    boost::ut::test("ray tracing fixed-input stability") = [&] {
        for (const auto &[scene, path] : Scenes) {
            const TestDir dir{"ray-stability"};
            f.Begin(dir);
            const auto extent = scene == FixtureScene::MixedGlass ? mtl::Extent2D{2560, 1600} : Preview;
            Load(f, scene, path, extent);
            f.Do(action::UpdateOf<&ViewportDisplay::RayTracing, &RayTracingSettings::Denoise>(FirstViewport(f.R), false));
            CheckStability(f, path, extent);
            boost::ut::expect(f.P->Close());
        }
    };
}

void WriteRayTracingBenchmarks(const fs::path &dir) {
    boost::ut::test("ray tracing benchmark documents") = [&] {
        Fixture f;
        fs::create_directories(dir);
        for (const auto &[scene, path, name] : BenchmarkScenes) {
            const TestDir project{"ray-benchmark"};
            f.Begin(project);
            Load(f, scene, path, {1280, 800});
            const auto workspace = workspace::Serialize(workspace::Capture(f.R, PanelsState{}));
            boost::ut::expect(f.P->SaveArchive(dir / std::format("{}.actions", name), project::ArchiveForm::Actions, workspace));
            boost::ut::expect(f.P->Close());
        }
    };
}
