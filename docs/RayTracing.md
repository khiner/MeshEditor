# MeshEditor and Blender EEVEE

Measured headlessly at 2560 × 1600 on a 32-core GPU Apple M5 Max running macOS 27.2.1. MeshEditor used its standard optimized editor build (`-O2`) with action validation off. Each timing in the motion table is the **median of 120-frame p95 measurements**, in milliseconds. Each entry uses three trials, except MeshEditor’s traced diffuse row, which includes both three-trial batches. A p95 of 16.67 ms is the 60 fps frame budget. The first two timing columns and both images have overlays off. The editor column includes MeshEditor overlays.

| Scene | MeshEditor (ms) | EEVEE (ms) | MeshEditor + overlays (ms) | Frame 32 · MeshEditor left / EEVEE right |
|---|---:|---:|---:|---|
| Opaque studio · viewport defaults for indirect lighting | 5.05 · Environment | 9.58 · Screen-Trace + Fast GI | 5.67 | [![Opaque studio, MeshEditor left and EEVEE right](ray-tracing/opaque-preview.webp)](ray-tracing/opaque.webp) |
| Studio · Fast GI | 7.18 · Fast GI | 9.61 · Fast GI | 7.80 | [![Studio · Fast GI, MeshEditor left and EEVEE right](ray-tracing/fast-gi-preview.webp)](ray-tracing/fast-gi.webp) |
| Studio · traced diffuse lighting | 12.62 · hardware scene rays | 10.65 · Screen-Trace, Fast GI off | 13.43 | [![Studio · traced diffuse bounce, MeshEditor left and EEVEE right](ray-tracing/diffuse-gi-preview.webp)](ray-tracing/diffuse-gi.webp) |
| Flower vases · blend and transmission | 3.93 | 10.30 | 4.74 | [![Flower vases · blend and transmission, MeshEditor left and EEVEE right](ray-tracing/vase-preview.webp)](ray-tracing/vase.webp) |
| Glass spheres · four shadow rays | 12.48 | 17.36 | 13.07 | [![Glass spheres · four shadow rays, MeshEditor left and EEVEE right](ray-tracing/glass-preview.webp)](ray-tracing/glass.webp) |
| Blended and tinted glass · four shadow rays | 13.75 | 19.16 | 15.20 | [![Blended and tinted glass · four shadow rays, MeshEditor left and EEVEE right](ray-tracing/mixed-preview.webp)](ray-tracing/mixed.webp) |
| Camera inside single-sided glass | 14.23 | 31.00 | 14.96 | [![Camera inside single-sided glass, MeshEditor left and EEVEE right](ray-tracing/inside-preview.webp)](ray-tracing/inside.webp) |
| Moving object behind glass | 5.56 | 8.66 | 6.25 | [![Moving object behind glass, MeshEditor left and EEVEE right](ray-tracing/behind-glass-preview.webp)](ray-tracing/behind-glass.webp) |
| Overlapping textured alpha layers | 1.89 | 5.88 | 2.52 | [![Overlapping textured alpha layers, MeshEditor left and EEVEE right](ray-tracing/alpha-preview.webp)](ray-tracing/alpha.webp) |
| Diffuse transmission · foliage | 2.53 | 6.65 | 3.12 | [![Diffuse transmission · foliage, MeshEditor left and EEVEE right](ray-tracing/foliage-preview.webp)](ray-tracing/foliage.webp) |
| Moving blocker · one area light | 3.53 | 7.76 | 4.19 | [![Moving blocker · one area light, MeshEditor left and EEVEE right](ray-tracing/blocker-preview.webp)](ray-tracing/blocker.webp) |
| Moving emitter · one area light | 3.38 | 8.23 | 3.96 | [![Moving emitter · one area light, MeshEditor left and EEVEE right](ray-tracing/emitter-preview.webp)](ray-tracing/emitter.webp) |
| 64 overlapping area lights | 16.10 | 33.71 | 16.22 | [![64 overlapping area lights, MeshEditor left and EEVEE right](ray-tracing/lights-64-preview.webp)](ray-tracing/lights-64.webp) |
| Flight Helmet · face selection | 3.38 | 7.96 | 3.86 | [![Flight Helmet · face selection, MeshEditor left and EEVEE right](ray-tracing/flight-helmet-preview.webp)](ray-tracing/flight-helmet.webp) |
| Sponza exterior · face selection | 3.36 | 6.91 | 3.71 | [![Sponza exterior · face selection, MeshEditor left and EEVEE right](ray-tracing/sponza-preview.webp)](ray-tracing/sponza.webp) |
| Sponza atrium · camera dolly, Fast GI | 13.82 | 14.15 | 14.42 | [![Sponza atrium · camera dolly, Fast GI, MeshEditor left and EEVEE right](ray-tracing/atrium-preview.webp)](ray-tracing/atrium.webp) |
| 64 animated armatures | 7.80 | 91.16 | 8.73 | [![64 animated armatures, MeshEditor left and EEVEE right](ray-tracing/skins-64-preview.webp)](ray-tracing/skins-64.webp) |
| Same armatures · reflective floor | 12.98 | 93.54 | 13.83 | [![Same armatures · reflective floor, MeshEditor left and EEVEE right](ray-tracing/reflective-skins-64-preview.webp)](ray-tracing/reflective-skins-64.webp) |

## Glass refinement after the camera stops

Both viewports replay the first 32 frames of the blended and tinted glass orbit, then hold the last camera pose. The clock starts when the last moving frame finishes. Each target is the root-mean-square difference from that renderer’s own settled image, measured in 8-bit display RGB over the 427,414 pixels inside the three sphere silhouettes. A crossing counts when the error stays at or below the target for the rest of the run.

| Target error (levels out of 255) | MeshEditor | EEVEE |
|---:|---:|---:|
| 4 | 13 ms · 1 frame | 79 ms · 4 frames |
| 2 | 79 ms · 6 frames | 192 ms · 10 frames |
| 1 | 212 ms · 16 frames | 395 ms · 21 frames |

Times are medians of three GPU-synchronized, unpaced headless trials with overlays off. Separate image runs use the same poses and viewport settings. The entire image settled after 63 stationary frames in MeshEditor (851 ms) and 64 in EEVEE (1,176 ms). Every subsequent capture through frame 72 matched its renderer’s settled RGB values exactly. This measures how quickly each viewport approaches its final appearance. The two renderers differ in their treatment of absorption and refraction. The EEVEE recovery sequence uses the identical camera log from the original comparison.

The top row shows the last moving frame and the bottom row shows the settled result. Click for the full-resolution crops. MeshEditor is on the left and EEVEE on the right.

[![Three glass spheres at camera stop and after refinement in both renderers](ray-tracing/recovery-preview.webp)](ray-tracing/recovery.webp)

Reproduce with `script/BenchmarkRayTracing OUTPUT --recovery --blender PATH_TO_BLENDER`. [Per-frame errors and elapsed times](ray-tracing/recovery.json)

## Reading the comparison

55 of 57 MeshEditor trials with overlays stayed within 16.67 ms at p95. The largest overlay p95 was 19.70 ms on the traced diffuse studio. Its slowest individual frame with overlays was 23.45 ms.

MeshEditor traces scene geometry. Blender 5.4.0 Alpha uses EEVEE Screen-Trace, light probes and shadow maps. Both use denoising and 64 warm-up frames. MeshEditor uses one indirect sample per pixel on its ray grid. The ray grid is half resolution except for the glass interior and armature scenes, which use quarter resolution. The glass sphere scenes use four area-shadow rays per light. The other scenes use one. EEVEE uses its default Fast GI routing above roughness 0.5 except in the traced diffuse studio row. There, disabling Fast GI extends Screen-Trace to all roughnesses, with light probes supplying misses. MeshEditor traces diffuse rays against scene geometry, including offscreen surfaces. The studio Fast GI row and Sponza atrium use a 10-unit Fast GI search radius in both renderers. The 64-light fixture splits a fixed total power among coincident emitters and sets EEVEE’s light cutoff to 100 scene units so every light reaches the receiver.

The alpha layers, vases, blended and tinted spheres, and moving object behind glass were remeasured after sharing the raster transparency pass. Alternating the previous and current executables over the same camera poses gave 8.15 → 1.89 ms for overlapping alpha and 15.43 → 13.75 ms for mixed glass, with overlays off. The table uses these paired trials. Mixed glass was more variable across batches: the first three current-build trials had a median p95 of 24.96 ms.

The traced diffuse studio row uses separate denoising tile lists for reflection and diffuse lighting. Only 0.6% of active 8 × 8 tiles needed both. Across six paired trials, p95 went from 12.99 to 12.62 ms without overlays and from 13.58 to 13.43 ms with overlays. Overlay timings varied substantially in the second batch, reaching 19.70 ms with separate lists and 25.50 ms with shared lists. The frame 32 image matched the original pixel for pixel. The remaining motion rows retain the original timings.

The timed sequences use camera or scene motion. Both engines replay the same camera matrices. MeshEditor times the headless frame through GPU completion. EEVEE times animation and viewport drawing through a synchronized one-pixel readback. Both exclude presentation and image export. The 64 characters share an animation, which lets MeshEditor reuse deformation data.

The tinted sphere shows stronger absorption in MeshEditor. The engines also shade different exit surfaces inside single-sided glass. MeshEditor retains selected-face tint in the two edit-mode images.

Click any pair for the full-resolution lossless images, cropped to the same camera gate.

Build with `PROFILE=1 script/Build` and `cmake --build build --target MeshEditorMslCompileTest`, then run `script/BenchmarkRayTracing OUTPUT --blender PATH_TO_BLENDER` using Blender revision `222853b0d7e4` with the offscreen viewport patch. The individual trials, build hashes and camera hashes are recorded alongside the images. [Timings](ray-tracing/timings.json) · [Benchmark](../script/BenchmarkRayTracing) · [Blender patch](../script/raytracing/blender-offscreen.patch)

The vase, foliage, Flight Helmet, Sponza and BrainStem scenes use Khronos glTF Sample Assets with added cameras, lighting or instances. Foliage materials by Eric Chadwick for Darmstadt Graphics Group, original asset by Rico Cilliers. [Asset credits and licenses](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/9429648735279342b4c32b8745f7904196607379/Models) · [Foliage license](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/9429648735279342b4c32b8745f7904196607379/Models/DiffuseTransmissionPlant/LICENSE.md)
