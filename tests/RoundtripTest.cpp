#include "SceneStores.h"
#include "SceneVulkanResources.h"
#include "gltf/EcsScene.h"
#include "gltf/GltfScene.h"
#include "vulkan/VulkanContext.h"

#include <boost/ut.hpp>
#include <simdjson.h>

#include <entt/entity/registry.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

std::vector<fs::path> CollectGltfSamples(const fs::path &root) {
    std::vector<fs::path> out;
    if (!fs::exists(root)) return out;
    for (const auto &entry : fs::directory_iterator{root}) {
        if (!entry.is_directory()) continue;
        // Khronos samples split variants into subdirectories: `glTF/` for the base form, plus
        // optional feature-specific variants like `glTF-IBL/`, `glTF-Draco/`, `glTF-Quantized/`.
        // Collect from every subdirectory prefixed with "glTF"; fall back to the sample root
        // when none exist (e.g. glTF_Physics samples, which put the .gltf at the top level).
        std::vector<fs::path> search_dirs;
        for (const auto &sub : fs::directory_iterator{entry.path()}) {
            if (sub.is_directory() && sub.path().filename().string().starts_with("glTF")) {
                search_dirs.emplace_back(sub.path());
            }
        }
        if (search_dirs.empty()) search_dirs.emplace_back(entry.path());
        for (const auto &search : search_dirs) {
            for (const auto &child : fs::directory_iterator{search}) {
                if (child.is_regular_file() && child.path().extension() == ".gltf") {
                    out.emplace_back(child.path());
                }
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

fs::path MakeRoundtripDir() {
    auto dir = fs::temp_directory_path() / "MeshEditor-roundtrip";
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

constexpr std::string_view SampleRoots[]{
    "../glTF-Sample-Assets/Models",
    "../glTF_Physics/samples",
};

// --- Expected-divergence list ---
// Each entry documents a JSON path where re-export is known to diverge from source,
// along with the specific detail of the divergence. Array indices in the pattern are written
// as `[*]` to match any position.
//
// Entries are grouped by topic so a single rationale applies to the whole block; each entry's
// `Why` only notes the specific default/field being filtered.
//
// Matching modes:
//   - Exact{}   matches the path literally (no descendants filtered).
//   - Subtree{} matches that path and any descendant path (e.g. `buffers` matches
//               `buffers[0].byteLength` but not `buffersInternal`).
// Prefer exact matches when filtering a specific JSON leaf; reserve Subtree for cases where
// the whole sub-tree is legitimately encoding-dependent so we don't mask unexpected semantic
// regressions deeper inside.
struct Exception {
    std::string_view Pattern;
    std::string_view Why;
};

constexpr Exception SubtreeExceptions[]{
    // --- Buffer / accessor layout: re-packed on export; semantic content preserved via accessor refs. ---
    {"buffers", "re-packed into a single buffer on export"},
    {"bufferViews", "bufferView layout and order depend on our emission order"},
    {"accessors", "accessor order depends on our emission choices; position-indexed JSON compare can't tell semantically equivalent accessors apart (semantic shape is verified indirectly via mesh/primitive/animation references)"},

    // --- extensions we don't import or re-emit yet. ---
    {"extensionsUsed", "extensionsUsed is computed from what we re-emit; membership/order diverges when extension data isn't fully re-emitted"},
    {"extensions.KHR_xmp_json_ld", "not imported"},
    {"extensions.KHR_xmp", "not imported"},
    {"materials[*].extensions.KHR_materials_volume_scatter", "not imported"},

    // --- Multi-scene files and unsupported animation channels. ---
    {"scenes", "multi-scene files are collapsed to the default scene on import"},
    {"animations[*].channels", "channels using KHR_animation_pointer are dropped on import"},
    {"animations[*].samplers", "samplers for dropped channels (e.g. KHR_animation_pointer) are also dropped"},

};

// --- Default-value omission ---
// fastgltf's writer omits spec-default field values. Several source writers emit defaults
// explicitly (notably COLLADA2GLTF, glTF Tools for Unity, hand-written samples with no generator,
// and the Blender v5.0.21 glTF I/O for physics samples). Each entry below filters one such path.
constexpr Exception DefaultOmissionExactExceptions[]{
    {"scene", "default scene index 0 omitted"},
    {"animations", "empty animations array omitted"},
    {"samplers[*].wrapS", "default 10497 (REPEAT) omitted"},
    {"samplers[*].wrapT", "default 10497 (REPEAT) omitted"},
    {"animations[*].samplers[*].interpolation", "default 'LINEAR' omitted"},
    {"materials[*].emissiveFactor", "default [0,0,0] omitted"},
    {"materials[*].alphaMode", "default 'OPAQUE' omitted"},
    {"materials[*].alphaCutoff", "default 0.5 omitted (emitted by source when alphaMode is MASK)"},
    {"materials[*].doubleSided", "default false omitted"},
    {"materials[*].pbrMetallicRoughness.baseColorFactor", "default [1,1,1,1] omitted"},
    {"materials[*].pbrMetallicRoughness.metallicFactor", "default 1.0 omitted"},
    {"materials[*].pbrMetallicRoughness.roughnessFactor", "default 1.0 omitted"},
    {"materials[*].pbrMetallicRoughness.baseColorTexture.texCoord", "default 0 omitted"},
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.texCoord", "default 0 omitted"},
    {"materials[*].normalTexture.texCoord", "default 0 omitted"},
    {"materials[*].occlusionTexture.texCoord", "default 0 omitted"},
    {"materials[*].emissiveTexture.texCoord", "default 0 omitted"},
    {"materials[*].extensions.KHR_materials_volume.thicknessTexture.texCoord", "default 0 omitted"},
    {"materials[*].extensions.KHR_materials_clearcoat.clearcoatTexture.texCoord", "default 0 omitted"},
    {"materials[*].extensions.KHR_materials_iridescence.iridescenceThicknessMaximum", "default 400 omitted"},
    {"materials[*].pbrMetallicRoughness.baseColorTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    // fastgltf's writer omits default KHR_texture_transform fields on the 5 base material textures
    // whenever the extension block is emitted (see TextureTransformMeta.SourceHadExtension).
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.scale", "default [1,1] scale omitted"},
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].normalTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].normalTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.scale", "default [1,1] scale omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].extensions.KHR_materials_diffuse_transmission.diffuseTransmissionColorFactor", "default [1,1,1] omitted"},
    {"materials[*].extensions.KHR_materials_anisotropy.anisotropyRotation", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_anisotropy.anisotropyStrength", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_specular.specularFactor", "default 1.0 omitted"},
    {"materials[*].extensions.KHR_materials_specular.specularColorFactor", "default [1,1,1] omitted"},
    {"materials[*].extensions.KHR_materials_sheen.sheenColorFactor", "default [0,0,0] omitted"},
    {"materials[*].extensions.KHR_materials_sheen.sheenRoughnessFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_clearcoat.clearcoatFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_clearcoat.clearcoatRoughnessFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_transmission.transmissionFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_iridescence.iridescenceFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_iridescence.iridescenceIor", "default 1.3 omitted"},
    {"materials[*].extensions.KHR_materials_volume.thicknessFactor", "default 0.0 omitted"},
    {"materials[*].extensions.KHR_materials_volume.attenuationColor", "default [1,1,1] omitted"},
    {"materials[*].extensions.KHR_materials_volume.attenuationDistance", "default infinity omitted"},
    {"extensions.KHR_lights_punctual.lights[*].color", "default [1,1,1] omitted"},
    {"extensions.KHR_lights_punctual.lights[*].intensity", "default 1.0 omitted"},
    {"extensions.EXT_lights_image_based.lights[*].intensity", "default 1.0 omitted"},
    {"nodes[*].extensions.KHR_physics_rigid_bodies.collider.geometry.convexHull", "default false omitted"},
    {"nodes[*].extensions.KHR_physics_rigid_bodies.motion.centerOfMass", "default [0,0,0] omitted"},
    {"meshes[*].primitives[*].mode", "default 4 (triangles) omitted"},
    // fastgltf's writer omits TRS fields equal to defaults on TRS-form nodes.
    // Matrix-form nodes now re-emit as matrix (see SourceMatrixTransform handling).
    {"nodes[*].translation", "default [0,0,0] omitted"},
    {"nodes[*].rotation", "default [0,0,0,1] omitted"},
    {"nodes[*].scale", "default [1,1,1] omitted"},
    // Empty-string / empty-object emissions the source carried explicitly:
    {"asset.copyright", "empty source copyright string omitted by fastgltf's writer"},
    {"extensions", "empty root-level extensions block omitted by fastgltf's writer (source emits `\"extensions\":{}`)"},
    {"materials[*].extensions", "empty material-level extensions block omitted"},
    {"nodes[*].extensions", "empty per-node extensions block omitted"},
};

// --- Always-emitted (we don't omit defaults) ---
// Symmetric to the above: fastgltf's writer always emits these fields (no "only if non-default"
// gate), so source files that legitimately omit them see a "present in roundtripped only" diff.
constexpr Exception AlwaysEmittedExactExceptions[]{
    {"materials[*].normalTexture.scale", "fastgltf always emits scale on NormalTextureInfo"},
    {"materials[*].occlusionTexture.strength", "fastgltf always emits strength on OcclusionTextureInfo"},
    {"materials[*].extensions.KHR_materials_clearcoat.clearcoatNormalTexture.scale", "fastgltf always emits scale on NormalTextureInfo"},
    {"extensions.KHR_physics_rigid_bodies.physicsMaterials[*].frictionCombine", "always-emitted even when equal to default 'average'"},
    {"extensions.KHR_physics_rigid_bodies.physicsMaterials[*].restitutionCombine", "always-emitted even when equal to default 'average'"},
    {"extensions.KHR_physics_rigid_bodies.physicsJoints[*].limits[*].damping", "always-emitted even when 0.0"},
    {"extensions.KHR_physics_rigid_bodies.physicsJoints[*].drives[*].maxForce", "always-emitted even when float::max ('no limit')"},
    {"materials[*].pbrMetallicRoughness", "emitted even for extension-only / specular-glossiness materials"},
};

// --- BufferView indexing ---
// Accessor references (meshes[*].primitives[*].attributes, indices, targets, skin IBMs,
// animation sampler inputs/outputs) compare semantically via the referenced accessor's
// (type, count, normalized) shape - see IsAccessorRefPath/CompareAccessorShape. That handles
// the emission-order divergence without masking shape regressions. BufferView refs outside
// the accessor path (just images[*].bufferView today) stay as exact exceptions.
constexpr Exception BufferViewIndexExactExceptions[]{
    // bufferView index depends on our emission order; only fires for images embedded as bufferView
    // (source form preserved: URI-form images emit URIs, no bufferView ref).
    {"images[*].bufferView", "bufferView index depends on emission order"},
    // Indexed vs non-indexed round-trips cleanly for Triangles (see HasSourceIndices on
    // MeshPrimitives). Non-triangle modes (TriangleStrip/Fan, LineStrip/Loop, Points) are
    // unfolded to TriangleList / LineList / Points at save and the indices accessor count
    // (or presence) diverges from source accordingly.
    {"meshes[*].primitives[*].indices", "strip/fan/loop primitive modes are unfolded to list mode on save; source Points with indices lose them"},
};

constexpr Exception OtherExactExceptions[]{
    {"meshes[*].primitives[*].material", "line/point primitives lose per-primitive material on import (merged across primitives without retaining material refs)"},
    {"meshes[*].extensions", "mesh-level extensions (e.g. KHR_materials_variants mappings) not re-emitted"},
    {"scenes[*].extras", "not tracked on Scene"},
    {"scenes[*].extensions", "scene-level extensions not re-emitted"},
};

std::string NormalizePath(std::string_view path) {
    std::string out;
    out.reserve(path.size());
    for (size_t i = 0; i < path.size();) {
        if (path[i] == '[') {
            const auto end = path.find(']', i);
            if (end == std::string_view::npos) {
                out.append(path.substr(i));
                break;
            }
            out.append("[*]");
            i = end + 1;
        } else {
            out.push_back(path[i++]);
        }
    }
    return out;
}

bool SubtreeMatches(std::string_view normalized_path, std::string_view pattern) {
    if (!normalized_path.starts_with(pattern)) return false;
    if (normalized_path.size() == pattern.size()) return true;
    const char next = normalized_path[pattern.size()];
    return next == '.' || next == '[';
}

template<std::size_t N>
bool MatchesExact(std::string_view norm, const Exception (&list)[N]) {
    for (const auto &ex : list) {
        if (norm == ex.Pattern) return true;
    }
    return false;
}

bool IsExpectedDivergence(std::string_view path) {
    const auto norm = NormalizePath(path);
    if (MatchesExact(norm, DefaultOmissionExactExceptions)) return true;
    if (MatchesExact(norm, AlwaysEmittedExactExceptions)) return true;
    if (MatchesExact(norm, BufferViewIndexExactExceptions)) return true;
    if (MatchesExact(norm, OtherExactExceptions)) return true;
    for (const auto &ex : SubtreeExceptions) {
        if (SubtreeMatches(norm, ex.Pattern)) return true;
    }
    return false;
}

// --- Generic JSON comparator ---

struct Diff {
    std::string Path;
    std::string Message;
};

// Float-ish tolerance: glTF numbers round-trip through JSON text, so small loss is expected.
// Some source files truncate to 3-4 significant digits (e.g. a quaternion component "0.688"),
// so relative tolerance has to be loose enough to absorb that.
constexpr double AbsEps = 1e-6;
constexpr double RelEps = 1e-3;

bool NumberEq(double a, double b) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) == std::isnan(b);
    const double diff = std::abs(a - b);
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return diff <= AbsEps || diff <= RelEps * scale;
}

using ElType = simdjson::dom::element_type;

bool IsNumber(ElType t) {
    return t == ElType::INT64 || t == ElType::UINT64 || t == ElType::DOUBLE;
}

double AsNumber(simdjson::dom::element e) {
    switch (e.type()) {
        case ElType::INT64: return double(int64_t(e));
        case ElType::UINT64: return double(uint64_t(e));
        case ElType::DOUBLE: return double(e);
        default: return std::nan("");
    }
}

std::string_view TypeName(ElType t) {
    switch (t) {
        case ElType::OBJECT: return "object";
        case ElType::ARRAY: return "array";
        case ElType::STRING: return "string";
        case ElType::INT64: return "int64";
        case ElType::UINT64: return "uint64";
        case ElType::DOUBLE: return "double";
        case ElType::BOOL: return "bool";
        case ElType::NULL_VALUE: return "null";
    }
    return "?";
}

std::string JoinKey(std::string_view path, std::string_view key) {
    return path.empty() ? std::string(key) : std::format("{}.{}", path, key);
}
std::string JoinIndex(std::string_view path, size_t i) {
    return std::format("{}[{}]", path, i);
}

// --- Semantic accessor-reference resolution ---
// Several JSON paths hold a numeric accessor index whose index value is emission-order-dependent
// (we pack accessors differently from any given source writer). Positional JSON compare would see
// `5 vs 12` and diff, even when both sides reference accessors with identical semantic shape.
// Comparing the RESOLVED accessor's (type, count, normalized) lets us diff genuine shape regressions
// (wrong vertex count, swapped attribute role, etc.) without false positives from emission order.
bool IsAccessorRefPath(std::string_view norm) {
    if (norm == "meshes[*].primitives[*].indices") return true;
    if (norm == "skins[*].inverseBindMatrices") return true;
    if (norm == "animations[*].samplers[*].input") return true;
    if (norm == "animations[*].samplers[*].output") return true;
    static constexpr std::string_view AnyChildPrefixes[]{
        "meshes[*].primitives[*].attributes.",
        "meshes[*].primitives[*].targets[*].",
        "nodes[*].extensions.EXT_mesh_gpu_instancing.attributes.",
    };
    for (const auto p : AnyChildPrefixes) {
        if (norm.starts_with(p)) {
            const auto rest = norm.substr(p.size());
            if (!rest.empty() && rest.find('.') == std::string_view::npos && rest.find('[') == std::string_view::npos) return true;
        }
    }
    return false;
}

std::optional<simdjson::dom::element> ResolveAccessor(simdjson::dom::element root, size_t index) {
    simdjson::dom::array accessors;
    if (root["accessors"].get_array().get(accessors) != simdjson::SUCCESS) return std::nullopt;
    if (index >= accessors.size()) return std::nullopt;
    return accessors.at(index);
}

// Compare only the semantic fields of an accessor: type and count. Encoding-level fields
// (componentType, normalized, bufferView, byteOffset, sparse, name) are intentionally skipped -
// our importer normalizes componentType (indices widen to uint32, joints narrow to uint16,
// quantized attributes decode to FLOAT), and `normalized` is a semantic pair with componentType
// that changes with it. bufferView layout is an emission choice; min/max are omitted by some
// source writers selectively.
void CompareAccessorShape(simdjson::dom::element src, simdjson::dom::element out, std::string_view ref_path, std::vector<Diff> &diffs) {
    std::string_view src_type, out_type;
    const bool src_has_type = src["type"].get_string().get(src_type) == simdjson::SUCCESS;
    const bool out_has_type = out["type"].get_string().get(out_type) == simdjson::SUCCESS;
    if (src_has_type && out_has_type && src_type != out_type) {
        diffs.emplace_back(std::string(ref_path), std::format("accessor type \"{}\" vs \"{}\"", src_type, out_type));
    }

    uint64_t src_cnt = 0, out_cnt = 0;
    const bool src_has_cnt = src["count"].get_uint64().get(src_cnt) == simdjson::SUCCESS;
    const bool out_has_cnt = out["count"].get_uint64().get(out_cnt) == simdjson::SUCCESS;
    if (src_has_cnt && out_has_cnt && src_cnt != out_cnt) {
        diffs.emplace_back(std::string(ref_path), std::format("accessor count {} vs {}", src_cnt, out_cnt));
    }
}

void CompareJson(simdjson::dom::element a, simdjson::dom::element b, std::string_view path, std::vector<Diff> &out, simdjson::dom::element root_a, simdjson::dom::element root_b) {
    const auto ta = a.type(), tb = b.type();
    // Numbers compare with epsilon regardless of int/uint/double subtype.
    if (IsNumber(ta) && IsNumber(tb)) {
        if (IsAccessorRefPath(NormalizePath(path))) {
            const auto src_acc = ResolveAccessor(root_a, size_t(AsNumber(a)));
            const auto out_acc = ResolveAccessor(root_b, size_t(AsNumber(b)));
            if (!src_acc || !out_acc) {
                out.emplace_back(std::string(path), "accessor reference out of range");
            } else {
                CompareAccessorShape(*src_acc, *out_acc, path, out);
            }
            return;
        }
        if (!NumberEq(AsNumber(a), AsNumber(b))) {
            out.emplace_back(std::string(path), std::format("number {} vs {}", AsNumber(a), AsNumber(b)));
        }
        return;
    }
    if (ta != tb) {
        out.emplace_back(std::string(path), std::format("type {} vs {}", TypeName(ta), TypeName(tb)));
        return;
    }
    switch (ta) {
        case ElType::OBJECT: {
            std::vector<std::pair<std::string_view, simdjson::dom::element>> va, vb;
            for (auto kv : simdjson::dom::object(a)) va.emplace_back(kv.key, kv.value);
            for (auto kv : simdjson::dom::object(b)) vb.emplace_back(kv.key, kv.value);
            const auto cmp = [](const auto &x, const auto &y) { return x.first < y.first; };
            std::sort(va.begin(), va.end(), cmp);
            std::sort(vb.begin(), vb.end(), cmp);
            size_t i = 0, j = 0;
            while (i < va.size() || j < vb.size()) {
                if (j >= vb.size() || (i < va.size() && va[i].first < vb[j].first)) {
                    out.emplace_back(JoinKey(path, va[i].first), "present in source only");
                    ++i;
                } else if (i >= va.size() || vb[j].first < va[i].first) {
                    out.emplace_back(JoinKey(path, vb[j].first), "present in roundtripped only");
                    ++j;
                } else {
                    CompareJson(va[i].second, vb[j].second, JoinKey(path, va[i].first), out, root_a, root_b);
                    ++i;
                    ++j;
                }
            }
            break;
        }
        case ElType::ARRAY: {
            simdjson::dom::array aa = a, bb = b;
            const auto asz = aa.size(), bsz = bb.size();
            if (asz != bsz) {
                out.emplace_back(std::string(path), std::format("array size {} vs {}", asz, bsz));
            }
            size_t i = 0;
            auto ia = aa.begin(), ib = bb.begin();
            while (ia != aa.end() && ib != bb.end()) {
                CompareJson(*ia, *ib, JoinIndex(path, i), out, root_a, root_b);
                ++ia;
                ++ib;
                ++i;
            }
            break;
        }
        case ElType::STRING: {
            const std::string_view sa = a, sb = b;
            if (sa != sb) out.emplace_back(std::string(path), std::format("string \"{}\" vs \"{}\"", sa, sb));
            break;
        }
        case ElType::BOOL: {
            const bool va = a, vb = b;
            if (va != vb) out.emplace_back(std::string(path), std::format("bool {} vs {}", va, vb));
            break;
        }
        case ElType::NULL_VALUE:
            break;
        default:
            break; // numbers handled above
    }
}
} // namespace

namespace {
// Compares two parsed gltf JSON files A and B with the existing exception list. Reports up to
// 20 unexpected diffs to stderr. Returns the count of unexpected diffs.
size_t CompareGltfJson(const fs::path &a_path, const fs::path &b_path, std::string_view sample_name) {
    using namespace boost::ut;
    simdjson::dom::parser pa, pb;
    simdjson::dom::element ea, eb;
    const auto err_a = pa.load(a_path.string()).get(ea);
    const auto err_b = pb.load(b_path.string()).get(eb);
    expect(err_a == simdjson::SUCCESS) << "Parse A: " << simdjson::error_message(err_a);
    expect(err_b == simdjson::SUCCESS) << "Parse B: " << simdjson::error_message(err_b);
    if (err_a != simdjson::SUCCESS || err_b != simdjson::SUCCESS) return 0;

    std::vector<Diff> all_diffs;
    CompareJson(ea, eb, "", all_diffs, ea, eb);

    std::vector<Diff> unexpected;
    size_t expected = 0;
    for (auto &d : all_diffs) {
        if (IsExpectedDivergence(d.Path)) ++expected;
        else unexpected.emplace_back(std::move(d));
    }
    if (!unexpected.empty()) {
        constexpr size_t MaxReport = 20;
        std::cerr << "  " << unexpected.size() << " unexpected JSON diff(s) in " << sample_name
                  << " (" << expected << " expected-divergence diff(s) filtered):\n";
        for (size_t i = 0; i < unexpected.size() && i < MaxReport; ++i) {
            std::cerr << "    " << unexpected[i].Path << ": " << unexpected[i].Message << "\n";
        }
        if (unexpected.size() > MaxReport) {
            std::cerr << "    ... and " << (unexpected.size() - MaxReport) << " more\n";
        }
    }
    return unexpected.size();
}
} // namespace

int main() {
    using namespace boost::ut;

    const auto tmp_root = MakeRoundtripDir();

    std::vector<fs::path> samples;
    for (const auto root : SampleRoots) {
        auto found = CollectGltfSamples(root);
        samples.insert(samples.end(), found.begin(), found.end());
    }

    // Headless Vulkan fixture shared across all ECS-roundtrip samples (device init is expensive).
    // Each sample test below builds its own SceneStores + registry on top of these shared handles.
    VulkanContext vk_ctx{{}, /*with_swapchain=*/false};
    const SceneVulkanResources vk_resources{*vk_ctx.Instance, vk_ctx.PhysicalDevice, *vk_ctx.Device, vk_ctx.QueueFamily, vk_ctx.Queue};
    const auto cmd_pool = vk_ctx.Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, vk_ctx.QueueFamily});
    const auto fence = vk_ctx.Device->createFenceUnique({});

    for (const auto &src : samples) {
        const auto sample_name = src.stem().string();

        test(sample_name) = [&] {
            auto loaded = gltf::LoadScene(src);
            if (!loaded) return; // Loader limitation on source (e.g., unsupported extension); not a roundtrip concern.

            const auto out_path = tmp_root / (sample_name + ".gltf");
            auto save_result = gltf::SaveScene(*loaded, out_path);
            expect(save_result.has_value()) << "SaveScene failed: " << (save_result ? "" : save_result.error());
            if (!save_result) return;

            const auto unexpected = CompareGltfJson(src, out_path, sample_name);
            expect(unexpected == 0) << unexpected << " unexpected JSON diff(s)";
        };

        test(sample_name + " (ecs)") = [&] {
            auto loaded = gltf::LoadScene(src);
            if (!loaded) return;

            entt::registry registry;
            SceneStores stores{vk_resources, *cmd_pool, *fence};
            const auto scene_entity = WireSceneRegistry(registry, stores);

            gltf::PopulateContext ctx{
                .R = registry,
                .SceneEntity = scene_entity,
                .Vk = vk_resources,
                .CommandPool = *cmd_pool,
                .OneShotFence = *fence,
                .Slots = *stores.Slots,
                .Buffers = *stores.Buffers,
                .Meshes = *stores.Meshes,
                .Textures = *stores.Textures,
                .Environments = *stores.Environments,
            };
            auto populate = gltf::PopulateGltfScene(*loaded, src, ctx);
            expect(populate.has_value()) << "PopulateGltfScene failed: " << (populate ? "" : populate.error());
            if (!populate) return;

            auto rebuilt = gltf::BuildGltfScene(registry, scene_entity);

            // Save both `loaded` and `rebuilt` through SaveScene; identical SaveScene paths cancel
            // out fastgltf-side asymmetries, so any diff is genuine ECS-roundtrip lossiness.
            const auto a_path = tmp_root / (sample_name + ".A.gltf");
            const auto b_path = tmp_root / (sample_name + ".B.gltf");
            auto save_a = gltf::SaveScene(*loaded, a_path);
            auto save_b = gltf::SaveScene(rebuilt, b_path);
            expect(save_a.has_value()) << "SaveScene(loaded) failed: " << (save_a ? "" : save_a.error());
            expect(save_b.has_value()) << "SaveScene(rebuilt) failed: " << (save_b ? "" : save_b.error());
            if (!save_a || !save_b) return;

            const auto unexpected = CompareGltfJson(a_path, b_path, sample_name + " (ecs)");
            expect(unexpected == 0) << unexpected << " unexpected JSON diff(s)";
        };
    }
}
