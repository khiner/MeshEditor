#include "Path.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "gltf/GltfScene.h"
#include "gpu/PunctualLight.h"
#include "image/ImageDecode.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/PrimitiveType.h"
#include "mesh/Primitives.h"
#include "render/GpuBuffers.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "snapshot/SaveState.h"
#include "snapshot/SceneSnapshot.h"
#include "snapshot/SnapshotRoles.h"
#include "viewport/Viewport.h"
#include "vulkan/VulkanContext.h"

#include <boost/ut.hpp>
#include <entt/entity/registry.hpp>
#include <simdjson.h>

#include <cstring>
#include <map>
#include <set>

namespace {
namespace fs = std::filesystem;
using std::ranges::to, std::views::join, std::views::transform;

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

// Sample assets are checked out as siblings of the repo root. Anchor every sample path at
// MESHEDITOR_SOURCE_DIR (the repo root, baked in by CMake) so the test finds them regardless of the
// working directory it's launched from.
constexpr std::string_view SampleRoots[]{
    "../glTF-Sample-Assets/Models",
    "../glTF_Physics/samples",
};

fs::path SamplePath(std::string_view relative_to_repo_root) { return fs::path{MESHEDITOR_SOURCE_DIR} / relative_to_repo_root; }

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

    // --- Unsupported animation channels. ---
    {"animations[*].channels", "channels using KHR_animation_pointer are dropped on import"},
    {"animations[*].samplers", "samplers for dropped channels (e.g. KHR_animation_pointer) are also dropped"},

    // Lights aren't shared across nodes on import, so the lights table and per-node `light`
    // indices diverge when source had multiple nodes pointing at the same `lights[i]`.
    {"extensions.KHR_lights_punctual.lights", "per-node PunctualLight components aren't deduped on save"},
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
    {"nodes[*].extensions.KHR_node_visibility", "extension block only emitted for visible:false (default true is omitted)"},
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
    {"meshes[*].extensions", "mesh-level extensions (e.g. KHR_xmp_json_ld) not re-emitted"},
    {"scenes[*].extras", "not tracked on Scene"},
    {"scenes[*].extensions", "scene-level extensions not re-emitted"},
    {"nodes[*].extensions.KHR_lights_punctual.light", "renumbered to match the un-deduped lights table"},
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

template<size_t N>
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
    std::string Path, Message;
};

// Float-ish tolerance: glTF numbers round-trip through JSON text, so small loss is expected.
// Some source files truncate to 3-4 significant digits (e.g. a quaternion component "0.688"),
// so relative tolerance has to be loose enough to absorb that.
constexpr double AbsEps = 1e-6;
constexpr double RelEps = 1e-3;

bool NumberEq(double a, double b) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) == std::isnan(b);
    const double diff = std::abs(a - b), scale = std::max({1.0, std::abs(a), std::abs(b)});
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

std::string JoinKey(std::string_view path, std::string_view key) { return path.empty() ? std::string{key} : std::format("{}.{}", path, key); }
std::string JoinIndex(std::string_view path, size_t i) { return std::format("{}[{}]", path, i); }

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

// q and -q describe the same rotation; `glm::decompose` doesn't preserve source sign so this
// surfaces as a per-component diff on `nodes[*].rotation`. Accept either sign within tolerance.
bool QuaternionsEqual(simdjson::dom::array a, simdjson::dom::array b) {
    if (a.size() != 4 || b.size() != 4) return false;
    const auto to_quat = [](simdjson::dom::array arr, std::array<double, 4> &out) {
        size_t i = 0;
        for (auto e : arr) {
            if (!IsNumber(e.type())) return false;
            out[i++] = AsNumber(e);
        }
        return true;
    };
    std::array<double, 4> va{}, vb{};
    if (!to_quat(a, va) || !to_quat(b, vb)) return false;
    const auto eq_signed = [&](double sign) {
        for (size_t i = 0; i < 4; ++i) {
            if (!NumberEq(va[i], sign * vb[i])) return false;
        }
        return true;
    };
    return eq_signed(1.0) || eq_signed(-1.0);
}

// glTF `scene.nodes` is an unordered set (schema: uniqueItems), so compare the root indices as a
// multiset — we emit them ascending, source order is arbitrary.
bool SameNumberMultiset(simdjson::dom::array a, simdjson::dom::array b) {
    const auto collect = [](simdjson::dom::array arr, std::vector<double> &out) {
        for (auto e : arr) {
            if (!IsNumber(e.type())) return false;
            out.emplace_back(AsNumber(e));
        }
        return true;
    };
    std::vector<double> va, vb;
    if (!collect(a, va) || !collect(b, vb) || va.size() != vb.size()) return false;
    std::ranges::sort(va);
    std::ranges::sort(vb);
    return va == vb;
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
    if (ta == ElType::ARRAY && NormalizePath(path) == "nodes[*].rotation") {
        if (!QuaternionsEqual(a, b)) {
            out.emplace_back(std::string(path), "quaternion mismatch");
        }
        return;
    }
    if (ta == ElType::ARRAY && NormalizePath(path) == "scenes[*].nodes") {
        if (!SameNumberMultiset(a, b)) out.emplace_back(std::string(path), "scene root set mismatch");
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

// Assert two registries match: same components per entity, and equal values for every comparable component.
// Even allocation handles (GPU slots, Jolt body ids) reconstruct identically from a clean per-scene baseline — nothing is exempt.
void CompareRegistries(std::string_view name, entt::registry &a, entt::registry &b) {
    using namespace boost::ut;
    const auto components_by_entity = [](entt::registry &r) {
        std::map<entt::entity, std::set<std::string>> m;
        for (auto [id, set] : r.storage()) {
            const std::string_view tn{set.info().name()};
            if (tn.starts_with("entt::")) continue; // entity / reactive storages, not components
            for (const auto e : set) {
                if (e != entt::tombstone) m[e].insert(std::string{tn});
            }
        }
        return m;
    };
    const auto ca = components_by_entity(a), cb = components_by_entity(b);

    std::map<std::string, int> only_a_comps, only_b_comps, diff_comps;
    int only_a = 0, only_b = 0, diffs = 0;
    for (const auto &[e, comps] : ca) {
        const auto it = cb.find(e);
        if (it == cb.end()) {
            ++only_a;
            for (const auto &c : comps) ++only_a_comps[c];
        } else if (comps != it->second) {
            ++diffs;
            for (const auto &c : comps)
                if (!it->second.contains(c)) ++diff_comps["-" + c];
            for (const auto &c : it->second)
                if (!comps.contains(c)) ++diff_comps["+" + c];
        }
    }
    for (const auto &[e, comps] : cb) {
        if (!ca.contains(e)) {
            ++only_b;
            for (const auto &c : comps) ++only_b_comps[c];
        }
    }

    // ComponentValuesEqual handles the per-type compare (field-wise where memcmp is unsafe). nullopt means the type
    // can't be compared (a derived component with no serializer, e.g. MeshBuffers), so it's skipped.
    std::map<entt::id_type, entt::sparse_set *> b_set;
    for (auto [id, set] : b.storage()) b_set[set.info().hash()] = &set;
    std::map<std::string, int> value_diffs;
    for (auto [id, a_set] : a.storage()) {
        const std::string_view tn{a_set.info().name()};
        if (tn.starts_with("entt::")) continue;
        const auto hash = a_set.info().hash();
        const auto bit = b_set.find(hash);
        if (bit == b_set.end()) continue;
        auto *b_set_p = bit->second;
        for (const auto e : a_set) {
            if (e == entt::tombstone || !b_set_p->contains(e)) continue;
            const auto eq = snapshot::ComponentValuesEqual(hash, a_set.value(e), b_set_p->value(e));
            if (eq && !*eq) ++value_diffs[std::string{tn}];
        }
    }

    // Presence must match exactly - no exclusions, unlike the value check below.
    const auto present_detail = [&] {
        std::string s;
        const auto dump = [&](const char *label, const std::map<std::string, int> &h) {
            if (h.empty()) return;
            s += std::format(" {}:", label);
            for (const auto &[c, n] : h) s += std::format(" {}({})", c, n);
        };
        dump("only-in-fx", only_a_comps);
        dump("only-in-restore", only_b_comps);
        dump("comp-set-diff(-fx/+restore)", diff_comps);
        return s;
    };
    expect(only_a == 0 && only_b == 0 && diffs == 0)
        << name << "presence diverged - only-in-fx=" << only_a << " only-in-restore=" << only_b
        << " comp-set-diffs=" << diffs << present_detail();

    // Every comparable component must hold an identical value in both registries.
    std::string value_detail;
    for (const auto &[type, n] : value_diffs) value_detail += std::format(" {}({})", type, n);
    expect(value_diffs.empty()) << name << "value diverged for component(s):" << value_detail;
}
} // namespace

int main() {
    using namespace boost::ut;

    // res/ and shaders/ are symlinked into the CMake build dir. InitEngine compiles shaders and loads LUTs from there.
    Paths::Init(MESHEDITOR_BUILD_DIR);

    const auto tmp_root = MakeRoundtripDir();
    const auto samples = SampleRoots | transform([](auto root) { return CollectGltfSamples(SamplePath(root)); }) | join | to<std::vector>();

    // Headless Vulkan fixture shared across all ECS-roundtrip samples (device init is expensive).
    const VulkanContext vk_ctx{{}, /*with_swapchain=*/false};
    const VulkanResources vk_resources = vk_ctx.Resources();

    struct SceneFixture {
        entt::registry R;
        entt::entity Viewport;

        explicit SceneFixture(VulkanResources vk) : Viewport(InitEngine(R, vk)) { SetupScene(R, Viewport); }
        ~SceneFixture() { DeinitViewport(R, Viewport); }
    };

    // SaveState → restore into a fresh registry->SaveState again must match byte-for-byte, exercising every
    // encoding (Tag/Bytes/Serialized), the mesh arena blob, and exact entity-handle recreation.
    "snapshot save/restore round trip"_test = [&] {
        std::vector<std::byte> before;
        {
            SceneFixture f{vk_resources};
            auto &meshes = f.R.ctx().get<MeshStore>();
            auto created = meshes.CreateMesh(primitive::CreateMesh(primitive::Cuboid{}), {}, {});
            const auto e = f.R.create();
            f.R.emplace<MeshConnectivity>(e, std::move(created.Connectivity)); // Serialized (heap)
            f.R.emplace<MeshHandle>(e, MeshHandle{created.StoreId}); // Bytes
            f.R.emplace<Name>(e, "Cube"); // Serialized (string)
            f.R.emplace<ObjectKind>(e, ObjectType::Mesh); // Bytes
            f.R.emplace<MeshActiveElement>(e, 7u); // Bytes
            f.R.emplace<Selected>(e); // Tag
            f.R.emplace<Path>(e, "/tmp/scene.gltf"); // Serialized (std::filesystem::path)

            const auto light = f.R.create();
            f.R.emplace<PunctualLight>(light, PunctualLight{.Range = 12.f, .Color = {0.2f, 0.4f, 0.6f}, .Intensity = 3.f}); // Bytes (generated POD)
            f.R.emplace<Name>(light, "Lamp");

            ProcessComponentEvents(f.R, f.Viewport);
            before = snapshot::SaveState(f.R);
            expect(before.size() > sizeof(uint64_t));
        }

        // Restore into a fresh registry - the re-saved image must match byte-for-byte.
        SceneFixture f{vk_resources};
        snapshot::LoadState(f.R, before);
        ProcessComponentEvents(f.R, f.Viewport);
        const auto after = snapshot::SaveState(f.R);
        const auto diff = snapshot::Compare(before, after);
        expect(diff.Equal) << "round-trip diverged at byte" << diff.FirstDifferingByte << "of" << before.size() << "/" << after.size();
    };

    // MeshConnectivity uses in_place_delete, so erasing a mesh leaves a tombstone slot in its pool.
    // SaveState must skip tombstones (a sparse_set yields them but value() asserts on them) — regression for the crash
    // when saving after a New->Empty clear removed the default cube.
    "snapshot save skips in_place_delete tombstones"_test = [&] {
        SceneFixture f{vk_resources};
        auto &meshes = f.R.ctx().get<MeshStore>();
        const auto keep = f.R.create();
        auto kept = meshes.CreateMesh(primitive::CreateMesh(primitive::Cuboid{}), {}, {});
        f.R.emplace<MeshConnectivity>(keep, std::move(kept.Connectivity));
        f.R.emplace<MeshHandle>(keep, MeshHandle{kept.StoreId});

        const auto gone = f.R.create();
        auto removed = meshes.CreateMesh(primitive::CreateMesh(primitive::Cuboid{}), {}, {});
        f.R.emplace<MeshConnectivity>(gone, std::move(removed.Connectivity));
        f.R.emplace<MeshHandle>(gone, MeshHandle{removed.StoreId});
        f.R.destroy(gone); // leaves a MeshConnectivity tombstone in the pool

        ProcessComponentEvents(f.R, f.Viewport);
        const auto before = snapshot::SaveState(f.R); // must not assert on the tombstone
        SceneFixture g{vk_resources};
        snapshot::LoadState(g.R, before);
        ProcessComponentEvents(g.R, g.Viewport);
        const auto after = snapshot::SaveState(g.R);
        const auto diff = snapshot::Compare(before, after);
        expect(diff.Equal) << "tombstone round-trip diverged at byte" << diff.FirstDifferingByte;
    };

    const auto load_ctx = [](entt::registry &r, entt::entity e) {
        return gltf::LoadContext{
            .R = r,
            .Viewport = e,
            .Slots = r.ctx().get<DescriptorSlots>(),
            .Buffers = r.ctx().get<GpuBuffers>(),
            .Meshes = r.ctx().get<MeshStore>(),
            .Textures = r.ctx().get<TextureStore>(),
            .Environments = r.ctx().get<EnvironmentStore>(),
        };
    };
    const auto save_ctx = [&](entt::registry &r, entt::entity e) {
        auto &buffers = r.ctx().get<GpuBuffers>();
        return gltf::SaveContext{
            .R = r,
            .Viewport = e,
            .Buffers = buffers,
            .Meshes = r.ctx().get<MeshStore>(),
            .Textures = r.ctx().get<TextureStore>(),
            .Vk = &vk_resources,
            .BufCtx = &buffers.Ctx,
        };
    };

    // The test never renders, so WaitForRender (the only caller of ReclaimRetiredBuffers) never runs and retired arena
    // buffers would pile up until the GPU OOMs. Reclaim at each clear - safe since no frame is ever in flight here.
    const auto clear_scene = [](entt::registry &r, entt::entity vp) {
        ClearScene(r, vp);
        r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    };

    SceneFixture fx{vk_resources};
    for (const auto &src : samples) {
        const auto sample_name = src.stem().string();

        test(sample_name) = [&] {
            ProcessComponentEvents(fx.R, fx.Viewport);
            clear_scene(fx.R, fx.Viewport);

            const auto load = gltf::LoadGltf(src, load_ctx(fx.R, fx.Viewport));
            if (!load) return; // Loader limitation on source (e.g., unsupported extension); not a roundtrip concern.

            const auto out_path = tmp_root / (sample_name + ".gltf");
            const auto save = gltf::SaveGltf(out_path, save_ctx(fx.R, fx.Viewport));
            expect(save.has_value()) << "SaveGltf failed: " << (save ? "" : save.error());
            if (!save) return;

            const auto unexpected = CompareGltfJson(src, out_path, sample_name);
            expect(unexpected == 0) << unexpected << " unexpected JSON diff(s)";
        };
    }

    // Drains PendingTextureUploads onto the GPU — ProcessComponentEvents minus the env /
    // sync passes — so the edit tests below can read texture pixels back.
    const auto materialize_textures = [&](entt::registry &r, entt::entity scene) {
        const auto pool = vk_resources.Device.createCommandPoolUnique({{}, vk_resources.QueueFamily});
        const auto fence = vk_resources.Device.createFenceUnique({});
        const auto *pending = r.try_get<const PendingTextureUploads>(scene);
        const auto *src = r.try_get<const gltf::SourceAssets>(scene);
        if (!pending || pending->Items.empty() || !src) return;
        auto &buffers = r.ctx().get<GpuBuffers>();
        auto &slots = r.ctx().get<DescriptorSlots>();
        auto &textures = r.ctx().get<TextureStore>();
        auto batch = BeginTextureUploadBatch(vk_resources.Device, *pool, buffers.Ctx);
        for (const auto &item : pending->Items) {
            if (auto entry = MaterializeTextureEntry(vk_resources, batch, slots, item, src->Images)) {
                textures.Textures.emplace_back(std::move(*entry));
            }
        }
        SubmitTextureUploadBatch(batch, vk_resources.Queue, *fence, vk_resources.Device);
        r.remove<PendingTextureUploads>(scene);
    };

    // Round-trip each import through SaveState -> restore -> SaveState and byte-compare, catching state that
    // doesn't reconstruct. restore_fx is the restore target. A SaveState that hits an unclassified component
    // throws and surfaces as a ut failure with the message.
    SceneFixture restore_fx{vk_resources};
    for (const auto &src : samples) {
        const auto sample_name = src.stem().string();
        test("snapshot round trip (" + sample_name + ")") = [&] {
            ProcessComponentEvents(fx.R, fx.Viewport);
            clear_scene(fx.R, fx.Viewport);
            const auto load = gltf::LoadGltf(src, load_ctx(fx.R, fx.Viewport));
            if (!load) return; // Loader limitation on source (e.g., unsupported extension); not a snapshot concern.

            ProcessComponentEvents(fx.R, fx.Viewport);
            const auto before = snapshot::SaveState(fx.R);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);
            clear_scene(restore_fx.R, restore_fx.Viewport);
            snapshot::LoadState(restore_fx.R, before);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);
            const auto diff = snapshot::Compare(before, snapshot::SaveState(restore_fx.R));
            expect(diff.Equal) << "glTF-import round-trip diverged at byte" << diff.FirstDifferingByte;
        };
    }

    // Assert fresh-load and restored registries match, including derived components the byte round-trip can't see.
    for (const auto &src : samples) {
        const auto sample_name = src.stem().string();
        test("full-state restore (" + sample_name + ")") = [&] {
            ProcessComponentEvents(fx.R, fx.Viewport);
            clear_scene(fx.R, fx.Viewport);
            const auto load = gltf::LoadGltf(src, load_ctx(fx.R, fx.Viewport));
            if (!load) return;
            ProcessComponentEvents(fx.R, fx.Viewport);
            const auto before = snapshot::SaveState(fx.R);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);
            clear_scene(restore_fx.R, restore_fx.Viewport);
            snapshot::LoadState(restore_fx.R, before);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);
            CompareRegistries(sample_name, fx.R, restore_fx.R);
        };
    }

    const auto edit_root = tmp_root / "edits";
    fs::create_directories(edit_root);

    // Mark the embedded variant's image dirty; saved bytes must pixel-equal the GPU readback.
    if (const fs::path box_embedded = SamplePath("../glTF-Sample-Assets/Models/BoxTextured/glTF-Embedded/BoxTextured.gltf"); fs::exists(box_embedded)) {
        test("dirty_image_re_encodes_pixel_equal") = [&] {
            SceneFixture fx{vk_resources};
            const auto load = gltf::LoadGltf(box_embedded, load_ctx(fx.R, fx.Viewport));
            expect(load.has_value()) << "load failed";
            if (!load) return;
            materialize_textures(fx.R, fx.Viewport);

            // Skip the WireRegistry default-white RawPixels texture (no SourceImageIndex link).
            const auto &textures = fx.R.ctx().get<TextureStore>();
            const TextureEntry *tex = nullptr;
            for (const auto &t : textures.Textures) {
                if (t.SourceImageIndex == 0) {
                    tex = &t;
                    break;
                }
            }
            expect(tex != nullptr) << "BoxTextured image was not materialized";
            if (!tex) return;
            const auto pool = vk_resources.Device.createCommandPoolUnique({{}, vk_resources.QueueFamily});
            const auto fence = vk_resources.Device.createFenceUnique({});
            const auto original_pixels = ReadbackTextureRgba8(vk_resources, fx.R.ctx().get<GpuBuffers>().Ctx, *pool, *fence, *tex);
            expect(original_pixels.has_value()) << "readback failed";
            if (!original_pixels) return;

            fx.R.get<gltf::SourceAssets>(fx.Viewport).Images.front().IsDirty = true;

            const auto out_path = edit_root / "BoxTextured-dirty.gltf";
            const auto save = gltf::SaveGltf(out_path, save_ctx(fx.R, fx.Viewport));
            expect(save.has_value()) << "save failed: " << (save ? "" : save.error());
            if (!save) return;

            SceneFixture fx2{vk_resources};
            const auto reload = gltf::LoadGltf(out_path, load_ctx(fx2.R, fx2.Viewport));
            expect(reload.has_value()) << "reload failed";
            if (!reload) return;
            const auto &reloaded = fx2.R.get<const gltf::SourceAssets>(fx2.Viewport).Images;
            expect(reloaded.size() == 1u);
            // PNG re-encode is lossless, so decoded pixels must match the pre-edit GPU readback.
            const auto decoded = DecodeImageRgba8(reloaded.front().Bytes, reloaded.front().Name);
            expect(decoded.has_value()) << "reloaded image failed to decode";
            if (!decoded) return;
            expect(decoded->Width == tex->Width);
            expect(decoded->Height == tex->Height);
            const bool pixels_match = decoded->Pixels == *original_pixels;
            expect(pixels_match) << "re-encoded pixels diverge from GPU readback";
        };
    }

    // Move the external PNG aside between load and save; the embed-as-PNG fallback should fire.
    const fs::path box_external = SamplePath("../glTF-Sample-Assets/Models/BoxTextured/glTF/BoxTextured.gltf");
    if (fs::exists(box_external)) {
        test("missing_external_source_falls_back_to_embedded_png") = [&] {
            const auto stage_dir = edit_root / "BoxTextured-external";
            fs::create_directories(stage_dir);
            for (const auto &entry : fs::directory_iterator{box_external.parent_path()}) {
                fs::copy_file(entry.path(), stage_dir / entry.path().filename(), fs::copy_options::overwrite_existing);
            }
            const auto staged_gltf = stage_dir / box_external.filename();
            const auto staged_png = stage_dir / "CesiumLogoFlat.png";
            expect(fs::exists(staged_png)) << "fixture missing PNG";

            SceneFixture fx{vk_resources};
            const auto load = gltf::LoadGltf(staged_gltf, load_ctx(fx.R, fx.Viewport));
            expect(load.has_value()) << "load failed";
            if (!load) return;
            materialize_textures(fx.R, fx.Viewport);

            fs::rename(staged_png, stage_dir / "CesiumLogoFlat.png.moved");

            const auto out_path = edit_root / "BoxTextured-fallback.gltf";
            const auto save = gltf::SaveGltf(out_path, save_ctx(fx.R, fx.Viewport));
            expect(save.has_value()) << "save failed: " << (save ? "" : save.error());
            if (!save) return;

            SceneFixture fx2{vk_resources};
            const auto reload = gltf::LoadGltf(out_path, load_ctx(fx2.R, fx2.Viewport));
            expect(reload.has_value()) << "reload failed";
            if (!reload) return;
            const auto &reloaded = fx2.R.get<const gltf::SourceAssets>(fx2.Viewport).Images;
            expect(reloaded.size() == 1u);
            if (reloaded.empty()) return;
            expect(reloaded.front().Uri.empty()) << "fallback should drop the URI";
            expect(reloaded.front().MimeType == gltf::MimeType::PNG);
        };
    }
}
