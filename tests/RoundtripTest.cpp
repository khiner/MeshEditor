#include "Path.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "RunSuites.h"
#include "action/Errors.h"
#include "audio/AcousticMaterial.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/ContactSurface.h"
#include "audio/ModalModelFile.h"
#include "audio/ModalModes.h"
#include "editor/Engine.h"
#include "numeric/VectorMath.h"
#include "project/Assets.h"
#include <barrier>
#include <future>
#ifdef SURFACE_AUDIO
#include "audio/surface/SurfaceAudio.h" // SurfaceRelief, UpdateSurfaceRelief
#endif
#include "gltf/GltfScene.h"
#include "gltf/SourceTexture.h"
#include "image/ImageDecode.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "mesh/PrimitiveType.h"
#include "mesh/Primitives.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "snapshot/SnapshotRoles.h"
#include "viewport/Viewport.h"

#include "numeric/FastGltf.h"
#include "state/Scene.h"
#include <boost/ut.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <simdjson.h>

#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <set>

namespace {
namespace fs = std::filesystem;
using std::ranges::to, std::views::join, std::views::transform;

std::vector<fs::path> CollectGltfSamples(const fs::path &root) {
    std::vector<fs::path> out;
    if (!fs::exists(root)) return out;
    for (const auto &entry : fs::directory_iterator{root}) {
        if (!entry.is_directory()) continue;
        // Collect all Khronos variant directories prefixed with glTF.
        // Use the sample root when no variant directory exists.
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

// Copy a sample and every file beside it into `dir`, so a test can move or rewrite the asset's external files.
// Returns the staged glTF.
fs::path StageSample(const fs::path &sample_gltf, const fs::path &dir) {
    fs::create_directories(dir);
    for (const auto &entry : fs::directory_iterator{sample_gltf.parent_path()}) {
        fs::copy_file(entry.path(), dir / entry.path().filename(), fs::copy_options::overwrite_existing);
    }
    return dir / sample_gltf.filename();
}

fs::path MakeRoundtripDir() {
    auto dir = fs::temp_directory_path() / "MeshEditor-roundtrip";
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// Resolve submodule samples from the CMake-provided repository root.
constexpr std::string_view SampleRoots[]{
    "external/glTF-Sample-Assets/Models",
    "external/glTF_Physics/samples",
};

fs::path SamplePath(std::string_view relative_to_repo_root) { return fs::path{MESHEDITOR_SOURCE_DIR} / relative_to_repo_root; }

// Exact patterns match one JSON path, while subtree patterns include descendants.
// Use subtree patterns only for encoding-dependent structures.
struct Exception {
    std::string_view Pattern;
    std::string_view Why;
};

constexpr Exception SubtreeExceptions[]{
    // Export repacks buffer and accessor layout while preserving referenced values.
    {"buffers", "re-packed into a single buffer on export"},
    {"bufferViews", "bufferView layout and order depend on our emission order"},
    {"accessors", "accessor order depends on our emission choices; position-indexed JSON compare can't tell semantically equivalent accessors apart (semantic shape is verified indirectly via mesh/primitive/animation references)"},

    // These extensions are omitted during import or export.
    {"extensionsUsed", "extensionsUsed is computed from what we re-emit; membership/order diverges when extension data isn't fully re-emitted"},
    {"extensions.KHR_xmp_json_ld", "not imported"},
    {"extensions.KHR_xmp", "not imported"},
    {"materials[*].extensions.KHR_materials_volume_scatter", "not imported"},
    {"materials[*].extensions.KHR_materials_retroreflection", "not imported"},

    {"animations[*].channels", "channels are re-emitted in scene entity order, so their positions and sampler indices differ from the source"},
    {"animations[*].samplers", "samplers follow the re-emitted channel order"},

    // Import duplicates lights shared by source nodes, changing table and node indices.
    {"extensions.KHR_lights_punctual.lights", "per-node PunctualLight components aren't deduped on save"},
};

constexpr Exception ExactExceptions[]{
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
    // fastgltf omits default KHR_texture_transform fields from base material textures.
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.scale", "default [1,1] scale omitted"},
    {"materials[*].pbrMetallicRoughness.metallicRoughnessTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].normalTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].normalTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.scale", "default [1,1] scale omitted"},
    {"materials[*].occlusionTexture.extensions.KHR_texture_transform.rotation", "default 0 rotation omitted"},
    {"materials[*].extensions.KHR_materials_volume.thicknessTexture.extensions.KHR_texture_transform.offset", "default [0,0] offset omitted"},
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
    // fastgltf omits default TRS fields and preserves matrix-form nodes as matrices.
    {"nodes[*].translation", "default [0,0,0] omitted"},
    {"nodes[*].rotation", "default [0,0,0,1] omitted"},
    {"nodes[*].scale", "default [1,1,1] omitted"},
    {"asset.copyright", "empty source copyright string omitted by fastgltf's writer"},
    {"extensions", "empty root-level extensions block omitted by fastgltf's writer (source emits `\"extensions\":{}`)"},
    {"materials[*].extensions", "empty material-level extensions block omitted"},
    {"nodes[*].extensions", "empty per-node extensions block omitted"},

    {"materials[*].normalTexture.scale", "fastgltf always emits scale on NormalTextureInfo"},
    {"materials[*].occlusionTexture.strength", "fastgltf always emits strength on OcclusionTextureInfo"},
    {"materials[*].extensions.KHR_materials_clearcoat.clearcoatNormalTexture.scale", "fastgltf always emits scale on NormalTextureInfo"},
    {"extensions.KHR_physics_rigid_bodies.physicsMaterials[*].frictionCombine", "always-emitted even when equal to default 'average'"},
    {"extensions.KHR_physics_rigid_bodies.physicsMaterials[*].restitutionCombine", "always-emitted even when equal to default 'average'"},
    {"extensions.KHR_physics_rigid_bodies.physicsJoints[*].limits[*].damping", "always-emitted even when 0.0"},
    {"extensions.KHR_physics_rigid_bodies.physicsJoints[*].drives[*].maxForce", "always-emitted even when float::max ('no limit')"},
    {"materials[*].pbrMetallicRoughness", "emitted even for extension-only / specular-glossiness materials"},

    {"images[*].bufferView", "bufferView index depends on emission order"},
    // Export converts strip, fan, and loop primitives to list form and changes their index accessors.
    {"meshes[*].primitives[*].indices", "strip/fan/loop primitive modes are unfolded to list mode on save; source Points with indices lose them"},

    {"meshes[*].primitives[*].material", "line/point primitives lose per-primitive material on import (merged across primitives without retaining material refs)"},
    {"meshes[*].extensions", "mesh-level extensions (e.g. KHR_xmp_json_ld) not re-emitted"},
    {"scenes[*].extras", "not tracked on Scene"},
    {"scenes[*].extensions", "scene-level extensions not re-emitted"},
    {"nodes[*].extensions.KHR_lights_punctual.light", "renumbered to match the un-deduped lights table"},
};

// Return the sole entity with C or null.
template<typename C> state::Entity NodeWith(state::Scene &r) {
    for (auto e : r.view<const C>()) return e;
    return state::Null;
}

// Check that a KHR_audio_rigid_bodies model's accessor reference has the given type and count.
// At namespace scope so `==` is the builtin comparison, not a boost::ut expression.
bool AccessorShapeIs(simdjson::dom::element root, simdjson::dom::element model, std::string_view key, std::string_view type, uint64_t count) {
    uint64_t idx;
    if (model[key].get_uint64().get(idx) != simdjson::SUCCESS) return false;
    auto acc = root["accessors"].at(idx);
    std::string_view t;
    uint64_t c = 0;
    return acc["type"].get_string().get(t) == simdjson::SUCCESS && t == type &&
        acc["count"].get_uint64().get(c) == simdjson::SUCCESS && c == count;
}

std::optional<simdjson::dom::element> OnlyAudioEntry(simdjson::dom::element doc, std::string_view key) {
    simdjson::dom::array table;
    if (doc["extensions"]["KHR_audio_rigid_bodies"][key].get_array().get(table) != simdjson::SUCCESS || table.size() != 1) return std::nullopt;
    simdjson::dom::element entry;
    if (table.at(0).get(entry) != simdjson::SUCCESS) return std::nullopt;
    return entry;
}

// The table index a node instances through `key` of the extension, or nullopt when no node does.
std::optional<uint64_t> NodeAudioIndex(simdjson::dom::element doc, std::string_view key) {
    for (auto node : doc["nodes"]) {
        uint64_t index;
        if (node["extensions"]["KHR_audio_rigid_bodies"][key].get_uint64().get(index) == simdjson::SUCCESS) return index;
    }
    return std::nullopt;
}

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

bool IsExpectedDivergence(std::string_view path) {
    const auto norm = NormalizePath(path);
    return std::ranges::any_of(ExactExceptions, [norm](const auto &ex) { return norm == ex.Pattern; }) ||
        std::ranges::any_of(SubtreeExceptions, [norm](const auto &ex) { return SubtreeMatches(norm, ex.Pattern); });
}

// --- Generic JSON comparator ---

struct Diff {
    std::string Path, Message;
};

// Allow relative error from JSON conversion and source values truncated to three or four significant digits.
constexpr double AbsEps = 1e-6;
constexpr double RelEps = 1e-3;

bool NumberEq(double a, double b) {
    if (a == b) return true;
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) == std::isnan(b);
    const double diff = std::abs(a - b), scale = std::max({1.0, std::abs(a), std::abs(b)});
    return diff <= AbsEps || diff <= RelEps * scale;
}

template<typename V> bool VecEq(const V &a, const V &b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) {
        if (!NumberEq(a[i], b[i])) return false;
    }
    return true;
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
        case ElType::BIGINT: return "bigint";
        case ElType::NULL_VALUE: return "null";
    }
    return "?";
}

std::string JoinKey(std::string_view path, std::string_view key) { return path.empty() ? std::string{key} : std::format("{}.{}", path, key); }
std::string JoinIndex(std::string_view path, size_t i) { return std::format("{}[{}]", path, i); }

// --- Semantic accessor-reference resolution ---
// Accessor indices are emission-order-dependent, so their numeric values never compare.
// Mesh-geometry references compare by dereferenced per-corner content (CompareMeshGeometry).
// Other references compare by the resolved accessor's (type, count) shape (CompareAccessorShape).
bool MatchesAnyChildOf(std::string_view norm, std::string_view prefix) {
    if (!norm.starts_with(prefix)) return false;
    const auto rest = norm.substr(prefix.size());
    return !rest.empty() && rest.find('.') == std::string_view::npos && rest.find('[') == std::string_view::npos;
}

bool IsMeshGeometryRefPath(std::string_view norm) {
    return norm == "meshes[*].primitives[*].indices" ||
        MatchesAnyChildOf(norm, "meshes[*].primitives[*].attributes.") ||
        MatchesAnyChildOf(norm, "meshes[*].primitives[*].targets[*].");
}

bool IsShapeAccessorRefPath(std::string_view norm) {
    return norm == "skins[*].inverseBindMatrices" ||
        norm == "animations[*].samplers[*].input" ||
        norm == "animations[*].samplers[*].output" ||
        MatchesAnyChildOf(norm, "nodes[*].extensions.EXT_mesh_gpu_instancing.attributes.");
}

std::optional<simdjson::dom::element> ResolveAccessor(simdjson::dom::element root, size_t index) {
    simdjson::dom::array accessors;
    if (root["accessors"].get_array().get(accessors) != simdjson::SUCCESS) return std::nullopt;
    if (index >= accessors.size()) return std::nullopt;
    return accessors.at(index);
}

// Compare accessor type and count after import normalizes component types and buffer layout.
// Indices widen to uint32, joints narrow to uint16, and quantized attributes decode to float.
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

template<typename T>
std::vector<T> DecodeAccessor(const fastgltf::Asset &asset, size_t accessor_index) {
    const auto &accessor = asset.accessors[accessor_index];
    std::vector<T> out(accessor.count);
    fastgltf::copyFromAccessor<T>(asset, accessor, out.data());
    return out;
}

bool IsTrianglePrimitive(fastgltf::PrimitiveType t) {
    return t == fastgltf::PrimitiveType::Triangles || t == fastgltf::PrimitiveType::TriangleStrip || t == fastgltf::PrimitiveType::TriangleFan;
}

// Corner vertex indices in the importer's triangulation order (strips/fans unfold to lists).
std::vector<uint32_t> CornerIndices(const fastgltf::Asset &asset, const fastgltf::Primitive &prim, size_t vertex_count) {
    std::vector<uint32_t> indices;
    if (prim.indicesAccessor) {
        indices = DecodeAccessor<uint32_t>(asset, *prim.indicesAccessor);
    } else {
        indices.resize(vertex_count);
        std::iota(indices.begin(), indices.end(), 0u);
    }
    std::vector<uint32_t> corners;
    if (indices.size() < 3) return corners;
    if (prim.type == fastgltf::PrimitiveType::TriangleStrip) {
        for (uint32_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) corners.insert(corners.end(), {indices[i], indices[i + 1], indices[i + 2]});
            else corners.insert(corners.end(), {indices[i + 1], indices[i], indices[i + 2]});
        }
    } else if (prim.type == fastgltf::PrimitiveType::TriangleFan) {
        for (uint32_t i = 1; i + 1 < indices.size(); ++i) corners.insert(corners.end(), {indices[0], indices[i], indices[i + 1]});
    } else {
        for (uint32_t i = 0; i + 2 < indices.size(); i += 3) corners.insert(corners.end(), {indices[i], indices[i + 1], indices[i + 2]});
    }
    return corners;
}

// Unit-direction compare: normalize both and accept a small angle.
// Faceted-face normals re-derive from positions on export, so exact equality is too strict for them.
bool DirectionEq(vec3 a, vec3 b) {
    const auto la = numeric::Length(a), lb = numeric::Length(b);
    if (la < 1e-6f || lb < 1e-6f) return VecEq(a, b);
    constexpr float CosTol = 0.999998f; // ~2 milliradians
    return numeric::Dot(a / la, b / lb) >= CosTol;
}

bool InfluencesEq(uvec4 ja, vec4 wa, uvec4 jb, vec4 wb) {
    const auto collect = [](uvec4 j, vec4 w) {
        std::vector<std::pair<uint32_t, float>> out;
        for (size_t i = 0; i < 4; ++i) {
            if (w[i] > 1e-6f) out.emplace_back(j[i], w[i]);
        }
        std::ranges::sort(out);
        return out;
    };
    const auto a = collect(ja, wa), b = collect(jb, wb);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].first != b[i].first || !NumberEq(a[i].second, b[i].second)) return false;
    }
    return true;
}

// Zip two corner streams over their dereferenced attribute values, reporting the first divergence.
template<typename T>
void CompareCornerValues(
    const std::vector<T> &a, const std::vector<T> &b,
    std::span<const uint32_t> corners_a, std::span<const uint32_t> corners_b,
    std::string_view path, auto &&eq, std::vector<Diff> &out
) {
    for (size_t k = 0; k < corners_a.size(); ++k) {
        const auto ia = corners_a[k], ib = corners_b[k];
        if (ia >= a.size() || ib >= b.size()) {
            out.emplace_back(std::string{path}, std::format("corner {} references out-of-range vertex", k));
            return;
        }
        if (!eq(a[ia], b[ib])) {
            out.emplace_back(std::string{path}, std::format("value diverged at corner {}", k));
            return;
        }
    }
}

// COLOR_n decodes to vec4 with the importer's VEC3 -> w=1 padding.
std::vector<vec4> DecodeColorAccessor(const fastgltf::Asset &asset, size_t accessor_index) {
    if (asset.accessors[accessor_index].type == fastgltf::AccessorType::Vec3) {
        const auto rgb = DecodeAccessor<vec3>(asset, accessor_index);
        std::vector<vec4> out;
        out.reserve(rgb.size());
        for (const auto &c : rgb) out.emplace_back(c, 1.f);
        return out;
    }
    return DecodeAccessor<vec4>(asset, accessor_index);
}

void CompareGeometryAttr(
    const fastgltf::Asset &a, const fastgltf::Asset &b,
    size_t acc_a, size_t acc_b, std::string_view name, bool morph_delta,
    std::span<const uint32_t> corners_a, std::span<const uint32_t> corners_b,
    std::string_view path, std::vector<Diff> &out
) {
    const auto compare = [&](auto &&decode, auto &&eq) {
        CompareCornerValues(decode(a, acc_a), decode(b, acc_b), corners_a, corners_b, path, eq, out);
    };
    if (morph_delta) {
        // Target deltas (POSITION/NORMAL/TANGENT) are all VEC3 and pass through import untouched.
        if (name == "POSITION" || name == "NORMAL" || name == "TANGENT") compare(DecodeAccessor<vec3>, VecEq<vec3>);
    } else if (name == "POSITION") {
        compare(DecodeAccessor<vec3>, VecEq<vec3>);
    } else if (name == "NORMAL") {
        compare(DecodeAccessor<vec3>, DirectionEq);
    } else if (name == "TANGENT") {
        compare(DecodeAccessor<vec4>, [](vec4 x, vec4 y) { return DirectionEq(vec3{x}, vec3{y}) && NumberEq(x.w, y.w); });
    } else if (name.starts_with("COLOR_")) {
        compare(DecodeColorAccessor, VecEq<vec4>);
    } else if (name.starts_with("TEXCOORD_")) {
        compare(DecodeAccessor<vec2>, VecEq<vec2>);
    }
    // JOINTS_n/WEIGHTS_n compare jointly in ComparePrimitiveGeometry.
    // Custom attributes stay uncompared since import drops them.
}

void ComparePrimitiveGeometry(
    const fastgltf::Asset &a, const fastgltf::Asset &b,
    const fastgltf::Primitive &pa, const fastgltf::Primitive &pb,
    std::string_view path, std::vector<Diff> &out
) {
    if (!IsTrianglePrimitive(pa.type) || !IsTrianglePrimitive(pb.type)) return;
    const auto *pos_a = pa.findAttribute("POSITION");
    const auto *pos_b = pb.findAttribute("POSITION");
    if (pos_a == pa.attributes.end() || pos_b == pb.attributes.end()) return;

    const auto corners_a = CornerIndices(a, pa, a.accessors[pos_a->accessorIndex].count);
    const auto corners_b = CornerIndices(b, pb, b.accessors[pos_b->accessorIndex].count);
    if (corners_a.size() != corners_b.size()) {
        out.emplace_back(std::string{path}, std::format("corner count {} vs {}", corners_a.size(), corners_b.size()));
        return;
    }
    if (corners_a.empty()) return;

    for (const auto &attr : pa.attributes) {
        const std::string_view name{attr.name};
        const auto *battr = pb.findAttribute(name);
        if (battr == pb.attributes.end()) continue; // presence diffs come from the JSON walk
        CompareGeometryAttr(a, b, attr.accessorIndex, battr->accessorIndex, name, false, corners_a, corners_b, std::format("{}.attributes.{}", path, name), out);
    }

    const auto *ja = pa.findAttribute("JOINTS_0");
    const auto *wa = pa.findAttribute("WEIGHTS_0");
    const auto *jb = pb.findAttribute("JOINTS_0");
    const auto *wb = pb.findAttribute("WEIGHTS_0");
    if (ja != pa.attributes.end() && wa != pa.attributes.end() && jb != pb.attributes.end() && wb != pb.attributes.end()) {
        const auto influences = [](const fastgltf::Asset &asset, size_t joints_acc, size_t weights_acc) {
            const auto joints = DecodeAccessor<uvec4>(asset, joints_acc);
            const auto weights = DecodeAccessor<vec4>(asset, weights_acc);
            std::vector<std::pair<uvec4, vec4>> out(std::min(joints.size(), weights.size()));
            for (size_t i = 0; i < out.size(); ++i) out[i] = {joints[i], weights[i]};
            return out;
        };
        CompareCornerValues(
            influences(a, ja->accessorIndex, wa->accessorIndex), influences(b, jb->accessorIndex, wb->accessorIndex),
            corners_a, corners_b, std::format("{}.attributes.JOINTS_0", path),
            [](const auto &x, const auto &y) { return InfluencesEq(x.first, x.second, y.first, y.second); }, out
        );
    }

    const auto target_count = std::min(pa.targets.size(), pb.targets.size());
    for (size_t t = 0; t < target_count; ++t) {
        for (const auto &attr : pa.targets[t]) {
            const std::string_view name{attr.name};
            const auto *battr = pb.findTargetAttribute(t, name);
            if (battr == pb.targets[t].end()) continue;
            CompareGeometryAttr(a, b, attr.accessorIndex, battr->accessorIndex, name, true, corners_a, corners_b, std::format("{}.targets[{}].{}", path, t, name), out);
        }
    }
}

void CompareMeshGeometry(const fastgltf::Asset &a, const fastgltf::Asset &b, std::vector<Diff> &out) {
    const auto mesh_count = std::min(a.meshes.size(), b.meshes.size());
    for (size_t mi = 0; mi < mesh_count; ++mi) {
        const auto prim_count = std::min(a.meshes[mi].primitives.size(), b.meshes[mi].primitives.size());
        for (size_t pi = 0; pi < prim_count; ++pi) {
            ComparePrimitiveGeometry(
                a, b, a.meshes[mi].primitives[pi], b.meshes[mi].primitives[pi],
                std::format("meshes[{}].primitives[{}]", mi, pi), out
            );
        }
    }
}

// The array's values, or nullopt when any element is not a number.
std::optional<std::vector<double>> NumberArray(simdjson::dom::array arr) {
    std::vector<double> out;
    for (auto e : arr) {
        if (!IsNumber(e.type())) return std::nullopt;
        out.emplace_back(AsNumber(e));
    }
    return out;
}

// Accept q and -q because matrix decomposition does not preserve quaternion sign.
bool QuaternionsEqual(simdjson::dom::array a, simdjson::dom::array b) {
    const auto va = NumberArray(a), vb = NumberArray(b);
    if (!va || !vb || va->size() != 4 || vb->size() != 4) return false;
    const auto eq_signed = [&](double sign) {
        for (size_t i = 0; i < 4; ++i) {
            if (!NumberEq((*va)[i], sign * (*vb)[i])) return false;
        }
        return true;
    };
    return eq_signed(1.0) || eq_signed(-1.0);
}

// glTF `scene.nodes` is an unordered set (schema: uniqueItems), so compare the root indices as a multiset — we emit them ascending, source order is arbitrary.
bool SameNumberMultiset(simdjson::dom::array a, simdjson::dom::array b) {
    auto va = NumberArray(a), vb = NumberArray(b);
    if (!va || !vb || va->size() != vb->size()) return false;
    std::ranges::sort(*va);
    std::ranges::sort(*vb);
    return va == vb;
}

void CompareJson(simdjson::dom::element a, simdjson::dom::element b, std::string_view path, std::vector<Diff> &out, simdjson::dom::element root_a, simdjson::dom::element root_b) {
    const auto ta = a.type(), tb = b.type();
    // Numbers compare with epsilon regardless of int/uint/double subtype.
    if (IsNumber(ta) && IsNumber(tb)) {
        const auto norm = NormalizePath(path);
        if (IsMeshGeometryRefPath(norm)) return; // content-compared by CompareMeshGeometry
        if (IsShapeAccessorRefPath(norm)) {
            const auto src_acc = ResolveAccessor(root_a, size_t(AsNumber(a)));
            const auto out_acc = ResolveAccessor(root_b, size_t(AsNumber(b)));
            if (!src_acc || !out_acc) out.emplace_back(std::string(path), "accessor reference out of range");
            else CompareAccessorShape(*src_acc, *out_acc, path, out);
            return;
        }
        if (!NumberEq(AsNumber(a), AsNumber(b))) out.emplace_back(std::string(path), std::format("number {} vs {}", AsNumber(a), AsNumber(b)));
        return;
    }
    if (ta != tb) {
        out.emplace_back(std::string(path), std::format("type {} vs {}", TypeName(ta), TypeName(tb)));
        return;
    }
    if (ta == ElType::ARRAY && NormalizePath(path) == "nodes[*].rotation") {
        if (!QuaternionsEqual(a, b)) out.emplace_back(std::string(path), "quaternion mismatch");
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
            if (asz != bsz) out.emplace_back(std::string(path), std::format("array size {} vs {}", asz, bsz));
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

// Compares parsed glTF JSON with known exceptions and reports up to 20 unexpected differences.
// Returns the total unexpected-difference count.
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

    // Geometry content compare over dereferenced accessor data (the JSON walk skips those refs).
    {
        const auto asset_a = gltf::ParseGltfAsset(a_path);
        const auto asset_b = gltf::ParseGltfAsset(b_path);
        if (asset_a && asset_b) CompareMeshGeometry(*asset_a, *asset_b, all_diffs);
        else all_diffs.emplace_back("meshes", std::format("geometry parse failed: {}", !asset_a ? asset_a.error() : asset_b.error()));
    }

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
        for (size_t i = 0; i < unexpected.size() && i < MaxReport; ++i) std::cerr << "    " << unexpected[i].Path << ": " << unexpected[i].Message << "\n";
        if (unexpected.size() > MaxReport) std::cerr << "    ... and " << (unexpected.size() - MaxReport) << " more\n";
    }
    return unexpected.size();
}

// Require identical component presence and comparable values for every entity.
void CompareRegistries(std::string_view name, state::Scene &a, state::Scene &b) {
    using namespace boost::ut;
    const auto components_by_entity = [](state::Scene &r) {
        std::map<state::Entity, std::set<std::string>> m;
        for (auto [id, set] : r.storage()) {
            const std::string_view tn{state::SchemaNames[id]};
            for (const auto e : set.entities()) m[e].insert(std::string{tn});
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
            for (const auto &c : comps) {
                if (!it->second.contains(c)) ++diff_comps["-" + c];
            }
            for (const auto &c : it->second) {
                if (!comps.contains(c)) ++diff_comps["+" + c];
            }
        }
    }
    for (const auto &[e, comps] : cb) {
        if (!ca.contains(e)) {
            ++only_b;
            for (const auto &c : comps) ++only_b_comps[c];
        }
    }

    // ComponentValuesEqual returns nullopt for derived components without serializers.
    // Meshlet arena ranges follow build order, and a restore rebuilds reclassified meshes after the import built them.
    std::map<std::string, int> value_diffs;
    for (auto [id, a_set] : a.storage()) {
        const auto tn = state::SchemaNames[id];
        const auto &b_set = b.storage(id);
        if (id == state::Type<MeshBuffers>()) continue;
        for (const auto e : a_set.entities()) {
            if (!b_set.contains(e)) continue;
            const auto eq = snapshot::ComponentValuesEqual(id, a_set.value(e), b_set.value(e));
            if (eq && !*eq) ++value_diffs[std::string{tn}];
        }
    }

    // Require exact component presence, including types excluded from value comparison.
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

    std::string value_detail;
    for (const auto &[type, n] : value_diffs) value_detail += std::format(" {}({})", type, n);
    expect(value_diffs.empty()) << name << "value diverged for component(s):" << value_detail;
}

const ModalModelData SampleModal{
    .Modes = {
        {
            .Freqs = {110.f, 275.5f},
            .T60s = {1.5f, 0.8f},
            .Shapes = {{{0.1f, 0.2f, -0.3f}, {0.02f, -0.11f, 0.4f}}, {{-0.2f, 0.15f, 0.25f}, {0.3f, 0.1f, -0.2f}}, {{0.05f, -0.3f, 0.12f}, {-0.4f, 0.22f, 0.07f}}},
            .Positions = {{0.f, 0.f, 0.f}, {0.4f, -0.2f, 0.1f}, {-0.1f, 0.5f, 0.3f}},
            .OriginalFundamentalFreq = 0.f,
        },
        {0, 1, 2},
        {0, 1, 2},
    },
    .Mass = {2.5, {0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {1.f, 0.f, 0.f, 0.f}},
    .Tets = {{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}}, {0, 1}},
    .Summary = {
        {
            .Eigenvalues = {4.7e5, 3.0e6},
            .Shapes = {{{0.1f, 0.2f, -0.3f}, {0.02f, -0.11f, 0.4f}}, {{-0.2f, 0.15f, 0.25f}, {0.3f, 0.1f, -0.2f}}, {{0.05f, -0.3f, 0.12f}, {-0.4f, 0.22f, 0.07f}}},
            .SolvedMaterial = materials::acoustic::Ceramic.Properties,
        },
        1,
        2,
        {0, 1, 2},
    },
};

struct SceneFixture : Engine {
    // Imports keep source image URIs while no asset store is present, so the glTF comparison sees the source layout.
    // Project operations need the store, so it exists only while a project is open.
    SceneFixture() : Engine{false} { R.ctx().erase<project::Assets>(); }
    void Check(bool ok) {
        boost::ut::expect(ok);
        if (ok) return;
        for (const auto &message : R.ctx().get<action::Errors>().Messages) std::cerr << "  project: " << message << "\n";
        if (const auto error = P->History.TakeIntegrityError(); !error.empty()) std::cerr << "  history: " << error << "\n";
    }
    // Start a project at `dir`, save live state into it, close it, and return the persistent image.
    std::vector<std::byte> SaveTo(const std::filesystem::path &dir) {
        R.ctx().emplace<project::Assets>();
        Check(P->Begin(dir));
        Check(P->Save());
        auto image = P->History.MaterializeLive();
        Check(P->Close());
        R.ctx().erase<project::Assets>();
        return image;
    }
    // Restore the project at `dir` and return the persistent image.
    std::vector<std::byte> LoadFrom(const std::filesystem::path &dir) {
        R.ctx().emplace<project::Assets>();
        Check(P->Open(dir));
        return P->History.MaterializeLive();
    }
};

// The first mesh-instance node, or null.
state::Entity FirstMeshNode(state::Scene &r) {
    for (auto e : r.view<const Instance, const GltfNode>()) {
        if (r.all_of<MeshHandle>(r.get<const Instance>(e).Entity)) return e;
    }
    return state::Null;
}

// The reloaded scene of a save/load round trip, with the first entity carrying T. Value is null when any step failed.
template<typename T> struct Roundtripped {
    std::unique_ptr<SceneFixture> Scene;
    state::Entity Node{state::Null};
    const T *Value{nullptr};
};

// Loads `sample` into a fresh scene, runs `author` on it with its first mesh-instance node, saves to `out_path`, and reloads the file into a second fresh scene.
template<typename T> Roundtripped<T> RoundtripComponent(const fs::path &sample, const fs::path &out_path, auto &&author) {
    using namespace boost::ut;
    Roundtripped<T> out;
    SceneFixture fx;
    const auto load = gltf::LoadGltf(sample, fx.R, fx.Viewport);
    expect(load.has_value()) << "load failed: " << (load ? "" : load.error());
    if (!load) return out;
    const auto node = FirstMeshNode(fx.R);
    expect(node != state::Null) << "no mesh instance node in " << sample.stem().string();
    if (node == state::Null) return out;
    author(fx, node);
    const auto save = gltf::SaveGltf(out_path, fx.R, fx.Viewport);
    expect(save.has_value()) << "save failed: " << (save ? "" : save.error());
    if (!save) return out;
    out.Scene = std::make_unique<SceneFixture>();
    const auto reload = gltf::LoadGltf(out_path, out.Scene->R, out.Scene->Viewport);
    expect(reload.has_value()) << "reload failed: " << (reload ? "" : reload.error());
    if (!reload) return out;
    out.Node = NodeWith<T>(out.Scene->R);
    expect(out.Node != state::Null) << "no entity carries the component after reload";
    state::Scene &reloaded = out.Scene->R;
    if (out.Node != state::Null) out.Value = &reloaded.get<const T>(out.Node);
    return out;
}

bool FloatsEq(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](float x, float y) { return NumberEq(x, y); });
}
bool Vec3sEq(std::span<const vec3> a, std::span<const vec3> b) {
    return a.size() == b.size() && std::ranges::equal(a, b, VecEq<vec3>);
}
// Tolerance equality over the fields a glTF round trip carries.
bool ModesEq(const ModalModes &a, const ModalModes &b) {
    return FloatsEq(a.Freqs, b.Freqs) && FloatsEq(a.T60s, b.T60s) && Vec3sEq(a.Positions, b.Positions) && a.Indices == b.Indices &&
        a.Shapes.size() == b.Shapes.size() && std::ranges::equal(a.Shapes, b.Shapes, Vec3sEq);
}
bool SurfacesEq(const ContactSurface &a, const ContactSurface &b) {
    const bool textures_eq = a.NormalTexture.has_value() == b.NormalTexture.has_value() &&
        (!a.NormalTexture || (a.NormalTexture->Texture == b.NormalTexture->Texture && a.NormalTexture->TexCoord == b.NormalTexture->TexCoord && NumberEq(a.NormalTexture->Scale, b.NormalTexture->Scale)));
    return a.Name == b.Name && NumberEq(a.Roughness, b.Roughness) && NumberEq(a.CorrelationLength, b.CorrelationLength) && NumberEq(a.SpectralSlope, b.SpectralSlope) &&
        NumberEq(a.ShortWavelength, b.ShortWavelength) && NumberEq(a.Waviness, b.Waviness) && NumberEq(a.WavinessLength, b.WavinessLength) &&
        FloatsEq(a.Profile, b.Profile) && NumberEq(a.SampleSpacing, b.SampleSpacing) && textures_eq;
}
bool AcousticMaterialsEq(const AcousticMaterial &a, const AcousticMaterial &b) {
    const auto &pa = a.Properties, &pb = b.Properties;
    return a.Name == b.Name && NumberEq(pa.Density, pb.Density) && NumberEq(pa.YoungModulus, pb.YoungModulus) && NumberEq(pa.PoissonRatio, pb.PoissonRatio) && NumberEq(pa.Alpha, pb.Alpha) && NumberEq(pa.Beta, pb.Beta);
}

// A glTF with one skinned triangle. The joint sits under a rig node the scene never reaches, so the armature root check fails, and dropping WEIGHTS_0 fails the mesh parse instead.
std::string BrokenSkinnedTriangleGltf(bool drop_weights) {
    // 3 positions, 3 ubyte4 joints, and 3 float4 weights, all zero.
    const std::string buffer(128, 'A');
    return std::format(R"({{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [
    {{"mesh": 0, "skin": 0, "name": "Skinned"}},
    {{"children": [2], "name": "Rig"}},
    {{"name": "Joint"}}
  ],
  "skins": [{{"joints": [2]}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "JOINTS_0": 1{}}}}}]}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [0, 0, 0]}},
    {{"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"}},
    {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"}}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": 48, "byteLength": 48}}
  ],
  "buffers": [{{"byteLength": 96, "uri": "data:application/octet-stream;base64,{}"}}]
}})",
                       drop_weights ? "" : ", \"WEIGHTS_0\": 2", buffer);
}

// Every count a load changes, so a failed load can prove it changed none of them.
// Every bindless slot an import allocates pairs with a pending upload or environment import, which these counts cover.
struct SceneCounts {
    size_t Entities, MeshHandles, Textures, PendingUploads, Materials, MaterialNames, MaterializedTextures;
    uint64_t VertexBytes;
    bool PendingEnvironment, SourceAssets;
    bool operator==(const SceneCounts &) const = default;
};
SceneCounts CountScene(SceneFixture &f) {
    auto &c = f.R.ctx();
    return {
        .Entities = f.R.storage<state::Entity>().size(),
        .MeshHandles = f.R.storage<MeshHandle>().size(),
        .Textures = c.get<TextureStore>().Textures.size(),
        .PendingUploads = c.get<TextureStore>().PendingUploads.size(),
        .Materials = c.get<GpuBuffers>().Materials.Count<PBRMaterial>(),
        .MaterialNames = c.get<MaterialStore>().Names.size(),
        .MaterializedTextures = f.R.storage<MaterializedTextures>().size(),
        .VertexBytes = c.get<MeshStore>().Arenas().Vertices.Buffer.UsedSize,
        .PendingEnvironment = c.get<EnvironmentStore>().PendingImport.has_value(),
        .SourceAssets = f.R.all_of<gltf::SourceAssets>(f.Viewport),
    };
}
} // namespace
int main(int argc, const char **argv) {
    using namespace boost::ut;

    // Accept a Boost.UT test-name filter.
    if (argc > 1) cfg<override> = {.filter = argv[1]};

    // Resolve symlinked resources and shaders from the build directory.
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);

    const auto tmp_root = MakeRoundtripDir();
    Paths::SetProject(tmp_root);
    const auto samples = SampleRoots | transform([](auto root) { return CollectGltfSamples(SamplePath(root)); }) | join | to<std::vector>();

    "snapshot encoding appends within reserved capacity"_test = [] {
        constexpr size_t prefix = 65536;
        std::vector<std::byte> bytes(prefix, std::byte{42});
        bytes.reserve(prefix + 256);
        const auto capacity = bytes.capacity();
        Name name{"appended record"}, decoded;
        snapshot::SnapshotTable()[state::Type<Name>()].Serialize(&name, bytes);
        expect(bytes.capacity() == capacity);
        expect(std::ranges::all_of(bytes | std::views::take(prefix), [](auto b) { return b == std::byte{42}; }));
        zpp::bits::in{std::span{bytes}.subspan(prefix)}(decoded).or_throw();
        expect(decoded.Value == name.Value);
    };

    // Require an identical persistent image and registry after restoring into a fresh scene.
    "project save/restore round trip"_test = [&] {
        SceneFixture f;
        {
            auto &meshes = f.R.ctx().get<MeshStore>();
            const auto created = CreateMesh(f.R, {.Data = primitive::CreateMesh(primitive::Cuboid{})});
            const auto e = f.R.create();
            f.R.emplace<MeshHandle>(e, MeshHandle{created.StoreId});
            f.R.emplace<Name>(e, "Cube");
            f.R.emplace<ObjectKind>(e, ObjectType::Mesh);
            f.R.emplace<MeshActiveElement>(e, 7u);
            f.R.emplace<Selected>(e);
            f.R.emplace<Path>(e, "/tmp/scene.gltf");

            const auto light = f.R.create();
            f.R.emplace<PunctualLight>(light, PunctualLight{.Range = 12.f, .Color = {0.2f, 0.4f, 0.6f}, .Intensity = 3.f});
            f.R.emplace<Name>(light, "Lamp");

            const auto sound = f.R.create();
            f.R.emplace<Instance>(sound, Instance{e});
            f.R.emplace<ModalModes>(sound, SampleModal.Modes);
            f.R.emplace<MassProperties>(sound, SampleModal.Mass);
            f.R.emplace<ModalEigenSummary>(sound, SampleModal.Summary);
            f.R.emplace<SoundVerticesModel>(sound, SoundVerticesModel::Modal);
            f.R.emplace<ModalGain>(sound, ModalGain{0.6f});
            f.R.emplace<ModalTuning>(sound, ModalTuning{440.f, 1.2f});
            f.R.emplace<AcousticMaterial>(e, materials::acoustic::Ceramic);
            f.R.emplace<TetBuffers>(e, meshes.AllocateTets(SampleModal.Tets.Positions, SampleModal.Tets.EdgeIndices));

            ProcessComponentEvents(f.R, f.Viewport);
        }
        const auto dir = tmp_root / "roundtrip.project";
        const auto before = f.SaveTo(dir);
        expect(before.size() > sizeof(uint64_t));

        SceneFixture g;
        const auto after = g.LoadFrom(dir);
        ProcessComponentEvents(g.R, g.Viewport);
        expect(after == before) << "round-trip diverged at byte" << std::ranges::mismatch(before, after).in1 - before.begin() << "of" << before.size() << "/" << after.size();
        CompareRegistries("roundtrip", f.R, g.R);
    };

    // Require exact modal-result round trips and reuse of identical content-addressed files.
    "modal model file round trip"_test = [] {
        project::Assets assets{.Directory = Paths::Project()};
        const auto stored = SaveModalModelFile(assets, SampleModal);
        expect(bool(stored));
        if (!stored) return;
        const auto loaded = LoadModalModelFile(assets.Resolve(*stored));
        expect(bool(loaded));
        if (!loaded) return;
        expect(*loaded == SampleModal);
        expect(SaveModalModelFile(assets, SampleModal) == stored);
        auto changed = SampleModal;
        changed.Modes.Freqs[0] += 1.f;
        std::barrier gate{4};
        std::array<std::future<std::expected<std::filesystem::path, std::string>>, 4> writes;
        for (auto &write : writes) write = std::async(std::launch::async, [&, destination = assets]() mutable {
                                       gate.arrive_and_wait();
                                       return SaveModalModelFile(destination, changed);
                                   });
        for (auto &write : writes) {
            const auto result = write.get();
            expect(bool(result));
            if (result) expect(LoadModalModelFile(assets.Resolve(*result)) == changed);
        }
    };

    // A destroyed entity leaves deletion history in its pools, and the persistent image reflects live state alone.
    "project save omits destroyed mesh entities"_test = [&] {
        SceneFixture f;
        const auto keep = f.R.create();
        const auto kept = CreateMesh(f.R, {.Data = primitive::CreateMesh(primitive::Cuboid{})});
        f.R.emplace<MeshHandle>(keep, MeshHandle{kept.StoreId});

        const auto gone = f.R.create();
        const auto removed = CreateMesh(f.R, {.Data = primitive::CreateMesh(primitive::Cuboid{})});
        f.R.emplace<MeshHandle>(gone, MeshHandle{removed.StoreId});
        f.R.destroy(gone);

        ProcessComponentEvents(f.R, f.Viewport);
        const auto dir = tmp_root / "destroyed.project";
        const auto before = f.SaveTo(dir);
        SceneFixture g;
        const auto after = g.LoadFrom(dir);
        ProcessComponentEvents(g.R, g.Viewport);
        expect(after == before) << "destroyed-entity round-trip diverged at byte" << std::ranges::mismatch(before, after).in1 - before.begin();
        CompareRegistries("destroyed", f.R, g.R);
    };

    // Reclaim retired arena buffers after each clear because this test has no render frames in flight.
    const auto clear_scene = [](state::Scene &r, state::Entity vp) {
        ClearScene(r, vp);
        r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    };

    // Check each sample through JSON comparison and byte-identical project restoration.
    // Compare registries for derived components omitted from the persistent image.
    SceneFixture fx;
    SceneFixture restore_fx;
    size_t sample_index = 0;
    for (const auto &src : samples) {
        const auto sample_name = src.stem().string();
        // Variants of one model share a stem, so number the project directories.
        const auto dir = tmp_root / std::format("{}_{}.project", sample_name, sample_index++);

        test(sample_name) = [&] {
            ProcessComponentEvents(fx.R, fx.Viewport);
            clear_scene(fx.R, fx.Viewport);

            const auto load = gltf::LoadGltf(src, fx.R, fx.Viewport);
            if (!load) return; // Loader limitation on source (e.g., unsupported extension); skips both round-trips.
            ProcessComponentEvents(fx.R, fx.Viewport); // mirror prod: a frame runs (posing skinned models) before save

            const auto out_path = tmp_root / (sample_name + ".gltf");
            const auto save = gltf::SaveGltf(out_path, fx.R, fx.Viewport);
            expect(save.has_value()) << "SaveGltf failed: " << (save ? "" : save.error());
            if (save) {
                const auto unexpected = CompareGltfJson(src, out_path, sample_name);
                expect(unexpected == 0) << unexpected << " unexpected JSON diff(s)";
            }

            const auto before = fx.SaveTo(dir);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);
            clear_scene(restore_fx.R, restore_fx.Viewport);
            const auto after = restore_fx.LoadFrom(dir);
            ProcessComponentEvents(restore_fx.R, restore_fx.Viewport);

            expect(after == before) << "persistent image diverged at byte" << std::ranges::mismatch(before, after).in1 - before.begin();
            CompareRegistries(sample_name, fx.R, restore_fx.R);
        };
    }

    const auto edit_root = tmp_root / "edits";
    fs::create_directories(edit_root);

    // Mark the embedded variant's image dirty; saved bytes must pixel-equal the GPU readback.
    if (const fs::path box_embedded = SamplePath("external/glTF-Sample-Assets/Models/BoxTextured/glTF-Embedded/BoxTextured.gltf"); fs::exists(box_embedded)) {
        test("dirty_image_re_encodes_pixel_equal") = [&] {
            std::vector<std::byte> original_pixels;
            uint32_t width = 0, height = 0;
            const auto reloaded = RoundtripComponent<gltf::SourceAssets>(box_embedded, edit_root / "BoxTextured-dirty.gltf", [&](SceneFixture &fx, state::Entity) {
                // A frame materializes the pending upload so the readback sees the texture.
                ProcessComponentEvents(fx.R, fx.Viewport);
                const auto &textures = fx.R.ctx().get<TextureStore>().Textures;
                const auto tex = std::ranges::find(textures, 0u, &TextureEntry::SourceImageIndex);
                expect(tex != textures.end()) << "BoxTextured image was not materialized";
                if (tex == textures.end()) return;
                auto pixels = ReadbackTextureRgba8(fx.R.ctx().get<const mtl::Context>(), *tex);
                expect(pixels.has_value()) << "readback failed";
                if (!pixels) return;
                original_pixels = std::move(*pixels);
                width = tex->Image.Extent.Width;
                height = tex->Image.Extent.Height;
                fx.R.edit<gltf::SourceAssets>(fx.Viewport).Images.front().IsDirty = true;
            });
            if (!reloaded.Value || original_pixels.empty()) return;
            const auto &images = reloaded.Value->Images;
            expect(images.size() == 1u);
            // PNG re-encode is lossless, so decoded pixels must match the pre-edit GPU readback.
            const auto decoded = DecodeImageRgba8(images.front().Bytes, images.front().Name);
            expect(decoded.has_value()) << "reloaded image failed to decode";
            if (!decoded) return;
            expect(decoded->Width == width && decoded->Height == height);
            expect(decoded->Pixels == original_pixels) << "re-encoded pixels diverge from GPU readback";
        };
    }

    // Move the external PNG aside between load and save; the embed-as-PNG fallback should fire.
    const fs::path box_external = SamplePath("external/glTF-Sample-Assets/Models/BoxTextured/glTF/BoxTextured.gltf");
    if (fs::exists(box_external)) {
        test("missing_external_source_falls_back_to_embedded_png") = [&] {
            const auto stage_dir = edit_root / "BoxTextured-external";
            const auto staged_gltf = StageSample(box_external, stage_dir);
            const auto staged_png = stage_dir / "CesiumLogoFlat.png";
            expect(fs::exists(staged_png)) << "fixture missing PNG";
            const auto reloaded = RoundtripComponent<gltf::SourceAssets>(staged_gltf, edit_root / "BoxTextured-fallback.gltf", [&](SceneFixture &fx, state::Entity) {
                ProcessComponentEvents(fx.R, fx.Viewport);
                fs::rename(staged_png, stage_dir / "CesiumLogoFlat.png.moved");
            });
            if (!reloaded.Value) return;
            const auto &images = reloaded.Value->Images;
            expect(images.size() == 1u);
            if (images.empty()) return;
            expect(images.front().Source == gltf::Image::SourceKind::Embedded) << "fallback should embed";
            expect(images.front().MimeType == gltf::MimeType::PNG);
        };
    }

    // Require KHR_audio_rigid_bodies modal data and schema shape to survive export and re-import.
    if (const fs::path box = SamplePath("external/glTF-Sample-Assets/Models/Box/glTF/Box.gltf"); fs::exists(box)) {
        test("audio_modal_round_trip") = [&] {
            // A small modal model on the node, with its derivation material.
            ModalModes modes{
                {
                    .Freqs = {110.f, 275.5f, 431.2f},
                    .T60s = {1.5f, 0.8f, 0.32f},
                    .Shapes = {
                        {{0.1f, 0.2f, -0.3f}, {0.02f, -0.11f, 0.4f}, {-0.05f, 0.06f, 0.07f}},
                        {{-0.2f, 0.15f, 0.25f}, {0.3f, 0.1f, -0.2f}, {0.01f, -0.02f, 0.03f}},
                        {{0.05f, -0.3f, 0.12f}, {-0.4f, 0.22f, 0.07f}, {0.08f, 0.09f, -0.01f}},
                    },
                    .Positions = {{0.f, 0.f, 0.f}, {0.4f, -0.2f, 0.1f}, {-0.1f, 0.5f, 0.3f}},
                },
                {},
                {0, 1, 2},
            };
            const auto out_path = edit_root / "audio_modal.gltf";
            const auto reloaded = RoundtripComponent<ModalModes>(box, out_path, [&](SceneFixture &fx, state::Entity node) {
                fx.R.emplace<ModalModes>(node, modes);
                fx.R.emplace<ModalGain>(node, ModalGain{0.6f});
                fx.R.emplace_or_replace<AcousticMaterial>(node, materials::acoustic::Ceramic);
            });
            if (!reloaded.Value) return;

            // Schema-shape checks on the emitted JSON.
            {
                simdjson::dom::parser p;
                simdjson::dom::element doc;
                expect(p.load(out_path.string()).get(doc) == simdjson::SUCCESS) << "emitted json failed to parse";

                bool lists_extension = false;
                for (auto e : doc["extensionsUsed"]) {
                    if (std::string_view{e} == "KHR_audio_rigid_bodies") lists_extension = true;
                }
                expect(lists_extension) << "KHR_audio_rigid_bodies missing from extensionsUsed";

                const auto model0 = OnlyAudioEntry(doc, "modalModels");
                expect(model0.has_value()) << "expected one modal model";
                expect(OnlyAudioEntry(doc, "acousticMaterials").has_value()) << "expected one acoustic material";
                if (!model0) return;
                expect(AccessorShapeIs(doc, *model0, "frequencies", "SCALAR", 3)) << "frequencies accessor shape";
                expect(AccessorShapeIs(doc, *model0, "decayRates", "SCALAR", 3)) << "decayRates accessor shape";
                expect(AccessorShapeIs(doc, *model0, "positions", "VEC3", 3)) << "positions accessor shape";
                expect(AccessorShapeIs(doc, *model0, "shapes", "VEC3", 9)) << "shapes accessor shape (mode-major M*P)";
                expect(AccessorShapeIs(doc, *model0, "indices", "SCALAR", 3)) << "indices accessor shape";

                const auto instanced = NodeAudioIndex(doc, "modalModel");
                expect(instanced.has_value()) << "no node instances the modal model";
                if (instanced) expect(*instanced == 0u) << "node model index";
            }

            auto &r = reloaded.Scene->R;
            const auto rnode = reloaded.Node;
            expect(ModesEq(*reloaded.Value, modes)) << "modal model diverged";
            const auto *rgain = r.try_get<const ModalGain>(rnode);
            expect(rgain != nullptr && NumberEq(rgain->Value, 0.6f)) << "gain diverged";
            expect(r.all_of<Instance>(rnode)) << "modal node lost its mesh instance";
            const auto *rmat = r.try_get<const AcousticMaterial>(rnode);
            expect(rmat != nullptr && AcousticMaterialsEq(*rmat, materials::acoustic::Ceramic)) << "acoustic material not restored on the node";
            // Import maps sample points to mesh vertices and marks the entity modal, so it is a playable sound object.
            expect(reloaded.Value->Vertices.size() == modes.Positions.size()) << "sample points not mapped to mesh vertices";
            expect(r.all_of<SoundVerticesModel>(rnode) && r.get<const SoundVerticesModel>(rnode) == SoundVerticesModel::Modal) << "imported model not set up as a modal sound object";
        };
    }

    // Require exact import of a hand-authored meshless KHR_audio_rigid_bodies node.
    if (const fs::path fixture = SamplePath("tests/fixtures/KHR_audio_rigid_bodies.gltf"); fs::exists(fixture)) {
        test("audio_modal_decode_fixture") = [&] {
            SceneFixture fx;
            const auto load = gltf::LoadGltf(fixture, fx.R, fx.Viewport);
            expect(load.has_value()) << "fixture load failed";
            if (!load) return;

            const auto node = NodeWith<ModalModes>(fx.R);
            expect(node != state::Null) << "fixture produced no modal model";
            if (node == state::Null) return;

            const auto &m = fx.R.get<const ModalModes>(node);
            expect(m.Freqs.size() == 1u && NumberEq(m.Freqs[0], 220.0)) << "frequency";
            constexpr double ExpectedT60 = 6.907755278982137 / 2.0; // ln(1000) / decayRate(2.0)
            expect(m.T60s.size() == 1u && NumberEq(m.T60s[0], ExpectedT60)) << "T60 from decay rate";
            expect(m.Positions.size() == 1u && VecEq(m.Positions[0], vec3{1.5f, -0.5f, 0.25f})) << "position";
            expect(m.Shapes.size() == 1u && m.Shapes[0].size() == 1u && VecEq(m.Shapes[0][0], vec3{0.3f, 0.4f, -0.6f})) << "shape";

            const auto *gain = fx.R.try_get<const ModalGain>(node);
            expect(gain != nullptr && NumberEq(gain->Value, 0.75)) << "gain";

            const auto floor_node = NodeWith<ContactSurface>(fx.R);
            expect(floor_node != state::Null) << "fixture produced no contact surface";
            if (floor_node == state::Null) return;
            const auto &cs = fx.R.get<const ContactSurface>(floor_node);
            expect(cs.Name == "TestFinish") << "surface name";
            expect(NumberEq(cs.Roughness, 3e-6) && NumberEq(cs.CorrelationLength, 7e-5) && NumberEq(cs.SpectralSlope, -1.25)) << "surface parameters";
            expect(cs.NormalTexture.has_value()) << "surface normal texture";
            if (cs.NormalTexture) {
                expect(cs.NormalTexture->Texture == 0u && cs.NormalTexture->TexCoord == 0u && NumberEq(cs.NormalTexture->Scale, 0.8)) << "normal texture info";
            }
            const auto *floor_material = fx.R.try_get<const AcousticMaterial>(floor_node);
            expect(floor_material != nullptr && floor_material->Name == "TestMat") << "surface material";
        };
    }

    // KHR_audio_rigid_bodies acoustic surfaces round-trip on their own, with a real texture reference.
    // A body may supply only its finish, which is the floor-and-table case sustained contact needs.
    if (const fs::path box_textured = SamplePath("external/glTF-Sample-Assets/Models/BoxTextured/glTF/BoxTextured.gltf"); fs::exists(box_textured)) {
        test("audio_surface_round_trip") = [&] {
            // Stage the asset so its external PNG sits beside the file this test writes back out.
            const auto stage_dir = edit_root / "audio-surface";
            const auto staged_gltf = StageSample(box_textured, stage_dir);
            const ContactSurface surface{
                .Name = "Tiled floor",
                .Roughness = 8e-6f,
                .CorrelationLength = 8e-5f,
                .SpectralSlope = -1.15f,
                .ShortWavelength = 4e-6f,
                .Waviness = 3e-5f,
                .WavinessLength = 6e-3f,
                .Profile = {0.f, 1e-6f, -1e-6f, 0.5e-6f},
                .SampleSpacing = 5e-6f,
                .NormalTexture = SurfaceNormalTexture{.Texture = 0, .TexCoord = 0, .Scale = 0.8f},
            };
            const auto out_path = stage_dir / "audio_surface.gltf";
            const auto reloaded = RoundtripComponent<ContactSurface>(staged_gltf, out_path, [&](SceneFixture &fx, state::Entity node) {
                fx.R.emplace_or_replace<ContactSurface>(node, surface);
#ifdef SURFACE_AUDIO
                // A texel spans a real distance along the surface, set by the mesh's own UV parameterization.
                const auto mesh_entity = fx.R.get<const Instance>(node).Entity;
                UpdateSurfaceRelief(fx.R, node, mesh_entity, true);
                const auto *relief = fx.R.try_get<const SurfaceRelief>(node);
                expect(relief != nullptr) << "no mesoscale relief derived from the normal map";
                if (relief) {
                    expect(relief->Track->Spacing > 1e-3f && relief->Track->Spacing < 1e-2f) << "texel size";
                }
                if (relief) {
                    const auto spacing = relief->Track->Spacing, rms = relief->Track->Rms;
                    for (auto e : fx.R.view<const Instance>()) {
                        if (fx.R.get<const Instance>(e).Entity == mesh_entity) fx.R.patch<Transform>(e, [](Transform &t) { t.S = vec3{3.f}; });
                    }
                    ProcessComponentEvents(fx.R, fx.Viewport);
                    UpdateSurfaceRelief(fx.R, node, mesh_entity, true);
                    const auto *rescaled = fx.R.try_get<const SurfaceRelief>(node);
                    expect(rescaled != nullptr) << "relief lost when the node was resized";
                    if (rescaled) {
                        expect(NumberEq(rescaled->Track->Spacing, spacing)) << "node scale leaked into the track spacing";
                        expect(NumberEq(rescaled->Track->Rms, rms)) << "node scale leaked into the track height";
                    }
                }
#endif
            });
            if (!reloaded.Value) return;
            {
                simdjson::dom::parser p;
                simdjson::dom::element doc;
                expect(p.load(out_path.string()).get(doc) == simdjson::SUCCESS) << "emitted json failed to parse";

                const auto surface0 = OnlyAudioEntry(doc, "acousticSurfaces");
                expect(surface0.has_value()) << "expected one acoustic surface";
                if (!surface0) return;
                expect(AccessorShapeIs(doc, *surface0, "profile", "SCALAR", 4)) << "profile accessor shape";
                uint64_t texture_index = ~0ull;
                expect((*surface0)["normalTexture"]["index"].get_uint64().get(texture_index) == simdjson::SUCCESS && texture_index == 0u) << "normal texture index";
                const auto instanced = NodeAudioIndex(doc, "acousticSurface");
                expect(instanced.has_value()) << "no node instances the acoustic surface";
                if (instanced) expect(*instanced == 0u) << "node surface index";
            }
            expect(SurfacesEq(*reloaded.Value, surface)) << "contact surface diverged";
        };
    }

#ifdef SURFACE_AUDIO
    if (const fs::path normal_tangent = SamplePath("external/glTF-Sample-Assets/Models/NormalTangentTest/glTF/NormalTangentTest.gltf"); fs::exists(normal_tangent)) {
        test("audio_surface_inherits_material_normal_map") = [&] {
            const auto stage_dir = edit_root / "audio-surface-inherit";
            const auto staged_gltf = StageSample(normal_tangent, stage_dir);

            SceneFixture fx;
            const auto load = gltf::LoadGltf(staged_gltf, fx.R, fx.Viewport);
            expect(load.has_value()) << "NormalTangentTest load failed: " << (load ? "" : load.error());
            if (!load) return;

            const auto node = FirstMeshNode(fx.R);
            expect(node != state::Null) << "no mesh instance node in NormalTangentTest";
            if (node == state::Null) return;
            const auto mesh_entity = fx.R.get<const Instance>(node).Entity;

            // The material's normal map is texture 2, which resolves to its own source image.
            const auto inherited = gltf::MeshMaterialNormalMap(fx.R, mesh_entity);
            expect(inherited.has_value()) << "material normal map not resolved";
            const auto expected_image = gltf::TextureImageIndex(fx.R, 2);
            expect(expected_image.has_value()) << "material normal texture has no source image";
            if (inherited && expected_image) expect(inherited->Image == *expected_image) << "wrong material normal map";

            // The surface names no map of its own, so the relief comes from the material's.
            fx.R.emplace_or_replace<ContactSurface>(node, ContactSurface{.Name = "Inherited"});
            UpdateSurfaceRelief(fx.R, node, mesh_entity, true);
            const auto *relief = fx.R.try_get<const SurfaceRelief>(node);
            expect(relief != nullptr) << "no relief derived from the material's normal map";
            // These are the relief's own properties, so they are asserted here, against a real tangent-space normal map.
            if (relief) {
                expect(relief->Track->Spacing > 0.f) << "texel size";
                expect(relief->Track->Rms > 0.f) << "relief has no height";
                expect(relief->Track->Heights.size() == relief->Track->Sum.size() - 1) << "relief track integral";
            }

            const auto out_path = stage_dir / "audio_surface_inherit.gltf";
            const auto save = gltf::SaveGltf(out_path, fx.R, fx.Viewport);
            expect(save.has_value()) << "save failed: " << (save ? "" : save.error());
            if (!save) return;

            simdjson::dom::parser p;
            simdjson::dom::element doc;
            expect(p.load(out_path.string()).get(doc) == simdjson::SUCCESS) << "emitted json failed to parse";
            const auto surface0 = OnlyAudioEntry(doc, "acousticSurfaces");
            expect(surface0.has_value()) << "expected one acoustic surface";
            if (surface0) expect((*surface0)["normalTexture"].error() != simdjson::SUCCESS) << "an inherited normal texture was written back out";
        };
    }
#endif

    // A load that fails validation leaves the scene and every store as they were.
    "failed_import_leaves_scene_untouched"_test = [&] {
        SceneFixture fx;
        const auto before = CountScene(fx);
        const std::pair<bool, std::string_view> cases[]{{false, "armature root node"}, {true, "JOINTS_0 without WEIGHTS_0"}};
        for (const auto &[drop_weights, expected_error] : cases) {
            const auto path = tmp_root / std::format("broken_{}.gltf", drop_weights ? "mesh" : "skin");
            std::ofstream{path} << BrokenSkinnedTriangleGltf(drop_weights);
            const auto load = gltf::LoadGltf(path, fx.R, fx.Viewport);
            expect(!load.has_value()) << "broken document loaded";
            if (!load) expect(load.error().contains(expected_error)) << "unexpected error: " << load.error();
            const auto after = CountScene(fx);
            expect(after == before) << "failed load changed the scene or a store: entities " << before.Entities << " -> " << after.Entities << ", materials " << before.Materials << " -> " << after.Materials;
            std::cerr << std::format("  failed load '{}': entities {} -> {}, mesh handles {} -> {}, textures {} -> {}, materials {} -> {}, vertex bytes {} -> {}\n", path.filename().string(), before.Entities, after.Entities, before.MeshHandles, after.MeshHandles, before.Textures, after.Textures, before.Materials, after.Materials, before.VertexBytes, after.VertexBytes);
        }
    };

    return RunSuites();
}
