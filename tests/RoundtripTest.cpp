#include "gltf/GltfScene.h"

#include <boost/ut.hpp>

#include <filesystem>
#include <vector>

namespace {
namespace fs = std::filesystem;

// Collects .gltf files in <root>/<Name>/glTF/ if that subdir exists, otherwise <root>/<Name>/
std::vector<fs::path> CollectGltfSamples(const fs::path &root) {
    std::vector<fs::path> out;
    if (!fs::exists(root)) return out;
    for (const auto &entry : fs::directory_iterator{root}) {
        if (!entry.is_directory()) continue;
        const fs::path canonical = entry.path() / "glTF";
        const fs::path &search = fs::is_directory(canonical) ? canonical : entry.path();
        for (const auto &child : fs::directory_iterator{search}) {
            if (child.is_regular_file() && child.path().extension() == ".gltf") {
                out.emplace_back(child.path());
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
} // namespace

int main() {
    using namespace boost::ut;

    const auto tmp_root = MakeRoundtripDir();

    std::vector<fs::path> samples;
    for (const auto root : SampleRoots) {
        auto found = CollectGltfSamples(root);
        samples.insert(samples.end(), found.begin(), found.end());
    }

    for (const auto &src : samples) {
        test(src.stem().string()) = [&] {
            auto loaded_a = gltf::LoadScene(src);
            if (!loaded_a) {
                // Loader failure on the source; not a roundtrip issue.
                expect(false) << "LoadScene(original) failed: " << loaded_a.error();
                return;
            }
            const auto &a = *loaded_a;

            const auto out_path = tmp_root / (src.stem().string() + ".gltf");
            auto save_result = gltf::SaveScene(a, out_path);
            expect(save_result.has_value()) << "SaveScene failed: " << (save_result ? "" : save_result.error());
            if (!save_result) return;

            auto loaded_b = gltf::LoadScene(out_path);
            expect(loaded_b.has_value()) << "LoadScene(roundtripped) failed: " << (loaded_b ? "" : loaded_b.error());
            if (!loaded_b) return;
            const auto &b = *loaded_b;

            expect(a.Meshes.size() == b.Meshes.size());
            expect(a.Materials.size() == b.Materials.size());
            expect(a.Textures.size() == b.Textures.size());
            expect(a.Images.size() == b.Images.size());
            expect(a.Samplers.size() == b.Samplers.size());
            expect(a.Nodes.size() == b.Nodes.size());
            expect(a.Objects.size() == b.Objects.size());
            expect(a.Skins.size() == b.Skins.size());
            expect(a.Animations.size() == b.Animations.size());
            expect(a.Cameras.size() == b.Cameras.size());
            expect(a.Lights.size() == b.Lights.size());
            expect(a.PhysicsMaterials.size() == b.PhysicsMaterials.size());
            expect(a.CollisionFilters.size() == b.CollisionFilters.size());
            expect(a.PhysicsJointDefs.size() == b.PhysicsJointDefs.size());

            const auto min_size = [](auto &x, auto &y) { return std::min(x.size(), y.size()); };
            for (size_t i = 0, n = min_size(a.Meshes, b.Meshes); i < n; ++i) {
                expect(a.Meshes[i].Name == b.Meshes[i].Name);
            }
            for (size_t i = 0, n = min_size(a.Materials, b.Materials); i < n; ++i) {
                expect(a.Materials[i].Name == b.Materials[i].Name);
            }
            for (size_t i = 0, n = min_size(a.Nodes, b.Nodes); i < n; ++i) {
                expect(a.Nodes[i].Name == b.Nodes[i].Name);
            }
            for (size_t i = 0, n = min_size(a.Animations, b.Animations); i < n; ++i) {
                expect(a.Animations[i].Name == b.Animations[i].Name);
            }
        };
    }
}
