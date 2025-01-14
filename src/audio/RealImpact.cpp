#include "RealImpact.h"
#include "npy.h"
#include "numeric/vec4.h"

#include <glm/gtc/matrix_transform.hpp>

#include <ranges>
#include <regex>

using std::ranges::max_element, std::ranges::iota_view, std::ranges::to;
using std::views::transform;

/**
All files (present for each object):
  - angle.npy (24K)
  - deconvolved_0db.npy (2.3G) // Loaded dynamically
  - distance.npy (24K)
  - listenerXYZ.npy (72K)
  - material_0.mtl (4.0K)
  - material_0.png (1.2M)
  - micID.npy (24K)
  - transformed.obj (3.4M)
  - vertexID.npy (24K)
  - vertexXYZ.npy (72K)
*/

namespace {
// The authors provide `.mtl` and `.png` material files, but they don't provide the material name.
// However, most object names include the material name, and the rest are easy to guess.
// A '*' indicates a guess based on the object name and the material image.
const std::unordered_map<std::string_view, std::string_view> MaterialNameForObject{
    {"CeramicKoiBowl", "Ceramic"},
    {"CeramicBowlFish", "Ceramic"},
    {"Bowl", "Ceramic"}, // *
    {"BowlCeramic", "Ceramic"},
    {"bowl", "Ceramic"}, // *
    {"IronSkillet", "Iron"},
    {"Pan", "Iron"}, // *
    {"Cup", "Glass"}, // *
    {"PurpleScoop", "Plastic"},
    {"WoodPlate", "Wood"},
    {"WoodPlateSquare", "Wood"},
    {"WoodSlab", "Wood"},
    {"WoodChalice", "Wood"},
    {"WoodWineGlass", "Wood"},
    {"WoodMug", "Wood"},
    {"MeasuringCup", "Polycarbonate"}, // *
    {"SmallMeasuringCup", "Polycarbonate"}, // *
    {"PiePan", "Steel"}, // *
    {"IronMortar", "Iron"},
    {"PlasticBowl", "Plastic"},
    {"PlasticBowl", "Plastic"},
    {"PlasticBowl", "Plastic"},
    {"ShellPlate", "Glass"}, // *
    {"stand", "Steel"}, // *
    {"SkullCup", "Glass"}, // *
    {"PlanterCeramic", "Ceramic"},
    {"Pot_Hexagonal", "Ceramic"}, // *
    {"SmallPlanterCeramic", "Ceramic"},
    {"CeramicMug", "Ceramic"},
    {"PitcherCeramic", "Ceramic"},
    {"IronPlate", "Iron"},
    {"WoodBoard", "Wood"},
    {"PlasticBin", "Plastic"},
    {"FlowerPotLargeCeramic", "Ceramic"},
    {"FlowerpotSmallCeramic", "Ceramic"},
    {"CeramicCup", "Ceramic"},
    {"LargeSwanCeramic", "Ceramic"},
    {"SmallSwanCeramic", "Ceramic"},
    {"WoodPad", "Wood"},
    {"WoodVase", "Wood"},
    {"MetalHoledSpoon", "Steel"}, // *
    {"MetalSpatula", "Steel"}, // *
    {"MetalLadle", "Steel"}, // *
    {"MetalSpoon", "Steel"}, // *
    {"MetalSpatula", "Steel"}, // *
    {"GreenGoblet", "Glass"}, // *
    {"GlassGoblet", "Glass"},
    {"Bowl", "Ceramic"}, // *
    {"PlasticScoop", "Plastic"},
    {"Frisbee", "Plastic"},
};
} // namespace

namespace RealImpact {
// Ascend up ancestor directories until we find the RealImpact object name directory.
std::optional<std::string> FindObjectName(const fs::path &start_path) {
    static const std::regex pattern("^\\d+_.*"); // Integer followed by an underscore and any characters
    for (auto path = start_path; path != path.root_path(); path = path.parent_path()) {
        const auto dirname = path.filename().string();
        if (std::regex_search(dirname, pattern)) {
            // Extract the object name part after the underscore.
            const auto underscore_pos = dirname.find('_');
            if (underscore_pos != std::string::npos && underscore_pos + 1 < dirname.length()) {
                return dirname.substr(underscore_pos + 1); // Extract the object name.
            }
        }
    }

    return {};
}

std::optional<std::string_view> FindMaterialName(std::string_view object_name) {
    if (const auto it = MaterialNameForObject.find(object_name); it != MaterialNameForObject.end()) {
        return it->second;
    }
    return {};
}

std::vector<ListenerPoint> LoadListenerPoints(const fs::path &directory) {
    const auto mic_ids = npy::read_npy<long>(directory / "micID.npy");
    const auto angles = npy::read_npy<long>(directory / "angle.npy");
    const auto distances = npy::read_npy<long>(directory / "distance.npy");
    return iota_view{0u, NumListenerPoints} | transform([&](uint i) {
               return ListenerPoint{
                   .Index = i,
                   .MicId = mic_ids.data[i],
                   .DistanceMm = distances.data[i],
                   .AngleDeg = angles.data[i]
               };
           }) |
        to<std::vector>();
}

std::array<vec3, NumImpactVertices> LoadPositions(const fs::path &directory) {
    std::array<vec3, NumImpactVertices> impact_positions;
    // const auto vertex_ids = npy::read_npy<long>(directory / "vertexID.npy").data;
    // for (uint i = 0; i < NumImpactVertices; ++i) VertexIndices[i] = vertex_ids[i * NumListenerPoints];
    const auto vertex_xyzs = npy::read_npy<double>(directory / "vertexXYZ.npy").data;
    for (uint i = 0; i < NumImpactVertices; ++i) {
        const size_t vi = i * 3 * NumListenerPoints;
        impact_positions[i] = {vertex_xyzs[vi], vertex_xyzs[vi + 1], vertex_xyzs[vi + 2]};
    }
    return impact_positions;
}

std::array<std::vector<float>, NumImpactVertices> LoadSamples(const fs::path &directory, long listener_point_index) {
    const auto file = directory / "deconvolved_0db.npy";
    if (!fs::exists(file)) return {};

    std::ifstream stream{std::move(file), std::ifstream::binary};
    const auto header = npy::read_header<float>(stream);

    const size_t frames_per_impact = header.shape[1];
    std::array<std::vector<float>, NumImpactVertices> all_samples;
    float max_sample = 0;
    for (uint i = 0; i < NumImpactVertices; ++i) {
        // All listener points are recorded before the vertex moves.
        // The offset is calculated relative to the current stream read position.
        const size_t advance_frames = (i == 0 ? listener_point_index : (NumListenerPoints - 1)) * frames_per_impact;
        auto frames = npy::read_npy<float>(stream, header, advance_frames, frames_per_impact).data;
        max_sample = std::max(max_sample, std::abs(*max_element(frames, [](auto a, auto b) { return std::abs(a) < std::abs(b); })));
        all_samples[i] = std::move(frames);
    }
    // Normalize audio to [-1, 1]
    for (auto &samples : all_samples) {
        for (float &sample : samples) sample /= max_sample;
    }
    return all_samples;
}

/*
See https://github.com/samuel-clarke/RealImpact/blob/main/preprocess_measurements.py
Here, we reproduce the `get_mic_world_space` function, to avoid redundantly storing positions.
The only difference is we use Y-up instead of Z-up.
    MIC_BAR_LENGTH = 1890 - 70
    def get_mic_world_space(angle, distance, ind):
        mic_z = -(MIC_BAR_LENGTH/2) + ind/14 * MIC_BAR_LENGTH
        mic_x = 230 + distance
        mic_y = -((45/2) + 20.95) * np.ones_like(angle)
        mic_points = np.vstack((mic_x, mic_y, mic_z)).transpose()
        rot = Rotation.from_euler('z', angle, degrees=True)
        pos_meters = rot.apply(mic_points) / 1000.0
        return pos_meters
*/
vec3 ListenerPoint::GetPosition(vec3 world_up, bool mic_center) const {
    const float angle = glm::radians(float(AngleDeg)), dist = float(DistanceMm);
    const vec3 pos{
        // 230 I believe is for the gantry (where the object is placed)
        230 + dist + (mic_center ? MicLengthMm / 2 : 0),
        -(MicBarLengthMm / 2) + (float(MicId) / (NumMics - 1)) * MicBarLengthMm,
        // I beleive these offseta are to accurately reflect the mic positions attached to _one side_ of microphone array bar.
        // You can see the same offsets in https://samuelpclarke.com/realimpact/ and https://www.youtube.com/watch?v=OeZMeze-oIs
        ((45.f / 2.f) + 20.95f),
    };
    return vec3{glm::rotate({1}, angle, world_up) * vec4{pos, 1}} / 1000.f;
}
} // namespace RealImpact
