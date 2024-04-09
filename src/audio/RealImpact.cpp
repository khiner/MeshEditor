#include "RealImpact.h"

#include <regex>

#include <glm/gtc/matrix_transform.hpp>

#include "helper/npy.h"
#include "numeric/vec4.h"

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

// A '*' indicates a guess based on the object name and the material image.
const std::unordered_map<std::string, std::string> RealImpact::MaterialNameForObjName = {
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

// Ascend up ancestor directories until we find the RealImpact object name directory.
std::optional<std::string> FindObjectNameInAncestors(const fs::path &start_path) {
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

RealImpact::RealImpact(const fs::path &directory) : Directory(directory), ObjPath(Directory / "transformed.obj") {
    if (const auto object_name = FindObjectNameInAncestors(Directory)) {
        if (const auto it = MaterialNameForObjName.find(*object_name); it != MaterialNameForObjName.end()) {
            MaterialName = it->second;
        }
    }
    const auto vertex_ids = npy::read_npy<long>(Directory / "vertexID.npy").data;
    for (uint i = 0; i < NumImpactVertices; ++i) VertexIndices[i] = vertex_ids[i * NumListenerPoints];
    const auto vertex_xyzs = npy::read_npy<double>(Directory / "vertexXYZ.npy").data;
    for (uint i = 0; i < NumImpactVertices; ++i) {
        const size_t vi = i * 3 * NumListenerPoints;
        ImpactPositions[i] = {vertex_xyzs[vi], vertex_xyzs[vi + 1], vertex_xyzs[vi + 2]};
    }
}

std::vector<RealImpactListenerPoint> RealImpact::LoadListenerPoints() const {
    const auto mic_ids = npy::read_npy<long>(Directory / "micID.npy");
    const auto angles = npy::read_npy<long>(Directory / "angle.npy");
    const auto distances = npy::read_npy<long>(Directory / "distance.npy");

    std::vector<RealImpactListenerPoint> listener_points;
    listener_points.reserve(NumListenerPoints);
    for (uint i = 0; i < NumListenerPoints; ++i) {
        listener_points.push_back(RealImpactListenerPoint{
            .Index = i,
            .MicId = mic_ids.data[i],
            .DistanceMm = distances.data[i],
            .AngleDeg = angles.data[i],
        });
    }
    return listener_points;
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
vec3 RealImpactListenerPoint::GetPosition(vec3 world_up, bool mic_center) const {
    const float angle = glm::radians(float(AngleDeg)), dist = float(DistanceMm);
    const vec3 pos{
        // 230 I believe is for the gantry (where the object is placed)
        230 + dist + (mic_center ? RealImpact::MicLengthMm / 2 : 0),
        -(RealImpact::MicBarLengthMm / 2) + (float(MicId) / (RealImpact::NumMics - 1)) * RealImpact::MicBarLengthMm,
        // I beleive these offseta are to accurately reflect the mic positions attached to _one side_ of microphone array bar.
        // You can see the same offsets in https://samuelpclarke.com/realimpact/ and https://www.youtube.com/watch?v=OeZMeze-oIs
        ((45.f / 2.f) + 20.95f),
    };
    return vec3{glm::rotate({1}, angle, world_up) * vec4{pos, 1}} / 1000.f;
}

std::unordered_map<uint, std::vector<float>> RealImpactListenerPoint::LoadImpactSamples(const RealImpact &parent) const {
    std::unordered_map<uint, std::vector<float>> all_samples;
    all_samples.reserve(RealImpact::NumImpactVertices);
    float max_sample = 0;
    for (uint i = 0; i < RealImpact::NumImpactVertices; ++i) {
        const uint vertex_index = parent.VertexIndices[i];
        const size_t offset = i * RealImpact::NumListenerPoints + Index;
        const size_t size = 209549; // todo get from shape
        auto frames = npy::read_npy<float>(parent.Directory / "deconvolved_0db.npy", offset, size).data;

        const float max_vertex_sample = std::abs(*std::max_element(frames.begin(), frames.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }));
        max_sample = std::max(max_sample, max_vertex_sample);
        all_samples.emplace(vertex_index, std::move(frames));
    }
    // Normalize audio to [-1, 1]
    for (auto &[_, samples] : all_samples) {
        for (float &sample : samples) sample /= max_sample;
    }
    return all_samples;
}
