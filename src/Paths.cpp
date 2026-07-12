#include "Paths.h"

namespace Paths {
namespace {
std::filesystem::path BaseDir;
std::filesystem::path ResDir;
std::filesystem::path ShadersDir;
std::filesystem::path UserDataDir;
std::filesystem::path ProjectDir;
} // namespace

void Init(std::filesystem::path base, std::filesystem::path user_data) {
    BaseDir = std::move(base);
    ResDir = BaseDir / "res";
    ShadersDir = BaseDir / "shaders";
    UserDataDir = std::move(user_data);
}

const std::filesystem::path &Base() { return BaseDir; }
const std::filesystem::path &Res() { return ResDir; }
const std::filesystem::path &Shaders() { return ShadersDir; }
const std::filesystem::path &UserData() { return UserDataDir; }

const std::filesystem::path &Project() { return ProjectDir; }
void SetProject(std::filesystem::path dir) { ProjectDir = std::move(dir); }
} // namespace Paths
