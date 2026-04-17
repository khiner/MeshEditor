#include "Paths.h"

namespace Paths {
namespace {
std::filesystem::path BaseDir;
std::filesystem::path ResDir;
std::filesystem::path ShadersDir;
} // namespace

void Init(std::filesystem::path base) {
    BaseDir = std::move(base);
    ResDir = BaseDir / "res";
    ShadersDir = BaseDir / "shaders";
}

const std::filesystem::path &Base() { return BaseDir; }
const std::filesystem::path &Res() { return ResDir; }
const std::filesystem::path &Shaders() { return ShadersDir; }
} // namespace Paths
