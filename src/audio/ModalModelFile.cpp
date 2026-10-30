#include "ModalModelFile.h"

#include "File.h"
#include "numeric/Serialize.h"
#include "project/Assets.h"

std::expected<std::filesystem::path, std::string> SaveModalModelFile(project::Assets &assets, const ModalModelData &data) {
    std::vector<std::byte> bytes;
    zpp::bits::out archive{bytes};
    if (zpp::bits::failure(archive(data))) return std::unexpected{"Cannot serialize modal model"};
    return assets.Store("modes.modal", bytes);
}

std::expected<ModalModelData, std::string> LoadModalModelFile(const std::filesystem::path &path) {
    const auto bytes = File::Read(path);
    if (!bytes) return std::unexpected{bytes.error()};
    ModalModelData data;
    if (zpp::bits::failure(zpp::bits::in{*bytes}(data))) return std::unexpected{"Invalid modal model data"};
    return data;
}
