#include "action/Errors.h"
#include "state/Registry.h"

namespace action {
void Fail(state::Registry &r, std::string message, bool unsupported_extensions) {
    auto &errors = r.Resources.get<Errors>();
    errors.OnlyUnsupportedExtensions = unsupported_extensions && (errors.Messages.empty() || errors.OnlyUnsupportedExtensions);
    errors.Messages.push_back(std::move(message));
}
} // namespace action
