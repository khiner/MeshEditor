#pragma once

#include "metal/Shader.h"
#include "project/store/Hash.h"

#include <deque>
#include <functional>
#include <unordered_map>

namespace MTL {
class VisibleFunctionTable;
} // namespace MTL
namespace MTL4 {
class BinaryFunction;
} // namespace MTL4

namespace mtl {
// A precompiled pipeline whose compute or fragment function calls programs through a visible function table, in which entry i is program i.
template<typename P> struct Driver {
    // The pipeline with every registered program linked, and its table, whose entries are handles from its pipeline state alone.
    struct Linked {
        P Pipeline;
        Resident<MTL::VisibleFunctionTable> Table;
    };
    // Replaced when programs link, and kept by each command that binds it until the command completes.
    std::shared_ptr<const Linked> Current;
    // Creates the pipeline with the programs linked, for a pipeline the driver recreates since it cannot grow, or empty for one that grows.
    std::function<P(const NS::Array *)> Make{};
};
using ComputeDriver = Driver<ComputePipeline>;
using RenderDriver = Driver<RenderPipeline>;

// Programs compiled to binary functions, and the drivers that call them.
// A program is Metal source defining one [[visible]] function named ProgramFunction.
// Programs compiled together share a library, so a definition that several sources repeat sits under an include guard.
// Identical source compiles once, and each distinct program keeps its index for the life of the registry.
struct ProgramRegistry {
    explicit ProgramRegistry(const Context &, std::optional<MTL::MathMode> = {});
    ~ProgramRegistry();
    ProgramRegistry(const ProgramRegistry &) = delete;
    ProgramRegistry &operator=(const ProgramRegistry &) = delete;

    const Context &Ctx;
    std::optional<MTL::MathMode> MathMode; // The math mode programs compile with, and Metal's default when empty
    NS::SharedPtr<MTL4::Compiler> Compiler;
    std::unordered_map<store::Hash128, uint32_t, store::Hash128Hasher> Indices; // Program index by source hash
    std::vector<NS::SharedPtr<MTL4::BinaryFunction>> Functions; // Binary function by program index
    // Drivers keep their addresses as others register.
    std::deque<ComputeDriver> ComputeDrivers;
    std::deque<RenderDriver> RenderDrivers;
};

// Compiles the sources not yet registered, links them into every driver, and returns each source's program index.
// Returns once every program is linked.
std::vector<uint32_t> AddPrograms(ProgramRegistry &, std::span<const std::string> sources);
// Registers a pipeline whose function reference sets BinaryLinking, and links every program into it.
ComputeDriver &AddDriver(ProgramRegistry &, ComputePipeline);
RenderDriver &AddDriver(ProgramRegistry &, RenderPipeline);
// Registers the pipeline `make` creates with every registered program linked into its fragment stage, which `make` creates again with every program each time programs are added.
// A mesh pipeline links its programs as it compiles, since the driver fails to grow one.
RenderDriver &AddDriver(ProgramRegistry &, std::function<RenderPipeline(const NS::Array *)> make);
// Binds the driver's pipeline state, and its function table at buffer `index` of the stage's argument table.
void Bind(ComputeEncoder, const ComputeDriver &, uint32_t index);
void Bind(RenderEncoder, const RenderDriver &, uint32_t index);
} // namespace mtl
