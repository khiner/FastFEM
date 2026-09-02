#include "FiniteCellMetal.h"

#import <Metal/Metal.h>

#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <dispatch/dispatch.h>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#if FASTFEM_HAS_EMBEDDED_FINITE_CELL_METALLIB
#include "FiniteCellMetallib.h"
#endif

namespace {
constexpr const char *Source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct Cell {
    uint Nodes[27];
};

struct AlgebraParams {
    uint Count;
    uint Rows;
    float LeftScale;
    float RightScale;
};

struct TransferParams {
    uint NumP1Nodes;
    uint NumNodes;
    uint Width;
};

struct SparseProductParams {
    uint OutputRows;
    uint InputRows;
    uint Width;
};

struct DenseProductParams {
    uint Rows;
    uint Width;
};

struct LocalActionParams {
    uint NumNodes;
    uint Width;
    uint LocalDofs;
    uint FirstEntry;
    uint EntryCount;
    uint ScratchBase;
};

struct P1Stencil {
    uint Nodes[8];
    float Weights[8];
    uint Count;
};

struct P1Occurrence {
    uint FineNode;
    float Weight;
};

kernel void Clear(
    device float *output [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) output[index] = 0;
}

kernel void ApplyCooperativeWideBatchedPackedCellMatrixAction(
    const device Cell *cells [[buffer(0)]],
    const device float *matrices [[buffer(1)]],
    const device uint *cell_indices [[buffer(2)]],
    const device float *input [[buffer(3)]],
    device float *scratch [[buffer(4)]],
    constant LocalActionParams &params [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint batches = (params.Width + 7) / 8;
    const uint group_count = params.EntryCount * batches;
    if (group >= group_count) return;
    const uint first_column = 8 * (group % batches);
    const uint entry = params.FirstEntry + group / batches;
    const uint cell_index = cell_indices[entry];
    const device Cell &cell = cells[cell_index];
    threadgroup float4 local_input_low[81], local_input_high[81];
    if (lane < params.LocalDofs) {
        const uint node = cell.Nodes[lane / 3];
        const uint dof = 3 * node + lane % 3;
        float4 low = 0, high = 0;
        for (uint offset = 0; offset < 4 && first_column + offset < params.Width; ++offset)
            low[offset] = input[(first_column + offset) * 3 * params.NumNodes + dof];
        for (uint offset = 0; offset < 4 && first_column + 4 + offset < params.Width; ++offset)
            high[offset] = input[(first_column + 4 + offset) * 3 * params.NumNodes + dof];
        local_input_low[lane] = low;
        local_input_high[lane] = high;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane >= params.LocalDofs) return;
    const uint packed_dofs = params.LocalDofs * (params.LocalDofs + 1) / 2;
    const uint matrix_offset = cell_index * packed_dofs;
    float4 low = 0, high = 0;
    for (uint local = 0; local < params.LocalDofs; ++local) {
        const uint row = max(lane, local), column = min(lane, local);
        const float coefficient = matrices[matrix_offset + row * (row + 1) / 2 + column];
        low += coefficient * local_input_low[local];
        high += coefficient * local_input_high[local];
    }
    for (uint offset = 0; offset < 4 && first_column + offset < params.Width; ++offset)
        scratch[((entry - params.ScratchBase) * params.Width + first_column + offset) * params.LocalDofs + lane] = low[offset];
    for (uint offset = 0; offset < 4 && first_column + 4 + offset < params.Width; ++offset)
        scratch[((entry - params.ScratchBase) * params.Width + first_column + 4 + offset) * params.LocalDofs + lane] = high[offset];
}

kernel void ScatterCellMatrixAction(
    const device Cell *cells [[buffer(0)]],
    const device uint *cell_indices [[buffer(1)]],
    const device float *scratch [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant LocalActionParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    const uint count = params.EntryCount * params.Width * params.LocalDofs;
    if (index >= count) return;
    const uint local_dof = index % params.LocalDofs;
    const uint column = (index / params.LocalDofs) % params.Width;
    const uint entry = params.FirstEntry + index / (params.LocalDofs * params.Width);
    const device Cell &cell = cells[cell_indices[entry]];
    const uint node = cell.Nodes[local_dof / 3];
    output[column * 3 * params.NumNodes + 3 * node + local_dof % 3] +=
        scratch[((entry - params.ScratchBase) * params.Width + column) * params.LocalDofs + local_dof];
}

kernel void ApplyJacobiSmoother(
    const device float *diagonal [[buffer(0)]],
    const device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant AlgebraParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.Count) return;
    output[index] = params.RightScale * input[index] / diagonal[index % params.Rows];
}

kernel void ApplyJacobiUpdate(
    const device float *diagonal [[buffer(0)]],
    const device float *right_hand_side [[buffer(1)]],
    const device float *action [[buffer(2)]],
    const device float *current [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant AlgebraParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.Count) return;
    output[index] = current[index] + params.RightScale *
        (right_hand_side[index] - action[index]) / diagonal[index % params.Rows];
}

kernel void ApplySparseProduct(
    const device uint *offsets [[buffer(0)]],
    const device uint *columns [[buffer(1)]],
    const device float *values [[buffer(2)]],
    const device float *input [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant SparseProductParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.OutputRows * params.Width) return;
    const uint block = index / params.OutputRows;
    const uint row = index % params.OutputRows;
    float value = 0;
    for (uint entry = offsets[row]; entry < offsets[row + 1]; ++entry)
        value += values[entry] * input[block * params.InputRows + columns[entry]];
    output[index] = value;
}

kernel void ApplyDenseProduct(
    const device float *matrix [[buffer(0)]],
    const device float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant DenseProductParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.Rows * params.Width) return;
    const uint block = index / params.Rows;
    const uint row = index % params.Rows;
    float value = 0;
    for (uint column = 0; column < params.Rows; ++column)
        value += matrix[column * params.Rows + row] * input[block * params.Rows + column];
    output[index] = value;
}

kernel void LinearCombination(
    const device float *left [[buffer(0)]],
    const device float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant AlgebraParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= params.Count) return;
    output[index] = params.LeftScale * left[index] + params.RightScale * right[index];
}

kernel void RestrictP1(
    const device float *fine [[buffer(0)]],
    const device uint *offsets [[buffer(1)]],
    const device P1Occurrence *occurrences [[buffer(2)]],
    device float *coarse [[buffer(3)]],
    constant TransferParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    const uint coarse_dofs = 3 * params.NumP1Nodes;
    if (index >= coarse_dofs * params.Width) return;
    const uint column = index / coarse_dofs;
    const uint row = index % coarse_dofs;
    const uint node = row / 3;
    const uint component = row % 3;
    float value = 0;
    for (uint occurrence = offsets[node]; occurrence < offsets[node + 1]; ++occurrence)
        value += occurrences[occurrence].Weight *
            fine[column * 3 * params.NumNodes + 3 * occurrences[occurrence].FineNode + component];
    coarse[index] = value;
}

kernel void ProlongP1(
    const device float *coarse [[buffer(0)]],
    const device P1Stencil *stencils [[buffer(1)]],
    device float *fine [[buffer(2)]],
    constant TransferParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    const uint fine_dofs = 3 * params.NumNodes;
    if (index >= fine_dofs * params.Width) return;
    const uint column = index / fine_dofs;
    const uint row = index % fine_dofs;
    const uint node = row / 3;
    const uint component = row % 3;
    const device P1Stencil &stencil = stencils[node];
    float value = 0;
    for (uint entry = 0; entry < stencil.Count; ++entry)
        value += stencil.Weights[entry] * coarse[column * 3 * params.NumP1Nodes + 3 * stencil.Nodes[entry] + component];
    fine[index] = value;
}

)metal";

struct MetalCell {
    std::array<uint32_t, 27> Nodes;
};

struct AlgebraParams {
    uint32_t Count, Rows;
    float LeftScale, RightScale;
};

struct TransferParams {
    uint32_t NumP1Nodes, NumNodes, Width;
};

struct SparseProductParams {
    uint32_t OutputRows, InputRows, Width;
};

struct DenseProductParams {
    uint32_t Rows, Width;
};

struct LocalActionParams {
    uint32_t NumNodes, Width, LocalDofs, FirstEntry, EntryCount, ScratchBase;
};

struct MetalP1Stencil {
    std::array<uint32_t, 8> Nodes;
    std::array<float, 8> Weights;
    uint32_t Count;
};

struct P1Occurrence {
    uint32_t FineNode;
    float Weight;
};

static_assert(sizeof(MetalCell) == 108);
static_assert(sizeof(AlgebraParams) == 16);
static_assert(sizeof(LocalActionParams) == 24);
static_assert(sizeof(TransferParams) == 12);
static_assert(sizeof(SparseProductParams) == 12);
static_assert(sizeof(DenseProductParams) == 8);
static_assert(sizeof(MetalP1Stencil) == 68);
static_assert(sizeof(P1Occurrence) == 8);

std::runtime_error MetalError(NSString *context, NSError *error = nil) {
    std::string message{context.UTF8String};
    if (error) message += ": " + std::string{error.localizedDescription.UTF8String};
    return std::runtime_error{message};
}

// Dispatches encoded on one compute pass execute in order, so an operation needs one encoder.
template<typename Params>
void Bind(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers, const Params &params
) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
    [encoder setBytes:&params length:sizeof(params) atIndex:index];
}

template<typename Params>
void EncodeLinear(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers, const Params &params, size_t threads
) {
    Bind(encoder, pipeline, buffers, params);
    const NSUInteger group = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

// One threadgroup per (cell, eight-column batch), with one lane per local degree of freedom.
template<typename Params>
void EncodeCooperative(
    id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
    std::initializer_list<id<MTLBuffer>> buffers, const Params &params,
    size_t groups, uint32_t lanes, NSString *overflow
) {
    const NSUInteger required = (lanes + 31) / 32 * 32;
    if (pipeline.maxTotalThreadsPerThreadgroup < required) throw MetalError(overflow);
    Bind(encoder, pipeline, buffers, params);
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(required, 1, 1)];
}

} // namespace

struct modal::FiniteCellMetal::Block::Implementation {
    id<MTLBuffer> Buffer;
};

struct modal::FiniteCellMetal::Implementation {
    struct MultigridLevel {
        id<MTLBuffer> Offsets, Columns, Values, Diagonal;
        id<MTLBuffer> ProlongationOffsets, ProlongationColumns, ProlongationValues;
        id<MTLBuffer> RestrictionOffsets, RestrictionColumns, RestrictionValues;
        id<MTLBuffer> Inverse;
        uint32_t Rows{}, CoarseRows{};
        float Maximum{};
    };

    id<MTLDevice> Device;
    id<MTLCommandQueue> Queue;
    id<MTLComputePipelineState> ClearPipeline, SmootherPipeline, JacobiUpdatePipeline;
    id<MTLComputePipelineState> SparseProductPipeline, DenseProductPipeline;
    id<MTLComputePipelineState> CombinationPipeline, RestrictPipeline, ProlongPipeline;
    id<MTLComputePipelineState> CooperativeWideBatchedPackedCellMatrixPipeline, CellMatrixScatterPipeline;
    id<MTLBuffer> Cells, CellIndices, LocalizedCells, P1Stencils, P1Offsets, P1Occurrences;
    mutable std::vector<MultigridLevel> Multigrid;
    mutable id<MTLBuffer> PackedPatchInverses, PackedElementMatrices, Scratch;
    mutable size_t ScratchCapacity{};
    // Upload and download delimit one command buffer, amortizing submission across a preconditioner application.
    mutable id<MTLCommandBuffer> PendingCommand;
    std::array<uint32_t, 9> ColorOffsets{};
    std::array<std::array<uint32_t, 9>, 8> LocalizedOffsets{};
    std::vector<uint32_t> CellOrder;
    uint32_t NumNodes{}, NumP1Nodes{}, NodesPerCell{};

    id<MTLCommandBuffer> Command() const {
        if (!PendingCommand) {
            PendingCommand = [Queue commandBuffer];
            if (!PendingCommand) throw MetalError(@"Creating finite-cell Metal command buffer failed");
        }
        return PendingCommand;
    }

    void ReserveScratch(uint32_t width) const {
        const size_t bytes = CellOrder.size() * width * 3 * NodesPerCell * sizeof(float);
        if (bytes <= ScratchCapacity) return;
        Scratch = [Device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!Scratch) throw MetalError(@"Allocating finite-cell Metal cell scratch failed");
        ScratchCapacity = bytes;
    }
};

modal::FiniteCellMetal::SharedFloats::SharedFloats() = default;

modal::FiniteCellMetal::SharedFloats::SharedFloats(size_t size) : Size{size} {
    const size_t bytes = size * sizeof(float), page = size_t(getpagesize());
    CapacityBytes = (bytes + page - 1) / page * page;
    if (CapacityBytes && posix_memalign(reinterpret_cast<void **>(&Data), page, CapacityBytes)) throw std::bad_alloc{};
}

modal::FiniteCellMetal::SharedFloats::SharedFloats(SharedFloats &&other) noexcept : Data{other.Data}, Size{other.Size}, CapacityBytes{other.CapacityBytes} {
    other.Data = nullptr;
    other.Size = other.CapacityBytes = 0;
}

modal::FiniteCellMetal::SharedFloats &modal::FiniteCellMetal::SharedFloats::operator=(SharedFloats &&other) noexcept {
    if (this == &other) return *this;
    std::free(Data);
    Data = other.Data;
    Size = other.Size;
    CapacityBytes = other.CapacityBytes;
    other.Data = nullptr;
    other.Size = other.CapacityBytes = 0;
    return *this;
}

modal::FiniteCellMetal::SharedFloats::~SharedFloats() { std::free(Data); }
std::span<float> modal::FiniteCellMetal::SharedFloats::Values() const { return {Data, Size}; }
float *modal::FiniteCellMetal::SharedFloats::Release() {
    auto *result = Data;
    Data = nullptr;
    Size = CapacityBytes = 0;
    return result;
}

modal::FiniteCellMetal::Block::Block() : Impl(std::make_unique<Implementation>()) {}
modal::FiniteCellMetal::Block::Block(Block &&) noexcept = default;
modal::FiniteCellMetal::Block &modal::FiniteCellMetal::Block::operator=(Block &&) noexcept = default;
modal::FiniteCellMetal::Block::~Block() = default;

modal::FiniteCellMetal::FiniteCellMetal(const FiniteCellOperator &operation) : Impl(std::make_unique<Implementation>()) {
    @autoreleasepool {
        Impl->Device = MTLCreateSystemDefaultDevice();
        if (!Impl->Device) throw MetalError(@"No Metal device");
        Impl->Queue = [Impl->Device newCommandQueue];
        if (!Impl->Queue) throw MetalError(@"Creating finite-cell Metal command queue failed");
        NSError *error = nil;
        [[maybe_unused]] bool precompiled = false;
#if FASTFEM_HAS_EMBEDDED_FINITE_CELL_METALLIB
        dispatch_data_t data = dispatch_data_create(FastFEMFiniteCellMetallib, FastFEMFiniteCellMetallib_len, nil, ^{});
        id<MTLLibrary> library = [Impl->Device newLibraryWithData:data error:&error];
        precompiled = library != nil;
#else
        id<MTLLibrary> library = nil;
#endif
        if (!library) {
            error = nil;
            library = [Impl->Device newLibraryWithSource:[NSString stringWithUTF8String:Source] options:nil error:&error];
        }
        if (!library) throw MetalError(@"Compiling finite-cell Metal kernels failed", error);
        id<MTLBinaryArchive> archive = nil;
#ifdef FASTFEM_FINITE_CELL_BINARY_ARCHIVE_PATH
        if (precompiled) {
            MTLBinaryArchiveDescriptor *descriptor = [MTLBinaryArchiveDescriptor new];
            descriptor.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:FASTFEM_FINITE_CELL_BINARY_ARCHIVE_PATH]];
            error = nil;
            archive = [Impl->Device newBinaryArchiveWithDescriptor:descriptor error:&error];
        }
#endif
        const auto pipeline = [&](NSString *name) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            id<MTLComputePipelineState> result;
            if (archive) {
                MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
                descriptor.computeFunction = function;
                descriptor.binaryArchives = @[ archive ];
                result = [Impl->Device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
            } else result = [Impl->Device newComputePipelineStateWithFunction:function error:&error];
            if (!result) throw MetalError(@"Creating finite-cell Metal pipeline failed", error);
            return result;
        };
        Impl->ClearPipeline = pipeline(@"Clear");
        Impl->SmootherPipeline = pipeline(@"ApplyJacobiSmoother");
        Impl->JacobiUpdatePipeline = pipeline(@"ApplyJacobiUpdate");
        Impl->SparseProductPipeline = pipeline(@"ApplySparseProduct");
        Impl->DenseProductPipeline = pipeline(@"ApplyDenseProduct");
        Impl->CombinationPipeline = pipeline(@"LinearCombination");
        Impl->CooperativeWideBatchedPackedCellMatrixPipeline = pipeline(@"ApplyCooperativeWideBatchedPackedCellMatrixAction");
        Impl->CellMatrixScatterPipeline = pipeline(@"ScatterCellMatrixAction");
        Impl->RestrictPipeline = pipeline(@"RestrictP1");
        Impl->ProlongPipeline = pipeline(@"ProlongP1");

        std::vector<MetalCell> cells;
        cells.reserve(operation.Cells.size());
        Impl->CellOrder.reserve(operation.Cells.size());
        for (uint32_t color = 0; color < 8; ++color) {
            Impl->ColorOffsets[color] = uint32_t(cells.size());
            for (uint32_t cell_index = 0; cell_index < operation.Cells.size(); ++cell_index) {
                if (operation.Cells[cell_index].Color != color) continue;
                cells.push_back({operation.Cells[cell_index].Nodes});
                Impl->CellOrder.push_back(cell_index);
            }
        }
        Impl->ColorOffsets[8] = uint32_t(cells.size());
        std::vector<uint32_t> localized_cells;
        std::vector<uint8_t> source_nodes(operation.Nodes.size());
        for (uint8_t source_color = 0; source_color < 8; ++source_color) {
            std::fill(source_nodes.begin(), source_nodes.end(), uint8_t{});
            for (uint32_t source = Impl->ColorOffsets[source_color]; source < Impl->ColorOffsets[source_color + 1]; ++source)
                for (uint32_t local = 0; local < operation.NodesPerCell; ++local)
                    source_nodes[cells[source].Nodes[local]] = 1;
            for (uint8_t output_color = 0; output_color < 8; ++output_color) {
                Impl->LocalizedOffsets[source_color][output_color] = uint32_t(localized_cells.size());
                for (uint32_t target = Impl->ColorOffsets[output_color]; target < Impl->ColorOffsets[output_color + 1]; ++target) {
                    bool affected{};
                    for (uint32_t local = 0; local < operation.NodesPerCell; ++local)
                        affected |= source_nodes[cells[target].Nodes[local]];
                    if (affected) localized_cells.push_back(target);
                }
            }
            Impl->LocalizedOffsets[source_color][8] = uint32_t(localized_cells.size());
        }
        std::vector<MetalP1Stencil> stencils(operation.P1Stencils.size());
        std::vector<uint32_t> p1_offsets(operation.NumP1Nodes + 1);
        for (uint32_t fine = 0; fine < operation.P1Stencils.size(); ++fine) {
            const auto &source = operation.P1Stencils[fine];
            stencils[fine].Nodes = source.Nodes;
            stencils[fine].Count = source.Count;
            for (uint32_t entry = 0; entry < 8; ++entry) stencils[fine].Weights[entry] = float(source.Weights[entry]);
            for (uint32_t entry = 0; entry < source.Count; ++entry) ++p1_offsets[source.Nodes[entry] + 1];
        }
        for (uint32_t node = 0; node < operation.NumP1Nodes; ++node) p1_offsets[node + 1] += p1_offsets[node];
        std::vector<uint32_t> p1_cursor = p1_offsets;
        std::vector<P1Occurrence> p1_occurrences(p1_offsets.back());
        for (uint32_t fine = 0; fine < operation.P1Stencils.size(); ++fine) {
            const auto &source = operation.P1Stencils[fine];
            for (uint32_t entry = 0; entry < source.Count; ++entry)
                p1_occurrences[p1_cursor[source.Nodes[entry]]++] = {fine, float(source.Weights[entry])};
        }
        const auto buffer = [&](const void *bytes, size_t size) {
            id<MTLBuffer> result = [Impl->Device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared];
            if (!result) throw MetalError(@"Allocating finite-cell Metal buffer failed");
            return result;
        };
        Impl->Cells = buffer(cells.data(), cells.size() * sizeof(MetalCell));
        std::vector<uint32_t> cell_indices(cells.size());
        std::iota(cell_indices.begin(), cell_indices.end(), 0);
        Impl->CellIndices = buffer(cell_indices.data(), cell_indices.size() * sizeof(uint32_t));
        Impl->LocalizedCells = buffer(localized_cells.data(), localized_cells.size() * sizeof(uint32_t));
        Impl->P1Stencils = buffer(stencils.data(), stencils.size() * sizeof(MetalP1Stencil));
        Impl->P1Offsets = buffer(p1_offsets.data(), p1_offsets.size() * sizeof(uint32_t));
        Impl->P1Occurrences = buffer(p1_occurrences.data(), p1_occurrences.size() * sizeof(P1Occurrence));
        Impl->NumNodes = uint32_t(operation.Nodes.size());
        Impl->NumP1Nodes = operation.NumP1Nodes;
        Impl->NodesPerCell = operation.NodesPerCell;
    }
}

modal::FiniteCellMetal::~FiniteCellMetal() = default;

namespace {
modal::FiniteCellMetal::Block NewBlock(
    id<MTLDevice> device, uint32_t rows, uint32_t width, MTLResourceOptions options
) {
    modal::FiniteCellMetal::Block result;
    result.Rows = rows;
    result.Width = width;
    if (width == 0) return result;
    result.Impl->Buffer = [device newBufferWithLength:size_t(rows) * width * sizeof(float) options:options];
    if (!result.Impl->Buffer) throw MetalError(@"Allocating finite-cell Metal block failed");
    return result;
}
} // namespace

modal::FiniteCellMetal::Block modal::FiniteCellMetal::CreateBlock(uint32_t width) const {
    return NewBlock(Impl->Device, 3 * Impl->NumNodes, width, MTLResourceStorageModePrivate);
}

modal::FiniteCellMetal::Block modal::FiniteCellMetal::CreateSharedBlock(uint32_t width) const {
    return NewBlock(Impl->Device, 3 * Impl->NumNodes, width, MTLResourceStorageModeShared);
}

modal::FiniteCellMetal::Block modal::FiniteCellMetal::CreateP1Block(uint32_t width) const {
    return NewBlock(Impl->Device, 3 * Impl->NumP1Nodes, width, MTLResourceStorageModePrivate);
}

void modal::FiniteCellMetal::Synchronize() const {
    if (!Impl->PendingCommand) return;
    id<MTLCommandBuffer> command = Impl->PendingCommand;
    Impl->PendingCommand = nil;
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
        throw MetalError(@"Finite-cell Metal command failed", command.error);
}

namespace {
Eigen::Map<Eigen::VectorXf> SharedValues(
    const modal::FiniteCellMetal::Implementation &impl, const modal::FiniteCellMetal::Block &block, const char *what
) {
    if (!block.Impl || !block.Impl->Buffer || !block.Impl->Buffer.contents ||
        block.Impl->Buffer.device != impl.Device)
        throw std::invalid_argument(std::string{"Finite-cell Metal "} + what + " block must be allocated by the same device.");
    return {static_cast<float *>(block.Impl->Buffer.contents), Eigen::Index(block.Rows) * block.Width};
}
} // namespace

// Host I/O blocks use shared FP32 storage, converting directly to and from the caller's FP64 panel.
void modal::FiniteCellMetal::Upload(Block &block, const double *input) const {
    Synchronize();
    auto values = SharedValues(*Impl, block, "upload");
    values = Eigen::Map<const Eigen::VectorXd>{input, values.size()}.cast<float>();
}

void modal::FiniteCellMetal::Download(const Block &block, double *output) const {
    Synchronize();
    const auto values = SharedValues(*Impl, block, "download");
    Eigen::Map<Eigen::VectorXd>{output, values.size()} = values.cast<double>();
}

namespace {
void RequireMatching(
    const modal::FiniteCellMetal::Implementation &impl,
    const modal::FiniteCellMetal::Block &input, const modal::FiniteCellMetal::Block &output
) {
    if (!input.Impl || !output.Impl || input.Rows != output.Rows || input.Width != output.Width ||
        (input.Width && (!input.Impl->Buffer || !output.Impl->Buffer || input.Impl->Buffer.device != impl.Device || output.Impl->Buffer.device != impl.Device)))
        throw std::invalid_argument("Finite-cell Metal blocks must be allocated and have matching dimensions.");
    if (input.Width && input.Impl->Buffer == output.Impl->Buffer)
        throw std::invalid_argument("Finite-cell Metal actions require distinct input and output blocks.");
}

void RequireFine(
    const modal::FiniteCellMetal::Implementation &impl,
    const modal::FiniteCellMetal::Block &input, const modal::FiniteCellMetal::Block &output
) {
    RequireMatching(impl, input, output);
    if (input.Rows != 3 * impl.NumNodes) throw std::invalid_argument("Finite-cell Metal actions require fine-grid blocks.");
}

void RequireTransfer(
    const modal::FiniteCellMetal::Implementation &impl,
    const modal::FiniteCellMetal::Block &fine, const modal::FiniteCellMetal::Block &coarse, const char *what
) {
    if (!fine.Impl || !coarse.Impl || fine.Rows != 3 * impl.NumNodes || coarse.Rows != 3 * impl.NumP1Nodes ||
        fine.Width != coarse.Width ||
        (fine.Width && (!fine.Impl->Buffer || !coarse.Impl->Buffer || fine.Impl->Buffer.device != impl.Device || coarse.Impl->Buffer.device != impl.Device)))
        throw std::invalid_argument(std::string{"Finite-cell Metal "} + what);
}

using RowSparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using GridPoint = std::array<double, 3>;

struct CpuMultigridLevel {
    RowSparse Operator, Prolongation, Restriction;
    std::vector<GridPoint> Nodes;
    Eigen::MatrixXf Inverse;
    double Maximum{};
};

double JacobiMaximum(const RowSparse &matrix) {
    const Eigen::VectorXd diagonal = matrix.diagonal();
    Eigen::VectorXd vector(matrix.rows()), action(matrix.rows());
    for (Eigen::Index row = 0; row < vector.size(); ++row) vector[row] = std::sin(0.37 * double(row + 1));
    vector /= std::sqrt(vector.dot(diagonal.cwiseProduct(vector)));
    for (uint32_t iteration = 0; iteration < 16; ++iteration) {
        action = matrix * vector;
        vector = action.cwiseQuotient(diagonal);
        vector /= std::sqrt(vector.dot(diagonal.cwiseProduct(vector)));
    }
    action = matrix * vector;
    return 1.05 * vector.dot(action) / vector.dot(diagonal.cwiseProduct(vector));
}

std::vector<GridPoint> P1Nodes(const modal::FiniteCellOperator &operation) {
    std::vector<GridPoint> result(operation.NumP1Nodes);
    std::vector<uint8_t> assigned(operation.NumP1Nodes);
    for (uint32_t fine = 0; fine < operation.P1Stencils.size(); ++fine) {
        const auto &stencil = operation.P1Stencils[fine];
        if (stencil.Count != 1 || stencil.Weights[0] != 1) continue;
        const uint32_t coarse = stencil.Nodes[0];
        result[coarse] = {operation.Nodes[fine].x, operation.Nodes[fine].y, operation.Nodes[fine].z};
        assigned[coarse] = true;
    }
    if (std::ranges::find(assigned, uint8_t{}) != assigned.end())
        throw std::runtime_error("Finite-cell P1 hierarchy could not locate every coarse node.");
    return result;
}

std::pair<RowSparse, std::vector<GridPoint>> Coarsen(const std::vector<GridPoint> &fine_nodes) {
    std::array<std::vector<double>, 3> axes;
    for (const auto &node : fine_nodes)
        for (uint32_t axis = 0; axis < 3; ++axis) axes[axis].push_back(node[axis]);
    for (auto &axis : axes) {
        std::ranges::sort(axis);
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    }
    std::array<std::vector<double>, 3> coarse_axes;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        for (uint32_t index = 0; index < axes[axis].size(); index += 2) coarse_axes[axis].push_back(axes[axis][index]);
        if (coarse_axes[axis].back() != axes[axis].back()) coarse_axes[axis].push_back(axes[axis].back());
    }

    std::map<GridPoint, uint32_t> coarse_map;
    std::vector<GridPoint> coarse_nodes;
    std::vector<Eigen::Triplet<double>> scalar_triplets;
    for (uint32_t fine = 0; fine < fine_nodes.size(); ++fine) {
        std::array<std::array<double, 2>, 3> coordinates{}, weights{};
        std::array<uint32_t, 3> counts{};
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const auto upper = std::ranges::lower_bound(coarse_axes[axis], fine_nodes[fine][axis]);
            if (upper == coarse_axes[axis].end() || *upper == fine_nodes[fine][axis] || upper == coarse_axes[axis].begin()) {
                coordinates[axis][0] = upper == coarse_axes[axis].end() ? coarse_axes[axis].back() : *upper;
                weights[axis][0] = 1;
                counts[axis] = 1;
            } else {
                coordinates[axis] = {*(upper - 1), *upper};
                const double fraction = (fine_nodes[fine][axis] - coordinates[axis][0]) /
                    (coordinates[axis][1] - coordinates[axis][0]);
                weights[axis] = {1 - fraction, fraction};
                counts[axis] = 2;
            }
        }
        for (uint32_t z = 0; z < counts[2]; ++z)
            for (uint32_t y = 0; y < counts[1]; ++y)
                for (uint32_t x = 0; x < counts[0]; ++x) {
                    const double weight = weights[0][x] * weights[1][y] * weights[2][z];
                    if (weight == 0) continue;
                    const GridPoint point{coordinates[0][x], coordinates[1][y], coordinates[2][z]};
                    const auto [entry, inserted] = coarse_map.try_emplace(point, uint32_t(coarse_nodes.size()));
                    if (inserted) coarse_nodes.push_back(point);
                    scalar_triplets.emplace_back(fine, entry->second, weight);
                }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * scalar_triplets.size());
    for (const auto &entry : scalar_triplets)
        for (uint32_t component = 0; component < 3; ++component)
            triplets.emplace_back(3 * entry.row() + component, 3 * entry.col() + component, entry.value());
    RowSparse prolongation(3 * fine_nodes.size(), 3 * coarse_nodes.size());
    prolongation.setFromTriplets(triplets.begin(), triplets.end());
    prolongation.makeCompressed();
    return {std::move(prolongation), std::move(coarse_nodes)};
}

std::vector<CpuMultigridLevel> BuildP1Hierarchy(
    const modal::FiniteCellOperator &operation, double alpha,
    const modal::FiniteCellOperator::AssembledLower &prepared
) {
    const RowSparse stiffness = prepared.Stiffness.selfadjointView<Eigen::Lower>();
    const RowSparse mass = prepared.Mass.selfadjointView<Eigen::Lower>();
    std::vector<CpuMultigridLevel> result{{.Operator = stiffness + alpha * mass, .Nodes = P1Nodes(operation)}};
    while (result.back().Operator.rows() > 96 && result.size() < 8) {
        auto [prolongation, coarse_nodes] = Coarsen(result.back().Nodes);
        if (prolongation.cols() >= prolongation.rows()) break;
        result.back().Prolongation = std::move(prolongation);
        result.back().Restriction = result.back().Prolongation.transpose();
        RowSparse coarse = result.back().Restriction * result.back().Operator * result.back().Prolongation;
        coarse.makeCompressed();
        result.push_back({.Operator = std::move(coarse), .Nodes = std::move(coarse_nodes)});
    }
    for (auto &level : result) level.Maximum = JacobiMaximum(level.Operator);
    const Eigen::MatrixXd coarsest = Eigen::MatrixXd{result.back().Operator};
    const Eigen::LLT<Eigen::MatrixXd> factor{coarsest};
    if (factor.info() != Eigen::Success)
        throw std::runtime_error("Finite-cell P1 multigrid coarse operator is not positive definite.");
    result.back().Inverse = factor.solve(Eigen::MatrixXd::Identity(coarsest.rows(), coarsest.cols())).cast<float>();
    return result;
}

} // namespace

struct modal::FiniteCellMetal::P1Multigrid::Implementation {
    std::vector<CpuMultigridLevel> Levels;
    uint32_t Dofs{}, P1Nodes{};
};

modal::FiniteCellMetal::P1Multigrid::P1Multigrid() : Impl(std::make_unique<Implementation>()) {}
modal::FiniteCellMetal::P1Multigrid::P1Multigrid(P1Multigrid &&) noexcept = default;
modal::FiniteCellMetal::P1Multigrid &modal::FiniteCellMetal::P1Multigrid::operator=(P1Multigrid &&) noexcept = default;
modal::FiniteCellMetal::P1Multigrid::~P1Multigrid() = default;

modal::FiniteCellMetal::P1Multigrid modal::FiniteCellMetal::PrepareP1Multigrid(
    const FiniteCellOperator &operation, double alpha, const FiniteCellOperator::AssembledLower &assembly
) {
    P1Multigrid result;
    result.Impl->Levels = BuildP1Hierarchy(operation, alpha, assembly);
    result.Impl->Dofs = operation.Dofs();
    result.Impl->P1Nodes = operation.NumP1Nodes;
    return result;
}

void modal::FiniteCellMetal::ConfigureP1Multigrid(P1Multigrid &&prepared) const {
    if (!prepared.Impl || prepared.Impl->Dofs != 3 * Impl->NumNodes || prepared.Impl->P1Nodes != Impl->NumP1Nodes)
        throw std::invalid_argument("Finite-cell Metal P1 hierarchy must match the configured fine operator.");
    const auto &hierarchy = prepared.Impl->Levels;
    Synchronize();
    Impl->Multigrid.clear();
    Impl->Multigrid.resize(hierarchy.size());
    @autoreleasepool {
        const auto buffer = [&](const void *bytes, size_t size) {
            id<MTLBuffer> result = [Impl->Device newBufferWithBytes:bytes length:size options:MTLResourceStorageModeShared];
            if (!result) throw MetalError(@"Allocating finite-cell Metal P1 hierarchy failed");
            return result;
        };
        const auto sparse = [&](const RowSparse &matrix, id<MTLBuffer> __strong &offsets,
                                id<MTLBuffer> __strong &columns, id<MTLBuffer> __strong &values) {
            std::vector<uint32_t> row_offsets(matrix.rows() + 1), column_indices(matrix.nonZeros());
            std::vector<float> entries(matrix.nonZeros());
            for (int row = 0; row <= matrix.rows(); ++row) row_offsets[row] = uint32_t(matrix.outerIndexPtr()[row]);
            for (int entry = 0; entry < matrix.nonZeros(); ++entry) {
                column_indices[entry] = uint32_t(matrix.innerIndexPtr()[entry]);
                entries[entry] = float(matrix.valuePtr()[entry]);
            }
            offsets = buffer(row_offsets.data(), row_offsets.size() * sizeof(uint32_t));
            columns = buffer(column_indices.data(), column_indices.size() * sizeof(uint32_t));
            values = buffer(entries.data(), entries.size() * sizeof(float));
        };
        for (uint32_t index = 0; index < hierarchy.size(); ++index) {
            const auto &source = hierarchy[index];
            auto &level = Impl->Multigrid[index];
            level.Rows = uint32_t(source.Operator.rows());
            level.Maximum = float(source.Maximum);
            sparse(source.Operator, level.Offsets, level.Columns, level.Values);
            const Eigen::VectorXf diagonal = source.Operator.diagonal().cast<float>();
            level.Diagonal = buffer(diagonal.data(), diagonal.size() * sizeof(float));
            if (index + 1 < hierarchy.size()) {
                level.CoarseRows = uint32_t(source.Prolongation.cols());
                sparse(
                    source.Prolongation, level.ProlongationOffsets,
                    level.ProlongationColumns, level.ProlongationValues
                );
                sparse(
                    source.Restriction, level.RestrictionOffsets,
                    level.RestrictionColumns, level.RestrictionValues
                );
            } else {
                level.Inverse = buffer(source.Inverse.data(), size_t(source.Inverse.size()) * sizeof(float));
            }
        }
    }
}

void modal::FiniteCellMetal::ApplyP1Multigrid(
    const Block &input, Block &output
) const {
    RequireMatching(*Impl, input, output);
    if (Impl->Multigrid.empty() || input.Rows != Impl->Multigrid.front().Rows)
        throw std::runtime_error("Finite-cell Metal P1 hierarchy is not configured for these blocks.");
    if (!input.Width) return;
    const auto create = [&](uint32_t rows) {
        Block result;
        result.Rows = rows;
        result.Width = input.Width;
        const size_t bytes = size_t(rows) * input.Width * sizeof(float);
        result.Impl->Buffer = [Impl->Device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!result.Impl->Buffer) throw MetalError(@"Allocating finite-cell Metal multigrid block failed");
        return result;
    };
    @autoreleasepool {
        id<MTLCommandBuffer> batch = Impl->Command();
        id<MTLComputeCommandEncoder> pass = [batch computeCommandEncoder];
        const auto sparse = [&](id<MTLBuffer> offsets, id<MTLBuffer> columns, id<MTLBuffer> values,
                                const Block &source, Block &destination) {
            EncodeLinear(
                pass, Impl->SparseProductPipeline,
                {offsets, columns, values, source.Impl->Buffer, destination.Impl->Buffer},
                SparseProductParams{destination.Rows, source.Rows, source.Width},
                size_t(destination.Rows) * destination.Width
            );
        };
        const auto smooth = [&](const Implementation::MultigridLevel &level, const Block &source, Block &destination,
                                float weight) {
            EncodeLinear(
                pass, Impl->SmootherPipeline, {level.Diagonal, source.Impl->Buffer, destination.Impl->Buffer},
                AlgebraParams{source.Rows * source.Width, source.Rows, 0, weight},
                size_t(source.Rows) * source.Width
            );
        };
        const auto combine = [&](
                                 const Block &left, float left_scale, const Block &right, float right_scale, Block &destination
                             ) {
            EncodeLinear(
                pass, Impl->CombinationPipeline, {left.Impl->Buffer, right.Impl->Buffer, destination.Impl->Buffer},
                AlgebraParams{left.Rows * left.Width, left.Rows, left_scale, right_scale},
                size_t(left.Rows) * left.Width
            );
        };
        // Chebyshev-weighted Jacobi sweeps around one coarse correction, ping-ponging the level blocks.
        const auto cycle = [&](this const auto &self, uint32_t index, const Block &right_hand_side) -> Block {
            const auto &level = Impl->Multigrid[index];
            auto result = create(level.Rows);
            if (index + 1 == Impl->Multigrid.size()) {
                EncodeLinear(
                    pass, Impl->DenseProductPipeline,
                    {level.Inverse, right_hand_side.Impl->Buffer, result.Impl->Buffer},
                    DenseProductParams{level.Rows, result.Width}, size_t(level.Rows) * result.Width
                );
                return result;
            }
            auto next = create(level.Rows), action = create(level.Rows);
            constexpr uint32_t smoothing_steps{4};
            const float lower = level.Maximum / 10, center = 0.5f * (level.Maximum + lower);
            const float radius = 0.5f * (level.Maximum - lower);
            const auto weight = [=](uint32_t root) {
                return 1 / (center - radius * std::cos(float(std::numbers::pi) * float(2 * root + 1) / float(2 * smoothing_steps)));
            };
            const auto sweep = [&](uint32_t root) {
                sparse(level.Offsets, level.Columns, level.Values, result, action);
                EncodeLinear(
                    pass, Impl->JacobiUpdatePipeline,
                    {level.Diagonal, right_hand_side.Impl->Buffer, action.Impl->Buffer,
                     result.Impl->Buffer, next.Impl->Buffer},
                    AlgebraParams{result.Rows * result.Width, result.Rows, 0, weight(root)},
                    size_t(result.Rows) * result.Width
                );
                std::swap(result, next);
            };
            smooth(level, right_hand_side, result, weight(0));
            for (uint32_t step = 1; step < smoothing_steps; ++step) sweep(step);
            sparse(level.Offsets, level.Columns, level.Values, result, action);
            combine(right_hand_side, 1, action, -1, next);
            auto coarse_residual = create(level.CoarseRows);
            sparse(
                level.RestrictionOffsets, level.RestrictionColumns, level.RestrictionValues,
                next, coarse_residual
            );
            const auto coarse_correction = self(index + 1, coarse_residual);
            sparse(
                level.ProlongationOffsets, level.ProlongationColumns, level.ProlongationValues,
                coarse_correction, action
            );
            combine(result, 1, action, 1, next);
            std::swap(result, next);
            for (uint32_t step = 0; step < smoothing_steps; ++step) sweep(smoothing_steps - 1 - step);
            return result;
        };
        const auto result = cycle(0, input);
        [pass endEncoding];
        id<MTLBlitCommandEncoder> copy = [batch blitCommandEncoder];
        [copy copyFromBuffer:result.Impl->Buffer
                 sourceOffset:0
                     toBuffer:output.Impl->Buffer
            destinationOffset:0
                         size:size_t(result.Rows) * result.Width * sizeof(float)];
        [copy endEncoding];
    }
}

void modal::FiniteCellMetal::LinearCombination(
    const Block &left, float left_scale, const Block &right, float right_scale, Block &output
) const {
    RequireMatching(*Impl, left, right);
    if (!output.Impl || left.Rows != output.Rows || left.Width != output.Width ||
        (left.Width && (!output.Impl->Buffer || output.Impl->Buffer.device != Impl->Device)))
        throw std::invalid_argument("Finite-cell Metal combination output must have matching dimensions.");
    if (!left.Width) return;
    @autoreleasepool {
        id<MTLCommandBuffer> command = Impl->Command();
        id<MTLComputeCommandEncoder> pass = [command computeCommandEncoder];
        EncodeLinear(
            pass, Impl->CombinationPipeline, {left.Impl->Buffer, right.Impl->Buffer, output.Impl->Buffer},
            AlgebraParams{left.Rows * left.Width, left.Rows, left_scale, right_scale},
            size_t(left.Rows) * left.Width
        );
        [pass endEncoding];
    }
}

void modal::FiniteCellMetal::ConfigurePackedPatch(
    SharedFloats inverse_matrices, std::span<const double> element_matrices
) const {
    const uint32_t local_dofs = 3 * Impl->NodesPerCell;
    const size_t values_per_cell = size_t(local_dofs) * (local_dofs + 1) / 2;
    const size_t matrix_values = Impl->CellOrder.size() * values_per_cell;
    if (inverse_matrices.Size != matrix_values || element_matrices.size() != matrix_values)
        throw std::invalid_argument("Finite-cell Metal owned packed patch data dimensions do not match the operator.");
    Synchronize();
    Impl->PackedPatchInverses = [Impl->Device newBufferWithBytesNoCopy:inverse_matrices.Data length:inverse_matrices.CapacityBytes options:MTLResourceStorageModeShared deallocator:^(void *pointer, NSUInteger) { std::free(pointer); }];
    if (!Impl->PackedPatchInverses) throw MetalError(@"Wrapping finite-cell Metal packed inverses failed");
    inverse_matrices.Release();
    Impl->PackedElementMatrices = [Impl->Device newBufferWithLength:matrix_values * sizeof(float) options:MTLResourceStorageModeShared];
    if (!Impl->PackedElementMatrices)
        throw MetalError(@"Allocating finite-cell Metal owned packed patch data failed");
    auto *elements = static_cast<float *>(Impl->PackedElementMatrices.contents);
    dispatch_apply(Impl->CellOrder.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t cell) {
      const size_t source = size_t(Impl->CellOrder[cell]) * values_per_cell;
      const size_t destination = cell * values_per_cell;
      for (size_t entry = 0; entry < values_per_cell; ++entry)
          elements[destination + entry] = float(element_matrices[source + entry]);
    });
}

void modal::FiniteCellMetal::ApplyElement(const Block &input, Block &output) const {
    RequireFine(*Impl, input, output);
    if (!Impl->PackedElementMatrices)
        throw std::logic_error("Finite-cell Metal element operators have not been configured.");
    if (!input.Width) return;
    @autoreleasepool {
        const uint32_t local_dofs = 3 * Impl->NodesPerCell;
        Impl->ReserveScratch(input.Width);
        id<MTLCommandBuffer> command = Impl->Command();
        id<MTLComputeCommandEncoder> pass = [command computeCommandEncoder];
        const uint32_t count = input.Rows * input.Width;
        const uint32_t cells = uint32_t(Impl->CellOrder.size());
        EncodeLinear(pass, Impl->ClearPipeline, {output.Impl->Buffer}, count, count);
        EncodeCooperative(
            pass, Impl->CooperativeWideBatchedPackedCellMatrixPipeline,
            {Impl->Cells, Impl->PackedElementMatrices, Impl->CellIndices, input.Impl->Buffer, Impl->Scratch},
            LocalActionParams{Impl->NumNodes, input.Width, local_dofs, 0, cells, 0},
            size_t(cells) * ((input.Width + 7) / 8), local_dofs,
            @"Finite-cell cooperative packed element action exceeds the Metal threadgroup limit"
        );
        for (uint32_t color = 0; color < 8; ++color) {
            const uint32_t first = Impl->ColorOffsets[color], color_cells = Impl->ColorOffsets[color + 1] - first;
            if (!color_cells) continue;
            EncodeLinear(
                pass, Impl->CellMatrixScatterPipeline,
                {Impl->Cells, Impl->CellIndices, Impl->Scratch, output.Impl->Buffer},
                LocalActionParams{Impl->NumNodes, input.Width, local_dofs, first, color_cells, 0},
                size_t(color_cells) * input.Width * local_dofs
            );
        }
        [pass endEncoding];
    }
}

void modal::FiniteCellMetal::ApplyPackedLocalizedMultiplicativePatchSweep(
    const Block &input, Block &correction, Block &remaining, bool reverse, bool final_residual
) const {
    RequireFine(*Impl, input, correction);
    RequireFine(*Impl, input, remaining);
    if (!Impl->PackedPatchInverses || !Impl->PackedElementMatrices)
        throw std::logic_error("Finite-cell Metal patch preconditioner has not been configured.");
    if (input.Impl->Buffer == correction.Impl->Buffer || input.Impl->Buffer == remaining.Impl->Buffer ||
        correction.Impl->Buffer == remaining.Impl->Buffer)
        throw std::invalid_argument("Finite-cell Metal multiplicative sweep requires distinct blocks.");
    if (!input.Width) return;
    @autoreleasepool {
        const uint32_t local_dofs = 3 * Impl->NodesPerCell;
        Impl->ReserveScratch(input.Width);
        const auto delta_block = CreateBlock(input.Width), action_block = CreateBlock(input.Width);
        id<MTLBuffer> delta = delta_block.Impl->Buffer, action = action_block.Impl->Buffer;
        id<MTLCommandBuffer> command = Impl->Command();
        id<MTLBlitCommandEncoder> copy = [command blitCommandEncoder];
        [copy copyFromBuffer:input.Impl->Buffer
                 sourceOffset:0
                     toBuffer:remaining.Impl->Buffer
            destinationOffset:0
                         size:size_t(input.Rows) * input.Width * sizeof(float)];
        [copy endEncoding];
        id<MTLComputeCommandEncoder> pass = [command computeCommandEncoder];
        const uint32_t elements = input.Rows * input.Width, batches = (input.Width + 7) / 8;
        const auto encode_clear = [&](id<MTLBuffer> output) {
            EncodeLinear(pass, Impl->ClearPipeline, {output}, elements, elements);
        };
        const auto encode_combination = [&](id<MTLBuffer> left, float left_scale, id<MTLBuffer> right, float right_scale, id<MTLBuffer> output) {
            EncodeLinear(
                pass, Impl->CombinationPipeline, {left, right, output},
                AlgebraParams{elements, input.Rows, left_scale, right_scale}, elements
            );
        };
        // A patch solve is the packed cell-matrix action against the stored inverses, over the
        // identity cell order so each color writes its own cells' scratch slots.
        const auto encode_patch = [&](id<MTLBuffer> residual, uint8_t color) {
            const uint32_t first = Impl->ColorOffsets[color], cells = Impl->ColorOffsets[color + 1] - first;
            if (!cells) return;
            const LocalActionParams params{Impl->NumNodes, input.Width, local_dofs, first, cells, 0};
            EncodeCooperative(
                pass, Impl->CooperativeWideBatchedPackedCellMatrixPipeline,
                {Impl->Cells, Impl->PackedPatchInverses, Impl->CellIndices, residual, Impl->Scratch},
                params, size_t(cells) * batches, local_dofs,
                @"Finite-cell cooperative patch exceeds the Metal threadgroup limit"
            );
            EncodeLinear(
                pass, Impl->CellMatrixScatterPipeline,
                {Impl->Cells, Impl->CellIndices, Impl->Scratch, delta}, params,
                size_t(cells) * input.Width * local_dofs
            );
        };
        const auto encode_local_action = [&](uint8_t source_color) {
            const auto &offsets = Impl->LocalizedOffsets[source_color];
            const uint32_t first = offsets[0], end = offsets[8];
            if (first == end) return;
            EncodeCooperative(
                pass, Impl->CooperativeWideBatchedPackedCellMatrixPipeline,
                {Impl->Cells, Impl->PackedElementMatrices, Impl->LocalizedCells, delta, Impl->Scratch},
                LocalActionParams{Impl->NumNodes, input.Width, local_dofs, first, end - first, first},
                size_t(end - first) * batches, local_dofs,
                @"Finite-cell cooperative cell action exceeds the Metal threadgroup limit"
            );
            for (uint8_t output_color = 0; output_color < 8; ++output_color) {
                const uint32_t color_first = offsets[output_color], color_end = offsets[output_color + 1];
                if (color_first == color_end) continue;
                EncodeLinear(
                    pass, Impl->CellMatrixScatterPipeline,
                    {Impl->Cells, Impl->LocalizedCells, Impl->Scratch, action},
                    LocalActionParams{Impl->NumNodes, input.Width, local_dofs, color_first, color_end - color_first, first},
                    size_t(color_end - color_first) * input.Width * local_dofs
                );
            }
        };

        encode_clear(correction.Impl->Buffer);
        for (uint8_t step = 0; step < 8; ++step) {
            const uint8_t color = reverse ? uint8_t(7 - step) : step;
            encode_clear(delta);
            encode_patch(remaining.Impl->Buffer, color);
            encode_combination(correction.Impl->Buffer, 1, delta, 1, correction.Impl->Buffer);
            if (step == 7 && !final_residual) continue;
            encode_clear(action);
            encode_local_action(color);
            encode_combination(remaining.Impl->Buffer, 1, action, -1, remaining.Impl->Buffer);
        }
        [pass endEncoding];
    }
}

void modal::FiniteCellMetal::RestrictP1(const Block &fine, Block &coarse) const {
    RequireTransfer(*Impl, fine, coarse, "restriction requires matching fine and P1 blocks.");
    if (!fine.Width) return;
    @autoreleasepool {
        id<MTLCommandBuffer> command = Impl->Command();
        id<MTLComputeCommandEncoder> pass = [command computeCommandEncoder];
        EncodeLinear(
            pass, Impl->RestrictPipeline,
            {fine.Impl->Buffer, Impl->P1Offsets, Impl->P1Occurrences, coarse.Impl->Buffer},
            TransferParams{Impl->NumP1Nodes, Impl->NumNodes, fine.Width}, size_t(coarse.Rows) * coarse.Width
        );
        [pass endEncoding];
    }
}

void modal::FiniteCellMetal::ProlongP1(const Block &coarse, Block &fine) const {
    RequireTransfer(*Impl, fine, coarse, "prolongation requires matching P1 and fine blocks.");
    if (!fine.Width) return;
    @autoreleasepool {
        id<MTLCommandBuffer> command = Impl->Command();
        id<MTLComputeCommandEncoder> pass = [command computeCommandEncoder];
        EncodeLinear(
            pass, Impl->ProlongPipeline, {coarse.Impl->Buffer, Impl->P1Stencils, fine.Impl->Buffer},
            TransferParams{Impl->NumP1Nodes, Impl->NumNodes, fine.Width}, size_t(fine.Rows) * fine.Width
        );
        [pass endEncoding];
    }
}
