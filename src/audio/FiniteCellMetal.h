#pragma once

#include "FiniteCell.h"

#include <memory>
#include <span>
#include <vector>

namespace modal {
struct FiniteCellMetal {
    struct SharedFloats {
        SharedFloats();
        explicit SharedFloats(size_t size);
        SharedFloats(SharedFloats &&) noexcept;
        SharedFloats &operator=(SharedFloats &&) noexcept;
        ~SharedFloats();
        std::span<float> Values() const;
        float *Release();
        float *Data{};
        size_t Size{}, CapacityBytes{};
    };
    struct Implementation;
    struct Block {
        struct Implementation;
        std::unique_ptr<Implementation> Impl;
        uint32_t Rows{}, Width{};

        Block();
        Block(Block &&) noexcept;
        Block &operator=(Block &&) noexcept;
        ~Block();
    };
    struct P1Multigrid {
        struct Implementation;
        std::unique_ptr<Implementation> Impl;

        P1Multigrid();
        P1Multigrid(P1Multigrid &&) noexcept;
        P1Multigrid &operator=(P1Multigrid &&) noexcept;
        ~P1Multigrid();
    };

    std::unique_ptr<Implementation> Impl;

    explicit FiniteCellMetal(const FiniteCellOperator &);
    ~FiniteCellMetal();

    Block CreateBlock(uint32_t width) const;
    Block CreateSharedBlock(uint32_t width) const;
    Block CreateP1Block(uint32_t width) const;
    void Upload(Block &, const double *) const;
    void Download(const Block &, double *) const;
    static P1Multigrid PrepareP1Multigrid(const FiniteCellOperator &, double alpha, const FiniteCellOperator::AssembledLower &);
    void ConfigureP1Multigrid(P1Multigrid &&) const;
    void ApplyP1Multigrid(const Block &input, Block &output) const;
    void LinearCombination(const Block &left, float left_scale, const Block &right, float right_scale, Block &output) const;
    void ConfigurePackedPatch(SharedFloats inverse_matrices, std::span<const double> element_matrices) const;
    void ApplyElement(const Block &input, Block &output) const;
    // Colored multiplicative patch sweep: `correction` accumulates the patch solves and
    // `remaining` carries the residual left after each color's exact local action.
    void ApplyPackedLocalizedMultiplicativePatchSweep(
        const Block &input, Block &correction, Block &remaining,
        bool reverse = false, bool final_residual = true
    ) const;
    void RestrictP1(const Block &fine, Block &coarse) const;
    void ProlongP1(const Block &coarse, Block &fine) const;

private:
    void Synchronize() const;
};
} // namespace modal
