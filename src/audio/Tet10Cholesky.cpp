#include "Tet10Cholesky.h"

#include "SparseExecution.h"
#include "Tet10Assembler.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Block = modal::Tet10Assembler::ElementBlock;

struct BlockPencil {
    int Nodes{};
    std::vector<long> ColumnStarts;
    std::vector<int> Rows;
    std::vector<Block> Stiffness;
    std::vector<double> Mass;
    std::vector<std::array<uint32_t, modal::Tet10Assembler::LowerBlocksPerElement>> ElementEntries;
};

uint32_t FindEntry(const BlockPencil &pencil, uint32_t row, uint32_t column) {
    const auto begin = pencil.Rows.begin() + pencil.ColumnStarts[column];
    const auto end = pencil.Rows.begin() + pencil.ColumnStarts[column + 1];
    const auto found = std::lower_bound(begin, end, int(row));
    if (found == end || *found != int(row)) throw std::logic_error("Tet10 Cholesky pencil pattern is incomplete.");
    return uint32_t(found - pencil.Rows.begin());
}

BlockPencil MakePattern(const modal::Tet10Assembler &fem) {
    if (fem.NumNodes == 0 || fem.Elements().empty()) throw std::invalid_argument("Tet10 Cholesky requires a nonempty Tet10 mesh.");
    BlockPencil pencil;
    pencil.Nodes = int(fem.NumNodes);
    std::vector<std::vector<int>> column_rows(fem.NumNodes);
    for (const auto &element : fem.Elements()) {
        for (uint32_t a = 0; a < modal::Tet10Assembler::NodesPerElement; ++a) {
            for (uint32_t c = 0; c <= a; ++c) {
                const uint32_t row = std::max(element.Nodes[a], element.Nodes[c]);
                column_rows[std::min(element.Nodes[a], element.Nodes[c])].push_back(int(row));
            }
        }
    }
    pencil.ColumnStarts.resize(size_t(pencil.Nodes) + 1);
    for (int column = 0; column < pencil.Nodes; ++column) {
        auto &rows = column_rows[column];
        std::ranges::sort(rows);
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        pencil.ColumnStarts[column + 1] = pencil.ColumnStarts[column] + long(rows.size());
        pencil.Rows.insert(pencil.Rows.end(), rows.begin(), rows.end());
    }
    pencil.Stiffness.resize(pencil.Rows.size());
    pencil.Mass.resize(pencil.Rows.size());
    pencil.ElementEntries.resize(fem.Elements().size());
    for (size_t element_index = 0; element_index < fem.Elements().size(); ++element_index) {
        const auto &nodes = fem.Elements()[element_index].Nodes;
        uint32_t pair{};
        for (uint32_t a = 0; a < modal::Tet10Assembler::NodesPerElement; ++a) {
            for (uint32_t c = 0; c <= a; ++c) {
                const uint32_t row = std::max(nodes[a], nodes[c]), column = std::min(nodes[a], nodes[c]);
                pencil.ElementEntries[element_index][pair++] = FindEntry(pencil, row, column);
            }
        }
    }
    return pencil;
}

void Assemble(const modal::Tet10Assembler &fem, BlockPencil &pencil) {
    std::ranges::fill(pencil.Stiffness, Block{});
    std::ranges::fill(pencil.Mass, 0.0);
    for (size_t element_index = 0; element_index < fem.Elements().size(); ++element_index) {
        const auto &element = fem.Elements()[element_index];
        uint32_t pair{};
        for (uint32_t a = 0; a < modal::Tet10Assembler::NodesPerElement; ++a) {
            for (uint32_t c = 0; c <= a; ++c) {
                const bool transpose = element.Nodes[a] < element.Nodes[c];
                Block stiffness;
                double mass;
                fem.EvaluateBlock(element, transpose ? c : a, transpose ? a : c, stiffness, mass);
                const uint32_t entry = pencil.ElementEntries[element_index][pair++];
                for (uint32_t value = 0; value < stiffness.size(); ++value) pencil.Stiffness[entry][value] += stiffness[value];
                pencil.Mass[entry] += mass;
            }
        }
    }
}

SparseMatrix_Double MatrixView(BlockPencil &pencil, std::vector<Block> &values) {
    return {
        .structure = {
            .rowCount = pencil.Nodes,
            .columnCount = pencil.Nodes,
            .columnStarts = pencil.ColumnStarts.data(),
            .rowIndices = pencil.Rows.data(),
            .attributes = {.transpose = false, .triangle = SparseLowerTriangle, .kind = SparseSymmetric, ._reserved = 0, ._allocatedBySparse = false},
            .blockSize = 3,
        },
        .data = values.front().data(),
    };
}

SparseSymbolicFactorOptions SymbolicOptions(bool use_metis, int *permutation = nullptr) {
    return {
        .control = SparseDefaultControl,
        .orderMethod = use_metis ? SparseOrderMetis : SparseOrderAMD,
        .order = permutation,
        .ignoreRowsAndColumns = nullptr,
        .malloc = std::malloc,
        .free = std::free,
        .reportError = nullptr,
    };
}

std::vector<int> Order(BlockPencil &pencil) {
    ConfigureAccelerateSparseExecution();
    std::vector<int> permutation(pencil.Nodes);
    auto matrix = MatrixView(pencil, pencil.Stiffness);
    auto symbolic = SparseFactor(SparseFactorizationCholesky, matrix.structure, SymbolicOptions(3 * pencil.Nodes >= 8'000, permutation.data()));
    if (symbolic.status != SparseStatusOK) {
        SparseCleanup(symbolic);
        throw std::runtime_error("Tet10 Cholesky symbolic factorization failed.");
    }
    SparseCleanup(symbolic);
    return permutation;
}

struct SupernodePanel {
    int Begin{}, End{};
    std::vector<int> Rows;
    std::unique_ptr<double[]> Values;
    std::vector<uint32_t> ParentRows;
    int Parent{-1};

    int WidthBlocks() const { return End - Begin; }
    int Width() const { return 3 * WidthBlocks(); }
    int Height() const { return 3 * int(Rows.size()); }
};

struct InputScatter {
    uint32_t Source, Row;
    uint16_t Column;
    bool Transpose;
};

size_t PackedBlockCount(size_t blocks) { return blocks * (blocks + 1) / 2; }

size_t PackedBlockIndex(size_t blocks, size_t row, size_t column) { return column * (2 * blocks - column + 1) / 2 + row - column; }

constexpr size_t SchurTileBlocks{64};

struct SymbolicAnalysis {
    std::vector<int> Parents;
    std::vector<int> Degrees;
};

struct PermutedPattern {
    std::vector<long> RowStarts;
    std::vector<int> Columns;
};

PermutedPattern BuildPattern(const BlockPencil &pencil, const std::vector<int> &permutation) {
    PermutedPattern pattern;
    pattern.RowStarts.assign(size_t(pencil.Nodes) + 1, 0);
    for (int original_column = 0; original_column < pencil.Nodes; ++original_column) {
        for (long source = pencil.ColumnStarts[original_column]; source < pencil.ColumnStarts[original_column + 1]; ++source) {
            const int original_row = pencil.Rows[source];
            const int permuted_row = permutation[original_row], permuted_column = permutation[original_column];
            ++pattern.RowStarts[size_t(std::max(permuted_row, permuted_column)) + 1];
        }
    }
    for (int row = 0; row < pencil.Nodes; ++row) pattern.RowStarts[row + 1] += pattern.RowStarts[row];
    pattern.Columns.resize(pencil.Rows.size());
    std::vector<long> next = pattern.RowStarts;
    for (int original_column = 0; original_column < pencil.Nodes; ++original_column) {
        for (long source = pencil.ColumnStarts[original_column]; source < pencil.ColumnStarts[original_column + 1]; ++source) {
            const int original_row = pencil.Rows[source];
            const int permuted_row = permutation[original_row], permuted_column = permutation[original_column];
            const int row = std::max(permuted_row, permuted_column), column = std::min(permuted_row, permuted_column);
            pattern.Columns[next[row]++] = column;
        }
    }
    for (int row = 0; row < pencil.Nodes; ++row)
        std::sort(pattern.Columns.begin() + pattern.RowStarts[row], pattern.Columns.begin() + pattern.RowStarts[row + 1]);
    return pattern;
}

SymbolicAnalysis Analyze(const PermutedPattern &pattern) {
    const int nodes = int(pattern.RowStarts.size()) - 1;
    SymbolicAnalysis analysis;
    analysis.Parents.assign(nodes, -1);
    analysis.Degrees.assign(nodes, 0);
    std::vector<int> marks(nodes, -1);
    for (int row = 0; row < nodes; ++row) {
        marks[row] = row;
        for (long entry = pattern.RowStarts[row]; entry < pattern.RowStarts[row + 1]; ++entry) {
            int column = pattern.Columns[entry];
            while (marks[column] != row) {
                marks[column] = row;
                ++analysis.Degrees[column];
                if (analysis.Parents[column] < 0) analysis.Parents[column] = row;
                column = analysis.Parents[column];
            }
        }
    }
    return analysis;
}

struct RelaxedOrdering {
    std::vector<int> Permutation, PanelSizes;
    std::vector<std::vector<int>> PanelRows;
};

RelaxedOrdering Relax(const BlockPencil &pencil, std::vector<int> permutation) {
    const auto analysis = Analyze(BuildPattern(pencil, permutation));
    std::vector<int> scalar_children(pencil.Nodes);
    for (const int parent : analysis.Parents)
        if (parent >= 0) ++scalar_children[parent];

    std::vector<std::pair<int, int>> fundamental;
    for (int begin = 0; begin < pencil.Nodes;) {
        int end = begin + 1;
        while (end < pencil.Nodes && analysis.Parents[end - 1] == end && scalar_children[end] == 1 && analysis.Degrees[end] == analysis.Degrees[end - 1] - 1) {
            ++end;
        }
        fundamental.emplace_back(begin, end);
        begin = end;
    }

    std::vector<int> node_to_supernode(pencil.Nodes), parents(fundamental.size(), -1), sizes(fundamental.size()), degrees(fundamental.size());
    for (size_t supernode = 0; supernode < fundamental.size(); ++supernode) {
        const auto [begin, end] = fundamental[supernode];
        sizes[supernode] = end - begin;
        for (int node = begin; node < end; ++node) node_to_supernode[node] = int(supernode);
    }
    for (size_t supernode = 0; supernode < fundamental.size(); ++supernode) {
        const auto [begin, end] = fundamental[supernode];
        degrees[supernode] = analysis.Degrees[end - 1];
        if (analysis.Parents[end - 1] >= 0) parents[supernode] = node_to_supernode[analysis.Parents[end - 1]];
    }

    std::vector<std::vector<int>> children(fundamental.size());
    for (int supernode = 0; supernode < int(fundamental.size()); ++supernode)
        if (parents[supernode] >= 0) children[parents[supernode]].push_back(supernode);
    std::vector<int64_t> zeros(fundamental.size());
    std::vector<int> merge_parents(fundamental.size(), -1), last_merged_children(fundamental.size(), -1);
    const auto Mergable = [&](int child, int parent) {
        const int64_t old_zeros = zeros[child] + zeros[parent];
        const int64_t new_zeros = int64_t(sizes[parent] + degrees[parent] - degrees[child]) * sizes[child];
        if (new_zeros == 0) return std::pair{true, old_zeros};
        const int64_t merged_zeros = old_zeros + new_zeros;
        const int64_t combined_size = 3 * int64_t(sizes[child] + sizes[parent]);
        const int64_t expanded_entries = combined_size * (combined_size + 1) / 2 + 3 * int64_t(degrees[parent]) * combined_size;
        const int64_t scalar_zeros = 9 * merged_zeros;
        constexpr std::array<std::pair<int, double>, 4> cutoffs{{{4, 1}, {16, 0.8}, {48, 0.1}, {std::numeric_limits<int>::max(), 0.05}}};
        for (const auto [limit, ratio] : cutoffs)
            if (combined_size <= limit && scalar_zeros <= int64_t(ratio * expanded_entries)) return std::pair{true, merged_zeros};
        return std::pair{false, merged_zeros};
    };
    for (int parent = 0; parent < int(fundamental.size()); ++parent) {
        while (true) {
            int selected{-1}, largest_size{};
            int64_t selected_new_zeros{}, selected_merged_zeros{};
            for (const int child : children[parent]) {
                if (merge_parents[child] >= 0 || sizes[child] < largest_size) continue;
                const auto [mergable, merged_zeros] = Mergable(child, parent);
                if (!mergable) continue;
                const int64_t new_zeros = merged_zeros - zeros[child] - zeros[parent];
                if (selected < 0 || sizes[child] > largest_size || new_zeros < selected_new_zeros) {
                    selected = child;
                    selected_new_zeros = new_zeros;
                    selected_merged_zeros = merged_zeros;
                    largest_size = sizes[child];
                }
            }
            if (selected < 0) break;
            sizes[parent] += sizes[selected];
            sizes[selected] = 0;
            zeros[parent] = selected_merged_zeros;
            merge_parents[selected] = last_merged_children[parent] < 0 ? parent : last_merged_children[parent];
            last_merged_children[parent] = last_merged_children[selected] < 0 ? selected : last_merged_children[selected];
        }
    }

    std::vector<int> new_to_old, roots, panel_owners(fundamental.size(), -1);
    RelaxedOrdering relaxed;
    new_to_old.reserve(pencil.Nodes);
    for (int root = 0; root < int(fundamental.size()); ++root) {
        if (merge_parents[root] >= 0) continue;
        roots.push_back(root);
        const int panel = int(roots.size()) - 1;
        const size_t panel_begin = new_to_old.size();
        for (int supernode = last_merged_children[root] < 0 ? root : last_merged_children[root]; supernode >= 0; supernode = merge_parents[supernode]) {
            panel_owners[supernode] = panel;
            const auto [begin, end] = fundamental[supernode];
            for (int node = begin; node < end; ++node) new_to_old.push_back(node);
        }
        relaxed.PanelSizes.push_back(int(new_to_old.size() - panel_begin));
    }
    if (new_to_old.size() != size_t(pencil.Nodes)) throw std::logic_error("Relaxed supernodal ordering is incomplete.");
    std::vector<int> old_to_new(pencil.Nodes);
    for (int node = 0; node < pencil.Nodes; ++node) old_to_new[new_to_old[node]] = node;
    for (int original = 0; original < pencil.Nodes; ++original) permutation[original] = old_to_new[permutation[original]];

    std::vector<int> node_to_panel(pencil.Nodes), panel_parents(roots.size(), -1), panel_begins(roots.size() + 1);
    for (size_t panel = 0; panel < roots.size(); ++panel) {
        panel_begins[panel + 1] = panel_begins[panel] + relaxed.PanelSizes[panel];
        for (int node = panel_begins[panel]; node < panel_begins[panel + 1]; ++node) node_to_panel[node] = int(panel);
        if (parents[roots[panel]] >= 0) panel_parents[panel] = panel_owners[parents[roots[panel]]];
    }

    relaxed.PanelRows.resize(roots.size());
    for (size_t panel = 0; panel < roots.size(); ++panel) {
        auto &rows = relaxed.PanelRows[panel];
        rows.reserve(relaxed.PanelSizes[panel] + degrees[roots[panel]]);
        for (int node = panel_begins[panel]; node < panel_begins[panel + 1]; ++node) rows.push_back(node);
    }
    const auto final_pattern = BuildPattern(pencil, permutation);
    std::vector<int> marks(roots.size(), -1);
    for (size_t panel = 0; panel < roots.size(); ++panel) {
        for (int row = panel_begins[panel]; row < panel_begins[panel + 1]; ++row) {
            marks[panel] = row;
            for (long entry = final_pattern.RowStarts[row]; entry < final_pattern.RowStarts[row + 1]; ++entry) {
                const int column = final_pattern.Columns[entry];
                if (column >= panel_begins[panel]) continue;
                for (int ancestor = node_to_panel[column]; marks[ancestor] != row; ancestor = panel_parents[ancestor]) {
                    relaxed.PanelRows[ancestor].push_back(row);
                    marks[ancestor] = row;
                }
            }
        }
    }
    relaxed.Permutation = std::move(permutation);
    return relaxed;
}

struct SupernodalFactor {
    std::vector<int> Permutation;
    std::vector<SupernodePanel> Panels;
    std::vector<std::vector<InputScatter>> Inputs;
    std::vector<std::vector<uint32_t>> Children;
    std::unique_ptr<double[]> NumericWorkspace;
    mutable std::vector<double> SolveWorkspace, SolveScratch;
    int MaximumBelow{};

    SupernodalFactor(const BlockPencil &pencil, std::vector<int> permutation) {
        auto relaxed = Relax(pencil, std::move(permutation));
        Permutation = std::move(relaxed.Permutation);

        std::vector<int> node_to_panel(pencil.Nodes);
        int begin{};
        for (const int panel_size : relaxed.PanelSizes) {
            const int end = begin + panel_size;
            const int panel_index = int(Panels.size());
            for (int node = begin; node < end; ++node) node_to_panel[node] = panel_index;
            auto &panel = Panels.emplace_back();
            panel.Begin = begin;
            panel.End = end;
            panel.Rows = std::move(relaxed.PanelRows[panel_index]);
            panel.Values = std::make_unique_for_overwrite<double[]>(size_t(panel.Height()) * panel.Width());
            MaximumBelow = std::max(MaximumBelow, panel.Height() - panel.Width());
            begin = end;
        }

        Inputs.resize(Panels.size());
        for (int original_column = 0; original_column < pencil.Nodes; ++original_column) {
            for (long source = pencil.ColumnStarts[original_column]; source < pencil.ColumnStarts[original_column + 1]; ++source) {
                const int original_row = pencil.Rows[source];
                const int permuted_row = Permutation[original_row], permuted_column = Permutation[original_column];
                const int row = std::max(permuted_row, permuted_column), column = std::min(permuted_row, permuted_column);
                const int panel_index = node_to_panel[column];
                auto &panel = Panels[panel_index];
                const auto found = std::lower_bound(panel.Rows.begin(), panel.Rows.end(), row);
                if (found == panel.Rows.end() || *found != row) throw std::logic_error("Supernodal input is outside its factor panel.");
                Inputs[panel_index].push_back({uint32_t(source), uint32_t(found - panel.Rows.begin()), uint16_t(column - panel.Begin), permuted_row < permuted_column});
            }
        }

        for (auto &panel : Panels) {
            if (panel.Rows.size() == size_t(panel.WidthBlocks())) continue;
            panel.Parent = node_to_panel[panel.Rows[panel.WidthBlocks()]];
            if (panel.Parent <= node_to_panel[panel.Begin]) throw std::logic_error("Supernodal assembly forest is not ordered.");
            const auto &parent = Panels[panel.Parent];
            panel.ParentRows.reserve(panel.Rows.size() - panel.WidthBlocks());
            for (size_t row = panel.WidthBlocks(); row < panel.Rows.size(); ++row) {
                const auto found = std::lower_bound(parent.Rows.begin(), parent.Rows.end(), panel.Rows[row]);
                if (found == parent.Rows.end() || *found != panel.Rows[row]) throw std::logic_error("Supernodal child front is outside its parent front.");
                panel.ParentRows.push_back(uint32_t(found - parent.Rows.begin()));
            }
        }
        Children.resize(Panels.size());
        std::vector<size_t> workspace_sizes(Panels.size());
        for (uint32_t panel = 0; panel < Panels.size(); ++panel)
            if (Panels[panel].Parent >= 0) Children[Panels[panel].Parent].push_back(panel);
        size_t workspace_size{};
        for (size_t panel = 0; panel < Panels.size(); ++panel) {
            const size_t below_blocks = Panels[panel].Rows.size() - Panels[panel].WidthBlocks();
            size_t child_size{};
            for (const uint32_t child : Children[panel]) child_size = std::max(child_size, workspace_sizes[child]);
            const size_t scratch_size = 9 * below_blocks * std::min(below_blocks, SchurTileBlocks);
            workspace_sizes[panel] = 9 * PackedBlockCount(below_blocks) + std::max(child_size, scratch_size);
            if (Panels[panel].Parent < 0) workspace_size = std::max(workspace_size, workspace_sizes[panel]);
        }
        NumericWorkspace = std::make_unique_for_overwrite<double[]>(workspace_size);
    }

    void FactorPanel(size_t panel_index, size_t workspace_offset) {
        auto &panel = Panels[panel_index];
        const int height = panel.Height(), width = panel.Width(), below_blocks = int(panel.Rows.size()) - panel.WidthBlocks(), below = 3 * below_blocks;
        const size_t schur_entries = 9 * PackedBlockCount(below_blocks);
        double *schur = below ? NumericWorkspace.get() + workspace_offset : nullptr;
        if (below) std::fill_n(schur, schur_entries, 0);
        const size_t child_offset = workspace_offset + schur_entries;
        for (const uint32_t child_index : Children[panel_index]) {
            FactorPanel(child_index, child_offset);
            const auto &child = Panels[child_index];
            const int child_below_blocks = int(child.Rows.size()) - child.WidthBlocks();
            const double *child_schur = NumericWorkspace.get() + child_offset;
            const int parent_width_blocks = panel.WidthBlocks();
            for (size_t block_column = 0; block_column < child.ParentRows.size(); ++block_column) {
                const uint32_t destination_column = child.ParentRows[block_column];
                for (size_t block_row = block_column; block_row < child.ParentRows.size(); ++block_row) {
                    const uint32_t destination_row = child.ParentRows[block_row];
                    const double *source = child_schur + 9 * PackedBlockIndex(child_below_blocks, block_row, block_column);
                    if (destination_column < uint32_t(parent_width_blocks)) {
                        for (int column_component = 0; column_component < 3; ++column_component)
                            for (int row_component = 0; row_component < 3; ++row_component)
                                panel.Values[size_t(3 * destination_column + column_component) * height + 3 * destination_row + row_component] += source[3 * column_component + row_component];
                    } else {
                        double *destination = schur + 9 * PackedBlockIndex(below_blocks, destination_row - parent_width_blocks, destination_column - parent_width_blocks);
                        for (int entry = 0; entry < 9; ++entry) destination[entry] += source[entry];
                    }
                }
            }
        }

        __LAPACK_int lapack_width = width, leading = height, info{};
        dpotrf_("L", &lapack_width, panel.Values.get(), &leading, &info);
        if (info != 0) throw std::runtime_error("Supernodal Cholesky factorization failed.");
        if (below == 0) return;
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, below, width, 1, panel.Values.get(), height, panel.Values.get() + width, height);
        double *scratch = NumericWorkspace.get() + child_offset;
        for (int block_begin = 0; block_begin < below_blocks; block_begin += int(SchurTileBlocks)) {
            const int block_columns = std::min(int(SchurTileBlocks), below_blocks - block_begin);
            const int rows = below - 3 * block_begin, columns = 3 * block_columns;
            const double *factor = panel.Values.get() + width + 3 * block_begin;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rows, columns, width, -1, factor, height, factor, height, 0, scratch, rows);
            for (int block_column = 0; block_column < block_columns; ++block_column) {
                const int packed_column = block_begin + block_column;
                for (int packed_row = packed_column; packed_row < below_blocks; ++packed_row) {
                    double *destination = schur + 9 * PackedBlockIndex(below_blocks, packed_row, packed_column);
                    for (int column_component = 0; column_component < 3; ++column_component) {
                        const double *source = scratch + size_t(3 * block_column + column_component) * rows + 3 * (packed_row - block_begin);
                        for (int row_component = 0; row_component < 3; ++row_component) destination[3 * column_component + row_component] += source[row_component];
                    }
                }
            }
        }
    }

    void Factor(const BlockPencil &pencil, double sigma) {
        for (size_t panel_index = 0; panel_index < Panels.size(); ++panel_index) {
            auto &panel = Panels[panel_index];
            std::fill_n(panel.Values.get(), size_t(panel.Height()) * panel.Width(), 0);
            const int height = panel.Height();
            for (const auto &[source, row, column, transpose] : Inputs[panel_index]) {
                for (int block_column = 0; block_column < 3; ++block_column) {
                    for (int block_row = 0; block_row < 3; ++block_row) {
                        const int source_entry = transpose ? block_column + 3 * block_row : block_row + 3 * block_column;
                        panel.Values[size_t(3 * column + block_column) * height + 3 * row + block_row] = pencil.Stiffness[source][source_entry];
                    }
                }
                for (int component = 0; component < 3; ++component)
                    panel.Values[size_t(3 * column + component) * height + 3 * row + component] -= sigma * pencil.Mass[source];
            }
        }
        for (size_t panel = 0; panel < Panels.size(); ++panel)
            if (Panels[panel].Parent < 0) FactorPanel(panel, 0);
    }

    void Solve(const double *input, double *output, int rhs_count) const {
        const int dofs = 3 * int(Permutation.size());
        const size_t workspace_size = size_t(dofs) * rhs_count;
        if (SolveWorkspace.size() < workspace_size) SolveWorkspace.resize(workspace_size);
        const size_t scratch_size = size_t(MaximumBelow) * rhs_count;
        if (SolveScratch.size() < scratch_size) SolveScratch.resize(scratch_size);
        for (int rhs = 0; rhs < rhs_count; ++rhs)
            for (int original = 0; original < int(Permutation.size()); ++original)
                std::memcpy(SolveWorkspace.data() + size_t(rhs) * dofs + 3 * Permutation[original], input + size_t(rhs) * dofs + 3 * original, 3 * sizeof(double));

        for (const auto &panel : Panels) {
            const int height = panel.Height(), width = panel.Width(), below = height - width;
            double *diagonal = SolveWorkspace.data() + 3 * panel.Begin;
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, width, rhs_count, 1, panel.Values.get(), height, diagonal, dofs);
            if (below == 0) continue;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, below, rhs_count, width, 1, panel.Values.get() + width, height, diagonal, dofs, 0, SolveScratch.data(), below);
            for (int rhs = 0; rhs < rhs_count; ++rhs)
                for (size_t row = panel.WidthBlocks(); row < panel.Rows.size(); ++row)
                    for (int component = 0; component < 3; ++component)
                        SolveWorkspace[size_t(rhs) * dofs + 3 * panel.Rows[row] + component] -= SolveScratch[size_t(rhs) * below + 3 * (row - panel.WidthBlocks()) + component];
        }

        for (auto panel_iterator = Panels.rbegin(); panel_iterator != Panels.rend(); ++panel_iterator) {
            const auto &panel = *panel_iterator;
            const int height = panel.Height(), width = panel.Width(), below = height - width;
            double *diagonal = SolveWorkspace.data() + 3 * panel.Begin;
            if (below > 0) {
                for (int rhs = 0; rhs < rhs_count; ++rhs)
                    for (size_t row = panel.WidthBlocks(); row < panel.Rows.size(); ++row)
                        std::memcpy(SolveScratch.data() + size_t(rhs) * below + 3 * (row - panel.WidthBlocks()), SolveWorkspace.data() + size_t(rhs) * dofs + 3 * panel.Rows[row], 3 * sizeof(double));
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, width, rhs_count, below, -1, panel.Values.get() + width, height, SolveScratch.data(), below, 1, diagonal, dofs);
            }
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, width, rhs_count, 1, panel.Values.get(), height, diagonal, dofs);
        }

        for (int rhs = 0; rhs < rhs_count; ++rhs)
            for (int original = 0; original < int(Permutation.size()); ++original)
                std::memcpy(output + size_t(rhs) * dofs + 3 * original, SolveWorkspace.data() + size_t(rhs) * dofs + 3 * Permutation[original], 3 * sizeof(double));
    }
};
} // namespace

struct modal::Tet10Cholesky::Factorization {
    BlockPencil Pencil;
    std::vector<std::array<uint32_t, modal::Tet10Assembler::NodesPerElement>> ElementNodes;
    SupernodalFactor Native;
    bool HasNumeric{};
    double Shift{};

    explicit Factorization(const modal::Tet10Assembler &fem) : Pencil(MakePattern(fem)), Native(Pencil, Order(Pencil)) {
        ElementNodes.reserve(fem.Elements().size());
        for (const auto &element : fem.Elements()) ElementNodes.push_back(element.Nodes);
        Assemble(fem, Pencil);
    }

    void SetShift(double sigma) {
        if (HasNumeric && sigma == Shift) return;
        HasNumeric = false;
        Native.Factor(Pencil, sigma);
        Shift = sigma;
        HasNumeric = true;
    }
};

modal::Tet10Cholesky::Tet10Cholesky(const Tet10Assembler &fem) : Factor(std::make_unique<Factorization>(fem)) {}

modal::Tet10Cholesky::~Tet10Cholesky() = default;

void modal::Tet10Cholesky::SetShift(double sigma) { Factor->SetShift(sigma); }

void modal::Tet10Cholesky::Reassemble(const Tet10Assembler &fem) {
    if (int(fem.NumNodes) != Factor->Pencil.Nodes || fem.Elements().size() != Factor->ElementNodes.size())
        throw std::invalid_argument("Tet10 Cholesky reassembly requires unchanged Tet10 connectivity.");
    for (size_t element = 0; element < Factor->ElementNodes.size(); ++element)
        if (fem.Elements()[element].Nodes != Factor->ElementNodes[element])
            throw std::invalid_argument("Tet10 Cholesky reassembly requires unchanged Tet10 connectivity.");
    Assemble(fem, Factor->Pencil);
    Factor->HasNumeric = false;
}

void modal::Tet10Cholesky::ScalePencil(double stiffness_scale, double mass_scale) {
    if (!(std::isfinite(stiffness_scale) && stiffness_scale > 0 && std::isfinite(mass_scale) && mass_scale > 0))
        throw std::invalid_argument("Tet10 Cholesky pencil scales must be finite and positive.");
    if (stiffness_scale == 1 && mass_scale == 1) return;
    for (auto &block : Factor->Pencil.Stiffness)
        for (double &value : block) value *= stiffness_scale;
    for (double &value : Factor->Pencil.Mass) value *= mass_scale;
    Factor->HasNumeric = false;
}

void modal::Tet10Cholesky::Solve(const double *input, double *output, int width) const {
    if (width < 1) throw std::invalid_argument("Tet10 Cholesky solve width must be positive.");
    if (!Factor->HasNumeric) throw std::logic_error("Tet10 Cholesky solve requires numeric factorization.");
    Factor->Native.Solve(input, output, width);
}
