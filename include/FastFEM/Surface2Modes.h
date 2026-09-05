#pragma once

#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fastfem {
struct SolveMonitor;

struct UVec3 {
    using value_type = uint32_t;
    static constexpr size_t ComponentCount = 3;
    uint32_t x{}, y{}, z{};
    constexpr UVec3() = default;
    constexpr UVec3(uint32_t v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr UVec3(X x, Y y, Z z) : x(uint32_t(x)), y(uint32_t(y)), z(uint32_t(z)) {}
    constexpr uint32_t &operator[](size_t i) { return (&x)[i]; }
    constexpr const uint32_t &operator[](size_t i) const { return (&x)[i]; }
    friend constexpr bool operator==(UVec3, UVec3) = default;
};

struct Vec3 {
    using value_type = float;
    static constexpr size_t ComponentCount = 3;
    float x{}, y{}, z{};
    constexpr Vec3() = default;
    constexpr Vec3(float v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr Vec3(X x, Y y, Z z) : x(float(x)), y(float(y)), z(float(z)) {}
    template<typename V> constexpr Vec3(const V &xy, float z)
        requires requires { xy.x; xy.y; }
        : x(xy.x), y(xy.y), z(z) {}
    template<typename V> constexpr explicit Vec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
    friend constexpr bool operator==(Vec3, Vec3) = default;
};

struct DVec3 {
    using value_type = double;
    static constexpr size_t ComponentCount = 3;
    double x{}, y{}, z{};
    constexpr DVec3() = default;
    constexpr DVec3(double v) : x(v), y(v), z(v) {}
    constexpr DVec3(double x, double y, double z) : x(x), y(y), z(z) {}
    template<typename V> constexpr explicit DVec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr double &operator[](size_t i) { return (&x)[i]; }
    constexpr const double &operator[](size_t i) const { return (&x)[i]; }
    friend constexpr bool operator==(DVec3, DVec3) = default;
};

struct Quat {
    float x{}, y{}, z{}, w{1};
    constexpr Quat() = default;
    constexpr Quat(float real, float imag_x, float imag_y, float imag_z) : x(imag_x), y(imag_y), z(imag_z), w(real) {}
    constexpr Quat(float real, Vec3 imaginary) : x(imaginary.x), y(imaginary.y), z(imaginary.z), w(real) {}
    template<typename V> constexpr explicit Quat(const V &xyzw)
        requires requires { xyzw.x; xyzw.y; xyzw.z; xyzw.w; }
        : x(xyzw.x), y(xyzw.y), z(xyzw.z), w(xyzw.w) {}
    explicit Quat(Vec3 euler_xyz) {
        const Vec3 cosine{std::cos(euler_xyz.x * .5f), std::cos(euler_xyz.y * .5f), std::cos(euler_xyz.z * .5f)};
        const Vec3 sine{std::sin(euler_xyz.x * .5f), std::sin(euler_xyz.y * .5f), std::sin(euler_xyz.z * .5f)};
        w = cosine.x * cosine.y * cosine.z + sine.x * sine.y * sine.z;
        x = sine.x * cosine.y * cosine.z - cosine.x * sine.y * sine.z;
        y = cosine.x * sine.y * cosine.z + sine.x * cosine.y * sine.z;
        z = cosine.x * cosine.y * sine.z - sine.x * sine.y * cosine.z;
    }
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
    friend constexpr bool operator==(Quat, Quat) = default;
};

struct AcousticMaterialProperties {
    double Density{}, YoungModulus{}, PoissonRatio{};
    double Alpha{}, Beta{};
    double Lambda() const { return (PoissonRatio * YoungModulus) / ((1 + PoissonRatio) * (1 - 2 * PoissonRatio)); }
    double Mu() const { return YoungModulus / (2 * (1 + PoissonRatio)); }
    auto operator<=>(const AcousticMaterialProperties &) const = default;
};

enum struct Discretization { Tet10,
                             FiniteCell };

struct SolverConfig {
    float MinModeFreq{20};
    float MaxModeFreq{16'000};
    uint32_t NumModes{30};
    uint32_t NumFemModes{45};
    double Tolerance{1e-8};
    uint32_t MaxRestarts{100};
    std::optional<float> FundamentalFreq{};
};

enum class TetRefinement { None,
                           Quality,
                           QualityAndResolution };

struct TetrahedralizationConfig {
    TetRefinement Refinement{TetRefinement::None};
    std::vector<DVec3> Holes;
};

struct FiniteCellConfig {
    uint32_t CutDepth{3};
    double FictitiousScale{1e-8};
    double PaddingCells{0.25};
    DVec3 GridOffsetCells{};
};

struct SurfaceSolveConfig {
    SolverConfig Modal{};
    TetrahedralizationConfig Tetrahedralization{};
    FiniteCellConfig FiniteCell{};
    // Divisions along the longest input axis. Sets finite-cell spacing and, for
    // QualityAndResolution, Tet10 surface edge lengths and tetrahedron volume targets.
    uint32_t Resolution{12};
    float SurfaceSimplificationRatio{1};
};

struct ModalModes {
    std::vector<float> Freqs;
    std::vector<float> T60s;
    std::vector<std::vector<Vec3>> Shapes;
    std::vector<Vec3> Positions;
    float OriginalFundamentalFreq{Freqs.empty() ? 0 : Freqs.front()};
    Vec3 BakedScale{1, 1, 1};
    bool operator==(const ModalModes &) const = default;
};

struct MassProperties {
    double Mass{};
    Vec3 CenterOfMass{};
    Vec3 InertiaDiagonal{};
    Quat InertiaOrientation{};
    bool operator==(const MassProperties &) const = default;
};

struct ModalEigenSummary {
    std::vector<double> Eigenvalues;
    std::vector<std::vector<Vec3>> Shapes;
    AcousticMaterialProperties SolvedMaterial{};
    bool operator==(const ModalEigenSummary &) const = default;
};

struct TetMesh {
    std::vector<DVec3> Points;
    std::vector<std::array<uint32_t, 4>> Tets;
    bool operator==(const TetMesh &) const = default;
};

struct ModeBasis {
    struct Storage;
    std::shared_ptr<const Storage> Data;

    explicit operator bool() const { return bool(Data); }
};

struct SolveReuse {
    const ModeBasis *SeedBasis{};
    bool KeepBasis{};
};

struct ModalResult {
    ModalModes Modes;
    MassProperties Mass;
    ModalEigenSummary Summary;
    ModeBasis Basis;
    std::vector<uint32_t> SamplePointOfExcitation;
    TetMesh Tetrahedra;
};

std::expected<ModalResult, std::string> Surface2Modes(
    std::span<const Vec3> positions, std::span<const uint32_t> triangle_indices,
    const AcousticMaterialProperties &, std::span<const Vec3> excitation_positions,
    Vec3 baked_scale, Discretization, SurfaceSolveConfig = {}, SolveReuse = {}, SolveMonitor * = nullptr
);

std::optional<ModalModes> RescaleModes(
    const ModalEigenSummary &, const ModalModes &current,
    const AcousticMaterialProperties &, SolverConfig = {}
);
} // namespace fastfem
