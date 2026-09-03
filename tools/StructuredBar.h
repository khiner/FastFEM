#pragma once

#include "mesh/TetMesh.h"

inline constexpr uint32_t HexTets[6][4]{
    {0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}
};

inline TetMesh MakeStructuredBar(int nx, int ny, int nz, dvec3 extent = {0.3, 0.05, 0.02}) {
    TetMesh mesh;
    const int sy = ny + 1, sz = nz + 1;
    const auto Node = [=](int x, int y, int z) { return uint32_t((x * sy + y) * sz + z); };
    mesh.Points.resize(size_t(nx + 1) * sy * sz);
    for (int x = 0; x <= nx; ++x)
        for (int y = 0; y <= ny; ++y)
            for (int z = 0; z <= nz; ++z)
                mesh.Points[Node(x, y, z)] = {
                    extent.x * x / nx, extent.y * y / ny, extent.z * z / nz
                };

    mesh.Tets.reserve(size_t(nx) * ny * nz * 6);
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const uint32_t nodes[8]{
                    Node(x, y, z), Node(x + 1, y, z), Node(x, y + 1, z), Node(x + 1, y + 1, z),
                    Node(x, y, z + 1), Node(x + 1, y, z + 1), Node(x, y + 1, z + 1), Node(x + 1, y + 1, z + 1)
                };
                for (const auto &tet : HexTets) mesh.Tets.push_back({nodes[tet[0]], nodes[tet[1]], nodes[tet[2]], nodes[tet[3]]});
            }
        }
    }
    return mesh;
}
