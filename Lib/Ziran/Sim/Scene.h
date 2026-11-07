#ifndef SCENE_H
#define SCENE_H

#include <Ziran/CS/Util/DataDir.h>
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Math/Geometry/ObjIO.h>
#include <Ziran/Math/Geometry/SimplexMesh.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Math/Geometry/PolyIO.h>

namespace ZIRAN {
/**
       scene needs:

       particles
       dirichlet_position_dofs
       segmesh_to_write
       trimesh_to_write
       element_managers
     */

template <class T, int dim>
class Scene {
public:
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    Particles<T, dim> particles;
    StdVector<int> dirichlet_position_dofs;
    StdVector<std::unique_ptr<ElementManager<T, dim>>> element_managers;
    SimplexMesh<2> trimesh_to_write;
    SimplexMesh<1> segmesh_to_write;

    Scene()
    {
        particles.X.registerReorderingCallback(
            [&](const DataArrayBase::ValueIDToNewValueID& old_to_new) {
                for (auto& element : trimesh_to_write.indices)
                    for (int d = 0; d < 3; d++)
                        element(d) = old_to_new(element(d));

                for (auto& element : segmesh_to_write.indices)
                    for (int d = 0; d < 2; d++)
                        element(d) = old_to_new(element(d));

                for (auto& dirichlet_dof : dirichlet_position_dofs) {
                    int p = dirichlet_dof / dim;
                    int off = dirichlet_dof % dim;
                    dirichlet_dof = dim * old_to_new(p) + off;
                }
            });
    }

    /* binary ver 1 */
    void writeState(std::ostream& out, const std::string& boundary_filename, const std::string& segment_filename, DataDir& output_dir, int binary_ver, bool write_meshes = true)
    {
        if (write_meshes && (trimesh_to_write.indices.size() != 0)) {
            std::ofstream boundary_file = output_dir.openTextOutput(boundary_filename);
            writePositionObj(boundary_file, particles.X.array);
            writeFacesObj(boundary_file, trimesh_to_write.indices);
        }
        if (write_meshes && segmesh_to_write.indices.size() != 0) {
            std::ofstream segmesh_file = output_dir.openTextOutput(segment_filename);
            writeSegmeshPoly(segmesh_file, particles.X.array, segmesh_to_write.indices);
        }
        particles.writeData(out);
        for (const auto& em : element_managers)
            em->writeData(out);

        writeSTDVector(out, trimesh_to_write.indices);
        writeSTDVector(out, segmesh_to_write.indices);
    }

    /* binary ver 1 */
    void readState(std::istream& in, int binary_ver)
    {
        particles.readData(in);
        for (auto& em : element_managers)
            em->readData(in);

        readSTDVector(in, trimesh_to_write.indices);
        readSTDVector(in, segmesh_to_write.indices);
    }
};
} // namespace ZIRAN

#endif
