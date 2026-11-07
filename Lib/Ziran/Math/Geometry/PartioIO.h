#ifndef PARTIO_IO_H
#define PARTIO_IO_H

#include <Partio.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Geometry/Particles.h>

namespace ZIRAN {

template <class T, int dim>
void writePartio(const std::string& particleFile,
    const Particles<T, dim>& particles)
{
    // AttributeName<T> mass = Particles<T, dim>::mass_name();
    AttributeName<Vector<T, dim>> x_name = Particles<T, dim>::X_name();
    // AttributeName<Vector<T, dim>> v_name = Particles<T, dim>::V_name();

    Partio::ParticlesDataMutable* parts = Partio::create();
    Partio::ParticleAttribute posH, vH, mH;
    // mH = parts->addAttribute("m", Partio::VECTOR, 1);
    posH = parts->addAttribute("position", Partio::VECTOR, 3);
    // vH = parts->addAttribute("v", Partio::VECTOR, 3);
    // auto iter = particles.iter(mass, x_name, v_name);
    auto iter = particles.iter(x_name);
    for (; iter; ++iter) {
        int idx = parts->addParticle();
        // float* m = parts->dataWrite<float>(mH, idx);
        float* p = parts->dataWrite<float>(posH, idx);
        // float* v = parts->dataWrite<float>(vH, idx);

        // T mass = iter.template get<0>();
        // m[0] = mass;

        Vector<T, dim> X = iter.template get<0>();
        // Vector<T, dim> V = iter.template get<2>();

        for (int k = 0; k < 3; k++)
            p[k] = (T)0;
        // for (int k = 0; k < 3; k++)
        //    v[k] = (T)0;

        for (int k = 0; k < dim; k++)
            p[k] = X(k);
        // for (int k = 0; k < dim; k++)
        //    v[k] = V(k);
    }

    Partio::write(particleFile.c_str(), *parts);
    parts->release();
}

} // namespace ZIRAN
#endif
