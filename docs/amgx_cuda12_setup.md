# CUDA 12 and AMGX setup

Use a persistent dependency directory beside this checkout:

```text
parent/
  DIQ-MPM/
  AMGX/
    build-ziran-cuda12/
  NVTX/
```

The examples below run from the repository root. They assume Linux, a working
NVIDIA driver, and system GCC/G++ 11 for CUDA 12.2's host compilation. Keep the
GPU environment separate from the CPU reference environment.

## Environment

The validated installation uses these compiler/library families. The commands
describe the dependencies; creating a new conda environment may resolve
different transitive package versions.

```bash
conda create -n ziran-cuda12 -c conda-forge \
  python=3.12 cmake ninja make git gcc_linux-64 gxx_linux-64 \
  eigen=5 tbb-devel=2023 openvdb=11 metis boost-cpp=1.84 \
  libboost-devel=1.84 zlib bzip2 opencl-headers ocl-icd

conda install -n ziran-cuda12 -c nvidia \
  cuda-nvcc=12.2.140 cuda-driver-dev=12.2 cuda-cudart-dev=12.2 \
  cuda-cudart-static=12.2.140 libcusparse-dev=12.2.0.103 \
  libcusparse=12.2.0.103 libnvjitlink=12.2.140 \
  libcublas-dev=12.2.5.6 libcusolver-dev=11.5.2.141 libcurand-dev=10.3.5.147
```

## Dependency versions

| Dependency | Version used |
|---|---|
| AMGX | `91a8413ef267b1c32aff4014c02820e1c5897ac2` |
| NVTX | `v3.2.2`, commit `4f37e799ccc6b70c39979b53a4c24488ee3fe0ae` |

Clone into persistent directories. Do not overwrite an existing installation:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/AMGX.git ../AMGX
git -C ../AMGX checkout 91a8413ef267b1c32aff4014c02820e1c5897ac2
git -C ../AMGX submodule update --init --recursive
git clone --branch v3.2.2 --depth 1 https://github.com/NVIDIA/NVTX.git ../NVTX
```

## Build AMGX

```bash
conda activate ziran-cuda12

cmake -S ../AMGX -B ../AMGX/build-ziran-cuda12 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_FLAGS="-I$PWD/../NVTX/c/include"

conda run -n ziran-cuda12 cmake --build ../AMGX/build-ziran-cuda12 \
  --target amgxsh -j 4
```

The result is `../AMGX/build-ziran-cuda12/libamgxsh.so`.
Use matching CUDA architectures for AMGX and this project. Architecture 89 is
the validated RTX 4080 configuration.

## Connect to DIQ-MPM

Follow the [root build instructions](../README.md#build-the-gpu-version).
For another dependency layout, pass absolute `AMGX_ROOT` and `AMGX_LIBRARY`
paths explicitly. The headers and library must come from the same AMGX build.
Keep that library available when running simulations.

Existing working installations can be reused by a separate checkout/worktree;
each checkout should have its own CMake build directory.
