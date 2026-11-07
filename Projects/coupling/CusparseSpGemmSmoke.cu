#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

void checkCuda(cudaError_t status, const char* label)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", label, cudaGetErrorString(status));
        std::exit(1);
    }
}

void checkCusparse(cusparseStatus_t status, const char* label)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuSPARSE error %d\n", label, static_cast<int>(status));
        std::exit(1);
    }
}

template <class T>
void deviceAlloc(T*& ptr, std::size_t count, const char* label)
{
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), label);
}

template <class T>
void copyToDevice(T* dst, const T* src, std::size_t count, const char* label)
{
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), label);
}

template <class T>
void copyToHost(T* dst, const T* src, std::size_t count, const char* label)
{
    checkCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), label);
}

} // namespace

int main()
{
    constexpr int a_rows = 3;
    constexpr int a_cols = 3;
    constexpr int a_nnz = 5;
    constexpr int b_rows = 3;
    constexpr int b_cols = 2;
    constexpr int b_nnz = 4;
    const int h_a_offsets[] = {0, 2, 3, 5};
    const int h_a_cols[] = {0, 2, 1, 0, 2};
    const double h_a_vals[] = {2.0, -1.0, 4.0, 0.5, 3.0};
    const int h_b_offsets[] = {0, 1, 3, 4};
    const int h_b_cols[] = {0, 0, 1, 1};
    const double h_b_vals[] = {1.0, -2.0, 0.5, 3.0};

    const int h_c_offsets_ref[] = {0, 2, 4, 6};
    const int h_c_cols_ref[] = {0, 1, 0, 1, 0, 1};
    const double h_c_vals_ref[] = {2.0, -3.0, -8.0, 2.0, 0.5, 9.0};

    int* d_a_offsets = nullptr;
    int* d_a_cols = nullptr;
    double* d_a_vals = nullptr;
    int* d_b_offsets = nullptr;
    int* d_b_cols = nullptr;
    double* d_b_vals = nullptr;
    int* d_c_offsets = nullptr;
    int* d_c_cols = nullptr;
    double* d_c_vals = nullptr;
    void* buffer1 = nullptr;
    void* buffer2 = nullptr;

    deviceAlloc(d_a_offsets, a_rows + 1, "cudaMalloc a offsets");
    deviceAlloc(d_a_cols, a_nnz, "cudaMalloc a cols");
    deviceAlloc(d_a_vals, a_nnz, "cudaMalloc a vals");
    deviceAlloc(d_b_offsets, b_rows + 1, "cudaMalloc b offsets");
    deviceAlloc(d_b_cols, b_nnz, "cudaMalloc b cols");
    deviceAlloc(d_b_vals, b_nnz, "cudaMalloc b vals");
    deviceAlloc(d_c_offsets, a_rows + 1, "cudaMalloc c offsets");
    copyToDevice(d_a_offsets, h_a_offsets, a_rows + 1, "copy a offsets");
    copyToDevice(d_a_cols, h_a_cols, a_nnz, "copy a cols");
    copyToDevice(d_a_vals, h_a_vals, a_nnz, "copy a vals");
    copyToDevice(d_b_offsets, h_b_offsets, b_rows + 1, "copy b offsets");
    copyToDevice(d_b_cols, h_b_cols, b_nnz, "copy b cols");
    copyToDevice(d_b_vals, h_b_vals, b_nnz, "copy b vals");

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_a = nullptr;
    cusparseSpMatDescr_t mat_b = nullptr;
    cusparseSpMatDescr_t mat_c = nullptr;
    cusparseSpGEMMDescr_t desc = nullptr;
    checkCusparse(cusparseCreate(&handle), "cusparseCreate");
    checkCusparse(cusparseCreateCsr(&mat_a, a_rows, a_cols, a_nnz,
                      d_a_offsets, d_a_cols, d_a_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr A");
    checkCusparse(cusparseCreateCsr(&mat_b, b_rows, b_cols, b_nnz,
                      d_b_offsets, d_b_cols, d_b_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr B");
    checkCusparse(cusparseCreateCsr(&mat_c, a_rows, b_cols, 0,
                      d_c_offsets, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr C");

    const double alpha = 1.0;
    const double beta = 0.0;
    checkCusparse(cusparseSpGEMM_createDescr(&desc), "cusparseSpGEMM_createDescr");
    std::size_t buffer_size1 = 0;
    checkCusparse(cusparseSpGEMM_workEstimation(handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b,
                      &beta, mat_c, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, desc,
                      &buffer_size1, nullptr),
        "workEstimation size");
    checkCuda(cudaMalloc(&buffer1, buffer_size1), "cudaMalloc buffer1");
    checkCusparse(cusparseSpGEMM_workEstimation(handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b,
                      &beta, mat_c, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, desc,
                      &buffer_size1, buffer1),
        "workEstimation");

    std::size_t buffer_size2 = 0;
    checkCusparse(cusparseSpGEMM_compute(handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b,
                      &beta, mat_c, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, desc,
                      &buffer_size2, nullptr),
        "compute size");
    checkCuda(cudaMalloc(&buffer2, buffer_size2), "cudaMalloc buffer2");
    checkCusparse(cusparseSpGEMM_compute(handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b,
                      &beta, mat_c, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, desc,
                      &buffer_size2, buffer2),
        "compute");

    int64_t c_rows = 0;
    int64_t c_cols = 0;
    int64_t c_nnz = 0;
    checkCusparse(cusparseSpMatGetSize(mat_c, &c_rows, &c_cols, &c_nnz),
        "cusparseSpMatGetSize");
    if (c_rows != a_rows || c_cols != b_cols || c_nnz != 6) {
        std::fprintf(stderr, "unexpected C shape rows=%lld cols=%lld nnz=%lld\n",
            static_cast<long long>(c_rows), static_cast<long long>(c_cols),
            static_cast<long long>(c_nnz));
        return 1;
    }

    deviceAlloc(d_c_cols, static_cast<std::size_t>(c_nnz), "cudaMalloc c cols");
    deviceAlloc(d_c_vals, static_cast<std::size_t>(c_nnz), "cudaMalloc c vals");
    checkCusparse(cusparseCsrSetPointers(mat_c, d_c_offsets, d_c_cols, d_c_vals),
        "cusparseCsrSetPointers");
    checkCusparse(cusparseSpGEMM_copy(handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a, mat_b,
                      &beta, mat_c, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, desc),
        "copy");

    int h_c_offsets[4] = {};
    int h_c_cols[6] = {};
    double h_c_vals[6] = {};
    copyToHost(h_c_offsets, d_c_offsets, 4, "copy c offsets");
    copyToHost(h_c_cols, d_c_cols, 6, "copy c cols");
    copyToHost(h_c_vals, d_c_vals, 6, "copy c vals");

    for (int i = 0; i < 4; ++i) {
        if (h_c_offsets[i] != h_c_offsets_ref[i])
            return 1;
    }
    for (int i = 0; i < 6; ++i) {
        if (h_c_cols[i] != h_c_cols_ref[i] ||
            std::abs(h_c_vals[i] - h_c_vals_ref[i]) > 1e-12)
            return 1;
    }

    checkCusparse(cusparseSpGEMM_destroyDescr(desc), "destroy desc");
    checkCusparse(cusparseDestroySpMat(mat_a), "destroy A");
    checkCusparse(cusparseDestroySpMat(mat_b), "destroy B");
    checkCusparse(cusparseDestroySpMat(mat_c), "destroy C");
    checkCusparse(cusparseDestroy(handle), "destroy handle");
    checkCuda(cudaFree(buffer1), "free buffer1");
    checkCuda(cudaFree(buffer2), "free buffer2");
    checkCuda(cudaFree(d_a_offsets), "free a offsets");
    checkCuda(cudaFree(d_a_cols), "free a cols");
    checkCuda(cudaFree(d_a_vals), "free a vals");
    checkCuda(cudaFree(d_b_offsets), "free b offsets");
    checkCuda(cudaFree(d_b_cols), "free b cols");
    checkCuda(cudaFree(d_b_vals), "free b vals");
    checkCuda(cudaFree(d_c_offsets), "free c offsets");
    checkCuda(cudaFree(d_c_cols), "free c cols");
    checkCuda(cudaFree(d_c_vals), "free c vals");
    return 0;
}
