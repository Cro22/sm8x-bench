// cuBLAS fp16 GEMV baseline (the dense vendor ceiling: 2 bytes/weight).
//
// Computes y[M,N] = x[M,K] * W[N,K]^T with fp16 inputs and fp32 accumulation
// (CUBLAS_COMPUTE_32F), reading the SAME fp16 W/x bytes MAX consumed and
// validating against our fp32 reference. Prints the timing format the Python
// runners parse. At M=1 this is a GEMV; nsys tells us whether cuBLAS dispatches
// a gemv kernel or a GEMM tile (the runner records the kernel name).
//
// Row-major y[M,N] = x[M,K] * W[N,K]^T. In cuBLAS (column-major) this is one
// call with the stored buffers reinterpreted (row-major [a,b] == col-major
// [b,a]), computing C_cm[N,M] = (W_cm[K,N])^T * x_cm[K,M]:
//   cublasGemmEx(OP_T, OP_N, m=N, n=M, k=K, W(lda=K), x(ldb=K), y(ldc=N)).
//
// argv: N K M W_path x_path ref_path
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

static std::vector<uint8_t> read_bytes(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s N K M W_path x_path ref_path\n", argv[0]);
        return 1;
    }
    const int64_t N = atoll(argv[1]);
    const int64_t K = atoll(argv[2]);
    const int64_t M = atoll(argv[3]);

    std::vector<uint8_t> W_raw = read_bytes(argv[4]); // fp16 [N,K]
    std::vector<uint8_t> x_raw = read_bytes(argv[5]); // fp16 [M,K]
    std::vector<uint8_t> r_raw = read_bytes(argv[6]); // f32  [M,N]
    if (W_raw.size() != (size_t)N * K * 2) { fprintf(stderr, "W size mismatch\n"); return 1; }
    if (x_raw.size() != (size_t)M * K * 2) { fprintf(stderr, "x size mismatch\n"); return 1; }
    if (r_raw.size() != (size_t)M * N * 4) { fprintf(stderr, "ref size mismatch\n"); return 1; }
    const float * ref = reinterpret_cast<const float *>(r_raw.data());

    __half *dW, *dX; float *dY;
    cudaMalloc(&dW, (size_t)N * K * 2);
    cudaMalloc(&dX, (size_t)M * K * 2);
    cudaMalloc(&dY, (size_t)M * N * 4);
    cudaMemcpy(dW, W_raw.data(), W_raw.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x_raw.data(), x_raw.size(), cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasCreate(&h);
    // fp32 accumulation, no fp16 reduced-precision reduction.
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    const float alpha = 1.0f, beta = 0.0f;

    auto run = [&]() {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                     /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                     dW, CUDA_R_16F, /*lda=*/K,
                     dX, CUDA_R_16F, /*ldb=*/K, &beta,
                     dY, CUDA_R_32F, /*ldc=*/N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };

    // ---- Compute once + validate. ----
    run();
    cudaDeviceSynchronize();
    std::vector<float> host_y((size_t)M * N);
    cudaMemcpy(host_y.data(), dY, host_y.size() * 4, cudaMemcpyDeviceToHost);

    double ssd = 0.0, ssr = 0.0, max_abs = 0.0, max_rel = 0.0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double got = host_y[m * N + n]; // col-major C[N,M] == row-major y[M,N]
            double r   = ref[m * N + n];
            double ae  = std::abs(got - r);
            double re  = ae / (std::abs(r) + 1e-12);
            if (ae > max_abs) max_abs = ae;
            if (re > max_rel) max_rel = re;
            ssd += (got - r) * (got - r);
            ssr += r * r;
        }
    double l2_rel = (ssr > 0.0) ? (std::sqrt(ssd) / (std::sqrt(ssr) + 1e-12)) : std::sqrt(ssd);
    const double rtol = 3e-2;
    bool ok = l2_rel < rtol;

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device: %s\n", prop.name);
    printf("correctness: %s l2_rel_err= %g max_abs_err= %g max_rel_err= %g\n",
           ok ? "PASS" : "FAIL", l2_rel, max_abs, max_rel);
    if (!ok) { printf("aborting: kernel output L2 error %g >= tol %g\n", l2_rel, rtol); return 1; }

    // ---- Warmup + timed samples (12 x 10 back-to-back / 10). ----
    for (int i = 0; i < 10; ++i) run();
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    std::vector<double> samples;
    for (int s = 0; s < 12; ++s) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int j = 0; j < 10; ++j) run();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
        samples.push_back((double)ms * 1000.0 / 10.0); // us per launch
    }
    printf("launches_per_sample= %d\n", 10);
    printf("samples_us= ");
    for (size_t i = 0; i < samples.size(); ++i) { if (i) printf(","); printf("%g", samples[i]); }
    printf("\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cublasDestroy(h);
    cudaFree(dW); cudaFree(dX); cudaFree(dY);
    return 0;
}
