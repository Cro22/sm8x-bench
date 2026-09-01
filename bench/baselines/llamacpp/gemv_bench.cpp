// llama.cpp/ggml Q4_0 GEMV baseline benchmark.
//
// Times ggml_mul_mat with a Q4_0 weight and an F32 activation at our canonical
// decode shapes, reading the SAME Q4_0 weight bytes MAX consumed, validating
// against our fp32 reference, and printing timing in the format bench/mojo/
// gemv_max.mojo emits so the Python runner parses both identically.
//
// ggml_mul_mat(a, b): a = weight [ne0=K, ne1=N] (Q4_0), b = [ne0=K, ne1=M]
// (F32), result = [ne0=N, ne1=M]. Our raw W bytes are row-major N x (K/32*18),
// which is exactly a ggml Q4_0 tensor of ne0=K, ne1=N -> single memcpy.
//
// argv: N K M W_path x_path ref_path
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>

static std::vector<uint8_t> read_bytes(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(n);
    if (fread(buf.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s N K M W_path x_path ref_path\n", argv[0]);
        return 1;
    }
    const int64_t N = atoll(argv[1]);
    const int64_t K = atoll(argv[2]);
    const int64_t M = atoll(argv[3]);
    const char * W_path   = argv[4];
    const char * x_path   = argv[5];
    const char * ref_path = argv[6];

    // ---- Load inputs. ----
    std::vector<uint8_t> W_bytes = read_bytes(W_path);
    const size_t expect_W = (size_t)N * (K / 32) * 18;
    if (W_bytes.size() != expect_W) {
        fprintf(stderr, "W size %zu != expected %zu (N*K/32*18)\n", W_bytes.size(), expect_W);
        return 1;
    }
    std::vector<uint8_t> x_raw = read_bytes(x_path); // bf16 [M,K]
    if (x_raw.size() != (size_t)M * K * 2) {
        fprintf(stderr, "x size %zu != expected %zu (M*K*2)\n", x_raw.size(), (size_t)M * K * 2);
        return 1;
    }
    std::vector<float> x(M * K);
    const uint16_t * xb = reinterpret_cast<const uint16_t *>(x_raw.data());
    for (int64_t i = 0; i < M * K; ++i) x[i] = bf16_to_f32(xb[i]);

    std::vector<uint8_t> ref_raw = read_bytes(ref_path); // f32 [M,N]
    if (ref_raw.size() != (size_t)M * N * 4) {
        fprintf(stderr, "ref size %zu != expected %zu (M*N*4)\n", ref_raw.size(), (size_t)M * N * 4);
        return 1;
    }
    const float * ref = reinterpret_cast<const float *>(ref_raw.data());

    // ---- Backend + context. ----
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { fprintf(stderr, "ggml_backend_cuda_init(0) failed\n"); return 1; }

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /* .mem_base   = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N); // [ne0=K, ne1=N]
    ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // [ne0=K, ne1=M]
    ggml_set_name(W, "W");
    ggml_set_name(x_t, "x");

    // Create the op tensor (out) BEFORE allocating, so its result buffer is
    // allocated on the backend too. out = mul_mat(W, x) -> [ne0=N, ne1=M].
    ggml_tensor * out = ggml_mul_mat(ctx, W, x_t);
    ggml_set_name(out, "out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "failed to allocate tensors\n"); return 1; }

    if (ggml_nbytes(W) != W_bytes.size()) {
        fprintf(stderr, "ggml W nbytes %zu != file %zu\n", ggml_nbytes(W), W_bytes.size());
        return 1;
    }
    ggml_backend_tensor_set(W, W_bytes.data(), 0, W_bytes.size());
    ggml_backend_tensor_set(x_t, x.data(), 0, x.size() * sizeof(float));

    // ---- Graph. ----
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // ---- Compute once + validate. ----
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n"); return 1;
    }
    std::vector<float> host_out((size_t)N * M);
    ggml_backend_tensor_get(out, host_out.data(), 0, host_out.size() * sizeof(float));

    double ssd = 0.0, ssr = 0.0, max_abs = 0.0, max_rel = 0.0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double got = host_out[m * N + n]; // out is [ne0=N, ne1=M]
            double r   = ref[m * N + n];
            double ae  = std::abs(got - r);
            double re  = ae / (std::abs(r) + 1e-12);
            if (ae > max_abs) max_abs = ae;
            if (re > max_rel) max_rel = re;
            double d = got - r;
            ssd += d * d;
            ssr += r * r;
        }
    }
    double l2_rel = (ssr > 0.0) ? (std::sqrt(ssd) / (std::sqrt(ssr) + 1e-12)) : std::sqrt(ssd);
    const double rtol = 3e-2;
    bool ok = l2_rel < rtol;

    const char * dev_desc = ggml_backend_dev_description(ggml_backend_get_device(backend));
    printf("device: %s\n", dev_desc ? dev_desc : "CUDA0");
    printf("correctness: %s l2_rel_err= %g max_abs_err= %g max_rel_err= %g\n",
           ok ? "PASS" : "FAIL", l2_rel, max_abs, max_rel);
    if (!ok) {
        printf("aborting: kernel output L2 error %g >= tol %g\n", l2_rel, rtol);
        return 1;
    }

    // ---- Warmup. ----
    const int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // ---- Timed samples: 12 samples, each = 10 back-to-back computes / 10. ----
    const int SAMPLES = 12;
    const int PER_BATCH = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> samples;
    for (int s = 0; s < SAMPLES; ++s) {
        ggml_backend_synchronize(backend);
        cudaEventRecord(start);
        for (int j = 0; j < PER_BATCH; ++j) ggml_backend_graph_compute(backend, gf);
        cudaEventRecord(stop);
        ggml_backend_synchronize(backend);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back((double)ms * 1000.0 / PER_BATCH); // us per launch
    }

    printf("launches_per_sample= %d\n", PER_BATCH);
    printf("samples_us= ");
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i > 0) printf(",");
        printf("%g", samples[i]);
    }
    printf("\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    return 0;
}
