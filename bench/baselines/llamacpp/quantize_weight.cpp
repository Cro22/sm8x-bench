// Quantize an fp32 weight [N, K] (row-major) to a GGUF block format using ggml's
// own quantizer, and write the raw block bytes. Needed for formats the pure-
// Python gguf-py quantizer does not implement (e.g. Q4_K); gguf-py CAN dequantize
// them, so bench/reference.py quantizes here and dequantizes in Python for the
// fp32 reference. Q4_0/Q8_0 are supported here too for symmetry.
//
// argv: fmt N K in_f32_path out_bytes_path      (fmt in {Q4_0,Q8_0,Q4_K})
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "ggml.h"

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
    if (argc != 6) {
        fprintf(stderr, "usage: %s fmt N K in_f32 out_bytes\n", argv[0]);
        return 1;
    }
    const char * fmt = argv[1];
    const int64_t N = atoll(argv[2]);
    const int64_t K = atoll(argv[3]);
    const char * in_path  = argv[4];
    const char * out_path = argv[5];

    ggml_type t;
    if      (!strcmp(fmt, "Q4_0")) t = GGML_TYPE_Q4_0;
    else if (!strcmp(fmt, "Q8_0")) t = GGML_TYPE_Q8_0;
    else if (!strcmp(fmt, "Q4_K")) t = GGML_TYPE_Q4_K;
    else { fprintf(stderr, "unsupported fmt %s\n", fmt); return 1; }

    std::vector<uint8_t> raw = read_bytes(in_path);
    if (raw.size() != (size_t)N * K * 4) {
        fprintf(stderr, "in size %zu != N*K*4 = %zu\n", raw.size(), (size_t)N * K * 4);
        return 1;
    }
    const float * src = reinterpret_cast<const float *>(raw.data());

    // Worst-case output size; ggml_quantize_chunk returns the exact byte count.
    std::vector<uint8_t> dst((size_t)N * K * 4);
    size_t nbytes = ggml_quantize_chunk(t, src, dst.data(),
                                        /*start=*/0, /*nrows=*/N, /*n_per_row=*/K,
                                        /*imatrix=*/nullptr);
    if (nbytes == 0) { fprintf(stderr, "ggml_quantize_chunk returned 0\n"); return 1; }

    FILE * f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    if (fwrite(dst.data(), 1, nbytes, f) != nbytes) { fprintf(stderr, "short write\n"); return 1; }
    fclose(f);
    printf("wrote %zu bytes (%s, N=%lld K=%lld)\n", nbytes, fmt,
           (long long)N, (long long)K);
    return 0;
}
