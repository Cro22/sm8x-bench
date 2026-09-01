# sm8x-bench

A rigorous benchmark workbench for LLM inference kernels written in Mojo, run on
consumer NVIDIA GPUs (RTX 3090 = sm_86, RTX 4090 = sm_89). It measures the
kernels that ship in Modular's open-source repo (`modular/max/kernels`) against
the best CUDA baselines (llama.cpp, FlashInfer, cuBLAS) and against the hardware
roofline, on hardware Modular does not tune for.

This is a workbench and a public record, **not** a competing kernel library.

See `CLAUDE.md` for the current phase and hard rules. Results tables live in
`reports/`; every number is backed by a JSON file in `bench/results/`.

Owner: Jesús ([Cro22](https://github.com/Cro22)). Prior work with the same
methodology: [mojo-cuda-ampere](https://github.com/Cro22/mojo-cuda-ampere).
