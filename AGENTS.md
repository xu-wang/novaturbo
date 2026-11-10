# NovaTurbo project instructions

@/home/wangjianyu/.codex/RTK.md

## Default GPU

- For every CUDA-capable training, inference, evaluation, benchmark, and GPU smoke test, use physical GPU index `6` by default.
- Set `CUDA_VISIBLE_DEVICES=6` in the same command or persistent execution session before starting the Python process.
- Because `CUDA_VISIBLE_DEVICES` remaps visible devices, application code should normally select `cuda:0`, which corresponds to physical GPU `6`; do not select `cuda:6` inside that restricted process.
- Before a material GPU run, verify physical GPU `6` with `nvidia-smi -i 6` and verify that PyTorch reports CUDA available in the actual execution context.
- The default Codex workspace sandbox may omit `/dev/nvidia*`. Run GPU workloads outside that sandbox with the required approval instead of silently falling back to CPU.
- Deviate from GPU `6` only when the user explicitly requests another device or GPU `6` is unavailable; report any deviation.
