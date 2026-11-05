# MI355X validation of vllm-project/vime#513 and #514

Data for RFC vllm-project/vime#515. The runs follow `docs/en/platform_support/amd_tutorial.md`: Qwen3-8B GRPO on dapo-math-17k, 2 GPUs for colocate and 4 GPUs for the async run. The only change to the scripts is that the wandb arguments are removed.

| Run | Script | Image | Steps |
|---|---|---|---|
| colocate-100 | `run-qwen3-8B-amd.sh` as documented (`--no-offload-train`) | #514 at `18de560a` | 100 |
| async-100 | `run-qwen3-8B-async-amd.sh` (disaggregated) | #514 at `18de560a` | 100 |
| pr3v2-offload-20 | `run-qwen3-8B-amd.sh` of #514 (`--offload-train`, train-side `NCCL_CUMEM_ENABLE=1`) | #514 at `e6d4de3f` | 20 |

The `18de560a` and `e6d4de3f` images have the same patched vLLM tree. Neither 100-step run uses train offload, which is the only part that `e6d4de3f` changes.

- `vime-rocm-tutorial-curves.png`: per-step `rollout/raw_reward` (with a 10-step mean), `rollout/truncated`, `train/train_rollout_logprob_abs_diff`, and `train/entropy_loss`.
- `vime-rocm-tutorial-curves.csv`: the same data, one row per run and step.
