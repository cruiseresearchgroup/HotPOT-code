# Reproducing the experiments

## Pre-training

Use the README's `torchrun` command. The fixed recipe is two ranks, 20 samples per rank, 545 optimizer updates per epoch, and 1,000 epochs. The six storage entries use the same sample counts and unit weights as the matched MoE-POT protocol. Each sample contains 10 history frames and a one-frame target. The phase-aware sampler chooses history windows; training is not a multi-step rollout loss.

Adam uses a maximum learning rate of 1e-3, betas (0.9, 0.9), and weight decay 1e-6. OneCycle uses a 20% warm-up and initial/final division factors of 1e4. Noise amplitude is 5e-4 outside CFDBench; CFDBench inputs remain clean. The source seed is 20260815. `configs/pretrain.json` records the recipe. Use `--smoke-updates 2` to test the distributed training path without a full run; this is not a trained checkpoint.

Use `--resume workspace/runs/full17/checkpoint_eNNNN.pt` with the original run's output directory to resume. The trainer restores the optimizer and schedule and checks its recorded protocol.

## Evaluation

Use the fixed E1000 source checkpoint. `hotpot.evaluate` reports a mean and all trajectory scores at the requested setting's horizons. It rolls forward using the model's own predictions. The scored horizons are 10 for NS 1e-5 and CFDBench, 20 for NS 1e-3, 11 for CNS, and 91 for SWE/DR. The evaluation computes relative error over the horizon using the released grid/channel mask; this differs from averaging independent per-step errors.

For CFDBench, add `--cfdb-test-sidecar workspace/metadata/cfdb_test.json`. Use `--target` when evaluating target-adapted full model states: source NS execution uses four substeps, while the target protocol uses eight.

## Known-task fine-tuning

Use `hotpot.transfer --mode known-train` as in the README. The protocol uses 200 epochs, a fixed final checkpoint, target seed 0, learning rate 1e-4, global batch 40, and microbatch 20. It trains the shared encoder and the target's private branch. One invocation uses one GPU with gradient accumulation. Other branches remain frozen.

Evaluate in a separate process:

```bash
python -m hotpot.transfer \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --target-checkpoint ./workspace/runs/ns_finetuned/target_full_state.pt \
  --mode known-test --task ns2d_fno_1e-5 \
  --output ./workspace/results/ns_finetuned
```

## Few-shot coefficient adaptation

```bash
python -m hotpot.transfer \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --mode fewshot-train --task fno_1e-4 --shots 32 \
  --arm transfer --selection original-val \
  --output ./workspace/runs/ns_32shot
```

Use `cns_m1_eta1e-2` for CNS. Shot counts are 16, 32, 64, and 128. Training runs for 500 epochs, with batch size `min(20, shots)`, Adam maximum learning rate 1e-4, and 20% OneCycle warm-up. Validation is performed every 10 epochs on 256 fixed train-origin row/window addresses. The minimum validation error selects a **full state, including buffers**. Test with the corresponding `fewshot-test` mode and `--target-checkpoint`; the saved validation error is reconstructed before test evaluation. Set `--arm scratch` to train the same target architecture without loading pretrained weights.

## New-family Wave transfer

```bash
python -m hotpot.transfer \
  --workspace ./workspace --paths configs/data_paths.json \
  --checkpoint ./workspace/runs/full17/checkpoint_e1000.pt --scale 17 \
  --mode wave-train --task Wave-Gauss --shots 32 \
  --arm transfer --selection original-val \
  --output ./workspace/runs/wave_gauss_32shot
```

Only the pretrained history encoder is copied. The Wave-private modules are freshly initialized, and all target parameters can train. Use `Wave-Layer` for the other Wave dataset and `--arm scratch` for the matched randomly initialized control. Train/test use the bundled split and file-identity checks. Evaluate with `wave-test` and the saved `--target-checkpoint`, using the same source checkpoint, task, shot count, and selection setting.

Wave validation/test metrics average relative L2 error across trajectories **and forecast steps**. Wave uses an observation-space residual predictor because the required physical coefficients and boundary metadata are not supplied. Array padding is not a claim about the physical boundary condition.

## CFDBench condition transfer

`--mode zero-shot --task cfdb_cavity`, `cfdb_cylinder`, or `cfdb_tube` evaluates the corresponding family with the fixed pretrained model. The condition-space audit uses the bundled case descriptions and separates boundary, parameter, and geometry changes. It reports both trajectory scores and the case-uniform aggregate.

## Baseline and compute

Use the [official MoE-POT implementation](https://github.com/haiyangxin/MoEPOT) for the baseline. Match dataset rows, checkpoint-selection rules, activated-parameter scale, and evaluation masks; comparing total stored HotPOT parameters to active MoE-POT parameters would use different definitions. The repository includes the HotPOT execution paths, not baseline training logs or pretrained weights.

The source release was smoke-tested on CPU. Full pre-training, transfer, and the paper's reported values require the datasets, the stated GPU training budgets, and their resulting checkpoints. No new full training experiment was run as part of packaging this release.
