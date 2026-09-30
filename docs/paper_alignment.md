# Paper–implementation map

| Paper component | Implementation |
| --- | --- |
| Visual artifact collection, coordinate ancestry | `vista/environment.py`, `vista/crop.py` |
| Deduplication, provenance, categorical hindsight | `vista/experience.py`, `vista/hindsight.py` |
| Decision-conditioned selection, ancestor closure, budget 4 | `ExperienceViewProjector` in `vista/experience.py` |
| Identical sampled response prefixes; teacher-only evidence | `vista/teacher.py`, `RayPPOTrainer._prepare_collective_experience_teacher_inputs` |
| HAPI and crossed routing controls | `vista/supervision.py` |
| Per-input, full-group token denominators | `vista/objectives.py`, `RayPPOTrainer.fit` |
| Clipped policy objective; full-group advantages before gating | `verl/trainer/ppo/core_algos.py` |
| Exact full-vocabulary JSD and KL controls | `vista/objectives.py`, `verl/workers/actor/dp_actor.py` |
| One update per rollout batch, followed by EMA | `DataParallelPPOActor.update_policy`, `vista/ema.py` |
| Training defaults | `verl/trainer/config/ppo_trainer.yaml` |
| Seeds and run records | `vista/reproducibility.py`, `scripts/train_three_seeds.sh` |

## Reduction across turns, microbatches and workers

For Q input questions, let N_x count all generated response tokens for input x,
and M_x count tokens eligible for distillation under the selected learning rule.
Every response row carries policy weight 1/(Q N_x) and distillation weight
1/(Q M_x), with zero weight for empty subsets. HAPI sets unsuccessful policy
advantages to zero only after estimating full-group statistics. The policy
denominator therefore includes unsuccessful trajectories. Distillation gates
select the numerator and denominator together.

Weights are computed before reordering, padding or distributing rows. Added
padding copies have zero weight and preserve trajectory IDs, so they neither
change objective contributions nor add independent samples to the advantage
baseline. Local microbatch contributions are summed; multiplying by the data
parallel world size cancels FSDP's gradient average. A complete rollout batch
has one optimizer step and one EMA update. Zero-response input groups still
count in Q. An entirely empty rollout batch raises an error.

Full-vocabulary JSD computes distributions over every token. It uses token
chunks with recomputation for probability tensors, detached teacher logits,
and float32 arithmetic (float64 for numerical tests). The unpadded actor gathers
response logits directly, avoiding a padded prompt-by-vocabulary tensor.
Student and teacher model logits themselves still consume substantial memory.

## Explicit implementation choices and limits

The manuscript does not give the precise formula for relative advantage A_k.
The default now uses standard GRPO: mean subtraction and sample-standard-deviation
scaling over distinct trajectories, before HAPI masking. The earlier release
used mean subtraction only. Set `algorithm.norm_adv_by_std_in_grpo=false` to
recover that alternative; it is not treated as a verified historical setting.
A single-trajectory group receives zero relative advantage.

Default rewards are binary correctness with protocol validity. Optional legacy
crop/invalid-action shaping remains available but is disabled. The clipping
objective has no dual-clip extension by default. Seed values 0, 1, 2 are provided
as reproducible choices, not asserted to be the unpublished experimental seeds.

This release implements the main VISTA method, learning-rule masks, lambda,
artifact-budget, divergence and frozen-teacher controls. It does not bundle
training data, checkpoints, regional annotations, other methods' implementations,
or scripts reproducing every paper table. CUDA training, exact-vocabulary memory
usage and reported scores have not been revalidated by the CPU suite. The supplied
verifier normalizes boxed answers and compares them; new datasets must provide
compatible answer labels or their own verified scoring protocol.
