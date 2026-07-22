# Stage B v2 Experiment Plan

Stage B v2 is a new experiment from the promoted Stage A checkpoint. It does
not resume or modify Stage B v1. The v1 artifacts remain the comparison
baseline, and LoRA remains a planned comparison rather than being rejected.

## Objective

Improve medical validation loss substantially while preserving useful general
language ability. Stage B v1 proved that unrestricted full-parameter continual
pretraining can improve medical loss but also showed severe catastrophic
forgetting. V2 changes the data mixture, optimization strength, trainable
parameter set, regularization, validation cadence, and promotion rule.

## Verified v2 data

- Medical/general mixture: 70.0134% / 29.9866% by exact stream tokens.
- Packed sequences: 1,028,134 at 256 targets per sequence.
- Supervised targets: 263,202,304.
- Binary shards: 126, all hash and size verified.
- Stage A document overlap: zero.
- Medical and general validation/test overlap: zero.
- Stage A replay: unnecessary because the remaining licensed general sources
  contain enough unused text for the 30% rehearsal allocation.
- Planned full-run updates: 8,033 at 128 sequences (32,768 tokens) per update.

See `DATASET_SPECIFICATION.md`, `source_audit.json`, and
`corpus_verification.json` in this directory for the exact evidence.

## Adaptation method

The main v2 candidate uses all of the following:

1. Start from Stage A `checkpoint_00007250`, not from Stage B v1.
2. Freeze the tied token embedding/output-head weight.
3. Freeze decoder blocks 0, 1, and 2.
4. Train blocks 3-7 and the final RMSNorm.
5. Use a peak learning rate of `4e-5`, below v1's `1e-4`.
6. Apply normalized L2-SP regularization to trainable parameters, anchored to
   their exact Stage A values.
7. Validate every 50 updates in pilots and every 100 updates in a full run.

Freezing and L2-SP solve different problems: freezing protects entire lower
representations exactly, while L2-SP discourages excessive movement in the
upper trainable layers. LoRA will later provide a third, parameter-efficient
adaptation family for comparison.

## Controlled pilot matrix

Run three 500-update pilots with identical data, initialization, seed, batch
size, scheduler, and validation sets:

| Arm | Trainable parameters | L2-SP | Config |
|---|---|---:|---|
| Control | All | 0 | `training_stage_b_v2_control.yaml` |
| Selective | Blocks 3-7 + final norm | 0 | `training_stage_b_v2_selective.yaml` |
| Selective + L2-SP | Blocks 3-7 + final norm | `1e6` pilot value | `training_stage_b_v2_selective_l2sp.yaml` |

The L2-SP penalty is the mean squared parameter displacement. Therefore its
coefficient has a different numerical scale from an unnormalized sum penalty.
`1e6` is a calibration candidate, not a value to accept without pilot metrics.

## Retention and safety gates

General retention is measured by perplexity relative to the unchanged Stage A
baseline, because a percentage change in cross-entropy loss is not equivalent
to the same percentage change in perplexity.

- Preferred band: no more than 20% general perplexity degradation.
- Promotion hard cap: no more than 25% degradation.
- Emergency band: more than 35% degradation.
- Automatic stop: two consecutive emergency-band validations.
- A promotable checkpoint must also improve medical loss over the Stage A
  medical baseline.

The trainer records separate `best_preferred`, `best_eligible`, `best_medical`,
`latest`, and final pointers. The medically best checkpoint is not automatically
promotable.

## Execution order

1. Verify each arm initializes from the same Stage A hashes.
2. Run a 10-update one-batch alignment check for the selective+L2-SP arm.
3. Run all three 500-update pilots.
4. Plot medical loss against general perplexity degradation at every 50-update
   validation point.
5. Select the best method inside the preferred band; use the hard-cap band only
   if the extra medical gain is material and documented.
6. Adjust the L2-SP strength once if it is visibly inactive or dominates the
   language-model loss, then repeat only that arm.
7. Convert the selected pilot profile to an 8,033-update full-run config.
8. Resume-test its checkpoint before the full Colab run.
9. Evaluate the selected checkpoint once on the untouched medical and general
   test splits.
10. Compare v1, v2, and the later LoRA run using the same reporting schema.

No v2 result should replace v1 artifacts. Promotion is a recorded comparison
decision, not an overwrite.
