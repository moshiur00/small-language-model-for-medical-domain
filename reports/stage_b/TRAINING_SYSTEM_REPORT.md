# Stage B Training System Report

**Date:** 2026-07-21  
**Status:** Implemented, regression-tested, trained for one complete epoch, and preserved as the v1 comparison run
**Training status:** Complete; the endpoint exhibits catastrophic general-domain forgetting

## Objective

Stage B continually pretrains the 35,463,680-parameter decoder from promoted Stage A checkpoint `checkpoint_00007250` on the document-disjoint continual medical corpus. It must improve medical validation loss without allowing uncontrolled regression on the original general validation distribution.

## Initialization contract

Stage B is a new training phase, not a Stage A resume. Initialization performs these operations in order:

1. Verify every artifact in the Stage A parent checkpoint.
2. Verify that the parent tokenizer SHA-256 matches the Stage B tokenizer.
3. Verify that the complete saved architecture configuration matches Stage B.
4. Strictly load only `model.pt`.
5. Initialize a new AdamW optimizer with no moment state.
6. Initialize a new warmup/cosine scheduler.
7. Initialize a new FP16 scaler when the selected hardware requires one.
8. Initialize Stage B counters at update zero and zero consumed tokens.
9. Retain parent checkpoint, manifest, model, and tokenizer hashes as lineage.

The real initialization-only preflight passed:

```text
parent=checkpoint_00007250
parameters=35,463,680
optimizer_state=empty
stage_b_update=0
stage_b_consumed_tokens=0
```

Stage A optimizer, scheduler, scaler, RNG, cursor, and update state are deliberately not imported.

## Training configuration

| Setting | Value |
|---|---:|
| Training targets | 224,120,320 |
| Planned updates | 6,840 |
| Sequence length | 256 |
| Micro-batch size | 16 |
| Gradient accumulation | 8 |
| Nominal targets/update | 32,768 |
| Peak learning rate | `1e-4` |
| Final learning rate | `1e-5` |
| Warmup updates | 68 |
| Adam betas | `(0.9, 0.95)` |
| Weight decay | `0.1` |
| Gradient clipping | `1.0` |
| Validation interval | 250 updates |
| Checkpoint interval | 250 updates |
| Milestone interval | 1,000 updates |
| General-loss degradation ceiling | 5% |

The architecture and tokenizer remain unchanged from Stage A.

## Dual validation

Before the first optimizer update, the Stage A parent model is evaluated on:

- Medical validation: `datasets/tokenized/evaluation_medical/validation`
- General retention validation: `datasets/tokenized/evaluation/validation`

These become immutable per-run baselines and are saved in training state. At every validation interval the trainer records separate medical and general loss, perplexity, sample count, token count, duration, and relative general-loss change.

## Promotion rule

A checkpoint is promotion-eligible only when:

1. Medical validation loss is lower than the untouched Stage A medical baseline.
2. General validation loss is no more than 5% above the untouched Stage A general baseline.

Among eligible checkpoints, the checkpoint with the lowest medical validation loss becomes `best_eligible`. The lowest medical loss regardless of retention is retained separately as `best_medical`. If no checkpoint satisfies both constraints, Stage B must not be promoted automatically.

The sealed medical test set is not loaded by the trainer and cannot affect checkpoint selection.

## Checkpoint lineage

Every Stage B manifest records:

- Stage name.
- Parent checkpoint name.
- Parent checkpoint-manifest SHA-256.
- Parent model SHA-256.
- Tokenizer SHA-256.
- Stage B training dataset-manifest SHA-256.
- Medical validation dataset-manifest SHA-256.
- General validation dataset-manifest SHA-256.

Resume rejects a checkpoint whose lineage differs from the configured parent or datasets. Exact Stage B resume restores optimizer, scheduler, scaler, RNG, update, epoch, batch cursor, consumed tokens, validation baselines, and best-checkpoint state.

## Commands

Initialization-only preflight:

```bash
python scripts/training/train_stage_b.py --verify-initialization-only
```

Baseline-only evaluation:

```bash
python scripts/training/train_stage_b.py --baseline-only
```

One-batch diagnostic:

```bash
python scripts/training/train_stage_b.py --overfit-one-batch --max-updates 10
```

Development run:

```bash
python scripts/training/train_stage_b.py --max-updates 50
```

Exact resume:

```bash
python scripts/training/train_stage_b.py --resume latest
```

## Verification

Regression coverage includes:

- Stage B configuration defaults and forgetting-budget validation.
- Strict parent model-only initialization.
- Empty fresh optimizer state.
- Zeroed Stage B counters.
- Parent and data lineage capture.
- Medical/general baseline recording.
- Promotion-eligibility threshold behavior.
- Best-medical and best-eligible pointers.
- Manifest lineage persistence.
- Exact Stage B checkpoint resume.
- Baseline restoration after resume.
- Correct epoch-boundary cursor normalization.
- Preservation of all Stage A checkpoint and resume tests.

Current complete result:

```text
365 passed in 10.17s
```

## Next gate

Stage B v1 completed all 6,840 updates over 224,120,320 targets with zero skipped or non-finite updates. The original retention-selected checkpoint was `checkpoint_00000250`; the full endpoint achieved substantially lower medical loss but severe general-domain forgetting. The medical test remains sealed.

Preserve the full v1 experiment before starting another adaptation run. See the [v1 Experiment Report](v1/EXPERIMENT_REPORT.md) and [Preservation and Comparison Workflow](v1/PRESERVATION_AND_COMPARISON.md). Stage B v2 and LoRA will restart independently from the Stage A parent and will be compared under the shared registry at [continual_adaptation_registry.json](../comparisons/continual_adaptation_registry.json).
