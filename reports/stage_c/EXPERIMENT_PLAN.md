# Stage C v1 Supervised Instruction Fine-Tuning Plan

**Status:** Training system implemented; execution gates pending
**Method:** Full-parameter response-only supervised instruction fine-tuning  
**Parent:** Stage B v2 `checkpoint_00008000`  
**Selection:** Validation only, subject to medical and general retention gates

The corrected dataset and post-build duplicate/license audit have passed for
internal research training. Public distribution of a Stage C checkpoint remains
blocked pending manual source-license review.

## Objective

Teach the promoted Stage B v2 base model to follow the repository's instruction
template using only the existing balanced seven-source corpus. Stage C v1 is a
controlled full-parameter baseline. It is not a clinical-safety claim and does not
replace a later LoRA comparison.

## Immutable parent identity

| Property | Value |
|---|---|
| Checkpoint | `checkpoint_00008000` |
| Model SHA-256 | `799fe9c34648044d21bf73258cd55a46716167f89dcf64e2c0487e0382d65c14` |
| Checkpoint manifest SHA-256 | `ea60b0d6b66ea3bd1987f9ff7bbdd75ba34bc0f05b7c21349ad6ef90615a9b71` |
| Tokenizer SHA-256 | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` |
| Parameters | 35,463,680 |

Stage C loads only parent model weights. Optimizer, scheduler, precision scaler, RNG
progression, counters, and sampler state begin fresh. Every checkpoint must embed
the complete parent and Stage C dataset identity.

## Training contract

- Dedicated Stage C configuration, trainer, evaluator, and CLI.
- Standard shifted causal loss over response/EOS labels only.
- Token-summed loss normalization across gradient accumulation.
- Attention masks passed through every decoder layer.
- Per-batch cropping to the longest non-padding sequence.
- Deterministic epoch-seeded ordering with an explicit next-batch cursor.
- FP32, BF16, or gradient-scaled FP16 selected by hardware capability.
- Atomic immutable checkpoints and exact resume.
- JSONL metrics and verified Google Drive mirroring.

## Initial optimization search

Run matched 100-150-update pilots from the same parent and identical batch order:

| Setting | Pilot A | Pilot B |
|---|---:|---:|
| Peak learning rate | `1e-5` | `2e-5` |
| Weight decay | 0.01 | 0.01 |
| AdamW betas | `(0.9, 0.95)` | `(0.9, 0.95)` |
| Warmup | 5% | 5% |
| Schedule | Cosine | Cosine |
| Gradient clipping | 1.0 | 1.0 |
| Seed | 42 | 42 |

The lower validation response loss wins only when both medical and general retention
remain acceptable. Test data is prohibited from pilot selection.

## Tentative full-run profile

| Setting | Initial value |
|---|---:|
| Maximum length | 1,024 |
| T4 micro-batch | 4 |
| Gradient accumulation | 8 |
| Effective batch | 32 examples |
| Epoch limit | 3 |
| Validation interval | 25 updates |
| Checkpoint interval | 50 updates |
| Early-stopping patience | 3 validations |
| Precision | Automatic; FP16 expected on T4 |

The corrected manifest contains 6,248 training examples. With micro-batch 4 and
gradient accumulation 8, each epoch has 1,562 micro-batches and 196 optimizer
updates (the final update consumes the partial accumulation). The locked three-epoch
schedule therefore contains exactly 588 optimizer updates.

## Baselines and retention gates

Before training, the unchanged parent is evaluated on SFT validation plus the
existing medical and general pretraining validation splits.

| Distribution | Stage B v2 PPL | Preferred ceiling (+10%) | Hard ceiling (+15%) |
|---|---:|---:|---:|
| Medical validation | 23.004 | 25.304 | 26.454 |
| General validation | 28.474 | 31.321 | 32.745 |

If both pilots breach the preferred band, reduce learning rate or exposure rather
than relaxing the limits after observing results.

## Promotion rule

Promote the lowest response-only SFT validation loss that:

1. remains inside the preferred medical retention band;
2. remains inside the preferred general retention band;
3. improves instruction metrics over the zero-update parent;
4. has no unexplained non-finite or skipped updates;
5. passes checkpoint, lineage, tokenizer, and dataset verification;
6. was selected without test data.

Hard-band candidates may be preserved as experiments but are not automatically
promoted.

## Evaluation

Development evaluation includes overall and per-source response loss, response-token
accuracy, formatting compliance, medical/general validation retention, and fixed
qualitative prompts. After selection is fixed, evaluate the sealed Stage C test once,
including MedMCQA/PubMedQA-style accuracy where the answer format is reliably
extractable.

Perplexity and generated examples do not establish factuality or clinical safety.

## Implementation milestones

1. ~~Lock this plan and the dataset specification.~~ Complete.
2. ~~Correct truncation, duplicate handling, grouped splitting, and manifests.~~ Complete.
3. ~~Rebuild, verify, and audit `datasets/tokenized/sft_stage_c_v1`.~~ Complete for internal research.
4. ~~Implement token-summed SFT training and evaluation.~~ Complete.
5. ~~Implement Stage C configuration, lineage, checkpoints, and exact resume.~~ Complete.
6. ~~Add regression tests and a self-contained Colab notebook.~~ Complete; runtime
   gates remain to be executed on Colab.
7. Run the zero-update and one-batch gates.
8. Run matched learning-rate pilots.
9. Lock and execute the full run.
10. Select on validation, open test once, promote, preserve, and add inference.
