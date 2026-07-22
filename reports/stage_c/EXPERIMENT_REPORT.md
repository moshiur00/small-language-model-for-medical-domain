# Stage C v1 Supervised Instruction Fine-Tuning Report

**Status:** Training and sealed evaluation complete; promotion and preservation tooling implemented  
**Parent:** Stage B v2 `checkpoint_00008000`  
**Method:** Full-parameter, response-only supervised instruction fine-tuning  
**Primary:** `medical_instruction_specialist` (`checkpoint_00000588`)  
**Comparator:** `balanced_retention` (`checkpoint_00000125`)

## Outcome

Stage C completed all 588 planned optimizer updates across three epochs without a
safety stop, skipped update, or non-finite event. Two checkpoints are retained on
purpose:

- the **medical-instruction specialist** is the pre-registered primary profile. It
  completed all three epochs and improved every one of the seven SFT validation and
  sealed-test sources;
- the **balanced-retention comparator** has weaker instruction behavior but smaller
  medical- and general-language-model retention degradation.

The sealed test was opened once only after both roles were immutably registered.
Test outcomes did not select or switch profiles.

## Data and training contract

The dataset contains 6,248 training, 369 validation, and 350 sealed-test examples
from Alpaca, ChatDoctor, MedAlpaca, MedInstruct, MedMCQA, OpenMedInstruct, and
PubMedQA. Prompt groups and record IDs are disjoint across splits. Public checkpoint
release remains disabled pending manual source-license review.

Only response and EOS targets contribute to loss. Prompt and padding labels are
masked with `-100`; the trainer performs exactly one causal shift. Stage C loads
only Stage B v2 model weights and starts fresh optimizer, scheduler, scaler, RNG,
sampler, and progress state.

| Setting | Value |
|---|---:|
| Parameters | 35,463,680 |
| Winning peak learning rate | `2e-5` |
| Final learning rate | `2e-6` |
| Weight decay | 0.01 |
| Micro-batch / accumulation | 4 / 8 |
| Effective batch | 32 examples |
| Epochs / updates | 3 / 588 |
| Maximum length | 1,024 |
| Precision on Colab T4 | FP16 with saved scaler |

## Validation-only profiles

| Profile | Checkpoint | SFT validation loss | Medical degradation | General degradation | Band |
|---|---:|---:|---:|---:|---|
| Balanced retention | 125 | 3.079058 | 9.91% | 5.87% | Preferred |
| Medical specialist | 588 | **2.824497** | 13.49% | 8.77% | Hard eligible |

Per-source validation favored checkpoint 588 on all seven sources. Its largest
relative perplexity reduction versus checkpoint 125 was on OpenMedInstruct
(-31.65%), followed by ChatDoctor (-20.49%).

## One-time sealed-test results

| Profile | SFT loss | SFT perplexity | Response-token accuracy |
|---|---:|---:|---:|
| Balanced retention | 3.120167 | 22.650 | 42.79% |
| **Medical specialist** | **2.879608** | **17.807** | **46.34%** |

The specialist reduced perplexity by 21.38% and improved response-token accuracy by
3.55 percentage points. It improved loss and accuracy on every individual source.

Packed-test retention preserves the reason for both profiles:

| Profile | Medical loss / PPL | General loss / PPL |
|---|---:|---:|
| Balanced retention | **3.254389 / 25.904** | **3.752449 / 42.625** |
| Medical specialist | 3.285625 / 26.726 | 3.782876 / 43.942 |

## Reproducibility and release controls

- Checkpoint manifests bind model, tokenizer, parent, dataset, and configuration
  identities with SHA-256 hashes.
- Pre-test profile registration prevents post-test model shopping.
- Promotion remains validation-selected and internal-research-only.
- Preservation includes both full checkpoints, metrics, reports, data contracts,
  tokenizer, and a file-level integrity manifest in a hashed archive.
- Inference reconstructs the exact SFT prompt and verifies promotion, identity,
  lineage, tokenizer compatibility, context bounds, and finite logits.

## Limitations and next experiment

Loss, perplexity, and token accuracy do not measure factual correctness, clinical
reasoning, calibration, refusal quality, or patient safety. Generated text is for
research inspection only and must not be used for medical decisions.

The next controlled experiment should compare full-parameter SFT with LoRA using
the same Stage B v2 parent and data contracts. Since these test results are now
known, continued tuning requires a new untouched final benchmark.
