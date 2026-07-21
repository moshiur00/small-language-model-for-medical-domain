# Stage B Continual-Pretraining Data Preparation Report

**Date:** 2026-07-21  
**Status:** Complete and verified  
**Purpose:** Prepare document-disjoint continual medical pretraining, medical validation, and sealed medical test datasets.

## Outcome

The canonical Stage B corpus is disjoint from Stage A and preserves the planned curriculum mixture. Separate source-stratified medical validation and test sets were selected from documents unused by either training stage. All packed binary shards passed hash verification and loader checks.

## Training corpus

| Property | Result |
|---|---:|
| Processed path | `datasets/processed/corpora/continual_medical_stage_b_225m` |
| Tokenized path | `datasets/tokenized/continual_medical_stage_b` |
| Documents | 218,812 |
| Exact processed stream tokens | 224,995,919 |
| Packed sequences | 875,470 |
| Effective supervised targets | 224,120,320 |
| Sequence length | 256 |
| Stored sample width | 257 |
| Binary shards | 107 |
| Stage A document overlap | 0 |
| Internal duplicate IDs | 0 |
| Medical share | 82.2223% |
| General rehearsal share | 17.7777% |
| Updates at 32,768 targets/update | 6,840 |

Source allocation:

| Source | Domain | Exact tokens | Documents |
|---|---|---:|---:|
| PMC Open Access | Medical | 94,996,966 | 10,183 |
| PubMed abstracts | Medical | 64,999,868 | 158,484 |
| WikiDoc | Medical | 24,999,884 | 14,130 |
| FineWeb-Edu | General | 39,999,201 | 36,015 |

Stage A had consumed all available post-deduplication Wikipedia records. The intended 15M-token Wikipedia rehearsal allocation was therefore reassigned to unused FineWeb-Edu records, preserving the full 40M-token general rehearsal budget and zero Stage A overlap.

The processed training JSONL SHA-256 recorded by the tokenized manifest is:

```text
86ed0a6d21d59f455a9fa0551aad2577bce9ef25845742a3c668b57d2593de38
```

The tokenized Stage B dataset-manifest SHA-256 is:

```text
1d8b828604c8e6ff9cc6ea4d6cffe604f78367dd87e8c171d73dd5dcfd4170d1
```

## Medical evaluation

Processed evaluation path:

```text
datasets/processed/evaluation_medical
```

Tokenized evaluation path:

```text
datasets/tokenized/evaluation_medical
```

| Property | Validation | Test |
|---|---:|---:|
| Documents | 997 | 1,041 |
| Exact processed stream tokens | 1,001,638 | 1,009,193 |
| Packed sequences | 3,897 | 3,926 |
| Effective supervised targets | 997,632 | 1,005,056 |
| Shards | 1 | 1 |
| Duplicate IDs | 0 | 0 |

Each split uses the same planned source token proportions:

- 500,000 target tokens from PMC Open Access.
- 350,000 target tokens from PubMed abstracts.
- 150,000 target tokens from WikiDoc.

Disjointness audit:

| Comparison | Overlapping IDs |
|---|---:|
| Stage A vs medical validation | 0 |
| Stage A vs medical test | 0 |
| Stage B vs medical validation | 0 |
| Stage B vs medical test | 0 |
| Medical validation vs medical test | 0 |

The medical test split is sealed and must not be used for configuration selection, learning-rate decisions, checkpoint selection, or early stopping. It should be evaluated only after a Stage B checkpoint has been selected using validation metrics.

Processed input hashes:

| Split | SHA-256 |
|---|---|
| Validation | `93b1029d49bd12cb8cf9b6038a9359294744f4edfc6d4b5383441c769215380f` |
| Test | `5d1504413bdb1712f84b6948d848c6ba18a472eb79b97e9da3cad594b4b42912` |

The tokenized medical evaluation dataset-manifest SHA-256 is:

```text
6537ac466894df9175009017198e7c9996e55705a9a9388130716f4e261a706d
```

## Provenance note

The canonical Stage B and medical evaluation corpora were selected from the complete `datasets/interim/license_validated` source stage. The configured `toxicity_audited` directory had been overwritten by a 100-document development run and was therefore incomplete. The toxicity policy was configured with `automatically_reject: false`, so that stage adds audit metadata but does not remove documents. This input-stage choice is recorded explicitly rather than silently treating the truncated development directory as production data.

Before any model is considered for medical use, content-safety evaluation remains necessary. Corpus toxicity annotation is not equivalent to clinical model-safety evaluation.

## Verification performed

- Corpus assembly uses stable document IDs and deterministic source order.
- Stage B excludes all 193,231 Stage A document IDs.
- Evaluation excludes all 412,043 Stage A and Stage B training IDs.
- Validation and test selection is mutually exclusive.
- Document-ID set hashes are recorded in processed manifests.
- Tokenizer SHA-256 matches the Stage A tokenizer.
- Packed format remains `next_token_shift_in_dataset`.
- Every one of the 109 new shard files passed size and SHA-256 verification.
- Packed loader returned aligned `[256]` input, label, and attention-mask tensors.
- The direct shifted-label relationship was visually and programmatically exercised by the loader check.
- Complete project regression suite: `359 passed in 11.08s`.

## Next implementation gate

The next step is Stage B training-system support:

1. Load only model weights from promoted Stage A checkpoint `checkpoint_00007250`.
2. Initialize a fresh Stage B optimizer, scheduler, scaler, and training state.
3. Record parent-checkpoint lineage in Stage B checkpoints.
4. Evaluate the untouched Stage A model on medical and general validation sets to establish baselines.
5. Add dual-validation and forgetting-aware checkpoint promotion.
6. Run one-batch overfit, 50-update development, and 50-to-100 exact-resume gates before full training.
