# Stage A Implementation and Training Report

**Project:** Medical Small Language Model from Scratch  
**Stage:** Stage A general-domain causal-language-model pretraining  
**Report date:** 2026-07-21  
**Status:** Complete, evaluated, integrity-verified, and promoted  
**Promoted checkpoint:** `checkpoint_00007250`  
**Final chronological checkpoint:** `checkpoint_00007310`

**Unique trainable parameters:** 35,463,680

**Primary evidence:** Saved checkpoint manifests, trainer state, evaluation JSON, notebook outputs, configuration files, and regression-test results

> [!IMPORTANT]
> This report records a language-model pretraining experiment. Neither the promoted checkpoint nor this report establishes medical correctness, clinical safety, diagnostic ability, or fitness for patient-facing use.

## 1. Executive summary

Stage A began with prepared tokenized datasets and a custom tokenizer, but without a language-model architecture or a training system. We designed and implemented a decoder-only causal language model, a deterministic and resumable training pipeline, mixed-precision support, evaluation, structured metrics, and crash-safe checkpoints. We then tested the system locally, exercised it through short development and resume runs, prepared a self-contained Google Colab workflow, and completed one full epoch of Stage A pretraining on a Tesla T4 GPU using FP16.

The completed run processed all 935,642 training sequences. The final training state reached update 7,310, epoch 1, with 239,524,352 consumed training tokens, no skipped optimizer updates, and no non-finite events. Validation comparison showed that checkpoint 7,250 was marginally better than the final chronological checkpoint at update 7,310, so checkpoint 7,250 was selected and promoted.

The promoted checkpoint achieved:

| Split | Loss | Perplexity | Samples | Tokens |
|---|---:|---:|---:|---:|
| Validation | 3.198383 | 24.492887 | 1,822 | 466,432 |
| Test | 3.679168 | 39.613418 | 1,185 | 303,360 |

The local promoted checkpoint was subsequently audited. All nine checkpoint artifacts matched their declared sizes and SHA-256 hashes. Its recorded tokenizer hash and Stage A dataset-manifest hash also matched the corresponding local files.

### Report navigation

- Sections 2–3: starting assets and model architecture.
- Sections 4–8: loss, sampling, optimization, precision, and evaluation design.
- Sections 9–11: checkpoints, command line, and regression testing.
- Sections 12–14: Colab workflow, preflight evidence, and full training run.
- Sections 15–17: selection, promotion, and integrity verification.
- Sections 18–22: repository state, delivered capabilities, limitations, and next steps.

### Evidence and interpretation convention

Exact values in this report come from machine-readable artifacts when available. Rounded values are labeled “approximately.” Statements about why validation and test differ, likely runtime behavior, or suitability for a later stage are interpretations and should not be treated as measured facts. The saved JSON files remain the authoritative source for final numeric evaluation results.

## 2. Starting point

At the start of this work, the repository already contained:

- A custom tokenizer with a vocabulary of 16,000 tokens.
- A packed Stage A pretraining dataset in `datasets/tokenized/stage_a`.
- Separate validation and test datasets under `datasets/tokenized/evaluation`.
- EOS-separated documents.
- Binary token shards suitable for memory-efficient loading.
- A dataset contract that produces already shifted causal targets.

The Stage A training data had the following properties:

| Property | Value |
|---|---:|
| Training sequences | 935,642 |
| Sequence length | 256 |
| Stored sample width | 257 tokens |
| Effective training tokens | 239,524,352 |
| Vocabulary size | 16,000 |
| Training shards | 115 |
| Token storage type | `uint16` |
| Document separator | EOS |
| Label strategy | Next-token shift performed by the dataset |

The apparent difference between the originally quoted approximately 240.46 million stored tokens and the final 239,524,352 consumed training tokens is explained by the packed format. Each stored sample contains 257 token IDs so it can produce 256 aligned input/target positions. Training-token accounting counts the 256 supervised target positions per sequence:

```text
935,642 sequences × 256 target tokens = 239,524,352 training tokens
```

No model architecture or end-to-end trainer existed yet, so those became the primary implementation objectives.

## 3. Architecture design

We chose a compact decoder-only Transformer appropriate for the available data volume and Colab-class hardware. The implemented model configuration was:

| Setting | Implemented value |
|---|---:|
| Unique trainable parameters | 35,463,680 |
| Vocabulary size | 16,000 |
| Hidden size | 512 |
| Decoder layers | 8 |
| Attention heads | 8 |
| Head dimension | 64 |
| Intermediate/SwiGLU size | 1,536 |
| Maximum positions | 1,024 |
| Stage A sequence length | 256 |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Positional method | RoPE |
| RoPE base | 10,000 |
| Tied input/output embeddings | Yes |
| Dropout | 0.0 |
| Initialization standard deviation | 0.02 |

The maximum context length was deliberately set to 1,024 even though Stage A trains at length 256. This leaves room for later continual pretraining or supervised fine-tuning at longer context lengths without changing the learned architecture.

The exact parameter total is derived from the tied-embedding implementation:

| Component | Parameters |
|---|---:|
| Token embedding and tied output head | 8,192,000 |
| Eight attention modules | 8,388,608 |
| Eight SwiGLU MLP modules | 18,874,368 |
| Per-layer RMSNorm scales | 8,192 |
| Final RMSNorm scale | 512 |
| **Total unique trainable parameters** | **35,463,680** |

Because the output head shares the token-embedding matrix, it does not add another 8,192,000 unique parameters.

The model was implemented as repository-native PyTorch modules:

- `src/medical_slm/model/config.py` — validated decoder configuration.
- `src/medical_slm/model/normalization.py` — RMSNorm.
- `src/medical_slm/model/rope.py` — rotary positional embeddings.
- `src/medical_slm/model/attention.py` — causal self-attention.
- `src/medical_slm/model/layers.py` — SwiGLU and decoder blocks.
- `src/medical_slm/model/decoder.py` — complete decoder language model.

## 4. Loss contracts

### 4.1 Direct packed shifted-label loss

The packed pretraining dataset stores 257 token IDs per record and returns:

```python
input_ids = sample[:-1]
labels = sample[1:]
```

Therefore, each logit at position `i` must be compared directly with `labels[i]`. The training loss must not shift the labels a second time. A second shift would accidentally train the model to predict two positions ahead and silently corrupt the learning objective.

We implemented `shifted_packed_causal_loss` to calculate cross-entropy directly between same-position logits and the already shifted labels. The trainer also validates the dataset manifest and rejects an incompatible label strategy.

### 4.2 Separate masked SFT loss

Supervised fine-tuning has a different contract. SFT examples normally retain standard causal alignment and use a mask so that prompt or instruction tokens do not contribute to the response loss. We therefore kept SFT loss separate from packed pretraining loss.

The masked SFT path:

1. Performs the normal one-token causal shift.
2. Applies the response mask after alignment.
3. Computes loss only on response tokens intended for supervision.

Keeping these two functions separate makes accidental double-shifting or incorrect response masking much less likely.

## 5. Deterministic data ordering and exact resume

We implemented a deterministic epoch-seeded batch sampler. Given the same base seed and epoch, it creates the same sample permutation and batches. A different epoch produces a different deterministic order.

The sampler exposes an explicit batch cursor. The checkpoint records both the current epoch and the next batch position. After interruption, training can rebuild the same epoch permutation and continue from the next unconsumed micro-batch instead of repeating or skipping data.

The training state tracks:

- Optimizer update.
- Epoch.
- Batch cursor.
- Consumed micro-batches.
- Consumed samples.
- Consumed tokens.
- Skipped updates.
- Non-finite events.
- Best validation loss.

An early regression test exposed an ambiguity at an epoch boundary: the resumed test expected a monotonically increasing cursor, while the trainer correctly normalized the cursor to zero after completing the epoch and incremented the epoch. The test and semantics were aligned so the cursor consistently means the position within the current epoch.

## 6. Optimizer and learning-rate schedule

We implemented AdamW with decay and no-decay parameter groups. Matrix-like learned weights are subject to weight decay, while bias terms and normalization scale parameters are excluded. This avoids applying regularization to parameters for which decay is generally undesirable.

The final optimization configuration was:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Final learning rate | `3e-5` |
| Betas | `(0.9, 0.95)` |
| Weight decay | `0.1` |
| Maximum gradient norm | `1.0` |
| Warmup updates | 73 |
| Total scheduled updates | 7,310 |
| Schedule after warmup | Cosine decay |
| Seed | 42 |

The scheduler was tested at its important boundaries: initialization, warmup transition, decay region, and final learning rate. Additional tests verified monotonic cosine decay after warmup.

## 7. Mixed precision and batching

The precision system supports automatic resolution among FP32, BF16, and FP16. BF16 is preferred when the GPU has reliable native support. On the Tesla T4 used for the full run, automatic precision correctly selected FP16 and enabled a gradient scaler.

The full training batch configuration was:

| Setting | Value |
|---|---:|
| Micro-batch size | 16 sequences |
| Gradient accumulation | 8 micro-batches |
| Global batch per optimizer update | 128 sequences |
| Sequence length | 256 tokens |
| Tokens per optimizer update | 32,768 |
| Evaluation batch size | 32 sequences |
| Data-loader workers | 2 |
| Pinned memory | Enabled |

The optimizer-update function handles autocast, gradient accumulation, unscaling when FP16 is active, gradient clipping, non-finite detection, optimizer stepping, scaler updates, and per-update metrics.

## 8. Evaluation and metrics

Evaluation uses the same direct packed shifted-label contract as pretraining. Loss is accumulated by token count rather than by averaging batch means, which makes the result invariant to evaluation batch size.

One regression test initially failed because perplexity differed by roughly 1.6 millionths across batch sizes. The underlying loss accumulation had a very small floating-point aggregation difference, and exponentiation amplified it enough to exceed an unusually strict relative tolerance. The evaluation aggregation was corrected/stabilized so loss and perplexity remain consistently batch-size invariant at the tested precision.

The metrics system records structured JSONL events including:

- Training loss.
- Validation loss.
- Perplexity.
- Learning rate.
- Gradient norm.
- Token accuracy where applicable.
- Tokens per second.
- Optimizer-step status.
- Consumed tokens and progress state.
- Skipped updates and non-finite events.

The development overfit run, for example, reached update 10 with finite loss, a valid gradient norm, an optimizer step, and measurable non-zero token accuracy. Its purpose was functional verification, not convergence.

## 9. Checkpoint system

The checkpoint subsystem was built to support exact crash recovery and safe long-running cloud training. Each checkpoint includes:

1. `model.pt` — model parameters.
2. `optimizer.pt` — AdamW state.
3. `scheduler.pt` — learning-rate scheduler state.
4. `scaler.pt` — FP16 gradient-scaler state.
5. `rng_state.pt` — Python, NumPy, CPU PyTorch, and CUDA RNG state.
6. `trainer_state.json` — update, cursor, token accounting, and validation state.
7. `config.json` — exact model and training configuration.
8. `environment.json` — relevant runtime environment information.
9. `metrics_tail.json` — recent metrics retained with the checkpoint.
10. `checkpoint_manifest.json` — artifact sizes, hashes, compatibility hashes, and format metadata.

Checkpoint directories are immutable once completed. Files are written through a temporary location and published atomically. Pointer files such as `latest.json`, `best_validation.json`, and `final_stage_a.json` are also updated atomically. Before loading, the system verifies the checkpoint manifest and rejects missing, modified, corrupt, or dataset-incompatible artifacts.

The retention policy preserves:

- The latest resumable checkpoint.
- The best-validation checkpoint.
- Recent checkpoints.
- Periodic milestone checkpoints.
- The final Stage A checkpoint.

For Colab, checkpoint mirroring to Google Drive was added and tested for verification and idempotence.

## 10. Command-line training entry point

The Stage A trainer is launched through:

```text
scripts/training/train_stage_a.py
```

The command supports:

- A training configuration path.
- A separate model configuration path.
- Maximum-update overrides.
- Resume from a named or pointer checkpoint.
- One-batch overfitting mode.

An early command-line attempt exposed that the configuration flag was `--config`, not a shortened or wrapped variant. The working development command was equivalent to:

```powershell
python scripts/training/train_stage_a.py `
  --config configs/training_stage_a.yaml `
  --overfit-one-batch `
  --max-updates 10
```

## 11. Regression testing history

Testing was performed incrementally as each subsystem was added. The visible suite progressed through 318, 326, 334, and 347 passing tests during training-system development. Four targeted generation tests and the other intervening coverage brought the final post-training suite to 356 passing tests.

Important targeted regressions include:

- Packed loss does not shift labels twice.
- SFT loss performs causal alignment and honors response masking.
- Deterministic sampler covers an epoch without duplication.
- Different epochs generate different deterministic orders.
- Sampler resume matches the uninterrupted remainder.
- Scheduler values are correct at warmup and final boundaries.
- Cosine decay behaves monotonically after warmup.
- Evaluation loss and perplexity are invariant to batch size.
- Checkpoint round-trip restores all training state.
- RNG state is restored exactly.
- A resumed update matches the uninterrupted trajectory.
- Corrupt checkpoint artifacts are rejected before loading.
- Dataset-incompatible checkpoints are rejected.
- Pointer updates are atomic and resolvable.
- Immutable checkpoint directories are not overwritten.
- Google Drive mirroring is verified and idempotent.
- Retention keeps pointer targets, recent checkpoints, and milestones.
- Tiny training validates, checkpoints, and resumes at the correct batch.
- One-batch overfit mode reuses the selected batch without advancing the normal data cursor.

The complete regression result after adding and exercising the post-training inference gate was:

```text
356 passed in 13.66s
```

During the earlier artifact audit, the restricted assistant sandbox could not launch the repository's Python environment. A later explicitly authorized run used the working project virtual environment successfully, so the 356-pass result supersedes that earlier environment limitation.

## 12. Colab and RunPod portability

Both Google Colab and RunPod were considered. RunPod offers more predictable GPU selection and persistent storage, while Colab Pro was selected for this run because it was immediately available and suitable for the 35-million-parameter-class model.

The repository contains platform-specific configurations:

- `configs/training_stage_a_colab.yaml`
- `configs/training_stage_a_runpod.yaml`

The Colab notebook is:

```text
notebooks/colab_stage_a.ipynb
```

The notebook was developed to provide:

- Repository cloning and editable dependency installation.
- Google Drive mounting.
- Dataset availability checks.
- Copying data from Drive to Colab's local SSD for faster shard access.
- Automatic precision inspection.
- Initial validation-baseline verification.
- A 50-update development run.
- Checkpoint mirroring to Drive.
- A restart/restore/resume test from update 50 to update 100.
- Fresh full-run and resume sections that include their own required setup.
- Full Stage A training.
- Checkpoint integrity verification.
- Validation comparison of best and final checkpoints.
- A single held-out test evaluation of the selected checkpoint.
- Promotion-pointer and evaluation-report generation.

The full-run and resume sections were made self-contained so they can restore code and necessary imports on a fresh runtime rather than relying on every earlier exploratory cell having been run.

## 13. Preflight and development results

### 13.1 Initial random validation baseline

For a randomly initialized 16,000-token classifier, the theoretical cross-entropy is approximately:

```text
ln(16000) = 9.680344
```

The initial full validation evaluation produced:

| Metric | Result |
|---|---:|
| Initial validation loss | 9.774141 |
| Expected random loss | 9.680344 |
| Initial perplexity | 17,573.392 |
| Evaluated tokens | 466,432 |

The validation gate checked that loss and perplexity were finite, that the exact expected number of validation tokens was processed, and that loss was reasonably close to the random-initialization expectation. The gate passed.

### 13.2 Development and resume run

The first Colab development run trained to update 50 and wrote a verified Drive checkpoint. After restoring from Drive, training resumed to update 100. The restored state showed:

| State field | Value at update 100 |
|---|---:|
| Update | 100 |
| Batch cursor | 800 |
| Consumed micro-batches | 800 |
| Consumed samples | 12,800 |
| Consumed tokens | 3,276,800 |
| Epoch | 0 |
| Skipped updates | 0 |
| Non-finite events | 0 |

Training and validation both improved strongly during this test. Visible validation metrics were:

| Update | Validation loss | Perplexity |
|---:|---:|---:|
| 0 | 9.7741 | 17,573.39 |
| 50 | 7.4758 | 1,764.80 |
| 100 | approximately 6.86 | approximately 954–957 |

This demonstrated that the loss path, optimizer, precision handling, checkpoint mirroring, restoration, data cursor, and resume logic all worked before committing to the full run.

## 14. Full Stage A run

The full run was executed on Google Colab Pro with:

| Runtime property | Value |
|---|---|
| GPU | Tesla T4 |
| CUDA capability | 7.5 |
| PyTorch CUDA runtime | 12.8 |
| Precision | FP16 |
| Gradient scaler | Enabled |
| Epochs | 1 |
| Planned optimizer updates | 7,310 |

The promoted checkpoint captured this software environment:

| Runtime component | Recorded value |
|---|---|
| Operating system | Linux 6.6.122+, x86-64, glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 91900 |

The T4 does not provide the preferred native BF16 behavior for this pipeline, so automatic precision correctly resolved to FP16. This was expected and safe because the trainer saved and restored the gradient scaler and tracked non-finite events.

The run produced milestones at updates 1,000 through 7,000, validation/checkpoint points including update 7,250, and the final chronological checkpoint at update 7,310. The observed pace was roughly 1,000 updates every 18–20 minutes on the T4, which was normal for the configured model, effective batch size, validation work, and Drive checkpoint mirroring.

The final Stage A state was independently displayed and verified in Colab:

| State field | Final value |
|---|---:|
| Update | 7,310 |
| Epoch | 1 |
| Batch cursor | 0 |
| Consumed micro-batches | 58,478 |
| Consumed samples | 935,642 |
| Consumed tokens | 239,524,352 |
| Skipped updates | 0 |
| Non-finite events | 0 |
| Best validation loss | 3.19838276440017 |

The final batch was smaller than a complete accumulated update because the dataset size is not an exact multiple of the nominal global batch. The accounting nevertheless confirms that every training sequence was consumed exactly once.

Colab temporarily exhausted GPU access after training, but the run had already completed and the checkpoints had been mirrored to Google Drive. The model was therefore not lost. When GPU access became available again, final validation and test evaluation proceeded from the Drive checkpoints.

## 15. Final checkpoint selection and test evaluation

Two checkpoints were compared on the full validation set:

| Checkpoint | Role | Validation loss | Validation perplexity | Tokens |
|---|---|---:|---:|---:|
| `checkpoint_00007250` | Best-validation candidate | 3.19838276440017 | 24.492887380448188 | 466,432 |
| `checkpoint_00007310` | Final chronological state | 3.1987046243962305 | 24.50077192987643 | 466,432 |

Checkpoint 7,250 had the lower validation loss, although the difference was very small. It was selected before looking at the test result. This preserves the test set as a held-out final measurement rather than using it for model selection.

The absolute validation-loss difference was approximately `0.00032186`, and the perplexity difference was approximately `0.007885`. These are small differences, but the predefined lower-validation-loss rule still selects checkpoint 7,250 deterministically.

The selected checkpoint was then evaluated once on the test set:

| Metric | Test result |
|---|---:|
| Loss | 3.679167894799388 |
| Perplexity | 39.61341782363614 |
| Samples | 1,185 |
| Tokens | 303,360 |
| Batches | 38 |

The validation/test gap should be recorded and monitored in later work. It may reflect differences in source composition or difficulty between the two evaluation splits. It is not evidence of a broken evaluation path: both evaluations used complete, finite, explicitly counted datasets, and the test split was not used for checkpoint selection.

## 16. Promotion and artifact preservation

The selected checkpoint was promoted with:

```json
{
  "checkpoint": "checkpoint_00007250"
}
```

The machine-readable permanent reports are:

- `reports/stage_a/promoted_stage_a.json`
- `reports/stage_a/stage_a_evaluation.json`

The local promoted checkpoint is stored at:

```text
artifacts/training/stage_a/checkpoints/checkpoint_00007250
```

Large training artifacts are intentionally excluded from Git. They should remain backed up in Google Drive or another durable object store. The final chronological checkpoint, `checkpoint_00007310`, should also be retained remotely even though it was not promoted, because it is the exact end-of-epoch training state.

## 17. Final integrity audit

After downloading the promoted checkpoint into the project, a local audit verified:

- Checkpoint name: `checkpoint_00007250`.
- Manifest format version: 1.
- Nine declared checkpoint artifacts present.
- Every artifact size matched the manifest.
- Every artifact SHA-256 hash matched the manifest.
- Total declared artifact size: approximately 405.94 MiB.
- Local Stage A dataset manifest exists and matches the checkpoint compatibility hash.
- Local tokenizer JSON exists and matches the checkpoint compatibility hash.
- The copy of the evaluation report under `artifacts/training/stage_a` is byte-for-byte identical to the copy under `reports/stage_a`.
- The copy of the promotion pointer under `artifacts/training/stage_a` is byte-for-byte identical to the copy under `reports/stage_a`.
- The promotion pointer and evaluation report both select checkpoint 7,250.

The promoted checkpoint's own saved trainer state is:

| Field | Value |
|---|---:|
| Update | 7,250 |
| Epoch | 0 |
| Batch cursor | 58,000 |
| Consumed micro-batches | 58,000 |
| Consumed samples | 928,000 |
| Consumed tokens | 237,568,000 |
| Skipped updates | 0 |
| Non-finite events | 0 |
| Best validation loss | 3.19838276440017 |

This state is exactly what is expected for the best checkpoint taken shortly before the end of the single training epoch.

### 17.1 Post-training inference gate

Before continual pretraining, a repository-native inference smoke test was added at:

```text
scripts/evaluation/check_stage_a_model.py
```

The gate uses the promotion pointer by default and performs the following checks in order:

1. Resolve `checkpoint_00007250` from `reports/stage_a/promoted_stage_a.json`.
2. Verify every checkpoint artifact against `checkpoint_manifest.json`.
3. Reconstruct the model from the checkpoint's saved model configuration.
4. Strictly load `model.pt`, rejecting missing or unexpected state-dictionary keys.
5. Hash the tokenizer and compare it with the checkpoint compatibility hash.
6. Confirm that tokenizer and model both use a vocabulary of 16,000 entries.
7. Confirm that `<bos>` and `<eos>` are present.
8. Run a forward pass and validate the logits shape.
9. Reject any NaN or Inf logits or sampling probabilities.
10. Generate autoregressive continuations from fixed or user-provided prompts.
11. Save settings, environment, checks, and outputs to a machine-readable JSON report.

The implementation supports greedy decoding and temperature/top-k/top-p sampling. It enforces the 1,024-token context limit and stops early on EOS. Sampling is seeded for repeatability within a compatible environment.

Default local execution is:

```bash
python scripts/evaluation/check_stage_a_model.py
```

The default report destination is:

```text
reports/stage_a/stage_a_generation_smoke_test.json
```

The inference gate is considered complete only after the script prints `Stage A model smoke test: PASSED`, the generated JSON is retained, and the sample text is manually reviewed for obvious corruption such as empty decoding, broken token boundaries, or pathological immediate failures. Repetition, weak factuality, or failure to follow instructions may still occur because Stage A is a base causal model rather than an instruction-tuned assistant.

Passing this gate establishes that the promoted files can be loaded and used for finite autoregressive inference. It does not establish medical knowledge, factual accuracy, or clinical safety. Those require separate evaluations in later stages.

The permanent gate was run on the promoted checkpoint using CPU/FP32, seed 42, temperature 0.8, top-k 50, top-p 0.95, and a maximum of 32 new tokens. It passed with:

| Verification | Observed result |
|---|---|
| Checkpoint | `checkpoint_00007250` |
| Manifest verification | Passed |
| Tokenizer compatibility | Passed |
| Loaded unique parameters | 35,463,680 |
| Vocabulary size | 16,000 |
| Forward logits | Correct shape and finite |
| Autoregressive generation | Passed for four prompts |
| EOS stopping | Observed on one sampled continuation |
| Regression suite after addition | 356 passed |

The complete machine-readable output is stored in `reports/stage_a/stage_a_generation_smoke_test.json`. Qualitative review found sentence-like, decodable text and sensible local continuations. It also found repetition and medically unreliable or awkward claims. This is not a checkpoint-loading failure; it is evidence that the Stage A base model is operational but not yet medically specialized, instruction-tuned, or safety-qualified.

## 18. Repository state at completion

At the time of the final audit:

- `notebooks/colab_stage_a.ipynb` had local modifications containing the completed Colab workflow.
- `.gitignore` had local modifications that exclude generated training artifacts, tokenizer artifacts, and tokenized datasets.
- `reports/` was untracked and needed to be added to Git.
- `.gitignore` contained a harmless duplicate `artifacts/training/` entry that can be cleaned up.

The large checkpoint should not be committed to normal Git history. The small JSON reports and this Markdown report should be committed because they provide the durable, reviewable record of the experiment.

## 19. What Stage A established

Stage A delivered more than a set of model weights. It established a reusable training foundation with:

- A from-scratch decoder-only Transformer implementation.
- Explicit and tested pretraining/SFT loss contracts.
- Deterministic data order.
- Exact resumable batch position.
- Mixed precision with hardware-aware automatic selection.
- Gradient accumulation and clipping.
- Stable token-weighted evaluation.
- Structured machine-readable metrics.
- Atomic, checksummed checkpoints.
- RNG-complete crash resume.
- Remote checkpoint mirroring.
- Retention and named checkpoint pointers.
- End-to-end Colab development, resume, full-run, evaluation, and promotion procedures.
- A promoted baseline suitable for the next curriculum stage.

## 20. Limitations and non-claims

The completed run has several important boundaries:

- **Perplexity is not medical competence.** Stage A measured next-token prediction, not medical factuality, reasoning, calibration, diagnosis, or treatment quality.
- **No clinical safety evaluation was performed.** The checkpoint must not be treated as suitable for patient-facing or professional decision-making use.
- **Only one pretraining epoch was run.** This was the planned Stage A budget, not a claim that one epoch is globally optimal.
- **Only two end-of-run candidates were compared in the final notebook evaluation.** Checkpoint 7,250 won under the recorded lower-validation-loss rule; the tiny margin should not be overinterpreted as a broad quality difference.
- **The test set was used once for the promoted checkpoint.** Future tuning must not optimize against this result; a new untouched test set may be needed if repeated decisions begin to depend on it.
- **The validation/test gap is unexplained.** Dataset composition or difficulty is a plausible explanation, but source-level analysis is required before drawing conclusions.
- **Cross-platform bitwise identity is not guaranteed.** RNG capture and batch cursors support exact continuation in a compatible environment, while different GPU kernels, PyTorch versions, or precision modes may produce small numerical differences.
- **Large binaries are outside Git.** Reproducibility depends on retaining the checkpoint directories and data artifacts in durable storage together with their manifests.
- **Qualitative generation remains weak.** The smoke test produced fluent fragments but also repetition and unreliable medical phrasing. Later training and evaluation must address this rather than treating basic fluency as correctness.

These limitations do not invalidate Stage A completion. They define what evidence must be added during continual medical pretraining, supervised fine-tuning, and downstream evaluation.

## 21. Recommended next steps

The immediate next phase should use `checkpoint_00007250` as its initialization point. Before beginning, the next-stage plan should define:

1. The continual medical-pretraining dataset and its manifest.
2. Whether the sequence length remains 256 or increases toward the architecture's 1,024-position capacity.
3. The next-stage learning-rate reset or continuation policy.
4. The number of epochs or target consumed tokens.
5. A separate validation set representing the desired medical-domain distribution.
6. Retention of a general-domain validation set to detect catastrophic forgetting.
7. Checkpoint compatibility rules for loading only the model weights while initializing a new optimizer and schedule where appropriate.
8. A final promotion protocol that again selects on validation and evaluates test only after selection.

Before committing the current work, the recommended repository housekeeping is:

```text
1. Remove the duplicate artifacts/training/ line from .gitignore.
2. Review the Colab notebook for outputs or credentials that should not be committed.
3. Add reports/stage_a/*.json and this Markdown report to Git.
4. Commit the Colab notebook and configuration changes intentionally.
5. Confirm both checkpoint_00007250 and checkpoint_00007310 remain backed up remotely.
```

Before starting the next run, define explicit acceptance gates. At minimum, require dataset/tokenizer compatibility, a finite random or resumed baseline, a short overfit smoke test, a successful checkpoint-resume test, decreasing medical-validation loss, bounded general-domain regression, zero unexplained non-finite updates, and integrity verification of the promoted artifacts.

## 22. Final conclusion

Stage A was completed successfully. The model trained over the complete Stage A dataset for one epoch, the loss decreased from the expected random baseline to a validation loss of approximately 3.20, resume behavior was demonstrated, no numerical failures were recorded, and the final artifacts passed integrity and compatibility checks. Checkpoint 7,250 was correctly promoted because it had a slightly better full-validation loss than the end-of-epoch checkpoint. The promoted model subsequently passed strict loading, finite-forward, tokenizer-compatibility, autoregressive-generation, and full-regression gates. Its sample text confirms that the base model works while also demonstrating that medical quality and safety are not yet sufficient. The project is ready to plan continual medical-domain pretraining from the promoted Stage A checkpoint.
