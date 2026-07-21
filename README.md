# Medical Small Language Model

A from-scratch decoder-only language model and training pipeline intended for a staged medical-language-model curriculum. The repository covers data preparation, tokenizer training and evaluation, packed causal pretraining, exact checkpoint resume, cloud execution, and final checkpoint promotion.

> [!IMPORTANT]
> This is a research and engineering project, not a clinical system. The Stage A checkpoint has only been evaluated with language-model loss and perplexity. It has not been evaluated for medical correctness, factuality, safety, diagnosis, treatment recommendations, or real-world patient use.

## Navigation

- [Current status](#current-status)
- [Stage A model](#stage-a-model)
- [Training data contract](#training-data-contract)
- [Installation](#installation)
- [Tests](#tests)
- [Use the promoted Stage A model](#use-the-promoted-stage-a-model)
- [Local Stage A commands](#local-stage-a-commands)
- [Google Colab](#google-colab)
- [Checkpoints and preservation](#checkpoints-and-preservation)
- [Reproducibility snapshot](#reproducibility-snapshot)
- [Stage B continual pretraining](#stage-b-continual-pretraining)

## Current status

**Stage A pretraining is complete.** The model was trained from random initialization for one full epoch over the packed Stage A dataset on Google Colab Pro using a Tesla T4 and FP16.

| Item | Result |
|---|---:|
| Training sequences | 935,642 |
| Consumed training tokens | 239,524,352 |
| Optimizer updates | 7,310 |
| Unique trainable parameters | 35,463,680 |
| Skipped updates | 0 |
| Non-finite events | 0 |
| Best checkpoint | `checkpoint_00007250` |
| Final chronological checkpoint | `checkpoint_00007310` |
| Best validation loss | 3.198383 |
| Best validation perplexity | 24.492887 |
| Held-out test loss | 3.679168 |
| Held-out test perplexity | 39.613418 |
| Post-training inference gate | Passed |
| Current regression suite | 365 passed |

Checkpoint 7,250 was promoted because its full-validation loss was marginally lower than the final update-7,310 checkpoint. The test set was evaluated only after checkpoint selection.

“Stage A complete” means that the planned one-epoch general-domain pretraining run, checkpoint selection, held-out evaluation, artifact verification, and promotion are complete. It does **not** mean that the model is ready for medical use; continual medical pretraining, supervised fine-tuning, task evaluation, and safety evaluation remain future stages.

For the complete implementation and experiment history, see [Stage A Implementation and Training Report](reports/stage_a/STAGE_A_IMPLEMENTATION_AND_TRAINING_REPORT.md).

Machine-readable results:

- [Stage A evaluation](reports/stage_a/stage_a_evaluation.json)
- [Promoted checkpoint pointer](reports/stage_a/promoted_stage_a.json)
- [Stage A generation smoke test](reports/stage_a/stage_a_generation_smoke_test.json)

## Stage A model

| Setting | Value |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Unique trainable parameters | 35,463,680 |
| Vocabulary size | 16,000 |
| Hidden size | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Head dimension | 64 |
| Intermediate size | 1,536 |
| Maximum positions | 1,024 |
| Stage A sequence length | 256 |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Positions | RoPE |
| Embeddings | Input/output weights tied |
| Dropout | 0.0 |

The 1,024-position architecture leaves room for later continual pretraining or supervised fine-tuning at longer sequence lengths even though Stage A uses length 256.

The implementation uses pre-normalized decoder blocks, PyTorch scaled-dot-product causal attention, rotary embeddings on queries and keys, bias-free attention/MLP projections, and a tied token-embedding/language-model head.

## Training data contract

The Stage A dataset is a packed causal-language-model dataset stored in binary `uint16` shards. Each record contains 257 token IDs and produces 256 supervised positions:

```python
input_ids = sample[:-1]
labels = sample[1:]
```

The labels are already shifted by the dataset. The pretraining loss therefore compares logits and labels directly and must not perform a second causal shift. Supervised fine-tuning uses a separate masked causal-loss function because its response mask follows standard causal-label alignment.

Key dataset properties:

| Property | Value |
|---|---:|
| Sequences | 935,642 |
| Binary shards | 115 |
| Sequence length | 256 |
| Stored sample width | 257 |
| Document separator | EOS |
| Label strategy | Shift performed in dataset |

Generated tokenized data and large model artifacts are intentionally excluded from Git.

Evaluation coverage used for final model selection and reporting:

| Split | Samples | Target tokens | Purpose |
|---|---:|---:|---|
| Validation | 1,822 | 466,432 | Compare candidate checkpoints |
| Test | 1,185 | 303,360 | Evaluate the selected checkpoint once |

## Implemented training system

The training system includes:

- Direct packed shifted-label causal loss with no second shift.
- Separate response-masked SFT causal loss.
- Deterministic epoch-seeded batch ordering.
- Explicit resumable epoch and batch cursor.
- AdamW decay and no-decay parameter groups.
- Linear warmup followed by cosine learning-rate decay.
- FP32, BF16, and gradient-scaled FP16 precision policies.
- Gradient accumulation and gradient clipping.
- Token-weighted evaluation and perplexity calculation.
- Structured JSONL training metrics.
- Atomic, immutable, checksummed checkpoints.
- Python, NumPy, PyTorch CPU, and CUDA RNG restoration.
- Dataset-manifest and tokenizer compatibility validation.
- Local retention and verified Google Drive checkpoint mirroring.
- Best, latest, milestone, and final checkpoint pointers.

## Repository layout

```text
configs/                       Model and platform training configurations
notebooks/colab_stage_a.ipynb  Stage A Colab training, resume, and evaluation workflow
notebooks/colab_stage_b.ipynb  Stage B zero-update Colab baseline gate
reports/stage_a/               Permanent Stage A reports and promotion metadata
scripts/training/              Stage A command-line entry point
src/medical_slm/model/         Decoder architecture
src/medical_slm/training/      Loss, sampler, optimizer, scheduler, trainer, and checkpoints
src/medical_slm/data/          Data preparation and tokenized-dataset implementation
src/medical_slm/tokenizer/     Tokenizer training and evaluation
tests/                         Unit and end-to-end regression tests
```

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm that the package and PyTorch runtime can be imported:

```bash
python -c "import torch, medical_slm; print(torch.__version__)"
```

## Tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

The current complete regression result, including the post-training generation tests, is:

```text
365 passed in 9.97s
```

The suite includes targeted tests for double-shifting, SFT masking, deterministic sampler resume, scheduler boundaries, batch-size-invariant evaluation, checkpoint corruption, compatibility checks, RNG restoration, exact resumed trajectories, Drive mirroring, and tiny end-to-end training.

The passing count is historical evidence from the completed Stage A implementation. Rerun the suite after changing code, dependencies, data contracts, or checkpoint logic.

## Use the promoted Stage A model

Before beginning continual pretraining, run the post-training inference gate against the promoted checkpoint:

```bash
python scripts/evaluation/check_stage_a_model.py
```

The command defaults to:

- Promotion pointer: `reports/stage_a/promoted_stage_a.json`
- Checkpoint root: `artifacts/training/stage_a/checkpoints`
- Tokenizer: `artifacts/tokenizer/tokenizer.json`
- Output report: `reports/stage_a/stage_a_generation_smoke_test.json`
- Device and precision: automatically selected

It verifies the complete checkpoint manifest before loading, checks the tokenizer SHA-256 and vocabulary size, strictly loads the saved weights, checks the parameter count and forward-logit shape, rejects NaN/Inf logits, and generates continuations from several fixed prompts.

Use custom prompts by repeating `--prompt`:

```bash
python scripts/evaluation/check_stage_a_model.py \
  --prompt "Once upon a time" \
  --prompt "The human heart pumps" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Use greedy decoding for a deterministic forward/generation check:

```bash
python scripts/evaluation/check_stage_a_model.py \
  --prompt "Scientists study the natural world by" \
  --max-new-tokens 32 \
  --temperature 0
```

On Colab, point directly to the verified Drive checkpoint if it has not been copied to the runtime SSD:

```bash
python scripts/evaluation/check_stage_a_model.py \
  --checkpoint /content/drive/MyDrive/medical-slm-runs/stage_a/checkpoints/checkpoint_00007250 \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --output /content/drive/MyDrive/medical-slm-runs/stage_a/stage_a_generation_smoke_test.json
```

A successful run prints `Stage A model smoke test: PASSED`. Review all generated continuations as a qualitative sanity check, but interpret them as raw next-token completions. Stage A was not instruction-tuned, so it is not expected to follow chat instructions reliably. Fluency is useful evidence that inference works; it is not evidence of medical correctness or safety.

The promoted checkpoint passed this gate on CPU/FP32 with four seeded prompts. The model produced decodable, sentence-like continuations and stopped on EOS where applicable. The samples also contained repetition and medically unreliable phrasing. That combination is expected for an early base model: it confirms that inference is operational while showing why continual medical pretraining, instruction tuning, and factual/safety evaluation remain necessary.

The next stage should begin only after the smoke test passes and its JSON report is retained with the Stage A evaluation artifacts.

## Local Stage A commands

Run the one-batch overfitting smoke test:

```bash
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --overfit-one-batch \
  --max-updates 10
```

Run a bounded development training:

```bash
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --max-updates 100
```

Start the configured Stage A training run:

```bash
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml
```

Resume from the latest checkpoint:

```bash
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --resume latest
```

The training CLI also accepts `--model-config` when a model configuration other than `configs/model_stage_a.yaml` is required.

## Google Colab

The complete Colab workflow is in [notebooks/colab_stage_a.ipynb](notebooks/colab_stage_a.ipynb). It includes:

- Repository checkout and installation.
- Google Drive mounting.
- Dataset staging on the runtime SSD.
- Hardware and precision checks.
- Initial random-validation baseline gate.
- Development training and checkpoint mirroring.
- Runtime restart and exact resume test.
- Self-contained fresh full-run and resume sections.
- Full validation comparison.
- One-time held-out test evaluation.
- Checkpoint integrity verification and promotion.

Automatic precision should remain enabled when GPU allocation is uncertain. The completed T4 run correctly selected FP16 with a gradient scaler. A compatible GPU with reliable native BF16 support can select BF16 without the scaler.

Platform profiles are available at:

- `configs/training_stage_a_colab.yaml`
- `configs/training_stage_a_runpod.yaml`

## Checkpoints and preservation

The promoted checkpoint is expected locally at:

```text
artifacts/training/stage_a/checkpoints/checkpoint_00007250
```

It contains model, optimizer, scheduler, scaler, RNG, configuration, environment, metrics, and trainer-state artifacts plus a SHA-256 manifest. The local copy was audited successfully: all nine declared artifacts passed size and hash verification, and its tokenizer and dataset-manifest compatibility hashes matched the project files.

Checkpoint binaries are excluded from Git because the promoted checkpoint is approximately 406 MiB. Keep both of these checkpoints in durable external storage:

- `checkpoint_00007250` — promoted best-validation model.
- `checkpoint_00007310` — exact final end-of-epoch state.

The small JSON and Markdown files under `reports/stage_a/` should be committed to preserve the experiment record.

| Artifact | Role | Git policy |
|---|---|---|
| `checkpoint_00007250` | Promoted best-validation checkpoint | External durable storage; excluded from Git |
| `checkpoint_00007310` | Exact final end-of-epoch checkpoint | External durable storage; excluded from Git |
| `reports/stage_a/stage_a_evaluation.json` | Machine-readable final evaluation | Commit |
| `reports/stage_a/promoted_stage_a.json` | Machine-readable promotion pointer | Commit |
| `reports/stage_a/stage_a_generation_smoke_test.json` | Post-training inference evidence | Commit |
| `reports/stage_a/STAGE_A_IMPLEMENTATION_AND_TRAINING_REPORT.md` | Full experiment record | Commit |

Do not retain only `model.pt` if exact resume matters. Optimizer, scheduler, scaler, RNG, trainer-state, configuration, and manifest files are all part of the resumable checkpoint contract.

## Reproducibility snapshot

The promoted checkpoint records the following run identity:

| Property | Recorded value |
|---|---|
| Checkpoint | `checkpoint_00007250` |
| Seed | 42 |
| GPU | Tesla T4 |
| Precision | FP16 with gradient scaler |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| Dataset-manifest SHA-256 | `ac5f8111b861c5665e3ee98e548c1813a793b6bcb245293150317ee1fae39c7c` |
| Tokenizer SHA-256 | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` |

Exact numerical reproduction still depends on compatible hardware, CUDA/PyTorch kernels, package versions, and deterministic behavior of the selected backend. The saved RNG states and explicit batch cursor support exact continuation in a compatible environment; they do not guarantee bit-identical results across arbitrary hardware or software stacks.

## Tokenizer comparison

The project trains a custom GPT-2-style ByteLevel BPE tokenizer and provides utilities to compare it with the original GPT-2 tokenizer. Comparison metrics include:

- Vocabulary size and utilization.
- Total token count.
- Tokens per word.
- Characters and bytes per token.
- Unknown-token rate.
- Document sequence lengths.
- Medical-term fragmentation.

Tokenizer tooling is located under `scripts/tokenizer/` and `src/medical_slm/tokenizer/`.

## Stage B continual pretraining

Stage B is prepared but training has not started. It will initialize from promoted Stage A checkpoint `checkpoint_00007250`, establish untouched medical and general validation baselines, and then continually pretrain for one epoch over the disjoint medical/rehearsal corpus. The immediate next gate is the zero-update dual-validation baseline on Colab.

The promoted Stage A checkpoint is a pretraining baseline, not a deployable medical assistant.

### Stage B data readiness

The Stage B data foundation is now prepared and verified:

- 224,120,320 effective training targets across 875,470 packed sequences.
- 82.22% medical data and 17.78% general rehearsal data.
- Zero document overlap with Stage A.
- Separate medical validation and sealed medical test splits.
- Zero overlap among either training stage, medical validation, and medical test.
- All 109 new binary shards passed SHA-256 verification.

See [Stage B Data Preparation Report](reports/stage_b/DATA_PREPARATION_REPORT.md) for exact source allocations, hashes, provenance, and audit results.

### Stage B training system

The continual-pretraining trainer is implemented and has passed its real parent-initialization gate. It:

- Verifies the complete promoted Stage A checkpoint.
- Loads only model weights from `checkpoint_00007250`.
- Starts a fresh optimizer, scheduler, precision scaler, RNG progression, and training state.
- Records parent model/checkpoint hashes in every Stage B checkpoint.
- Establishes medical-adaptation and general-retention baselines before training.
- Selects the best medical checkpoint subject to a 5% general-loss degradation ceiling.
- Writes `best_medical`, `best_eligible`, `latest`, milestone, and `final_stage_b` pointers.
- Supports exact Stage B interruption and resume.

Verify model-only initialization without evaluation or training:

```bash
python scripts/training/train_stage_b.py --verify-initialization-only
```

Evaluate the untouched Stage A model on medical and general validation:

```bash
python scripts/training/train_stage_b.py --baseline-only
```

Run a short alignment diagnostic:

```bash
python scripts/training/train_stage_b.py \
  --overfit-one-batch \
  --max-updates 10
```

Run a bounded development training:

```bash
python scripts/training/train_stage_b.py --max-updates 50
```

Resume a Stage B development or full run:

```bash
python scripts/training/train_stage_b.py --resume latest
```

Platform profiles are available at:

- `configs/training_stage_b.yaml`
- `configs/training_stage_b_colab.yaml`
- `configs/training_stage_b_runpod.yaml`

See [Stage B Training System Report](reports/stage_b/TRAINING_SYSTEM_REPORT.md) for initialization, lineage, dual-validation, promotion, and test details.

For the complete Colab workflow, use [notebooks/colab_stage_b.ipynb](notebooks/colab_stage_b.ipynb) and follow the [Stage B Colab Baseline Guide](reports/stage_b/COLAB_BASELINE_GUIDE.md). The notebook includes the zero-update baseline, development run, 50-to-100 resume test, standalone fresh full run, standalone interrupted-run resume, and final-state verification. Development and full checkpoints use separate Drive directories. The upload-ready `stage-b-data.tar` archive is generated locally and excluded from Git.
