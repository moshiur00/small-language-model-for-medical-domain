# Medical Small Language Model

A from-scratch decoder-only language model and training pipeline intended for a staged medical-language-model curriculum. The repository covers data preparation, tokenizer training and evaluation, packed causal pretraining, exact checkpoint resume, cloud execution, and final checkpoint promotion.

## Current status

**Stage A pretraining is complete.** The model was trained from random initialization for one full epoch over the packed Stage A dataset on Google Colab Pro using a Tesla T4 and FP16.

| Item | Result |
|---|---:|
| Training sequences | 935,642 |
| Consumed training tokens | 239,524,352 |
| Optimizer updates | 7,310 |
| Skipped updates | 0 |
| Non-finite events | 0 |
| Best checkpoint | `checkpoint_00007250` |
| Final chronological checkpoint | `checkpoint_00007310` |
| Best validation loss | 3.198383 |
| Best validation perplexity | 24.492887 |
| Held-out test loss | 3.679168 |
| Held-out test perplexity | 39.613418 |

Checkpoint 7,250 was promoted because its full-validation loss was marginally lower than the final update-7,310 checkpoint. The test set was evaluated only after checkpoint selection.

For the complete implementation and experiment history, see [Stage A Implementation and Training Report](reports/stage_a/STAGE_A_IMPLEMENTATION_AND_TRAINING_REPORT.md).

Machine-readable results:

- [Stage A evaluation](reports/stage_a/stage_a_evaluation.json)
- [Promoted checkpoint pointer](reports/stage_a/promoted_stage_a.json)

## Stage A model

| Setting | Value |
|---|---:|
| Architecture | Decoder-only causal Transformer |
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
notebooks/colab_stage_a.ipynb  Colab development, full-run, resume, and evaluation workflow
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

## Tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

The last complete user-run Stage A regression result was:

```text
347 passed in 9.61s
```

The suite includes targeted tests for double-shifting, SFT masking, deterministic sampler resume, scheduler boundaries, batch-size-invariant evaluation, checkpoint corruption, compatibility checks, RNG restoration, exact resumed trajectories, Drive mirroring, and tiny end-to-end training.

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

## Next phase

The next curriculum step is continual medical-domain pretraining initialized from `checkpoint_00007250`. Its implementation plan should define the medical corpus and validation distribution, sequence length, learning-rate policy, token or epoch budget, general-domain retention evaluation, and a new promotion protocol before training begins.
