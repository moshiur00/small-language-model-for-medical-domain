# Medical Small Language Model

**A 35.5M-parameter decoder-only Transformer built and trained from scratch, then
adapted to medical text with retention-aware continual pretraining.**

`Python 3.11+` · `PyTorch` · `Custom 16K BPE tokenizer` · `396 tests` ·
`Deterministic resume` · `Colab/RunPod workflows`

This portfolio project implements the complete path from raw text to a promoted
language-model checkpoint: source standardization, licensing and quality controls,
deduplication, tokenizer training, packed binary datasets, native Transformer
architecture, mixed-precision training, exact crash resume, controlled continual-
pretraining experiments, dual-domain evaluation, artifact preservation, and
autoregressive inference.

> [!WARNING]
> This is a research base model, not a clinical system. It is not instruction-tuned
> and has not been validated for medical factuality, diagnosis, treatment advice,
> patient safety, or real-world clinical use. Generated text must not be used for
> medical decisions.

## Project status

| Stage | Status | Outcome |
|---|---|---|
| Data and tokenizer | Complete | Licensed, filtered corpora; custom 16,000-token ByteLevel BPE |
| Stage A general pretraining | Complete | One epoch from random initialization; checkpoint `00007250` promoted |
| Stage B v1 continual pretraining | Complete, comparison only | Exposed catastrophic forgetting under an overly strict retention rule |
| Stage B v2 retention-aware pretraining | **Complete and promoted** | Checkpoint `00008000` improved medical validation while remaining inside the declared retention band |
| Stage B v2 inference gate | **Passed** | Checkpoint identity, lineage, tokenizer compatibility, finite logits, and generation verified |
| LoRA comparison | Planned | Must use the same Stage A parent and validation-only selection contract |
| Stage C supervised fine-tuning | Dataset v1 rebuilt and verified; training not started | 6,967 examples in grouped train/validation/sealed-test splits |

The current promoted domain-adapted model is **Stage B v2
`checkpoint_00008000`**. Its full checkpoint is preserved locally and in Google
Drive; large binary artifacts are intentionally excluded from Git.

## Results at a glance

### Validation-controlled experiment comparison

| Experiment | Training tokens | Medical validation loss | General validation loss | Selection outcome |
|---|---:|---:|---:|---|
| Stage A parent | 239,524,352 | 3.467505 | **3.198383** | General-pretraining baseline |
| Stage B v1 best eligible, update 250 | 8,192,000 | 3.374685 | 3.330027 | Promoted under v1's 5% loss ceiling |
| Stage B v1 full endpoint, update 6,840 | 224,120,320 | **3.009167** | 3.749123 | Rejected: severe general-domain forgetting |
| **Stage B v2, update 8,000** | **262,144,000** | **3.135652** | **3.348985** | **Promoted inside the preferred retention band** |

Stage B v1 is intentionally retained as a negative experimental result: optimizing
medical loss alone produced a stronger in-domain loss but unacceptable general-
domain regression. Stage B v2 addressed that failure with more general rehearsal,
a lower learning rate, earlier dual validation, explicit retention bands, and
matched pilot experiments.

### Final Stage B v2 evaluation

| Distribution | Validation loss | Validation perplexity | Test loss | Test perplexity |
|---|---:|---:|---:|---:|
| Medical | **3.135652** | **23.004** | **3.162896** | **23.639** |
| General | 3.348985 | 28.474 | 3.691913 | 40.122 |

Compared with the Stage A parent:

- Medical validation perplexity decreased from **32.057 to 23.004**.
- General validation perplexity increased by **16.25%**, inside the predeclared
  preferred 20% retention budget.
- General test perplexity increased from **39.613 to 40.122**, approximately 1.28%.
- Medical test perplexity was approximately **20.75% lower** than Stage B v1's
  retention-selected checkpoint.

Checkpoint selection used validation data only. Medical and general test splits
were opened once after the selected checkpoint was fixed. Those test results are now
known and must not be used to tune future LoRA or SFT experiments.

## What this project demonstrates

- A decoder-only Transformer implemented directly in PyTorch rather than delegated
  to a pretrained Hugging Face model.
- A custom tokenizer and binary packed-dataset format with manifest-level
  provenance and SHA-256 integrity checks.
- Correct direct loss for already-shifted packed labels, preventing the common
  double-shifting failure mode.
- Deterministic, epoch-seeded shuffling with an explicit resumable batch cursor.
- Mixed FP32/BF16/FP16 execution, gradient accumulation, clipping, and FP16 loss
  scaling.
- Atomic, immutable, resumable checkpoints containing all optimizer and RNG state.
- Parent-child checkpoint lineage for continual pretraining.
- Controlled adaptation experiments that treat catastrophic forgetting as a
  measurable constraint rather than a qualitative afterthought.
- Validation-only promotion, sealed-test discipline, structured JSONL metrics, and
  machine-readable experiment records.
- Google Colab Pro workflows with local-SSD data staging and verified Google Drive
  checkpoint mirroring.
- Physical preservation of promoted and final checkpoints with archive-level hashes.

## System overview

```mermaid
flowchart LR
    A[Raw general and medical sources] --> B[Standardize and clean]
    B --> C[License, language, quality, toxicity checks]
    C --> D[Exact and MinHash deduplication]
    D --> E[Custom 16K ByteLevel BPE]
    E --> F[EOS-packed uint16 shards]
    F --> G[Stage A: train from scratch]
    G --> H[Promote checkpoint 00007250]
    H --> I[Stage B v2 matched pilots]
    I --> J[Retention-aware full run]
    J --> K[Medical + general validation]
    K --> L[Promote checkpoint 00008000]
    L --> M[Integrity and inference gates]
```

The same model architecture and tokenizer are retained across Stage A and Stage B.
Stage B starts from Stage A model weights but deliberately initializes a fresh
optimizer, scheduler, precision scaler, RNG progression, and training state.

## Model architecture

| Component | Final configuration |
|---|---:|
| Architecture | Decoder-only causal Transformer |
| Unique parameters | 35,463,680 |
| Vocabulary | 16,000 |
| Hidden size | 512 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Head dimension | 64 |
| SwiGLU intermediate size | 1,536 |
| Maximum positions | 1,024 |
| Training sequence length | 256 |
| Normalization | Pre-norm RMSNorm, epsilon `1e-5` |
| Positional encoding | RoPE, theta `10,000` |
| Attention | PyTorch scaled-dot-product causal attention |
| Projections | Bias-free attention and MLP projections |
| Embeddings | Input embedding and LM head weights tied |
| Dropout | 0.0 |
| Initialization standard deviation | 0.02 |

The 1,024-position architecture permits later training or fine-tuning at longer
contexts even though current packed pretraining examples contain 256 supervised
positions.

Implementation: [`src/medical_slm/model/`](src/medical_slm/model/)

## Data engineering

The repository includes reusable pipelines for:

- source-specific schema standardization;
- text normalization and cleaning;
- license normalization and policy enforcement;
- FastText language verification;
- heuristic quality scoring and filtering;
- context-aware toxicity auditing;
- within-source and global exact deduplication;
- MinHash near-duplicate detection;
- corpus assembly with token budgets and document exclusions;
- tokenizer training, evaluation, and GPT-2 comparison;
- deterministic EOS-separated packing into binary shards.

Source adapters cover general and medical datasets including FineWeb-Edu,
Wikipedia, WikiText, Project Gutenberg, TinyStories, PubMed abstracts, PMC Open
Access, WikiDoc, PubMedQA, MedMCQA, ChatDoctor, MedAlpaca, MedInstruct, and
OpenMedicalInstruct-style records. Inclusion in a final corpus remains governed by
the configured license, quality, overlap, and phase-allocation policies.

### Tokenizer

The model uses a project-trained GPT-2-style ByteLevel BPE tokenizer:

| Property | Value |
|---|---:|
| Vocabulary size | 16,000 |
| Document separator | `<eos>` |
| Required generation tokens | `<bos>`, `<eos>` |
| Tokenizer SHA-256 | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` |

Tokenizer evaluation reports vocabulary utilization, token/word and byte/token
efficiency, unknown-token rate, document lengths, and medical-term fragmentation.

### Packed causal-training contract

Each stored record contains 257 `uint16` token IDs and yields 256 supervised targets:

```python
input_ids = sample[:-1]
labels = sample[1:]
```

The dataset has already performed the causal shift. Pretraining therefore computes
cross-entropy directly between `logits[:, t]` and `labels[:, t]`; shifting again
inside the model would train against the wrong target. SFT uses a separate loss path
that performs standard causal alignment and excludes non-response tokens through an
ignore mask.

### Stage A corpus

| Property | Value |
|---|---:|
| Packed sequences | 935,642 |
| Supervised tokens | 239,524,352 |
| Binary shards | 115 |
| Sequence length | 256 |
| General validation | 1,822 samples / 466,432 tokens |
| General test | 1,185 samples / 303,360 tokens |

### Stage B v2 corpus

| Property | Verified value |
|---|---:|
| Processed documents | 237,958 |
| Exact stream tokens | 264,230,631 |
| Medical tokens | 184,996,718 (70.0134%) |
| General rehearsal tokens | 79,233,913 (29.9866%) |
| Packed sequences | 1,028,134 |
| Supervised targets | 263,202,304 |
| Binary shards | 126 |
| Stage A document overlap | **0** |
| Medical/general evaluation overlap | **0** |

The corpus uses PMC Open Access, PubMed abstracts, and WikiDoc for medical exposure,
with FineWeb-Edu, WikiText-103, and public-domain Project Gutenberg text for general
rehearsal. All 126 shard sizes and SHA-256 values match the generated manifest.

Detailed data evidence:

- [Stage B v2 dataset specification](reports/stage_b/v2/DATASET_SPECIFICATION.md)
- [Source audit](reports/stage_b/v2/source_audit.json)
- [Corpus verification](reports/stage_b/v2/corpus_verification.json)
- [Stage B data-preparation report](reports/stage_b/DATA_PREPARATION_REPORT.md)

## Stage C supervised instruction data

Stage C v1 uses only the existing balanced seven-source instruction pool. The
corrected builder preserved instructions during truncation, rejected 33 invalid or
overlength examples, and produced 6,248 train, 369 validation, and 350 sealed test
examples. All tensor and structured artifacts match their manifest hashes, with zero
record-ID or normalized prompt-group overlap across splits. Cross-source near-
duplicate analysis found no cross-source or cross-split leakage. The dataset is
approved for internal research training. Public checkpoint release remains blocked
because 2,999 examples carry noncommercial or manual-review license metadata.

- [Stage C dataset specification](reports/stage_c/DATASET_SPECIFICATION.md)
- [Stage C experiment plan](reports/stage_c/EXPERIMENT_PLAN.md)
- [Stage C duplicate and license audit](reports/stage_c/stage_c_data_audit.json)

## Training system

### Optimization

| Setting | Stage A | Stage B v2 |
|---|---:|---:|
| Initialization | Random | Stage A model weights |
| Optimizer | AdamW | AdamW |
| Peak learning rate | `3e-4` | `4e-5` |
| Final learning rate | `3e-5` | `4e-6` |
| Betas | `(0.9, 0.95)` | `(0.9, 0.95)` |
| Weight decay | 0.1 | 0.05 |
| Schedule | Linear warmup + cosine | Linear warmup + cosine |
| Warmup updates | 73 | 161 |
| Micro-batch | 16 sequences | 16 sequences |
| Gradient accumulation | 8 | 8 |
| Nominal global batch | 32,768 tokens | 32,768 tokens |
| Gradient clipping | 1.0 | 1.0 |
| Precision on completed T4 runs | FP16 + GradScaler | FP16 + GradScaler |
| Epochs | 1 | 1 |

BF16 is selected automatically on compatible hardware; FP16 uses a saved gradient
scaler, and CPU execution falls back to FP32.

### Determinism and exact resume

The sampler derives each epoch's batch order from the configured seed. Training
state stores both `epoch` and the next `batch_cursor`, avoiding replay or omission
after interruption. A resumable checkpoint contains:

- model, optimizer, scheduler, and optional FP16 scaler state;
- update count, consumed samples/tokens, epoch, and batch cursor;
- Python, NumPy, PyTorch CPU, and CUDA RNG states;
- model and training configurations;
- tokenizer and dataset-manifest compatibility hashes;
- recent structured metrics and best-validation state;
- environment metadata and an artifact SHA-256 manifest.

Checkpoint directories are written atomically and treated as immutable. Small JSON
pointers identify `latest`, best, milestones, and final states. Resume tests verify
that interrupted training continues at the next exact batch rather than restarting
an epoch.

### Stage B v2 retention-aware selection

All Stage B experiments begin from the exact Stage A parent identity. Three
500-update pilots consumed the same batches in the same order:

| Pilot | Trainable scope | L2-SP | Medical validation loss | General validation loss | General PPL degradation |
|---|---|---:|---:|---:|---:|
| **Control** | All parameters | 0 | **3.371878** | 3.242126 | 4.47% |
| Selective freezing | Upper layers | 0 | 3.406951 | 3.207393 | 0.91% |
| Selective freezing + L2-SP | Upper layers | Enabled | 3.451979 | **3.199858** | **0.15%** |

Because all candidates were inside the preferred retention band, the predeclared
rule selected the control arm with the best medical loss. The full run then selected
the lowest medical-validation loss among checkpoints satisfying the retention
contract:

- preferred general-perplexity degradation: at most 20%;
- hard fallback limit: at most 25%;
- emergency threshold: 35% for two consecutive validations;
- test data prohibited from model or checkpoint selection.

This design makes the adaptation/retention trade-off explicit and preserves the
selective-freezing and L2-SP pilots as reproducible evidence rather than assuming
one regularization strategy would be best.

Implementation: [`src/medical_slm/training/`](src/medical_slm/training/)

## Evaluation and inference

Evaluation aggregates summed token loss before division, so loss and perplexity are
invariant to evaluation batch size. Training emits structured JSONL events for loss,
learning rate, gradient norm, throughput, validation metrics, memory, skipped
updates, and non-finite events.

Both promoted checkpoints have repository-native inference gates. They verify the
full checkpoint before loading, enforce tokenizer compatibility, strictly load
weights, reject non-finite logits, validate output shape, and run seeded top-k/top-p
or greedy autoregressive decoding.

Run the promoted Stage B v2 model:

```bash
python scripts/evaluation/check_stage_b_v2_model.py
```

Use custom prompts:

```bash
python scripts/evaluation/check_stage_b_v2_model.py \
  --prompt "The human heart pumps blood through" \
  --prompt "Antibiotic resistance occurs when" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Use deterministic greedy decoding with `--temperature 0`. The default report is
written to
[`reports/stage_b/v2/stage_b_v2_generation_smoke_test.json`](reports/stage_b/v2/stage_b_v2_generation_smoke_test.json).

The permanent Stage B v2 smoke test passed on CPU/FP32 with four seeded prompts,
verified all nine checkpoint artifacts, loaded all 35,463,680 parameters, and
produced finite, decodable continuations. This demonstrates operational inference,
not factual or clinical quality. Generation currently recomputes the full context
without a key/value cache and is intended as a verification path rather than a
production serving stack.

Stage A inference remains available:

```bash
python scripts/evaluation/check_stage_a_model.py
```

## Installation and quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/moshiru00/small-language-model-for-medical-domain.git
cd small-language-model-for-medical-domain
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the complete regression suite:

```bash
python -m pytest -q
```

Verified result at the current repository state:

```text
396 passed in 11.33s
```

Targeted coverage includes packed-label alignment, SFT response masking,
deterministic sampler order, exact batch-cursor resume, scheduler boundaries,
precision policy, token-weighted evaluation, optimizer grouping, checkpoint
corruption and compatibility failures, RNG restoration, Drive mirroring, Stage B
lineage and retention policy, preservation exports, inference identity checks, and
tiny end-to-end training.

## Training commands

Large training runs require the prepared binary datasets and parent checkpoints,
which are not stored in Git.

### Stage A

```bash
# One-batch alignment/overfit test
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --overfit-one-batch \
  --max-updates 10

# Bounded development run
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --max-updates 100

# Full configured run
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml

# Exact resume
python scripts/training/train_stage_a.py \
  --config configs/training_stage_a.yaml \
  --resume latest
```

### Stage B v2

```bash
# Reproduce the 500-update full-parameter control pilot
python scripts/training/train_stage_b_v2.py \
  --config configs/training_stage_b_v2_control.yaml \
  --max-updates 500

# Resume an interrupted run from its latest verified checkpoint
python scripts/training/train_stage_b_v2.py \
  --config configs/training_stage_b_v2_control.yaml \
  --resume latest
```

The tracked pilot profile is intentionally bounded to 500 updates. The full
8,033-update experiment used an isolated runtime configuration generated by the
Stage B v2 Colab notebook, with separate output and Drive-backup directories so a
pilot could never be mistaken for or overwrite the production run.

Alternative matched-pilot configurations are retained at:

- [`configs/training_stage_b_v2_selective.yaml`](configs/training_stage_b_v2_selective.yaml)
- [`configs/training_stage_b_v2_selective_l2sp.yaml`](configs/training_stage_b_v2_selective_l2sp.yaml)

## Cloud execution

The completed runs used Google Colab Pro with a Tesla T4 and automatic FP16
selection. The notebooks stage data from Drive onto the runtime SSD, mirror verified
checkpoints back to Drive, and support recovery after runtime disconnection.

- [Stage A Colab notebook](notebooks/colab_stage_a.ipynb)
- [Stage B v1 Colab notebook](notebooks/colab_stage_b.ipynb)
- [Stage B v2 self-contained notebook](notebooks/colab_stage_b_v2.ipynb)
- [Stage B v2 Colab pilot guide](reports/stage_b/v2/COLAB_PILOT_GUIDE.md)

Stage A also provides Colab and RunPod profiles:

- [`configs/training_stage_a_colab.yaml`](configs/training_stage_a_colab.yaml)
- [`configs/training_stage_a_runpod.yaml`](configs/training_stage_a_runpod.yaml)

## Artifact preservation and reproducibility

### Promoted identities

| Artifact | Stage A parent | Stage B v2 promoted |
|---|---|---|
| Checkpoint | `checkpoint_00007250` | `checkpoint_00008000` |
| Model SHA-256 | `2443cb5875c11e9c0c027ead53d4f9adab099e0cd4b19fd47fe08181b0640423` | `799fe9c34648044d21bf73258cd55a46716167f89dcf64e2c0487e0382d65c14` |
| Checkpoint manifest SHA-256 | `a7e132692f31b505b7d7db9fa7e6d773d5c006904fa0d774edb1ba6c7f0408a4` | `ea60b0d6b66ea3bd1987f9ff7bbdd75ba34bc0f05b7c21349ad6ef90615a9b71` |
| Tokenizer SHA-256 | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` | Same tokenizer |

The Stage B v2 lineage embeds the exact Stage A parent model, checkpoint manifest,
and tokenizer hashes, plus train, medical-validation, and general-validation
manifest hashes.

### Preservation policy

| Asset | Storage policy |
|---|---|
| Source code, configs, tests, reports, pointers | Commit to Git |
| Tokenized shards and raw/intermediate datasets | External storage; excluded from Git |
| Full resumable checkpoints | Local protected artifacts plus durable cloud storage |
| Preservation archives | External storage with adjacent SHA-256 record |

The Stage B v2 preservation archive contains the promoted update-8,000 checkpoint,
the final update-8,033 checkpoint, metrics, configs, tokenizer, manifests, and reports.
It is **853,012,480 bytes** with SHA-256:

```text
b5e4f75623d1af74dd825c873d549109bd9ad42e0d150cfffc279f0dd9c64263
```

All 41 inventoried files and both preserved checkpoints passed size and hash
verification after local extraction.

Exact cross-machine numerical identity still depends on compatible PyTorch, CUDA,
GPU kernels, and deterministic backend behavior. The saved state supports exact
continuation in a compatible environment; it does not promise bitwise equality
across arbitrary hardware stacks.

## Repository layout

```text
configs/                   Model, corpus, platform, and experiment configurations
datasets/                  Generated corpora and packed shards; mostly Git-ignored
notebooks/                 Self-contained Colab training and recovery workflows
reports/
  stage_a/                 Stage A evaluation, promotion, and inference evidence
  stage_b/v1/              Catastrophic-forgetting comparison experiment
  stage_b/v2/              V2 plan, audits, evaluation, promotion, and preservation
  comparisons/             Cross-experiment adaptation registry
scripts/
  assembly/                Corpus construction and overlap audits
  artifacts/               Preservation export and verification
  evaluation/              Promoted-model inference gates
  training/                Stage A, Stage B v1, and Stage B v2 entry points
  tokenizer/               Tokenizer train/evaluate/compare utilities
src/medical_slm/
  data/                    Standardization, filtering, deduplication, packing, SFT
  inference/               Autoregressive decoding
  model/                   Transformer architecture
  tokenizer/               Tokenizer implementation and metrics
  training/                Loss, sampler, optimizer, scheduler, trainer, checkpoints
tests/                     Unit, regression, and tiny end-to-end tests
```

## Experiment records

- [Stage A implementation and training report](reports/stage_a/STAGE_A_IMPLEMENTATION_AND_TRAINING_REPORT.md)
- [Stage A final evaluation](reports/stage_a/stage_a_evaluation.json)
- [Stage B training-system report](reports/stage_b/TRAINING_SYSTEM_REPORT.md)
- [Stage B v1 experiment report](reports/stage_b/v1/EXPERIMENT_REPORT.md)
- [Stage B v2 experiment plan](reports/stage_b/v2/EXPERIMENT_PLAN.md)
- [Stage B v2 final report](reports/stage_b/v2/EXPERIMENT_REPORT.md)
- [Stage B v2 evaluation](reports/stage_b/v2/stage_b_v2_evaluation.json)
- [Promoted Stage B v2 pointer](reports/stage_b/v2/promoted_stage_b_v2.json)
- [Continual-adaptation comparison registry](reports/comparisons/continual_adaptation_registry.json)

## Limitations and next work

- Perplexity measures next-token prediction, not medical truthfulness or safety.
- The 35.5M-parameter model has limited capacity and should not be compared directly
  with production-scale medical LLMs.
- Generation is raw causal completion; the model has not undergone instruction
  tuning, preference optimization, retrieval augmentation, or clinical calibration.
- Evaluation currently emphasizes held-out language-model loss rather than medical
  QA accuracy, hallucination rate, calibration, bias, privacy, or adversarial safety.
- The known test sets are sealed from all future tuning decisions; new untouched
  benchmarks will be required for repeated model-development comparisons.
- The next active experiment is full-parameter Stage C supervised instruction fine-
  tuning from the promoted Stage B v2 checkpoint. A later LoRA run remains planned as
  a controlled parameter-efficient comparison.

The project demonstrates a reproducible training and experimentation system for a
small domain language model. It does **not** claim that the resulting checkpoint is a
medical expert or a deployable assistant.

## License

See [`LICENSE`](LICENSE). Dataset licenses and inclusion decisions are tracked
separately by the data-policy pipeline; the repository license does not override the
license of any upstream dataset.
