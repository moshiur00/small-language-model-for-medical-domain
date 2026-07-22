# Small Language Model - Medical Domain

**A 35.5M-parameter decoder-only Transformer built and trained from scratch,
adapted to medical text with retention-aware continual pretraining, and evaluated
through response-masked supervised instruction tuning.**

`Python 3.11+` · `PyTorch` · `Custom 16K BPE tokenizer` · `430+ regression tests` ·
`Deterministic resume` · `Colab/RunPod workflows`

This project implements the complete path from raw text to a promoted
language-model checkpoint: source standardization, licensing and quality controls,
deduplication, tokenizer training, packed binary datasets, native Transformer
architecture, mixed-precision training, exact crash resume, controlled continual-
pretraining experiments, dual-domain evaluation, artifact preservation, and
autoregressive inference.

> [!WARNING]
> This is a research model, not a clinical system. Its Stage C instruction tuning
> has not been validated for medical factuality, diagnosis, treatment advice,
> patient safety, or real-world clinical use. Generated text must not be used for
> medical decisions.

## Project status

| Stage                                  | Status                      | Outcome                                                                                                                                                                            |
| -------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data and tokenizer                     | Complete                    | Licensed, filtered corpora; custom 16,000-token ByteLevel BPE                                                                                                                      |
| Stage A general pretraining            | Complete                    | One epoch from random initialization; checkpoint `00007250` promoted                                                                                                               |
| Stage B v1 continual pretraining       | Complete, comparison only   | Exposed catastrophic forgetting under an overly strict retention rule                                                                                                              |
| Stage B v2 retention-aware pretraining | **Complete and promoted**   | Checkpoint `00008000` improved medical validation while remaining inside the declared retention band                                                                               |
| Stage B v2 inference gate              | **Passed (operational)**    | Checkpoint identity, lineage, tokenizer compatibility, finite logits, and decoding verified; this is not a factuality result                                                       |
| LoRA comparison                        | Planned                     | Must use the same Stage A parent and validation-only selection contract                                                                                                            |
| Stage C supervised fine-tuning         | **Complete; research-only** | Specialist checkpoint `00000588` improved token metrics on all seven test sources, but failed qualitative generation review; balanced checkpoint `00000125` retained as comparator |
| Stage D distillation/alignment         | Planned                     | Verified teacher-response distillation first, then conservative preference optimization on new validation and test contracts                                                       |

The current instruction model is Stage C's **medical-instruction specialist
`checkpoint_00000588`**, with balanced-retention `checkpoint_00000125` retained as
an explicit comparator. Both descend from promoted Stage B v2 `checkpoint_00008000`.
Large binary artifacts are preserved outside Git with cryptographic manifests.

## Results and evaluation

This project reports three different kinds of evidence and does not treat them as
interchangeable:

1. **Language-model evaluation** measures next-token loss and perplexity on packed
   general and medical text.
2. **Instruction evaluation** measures response-only loss, perplexity, and token
   accuracy on held-out instruction/response examples.
3. **Qualitative generation review** asks whether decoded answers are coherent,
   relevant, non-repetitive, and medically defensible.

The first two improved during training. The third exposed a significant limitation:
better teacher-forced token prediction did **not** produce reliable free-form medical
answers at this model scale.

### Validation-controlled experiment comparison

| Experiment                             | Training tokens | Medical validation loss | General validation loss | Selection outcome                                |
| -------------------------------------- | --------------: | ----------------------: | ----------------------: | ------------------------------------------------ |
| Stage A parent                         |     239,524,352 |                3.467505 |            **3.198383** | General-pretraining baseline                     |
| Stage B v1 best eligible, update 250   |       8,192,000 |                3.374685 |                3.330027 | Promoted under v1's 5% loss ceiling              |
| Stage B v1 full endpoint, update 6,840 |     224,120,320 |            **3.009167** |                3.749123 | Rejected: severe general-domain forgetting       |
| **Stage B v2, update 8,000**           | **262,144,000** |            **3.135652** |            **3.348985** | **Promoted inside the preferred retention band** |

Stage B v1 is intentionally retained as a negative experimental result: optimizing
medical loss alone produced a stronger in-domain loss but unacceptable general-
domain regression. Stage B v2 addressed that failure with more general rehearsal,
a lower learning rate, earlier dual validation, explicit retention bands, and
matched pilot experiments.

### Final Stage B v2 evaluation

| Distribution | Validation loss | Validation perplexity |    Test loss | Test perplexity |
| ------------ | --------------: | --------------------: | -----------: | --------------: |
| Medical      |    **3.135652** |            **23.004** | **3.162896** |      **23.639** |
| General      |        3.348985 |                28.474 |     3.691913 |          40.122 |

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

### Stage C instruction-tuning results

| Profile                            | Checkpoint | SFT test loss | SFT test PPL | Response-token accuracy | Role                      |
| ---------------------------------- | ---------: | ------------: | -----------: | ----------------------: | ------------------------- |
| Balanced retention                 |        125 |      3.120167 |       22.650 |                  42.79% | Preferred-band comparator |
| **Medical instruction specialist** |    **588** |  **2.879608** |   **17.807** |              **46.34%** | **Primary profile**       |

The primary specialist was registered before sealed-test access. Relative to the
balanced profile, it reduced sealed SFT perplexity by **21.38%**, improved response-
token accuracy by **3.55 percentage points**, and improved all seven constituent
sources. The balanced model remains available because it has better packed medical
and general retention. Test results report this tradeoff; they did not choose the
profiles.

### Stage C sealed-test evaluation by source

The sealed SFT test contains 350 examples and 68,903 supervised response tokens.
Checkpoint `588` improved response-only loss and token accuracy over checkpoint
`125` on every source, but the size of the gain varied considerably.

| Source          | Examples | Balanced loss | Specialist loss | Specialist token accuracy | Accuracy change |
| --------------- | -------: | ------------: | --------------: | ------------------------: | --------------: |
| Alpaca          |       54 |         3.142 |       **3.124** |                    41.08% |        +0.33 pp |
| ChatDoctor      |       41 |         3.698 |       **3.509** |                    37.62% |        +2.37 pp |
| MedAlpaca       |       48 |         2.216 |       **2.021** |                **59.13%** |        +3.38 pp |
| MedInstruct     |       48 |         2.508 |       **2.400** |                    50.92% |        +1.55 pp |
| MedMCQA         |       58 |         3.683 |       **3.444** |                    41.68% |        +3.23 pp |
| OpenMedInstruct |       52 |         3.102 |       **2.773** |                    47.48% |    **+4.98 pp** |
| PubMedQA        |       49 |         3.309 |       **3.119** |                    44.06% |        +3.06 pp |

**Interpretation.** The specialist consistently learned the response distributions
present in the seven datasets. The strongest relative perplexity improvement was on
OpenMedInstruct (28.05%), while Alpaca was nearly flat (1.80%). These measurements
are teacher-forced: the model sees the correct preceding response tokens while each
next token is scored. They therefore measure imitation of reference responses, not
whether the model can independently construct a correct answer.

### Adaptation versus retention

SFT improved instruction imitation while weakening the Stage B v2 language-model
capabilities. This is the central trade-off in the current result.

| Checkpoint                 | Medical packed-test PPL | Change vs Stage B v2 | General packed-test PPL | Change vs Stage B v2 |
| -------------------------- | ----------------------: | -------------------: | ----------------------: | -------------------: |
| Stage B v2 parent (`8000`) |              **23.639** |                    — |              **40.122** |                    — |
| Stage C balanced (`125`)   |                  25.904 |               +9.58% |                  42.625 |               +6.24% |
| Stage C specialist (`588`) |                  26.726 |              +13.06% |                  43.942 |               +9.52% |

The balanced profile stays within the predeclared preferred validation-retention
bands. The specialist trades additional retention for lower SFT loss and remains
inside the hard bands. Neither result establishes medical correctness.

### Qualitative generation review: failed

The specialist checkpoint passed the **operational** inference gate—its artifacts
verified, weights loaded, logits were finite, and decoding completed—but its sample
answers did not pass a human-readable quality check:

- A hypertension prompt produced a repetitive response dominated by variants of
  “I can understand your concern,” without explaining hypertension or when urgent
  care may be needed.
- An antibiotics prompt produced internally contradictory claims and suggested that
  antibiotics could be a strategy for viral infection, which is medically unsafe.
- Sampling exposed repetition, poor answer planning, weak instruction adherence,
  and factual instability even though sealed response-token metrics improved.

This is a useful negative result: **the loss curves and sealed token metrics are
valid, but they are insufficient proxies for generation quality.** The Stage C
checkpoints are retained as reproducible research artifacts and baselines, not as a
usable medical assistant.

### Evaluation discipline

- Pilot and checkpoint selection used validation data only.
- The two Stage C profiles were immutably registered before sealed-test access.
- The sealed SFT, medical, and general tests were evaluated once and were not used
  to change the registered profile roles.
- Every reported checkpoint is tied to model, tokenizer, dataset, parent-lineage,
  and manifest SHA-256 identities.
- Since the test results and qualitative prompts are now known, future distillation,
  LoRA, or preference experiments require new development sets and a new untouched
  final test set.

Machine-readable evidence is available in the
[Stage C test report](reports/stage_c/stage_c_test_evaluation.json),
[profile registration](reports/stage_c/stage_c_profile_registration.json), and
[promotion record](reports/stage_c/promoted_stage_c.json).

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
    L --> M[Stage C response-only SFT]
    M --> N[Register balanced + specialist]
    N --> O[One-time sealed test]
    O --> P[Promote, preserve, infer]
```

The same model architecture and tokenizer are retained across Stage A and Stage B.
Stage B starts from Stage A model weights but deliberately initializes a fresh
optimizer, scheduler, precision scaler, RNG progression, and training state.

## Model architecture

| Component                         |                         Final configuration |
| --------------------------------- | ------------------------------------------: |
| Architecture                      |             Decoder-only causal Transformer |
| Unique parameters                 |                                  35,463,680 |
| Vocabulary                        |                                      16,000 |
| Hidden size                       |                                         512 |
| Transformer layers                |                                           8 |
| Attention heads                   |                                           8 |
| Head dimension                    |                                          64 |
| SwiGLU intermediate size          |                                       1,536 |
| Maximum positions                 |                                       1,024 |
| Training sequence length          |                                         256 |
| Normalization                     |            Pre-norm RMSNorm, epsilon `1e-5` |
| Positional encoding               |                        RoPE, theta `10,000` |
| Attention                         | PyTorch scaled-dot-product causal attention |
| Projections                       |     Bias-free attention and MLP projections |
| Embeddings                        |    Input embedding and LM head weights tied |
| Dropout                           |                                         0.0 |
| Initialization standard deviation |                                        0.02 |

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

| Property                   |                                                              Value |
| -------------------------- | -----------------------------------------------------------------: |
| Vocabulary size            |                                                             16,000 |
| Document separator         |                                                            `<eos>` |
| Required generation tokens |                                                   `<bos>`, `<eos>` |
| Tokenizer SHA-256          | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` |

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

| Property           |                          Value |
| ------------------ | -----------------------------: |
| Packed sequences   |                        935,642 |
| Supervised tokens  |                    239,524,352 |
| Binary shards      |                            115 |
| Sequence length    |                            256 |
| General validation | 1,822 samples / 466,432 tokens |
| General test       | 1,185 samples / 303,360 tokens |

### Stage B v2 corpus

| Property                           |         Verified value |
| ---------------------------------- | ---------------------: |
| Processed documents                |                237,958 |
| Exact stream tokens                |            264,230,631 |
| Medical tokens                     | 184,996,718 (70.0134%) |
| General rehearsal tokens           |  79,233,913 (29.9866%) |
| Packed sequences                   |              1,028,134 |
| Supervised targets                 |            263,202,304 |
| Binary shards                      |                    126 |
| Stage A document overlap           |                  **0** |
| Medical/general evaluation overlap |                  **0** |

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

Stage C now has a dedicated response-only SFT path. It loads only model weights
from the promoted Stage B v2 checkpoint, creates fresh optimizer/scheduler/scaler
state, crops each fixed-width batch to its longest active sequence, and normalizes
the accumulated gradient by the total number of supervised response tokens. Prompt
and padding labels remain `-100`; the SFT loss performs exactly one standard causal
shift. The trainer never loads the sealed SFT test split during training or
selection.

The Stage C runtime evaluates three validation distributions every 25 updates:
response-only SFT validation, medical language-model retention, and general
language-model retention. Preferred and hard perplexity bands are 10% and 15% for
both retention distributions. Checkpoints preserve exact batch position, complete
RNG state, parent/model/tokenizer/dataset identity, and FP16 scaler state. The
canonical profile is 4 examples per micro-batch, 8-way accumulation, 3 epochs, and
588 optimizer updates.

Stage C entry points:

```powershell
# Verify parent loading and fresh optimization state
python scripts/training/train_stage_c_sft.py --verify-initialization-only

# Record the zero-update SFT/medical/general baselines
python scripts/training/train_stage_c_sft.py --baseline-only

# Confirm response-mask alignment on one repeated batch
python scripts/training/train_stage_c_sft.py --overfit-one-batch --max-updates 10

# Run or exactly resume training
python scripts/training/train_stage_c_sft.py
python scripts/training/train_stage_c_sft.py --resume latest
```

After the immutable dual-profile promotion has been written, inference verifies the
selected profile's identity and reproduces the exact Stage C prompt template:

```powershell
python scripts/evaluation/check_stage_c_model.py `
  --profile medical_instruction_specialist `
  --checkpoint-root artifacts/training/stage_c_sft_v1/checkpoints `
  --promotion reports/stage_c/promoted_stage_c.json `
  --test-evaluation reports/stage_c/stage_c_test_evaluation.json `
  --instruction "Explain hypertension in plain language."
```

Use `--profile balanced_retention` to inspect the preferred-retention comparator.
The generation gate establishes artifact integrity and operational inference, not
medical correctness or safety.

For Colab, create and upload the deterministic data bundle together with its
generated SHA-256 sidecar:

```powershell
python scripts/artifacts/create_stage_c_data_archive.py
```

Then open [the Stage C Colab notebook](notebooks/colab_stage_c_sft.ipynb). It
contains focused regression and initialization checks, the zero-update triple
baseline, one-batch response-mask verification, matched `1e-5`/`2e-5` pilots,
validation-only pilot selection, standalone fresh/resume full-run cells, immutable
balanced/specialist registration, and a guarded one-time test evaluator.

The test tensors are intentionally absent from the training archive. Only after
profile registration, create and upload the separate sealed archive and checksum:

```powershell
python scripts/artifacts/create_stage_c_test_archive.py
```

### Optimization

| Setting               |         Stage A |      Stage B v2 |     Stage C SFT v1 |
| --------------------- | --------------: | --------------: | -----------------: |
| Initialization        |          Random | Stage A weights | Stage B v2 weights |
| Optimizer             |           AdamW |           AdamW |              AdamW |
| Peak learning rate    |          `3e-4` |          `4e-5` |    `2e-5` selected |
| Final learning rate   |          `3e-5` |          `4e-6` |             `2e-6` |
| Betas                 |   `(0.9, 0.95)` |   `(0.9, 0.95)` |      `(0.9, 0.95)` |
| Weight decay          |             0.1 |            0.05 |               0.01 |
| Schedule              | Warmup + cosine | Warmup + cosine |    Warmup + cosine |
| Warmup updates        |              73 |             161 |                 30 |
| Micro-batch           |    16 sequences |    16 sequences |         4 examples |
| Gradient accumulation |               8 |               8 |                  8 |
| Nominal global batch  |   32,768 tokens |   32,768 tokens |        32 examples |
| Gradient clipping     |             1.0 |             1.0 |                1.0 |
| Precision on T4       |   FP16 + scaler |   FP16 + scaler |      FP16 + scaler |
| Epochs                |               1 |               1 |            Up to 3 |

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

| Pilot                      | Trainable scope |   L2-SP | Medical validation loss | General validation loss | General PPL degradation |
| -------------------------- | --------------- | ------: | ----------------------: | ----------------------: | ----------------------: |
| **Control**                | All parameters  |       0 |            **3.371878** |                3.242126 |                   4.47% |
| Selective freezing         | Upper layers    |       0 |                3.406951 |                3.207393 |                   0.91% |
| Selective freezing + L2-SP | Upper layers    | Enabled |                3.451979 |            **3.199858** |               **0.15%** |

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

The suite contains more than 430 tests. Targeted coverage includes packed-label
alignment, SFT response masking,
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
- [Stage C self-contained SFT notebook](notebooks/colab_stage_c_sft.ipynb)

Stage A also provides Colab and RunPod profiles:

- [`configs/training_stage_a_colab.yaml`](configs/training_stage_a_colab.yaml)
- [`configs/training_stage_a_runpod.yaml`](configs/training_stage_a_runpod.yaml)

## Artifact preservation and reproducibility

### Promoted identities

| Artifact                    | Stage A parent                                                     | Stage B v2 promoted                                                |
| --------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Checkpoint                  | `checkpoint_00007250`                                              | `checkpoint_00008000`                                              |
| Model SHA-256               | `2443cb5875c11e9c0c027ead53d4f9adab099e0cd4b19fd47fe08181b0640423` | `799fe9c34648044d21bf73258cd55a46716167f89dcf64e2c0487e0382d65c14` |
| Checkpoint manifest SHA-256 | `a7e132692f31b505b7d7db9fa7e6d773d5c006904fa0d774edb1ba6c7f0408a4` | `ea60b0d6b66ea3bd1987f9ff7bbdd75ba34bc0f05b7c21349ad6ef90615a9b71` |
| Tokenizer SHA-256           | `6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e` | Same tokenizer                                                     |

The Stage B v2 lineage embeds the exact Stage A parent model, checkpoint manifest,
and tokenizer hashes, plus train, medical-validation, and general-validation
manifest hashes.

Stage C preserves two registered profiles descended from Stage B v2:

| Profile                        | Checkpoint            | Model SHA-256                                                      | Checkpoint manifest SHA-256                                        |
| ------------------------------ | --------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Balanced retention             | `checkpoint_00000125` | `275939cf7bb544aecb461f98956da2791b820757beae818ce4e7590029a08f39` | `0cf0080de0b74f2a5406e2e47f83b58b92a9f01442fa9abd15a085f12a04fa8f` |
| Medical-instruction specialist | `checkpoint_00000588` | `231c0376ee4cd3b58216605c7ab24e1d76aae80404257b298e589f0695dd10d4` | `87200c551c89b418e5266ef1fd2e94b62557fb8cb20f2f02c5a97375285f3253` |

Both use the same verified tokenizer identity. The immutable registration and
promotion reports record that test data did not select either profile.

### Preservation policy

| Asset                                          | Storage policy                                       |
| ---------------------------------------------- | ---------------------------------------------------- |
| Source code, configs, tests, reports, pointers | Commit to Git                                        |
| Tokenized shards and raw/intermediate datasets | External storage; excluded from Git                  |
| Full resumable checkpoints                     | Local protected artifacts plus durable cloud storage |
| Preservation archives                          | External storage with adjacent SHA-256 record        |

The Stage B v2 preservation archive contains the promoted update-8,000 checkpoint,
the final update-8,033 checkpoint, metrics, configs, tokenizer, manifests, and reports.
It is **853,012,480 bytes** with SHA-256:

```text
b5e4f75623d1af74dd825c873d549109bd9ad42e0d150cfffc279f0dd9c64263
```

All 41 inventoried files and both preserved checkpoints passed size and hash
verification after local extraction.

The Stage C preservation archive contains both complete profile checkpoints,
metrics, contracts, tokenizer, promotion/evaluation records, and lineage evidence.
It is **852,684,800 bytes** with SHA-256:

```text
f96c66d9ced820aa5056473a9fdcec741939cb0ef826fffbaff88cc9037d4994
```

The extracted 37-file bundle and both checkpoints passed the repository's
preservation verifier. Large archives and model weights remain outside Git; their
small manifests and reports are tracked in the repository.

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
  stage_c/                 SFT data, selection, sealed test, promotion, and limitations
  comparisons/             Cross-experiment adaptation registry
scripts/
  assembly/                Corpus construction and overlap audits
  artifacts/               Preservation export and verification
  evaluation/              Selection, promotion, sealed evaluation, and inference gates
  training/                Stage A, Stage B, and Stage C entry points
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
- [Stage C experiment plan](reports/stage_c/EXPERIMENT_PLAN.md)
- [Stage C final experiment report](reports/stage_c/EXPERIMENT_REPORT.md)
- [Stage C sealed-test evaluation](reports/stage_c/stage_c_test_evaluation.json)
- [Stage C promotion record](reports/stage_c/promoted_stage_c.json)
- [Stage C preservation manifest](reports/stage_c/stage_c_preservation_manifest.json)
- [Continual-adaptation comparison registry](reports/comparisons/continual_adaptation_registry.json)

## Known limitations

### Model capacity and generation quality

- At 35.5M parameters, the model can learn local language and dataset patterns but
  has limited capacity for robust medical knowledge, multi-step reasoning, long-form
  answer planning, and reliable instruction following.
- Stage C's specialist improved response-only test perplexity and token accuracy on
  all seven sources, yet its open-ended answers were repetitive, poorly grounded,
  and sometimes medically incorrect. This metric/behavior mismatch is the most
  important unresolved result.
- The current model has no retrieval system, external knowledge source, citation
  verification, uncertainty calibration, or mechanism for checking its own claims.
- Decoding parameters may change fluency or repetition, but they cannot repair
  missing knowledge or make an unsafe checkpoint clinically reliable.

### Evaluation limits

- Perplexity and response-token accuracy measure prediction of a reference sequence;
  they do not establish factuality, reasoning, helpfulness, calibration, or safety.
- The project does not yet report clinician-reviewed correctness, hallucination
  rate, emergency-triage sensitivity, contraindication safety, demographic bias,
  memorization/privacy risk, or adversarial robustness.
- The qualitative review used a small diagnostic prompt set. It is enough to reject
  a claim of readiness, but not enough to estimate a population-level failure rate.
- All existing Stage C tests and diagnostic prompts are now known. Reusing them for
  model decisions would leak test information into development.

### Data and release limits

- Instruction data contain heterogeneous styles and quality levels; optimizing
  likelihood can reproduce verbosity, contradictions, and weak source answers.
- Some Stage C source licenses remain noncommercial or require manual review.
  Consequently, the promoted Stage C profiles are restricted to internal research
  and the checkpoints are not approved for public release.
- This is not a clinical system and must not be used for diagnosis, treatment,
  medication decisions, triage, or patient-facing advice.

## Planned distillation and alignment work

The next iteration will address generation behavior rather than merely extending
the existing SFT run:

1. **Create new evaluation contracts.** Build fresh development, preference-
   validation, safety, and sealed-test sets before training. Include factuality,
   relevance, repetition, refusal, uncertainty, emergency escalation, and concise
   answer rubrics.
2. **Curate verified teacher responses.** Generate and filter approximately
   10,000–20,000 concise medical-education examples from a stronger teacher. Retain
   provenance, reject unsupported claims, and use expert or evidence-backed review
   for safety-critical material.
3. **Run sequence-level distillation.** Compare the Stage B v2 parent and Stage C
   balanced checkpoint as initialization points. Train first with LoRA to limit
   destructive weight movement and preserve a cheap, reversible baseline; keep a
   matched full-parameter arm only if resources permit.
4. **Use behavior-aware validation.** Select checkpoints with a composite gate:
   teacher-response loss plus factuality, instruction adherence, repetition, and
   medical/general retention. Token loss alone cannot select the winner.
5. **Apply conservative preference alignment.** Only after distilled generations
   are coherent, construct roughly 2,000–5,000 verified chosen/rejected pairs and
   compare DPO against the distilled checkpoint. Preference optimization is not a
   substitute for missing knowledge and will not be used to hide a weak SFT base.
6. **Perform one new sealed evaluation.** Compare Stage B v2, Stage C balanced,
   Stage C specialist, distilled LoRA, and any DPO model on the same untouched test
   contract. Preserve both successful and negative results.

The intended endpoint is a narrow, research-oriented medical education model with
measured boundaries—not a general medical expert or deployable assistant.

## License

See [`LICENSE`](LICENSE). Dataset licenses and inclusion decisions are tracked
separately by the data-policy pipeline; the repository license does not override the
license of any upstream dataset.
