# Stage B v1 Preservation and Comparison Workflow

Stage B v1 is an immutable pilot/reference experiment. Preserve it before creating Stage B v2 data, changing retention rules, or implementing LoRA adapters.

## Repository storage policy

Small audit material is tracked by Git under:

```text
reports/stage_b/v1/
reports/comparisons/continual_adaptation_registry.json
```

Large and generated artifacts remain Git-ignored under:

```text
artifacts/training/stage_b_v1/
```

Do not commit `.pt` checkpoint files or the preservation archive to GitHub.

## Step 1: rescue the raw metrics immediately

The checkpoint mirror does not contain the full `metrics.jsonl`. While the original full-run Colab runtime is still available, run:

```python
from pathlib import Path
import shutil

source = Path("/content/stage_b/metrics.jsonl")
destination = Path(
    "/content/drive/MyDrive/medical-slm-runs/"
    "stage_b/v1_source/metrics.jsonl"
)

assert source.is_file() and source.stat().st_size > 0
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
assert source.read_bytes() == destination.read_bytes()

print("Stage B v1 metrics preserved:", destination)
print("Bytes:", destination.stat().st_size)
```

This is the only time-sensitive step. The protected checkpoint directories already reside in Drive.

## Step 2: create the preservation bundle in Colab

After committing and pushing the preservation utility, update the Colab clone and run from the repository root:

```python
import os
import shutil
import subprocess
from pathlib import Path

repository = Path("/content/medical-slm")
os.chdir(repository)

subprocess.run(
    ["git", "pull", "--ff-only"],
    check=True,
)

bundle = Path("/content/stage_b_v1")
archive = Path("/content/stage_b_v1.tar")

assert not bundle.exists()
assert not archive.exists()

subprocess.run(
    [
        "python",
        "scripts/artifacts/export_stage_b_v1.py",
        "--checkpoint-root",
        "/content/drive/MyDrive/medical-slm-runs/stage_b/checkpoints",
        "--run-output",
        "/content/drive/MyDrive/medical-slm-runs/stage_b/v1_source",
        "--report-root",
        "/content/drive/MyDrive/medical-slm-runs/stage_b",
        "--destination",
        str(bundle),
        "--archive",
        str(archive),
    ],
    check=True,
)

drive_preservation = Path(
    "/content/drive/MyDrive/medical-slm-runs/stage_b/v1_preservation"
)
drive_preservation.mkdir(parents=True, exist_ok=True)

for source in (archive, Path(str(archive) + ".sha256")):
    shutil.copy2(source, drive_preservation / source.name)

print("Preservation archive copied to:", drive_preservation)
```

The exporter resolves and verifies these pointers:

- `best_eligible`
- `best_medical`
- `final_stage_b`

If multiple pointers reference one checkpoint, that checkpoint is copied only once. Every original checkpoint remains complete and resumable; the exporter does not strip optimizer, scheduler, scaler, or RNG artifacts.

The bundle also contains:

- Raw full-run metrics.
- All available machine-readable Stage B reports.
- Pointer files.
- Stage B v1 training configuration.
- Model configuration.
- Tokenizer JSON.
- Train, medical-validation, and general-validation dataset manifests.
- A preservation manifest with SHA-256 and byte size for every file.

## Step 3: import into the local project

Download `stage_b_v1.tar` and `stage_b_v1.tar.sha256` from Drive to the repository root. In PowerShell:

```powershell
Get-FileHash .\stage_b_v1.tar -Algorithm SHA256
Get-Content .\stage_b_v1.tar.sha256
```

The two SHA-256 values must match. Then extract:

```powershell
tar -xf .\stage_b_v1.tar -C .\artifacts\training
```

Verify every preserved file and every checkpoint manifest:

```powershell
python scripts/artifacts/verify_preserved_run.py `
  --root artifacts/training/stage_b_v1
```

Expected local structure:

```text
artifacts/training/stage_b_v1/
├── checkpoints/
│   ├── best_eligible.json
│   ├── best_medical.json
│   ├── final_stage_b.json
│   └── checkpoint_.../
├── contracts/
├── reports/
├── metrics.jsonl
└── preservation_manifest.json
```

Copy the small JSON reports into the tracked report directory:

```powershell
Copy-Item `
  .\artifacts\training\stage_b_v1\reports\*.json `
  .\reports\stage_b\v1\
```

The raw bundle remains ignored by Git. The copied JSON reports can be reviewed and committed.

## Comparison contract

Stage B v1, Stage B v2, and LoRA must be compared using:

- The same Stage A parent checkpoint.
- The same tokenizer and decoder architecture.
- The same medical and general validation manifests.
- The same shifted-label loss and token-weighted evaluation.
- Explicit medical/general token counts.
- The same validation-only checkpoint-selection policy declared before testing.
- Medical test evaluation only after the final experiment and checkpoint are locked.

The comparison table must report at least:

| Field | Meaning |
|---|---|
| Trainable parameters | Full model or LoRA adapter parameter count |
| Training tokens | Exact consumed targets |
| Peak learning rate | Strategy-specific optimization strength |
| Medical validation loss/perplexity | Adaptation benefit |
| General validation loss/perplexity | Retention cost |
| General perplexity ratio | Interpretable forgetting measure |
| Wall time and GPU | Efficiency |
| Checkpoint/model SHA-256 | Exact identity |

## LoRA comparison rationale

The LoRA experiment will start from the immutable Stage A parent and train only low-rank adapters. The frozen base weights remain recoverable by disabling the adapter, but the active adapter can still cause general-domain forgetting. It must therefore pass the same dual-validation and test-sealing rules as full continual pretraining.

LoRA is not automatically superior: it trades capacity and potentially weaker medical adaptation for lower trainable-state size, easier rollback, and often better base-model retention. The experiment registry keeps this comparison explicit.
