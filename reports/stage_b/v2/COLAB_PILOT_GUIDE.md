# Stage B v2 Colab Pilot Guide

## Purpose

Stage B v2 compares three adaptation methods before any new full continual
pretraining run is authorized. All three pilots start independently from the
promoted Stage A checkpoint and consume the same deterministic v2 batches.
Stage B v1 is retained as comparison evidence and is never used as a v2 parent.

Use [the Stage B v2 notebook](../../../notebooks/colab_stage_b_v2.ipynb).

## Required Drive inputs

Upload the locally generated archive to:

```text
MyDrive/medical-slm/stage-b-v2-data.tar
```

Keep the promoted Stage A checkpoint at:

```text
MyDrive/medical-slm-runs/stage_a/checkpoints/checkpoint_00007250/
```

The verified local archive has:

| Property | Value |
|---|---:|
| Size | 535,950,336 bytes |
| Tar entries | 157 |
| V2 training shards | 126 |
| SHA-256 | `9aacec980f8e600b1c40a55edfb4c942e310ca84ce79b61b8036ceed2e522a7f` |

It contains only the v2 packed training set, medical evaluation dataset,
general evaluation dataset, and tokenizer. The notebook streams the archive
hash calculation so it does not load the complete archive into RAM.

Rebuild it locally if necessary:

```powershell
tar -cf stage-b-v2-data.tar `
  datasets/tokenized/continual_medical_stage_b_v2 `
  datasets/tokenized/evaluation_medical `
  datasets/tokenized/evaluation `
  artifacts/tokenizer

Get-FileHash stage-b-v2-data.tar -Algorithm SHA256
```

Do not use a rebuilt archive unless its contents and new hash are reviewed and
the expected hash in the notebook is deliberately updated.

## Colab execution order

1. Commit and push the v2 implementation and notebook to GitHub.
2. Upload `stage-b-v2-data.tar` to the exact Drive path above.
3. Open `notebooks/colab_stage_b_v2.ipynb` in Colab Pro.
4. Select a GPU runtime.
5. Run bootstrap, focused tests, parent initialization, and the zero-update
   baseline.
6. Run the isolated ten-update one-batch diagnostic.
7. Run all three 500-update pilots:
   - full-parameter control;
   - selective freezing;
   - selective freezing plus L2-SP.
8. If a runtime disconnects, use the pilot resume cell for that arm.
9. Run the comparison cell. It selects validation checkpoints only and writes
   `pilot_selection.json` to Drive.
10. Inspect the comparison report before running the standalone full-run cell.
11. After the full run, execute the validation-only final selection cell. It
    compares `best_preferred` with the unvalidated 8,033-update endpoint.
12. Review `stage_b_v2_candidate_validation.json`, then run the one-time sealed
    medical/general test cell exactly once.

The notebook resolves precision automatically: T4 uses FP16 with a gradient
scaler, while a BF16-capable GPU uses BF16 without a scaler.

## Drive outputs

```text
MyDrive/medical-slm-runs/stage_b_v2/
├── stage_b_v2_baseline.json
├── pilot_selection.json
├── stage_b_v2_candidate_validation.json
├── stage_b_v2_evaluation.json
├── promoted_stage_b_v2.json
├── stage_b_v2_test_evaluation_status.json
├── pilots/
│   ├── control/
│   ├── selective/
│   └── selective_l2sp/
└── full/
```

Each pilot directory has its own checkpoints and `metrics.jsonl`. Metrics are
atomically mirrored whenever a checkpoint is written, so an interrupted Colab
process retains the learning curve through its last durable checkpoint.

## Selection rules

- Medical validation loss must improve over the Stage A medical baseline.
- Preferred candidates remain within 20% general perplexity degradation.
- The hard promotion cap is 25% general perplexity degradation.
- Two consecutive validations above 35% degradation stop the run.
- The comparison chooses the lowest medical loss inside the preferred band.
- It falls back to the hard-cap band only when no preferred checkpoint exists.
- Medical and general test splits remain sealed throughout pilot selection.

The L2-SP value `1,000,000` is a calibration candidate for a mean-squared
parameter-displacement penalty. Compare its reported regularization and total
losses before treating it as the final coefficient.

## Full-run rule

The selected full run starts again from Stage A and trains for at most 8,033
updates. It does not continue the winning 500-update pilot. This preserves a
clean, reproducible full-run schedule and prevents pilot-specific optimizer
history from silently influencing the result.

The final endpoint is not promoted automatically. The notebook independently
re-evaluates both it and `best_preferred` on medical/general validation, selects
without test data, and records that decision before the test datasets are
opened. A durable status file prevents accidental repeated test evaluation.

## Physical checkpoint preservation

After promotion, run the notebook's physical-preservation cell. It exports the
unique checkpoints referenced by `best_preferred`, `final_stage_b_v2`, and all
available supporting pointers. For this run that means the promoted update
8,000 checkpoint and chronological update 8,033 endpoint. The bundle also
contains full metrics, reports, configs, tokenizer, dataset manifests, lineage,
and a per-file SHA-256 inventory.

The verified archive and checksum are copied atomically to:

```text
MyDrive/medical-slm-runs/stage_b_v2/preservation/
├── stage_b_v2_preservation.tar
└── stage_b_v2_preservation.tar.sha256
```

Download both files into the repository root. Verify the archive hash in
PowerShell:

```powershell
$expected = (Get-Content stage_b_v2_preservation.tar.sha256).Split()[0]
$actual = (Get-FileHash stage_b_v2_preservation.tar -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected.ToLower()) { throw "Archive SHA-256 mismatch" }
```

Then extract into the Git-ignored training-artifact directory and verify every
file and checkpoint:

```powershell
tar -xf stage_b_v2_preservation.tar -C artifacts/training
python scripts/artifacts/verify_preserved_run.py `
  --root artifacts/training/stage_b_v2
```

Do not delete the Drive checkpoint directories until both the archive hash and
the extracted preservation manifest pass locally.
