# Stage B Colab Baseline Guide

**Status:** Ready to run; Stage B optimizer updates have not started

The next gate evaluates the promoted Stage A model on the complete medical and general validation splits before any continual-pretraining update. The resulting losses are the immutable adaptation and retention baselines for the Stage B run.

## Required files

The repository root now contains the local upload artifact `stage-b-data.tar`. It is intentionally ignored by Git.

| Property | Value |
|---|---:|
| Archive size | 457,461,248 bytes (436.27 MiB) |
| Archive entries | 138 |
| Stage B training shards | 107 |
| SHA-256 | `62034CE651C733F6EF0F26D4526D9F4E62D3AB991D52B3A0487A550785185AB3` |

The archive contains only:

- `datasets/tokenized/continual_medical_stage_b`
- `datasets/tokenized/evaluation_medical`
- `datasets/tokenized/evaluation`
- `artifacts/tokenizer`

The promoted parent checkpoint is not duplicated in the archive. The notebook restores it separately from the existing Drive checkpoint directory.

## Drive layout

Upload the archive and retain the promoted checkpoint at these exact paths:

```text
MyDrive/medical-slm/stage-b-data.tar
MyDrive/medical-slm-runs/stage_a/checkpoints/checkpoint_00007250/
```

The verified baseline report will be written to:

```text
MyDrive/medical-slm-runs/stage_b/stage_b_baseline.json
```

## Run procedure

1. Commit and push the Stage B source, configurations, tests, and notebook to GitHub.
2. Upload `stage-b-data.tar` from the repository root to the Drive path above.
3. Open `notebooks/colab_stage_b.ipynb` in Google Colab.
4. Select a GPU runtime.
5. Run the notebook from top to bottom.
6. Continue only if the last cell prints `STAGE B BASELINE GATE: PASSED`.

The notebook is self-contained for a fresh Colab runtime. It mounts Drive, clones or updates the repository, installs development dependencies, extracts the data to the runtime SSD, copies and verifies the parent checkpoint, checks the GPU precision policy, runs focused regressions, verifies model-only initialization, performs both full validation passes, and copies the verified report to Drive.

## Enforced baseline conditions

The CLI refuses to report a passing baseline after training has begun. The report must show:

- `optimizer_updates: 0`
- `consumed_tokens: 0`
- 35,463,680 model parameters
- parent `checkpoint_00007250`
- 997,632 medical validation target tokens
- 466,432 general validation target tokens
- finite loss and perplexity for both distributions

The report also records the GPU precision, parent and dataset lineage hashes, the 5% allowed general-loss degradation fraction, and the resulting general-loss ceiling.

No medical test data is evaluated at this gate. It remains sealed until a Stage B checkpoint has been selected using validation results alone.

## Development and full-training cells

The same notebook now includes the remaining training workflow. Run the sections in this order:

1. Isolated ten-update one-batch alignment diagnostic.
2. Fifty-update development run in `stage_b_dev`.
3. Restore from Drive and continue from update 50 to update 100.
4. Compare medical improvement and general retention.
5. Start the standalone full 6,840-update, one-epoch Stage B run.
6. If Colab disconnects, use the standalone full-resume cell; do not start fresh again.
7. Run the full-completion verification after `final_stage_b.json` appears.

Development checkpoints are stored in `MyDrive/medical-slm-runs/stage_b_dev/checkpoints`. Full-run checkpoints are stored separately in `MyDrive/medical-slm-runs/stage_b/checkpoints`, preventing a development checkpoint from being mistaken for a full-run resume point.

The Colab profile uses explicit FP16 rather than `auto`. This preserves the presence and semantics of the gradient-scaler state if Colab assigns a T4 in one session and an A100 in a later resume session. Both GPUs support this policy.

The fresh-full-run cell refuses to run if `latest.json` already exists locally or in the full Drive checkpoint directory. The resume cell verifies the checkpoint manifest and saved FP16 configuration before restoring optimizer, scheduler, scaler, RNG, counters, and the next batch cursor.
