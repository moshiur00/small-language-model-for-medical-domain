# Stage B v1 Full Continual-Pretraining Experiment

**Experiment status:** Training complete; preserved comparison baseline  
**Medical test status:** Not evaluated  
**Parent:** Stage A `checkpoint_00007250`

## Purpose

Stage B v1 tested full-parameter continual pretraining on the disjoint medical/rehearsal corpus. It is retained as the reference experiment for comparison with Stage B v2 and future LoRA-based medical adaptation. It must not be overwritten by later runs.

## Completed training

| Property | Result |
|---|---:|
| Epochs | 1 |
| Optimizer updates | 6,840 |
| Consumed sequences | 875,470 |
| Consumed tokens | 224,120,320 |
| Skipped updates | 0 |
| Non-finite events | 0 |
| Peak learning rate | `1e-4` |
| Data mixture | 82.22% medical / 17.78% general |
| Validation interval | 250 updates |

## Validation outcome

Untouched Stage A baselines:

| Distribution | Loss | Perplexity |
|---|---:|---:|
| Medical validation | 3.467505 | 32.057 |
| General validation | 3.198383 | 24.493 |

The original promotion rule allowed at most 5% relative general-loss degradation. This corresponds to a general-loss ceiling of 3.358302 and approximately 17% perplexity degradation.

### Retention-selected checkpoint

`checkpoint_00000250` is the preserved `best_eligible` checkpoint.

| Distribution | Loss | Perplexity | Change from baseline |
|---|---:|---:|---:|
| Medical validation | 3.374685 | 29.215 | 2.68% lower loss |
| General validation | 3.330027 | 27.939 | 4.12% higher loss |

Independent full-validation evaluation reproduced both values and passed the original selection rule. It had processed 8,192,000 Stage B tokens, approximately 3.66% of the full corpus.

### Full-run endpoint

The completed run state at update 6,840 recorded:

- Best medical-validation loss: 3.009167.
- Latest recorded general-validation loss: 3.749123.
- Best eligible medical loss: 3.374685.
- Zero skipped or non-finite updates.

The endpoint demonstrates substantial medical adaptation accompanied by catastrophic general-domain forgetting. It is an experimental comparison endpoint, not the promoted model.

## Preserved checkpoints

The preservation bundle contains the full immutable checkpoints referenced by:

- `best_eligible.json`: retention-constrained v1 candidate.
- `best_medical.json`: strongest medical-validation checkpoint regardless of retention.
- `final_stage_b.json`: chronological one-epoch endpoint.

Full checkpoints include model, optimizer, scheduler, FP16 scaler, RNG state, configuration, environment, state, metrics tail, and checksummed checkpoint manifest. They belong under the Git-ignored path:

```text
artifacts/training/stage_b_v1/checkpoints/
```

The raw full-run `metrics.jsonl`, dataset manifests, tokenizer, configurations, machine-readable reports, pointer files, and preservation manifest are retained beside them.

## Interpretation

Stage B v1 is evidence that full-parameter adaptation at `1e-4` with only 17.78% general rehearsal moves the 35M-parameter model rapidly toward the medical distribution but exceeds the desired retention budget. This is not a loss-alignment, checkpoint, data-lineage, or numerical-stability failure. It is the observed optimization trade-off for this recipe.

No medical-test result should be added to v1 until the project has predeclared which experiment and checkpoint will receive the one-time test evaluation.

## Planned comparisons

Stage B v2 will restart from the same Stage A parent with a lower learning rate, more general rehearsal, more frequent validation, and a perplexity-based retention rule. A later LoRA experiment will freeze the Stage A base weights and train adapters on a matched medical/rehearsal curriculum. All strategies must use the same tokenizer, architecture, validation splits, token accounting, and reporting schema.
