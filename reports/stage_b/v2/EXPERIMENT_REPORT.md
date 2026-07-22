# Stage B v2 Retention-Aware Continual Pretraining Report

**Experiment status:** Complete and validation-selected  
**Selected checkpoint:** `checkpoint_00008000`  
**Physical checkpoint status:** Verified in Google Drive; local preservation pending  
**Medical/general test status:** Evaluated once after validation-only selection  
**Parent:** Stage A `checkpoint_00007250`

## Outcome

Stage B v2 achieved the intended balance: substantially stronger medical
language modeling than the Stage A parent and Stage B v1's retention-selected
checkpoint, while remaining inside the predeclared preferred general-retention
band. The selected checkpoint is a continued-pretraining base model, not a
clinically safe assistant.

## Training recipe

| Property | Value |
|---|---:|
| Architecture | 35,463,680-parameter decoder |
| Data mixture | 70.0134% medical / 29.9866% general |
| Packed training sequences | 1,028,134 |
| Full-run updates | 8,033 |
| Selected update | 8,000 |
| Tokens consumed at selected update | 262,144,000 |
| Peak learning rate | `4e-5` |
| Final learning rate target | `4e-6` |
| Global tokens per full update | 32,768 |
| Trainable parameters | All parameters |
| L2-SP | Disabled in selected arm |
| Validation/checkpoint interval | 100 updates |
| Skipped updates | 0 |
| Non-finite events | 0 |

The full-parameter control arm was selected after matched 500-update pilots
against selective freezing and selective freezing plus L2-SP. All pilots began
from the same Stage A checkpoint and used identical deterministic data order.

## Pilot selection

| Pilot | Medical validation loss | General validation loss | General perplexity degradation |
|---|---:|---:|---:|
| Full-parameter control | **3.371878** | 3.242126 | 4.47% |
| Selective freezing | 3.406951 | 3.207393 | 0.91% |
| Selective freezing + L2-SP | 3.451979 | **3.199858** | **0.15%** |

All three were inside the preferred 20% retention band. The predetermined rule
therefore selected the full-parameter control because it had the lowest medical
validation loss. The L2-SP coefficient was overly restrictive when combined
with freezing.

## Final validation-only selection

The full run ended at update 8,033, but its last scheduled validation was at
update 8,000. Both models were independently re-evaluated before opening test
data:

| Checkpoint | Medical loss | Medical PPL | General loss | General PPL | General PPL degradation |
|---|---:|---:|---:|---:|---:|
| `checkpoint_00008000` | **3.135652** | **23.004** | 3.348985 | 28.474 | 16.253% |
| `checkpoint_00008033` | 3.135933 | 23.010 | **3.348532** | **28.461** | **16.201%** |

Both improved the medical baseline and remained inside the preferred band. The
predeclared selection rule chose update 8,000 for its slightly lower medical
loss. Test data did not influence this decision.

Checkpoint identity:

```text
checkpoint manifest: ea60b0d6b66ea3bd1987f9ff7bbdd75ba34bc0f05b7c21349ad6ef90615a9b71
model:               799fe9c34648044d21bf73258cd55a46716167f89dcf64e2c0487e0382d65c14
tokenizer:           6c569241e2d166cfba709d8d260cdcbdd6b0907ce45dfa644e0426f1aecb078e
```

## One-time sealed test evaluation

| Distribution | Loss | Perplexity | Samples | Tokens |
|---|---:|---:|---:|---:|
| Medical test | **3.162896** | **23.639** | 3,926 | 1,005,056 |
| General test | 3.691913 | 40.122 | 1,185 | 303,360 |

The Stage A general-test result was loss 3.679168 and perplexity 39.613. Stage B
v2 therefore increased general-test perplexity by approximately 1.28%. This is
smaller than the 16.25% validation-perplexity degradation; both results must be
reported rather than using one to erase the other.

Stage B v1's retention-selected checkpoint had medical-test loss 3.395429 and
perplexity 29.827. V2 reduced medical-test perplexity by approximately 20.75%
relative to that checkpoint.

## Promotion decision

`checkpoint_00008000` is the promoted Stage B v2 base checkpoint because:

1. Selection was made exclusively from validation results.
2. Medical validation improved from 3.467505 to 3.135652.
3. General validation remained inside the preferred 20% perplexity budget.
4. Training completed without skipped or non-finite updates.
5. Checkpoint, model, and tokenizer identities were recorded and reproduced.
6. The test evaluation was performed only after the selection was fixed.

The promotion is for subsequent model-development stages. It does not imply
medical factuality, clinical safety, instruction-following ability, or fitness
for patient care.

## Preserved reports

- `stage_b_v2_evaluation.json`: complete validation selection and test results.
- `promoted_stage_b_v2.json`: promotion pointer and provenance.
- `stage_b_v2_test_evaluation_status.json`: durable one-time-test completion record.
- `stage_b_v2_candidate_validation.json`: retained in the Drive run directory.

The full binary checkpoint remains under:

```text
MyDrive/medical-slm-runs/stage_b_v2/full/checkpoints/checkpoint_00008000/
```

It must be exported and verified locally before the Drive copy can be treated as
fully preserved.

## Next experiments

The next comparison is LoRA-based continual adaptation from the same Stage A
parent. The medical and general test results above are now known and must not be
used to tune LoRA hyperparameters or choose its checkpoint. LoRA selection must
remain validation-only under a predeclared retention rule.
