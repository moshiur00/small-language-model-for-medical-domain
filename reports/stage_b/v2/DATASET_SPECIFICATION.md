# Stage B v2 Dataset Specification

**Status:** Processed corpus and packed binary dataset verified  
**Parent exclusion:** Stage A `stage_a_325m`  
**Target mixture:** 70% medical / 30% general rehearsal

## Design

Stage B v2 retains the complete Stage B v1 medical allocation and adds enough licensed general data to reach a 70/30 token mixture. The source audit found 42,235,053 exact unused general tokens after excluding all Stage A and Stage B v1 documents. Only 39,285,107 additional tokens are required, so no Stage A replay is necessary.

V2 remains document-disjoint from Stage A. It deliberately reuses the Stage B v1 medical and general documents so v1 and v2 can be compared under a closely related medical curriculum. Medical validation, medical test, general validation, and general test remain excluded from training.

## Locked allocation

| Source | Domain | Target tokens |
|---|---|---:|
| PMC Open Access | Medical | 95,000,000 |
| PubMed abstracts | Medical | 65,000,000 |
| WikiDoc | Medical | 25,000,000 |
| FineWeb-Edu | General | 57,000,000 |
| WikiText-103 | General | 15,000,000 |
| Project Gutenberg public domain | General | 7,284,308 |
| **Total** | | **264,284,308** |

The exact build may finish slightly below individual quotas when adding the next complete document would exceed its source budget. Final percentages, sequences, shards, discarded tail tokens, and optimizer-update count must be taken from the generated manifests rather than the planning total.

## Verified build result

| Property | Result |
|---|---:|
| Processed documents | 237,958 |
| Exact stream tokens | 264,230,631 |
| Medical tokens | 184,996,718 (70.0134%) |
| General tokens | 79,233,913 (29.9866%) |
| New documents beyond v1 | 19,146 |
| Packed sequences | 1,028,134 |
| Supervised targets | 263,202,304 |
| Binary shards | 126 |
| Discarded tail tokens | 193 |
| Planned optimizer updates | 8,033 |

All 126 shard byte sizes and SHA-256 hashes match the tokenized dataset manifest. The packed loader returned 256-token inputs, directly shifted 256-token labels, and 256-token attention masks.

Corpus verification passed with zero Stage A overlap, complete retention of all 218,812 v1 documents, and zero overlap with medical validation, medical test, general validation, or general test.

## Unused general inventory

| Source | Unused documents | Exact unused tokens |
|---|---:|---:|
| FineWeb-Edu | 16,477 | 17,892,554 |
| WikiText-103 | 3,891 | 16,946,738 |
| Project Gutenberg | 111 | 7,395,761 |
| Wikipedia | 0 | 0 |
| TinyStories | 0 | 0 |
| **Total** | **20,479** | **42,235,053** |

See [source_audit.json](source_audit.json) for document-ID hashes, exact tokenizer counts, source availability, and the combined Stage A/v1 exclusion hash.

## Build contract

The canonical phase name is:

```text
continual_medical_stage_b_v2_70_30
```

Build from the complete licensed source stage with the existing tokenizer:

```powershell
python scripts/assembly/build_phase_corpora.py `
  --config configs/corpora.yaml `
  --input-directory datasets/interim/license_validated `
  --output-directory datasets/processed/corpora `
  --tokenizer-path artifacts/tokenizer `
  --phases continual_medical_stage_b_v2_70_30
```

The build must satisfy:

1. Zero document overlap with Stage A.
2. Zero overlap with medical validation and medical test.
3. Zero overlap with general validation and general test.
4. No duplicate IDs within v2.
5. Exact tokenizer identity preserved.
6. Every generated binary shard passes its SHA-256 manifest.
7. The final medical/general token mixture is reported from exact counts.

## Comparison implications

The v2 corpus is larger than v1 because medical exposure is preserved rather than reduced. Comparisons must therefore include both matched-token pilot results and full-run results. The selective-freezing pilot, full-parameter control, and L2-SP pilot must consume the same v2 batches in the same deterministic order.
