# Data preparation plan

The project prepares four independent corpora: a 60M-token mixed tokenizer
corpus, a 325M-token Stage A corpus, a 225M-token continual-medical corpus, and
a balanced SFT corpus. Targets live in `configs/corpora.yaml`.

Run every source through this order:

1. download through an official API, bulk endpoint, or pinned Hugging Face revision;
2. standardize to the project JSONL schema;
3. clean text and retain provenance;
4. exact-deduplicate within sources, then globally;
5. near-deduplicate globally while protecting evaluation splits;
6. verify English with fastText;
7. apply source-aware quality filters;
8. validate the record-level license;
9. audit toxicity, routing medical-context false positives to review;
10. assemble by source budget with a deterministic seed.

Do not treat all of PubMed or PMC as one permissively licensed collection.
PubMed distributes citation data under its own terms and abstracts may retain
publisher copyright. PMC Open Access licenses vary per article. Preserve PMID,
PMCID, URL, title, authors, publication date, and the exact license statement.

SFT sources must remain separate through processing. Balance with capped
round-robin sampling so a large dataset such as MedMCQA cannot dominate. Keep
official validation/test splits out of training and globally deduplicate them
before training records.

The `pubmed_abstracts`, `pmc_open_access`, `wikidoc`, `medalpaca`, and
`medinstruct` entries require their configured license gate before they can
enter a redistributable or commercial-use corpus.

## Download and standardize

Downloading writes canonical JSONL directly to `datasets/raw/<dataset>`.
Every dataset has its own standardizer module under
`src/medical_slm/data/standardizers`.

Set the NCBI contact email in the repository-root `.env` before using PubMed or
PMC. An API key is optional but raises the permitted request rate. The
downloader loads this file automatically and never overrides variables already
set by the shell:

```dotenv
NCBI_EMAIL=researcher@example.org
NCBI_API_KEY=optional-key
```

Download one source:

```bash
python scripts/download/download_dataset.py fineweb_edu
python scripts/download/download_dataset.py pubmed_abstracts
python scripts/download/download_dataset.py pmc_open_access
python scripts/download/download_dataset.py openmedinstruct
```

Download every configured source sequentially:

```bash
python scripts/download/download_dataset.py all
```

The downloader checks that every `max_documents` value in `configs/data.yaml`
exactly matches `download_plan` in `configs/corpora.yaml`. The caps provide a
bounded acquisition margin; they are not final corpus sizes. Cleaning and
token-budgeted corpus assembly determine the final retained counts.

The WikiDoc pretraining source is the `source == "wikidoc"` subset of
`epfl-llm/guidelines`. The public OpenMedInstruct source is pinned to
`OrionLLM/OpenMedicalInstruct`. Dataset revisions are commit-pinned so later
upstream changes cannot silently alter the training corpus.

## Run downstream preparation

All downstream stages discover every dataset and split from `datasets:` in
`configs/data.yaml`; priority lists are generated with test and validation
before train so evaluation records win deduplication conflicts.

Run the complete post-download workflow:

```bash
python scripts/prepare_data.py
```

Resume from a named stage after correcting or rerunning one step:

```bash
python scripts/prepare_data.py --start-at quality
python scripts/prepare_data.py --start-at toxicity --stop-after assembly
```

Stages are `clean`, `exact_dedup`, `global_dedup`, `near_dedup`, `language`,
`quality`, `license`, `toxicity`, and `assembly`.

Quality filtering assigns the `pretraining` profile to document corpora and
the more permissive `sft` profile to instruction datasets. License values that
are noncommercial, source-level, jurisdiction-dependent, or underspecified are
retained with a `review` decision and explicit obligations; they are never
silently recorded as fully cleared.

Phase assembly consumes only `datasets/interim/toxicity_audited` and writes:

```text
datasets/processed/corpora/tokenizer_60m/
datasets/processed/corpora/stage_a_325m/
datasets/processed/corpora/continual_medical_225m/
datasets/processed/corpora/sft_balanced/
datasets/processed/corpora/manifest.json
```

Pre-tokenizer token counts are estimates using four characters per token. The
manifest reports sources that run out before their configured allocation. Once
the tokenizer exists, rebuild quotas from exact tokenizer counts.
