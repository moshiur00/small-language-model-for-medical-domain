"""Generic Hugging Face dataset download utilities."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

from medical_slm.data.jsonl import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)

DatasetExample: TypeAlias = Mapping[str, Any]
StandardizedRecord: TypeAlias = dict[str, Any]

Standardizer: TypeAlias = Callable[
    ...,
    Iterator[StandardizedRecord],
]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    if "datasets" not in config:
        raise ValueError("Configuration must contain a 'datasets' section.")

    return config


def create_document_id(
    source: str,
    split: str,
    index: int,
    text: str,
) -> str:
    """
    Create a deterministic document identifier.

    The identifier contains the source, split, source index and a truncated
    SHA-256 hash of the standardized text.
    """
    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source}-{split}-{index:09d}-{text_hash}"


def resolve_limit(value: Any) -> int | None:
    """Validate and normalize a configured document limit."""
    if value is None:
        return None

    if not isinstance(value, int):
        raise TypeError(
            f"Document limit must be an integer or null, received {value!r}."
        )

    if value <= 0:
        raise ValueError(
            f"Document limit must be positive, received {value}."
        )

    return value


def write_metadata(
    *,
    output_directory: Path,
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    split_counts: Mapping[str, int],
) -> None:
    """Write dataset provenance and ingestion metadata."""
    metadata = {
        "dataset_name": dataset_name,
        "source_type": dataset_config.get("source_type", "huggingface"),
        "hub_name": dataset_config.get("hub_name"),
        "source_locator": dataset_config.get("hub_name", dataset_config.get("base_url")),
        "hub_config_name": dataset_config.get("config_name"),
        "source_name": dataset_config["source_name"],
        "license": dataset_config["license"],
        "language": dataset_config["language"],
        "streaming": bool(dataset_config.get("streaming", True)),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "format": "jsonl",
        "split_document_counts": dict(split_counts),
        "configured_splits": dict(dataset_config["splits"]),
        "schema": {
            "id": "string",
            "source": "string",
            "source_dataset": "string",
            "source_config": "string | null",
            "source_split": "string",
            "license": "string",
            "language": "string",
            "text": "string",
            "metadata": "object",
        },
    }

    metadata_path = output_directory / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    LOGGER.info("Wrote metadata to %s", metadata_path)


def download_dataset(
    *,
    config_path: Path,
    dataset_name: str,
    standardizer: Standardizer,
) -> dict[str, int]:
    """
    Download and standardize a configured Hugging Face dataset.

    Args:
        config_path:
            Path to the YAML data configuration.
        dataset_name:
            Dataset key under the configuration's ``datasets`` section.
        standardizer:
            Dataset-specific generator that converts source examples into
            the project's unified schema.

    Returns:
        Number of standardized documents written for each output split.
    """
    project_root = config_path.resolve().parent.parent
    load_dotenv(project_root / ".env", override=False)
    config = load_config(config_path)

    datasets_config = config["datasets"]

    if dataset_name not in datasets_config:
        available = ", ".join(sorted(datasets_config))
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )

    dataset_config = datasets_config[dataset_name]
    validate_download_limit_plan(
        root_config=config,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

    required_fields = {
        "source_name",
        "license",
        "language",
        "splits",
        "output_directory",
    }

    missing_fields = required_fields - dataset_config.keys()

    source_type = str(dataset_config.get("source_type", "huggingface"))
    if source_type == "huggingface" and "hub_name" not in dataset_config:
        missing_fields.add("hub_name")
    if source_type == "ncbi_eutils" and "database" not in dataset_config:
        missing_fields.add("database")

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Dataset '{dataset_name}' is missing fields: {missing}"
        )

    hub_name = str(
        dataset_config.get(
            "hub_name",
            dataset_config.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"),
        )
    )
    config_name = dataset_config.get("config_name")
    source = str(dataset_config["source_name"])
    license_name = str(dataset_config["license"])
    language = str(dataset_config["language"])
    streaming = bool(dataset_config.get("streaming", True))

    split_mapping = dataset_config["splits"]
    limits = dataset_config.get("max_documents", {})

    if not isinstance(split_mapping, dict) or not split_mapping:
        raise ValueError(
            f"Dataset '{dataset_name}' must define at least one split."
        )

    output_directory = Path(dataset_config["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}

    for output_split, source_split in split_mapping.items():
        max_documents = resolve_limit(limits.get(output_split))

        LOGGER.info(
            "Loading dataset=%s config=%s source_split=%s "
            "output_split=%s streaming=%s limit=%s",
            hub_name,
            config_name,
            source_split,
            output_split,
            streaming,
            max_documents,
        )

        dataset = load_source_dataset(
            dataset_config=dataset_config,
            source_split=str(source_split),
            max_documents=max_documents,
        )

        records = standardizer(
            dataset,
            hub_name=hub_name,
            config_name=config_name,
            source=source,
            source_split=str(source_split),
            output_split=str(output_split),
            license_name=license_name,
            language=language,
            max_documents=max_documents,
        )

        output_path = output_directory / f"{output_split}.jsonl"
        if bool(dataset_config.get("resume", False)):
            written_count = write_resumable_jsonl(
                records,
                output_path,
                max_documents=max_documents,
            )
        else:
            written_count = write_jsonl(records, output_path)

        split_counts[str(output_split)] = written_count

        LOGGER.info(
            "Wrote %d standardized documents to %s",
            written_count,
            output_path,
        )

    write_metadata(
        output_directory=output_directory,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split_counts=split_counts,
    )

    return split_counts


def _resume_key(record: Mapping[str, Any]) -> str:
    """Return a stable source identifier for resumable NCBI downloads."""
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for field in ("pmcid", "pmid", "doi", "source_document_id"):
            value = metadata.get(field)
            if value is not None and str(value).strip():
                return f"{record.get('source')}:{field}:{str(value).strip()}"
    record_id = record.get("id")
    if record_id is None:
        text = str(record.get("text", ""))
        return f"text:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return f"id:{record_id}"


def write_resumable_jsonl(
    records: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    max_documents: int | None,
) -> int:
    """Append unique records to a valid partial JSONL file up to its cap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[str] = set()
    existing_count = 0
    if output_path.exists():
        for record in read_jsonl(output_path):
            existing_keys.add(_resume_key(record))
            existing_count += 1
        LOGGER.info("Resuming %s with %d existing records", output_path, existing_count)
    if max_documents is not None and existing_count >= max_documents:
        return existing_count

    total = existing_count
    pending_flush = 0
    with output_path.open("a", encoding="utf-8") as file:
        for record in records:
            key = _resume_key(record)
            if key in existing_keys:
                continue
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
            existing_keys.add(key)
            total += 1
            pending_flush += 1
            if pending_flush >= 100:
                file.flush()
                pending_flush = 0
            if max_documents is not None and total >= max_documents:
                break
    return total


def validate_download_limit_plan(
    *,
    root_config: Mapping[str, Any],
    dataset_name: str,
    dataset_config: Mapping[str, Any],
) -> None:
    """Ensure data.yaml limits remain synchronized with corpora.yaml."""
    project = root_config.get("project")
    if not isinstance(project, Mapping) or not project.get("corpora_config"):
        return
    corpora_path = Path(str(project["corpora_config"]))
    if not corpora_path.exists():
        raise FileNotFoundError(f"Corpus configuration does not exist: {corpora_path}")
    with corpora_path.open("r", encoding="utf-8") as file:
        corpora_config = yaml.safe_load(file)
    if not isinstance(corpora_config, Mapping):
        raise ValueError("Corpus configuration root must be a mapping.")
    plan = corpora_config.get("download_plan")
    if not isinstance(plan, Mapping) or dataset_name not in plan:
        raise ValueError(f"download_plan is missing dataset '{dataset_name}'.")
    planned_dataset = plan[dataset_name]
    if not isinstance(planned_dataset, Mapping):
        raise TypeError(f"download_plan.{dataset_name} must be a mapping.")
    planned_limits = planned_dataset.get("max_documents")
    configured_limits = dataset_config.get("max_documents")
    if planned_limits != configured_limits:
        raise ValueError(
            f"max_documents for '{dataset_name}' differs between data.yaml "
            "and corpora.yaml download_plan."
        )


def load_source_dataset(
    *,
    dataset_config: Mapping[str, Any],
    source_split: str,
    max_documents: int | None,
) -> Iterable[Mapping[str, Any]]:
    """Load one configured Hugging Face or NCBI source split."""
    source_type = str(dataset_config.get("source_type", "huggingface"))
    if source_type == "huggingface":
        if bool(dataset_config.get("requires_authentication", False)) and not os.environ.get(
            "HF_TOKEN"
        ):
            raise ValueError(
                "This dataset is gated. Set HF_TOKEN after accepting its access terms."
            )
        load_arguments: dict[str, Any] = {
            "path": str(dataset_config["hub_name"]),
            "split": source_split,
            "streaming": bool(dataset_config.get("streaming", True)),
        }
        config_name = dataset_config.get("config_name")
        if config_name is not None:
            load_arguments["name"] = config_name
        revision = dataset_config.get("revision")
        if revision is not None:
            load_arguments["revision"] = str(revision)
        data_files = dataset_config.get("data_files")
        if data_files is not None:
            load_arguments["data_files"] = data_files
        return load_dataset(**load_arguments)
    if source_type == "ncbi_eutils":
        if max_documents is None:
            raise ValueError("NCBI downloads require a finite max_documents limit.")
        return iter_ncbi_xml_records(
            database=str(dataset_config["database"]),
            search_term=str(dataset_config["search_term"]),
            max_documents=max_documents,
            batch_size=int(dataset_config.get("batch_size", 100)),
            base_url=str(
                dataset_config.get(
                    "base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
                )
            ),
            email_environment_variable=str(
                dataset_config.get("email_environment_variable", "NCBI_EMAIL")
            ),
            api_key_environment_variable=str(
                dataset_config.get("api_key_environment_variable", "NCBI_API_KEY")
            ),
            date_start=(
                str(dataset_config["date_start"])
                if dataset_config.get("date_start") is not None
                else None
            ),
            date_end=(
                str(dataset_config["date_end"])
                if dataset_config.get("date_end") is not None
                else None
            ),
            date_field=str(dataset_config.get("date_field", "PDAT")),
        )
    raise ValueError(f"Unsupported source_type: {source_type}")


def _open_ncbi_request(
    url: str,
    params: Mapping[str, Any],
    *,
    timeout: int,
    max_attempts: int = 5,
) -> bytes:
    """POST an NCBI request with bounded retry and useful error messages."""
    payload = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "medical-slm-data-pipeline/0.1",
        },
        method="POST",
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")[:1000]
            retryable = error.code in {400, 408, 429, 500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise RuntimeError(
                    f"NCBI request failed after {attempt} attempt(s): "
                    f"HTTP {error.code} {error.reason}; response={response_text!r}"
                ) from error
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            LOGGER.warning(
                "NCBI HTTP %d; retrying request in %.1f seconds (attempt %d/%d)",
                error.code,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)
        except URLError as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"NCBI network request failed after {attempt} attempts: {error.reason}"
                ) from error
            delay = 2 ** attempt
            LOGGER.warning(
                "NCBI network error; retrying in %.1f seconds (attempt %d/%d): %s",
                delay,
                attempt,
                max_attempts,
                error.reason,
            )
            time.sleep(delay)
        except (http.client.HTTPException, TimeoutError, ConnectionError, OSError) as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    "NCBI response was interrupted after "
                    f"{attempt} attempts: {type(error).__name__}: {error}"
                ) from error
            delay = 2 ** attempt
            LOGGER.warning(
                "NCBI response interrupted (%s); retrying in %.1f seconds "
                "(attempt %d/%d)",
                type(error).__name__,
                delay,
                attempt,
                max_attempts,
            )
            time.sleep(delay)
    raise RuntimeError("NCBI request retry loop ended unexpectedly.")


def _request_json(url: str, params: Mapping[str, Any]) -> dict[str, Any]:
    payload = _open_ncbi_request(url, params, timeout=120)
    with io.BytesIO(payload) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("NCBI returned a non-object JSON response.")
    return value


def iter_ncbi_xml_records(
    *,
    database: str,
    search_term: str,
    max_documents: int,
    batch_size: int,
    base_url: str,
    email_environment_variable: str,
    api_key_environment_variable: str,
    date_start: str | None = None,
    date_end: str | None = None,
    date_field: str = "PDAT",
) -> Iterator[Mapping[str, Any]]:
    """Retrieve bounded NCBI records using ESearch history and batched EFetch."""
    if batch_size <= 0 or batch_size > 500:
        raise ValueError("NCBI batch_size must be between 1 and 500.")
    email = os.environ.get(email_environment_variable)
    if not email:
        raise ValueError(
            f"Set {email_environment_variable} to a contact email required for NCBI requests."
        )
    api_key = os.environ.get(api_key_environment_variable)
    common: dict[str, Any] = {"tool": "medical_slm", "email": email}
    if api_key:
        common["api_key"] = api_key

    if max_documents > 9_999:
        if date_start is None or date_end is None:
            raise ValueError(
                "NCBI downloads above 9,999 records require date_start and date_end."
            )
        start_date = _parse_ncbi_date(date_start, "date_start")
        end_date = _parse_ncbi_date(date_end, "date_end")
        if start_date > end_date:
            raise ValueError("date_start must not be later than date_end.")
        normalized_date_field = date_field.upper()
        if normalized_date_field not in {"CRDT", "EDAT", "PDAT"}:
            raise ValueError("date_field must be one of CRDT, EDAT, or PDAT.")

        remaining = max_documents
        current_date = end_date
        while current_date >= start_date and remaining > 0:
            formatted_date = current_date.strftime("%Y/%m/%d")
            partition_term = (
                f'{search_term} AND ("{formatted_date}"[{normalized_date_field}] : '
                f'"{formatted_date}"[{normalized_date_field}])'
            )
            partition_count = 0
            for record in _iter_ncbi_history_records(
                database=database,
                search_term=partition_term,
                max_documents=remaining,
                batch_size=batch_size,
                base_url=base_url,
                common=common,
                api_key_present=bool(api_key),
                truncate_pubmed_overflow=True,
            ):
                partition_count += 1
                remaining -= 1
                yield record
            LOGGER.info(
                "Completed %s date partition %s: records=%d remaining=%d",
                database,
                formatted_date,
                partition_count,
                remaining,
            )
            current_date -= timedelta(days=1)
        if remaining:
            LOGGER.warning(
                "%s date range exhausted before max_documents: requested=%d written=%d",
                database,
                max_documents,
                max_documents - remaining,
            )
        return

    yield from _iter_ncbi_history_records(
        database=database,
        search_term=search_term,
        max_documents=max_documents,
        batch_size=batch_size,
        base_url=base_url,
        common=common,
        api_key_present=bool(api_key),
    )


def _parse_ncbi_date(value: str, field_name: str) -> date:
    """Parse a deterministic NCBI publication-date boundary."""
    try:
        return datetime.strptime(value, "%Y/%m/%d").date()
    except ValueError as error:
        raise ValueError(f"{field_name} must use YYYY/MM/DD format, received {value!r}.") from error


def _iter_ncbi_history_records(
    *,
    database: str,
    search_term: str,
    max_documents: int,
    batch_size: int,
    base_url: str,
    common: Mapping[str, Any],
    api_key_present: bool,
    truncate_pubmed_overflow: bool = False,
) -> Iterator[Mapping[str, Any]]:
    """Retrieve records from one ESearch history query below its result ceiling."""
    search = _request_json(
        f"{base_url.rstrip('/')}/esearch.fcgi",
        {
            **common,
            "db": database,
            "term": search_term,
            "retmax": 0,
            "retmode": "json",
            "usehistory": "y",
        },
    )["esearchresult"]
    available = int(search["count"])
    target = min(max_documents, available)
    if database == "pubmed" and target > 9_999:
        if not truncate_pubmed_overflow:
            raise RuntimeError(
                "A PubMed query partition contains more than 9,999 records; use a narrower "
                f"publication-date range. Query={search_term!r} count={available}"
            )
        LOGGER.warning(
            "PubMed partition has %d records; retaining the retrievable first 9,999",
            available,
        )
        target = 9_999
    query_key = str(search["querykey"])
    web_environment = str(search["webenv"])
    delay = 0.11 if api_key_present else 0.34

    for start in range(0, target, batch_size):
        count = min(batch_size, target - start)
        parameters = {
            **common,
            "db": database,
            "query_key": query_key,
            "WebEnv": web_environment,
            "retstart": start,
            "retmax": count,
            "retmode": "xml",
        }
        payload = _open_ncbi_request(
            f"{base_url.rstrip('/')}/efetch.fcgi",
            parameters,
            timeout=180,
        )
        root = ET.fromstring(payload)
        tag = "PubmedArticle" if database == "pubmed" else "article"
        for article in root.findall(f".//{tag}"):
            yield {"xml": ET.tostring(article, encoding="unicode")}
        time.sleep(delay)
