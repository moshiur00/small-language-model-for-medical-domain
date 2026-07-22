"""Create a deterministic, checksum-addressed Stage C Colab data archive."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


DEFAULT_INPUTS = (
    Path("artifacts/tokenizer"),
    Path("datasets/tokenized/sft_stage_c_v1/manifest.json"),
    Path("datasets/tokenized/sft_stage_c_v1/train"),
    Path("datasets/tokenized/sft_stage_c_v1/validation"),
    Path("datasets/tokenized/evaluation/validation"),
    Path("datasets/tokenized/evaluation/dataset_manifest.json"),
    Path("datasets/tokenized/evaluation_medical/validation"),
    Path("datasets/tokenized/evaluation_medical/dataset_manifest.json"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/exports/stage-c-sft-data.tar"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove host-specific metadata so repeated archives have one identity."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    info.pax_headers = {}
    return info


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    yield path
    yield from sorted(path.rglob("*"), key=lambda item: item.as_posix())


def create_archive(
    output: Path,
    *,
    inputs: tuple[Path, ...] | None = None,
) -> tuple[Path, Path]:
    selected_inputs = DEFAULT_INPUTS if inputs is None else inputs
    missing = [path for path in selected_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage C archive inputs: " + ", ".join(map(str, missing))
        )
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise FileExistsError(f"Refusing to replace an existing archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
            for root in selected_inputs:
                for path in iter_files(root):
                    archive.add(
                        path,
                        arcname=path.as_posix(),
                        recursive=False,
                        filter=normalized_tar_info,
                    )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = Path(str(output) + ".sha256")
    checksum.write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    return output, checksum


def main() -> None:
    archive, checksum = create_archive(parse_arguments().output)
    print({
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": checksum.read_text(encoding="utf-8").split()[0],
        "checksum": str(checksum),
    })


if __name__ == "__main__":
    main()
