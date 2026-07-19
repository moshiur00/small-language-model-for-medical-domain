"""Run the complete post-download data preparation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys


STAGES = [
    ("clean", ["scripts/clean/clean_dataset.py", "all"]),
    ("exact_dedup", ["scripts/deduplicate/deduplicate_dataset.py", "all"]),
    ("global_dedup", ["scripts/deduplicate/deduplicate_global.py"]),
    ("near_dedup", ["scripts/deduplicate/deduplicate_near.py"]),
    ("language", ["scripts/language/verify_language.py"]),
    ("quality", ["scripts/quality/filter_quality.py"]),
    ("license", ["scripts/license/validate_licenses.py"]),
    ("toxicity", ["scripts/toxicity/audit_toxicity.py"]),
    ("assembly", ["scripts/assembly/build_phase_corpora.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=[name for name, _ in STAGES], default="clean")
    parser.add_argument("--stop-after", choices=[name for name, _ in STAGES], default="assembly")
    arguments = parser.parse_args()
    names = [name for name, _ in STAGES]
    start = names.index(arguments.start_at)
    stop = names.index(arguments.stop_after)
    if start > stop:
        parser.error("--start-at must not come after --stop-after")
    for name, command in STAGES[start : stop + 1]:
        print(f"\n=== Running {name} ===", flush=True)
        subprocess.run([sys.executable, *command], check=True)


if __name__ == "__main__":
    main()
