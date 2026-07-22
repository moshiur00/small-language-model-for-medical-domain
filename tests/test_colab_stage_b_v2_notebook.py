"""Static safety checks for the generated Stage B v2 Colab workflow."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_stage_b_v2.ipynb")


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_stage_b_v2_notebook_code_cells_compile() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert len(cells) == 28
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")


def test_stage_b_v2_notebook_has_isolated_pilots_and_safe_selection() -> None:
    notebook = load_notebook()
    code_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for arm in ("control", "selective", "selective_l2sp"):
        assert f"run_fresh_pilot('{arm}')" in code_text
    assert "best_preferred.json" in code_text
    assert "selection_uses_test_data': False" in code_text
    assert "evaluation_medical/test" not in code_text
    assert "datasets/tokenized/evaluation/test" not in code_text
    assert (
        "https://github.com/moshiur00/small-language-model-for-medical-domain.git"
        in code_text
    )
    assert "github.com/moshiru00/" not in code_text
    assert "REPOSITORY_BRANCH = 'main'" in code_text
    assert "'--branch', REPOSITORY_BRANCH, '--single-branch'" in code_text


def test_stage_b_v2_full_start_and_resume_cells_are_standalone() -> None:
    cells = load_notebook()["cells"]
    fresh = "".join(cells[23]["source"])
    resume = "".join(cells[25]["source"])
    for source in (fresh, resume):
        assert "drive.mount('/content/drive')" in source
        assert "git', 'clone'" in source
        assert "pip', 'install'" in source
        assert "stage-b-v2-data.tar" in source
        assert "checkpoint_00007250" in source
        assert "Removing stale non-Git runtime directory" in source
        assert (
            "rglob('checkpoint_00007250/checkpoint_manifest.json')" in source
        )
    assert "--resume', 'latest'" not in fresh
    assert "--resume', 'latest'" in resume
    assert "drive_metrics" in resume
