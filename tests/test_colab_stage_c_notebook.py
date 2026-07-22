"""Static safety checks for the Stage C Colab workflow."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_stage_c_sft.ipynb")


def cells() -> list[dict[str, object]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return notebook["cells"]


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in cells()
        if cell["cell_type"] == "code"
    ]


def test_stage_c_notebook_code_cells_compile() -> None:
    for index, source in enumerate(code_cells()):
        compile(source, f"stage_c_cell_{index}", "exec")


def test_stage_c_notebook_contains_all_pretraining_gates_and_pilots() -> None:
    source = "\n".join(code_cells())
    assert "STAGE C INITIALIZATION GATE: PASSED" in source
    assert "STAGE C BASELINE GATE: PASSED" in source
    assert "ONE-BATCH ALIGNMENT GATE: PASSED" in source
    assert "run_fresh_pilot('lr_1e5')" in source
    assert "run_fresh_pilot('lr_2e5')" in source
    assert "selection_uses_test_data': False" in source
    assert "stage-c-sft-data.tar.sha256" not in source
    assert "Path(str(DATA_ARCHIVE) + '.sha256')" in source


def test_stage_c_selection_never_reads_sealed_test_data() -> None:
    selection = next(
        source
        for source in code_cells()
        if "selection_rule" in source and "selected_learning_rate" in source
    )
    assert "sft_stage_c_v1/test" not in selection
    assert "evaluation_medical/test" not in selection
    assert "datasets/tokenized/evaluation/test" not in selection

    final_selection = next(
        source
        for source in code_cells()
        if "STAGE C VALIDATION-ONLY SELECTION: VERIFIED" in source
    )
    assert "select_stage_c_checkpoint.py" in final_selection
    assert "sft_stage_c_v1/test" not in final_selection
    assert "evaluation_medical/test" not in final_selection
    assert "datasets/tokenized/evaluation/test" not in final_selection


def test_stage_c_full_start_and_resume_are_standalone() -> None:
    notebook_cells = cells()
    fresh = "".join(notebook_cells[21]["source"])
    resume = "".join(notebook_cells[23]["source"])
    for source in (fresh, resume):
        assert "drive.mount('/content/drive')" in source
        assert "git', 'clone'" in source
        assert "pip', 'install'" in source
        assert "stage-c-sft-data.tar" in source
        assert "checkpoint_00008000" in source
        assert "selected_learning_rate" in source
    assert "--resume', 'latest'" not in fresh
    assert "--resume', 'latest'" in resume
    assert "drive_metrics" in resume


def test_stage_c_source_analysis_is_validation_only() -> None:
    source_analysis = next(
        source
        for source in code_cells()
        if "STAGE C PER-SOURCE VALIDATION ANALYSIS: VERIFIED" in source
    )
    assert "analyze_stage_c_sources.py" in source_analysis
    assert "checkpoint_00000125" in source_analysis
    assert "checkpoint_00000588" in source_analysis
    assert "analysis_uses_test_data" in source_analysis
    assert "sft_stage_c_v1/test" not in source_analysis
    assert "evaluation_medical/test" not in source_analysis


def test_stage_c_profiles_are_locked_before_guarded_test_access() -> None:
    registration = next(
        source
        for source in code_cells()
        if "STAGE C PROFILE REGISTRATION: VERIFIED AND LOCKED" in source
    )
    sealed_test = next(
        source
        for source in code_cells()
        if "STAGE C SEALED TEST EVALUATION: VERIFIED ONCE" in source
    )
    assert "register_stage_c_profiles.py" in registration
    assert "checkpoint_00000125" in registration
    assert "checkpoint_00000588" in registration
    assert "evaluate_stage_c_test.py" in sealed_test
    assert "stage-c-sealed-test.tar" in sealed_test
    assert "stage_c_test_evaluation_status.json" in sealed_test
    assert "assert not TEST_REPORT.exists() and not TEST_SENTINEL.exists()" in sealed_test
    assert sealed_test.index("PROFILE_REGISTRATION.is_file()") < sealed_test.index(
        "tar', '-xf'"
    )
