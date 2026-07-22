"""Generate the self-contained Stage B v2 Colab experiment notebook."""

from __future__ import annotations

from textwrap import dedent
import json
from pathlib import Path


OUTPUT = Path("notebooks/colab_stage_b_v2.ipynb")
ARCHIVE_SHA256 = "9aacec980f8e600b1c40a55edfb4c942e310ca84ce79b61b8036ceed2e522a7f"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


BOOTSTRAP = f"""
from google.colab import drive
from pathlib import Path
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys

drive.mount('/content/drive')

REPOSITORY_URL = 'https://github.com/moshiur00/small-language-model-for-medical-domain.git'
REPOSITORY_BRANCH = 'main'
REPOSITORY = Path('/content/medical-slm')
DATA_ARCHIVE = Path('/content/drive/MyDrive/medical-slm/stage-b-v2-data.tar')
DRIVE_PARENT = Path('/content/drive/MyDrive/medical-slm-runs/stage_a/checkpoints/checkpoint_00007250')
LOCAL_PARENT = Path('/content/stage_a_parent/checkpoint_00007250')
DRIVE_ROOT = Path('/content/drive/MyDrive/medical-slm-runs/stage_b_v2')
EXPECTED_ARCHIVE_SHA256 = '{ARCHIVE_SHA256}'

assert DATA_ARCHIVE.is_file(), f'Missing {{DATA_ARCHIVE}}'
if not (DRIVE_PARENT / 'checkpoint_manifest.json').is_file():
    search_root = Path('/content/drive/MyDrive/medical-slm-runs')
    parent_manifests = sorted(
        search_root.rglob('checkpoint_00007250/checkpoint_manifest.json')
    ) if search_root.is_dir() else []
    if parent_manifests:
        DRIVE_PARENT = parent_manifests[0].parent
        print('Auto-discovered Stage A parent:', DRIVE_PARENT)
    else:
        raise FileNotFoundError(
            'Cannot find checkpoint_00007250/checkpoint_manifest.json under '
            f'{{search_root}}. Restore or upload the promoted Stage A checkpoint.'
        )

if REPOSITORY.exists() and not (REPOSITORY / '.git').is_dir():
    print('Removing stale non-Git runtime directory:', REPOSITORY)
    shutil.rmtree(REPOSITORY)

if not (REPOSITORY / '.git').is_dir():
    subprocess.run([
        'git', 'clone', '--branch', REPOSITORY_BRANCH, '--single-branch',
        REPOSITORY_URL, str(REPOSITORY),
    ], check=True)
else:
    subprocess.run([
        'git', '-C', str(REPOSITORY), 'fetch', 'origin', REPOSITORY_BRANCH,
    ], check=True)
    subprocess.run([
        'git', '-C', str(REPOSITORY), 'checkout', REPOSITORY_BRANCH,
    ], check=True)
    subprocess.run([
        'git', '-C', str(REPOSITORY), 'pull', '--ff-only',
        'origin', REPOSITORY_BRANCH,
    ], check=True)

os.chdir(REPOSITORY)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[dev]'
], check=True)
source_path = str(REPOSITORY / 'src')
if source_path not in sys.path:
    sys.path.insert(0, source_path)
importlib.invalidate_caches()

archive_digest = hashlib.sha256()
with DATA_ARCHIVE.open('rb') as archive_file:
    for chunk in iter(lambda: archive_file.read(8 * 1024 * 1024), b''):
        archive_digest.update(chunk)
archive_hash = archive_digest.hexdigest()
assert archive_hash == EXPECTED_ARCHIVE_SHA256, (archive_hash, EXPECTED_ARCHIVE_SHA256)
train_metadata = REPOSITORY / 'datasets/tokenized/continual_medical_stage_b_v2/train/metadata.json'
if not train_metadata.is_file():
    subprocess.run(['tar', '-xf', str(DATA_ARCHIVE), '-C', str(REPOSITORY)], check=True)

LOCAL_PARENT.parent.mkdir(parents=True, exist_ok=True)
if not (LOCAL_PARENT / 'checkpoint_manifest.json').is_file():
    subprocess.run(['rsync', '-a', f'{{DRIVE_PARENT}}/', f'{{LOCAL_PARENT}}/'], check=True)

from medical_slm.training.checkpoint import verify_checkpoint
verify_checkpoint(LOCAL_PARENT)
metadata = json.loads(train_metadata.read_text())
assert metadata['statistics']['sequences_written'] == 1_028_134
assert len(metadata['packing']['shards']) == 126
print({{
    'archive_sha256': archive_hash,
    'training_sequences': metadata['statistics']['sequences_written'],
    'training_shards': len(metadata['packing']['shards']),
    'parent': LOCAL_PARENT.name,
}})
"""


PILOT_HELPERS = """
import json
from pathlib import Path
import shutil
import subprocess
import yaml
from medical_slm.training.metrics import mirror_metric_log

ARM_CONFIGS = {
    'control': 'configs/training_stage_b_v2_control.yaml',
    'selective': 'configs/training_stage_b_v2_selective.yaml',
    'selective_l2sp': 'configs/training_stage_b_v2_selective_l2sp.yaml',
}

def pilot_paths(arm):
    local_output = Path(f'/content/stage_b_v2_pilots/{arm}')
    drive_output = DRIVE_ROOT / 'pilots' / arm
    return local_output, drive_output

def write_pilot_config(arm):
    assert arm in ARM_CONFIGS
    local_output, drive_output = pilot_paths(arm)
    values = yaml.safe_load(Path(ARM_CONFIGS[arm]).read_text())
    values.update({
        'output_directory': str(local_output),
        'checkpoint_backup_directory': str(drive_output / 'checkpoints'),
        'parent_checkpoint_directory': str(LOCAL_PARENT),
        'precision': 'auto',
        'max_updates': 500,
        'validation_interval': 50,
        'checkpoint_interval': 50,
    })
    path = Path(f'/content/training_stage_b_v2_{arm}_pilot.yaml')
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path

def preserve_pilot_metrics(arm):
    local_output, drive_output = pilot_paths(arm)
    drive_output.mkdir(parents=True, exist_ok=True)
    metrics = local_output / 'metrics.jsonl'
    if metrics.is_file():
        mirror_metric_log(metrics, drive_output / 'metrics.jsonl')

def run_fresh_pilot(arm):
    local_output, drive_output = pilot_paths(arm)
    assert not (drive_output / 'checkpoints/latest.json').exists(), (
        f'{arm} already has a Drive checkpoint; use the resume cell.'
    )
    if local_output.exists():
        shutil.rmtree(local_output)
    config = write_pilot_config(arm)
    subprocess.run([
        'python', 'scripts/training/train_stage_b_v2.py',
        '--config', str(config),
    ], check=True)
    preserve_pilot_metrics(arm)
    pointer = json.loads((drive_output / 'checkpoints/latest.json').read_text())
    state_path = drive_output / 'checkpoints' / pointer['checkpoint'] / 'trainer_state.json'
    state = json.loads(state_path.read_text())
    print(arm, state)
    return state

def resume_pilot(arm):
    local_output, drive_output = pilot_paths(arm)
    drive_checkpoints = drive_output / 'checkpoints'
    assert (drive_checkpoints / 'latest.json').is_file(), f'No checkpoint for {arm}'
    local_checkpoints = local_output / 'checkpoints'
    local_checkpoints.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', f'{drive_checkpoints}/', f'{local_checkpoints}/'], check=True)
    drive_metrics = drive_output / 'metrics.jsonl'
    if drive_metrics.is_file():
        shutil.copy2(drive_metrics, local_output / 'metrics.jsonl')
    config = write_pilot_config(arm)
    subprocess.run([
        'python', 'scripts/training/train_stage_b_v2.py',
        '--config', str(config), '--resume', 'latest',
    ], check=True)
    preserve_pilot_metrics(arm)
"""


def standalone_full(*, resume: bool) -> str:
    action = "resume" if resume else "fresh"
    restore = """
drive_checkpoints = DRIVE_FULL / 'checkpoints'
assert (drive_checkpoints / 'latest.json').is_file(), 'No full-run checkpoint to resume.'
local_checkpoints = LOCAL_FULL / 'checkpoints'
local_checkpoints.mkdir(parents=True, exist_ok=True)
subprocess.run(['rsync', '-a', f'{drive_checkpoints}/', f'{local_checkpoints}/'], check=True)
drive_metrics = DRIVE_FULL / 'metrics.jsonl'
if drive_metrics.is_file():
    shutil.copy2(drive_metrics, LOCAL_FULL / 'metrics.jsonl')
""" if resume else ""
    fresh_guard = """
assert not (DRIVE_FULL / 'checkpoints/latest.json').exists(), (
    'A full run already exists. Use the standalone resume cell.'
)
if LOCAL_FULL.exists():
    shutil.rmtree(LOCAL_FULL)
""" if not resume else ""
    resume_args = ", '--resume', 'latest'" if resume else ""
    return BOOTSTRAP + f"""

import yaml
from medical_slm.training.metrics import mirror_metric_log

selection_path = DRIVE_ROOT / 'pilot_selection.json'
assert selection_path.is_file(), 'Run and compare all pilots before full training.'
selection = json.loads(selection_path.read_text())
selected_arm = selection['selected_arm']
arm_configs = {{
    'control': 'configs/training_stage_b_v2_control.yaml',
    'selective': 'configs/training_stage_b_v2_selective.yaml',
    'selective_l2sp': 'configs/training_stage_b_v2_selective_l2sp.yaml',
}}
assert selected_arm in arm_configs

LOCAL_FULL = Path('/content/stage_b_v2_full')
DRIVE_FULL = DRIVE_ROOT / 'full'
{fresh_guard}
{restore}
values = yaml.safe_load(Path(arm_configs[selected_arm]).read_text())
values.update({{
    'output_directory': str(LOCAL_FULL),
    'checkpoint_backup_directory': str(DRIVE_FULL / 'checkpoints'),
    'parent_checkpoint_directory': str(LOCAL_PARENT),
    'precision': 'auto',
    'max_updates': 8033,
    'validation_interval': 100,
    'checkpoint_interval': 100,
}})
FULL_CONFIG = Path('/content/training_stage_b_v2_full.yaml')
FULL_CONFIG.write_text(yaml.safe_dump(values, sort_keys=False))
print({{'mode': '{action}', 'selected_arm': selected_arm, 'config': values}})
subprocess.run([
    'python', 'scripts/training/train_stage_b_v2.py',
    '--config', str(FULL_CONFIG){resume_args}
], check=True)
DRIVE_FULL.mkdir(parents=True, exist_ok=True)
if (LOCAL_FULL / 'metrics.jsonl').is_file():
    mirror_metric_log(LOCAL_FULL / 'metrics.jsonl', DRIVE_FULL / 'metrics.jsonl')
"""


cells = [
    markdown("""
    # Stage B v2 retention-aware continual pretraining

    This notebook compares full-parameter training, selective freezing, and
    selective freezing plus L2-SP before allowing a full run. It starts every
    experiment from the promoted Stage A checkpoint, never from Stage B v1.
    """),
    markdown(f"""
    ## Required Drive inputs

    Upload `stage-b-v2-data.tar` to
    `MyDrive/medical-slm/stage-b-v2-data.tar`.

    - Archive size: 535,950,336 bytes
    - Entries: 157
    - SHA-256: `{ARCHIVE_SHA256}`

    Keep the Stage A parent at
    `MyDrive/medical-slm-runs/stage_a/checkpoints/checkpoint_00007250`.
    Select a GPU runtime before continuing. No medical or general test split is
    evaluated in this notebook's pilot-selection phase.
    """),
    markdown("## Bootstrap repository, data, tokenizer, and Stage A parent"),
    code(BOOTSTRAP),
    markdown("## Verify GPU, precision, focused regressions, and initialization"),
    code("""
    import torch
    from medical_slm.training.precision import resolve_precision

    assert torch.cuda.is_available(), 'Select a GPU runtime.'
    policy = resolve_precision('auto', 'cuda')
    print({'gpu': torch.cuda.get_device_name(0), 'precision': policy.name})
    subprocess.run([
        'python', '-m', 'pytest',
        'tests/test_training_adaptation.py',
        'tests/test_training_step.py',
        'tests/test_stage_b_trainer.py', '-q',
    ], check=True)
    subprocess.run([
        'python', 'scripts/training/train_stage_b_v2.py',
        '--config', 'configs/training_stage_b_v2_selective_l2sp.yaml',
        '--parent-checkpoint', str(LOCAL_PARENT),
        '--verify-initialization-only',
    ], check=True)
    print('STAGE B V2 INITIALIZATION GATE: PASSED')
    """),
    markdown("## Zero-update medical/general baseline"),
    code("""
    BASELINE_LOCAL = Path('/content/stage_b_v2_baseline.json')
    BASELINE_DRIVE = DRIVE_ROOT / 'stage_b_v2_baseline.json'
    subprocess.run([
        'python', 'scripts/training/train_stage_b_v2.py',
        '--config', 'configs/training_stage_b_v2_selective_l2sp.yaml',
        '--parent-checkpoint', str(LOCAL_PARENT),
        '--baseline-only', '--baseline-output', str(BASELINE_LOCAL),
    ], check=True)
    baseline = json.loads(BASELINE_LOCAL.read_text())
    assert baseline['optimizer_updates'] == 0
    assert baseline['consumed_tokens'] == 0
    assert baseline['medical_validation']['tokens'] == 997_632
    assert baseline['general_validation']['tokens'] == 466_432
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASELINE_LOCAL, BASELINE_DRIVE)
    print('STAGE B V2 BASELINE GATE: PASSED', baseline)
    """),
    markdown("## Ten-update one-batch alignment diagnostic"),
    code("""
    import yaml

    values = yaml.safe_load(Path('configs/training_stage_b_v2_selective_l2sp.yaml').read_text())
    values.update({
        'output_directory': '/content/stage_b_v2_overfit',
        'checkpoint_backup_directory': None,
        'parent_checkpoint_directory': str(LOCAL_PARENT),
        'precision': 'auto',
        'max_updates': 10,
        'log_interval': 1,
    })
    path = Path('/content/training_stage_b_v2_overfit.yaml')
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    subprocess.run([
        'python', 'scripts/training/train_stage_b_v2.py', '--config', str(path),
        '--overfit-one-batch', '--max-updates', '10',
    ], check=True)
    records = [json.loads(line) for line in Path('/content/stage_b_v2_overfit/metrics.jsonl').read_text().splitlines()]
    losses = [record['metrics']['loss'] for record in records if record['event'] == 'overfit_one_batch']
    assert len(losses) == 10 and losses[-1] < losses[0]
    print('ONE-BATCH ALIGNMENT GATE: PASSED', losses[0], '->', losses[-1])
    """),
    markdown("## Pilot helpers"),
    code(PILOT_HELPERS),
    markdown("## Pilot 1: full-parameter control, updates 0-500"),
    code("control_state = run_fresh_pilot('control')\nassert control_state['update'] == 500"),
    markdown("## Pilot 2: selective freezing, updates 0-500"),
    code("selective_state = run_fresh_pilot('selective')\nassert selective_state['update'] == 500"),
    markdown("## Pilot 3: selective freezing + L2-SP, updates 0-500"),
    code("l2sp_state = run_fresh_pilot('selective_l2sp')\nassert l2sp_state['update'] == 500"),
    markdown("""
    ## Resume an interrupted pilot

    On a new runtime, rerun the bootstrap and pilot-helper cells, set the arm
    below, and run this cell. It restores the latest verified Drive checkpoint.
    """),
    code("ARM_TO_RESUME = 'selective_l2sp'\nresume_pilot(ARM_TO_RESUME)"),
    markdown("## Compare pilots and select a profile without using test data"),
    code("""
    rows = []
    for arm in ARM_CONFIGS:
        checkpoint_root = DRIVE_ROOT / 'pilots' / arm / 'checkpoints'
        preferred = checkpoint_root / 'best_preferred.json'
        eligible = checkpoint_root / 'best_eligible.json'
        pointer_path = preferred if preferred.is_file() else eligible
        assert pointer_path.is_file(), f'{arm} produced no retention-eligible checkpoint.'
        pointer = json.loads(pointer_path.read_text())
        checkpoint = checkpoint_root / pointer['checkpoint']
        verify_checkpoint(checkpoint)
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        degradation = __import__('math').exp(
            state['latest_general_validation_loss']
            - state['general_validation_baseline_loss']
        ) - 1.0
        rows.append({
            'arm': arm,
            'pointer': pointer_path.stem,
            'checkpoint': checkpoint.name,
            'update': state['update'],
            'medical_loss': state['best_preferred_medical_loss']
                if pointer_path is preferred else state['best_eligible_medical_loss'],
            'general_loss': state['latest_general_validation_loss'],
            'general_perplexity_degradation_fraction': degradation,
        })

    preferred_rows = [row for row in rows if row['pointer'] == 'best_preferred']
    candidates = preferred_rows or rows
    selected = min(candidates, key=lambda row: row['medical_loss'])
    report = {
        'selection_uses_test_data': False,
        'selection_rule': 'lowest medical loss inside preferred retention band',
        'selected_arm': selected['arm'],
        'selected_checkpoint': selected['checkpoint'],
        'pilots': rows,
    }
    selection_path = DRIVE_ROOT / 'pilot_selection.json'
    selection_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\\n')
    print(json.dumps(report, indent=2))
    """),
    markdown("""
    # Fresh full Stage B v2 training — standalone

    Run only after `pilot_selection.json` has been created. This cell is
    self-contained for a fresh Colab runtime and starts again from Stage A; it
    does not continue the 500-update pilot.
    """),
    code(standalone_full(resume=False)),
    markdown("""
    # Resume interrupted full Stage B v2 training — standalone

    Use this cell after a Colab disconnect. It is self-contained, restores the
    latest verified Drive checkpoint, and continues the original data cursor.
    """),
    code(standalone_full(resume=True)),
    markdown("## Full-run completion and safety verification"),
    code("""
    import json
    from pathlib import Path
    from medical_slm.training.checkpoint import verify_checkpoint

    drive_full = Path('/content/drive/MyDrive/medical-slm-runs/stage_b_v2/full')
    checkpoint_root = drive_full / 'checkpoints'
    latest = json.loads((checkpoint_root / 'latest.json').read_text())
    checkpoint = checkpoint_root / latest['checkpoint']
    verify_checkpoint(checkpoint)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    final_pointer = checkpoint_root / 'final_stage_b_v2.json'
    stopped_for_retention = state['consecutive_emergency_retention_breaches'] >= 2
    assert final_pointer.is_file() or stopped_for_retention
    assert state['non_finite_events'] == 0
    assert state['skipped_updates'] == 0
    print({
        'latest': checkpoint.name,
        'update': state['update'],
        'epoch': state['epoch'],
        'completed': final_pointer.is_file(),
        'stopped_for_retention': stopped_for_retention,
        'best_preferred_medical_loss': state['best_preferred_medical_loss'],
        'best_eligible_medical_loss': state['best_eligible_medical_loss'],
        'latest_general_validation_loss': state['latest_general_validation_loss'],
    })
    """),
    markdown("""
    # Validation-only final checkpoint selection

    This gate independently evaluates the validation-selected checkpoint and
    the 8,033-update endpoint. It does not open either test split.
    """),
    code("""
    import json
    import math
    import os
    from dataclasses import asdict
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    import yaml
    from torch.utils.data import DataLoader

    from medical_slm.data.tokenization.dataset import PackedTokenDataset
    from medical_slm.data.tokenization.manifest import calculate_sha256
    from medical_slm.model import DecoderConfig, DecoderModel
    from medical_slm.training.checkpoint import load_model_weights, verify_checkpoint
    from medical_slm.training.evaluation import evaluate_shifted_packed
    from medical_slm.training.precision import resolve_precision

    DRIVE_V2 = Path('/content/drive/MyDrive/medical-slm-runs/stage_b_v2')
    CHECKPOINT_ROOT = DRIVE_V2 / 'full/checkpoints'
    BASELINE_REPORT = json.loads((DRIVE_V2 / 'stage_b_v2_baseline.json').read_text())
    MEDICAL_BASELINE = BASELINE_REPORT['medical_validation']['loss']
    GENERAL_BASELINE = BASELINE_REPORT['general_validation']['loss']
    PREFERRED_LIMIT = 0.20
    HARD_LIMIT = 0.25

    device = torch.device('cuda')
    assert torch.cuda.is_available()
    precision = resolve_precision('auto', device)
    model_values = yaml.safe_load(Path('configs/model_stage_a.yaml').read_text())
    model_config = DecoderConfig.from_mapping(model_values)
    tokenizer_hash = calculate_sha256(Path('artifacts/tokenizer/tokenizer.json'))

    def packed_loader(path):
        return DataLoader(
            PackedTokenDataset(path), batch_size=32, shuffle=False,
            num_workers=2, pin_memory=True,
        )

    def pointer_checkpoint(pointer_name):
        pointer = json.loads((CHECKPOINT_ROOT / f'{pointer_name}.json').read_text())
        checkpoint = CHECKPOINT_ROOT / pointer['checkpoint']
        verify_checkpoint(checkpoint)
        return checkpoint

    def evaluate_validation(checkpoint):
        model = DecoderModel(model_config).to(device)
        identity = load_model_weights(
            checkpoint_directory=checkpoint,
            model=model,
            expected_model_config=model_config.to_dict(),
            expected_tokenizer_sha256=tokenizer_hash,
            map_location=device,
        )
        medical = evaluate_shifted_packed(
            model=model,
            batches=packed_loader('datasets/tokenized/evaluation_medical/validation'),
            device=device,
            precision=precision,
        )
        general = evaluate_shifted_packed(
            model=model,
            batches=packed_loader('datasets/tokenized/evaluation/validation'),
            device=device,
            precision=precision,
        )
        del model
        torch.cuda.empty_cache()
        degradation = math.exp(general.loss - GENERAL_BASELINE) - 1.0
        return {
            'checkpoint': checkpoint.name,
            'checkpoint_identity': identity,
            'medical_validation': asdict(medical),
            'general_validation': asdict(general),
            'general_perplexity_degradation_fraction': degradation,
            'medical_improved': medical.loss < MEDICAL_BASELINE,
            'preferred_retention': degradation <= PREFERRED_LIMIT,
            'promotion_eligible': degradation <= HARD_LIMIT,
        }

    candidate_paths = {
        pointer_checkpoint('best_preferred'),
        pointer_checkpoint('final_stage_b_v2'),
    }
    validation_candidates = [
        evaluate_validation(checkpoint)
        for checkpoint in sorted(candidate_paths, key=lambda path: path.name)
    ]
    preferred = [
        result for result in validation_candidates
        if result['medical_improved'] and result['preferred_retention']
    ]
    eligible = [
        result for result in validation_candidates
        if result['medical_improved'] and result['promotion_eligible']
    ]
    selectable = preferred or eligible
    assert selectable, 'No validation candidate satisfies the promotion gate.'
    selected = min(
        selectable,
        key=lambda result: result['medical_validation']['loss'],
    )

    selection_report = {
        'stage': 'continual_medical_stage_b_v2',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'selection_uses_test_data': False,
        'selection_rule': (
            'Lowest medical-validation loss inside the preferred general-'
            'perplexity retention band; hard-cap fallback only if needed.'
        ),
        'medical_baseline_loss': MEDICAL_BASELINE,
        'general_baseline_loss': GENERAL_BASELINE,
        'preferred_degradation_fraction': PREFERRED_LIMIT,
        'hard_degradation_fraction': HARD_LIMIT,
        'selected_checkpoint': selected['checkpoint'],
        'selected_from_preferred_band': bool(preferred),
        'candidates': validation_candidates,
    }

    def atomic_json(path, payload):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')
        os.replace(temporary, path)

    validation_report_path = DRIVE_V2 / 'stage_b_v2_candidate_validation.json'
    atomic_json(validation_report_path, selection_report)
    print(json.dumps(selection_report, indent=2))
    print('VALIDATION-ONLY SELECTION: PASSED')
    """),
    markdown("""
    # One-time sealed medical and general test evaluation

    Run only after the validation-only report is reviewed. A durable sentinel
    prevents accidental repeated test evaluation. Test results never influence
    checkpoint selection.
    """),
    code("""
    validation_report_path = DRIVE_V2 / 'stage_b_v2_candidate_validation.json'
    evaluation_report_path = DRIVE_V2 / 'stage_b_v2_evaluation.json'
    promotion_path = DRIVE_V2 / 'promoted_stage_b_v2.json'
    test_sentinel_path = DRIVE_V2 / 'stage_b_v2_test_evaluation_status.json'

    if evaluation_report_path.is_file():
        raise RuntimeError(
            f'Test evaluation already completed: {evaluation_report_path}'
        )
    if test_sentinel_path.is_file():
        raise RuntimeError(
            'A test-evaluation attempt is already recorded. Audit it before rerunning.'
        )

    validation_report = json.loads(validation_report_path.read_text())
    selected_checkpoint = (
        CHECKPOINT_ROOT / validation_report['selected_checkpoint']
    )
    verify_checkpoint(selected_checkpoint)
    atomic_json(test_sentinel_path, {
        'status': 'started',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': selected_checkpoint.name,
    })

    model = DecoderModel(model_config).to(device)
    selected_identity = load_model_weights(
        checkpoint_directory=selected_checkpoint,
        model=model,
        expected_model_config=model_config.to_dict(),
        expected_tokenizer_sha256=tokenizer_hash,
        map_location=device,
    )
    medical_test = evaluate_shifted_packed(
        model=model,
        batches=packed_loader('datasets/tokenized/evaluation_medical/test'),
        device=device,
        precision=precision,
    )
    general_test = evaluate_shifted_packed(
        model=model,
        batches=packed_loader('datasets/tokenized/evaluation/test'),
        device=device,
        precision=precision,
    )
    assert medical_test.samples == 3_926 and medical_test.tokens == 1_005_056
    assert general_test.samples == 1_185 and general_test.tokens == 303_360
    assert math.isfinite(medical_test.loss) and math.isfinite(general_test.loss)

    selected_validation = next(
        candidate for candidate in validation_report['candidates']
        if candidate['checkpoint'] == selected_checkpoint.name
    )
    evaluation_report = {
        'stage': 'continual_medical_stage_b_v2',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'selected_checkpoint': selected_checkpoint.name,
        'checkpoint_identity': selected_identity,
        'selection_uses_test_data': False,
        'validation_selection': validation_report,
        'selected_candidate_validation': selected_validation,
        'medical_test': asdict(medical_test),
        'general_test': asdict(general_test),
        'test_evaluated_once': True,
        'limitations': [
            'Loss and perplexity do not establish medical factuality.',
            'The model has not been evaluated for clinical safety.',
            'The checkpoint is a base model, not a medical assistant.',
        ],
    }
    promotion = {
        'stage': 'continual_medical_stage_b_v2',
        'checkpoint': selected_checkpoint.name,
        'checkpoint_root': str(CHECKPOINT_ROOT),
        'evaluation_report': evaluation_report_path.name,
        'validation_selected': True,
        'test_used_for_selection': False,
    }
    atomic_json(evaluation_report_path, evaluation_report)
    atomic_json(promotion_path, promotion)
    atomic_json(test_sentinel_path, {
        'status': 'completed',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': selected_checkpoint.name,
        'evaluation_report': evaluation_report_path.name,
    })
    print({
        'selected_checkpoint': selected_checkpoint.name,
        'medical_test_loss': medical_test.loss,
        'medical_test_perplexity': medical_test.perplexity,
        'general_test_loss': general_test.loss,
        'general_test_perplexity': general_test.perplexity,
    })
    print('STAGE B V2 TEST AND PROMOTION ARTIFACTS: WRITTEN')
    """),
    markdown("""
    # Preserve promoted and final Stage B v2 checkpoints

    Run after promotion artifacts exist. This creates a checksummed bundle with
    the selected checkpoint, final endpoint, metrics, reports, pointer files,
    configurations, tokenizer, and dataset manifests, then copies it to Drive.
    """),
    code("""
    from google.colab import drive
    from pathlib import Path
    import hashlib
    import os
    import shutil
    import subprocess

    drive.mount('/content/drive')
    repository = Path('/content/medical-slm')
    assert (repository / '.git').is_dir()
    os.chdir(repository)
    subprocess.run(['git', 'fetch', 'origin', 'main'], check=True)
    subprocess.run(['git', 'checkout', 'main'], check=True)
    subprocess.run(['git', 'pull', '--ff-only', 'origin', 'main'], check=True)

    drive_v2 = Path('/content/drive/MyDrive/medical-slm-runs/stage_b_v2')
    checkpoint_root = drive_v2 / 'full/checkpoints'
    run_output = drive_v2 / 'full'
    bundle = Path('/content/stage_b_v2')
    archive = Path('/content/stage_b_v2_preservation.tar')
    checksum = Path(str(archive) + '.sha256')
    assert not bundle.exists(), f'Refusing overwrite: {bundle}'
    assert not archive.exists(), f'Refusing overwrite: {archive}'
    assert (drive_v2 / 'promoted_stage_b_v2.json').is_file()
    assert (run_output / 'metrics.jsonl').is_file()

    subprocess.run([
        'python', 'scripts/artifacts/export_stage_b_v2.py',
        '--checkpoint-root', str(checkpoint_root),
        '--run-output', str(run_output),
        '--report-root', str(drive_v2),
        '--destination', str(bundle),
        '--archive', str(archive),
    ], check=True)
    subprocess.run([
        'python', 'scripts/artifacts/verify_preserved_run.py',
        '--root', str(bundle),
    ], check=True)

    preservation_drive = drive_v2 / 'preservation'
    preservation_drive.mkdir(parents=True, exist_ok=True)

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open('rb') as source_file:
            for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    for source in (archive, checksum):
        destination = preservation_drive / source.name
        assert not destination.exists(), f'Refusing overwrite: {destination}'
        temporary = destination.with_suffix(destination.suffix + '.partial')
        temporary.unlink(missing_ok=True)
        shutil.copy2(source, temporary)
        assert sha256_file(source) == sha256_file(temporary)
        os.replace(temporary, destination)

    print({
        'archive': str(preservation_drive / archive.name),
        'archive_bytes': archive.stat().st_size,
        'sha256_record': checksum.read_text().strip(),
        'bundle': str(bundle),
    })
    print('STAGE B V2 PHYSICAL PRESERVATION: VERIFIED')
    """),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "colab_stage_b_v2.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells.")
