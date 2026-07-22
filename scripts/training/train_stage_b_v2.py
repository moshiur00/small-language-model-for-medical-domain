"""Run retention-aware Stage B v2 continual medical pretraining."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import yaml

from medical_slm.model import DecoderConfig
from medical_slm.training.config import (
    StageBV2TrainingConfig,
    load_stage_b_v2_config,
)
from medical_slm.training.trainer import StageBV2Trainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_stage_b_v2_selective_l2sp.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_stage_a.yaml"),
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--checkpoint-backup-directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-initialization-only", action="store_true")
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--baseline-output", type=Path)
    return parser.parse_args()


def load_model_config(path: Path) -> DecoderConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    return DecoderConfig.from_mapping(values)


def override_config(
    config: StageBV2TrainingConfig,
    arguments: argparse.Namespace,
) -> StageBV2TrainingConfig:
    values = config.to_dict()
    if arguments.parent_checkpoint is not None:
        values["parent_checkpoint_directory"] = str(arguments.parent_checkpoint)
    if arguments.max_updates is not None:
        values["max_updates"] = arguments.max_updates
    if arguments.checkpoint_backup_directory is not None:
        values["checkpoint_backup_directory"] = str(
            arguments.checkpoint_backup_directory
        )
    return StageBV2TrainingConfig.from_mapping(values)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    arguments = parse_arguments()
    config = override_config(load_stage_b_v2_config(arguments.config), arguments)
    trainer = StageBV2Trainer(config, load_model_config(arguments.model_config))
    if arguments.resume is not None:
        trainer.resume(arguments.resume)

    if arguments.verify_initialization_only:
        if trainer.optimizer.state_dict()["state"]:
            raise RuntimeError("Stage B v2 optimizer unexpectedly contains state.")
        if trainer.state.update != 0 or trainer.state.consumed_tokens != 0:
            raise RuntimeError("Stage B v2 counters did not start at zero.")
        logging.getLogger(__name__).info(
            "Stage B v2 initialization verified: parent=%s trainable=%d frozen=%d",
            trainer.checkpoint_lineage["parent"]["checkpoint_name"],
            trainer.freezing_report.trainable_parameters,
            trainer.freezing_report.frozen_parameters,
        )
        trainer.metric_logger.close()
        return

    if arguments.baseline_only:
        medical = trainer.evaluate()
        general = trainer.last_general_evaluation
        if general is None:
            raise RuntimeError("General validation did not produce a result.")
        output = arguments.baseline_output or (
            Path(config.output_directory) / "stage_b_v2_baseline.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "optimizer_updates": trainer.state.update,
                    "consumed_tokens": trainer.state.consumed_tokens,
                    "device": str(trainer.device),
                    "precision": trainer.precision.name,
                    "lineage": trainer.checkpoint_lineage,
                    "freezing": asdict(trainer.freezing_report),
                    "medical_validation": asdict(medical),
                    "general_validation": asdict(general),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        trainer.metric_logger.close()
        logging.getLogger(__name__).info("Stage B v2 baseline saved: %s", output)
        return

    if arguments.overfit_one_batch:
        updates = arguments.max_updates or min(10, config.total_updates)
        state = trainer.train_overfit_one_batch(max_updates=updates)
    else:
        state = trainer.train()
    logging.getLogger(__name__).info(
        "Stage B v2 finished: updates=%d tokens=%d epoch=%d early_stop=%s",
        state.update,
        state.consumed_tokens,
        state.epoch,
        trainer.stop_requested,
    )


if __name__ == "__main__":
    main()
