"""Run Stage C response-only supervised instruction fine-tuning."""

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
    StageCSFTTrainingConfig,
    load_stage_c_sft_config,
)
from medical_slm.training.sft_trainer import StageCSFTTrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_stage_c_sft.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_stage_a.yaml")
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output-directory", type=Path)
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
    config: StageCSFTTrainingConfig,
    arguments: argparse.Namespace,
) -> StageCSFTTrainingConfig:
    values = config.to_dict()
    overrides = {
        "parent_checkpoint_directory": arguments.parent_checkpoint,
        "max_updates": arguments.max_updates,
        "learning_rate": arguments.learning_rate,
        "output_directory": arguments.output_directory,
        "checkpoint_backup_directory": arguments.checkpoint_backup_directory,
    }
    for name, value in overrides.items():
        if value is not None:
            values[name] = str(value) if isinstance(value, Path) else value
    if arguments.learning_rate is not None:
        ratio = config.final_learning_rate / config.learning_rate
        values["final_learning_rate"] = arguments.learning_rate * ratio
    return StageCSFTTrainingConfig.from_mapping(values)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    arguments = parse_arguments()
    config = override_config(load_stage_c_sft_config(arguments.config), arguments)
    trainer = StageCSFTTrainer(config, load_model_config(arguments.model_config))
    if arguments.resume is not None:
        trainer.resume(arguments.resume)

    if arguments.verify_initialization_only:
        if trainer.optimizer.state_dict()["state"]:
            raise RuntimeError("Stage C optimizer unexpectedly contains parent state.")
        if trainer.state.update or trainer.state.consumed_tokens:
            raise RuntimeError("Stage C progress counters did not start at zero.")
        logging.getLogger(__name__).info(
            "Stage C initialization verified: parent=%s",
            trainer.checkpoint_lineage["parent"]["checkpoint_name"],
        )
        trainer.metric_logger.close()
        return

    if arguments.baseline_only:
        sft = trainer.evaluate()
        medical = trainer.last_medical_evaluation
        general = trainer.last_general_evaluation
        if medical is None or general is None:
            raise RuntimeError("Stage C retention evaluation was incomplete.")
        output = arguments.baseline_output or (
            Path(config.output_directory) / "stage_c_baseline.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "status": "passed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "optimizer_updates": trainer.state.update,
            "consumed_supervised_tokens": trainer.state.consumed_tokens,
            "device": str(trainer.device),
            "precision": trainer.precision.name,
            "lineage": trainer.checkpoint_lineage,
            "sft_validation": asdict(sft),
            "medical_retention_validation": asdict(medical),
            "general_retention_validation": asdict(general),
        }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        trainer.metric_logger.close()
        logging.getLogger(__name__).info("Stage C baseline saved: %s", output)
        return

    if arguments.overfit_one_batch:
        state = trainer.train_overfit_one_batch(
            max_updates=arguments.max_updates or min(10, config.total_updates)
        )
    else:
        state = trainer.train()
    logging.getLogger(__name__).info(
        "Stage C finished: updates=%d supervised_tokens=%d epoch=%d",
        state.update,
        state.consumed_tokens,
        state.epoch,
    )


if __name__ == "__main__":
    main()
