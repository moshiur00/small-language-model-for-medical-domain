"""Continually pretrain the Stage A decoder on the disjoint medical corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import yaml

from medical_slm.model import DecoderConfig
from medical_slm.training.config import StageBTrainingConfig, load_stage_b_config
from medical_slm.training.trainer import StageBTrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training_stage_b.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_stage_a.yaml"))
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument(
        "--checkpoint-backup-directory",
        type=Path,
        default=None,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-initialization-only", action="store_true")
    mode.add_argument("--baseline-only", action="store_true")
    mode.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--baseline-output", type=Path, default=None)
    return parser.parse_args()


def load_model_config(path: Path) -> DecoderConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    return DecoderConfig.from_mapping(values)


def override_config(
    config: StageBTrainingConfig,
    arguments: argparse.Namespace,
) -> StageBTrainingConfig:
    values = config.to_dict()
    if arguments.parent_checkpoint is not None:
        values["parent_checkpoint_directory"] = str(arguments.parent_checkpoint)
    if arguments.max_updates is not None:
        values["max_updates"] = arguments.max_updates
    if arguments.checkpoint_backup_directory is not None:
        values["checkpoint_backup_directory"] = str(
            arguments.checkpoint_backup_directory
        )
    return StageBTrainingConfig.from_mapping(values)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    arguments = parse_arguments()
    config = override_config(load_stage_b_config(arguments.config), arguments)
    trainer = StageBTrainer(config, load_model_config(arguments.model_config))
    if arguments.resume is not None:
        trainer.resume(arguments.resume)

    if arguments.verify_initialization_only:
        if trainer.optimizer.state_dict()["state"]:
            raise RuntimeError("Stage B optimizer unexpectedly contains parent state.")
        if trainer.state.update != 0 or trainer.state.consumed_tokens != 0:
            raise RuntimeError("Stage B training counters did not start at zero.")
        parent = trainer.checkpoint_lineage["parent"]
        trainer.metric_logger.close()
        logging.getLogger(__name__).info(
            "Stage B initialization verified: parent=%s parameters=%d ",
            parent["checkpoint_name"],
            trainer.model.parameter_count(),
        )
        return
    if arguments.baseline_only:
        medical = trainer.evaluate()
        general = trainer.last_general_evaluation
        if general is None:
            raise RuntimeError("General validation did not produce a result.")
        if trainer.state.update != 0 or trainer.state.consumed_tokens != 0:
            raise RuntimeError(
                "Baseline evaluation must run before any Stage B optimizer update."
            )
        general_limit = general.loss * (
            1.0 + config.general_loss_max_degradation_fraction
        )
        baseline_output = arguments.baseline_output or (
            Path(config.output_directory) / "stage_b_baseline.json"
        )
        baseline_output.parent.mkdir(parents=True, exist_ok=True)
        baseline_output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "optimizer_updates": trainer.state.update,
                    "consumed_tokens": trainer.state.consumed_tokens,
                    "model_parameters": trainer.model.parameter_count(),
                    "device": str(trainer.device),
                    "precision": trainer.precision.name,
                    "parent": trainer.checkpoint_lineage["parent"],
                    "data": trainer.checkpoint_lineage["data"],
                    "medical_validation": asdict(medical),
                    "general_validation": asdict(general),
                    "general_loss_max_degradation_fraction": (
                        config.general_loss_max_degradation_fraction
                    ),
                    "general_loss_limit": general_limit,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        trainer.metric_logger.close()
        logging.getLogger(__name__).info(
            "Stage A baselines: medical_loss=%.6f general_loss=%.6f report=%s",
            medical.loss,
            general.loss,
            baseline_output,
        )
        return
    if arguments.overfit_one_batch:
        updates = arguments.max_updates or min(300, config.total_updates)
        state = trainer.train_overfit_one_batch(max_updates=updates)
    else:
        state = trainer.train()
    logging.getLogger(__name__).info(
        "Stage B finished: updates=%d consumed_tokens=%d epoch=%d",
        state.update,
        state.consumed_tokens,
        state.epoch,
    )


if __name__ == "__main__":
    main()
