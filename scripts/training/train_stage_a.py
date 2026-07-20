"""Train the Stage A decoder-only causal language model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from medical_slm.model import DecoderConfig
from medical_slm.training.config import StageATrainingConfig, load_stage_a_config
from medical_slm.training.trainer import StageATrainer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training_stage_a.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_stage_a.yaml"))
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument(
        "--checkpoint-backup-directory",
        type=Path,
        default=None,
        help="Optional second filesystem for verified checkpoint mirrors.",
    )
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Repeatedly optimize one real Stage A batch as an alignment diagnostic.",
    )
    return parser.parse_args()


def load_model_config(path: Path) -> DecoderConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Model configuration root must be a mapping.")
    return DecoderConfig.from_mapping(values)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    arguments = parse_arguments()
    training_config = load_stage_a_config(arguments.config)
    if arguments.max_updates is not None:
        values = training_config.to_dict()
        values["max_updates"] = arguments.max_updates
        training_config = StageATrainingConfig.from_mapping(values)
    if arguments.checkpoint_backup_directory is not None:
        values = training_config.to_dict()
        values["checkpoint_backup_directory"] = str(
            arguments.checkpoint_backup_directory
        )
        training_config = StageATrainingConfig.from_mapping(values)
    trainer = StageATrainer(training_config, load_model_config(arguments.model_config))
    if arguments.resume is not None:
        trainer.resume(arguments.resume)
    if arguments.overfit_one_batch:
        overfit_updates = arguments.max_updates or min(300, training_config.total_updates)
        final_state = trainer.train_overfit_one_batch(max_updates=overfit_updates)
    else:
        final_state = trainer.train()
    logging.getLogger(__name__).info(
        "Training finished: updates=%d consumed_tokens=%d epoch=%d",
        final_state.update,
        final_state.consumed_tokens,
        final_state.epoch,
    )


if __name__ == "__main__":
    main()
