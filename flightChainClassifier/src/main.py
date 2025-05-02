# ────────────────────────────────────────────────────────────────
#  flightChainClassifier / src / main.py
# ────────────────────────────────────────────────────────────────
"""
Top-level orchestration script for the Flight-Chain-Delay pipeline.

CLI options:
  • Stage control: --skip-data / --skip-train / --skip-eval
  • Model choice  : --model {cbam, simam, qtsimam}
  • Data options  : --subsample, --balanced, --sim-factor
  • Network params: --lstm-layers, --lstm-hidden-size
  • Training opts : --batch-size, --epochs
  • Hyper-params  : --use-best-params  (loads Optuna JSON if present)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

try:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
        print(f"Adjusted sys.path: added project root → {project_dir}")
except Exception:
    project_dir = Path.cwd()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    print(
        "Warning: could not determine script path; "
        "assuming current working directory is project root."
    )

try:
    from src import config
    from src.data_processing.chain_constructor import run_chain_construction
    from src.training.trainer import run_training
    from src.evaluation.evaluate import run_evaluation
except Exception as e:
    print("\n‼️  Error importing pipeline modules:")
    traceback.print_exc()
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flight-Chain classification pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage-control flags
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data processing (chains already built).",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip model training."
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation.")

    # Model + data options
    parser.add_argument(
        "--model",
        default="simam",
        choices=["cbam", "simam", "qtsimam", "qtsimam_mp"],
        help="Which architecture to train/evaluate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=config.SUBSAMPLE_DATA,
        help="Fraction of rows kept when building chains " "(1.0 = full data).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Enable class-balanced oversampling on train set.",
    )
    parser.add_argument(
        "--sim-factor",
        type=int,
        default=config.SIM_FACTOR,
        metavar="K",
        help="Synthetic jitter copies per real chain; " "1 = no augmentation.",
    )

    # Network hyper-parameters
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=None,
        help="Override number of LSTM/QMogrifier layers.",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="Override LSTM hidden state size.",
    )
    # Training specifics
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Training epochs (overrides config).",
    )

    # Misc
    parser.add_argument(
        "--use-best-params",
        action="store_true",
        help="Load Optuna-tuned hyper-parameters if JSON exists.",
    )
    parser.add_argument(
        "--no-aug", action="store_true", help="Disable synthetic jitter altogether"
    )

    args = parser.parse_args()

    config.BALANCED = args.balanced
    config.SIM_FACTOR = max(1, args.sim_factor)
    config.USE_SIM_AUG = not args.no_aug and config.SIM_FACTOR > 1

    config.SUBSAMPLE_DATA = args.subsample
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.EPOCHS = args.epochs

    print("\n========== Pipeline ==========")
    print(f" model            : {args.model.upper()}")
    print(f" subsample        : {config.SUBSAMPLE_DATA}")
    print(f" balanced         : {config.BALANCED}")
    print(f" sim-factor       : {config.SIM_FACTOR}")
    print(f" lstm layers      : {args.lstm_layers or 'auto'}")
    print(f" lstm hidden size : {args.lstm_hidden_size or 'auto'}")
    print(f" batch-size       : {config.BATCH_SIZE}")
    print(f" epochs           : {config.EPOCHS}")
    print(
        f" augmentation     : {'ON ×'+str(config.SIM_FACTOR) if config.USE_SIM_AUG else 'OFF (pure chains)'}"
    )
    print("================================\n")

    if not args.skip_data:
        print("=== Stage 1 — Chain construction ===")
        t0 = time.time()
        run_chain_construction()
        print(f"Data processing complete ({time.time() - t0:.1f}s)\n")
    else:
        required = [
            config.TRAIN_CHAINS_FILE,
            config.TRAIN_LABELS_FILE,
            config.VAL_CHAINS_FILE,
            config.VAL_LABELS_FILE,
            config.TEST_CHAINS_FILE,
            config.TEST_LABELS_FILE,
        ]
        if not all(p.exists() for p in required):
            print("‼️  --skip-data used but processed .npy files are missing.")
            sys.exit(1)

    if not args.skip_train:
        print(f"=== Stage 2 — Training ({args.model.upper()}) ===")
        t0 = time.time()
        hp = None
        if args.use_best_params:
            hp = config.load_best_hyperparameters()
        run_training(
            model_type=args.model,
            lstm_layers=args.lstm_layers,
            lstm_hidden_size=args.lstm_hidden_size,
            hyperparams=hp,
        )
        print(f"Training finished ({time.time() - t0:.1f}s)\n")
    else:
        if not args.skip_eval and not config.MODEL_SAVE_PATH.exists():
            print("‼️  --skip-train given but model checkpoint is missing.")
            sys.exit(1)

    if not args.skip_eval:
        print(f"=== Stage 3 — Evaluation ({args.model.upper()}) ===")
        t0 = time.time()
        run_evaluation(
            model_type=args.model,
            lstm_layers=args.lstm_layers,
            lstm_hidden_size=args.lstm_hidden_size,
        )
        print(f"Evaluation finished ({time.time() - t0:.1f}s)\n")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
