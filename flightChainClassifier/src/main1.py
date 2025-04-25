# flightChainClassifier/src/main.py
import sys
import os
import time
import argparse
import traceback

# --- Path Setup ---
try:
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(script_path)
    project_dir = os.path.dirname(src_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
        print(f"Adjusted sys.path: Added '{project_dir}'")
except NameError:
    project_dir = os.path.abspath('.')
    src_dir = os.path.join(project_dir, 'src')
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    print("Warning: Could not determine script path automatically. Assuming running from project root.")

# --- Imports ---
try:
    from src import config
    from src.data_processing.chain_constructor import run_chain_construction
    from src.training.trainer import run_training
    from src.evaluation.evaluate import run_evaluation
    print("Successfully imported pipeline modules.")
except ModuleNotFoundError as e:
    print(f"Error: Could not import pipeline module - {e}.")
    print("Please ensure all necessary __init__.py files exist.")
    print(f"PYTHONPATH might need adjustment. Current sys.path: {sys.path}")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """
    Main function to orchestrate the flight chain classification pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Flight Chain Delay Classification Pipeline: Process data, train, and evaluate models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--skip-data', action='store_true',
                        help="Skip data processing (chain construction). Assumes data exists.")
    parser.add_argument('--skip-train', action='store_true',
                        help="Skip model training. Assumes a trained model exists for evaluation.")
    parser.add_argument('--skip-eval', action='store_true',
                        help="Skip model evaluation.")
    parser.add_argument('--model', type=str, default='simam', choices=['cbam', 'simam', 'qtsimam'],
                        help="Model architecture type to train and evaluate.")
    parser.add_argument('--use-best-params', action='store_true',
                        help=f"Load hyperparameters from '{config.BEST_PARAMS_FILE.name}' for training (if it exists).")
    parser.add_argument('--subsample', type=float, default=config.SUBSAMPLE_DATA,
                        help="Override subsample fraction for data processing/training (e.g., 0.1 for 10%). Set to 1.0 for full data.")
    # NEW: Flag for balancing the training data.
    parser.add_argument('--balanced', action='store_true',
                        help="If provided, balance the training dataset via oversampling of minority classes.")

    args = parser.parse_args()

    # Set the balanced flag in config
    config.BALANCED = args.balanced

    print(f"\n=========================================")
    print(f"--- Running Pipeline ---")
    print(f"Model selected: {args.model.upper()}")
    print(f"Use best params: {args.use_best_params}")
    print(f"Subsample override: {args.subsample if args.subsample != config.SUBSAMPLE_DATA else 'Using config default'}")
    print(f"Balanced Training Data: {config.BALANCED}")
    print(f"=========================================")
    pipeline_start = time.time()

    original_subsample = config.SUBSAMPLE_DATA
    if args.subsample != config.SUBSAMPLE_DATA:
        print(f"Overriding config.SUBSAMPLE_DATA with command line value: {args.subsample}")
        config.SUBSAMPLE_DATA = args.subsample

    best_params = None
    if args.use_best_params:
        if not args.skip_train:
            best_params = config.load_best_hyperparameters()
            if best_params is None:
                print(f"Warning: --use-best-params specified, but '{config.BEST_PARAMS_FILE.name}' not found or failed to load. Training will use defaults.")
        else:
            print("Info: --use-best-params flag ignored because training is skipped.")

    if not args.skip_data:
        stage_start = time.time()
        print("\n=== Stage 1: Data Processing (Chain Construction) ===")
        try:
            run_chain_construction()
            print(f"\n--- Data Processing Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("\nPipeline halted during data processing.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nData processing stopped manually.")
            sys.exit(1)
        except Exception as e:
            print(f"\n--- ERROR during Data Processing Stage ---")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Data Processing ---")
        required_files = [config.TRAIN_CHAINS_FILE, config.TRAIN_LABELS_FILE,
                          config.VAL_CHAINS_FILE, config.VAL_LABELS_FILE,
                          config.TEST_CHAINS_FILE, config.TEST_LABELS_FILE,
                          config.DATA_STATS_FILE]
        if not all(f.exists() for f in required_files):
            print("Error: --skip-data used, but required processed data files are missing in:")
            print(f"  {config.PROCESSED_DATA_DIR}")
            print("Please run the pipeline without --skip-data first.")
            sys.exit(1)
        else:
            print("Required processed data files found.")

    if not args.skip_train:
        stage_start = time.time()
        print(f"\n=== Stage 2: Training ({args.model.upper()} Model) ===")
        try:
            if args.model == 'qtsimam':
                from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
                run_training(model_type='qtsimam', hyperparams=best_params)
            else:
                run_training(model_type=args.model, hyperparams=best_params)
            print(f"\n--- Training Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("\nPipeline halted during training.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nTraining stopped manually.")
        except Exception as e:
            print(f"\n--- ERROR during Training Stage ---")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Training ---")
        if not args.skip_eval and not config.MODEL_SAVE_PATH.exists():
            print(f"Error: --skip-train used, but model file not found for evaluation at:")
            print(f"  {config.MODEL_SAVE_PATH}")
            print("Please run training first or ensure the model path is correct.")
            sys.exit(1)

    if not args.skip_eval:
        stage_start = time.time()
        print(f"\n=== Stage 3: Evaluation ({args.model.upper()} Model) ===")
        try:
            if args.model == 'qtsimam':
                from src.modeling.queue_augment_models import QTSimAM_CNN_LSTM_Model
                run_evaluation(model_type='qtsimam')
            else:
                run_evaluation(model_type=args.model)
            print(f"\n--- Evaluation Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("\nPipeline halted during evaluation.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nEvaluation stopped manually.")
            sys.exit(1)
        except Exception as e:
            print(f"\n--- ERROR during Evaluation Stage ---")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Evaluation ---")

    pipeline_end = time.time()
    total_duration = pipeline_end - pipeline_start
    print(f"\n--- Pipeline Finished ---")
    print(f"Total Execution Time: {total_duration:.2f} seconds ({total_duration / 60.0:.2f} minutes)")

    config.SUBSAMPLE_DATA = original_subsample

if __name__ == "__main__":
    main()

