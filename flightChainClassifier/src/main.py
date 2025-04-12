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
    print("Warning: Could not determine script path automatically.")

# --- Imports ---
try:
    from src import config # To access global config if needed
    from src.data_processing.chain_constructor import run_chain_construction
    from src.training.trainer import run_training
    from src.evaluation.evaluate import run_evaluation
    print("Successfully imported pipeline modules.")
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Ensure __init__.py files exist and PYTHONPATH is correct.")
    traceback.print_exc()
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Flight Chain Delay Classification Pipeline")
    parser.add_argument('--skip-data', action='store_true', help="Skip data processing (chain construction)")
    parser.add_argument('--skip-train', action='store_true', help="Skip model training")
    parser.add_argument('--skip-eval', action='store_true', help="Skip model evaluation")
    parser.add_argument('--model', type=str, default='simam', choices=['cbam', 'simam'],
                        help="Model type to train and evaluate ('cbam' or 'simam')")

    args = parser.parse_args()

    print(f"--- Running Pipeline ---")
    print(f"Model selected: {args.model.upper()}")
    pipeline_start = time.time()

    # --- 1. Data Processing ---
    if not args.skip_data:
        stage_start = time.time()
        print("\n=== Stage 1: Data Processing (Chain Construction) ===")
        try:
            run_chain_construction()
            print(f"--- Data Processing Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("Pipeline halted during data processing.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during data processing: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Data Processing ---")

    # --- 2. Training ---
    if not args.skip_train:
        stage_start = time.time()
        print(f"\n=== Stage 2: Training ({args.model.upper()} Model) ===")
        try:
            run_training(model_type=args.model)
            print(f"--- Training Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("Pipeline halted during training.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Training ---")

    # --- 3. Evaluation ---
    if not args.skip_eval:
        stage_start = time.time()
        print(f"\n=== Stage 3: Evaluation ({args.model.upper()} Model) ===")
        try:
            run_evaluation(model_type=args.model)
            print(f"--- Evaluation Complete ({time.time() - stage_start:.2f}s) ---")
        except SystemExit:
            print("Pipeline halted during evaluation.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n--- Skipping Evaluation ---")

    print(f"\n--- Pipeline Finished ({time.time() - pipeline_start:.2f}s) ---")


if __name__ == "__main__":
    main()
