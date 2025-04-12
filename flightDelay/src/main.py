# flightDelay/src/main.py

import sys
import os
import time
import traceback
import warnings
# --- Path Setup ---
# Add the 'src' directory parent (e.g., 'flightDelay') to the path
# to allow imports like 'from src.data_processing import ...'
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*X does not have valid feature names.*"
)

try:
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(script_path)
    flight_delay_dir = os.path.dirname(src_dir)
    if flight_delay_dir not in sys.path:
        sys.path.insert(0, flight_delay_dir)
    print(f"Adjusted sys.path: Added '{flight_delay_dir}'")
except NameError:
    print("Warning: Could not determine script path automatically (__file__ not defined).")
    print("Assuming script is run from a location where 'src' package is accessible.")

# Import necessary functions from modules within the 'src' package
try:
    from src.data_processing import load_merge
    from src.data_processing import preprocess_reframe
    from src.data_processing import feature_calculation
    from src.data_processing import partition_prepare
    from src.modeling import train
    from src.modeling import predict
    print("Successfully imported pipeline modules.")
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Ensure __init__.py files exist in 'src', 'src/data_processing', 'src/modeling'.")
    print(f"Current sys.path: {sys.path}")
    traceback.print_exc()
    sys.exit(1)

def run_pipeline():
    """Executes the full FDPP-ML pipeline: Data Processing -> Training -> Prediction."""
    pipeline_start_time = time.time()
    print("--- Starting FDPP-ML Pipeline ---")

    # === Stage 1: Data Processing ===
    print("\n=== Stage 1: Data Processing ===")
    stage1_start_time = time.time()
    try:
        print("\nStep 1.1: Merging raw files...")
        load_merge.run_merge()
        print("-" * 30)

        print("Step 1.2: Preprocessing and Reframing to Points...")
        preprocess_reframe.run_preprocess_reframe()
        print("-" * 30)

        print("Step 1.3: Calculating Features (FTD, Historical PFD) & Partitioning...")
        feature_calculation.run_feature_calculation()
        print("-" * 30)

        print("Step 1.4: Preparing Future Dataset (Merging last historical points)...")
        partition_prepare.run_partition_prepare()
        print("-" * 30)

        stage1_end_time = time.time()
        print(f"=== Data Processing Stage Complete ({stage1_end_time - stage1_start_time:.2f} seconds) ===")
    except SystemExit: # Catch sys.exit calls from within modules
        print("\n--- Data Processing Stage Halted due to Critical Error ---")
        sys.exit(1) # Re-raise to stop pipeline
    except Exception as e:
        print(f"\n--- ERROR during Data Processing Stage ---")
        traceback.print_exc()
        sys.exit(1)


    # === Stage 2: Modeling (Training and Prediction) ===
    print("\n=== Stage 2: Modeling ===")
    stage2_start_time = time.time()
    try:
        print("\nStep 2.1: Training Model...")
        train.run_training() # Trains and saves the model
        print("-" * 30)

        print("Step 2.2: Performing Iterative Prediction...")
        predict.run_prediction() # Loads model, predicts on future data, saves predictions
        print("-" * 30)

        stage2_end_time = time.time()
        print(f"=== Modeling Stage Complete ({stage2_end_time - stage2_start_time:.2f} seconds) ===")
    except SystemExit:
        print("\n--- Modeling Stage Halted due to Critical Error ---")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- ERROR during Modeling Stage ---")
        traceback.print_exc()
        sys.exit(1)


    # --- Pipeline End ---
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    print("\n--- FDPP-ML Pipeline Finished Successfully ---")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time / 60.0:.2f} minutes)")

if __name__ == "__main__":
    # This ensures the script runs the pipeline when executed directly
    run_pipeline()
