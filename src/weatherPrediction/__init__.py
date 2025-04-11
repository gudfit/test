# src/weatherPrediction/__init__.py
"""
Weather prediction module initialization.
This module serves as an interface to the actual implementation in 2_weatherPrediction.
"""
import os
import sys
import importlib.util

# Path to the actual implementation directory
_impl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "2_weatherPrediction")

# Import train.py first
_train_path = os.path.join(_impl_dir, "train.py")
if os.path.exists(_train_path):
    _train_spec = importlib.util.spec_from_file_location(
        "src.2_weatherPrediction.train", 
        _train_path
    )
    train = importlib.util.module_from_spec(_train_spec)
    sys.modules["src.weatherPrediction.train"] = train
    _train_spec.loader.exec_module(train)

# Then import evaluate.py
_eval_path = os.path.join(_impl_dir, "evaluate.py")
if os.path.exists(_eval_path):
    _eval_spec = importlib.util.spec_from_file_location(
        "src.2_weatherPrediction.evaluate", 
        _eval_path
    )
    evaluate = importlib.util.module_from_spec(_eval_spec)
    sys.modules["src.weatherPrediction.evaluate"] = evaluate
    _eval_spec.loader.exec_module(evaluate)

# Clean up temporary variables
del _impl_dir, _train_path, _train_spec, _eval_path, _eval_spec
