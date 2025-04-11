# src/dataPreprocessing/weatherPredictionProcessing/__init__.py
"""
Weather prediction preprocessing module initialization.
This module serves as an interface to the actual implementation in 1_dataPreprocessing/weatherPredictionProcessing.
"""
import os
import sys
import importlib.util

# Path to the actual implementation directory
_impl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       "1_dataPreprocessing", "weatherPredictionProcessing")

# First, import the feature_engineering module and make its functions available
_fe_path = os.path.join(_impl_dir, "feature_engineering.py")
if os.path.exists(_fe_path):
    _fe_spec = importlib.util.spec_from_file_location(
        "src.1_dataPreprocessing.weatherPredictionProcessing.feature_engineering", 
        _fe_path
    )
    feature_engineering = importlib.util.module_from_spec(_fe_spec)
    sys.modules["src.dataPreprocessing.weatherPredictionProcessing.feature_engineering"] = feature_engineering
    _fe_spec.loader.exec_module(feature_engineering)

# Now import the merge_data module
_md_path = os.path.join(_impl_dir, "merge_data.py")
if os.path.exists(_md_path):
    _md_spec = importlib.util.spec_from_file_location(
        "src.1_dataPreprocessing.weatherPredictionProcessing.merge_data", 
        _md_path
    )
    merge_data = importlib.util.module_from_spec(_md_spec)
    sys.modules["src.dataPreprocessing.weatherPredictionProcessing.merge_data"] = merge_data
    _md_spec.loader.exec_module(merge_data)

# Finally, import preprocessing after feature_engineering is already loaded
_pp_path = os.path.join(_impl_dir, "preprocessing.py")
if os.path.exists(_pp_path):
    _pp_spec = importlib.util.spec_from_file_location(
        "src.1_dataPreprocessing.weatherPredictionProcessing.preprocessing", 
        _pp_path
    )
    preprocessing = importlib.util.module_from_spec(_pp_spec)
    sys.modules["src.dataPreprocessing.weatherPredictionProcessing.preprocessing"] = preprocessing
    _pp_spec.loader.exec_module(preprocessing)

# Clean up temporary variables
del _impl_dir, _fe_path, _fe_spec, _md_path, _md_spec, _pp_path, _pp_spec
