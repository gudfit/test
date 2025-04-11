# src/dataPreprocessing/__init__.py
"""
Data preprocessing module initialization.
This module serves as an interface to the actual implementation in 1_dataPreprocessing.
"""
import os
import sys
import importlib.util

# Path to the actual implementation directory
_impl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "1_dataPreprocessing")

# Import the actual implementation module
_spec = importlib.util.spec_from_file_location(
    "src.1_dataPreprocessing", 
    os.path.join(_impl_dir, "__init__.py")
)
_impl_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl_module)

# Copy attributes from the implementation module to this module
for _attr in dir(_impl_module):
    if not _attr.startswith('_'):
        globals()[_attr] = getattr(_impl_module, _attr)

# Clean up temporary variables
del _impl_dir, _spec, _impl_module, _attr
