# E1/E1B/knowledge_quant_utils.py
import os
import gzip

def get_dir_size(path="."):
    """Calculates the total size of a directory in megabytes (MB)."""
    total = 0
    # Check if path exists to prevent errors
    if not os.path.exists(path):
        return 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total / (1024 * 1024)

def get_gzip_size_and_compress(file_path):
    """
    Compresses a local text file with Gzip and returns the compressed size in MB.
    """
    with open(file_path, 'rb') as f_in:
        compressed_data = gzip.compress(f_in.read())
    
    return len(compressed_data) / (1024 * 1024)
