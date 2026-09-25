import sys
import platform
import multiprocessing

print("Python:", sys.version)
print("Platform:", platform.platform())
print("CPUs:", multiprocessing.cpu_count())

try:
    import torch
    print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch not installed")

packages = [
    "numpy", "pandas", "scipy", "sklearn", "lightgbm", "catboost", "xgboost",
    "sentence_transformers", "transformers", "polars", "rapidfuzz", "faiss", "tqdm"
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"{pkg}: available")
    except ImportError:
        print(f"{pkg}: NOT available")
