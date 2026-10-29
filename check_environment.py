import platform
import sys
import torch

print("--- სისტემური ინფორმაცია ---")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: არ არის ხელმისაწვდომი (გამოიყენება CPU)")