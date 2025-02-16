import torch

# Проверка доступности CUDA
print("CUDA available:", torch.cuda.is_available())

# Если CUDA доступна, выведем информацию о GPU
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))