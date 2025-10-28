# Object detection by BoVW
- python 3.13.7

## Installation
Tạo môi trường ảo
```python
# Khởi tạo
python -m venv .venv

# Kích hoạt
.venv\Scripts\activate     # Window
# hoặc
source .venv/bin/activate  # Linux/Mac

# Xuất danh sách thư viện
pip freeze > requirements.txt

# Hủy kích hoạt
deactivate
```

Cài đặt thư viện
```python
pip install -r requirements.txt
```

Nếu có GPU NVIDIA, cài PyTorch có CUDA để tăng tốc [Tham khảo](https://pytorch.org/get-started/locally)

```python
#clean
pip uninstall torch torchvision torchaudio -y
pip list | findstr torch
pip cache purge

# GPU CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Kiểm tra lại
```python
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("CUDA version:", torch.version.cuda)

# Result:
# Torch version: 2.6.0+cu124
# CUDA available: True
# GPU name: Quadro P1000
# CUDA version: 12.4

```


## Tham khảo
- [ChatGPT](https://chatgpt.com/c/69004392-eb24-8320-9bd9-7a835d1da0b9)