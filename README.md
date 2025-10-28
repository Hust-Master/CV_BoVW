# Object detection by BoVW
## Tổng quan

Đây là một implementation hoàn chỉnh của pipeline nhận dạng đối tượng sử dụng Bag-of-Visual-Words (BoVW) áp dụng cho bộ dữ liệu CIFAR-10.  
Pipeline sử dụng CNN pretrained (ResNet18) làm bộ trích xuất descriptor cục bộ, xây dựng từ vựng thị giác với MiniBatchKMeans, biểu diễn ảnh dưới dạng histogram của visual words, và huấn luyện Linear SVM.


## Giải thích các thành phần chính

### 1. Tại sao sử dụng CNN cho descriptors?

- Ảnh CIFAR có kích thước rất nhỏ (32×32 pixels)
- Các bộ phát hiện keypoint cổ điển (SIFT) hoạt động kém trên ảnh nhỏ
- Giải pháp: Resize ảnh lên 224×224 và trích xuất dense local descriptors từ lớp convolutional trung gian của mạng pretrained
- Kết quả: Descriptors hiện đại, có ý nghĩa semantic

### 2. Dense descriptors → Bag of Visual Words

- Feature map của mỗi ảnh tạo ra nhiều local descriptors (một vector cho mỗi vị trí không gian)
- KMeans cluster các descriptors này thành K visual words (codebook)
- Mỗi ảnh được biểu diễn bằng histogram về số lượng descriptors rơi vào mỗi visual word

### 3. Tại sao dùng MiniBatchKMeans?

- Số lượng descriptors rất lớn (hàng trăm nghìn)
- MiniBatchKMeans nhanh hơn và tiết kiệm bộ nhớ hơn

### 4. Classifier

- Linear SVM (LinearSVC) hoạt động tốt với histogram features
- Có thể thử SVC(kernel='rbf') (chậm hơn) hoặc logistic regression

### 5. Evaluation

- Script in ra: overall accuracy, per-class precision/recall/F1, confusion matrix

### 6. Lưu trữ

- Codebook và SVM được lưu với joblib để tái sử dụng

## Hyperparameters có thể điều chỉnh

- **CODEBOOK_SIZE**: 128, 256, 512
- **NUM_DESCRIPTORS_FOR_KMEANS**: nhiều hơn = tốt hơn nhưng chậm hơn
- **Conv layer**: layer3 vs layer4
- **SVM regularization C**

## Performance Tips

1. Sử dụng GPU cho descriptor extraction (PyTorch tự động dùng DEVICE)
2. Tăng BATCH_SIZE nếu GPU memory cho phép
3. Nếu KMeans là bottleneck: giảm số lượng descriptors hoặc dùng PCA để giảm chiều descriptor trước khi clustering

## Kết quả mong đợi

BoVW với CNN descriptors trên CIFAR-10 thường đạt hiệu suất hợp lý (accuracy phụ thuộc vào lựa chọn). Không đạt được hiệu suất của CNN classifiers end-to-end hiện đại nhưng là pipeline cổ điển tốt cho học tập và phân tích.


## Hướng phát triển

1. **Sử dụng SIFT/ORB cổ điển**: Cung cấp variant sử dụng OpenCV với SIFT hoặc ORB và so sánh sự khác biệt
2. **Chạy demo ngắn**: Hiển thị expected outputs với dataset nhỏ hơn
3. **Thêm PCA**: Giảm chiều của descriptors trước KMeans để tiết kiệm bộ nhớ và tăng tốc độ
4. **Nâng cấp encoding**: Thay KMeans bằng VLAD hoặc Fisher vectors để có hiệu suất tốt hơn
5. **Colab notebook**: Cung cấp notebook với checkpoints và plots để dễ sử dụng


## Installation
- python 3.13.7

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


## 🚀 Run
1. Đảm bảo đã cài đặt các thư viện cần thiết
2. Chạy script: `python bovw_cifar10.py`
3. Script sẽ tự động:
   - Tải CIFAR-10 dataset
   - Trích xuất descriptors bằng ResNet18
   - Xây dựng visual vocabulary
   - Huấn luyện SVM classifier
   - Đánh giá và in kết quả
4. Models được lưu trong thư mục `.output/main/`


## Tham khảo
- [ChatGPT](https://chatgpt.com/c/69004392-eb24-8320-9bd9-7a835d1da0b9)