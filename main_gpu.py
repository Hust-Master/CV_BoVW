"""
bovw_cifar10.py

Full BoVW pipeline on CIFAR-10 using CNN dense descriptors (ResNet18),
MiniBatchKMeans codebook, and Linear SVM.

Dependencies:
- python >= 3.8
- torch, torchvision
- scikit-learn
- numpy
- matplotlib
- tqdm
- joblib

Install (example):
pip install torch torchvision scikit-learn numpy matplotlib tqdm joblib
"""

import time
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------
# Config / hyperparams
# -----------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- GPU / CUDA check ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
else:
    DEVICE = torch.device("cpu")
    gpu_name = None
    cuda_version = None

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {gpu_name}")
print(f"CUDA version: {cuda_version}")
print(f"Using device: {DEVICE}")

# --- other configs ---
BATCH_SIZE = 256            # for descriptor extraction
NUM_DESCRIPTORS_FOR_KMEANS = 200_000  # sample descriptors from training images
CODEBOOK_SIZE = 256         # number of visual words
SVM_C = 1.0
N_JOBS = -1                 # for scikit-learn where supported
RESIZE = 224                # resize CIFAR images to 224x224 for ResNet conv features

# Output files
CODEBOOK_PATH = "./output/main_gpu/bovw_codebook.joblib"
SVM_PATH = "./output/main_gpu/bovw_svm.joblib"
HIST_CACHE_TRAIN = "./output/main_gpu/hist_train.npy"
HIST_CACHE_TEST = "./output/main_gpu/hist_test.npy"

# -----------------------
# Data loading
# -----------------------
transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_set.classes

# -----------------------
# Feature extractor
# -----------------------
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = model.to(DEVICE)
model.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, backbone, layer='layer3'):
        super().__init__()
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2
        )
        if layer in ['layer3', 'layer4']:
            self.features.add_module('layer3', backbone.layer3)
        if layer == 'layer4':
            self.features.add_module('layer4', backbone.layer4)

    def forward(self, x):
        return self.features(x)

feat_extractor = FeatureExtractor(model, layer='layer3').to(DEVICE)
feat_extractor.eval()

# -----------------------
# Descriptor extraction
# -----------------------
def extract_descriptors_from_batch(images_tensor):
    with torch.no_grad():
        feats = feat_extractor(images_tensor)
        B, C, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).contiguous()
        feats = feats.view(B, h*w, C)
        feats = feats.cpu().numpy()
    return [feats[i] for i in range(feats.shape[0])]

# -----------------------
# 1) Build codebook
# -----------------------
def build_codebook(sample_descriptors, k=CODEBOOK_SIZE, batch_size=1000):
    print("Fitting MiniBatchKMeans (this can take time)...")
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=RANDOM_SEED, verbose=1)
    mbk.fit(sample_descriptors)
    return mbk

def gather_descriptors_for_kmeans(max_descriptors=NUM_DESCRIPTORS_FOR_KMEANS):
    descriptors = []
    collected = 0
    for imgs, _ in tqdm(train_loader, desc="Gathering descriptors"):
        imgs = imgs.to(DEVICE)
        ds_batch = extract_descriptors_from_batch(imgs)
        for d in ds_batch:
            descriptors.append(d)
            collected += d.shape[0]
            if collected >= max_descriptors:
                break
        if collected >= max_descriptors:
            break
    descriptors = np.vstack(descriptors)
    if descriptors.shape[0] > max_descriptors:
        idx = np.random.choice(descriptors.shape[0], max_descriptors, replace=False)
        descriptors = descriptors[idx]
    print(f"Collected descriptors shape: {descriptors.shape}")
    return descriptors

# -----------------------
# 2) Build histograms
# -----------------------
def image_to_bovw_hist(descriptors, kmeans):
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters+1))
    hist = hist.astype(np.float32)
    hist = normalize(hist.reshape(1, -1), norm='l2').ravel()
    return hist

def compute_histograms_for_loader(loader, kmeans, cache_path=None):
    all_hists, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Computing histograms"):
        imgs = imgs.to(DEVICE)
        ds_batch = extract_descriptors_from_batch(imgs)
        for d in ds_batch:
            h = image_to_bovw_hist(d, kmeans)
            all_hists.append(h)
        all_labels.extend(labels.numpy().tolist())
    H = np.vstack(all_hists)
    Y = np.array(all_labels)
    if cache_path:
        np.save(cache_path, {'hist': H, 'labels': Y})
    return H, Y

# -----------------------
# 3) Train classifier
# -----------------------
def train_svm(X_train, y_train, C=SVM_C):
    print("Training LinearSVC...")
    clf = LinearSVC(C=C, max_iter=5000)
    clf.fit(X_train, y_train)
    return clf

# -----------------------
# Main pipeline
# -----------------------
def main():
    start_total = time.time()

    if os.path.exists(CODEBOOK_PATH):
        print("Loading existing codebook:", CODEBOOK_PATH)
        kmeans = joblib.load(CODEBOOK_PATH)
    else:
        t0 = time.time()
        sampled_desc = gather_descriptors_for_kmeans(NUM_DESCRIPTORS_FOR_KMEANS)
        kmeans = build_codebook(sampled_desc, k=CODEBOOK_SIZE, batch_size=4096)
        joblib.dump(kmeans, CODEBOOK_PATH)
        print("Saved codebook to", CODEBOOK_PATH, " (took %.1f s)" % (time.time()-t0))

    if os.path.exists(HIST_CACHE_TRAIN):
        print("Loading train hist cache")
        data = np.load(HIST_CACHE_TRAIN, allow_pickle=True).item()
        X_train, y_train = data['hist'], data['labels']
    else:
        t1 = time.time()
        X_train, y_train = compute_histograms_for_loader(train_loader, kmeans, cache_path=None)
        np.save(HIST_CACHE_TRAIN, {'hist': X_train, 'labels': y_train})
        print("Computed train histograms (took %.1f s)" % (time.time()-t1))

    if os.path.exists(HIST_CACHE_TEST):
        print("Loading test hist cache")
        data = np.load(HIST_CACHE_TEST, allow_pickle=True).item()
        X_test, y_test = data['hist'], data['labels']
    else:
        t2 = time.time()
        X_test, y_test = compute_histograms_for_loader(test_loader, kmeans, cache_path=None)
        np.save(HIST_CACHE_TEST, {'hist': X_test, 'labels': y_test})
        print("Computed test histograms (took %.1f s)" % (time.time()-t2))

    if os.path.exists(SVM_PATH):
        print("Loading SVM:", SVM_PATH)
        clf = joblib.load(SVM_PATH)
    else:
        t3 = time.time()
        clf = train_svm(X_train, y_train, C=SVM_C)
        joblib.dump(clf, SVM_PATH)
        print("Saved SVM to", SVM_PATH, " (took %.1f s)" % (time.time()-t3))

    t4 = time.time()
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix shape:", cm.shape)
    print("Evaluation time: %.1f s" % (time.time()-t4))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print("Total pipeline time: %.1f s" % (time.time() - start_total))

if __name__ == "__main__":
    directory_path = Path(CODEBOOK_PATH).parent
    directory_path.mkdir(parents=True, exist_ok=True)
    directory_path = Path(SVM_PATH).parent
    directory_path.mkdir(parents=True, exist_ok=True)
    directory_path = Path(HIST_CACHE_TRAIN).parent
    directory_path.mkdir(parents=True, exist_ok=True)
    directory_path = Path(HIST_CACHE_TEST).parent
    directory_path.mkdir(parents=True, exist_ok=True)
    
    main()
