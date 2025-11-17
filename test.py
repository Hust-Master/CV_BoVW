import os

import joblib
from pathlib import Path

import torch
import torchvision
from main import BATCH_SIZE, CODEBOOK_PATH, DEVICE, RESIZE, SVM_PATH, extract_descriptors_from_batch, image_to_bovw_hist
import torchvision.transforms as T


transform = T.Compose([
    T.Resize((RESIZE, RESIZE)),
    T.ToTensor(),
    # normalize with ImageNet stats because we use pretrained ImageNet model
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_set.classes

def test_demo():
    """
    Test object recognition on images in data/demo folder.
    Uses trained codebook and SVM to predict object classes.
    """
    # Load trained models
    if not os.path.exists(CODEBOOK_PATH) or not os.path.exists(SVM_PATH):
        print("Error: Codebook or SVM model not found. Please run main() first.")
        return
    
    print("\n" + "="*50)
    print("Testing on demo images")
    print("="*50)
    
    kmeans = joblib.load(CODEBOOK_PATH)
    clf = joblib.load(SVM_PATH)
    
    demo_dir = Path("./data/demo")
    if not demo_dir.exists():
        print(f"Demo directory not found: {demo_dir}")
        return
    
    # Get all image files (png, jpg, jpeg)
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(demo_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"No images found in {demo_dir}")
        return
    
    print(f"Found {len(image_files)} images in demo folder\n")
    
    # Process each image
    results = []
    for img_path in image_files:
        try:
            # Load and transform image
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dimension
            
            # Extract descriptors
            descriptors = extract_descriptors_from_batch(img_tensor)[0]
            
            # Compute BoVW histogram
            hist = image_to_bovw_hist(descriptors, kmeans)
            
            # Predict class
            pred_label = clf.predict(hist.reshape(1, -1))[0]
            pred_class = class_names[pred_label]
            
            # Get prediction probabilities (for LinearSVC, we use decision function)
            decision = clf.decision_function(hist.reshape(1, -1))[0]
            
            results.append({
                'filename': img_path.name,
                'predicted_class': pred_class,
                'pred_label': pred_label,
                'decision_score': decision
            })
            
            print(f"✓ {img_path.name:40s} → {pred_class}")
            
        except Exception as e:
            print(f"✗ {img_path.name:40s} → Error: {str(e)}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Total images tested: {len(results)}")
    if results:
        pred_classes = [r['predicted_class'] for r in results]
        unique_classes = set(pred_classes)
        print(f"Predicted classes: {', '.join(sorted(unique_classes))}")
        print("\nDetailed Results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['filename']:40s} → {r['predicted_class']}")
    
    return results

if __name__ == "__main__":
    test_demo()
