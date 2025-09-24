import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.cnn3d import CNN3D
from models.resnet3d import ResNet3D
from models.densenet3d import DenseNet3D
from models.r2plus1d import R2Plus1D
from data.loaders import ABIDEDataset
from training.trainer import train
from main_helpers import preprocess_fmri, fetch_abide_subjects  # helper functions

# ----------------- Config -----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 2
batch_size = 1
n_subjects = 3  # small subset for testing

# ----------------- Models to compare -----------------
models_dict = {
    "CNN3D": CNN3D,
    "ResNet3D": ResNet3D,
    "DenseNet3D": DenseNet3D,
    "R2Plus1D": R2Plus1D
}

# ----------------- Load and preprocess ABIDE -----------------
subjects = fetch_abide_subjects(n_subjects=n_subjects)
loader = DataLoader(ABIDEDataset(subjects), batch_size=batch_size, shuffle=True)

# ----------------- Train and evaluate all models -----------------
results = {}
for name, ModelClass in models_dict.items():
    print(f"\nTraining model: {name}")
    model = ModelClass()
    _, metrics = train(model, loader, epochs=epochs, lr=1e-3, device=device, verbose=True)
    results[name] = metrics

# ----------------- Print summary -----------------
print("\n--- Model Comparison ---")
for name, metrics in results.items():
    print(f"{name}: Acc={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

# ----------------- Plot comparison -----------------
model_names = list(results.keys())
accuracy = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1'] for m in model_names]

x = range(len(model_names))
plt.figure(figsize=(8,5))
plt.bar(x, accuracy, width=0.4, label='Accuracy', align='center')
plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1 Score', align='center')
plt.xticks([i + 0.2 for i in x], model_names)
plt.ylabel("Score")
plt.title("Model Comparison (Accuracy vs F1)")
plt.legend()
plt.show()
