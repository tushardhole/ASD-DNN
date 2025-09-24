import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses, title="Training Loss", save_path=None):
    """
    Plot training loss curve.

    Args:
        losses: list of loss values per epoch
        title: plot title
        save_path: if provided, saves the plot to file
    """
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_prediction_heatmap(volume, title="Prediction Heatmap", save_path=None):
    """
    Plot a 2D slice from a 3D volume as a heatmap.

    Args:
        volume: 3D numpy array (X x Y x Z)
        title: plot title
        save_path: if provided, saves the plot to file
    """
    # Take the middle slice along Z-axis
    z_mid = volume.shape[2] // 2
    slice_2d = volume[:, :, z_mid]

    plt.figure(figsize=(5, 5))
    plt.imshow(slice_2d, cmap='hot', origin='lower')
    plt.colorbar()
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
