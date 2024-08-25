from torch.utils.data import DataLoader
import numpy as np
from dataset import FootballDatasets


def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    test_img_dir,
    test_mask_dir,
    train_transforms,
    val_transforms,
    test_transforms,
    batch_size,
    num_of_workers,
    pin_memory,
):
    train_ds = FootballDatasets(
        image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transforms
    )
    val_ds = FootballDatasets(
        image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transforms
    )
    test_ds = FootballDatasets(
        image_dir=test_img_dir, mask_dir=test_mask_dir, transform=test_transforms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_of_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_of_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_of_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def iou(outputs, labels) -> float:
    intersection = np.sum(np.logical_and(outputs, labels), axis=(1, 2, 3))
    union = np.sum(np.logical_or(outputs, labels), axis=(1, 2, 3))
    iou = intersection / union
    return np.mean(iou)
