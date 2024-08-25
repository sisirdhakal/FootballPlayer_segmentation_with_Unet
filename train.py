import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from unet import Unet
from utils import get_loaders
import os
from PIL import Image

# Hyperparameters
img_width = 512
img_height = 512
epochs = 2
lr = 1e-4
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
num_of_workers = 2
pin_memory = True

# directories for the datasets

base_dir = "football_dataset"
train_images = f"{base_dir}/train/images"
train_masks = f"{base_dir}/train/masks"

val_images = f"{base_dir}/val/images"
val_masks = f"{base_dir}/val/masks"

test_images = f"{base_dir}/test/images"
test_masks = f"{base_dir}/test/masks"

len(os.listdir(train_images)), len(os.listdir(train_masks))


# Train function
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    losses = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        with torch.amp.autocast(device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # updating the tqdm
        loop.set_postfix(loss=loss.item())

    return sum(losses) / len(losses)


# Validation function
def eval_fn(loader, model, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.float().unsqueeze(1).to(device=device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def Visualize_data():
    train_images_len = len(os.listdir(train_images))

    plt.figure(figsize=(12, 10))

    rows = 2
    columns = 2

    torch.manual_seed(42)

    random_indices = torch.randint(0, train_images_len, (rows,))
    # print("Random indices:", random_indices)

    for i, idx in enumerate(random_indices):

        image_filename = os.listdir(train_images)[idx.item()]

        img_path = os.path.join(train_images, image_filename)
        img = Image.open(img_path)

        img_size = img.size
        #     print(f'Image: {image_filename} - Size: {img_size}')

        mask_filename = os.path.splitext(image_filename)[0] + ".png"

        mask_path = os.path.join(train_masks, mask_filename)
        mask = Image.open(mask_path)

        mask_size = mask.size
        #     print(f'Mask: {mask_filename} - Size: {mask_size}')

        plt.subplot(rows, columns, i * 2 + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image: {image_filename}")

        plt.subplot(rows, columns, i * 2 + 1)
        plt.imshow(mask, alpha=0.5)
        plt.axis("off")
        plt.title(f"Mask: {mask_filename}")

    plt.tight_layout()
    plt.show()


# Main function
def main():
    Visualize_data()
    train_transform = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    test_val_transforms = A.Compose(
        [
            A.Resize(height=img_height, width=img_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = Unet(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader, test_loader = get_loaders(
        train_img_dir=train_images,
        train_mask_dir=train_masks,
        val_img_dir=val_images,
        val_mask_dir=val_masks,
        test_img_dir=test_images,
        test_mask_dir=test_masks,
        train_transforms=train_transform,
        val_transforms=test_val_transforms,
        test_transforms=test_val_transforms,
        batch_size=batch_size,
        num_of_workers=num_of_workers,
        pin_memory=pin_memory,
    )

    scaler = torch.amp.GradScaler()

    epoch_count = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = train_fn(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss = eval_fn(loader=val_loader, model=model, loss_fn=loss_fn)

        epoch_count.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
        )

    # Plotting the learning curve
    plt.plot(epoch_count, train_losses, label="Training Loss")
    plt.plot(epoch_count, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Evaluation on test data
    model.eval()
    with torch.no_grad():
        for i in range(3):
            data, targets = next(iter(test_loader))
            data = data.to(device=device)
            targets = targets.to(device=device)

            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

            plt.figure(figsize=(12, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(targets[0].squeeze().cpu())
            plt.title("Target")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(predictions[0].squeeze().cpu())
            plt.title("Prediction")
            plt.axis("off")

            plt.show()


main()
