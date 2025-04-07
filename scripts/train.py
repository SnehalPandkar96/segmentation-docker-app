import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image

# --- Config ---
IMAGE_SIZE = (512, 512)  # ⬅️ Ensures clean skip connections
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/home/snehal/Documents/threev/dataset"
RESULT_DIR = os.path.join(DATASET_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.valid_pairs = []

        for img_name in os.listdir(image_dir):
            base_name, _ = os.path.splitext(img_name)
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, base_name + ".png")

            if os.path.exists(mask_path) and cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) is not None:
                self.valid_pairs.append((img_path, mask_path))

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

# --- Center Crop helper (for U-Net skip connections) ---
def center_crop(tensor, target_tensor):
    _, _, h, w = target_tensor.shape
    return transforms.CenterCrop([h, w])(tensor)

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        self.center = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        center = self.center(self.pool4(enc4))

        dec4 = self.up4(center)
        dec4 = self.dec4(torch.cat([dec4, center_crop(enc4, dec4)], dim=1))

        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, center_crop(enc3, dec3)], dim=1))

        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, center_crop(enc2, dec2)], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, center_crop(enc1, dec1)], dim=1))

        return torch.sigmoid(self.final(dec1))

# --- Load Data ---
train_dataset = SegmentationDataset(
    os.path.join(DATASET_DIR, "train/images"),
    os.path.join(DATASET_DIR, "train/masks"),
)
val_dataset = SegmentationDataset(
    os.path.join(DATASET_DIR, "val/images"),
    os.path.join(DATASET_DIR, "val/masks"),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- Model, Loss, Optimizer ---
model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
        print(f"📉 Val Loss: {val_loss / len(val_loader):.4f}")

    # --- Save Model ---
    model_path = os.path.join(RESULT_DIR, f"unet_epoch{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved at: {model_path}")

print("🏁 Training complete!")


