import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torch import nn
import base64
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use specific origin like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config ---
DEVICE = torch.device("cpu")
IMAGE_SIZE = (512, 512)
THRESHOLD = 0.7
MODEL_PATH = "unet_epoch70.pth"  # Update if needed

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UNet Model Definition ---
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
        dec4 = self.dec4(torch.cat([dec4, self.center_crop(enc4, dec4)], dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([dec3, self.center_crop(enc3, dec3)], dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([dec2, self.center_crop(enc2, dec2)], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, self.center_crop(enc1, dec1)], dim=1))
        return torch.sigmoid(self.final(dec1))

    def center_crop(self, enc_feature, target_feature):
        _, _, h, w = target_feature.shape
        return transforms.CenterCrop([h, w])(enc_feature)

# --- Load model ---
model = UNet().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ UNet model loaded.")
except Exception as e:
    print(f"❌ Error loading UNet model: {e}")
    raise SystemExit("Failed to load model.")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# --- Endpoint ---
@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        orig_resized = img.resize(IMAGE_SIZE)
        image = transform(orig_resized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            pred_mask = output.squeeze().cpu().numpy()
            pred_bin = (pred_mask > THRESHOLD).astype(np.uint8) * 255

        # Convert to OpenCV format
        orig_np = np.array(orig_resized)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        overlay = orig_bgr.copy()

        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Encode binary mask and overlay
        _, mask_buf = cv2.imencode(".png", pred_bin)
        _, overlay_buf = cv2.imencode(".jpg", overlay)

        mask_base64 = base64.b64encode(mask_buf).decode("utf-8")
        overlay_base64 = base64.b64encode(overlay_buf).decode("utf-8")

        return {
            "mask_base64": mask_base64,
            "overlay_base64": overlay_base64,
            "contours_detected": len(contours)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")

# --- Run ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

