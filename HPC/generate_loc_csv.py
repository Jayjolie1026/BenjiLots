from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import os
import json
import pandas as pd

PARKING_CLASS_ID = 1

feature_extractor = SegformerImageProcessor(do_resize=True, size=(256, 256))

class EvalDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.json_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".json")
        ])
        # Grab image files only (modify extension if needed)
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".tif")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        json_name = self.json_files[idx]
        json_path = os.path.join(self.root_dir, json_name)
        
        with open(json_path, "r") as f:
            meta = json.load(f)

        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        encoding = self.feature_extractor(image, return_tensors="pt")
        
        return {
            "pixel_values": encoding["pixel_values"],
            "image_name": meta["image"],
            "crs": meta["crs"],
            "bbox_ll_epsg4326": meta["bbox_ll_epsg4326"],
            "bbox_3857_m": meta["bbox_3857_m"],
            "width_px": meta["width_px"],
            "image": image,
            "height_px": meta["height_px"],
            "pixel_size_m": meta["pixel_size_m"],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "image_name": [x["image_name"] for x in batch],
        "crs": [x["crs"] for x in batch],
        "bbox_ll_epsg4326": [x["bbox_ll_epsg4326"] for x in batch],
        "bbox_3857_m": [x["bbox_3857_m"] for x in batch],
        "width_px": [x["width_px"] for x in batch],
        "height_px": [x["height_px"] for x in batch],
        "pixel_size_m": [x["pixel_size_m"] for x in batch],
        "image": [x["image"] for x in batch],
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = "downtown_nashville_2023"

test_dataset = EvalDataset(root_dir, feature_extractor)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


results = []
model = SegformerForSemanticSegmentation.from_pretrained(
    "./segformer-parkseg12k_2"
)
model.to(device)
model.eval()


with torch.no_grad():
    output_dir = "nash_data"
    os.makedirs(output_dir, exist_ok=True)
    for example in test_dataset:
        # Get the input tensor and move to device
        pixel_values = example["pixel_values"].to(device)  # (1, C, H, W) if already unsqueezed
        image = example["image"]

        # Forward pass
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, num_classes, H_out, W_out)

        # Upsample to original image size
        upsample = nn.Upsample(size=(example["height_px"], example["width_px"]),
                               mode="bilinear", align_corners=False)
        logits_up = upsample(logits)  # (1, num_classes, H, W)

        # Get predicted class per pixel
        pred_class = torch.argmax(logits_up[0], dim=0).cpu().numpy().astype(np.uint8)  # (H, W)

        # Create parking mask
        parking_mask = (pred_class == PARKING_CLASS_ID).astype(np.uint8) * 255  # (H, W)

        # Overlay mask on original image
        image_dir = os.path.join(output_dir, example['image_name'])
        os.makedirs(image_dir, exist_ok=True)

        mask_l = Image.fromarray(parking_mask).convert("L")
        mask_file = os.path.join(image_dir, "mask.png")
        mask_l.save(mask_file)
        print(f"Saved raw mask -> {mask_file}")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            image = image.convert("RGB")

        # Save original image
        image_file = os.path.join(image_dir, "original.png")
        image.save(image_file)
        print(f"Saved original image -> {image_file}")
        red = Image.new("RGB", image.size, (255, 0, 0))
        overlay = Image.composite(red, image, mask_l)
        blended = Image.blend(image, overlay, alpha=0.3)

        # Save overlay
        overlay_file = os.path.join(image_dir, "overlay.png")
        blended.save(overlay_file)
        print(f"Saved blended overlay -> {overlay_file}")
        # Append results
        results.append({
            "image_name": example["image_name"],
            "crs": example["crs"],
            "bbox_ll_epsg4326": example["bbox_ll_epsg4326"],
            "bbox_3857_m": example["bbox_3857_m"],
            "width_px": example["width_px"],
            "height_px": example["height_px"],
            "pixel_size_m": example["pixel_size_m"],
            "parking_mask_flat": parking_mask.flatten().tolist()
        })

#df = pd.DataFrame(results)
#df.to_csv("inference_parking_masks_cook.csv", index=False)
print("Done")
