import json
import torch
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import PIL
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


feature_extractor = SegformerImageProcessor(do_resize=True, size=(256, 256))

def mask_to_tensor(mask, size=(256,256)): # Ensure PIL Image 
    if not isinstance(mask, PIL.Image.Image): 
        mask = PIL.Image.fromarray(mask) 
        # Resize 
        mask = mask.resize(size, resample=PIL.Image.NEAREST) 
        # Convert to tensor, long type 
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long) # shape [H,W] 
        return mask_tensor 
    
def collate_fn(batch):
    images = []
    masks = []

    for example in batch:
        # --- Image ---
        image = example["rgb"].convert("RGB")
        encoded = feature_extractor(image, return_tensors="pt")
        images.append(encoded["pixel_values"].squeeze(0))

        # --- Mask ---
        mask = example["mask"].convert("L")
        mask = mask.resize((256,256), resample=PIL.Image.NEAREST)
        mask_array = np.array(mask)
        mask_array = (mask_array > 0).astype(np.uint8)  # binary
        masks.append(torch.tensor(mask_array, dtype=torch.long))

    return {
        "pixel_values": torch.stack(images),
        "labels": torch.stack(masks)
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("UTEL-UIUC/parkseg12k")
print(ds["test"][0])
#ds["test"].set_format(type="torch", columns=["pixel_values", "labels"])
test_loader = DataLoader(
    ds["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "./segformer-parkseg12k_2"
)
model.to(device)
model.eval()

all_preds = []
all_labels = []
lot_stats_per_image = []

min_distance=20

with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)        # [B,H,W]

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits,
            size=labels.shape[-2:],  # (256,256)
            mode="bilinear",
            align_corners=False
        )

        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        
        preds_cpu = preds.cpu().numpy()  # [B,H,W]
        batch_stats = []
        all_stats = []

        for pred_mask in preds_cpu:
            lot_mask = (pred_mask == 1).astype(np.uint8)
        
            num_lots, lot_labels = cv2.connectedComponents(lot_mask)
            lots = [lot_labels == i for i in range(1, num_lots)]
            
            overlay_img = cv2.cvtColor(lot_mask, cv2.COLOR_GRAY2BGR)

            spots_per_lot = []
            spots_coords_per_lot = []

            for lot in lots:
                lot_uint8 = lot.astype(np.uint8) * 255
                # distance transform
                blurred = cv2.GaussianBlur(lot_uint8, (5,5), 0)
                edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                                minLineLength=10, maxLineGap=5)
                if lines is None or len(lines) < 50:
                    # fallback: use area formula or distance transform
                    distance = ndimage.distance_transform_edt(lot)
                    coords = peak_local_max(distance, min_distance=min_distance)
                    spot_count = len(coords)
                else:
                    # crude estimate: one spot per 2 lines
                    spot_count = len(lines) // 2
                    coords = []

                spots_per_lot.append(spot_count)
                spots_coords_per_lot.append(coords)

                contours, _ = cv2.findContours(lot_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_img, contours, -1, (0,255,0), 2)
                for y, x in coords:
                    cv2.circle(overlay_img, (x,y), 3, (0,0,255), -1)
                
                stats = {
                    "num_lots": len(lots),
                    "spots_per_lot": spots_per_lot,
                    "total_spots": sum(spots_per_lot),
                    "spots_coords_per_lot": [c.tolist() for c in spots_coords_per_lot]
                }
                
                all_stats.append({
                    "stats": stats,
                    "pred_mask": pred_mask.tolist()  # store as list for JSON compatibility
                })

            batch_stats.append({
                "num_lots": len(lots),
                "spots_per_lot": spots_per_lot,
                "total_spots": sum(spots_per_lot)
            })

            lot_stats_per_image.extend(batch_stats)

all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

output_dir = "parking_overlays"
os.makedirs(output_dir, exist_ok=True)

for i, item in enumerate(all_stats[:10]):
    stats = item["stats"]
    pred_mask = np.array(item["pred_mask"], dtype=np.uint8)  # convert list back to np.array

    # 1️⃣ Save mask as PNG (1 = parking lot)
    mask_file = os.path.join(output_dir, f"image_{i}_pred_mask.png")
    cv2.imwrite(mask_file, pred_mask * 255)  # scale to 0-255
    print(f"Saved predicted mask for Image {i} -> {mask_file}")

    # 2️⃣ Save stats separately as JSON
    stats_file = os.path.join(output_dir, f"image_{i}_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats for Image {i} -> {stats_file}")

    # 3️⃣ Optional: preview with matplotlib (terminal-friendly)
    plt.figure(figsize=(5,5))
    plt.title(f"Image {i} Pred Mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    plt.show()

def compute_iou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            continue  # class not present
        ious.append(intersection / union)

    return ious, sum(ious) / len(ious)


num_classes = 2  # adjust according to your dataset

ious, mean_iou = compute_iou(all_preds, all_labels, num_classes)

print(f"Per-class IoU: {ious}")
print(f"Mean IoU: {mean_iou}")

for i, stats in enumerate(lot_stats_per_image[:50]):  # show first 5 images
    print(f"Image {i}: {stats}")
