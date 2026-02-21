import torch
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import PIL
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    "./segformer-parkseg12k"
)
model.to(device)
model.eval()

all_preds = []
all_labels = []


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


all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

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
