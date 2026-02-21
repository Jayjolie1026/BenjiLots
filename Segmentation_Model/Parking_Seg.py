from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm
import PIL
import numpy as np
import torch.nn.functional as F

torch.cuda.empty_cache()

feature_extractor = SegformerImageProcessor(do_resize=True, size=(256, 256))
mask_transform = Compose([
    Resize((256, 256)),  # Resize expects PIL Image
    ToTensor(),           # Converts to float tensor [0,1], shape [1,H,W]
])

def mask_to_tensor(mask, size=(256,256)):
    # Ensure PIL Image
    if not isinstance(mask, PIL.Image.Image):
        mask = PIL.Image.fromarray(mask)

    # Resize
    mask = mask.resize(size, resample=PIL.Image.NEAREST)

    # Convert to tensor, long type
    mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)  # shape [H,W]

    return mask_tensor

def preprocess(example):
    # --- Image ---
    image = example["rgb"].convert("RGB")
    image_dict = feature_extractor(image, return_tensors="pt")
    image_tensor = image_dict["pixel_values"]

    # Ensure it's a tensor, not a list
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)
    image_tensor = image_tensor.squeeze(0)  # [3,H,W]

    # --- Mask ---
    mask = example["mask"].convert("L")
    mask_tensor = mask_to_tensor(mask, size=(256,256))  # shape [H,W]
    
    return {"pixel_values": image_tensor, "labels": mask_tensor}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("UTEL-UIUC/parkseg12k")
    
    print(ds['train'][0])

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    ds.set_format(type="torch", columns=["pixel_values", "labels"]) 

    train_loader = DataLoader(ds["train"], batch_size=16, shuffle=True)

    num_classes = 2

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 10
    print("Begin Training:")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:    
            optimizer.zero_grad()
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs).logits
            labels_resized = F.interpolate(
                labels.unsqueeze(1).float(),      # [B,1,H,W] as float
                size=outputs.shape[2:],           # [H_out, W_out]
                mode="nearest"                     # nearest for masks
            ).squeeze(1).long()
            labels_resized = labels_resized.clamp(0, outputs.shape[1]-1)
            loss = criterion(outputs, labels_resized)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

    
    model.save_pretrained("./segformer-parkseg12k_3")
    print("Model saved to ./segformer-parkseg12k_3")

if __name__ == "__main__":
    main()


