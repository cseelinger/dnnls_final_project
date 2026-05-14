# 1.1 Loading and parsing data
# [6]

import re

def extract_boxes_from_cot(cot_text):
    """
    Extracts all bounding boxes per frame from the chain-of-thought text.
    Output: Dictionary {frame_idx: [[y1, x1, y2, x2], ...]}
    """
    # 1. Subdivide text into section per picture
    image_sections = re.split(r'## Image \d+', cot_text)[1:]

    all_frame_boxes = {}

    for i, section in enumerate(image_sections):
        # 2. Search for sequence of numbers in the last column of the table | 318,48,345,135 |
        # Searching for numbers, seperated by commas, within pipes
        box_matches = re.findall(r'\|\s*([\d,]+)\s*\|', section)

        parsed_boxes = []
        for match in box_matches:
            coords = [float(c) for c in match.split(',')]
            if len(coords) == 4:
                # Normalize from 0-1000 to 0.0-1.0
                coords = [c / 1000.0 for c in coords]
                parsed_boxes.append(coords)

        all_frame_boxes[i] = parsed_boxes

    return all_frame_boxes

# [7]
def add_parsed_boxes(example):
    # Uses already tested function
    boxes_dict = extract_boxes_from_cot(example['chain_of_thought'])

    # Create a list of Box-Lists (one list per frame)
    # Example: [ [[y1,x1,y2,x2], [y1,x1,y2,x2]], [], [[...]] ]
    parsed_boxes = []
    for i in range(example['frame_count']):
        parsed_boxes.append(boxes_dict.get(i, []))

    example['parsed_boxes'] = parsed_boxes
    return example

# Apply mapping
print("Apply mapping the boxes to the datasets...")
train_dataset = train_dataset.map(add_parsed_boxes)
test_dataset = test_dataset.map(add_parsed_boxes)
print("Finished! Example 'parsed_boxes':", train_dataset[0]['parsed_boxes'][0])


# [11] SequencePredictionDataset

# @title Main dataset
"""
Defines the `SequencePredictionDataset` class, which is the core data provider for the main task.
1. `__getitem__`:
   - Loads 5 frames (4 context + 1 target).
   - Parses text descriptions and optionally appends CoT text.
   - Extracts bounding box crops (ROIs) for grounding tasks if CoT data is available.
   - Returns a tuple containing: sequence images, descriptions, target image, target text, ROI crops, and validity flags.
"""
class SequencePredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, K: int = 4, max_len: int = 120, image_hw=(60, 125)):
        super(SequencePredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw

        self.transform = transforms.Compose([
          transforms.Resize(image_hw),
          transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
        as a target.

        Returns:
          frames:        [K, C, H, W]
          descriptions:  [K, T]
          image_target:  [C, H, W]
          target_ids:    [1, T]
          roi1, roi2:    [C, H, W] (cropped from CoT bboxes, if available)
          roi_valid:     0/1
          roi_frame:     frame index for roi1 (0..K-1) if available else -1
          ent_id:        string id for the ROI entity (empty if none)
        """
        # 1. Get data of the original dataset (incl. parsed_boxes)
        item = self.dataset[idx]
        frames = item["images"]
        cot = item.get("chain_of_thought", "")

        # 2. parse text explanation (important: definition first!)
        image_attributes = parse_gdi_text(item["story"])
        cot_frames = parse_cot_grounding(cot) # keep template-function

        frame_tensors = []
        description_list = []

        # 3. Loop for Kontext-Frames (pictures and text)
        for frame_idx in range(self.K):
            # Bild verarbeiten
            image = FT.equalize(frames[frame_idx])
            input_frame = self.transform(image)
            frame_tensors.append(input_frame)

            # process text
            description = image_attributes[frame_idx]["description"]
            if USE_COT_TEXT:
                cot_txt = extract_cot_text_for_frame(cot, frame_idx)
                if cot_txt:
                    description = description + " [COT] " + cot_txt

            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).input_ids.squeeze(0)
            description_list.append(input_ids)

        # 4. process Target-Frame (future)
        image_target = FT.equalize(frames[self.K])
        image_target = self.transform(image_target)

        target_desc = image_attributes[self.K]["description"]
        target_ids = self.tokenizer(
            target_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).input_ids

        # 5. --- GROUNDING MODULE: Extract context ROIs ---
        all_context_rois = []
        for f_idx in range(self.K):
            frame_boxes = item.get('parsed_boxes', [[] for _ in range(self.K)])[f_idx]

            if len(frame_boxes) > 0:
                ny1, nx1, ny2, nx2 = frame_boxes[0] # normalized coordinates
                W, H = frames[f_idx].size
                # calculate pixel coordinates for PIL (x1, y1, x2, y2)
                pixel_bbox = [nx1 * W, ny1 * H, nx2 * W, ny2 * H]
                roi = crop_and_resize(frames[f_idx], pixel_bbox, out_hw=self.image_hw)
            else:
                roi = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
            all_context_rois.append(roi)

        context_rois_tensor = torch.stack(all_context_rois)

        # 6. --- Template ROI Pair (for ReID) ---
        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2, ent_id = pair
            if (0 <= f1 < self.K) and (0 <= f2 < self.K):
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        # 7. convert everything in tensors
        sequence_tensor = torch.stack(frame_tensors)
        description_tensor = torch.stack(description_list)

        # 8. return all 10 elements
        return (
            sequence_tensor,
            description_tensor,
            image_target,
            target_ids,
            context_rois_tensor, # <--- DEIN NEUES FEATURE (Pos 5)
            roi1, roi2, roi_valid, roi_frame, ent_id
        )

# -----------------------------------------
# 1.3 Creating and testing our dataset objects and loaders
# [17]

# @title Sanity check cell (Final Fix)
import numpy as np
import matplotlib.pyplot as plt

# 1. Test with single example (Dataset)
(frames, descriptions, image_target, text_target,
 context_rois,
 roi1, roi2, roi_valid, roi_frame, ent_id) = sp_train_dataset[np.random.randint(0,400)]

print("Dataset-Test: Context ROIs shape:", context_rois.shape)

# 2. Test with batch (DataLoader)
(frames, descriptions, image_target, text_target,
 context_rois,
 roi1, roi2, roi_valid, roi_frame, ent_id) = next(iter(train_dataloader))

print("Batch-Test: Frames shape:", frames.shape)          # [Batch, 4, 3, 60, 125]
print("Batch-Test: Context ROIs shape:", context_rois.shape) # [Batch, 4, 3, 60, 125]

# --- Visualize ---
# Show the first target picture of the batch
figure, ax = plt.subplots(1, 1)
show_image(ax, image_target[0])
plt.title("Erstes Target-Bild aus dem Batch")
plt.show()

# Show the first ROI of the first frame of the batch
figure, ax = plt.subplots(1, 1)
show_image(ax, context_rois[0][0])
plt.title("Erste extrahierte ROI (Grounding Check)")
plt.show()

# -----------------------------
# 2.3 The Main Architecture
# [23]
class SequencePredictor(nn.Module):
    # ...

    def forward(self, image_seq, text_seq, target_seq, context_rois=None):
        # ...

        # --- ROI ENCODING ---
        z_roi_seq = None
        if context_rois is not None:
            # Same trick like above: Flatten for Encoder
            rois_flat = context_rois.view(batch_size * seq_len, C, H, W)
            z_roi_flat = self.image_encoder(rois_flat) # use same encoder!
            z_roi_seq = z_roi_flat.view(batch_size, seq_len, -1) # [b, s, latent]
        # ...

        # --- EXPERIMENT 2 LOGIC ---
        if getattr(self, 'use_global_matching', False):
            # investigate the text for all 4 frames
            # z_t_seq has shape: [batch, 4, latent]
            global_z_t = z_t_seq.mean(dim=1, keepdim=True) # result: [batch, 1, latent]

            # "copy" the average back to all 4 timeslots
            # the model now sees for every frame just the "global theme"
            z_t_seq_final = global_z_t.expand_as(z_t_seq)
        else:
            # Standard: Frame-Aware (precise assign)
            z_t_seq_final = z_t_seq

        # Normalization
        if z_roi_seq is not None:
            # L2-Normalization for clean Cosine Similarity in the heatmap
            z_roi_seq = F.normalize(z_roi_seq, p=2, dim=-1)
            z_t_seq_final = F.normalize(z_t_seq_final, p=2, dim=-1)

        # return z_t_seq_final instead of statt z_t_seq
        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_seq, z_t_seq_final, z_roi_seq

# ----------------------------
# 3.2 Training Loops
# [30]

# @title Training loop for the sequence predictor
"""
The main training loop:
1. Iterates over epochs and batches.
2. Performs the forward pass to get predictions and latent representations.
3. Computes the **Base Losses**: Image L1, Context MSE, Text CrossEntropy.
4. Computes **CoT Grounding Losses** (if data is valid):
   - `loss_reid`: Visual consistency for re-identified entities.
   - `loss_ground_mse`: Embedding alignment between ROI and text.
   - `loss_contrast`: Contrastive loss for ROI-Text alignment.
   - `loss_entity_pool`: Consistency within the batch for the same entity.
5. Backpropagates total loss and updates weights.
6. Runs the validation visualization at the end of each epoch.
"""

# Instantiate the model, define loss and optimizer

# --- CoT-loss weights ---
LAMBDA_REID = 0.10            # pulls same-entity ROIs together (student idea)
LAMBDA_GROUND_MSE = 0.10      # Option 2: frame-aware ROI↔text MSE grounding
LAMBDA_CONTRAST = 0.10        # Option 1: contrastive ROI↔text grounding (InfoNCE)
LAMBDA_ENTITY_POOL = 0.05     # Option 3: within-batch entity pooling loss

sequence_predictor.train()
losses = []

# Lists for the documentation
history = {
    'total': [],
    'image': [],
    'text': [],
    'ground_mse': [],
    'contrast': []
}

for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for (frames, descriptions, image_target, text_target,
         context_rois,
         roi1, roi2, roi_valid, roi_frame, ent_id) in train_dataloader:

        # Send images and tokens to the GPU
        descriptions = descriptions.to(device)
        frames = frames.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)
        context_rois = context_rois.to(device)

        roi1, roi2 = roi1.to(device), roi2.to(device)
        roi_valid, roi_frame = roi_valid.to(device), roi_frame.to(device)

        optimizer.zero_grad()

        # 2. Call model with context_rois
        # 8 instead of 7 returns (z_roi_seq at the end)
        (pred_image_content, pred_image_context, predicted_text_logits_k,
         _, _, z_v_seq, z_t_seq, z_roi_seq) = sequence_predictor(
             frames, descriptions, text_target, context_rois=context_rois
         )

        # -------------------------
        # Base losses (Image & Text)
        # -------------------------
        loss_im = criterion_images(pred_image_content, image_target)
        mu_global = frames.mean(dim=[0, 1]).unsqueeze(0).expand_as(pred_image_context)
        loss_context = criterion_ctx(pred_image_context, mu_global)

        prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
        target_flat = text_target.squeeze(1)[:, 1:].reshape(-1)
        loss_text = criterion_text(prediction_flat, target_flat)

        # -------------------------
        # CoT Grounding Losses (Optimiert)
        # -------------------------
        loss_reid = torch.tensor(0.0, device=device)
        loss_ground_mse = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)

        # A) NEW SEQUENCE-LOSS (Experiment 1)
        # Compare z_roi_seq [B, K, D] with z_t_seq [B, K, D]
        if z_roi_seq is not None:
            if USE_GLOBAL_MATCHING:
                # GLOBAL: Mean the text over the time
                global_z_t = z_t_seq.mean(dim=1, keepdim=True).expand_as(z_t_seq)
                loss_ground_mse = F.mse_loss(z_roi_seq, global_z_t)
            else:
                # FRAME-AWARE: Standard (ROI_t matches Text_t)
                loss_ground_mse = F.mse_loss(z_roi_seq, z_t_seq)

            # InfoNCE set to 0 (for clear MSE-comparison)
            loss_contrast = torch.tensor(0.0, device=device)

            # Contrastive Grounding (InfoNCE)
            if USE_CONTRASTIVE_ROI:
                # flattening batch and time for Contrastive Learning
                z_img = F.normalize(z_roi_seq.reshape(-1, z_roi_seq.size(-1)), dim=-1)
                z_txt = F.normalize(z_t_seq.reshape(-1, z_t_seq.size(-1)), dim=-1)

                logits = (z_img @ z_txt.t()) / CONTRASTIVE_TAU
                labels = torch.arange(logits.size(0), device=device)
                loss_contrast = F.cross_entropy(logits, labels)

        # -------------------------
        # Total Loss & Update
        # -------------------------
        loss = loss_im + loss_context + loss_text
        loss += LAMBDA_REID * loss_reid
        loss += LAMBDA_GROUND_MSE * loss_ground_mse
        loss += LAMBDA_CONTRAST * loss_contrast

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Logging & Validation
    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}] Loss: {epoch_loss:.4f} (im={loss_im.item():.3f}, txt={loss_text.item():.3f}, g_mse={loss_ground_mse.item():.3f})")

    # ... (at the end of one epoch,  after the print)
    history['total'].append(epoch_loss)
    history['image'].append(loss_im.item())
    history['text'].append(loss_text.item())
    history['ground_mse'].append(loss_ground_mse.item())
    history['contrast'].append(loss_contrast.item())

    validation(sequence_predictor, val_dataloader)
    sequence_predictor.train()

# [31]
def plot_training_results(history):
    epochs = range(1, len(history['total']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Basis Loss (Image & Text)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['image'], label='Image L1 Loss')
    plt.plot(epochs, history['text'], label='Text CrossEntropy')
    plt.title('Base Model Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Grounding Loss (Experiment 1)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['ground_mse'], color='orange', label='Grounding MSE')
    if any(v > 0 for v in history['contrast']):
        plt.plot(epochs, history['contrast'], color='green', label='InfoNCE Loss')
    plt.title('Grounding Module Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_mse2.png') # Save picture for doku
    plt.show()

# Call after training:
plot_training_results(history)

# [32]
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_final_heatmap(model, dataloader, device):
    model.eval()
    # Get a Batch from the validation data
    frames, descriptions, _, text_target, context_rois, _, _, _, _, _ = next(iter(dataloader))

    with torch.no_grad():
        # Forward pass, for keeping the Embeddings
        _, _, _, _, _, _, z_t_seq, z_roi_seq = model(
            frames.to(device),
            descriptions.to(device),
            text_target.to(device),
            context_rois=context_rois.to(device)
        )

        # Analyze the first example in batch
        # Normalization for the cosinus similarity
        roi_emb = F.normalize(z_roi_seq[0], dim=-1) # [4, latent]
        txt_emb = F.normalize(z_t_seq[0], dim=-1)   # [4, latent]

        # Calculate the similarity matrix (Frames x Frames)
        sim_matrix = torch.mm(roi_emb, txt_emb.t()).cpu().numpy()

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')

    plt.xticks(range(4), [f"Text T{i+1}" for i in range(4)])
    plt.yticks(range(4), [f"ROI T{i+1}" for i in range(4)])

    plt.title("Visual Grounding: Temporal Alignment Matrix")
    plt.xlabel("Textual Context (Time)")
    plt.ylabel("Visual Regions (ROIs)")

    # Mark the diagonal (because of Frame-Aware Training)
    for i in range(4):
        plt.gca().add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=3))

    plt.tight_layout()
    plt.savefig('similarity_heatmap_mse5.png')
    plt.show()

# Call
generate_final_heatmap(sequence_predictor, val_dataloader, device)

#[33]
import pandas as pd
# Create a dataframe of the history
df_infonce = pd.DataFrame(history)
# Save as csv
df_infonce.to_csv('history_experiment1_mse2.csv', index=False)

print("Daten erfolgreich als 'history_experiment1_mse2.csv' gespeichert!")

#[35]
# Quick Sanity Plot
sample = sp_train_dataset[np.random.randint(0, 100)]
fig, ax = plt.subplots(2, 4, figsize=(15, 6))
for i in range(4):
    show_image(ax[0, i], sample[0][i]) # Original Frame
    show_image(ax[1, i], sample[4][i]) # my ROI
plt.show()

#[36]
# --- DEBUG SANITY CHECK ---
import matplotlib.pyplot as plt
import numpy as np

# 1. get random sample of the dataset
idx = np.random.randint(0, len(sp_train_dataset))
item = sp_train_dataset.dataset[idx] # go to Original-Item

frames = item["images"]
all_boxes = item.get('parsed_boxes', [[]]*4)

fig, ax = plt.subplots(2, 4, figsize=(20, 8))

for f_idx in range(4):
    # show Original Frame
    img = frames[f_idx]
    ax[0, f_idx].imshow(img)
    ax[0, f_idx].set_title(f"Original Frame {f_idx}")
    ax[0, f_idx].axis('off')

    frame_boxes = all_boxes[f_idx]

    if len(frame_boxes) > 0:
        # check the values
        coords = frame_boxes[0]
        print(f"Frame {f_idx} Coords: {coords}")

        # if values > 1: (1000 Grid), scale
        if any(c > 1.1 for c in coords):
            coords = [c/1000.0 for c in coords]

        # switch v1, v2 to ny1, nx1
        ny1, nx1, ny2, nx2 = coords

        W, H = img.size
        # PIL needs (left, top, right, bottom) -> (x1, y1, x2, y2)
        pixel_bbox = [nx1 * W, ny1 * H, nx2 * W, ny2 * H]

        roi = img.crop((pixel_bbox[0], pixel_bbox[1], pixel_bbox[2], pixel_bbox[3]))
        ax[1, f_idx].imshow(roi)
        ax[1, f_idx].set_title(f"Extracted ROI {f_idx}")
    else:
        ax[1, f_idx].text(0.5, 0.5, "Keine Box", ha='center')

    ax[1, f_idx].axis('off')

plt.tight_layout()
plt.show()


# ---------------
# Experiments
#----------------

# MSE-Run
USE_GLOBAL_MATCHING = False
USE_CONTRASTIVE_ROI = False

LAMBDA_GROUND_MSE = 0.10
LAMBDA_CONTRAST = 0.0
LAMBDA_REID = 0.0
LAMBDA_ENTITY_POOL = 0.0

# InfoNCE-Run
USE_GLOBAL_MATCHING = False
USE_CONTRASTIVE_ROI = True

LAMBDA_GROUND_MSE = 0.0
LAMBDA_CONTRAST = 0.10
LAMBDA_REID = 0.0
LAMBDA_ENTITY_POOL = 0.0

# Global Matching Run
USE_GLOBAL_MATCHING = True
USE_CONTRASTIVE_ROI = False

LAMBDA_GROUND_MSE = 0.10
LAMBDA_CONTRAST = 0.0
LAMBDA_REID = 0.0
LAMBDA_ENTITY_POOL = 0.0
