# dnnls_final_project

## Project Roadmap & Execution Plan
To improve the baseline multimodal model, I am following a structured three-phase approach focusing on Visual Grounding and Temporal Awareness.

### Phase 1: Grounding Module Enhancement
1. **Data Parsing:** Extracting entity-level bounding boxes from the chain_of_thought markdown tables using Regex.
2. **ROI Integration:** Implementing a region-of-interest (ROI) cropping mechanism to provide the model with local visual features alongside global frame information.
3. **Architecture Update:** Injecting ROI embeddings into the sequence predictor to align text tokens with specific image regions.

### Phase 2: Implementation of Training Objectives
1. **Contrastive Alignment (Experiment 1):** Implementing an InfoNCE Loss to minimize the distance between ROI embeddings and their corresponding text descriptions in the joint embedding space.
2. **Loss Ablation:** Comparing the performance of InfoNCE vs. MSE-based alignment to determine the most effective grounding signal.

### Phase 3: Temporal & Global Evaluation
1. **Frame-Aware Matching (Experiment 2):** Testing frame-specific ROI alignment against global context matching to evaluate improvements in temporal reasoning ("who did what when").
2. **Final Analysis:** Generating similarity heatmaps and retrieval metrics to provide evidence of improved multimodal grounding.

## Implementation Progress

### Component 1: Grounding Module

**Task:** Extraction of Grounding Information from Chain-of-Thought (CoT).
Since the `daniel3303/StoryReasoning` dataset stores grounding information (bounding boxes) within the `chain_of_thought` markdown string, I implemented a custom parser.
**Technical Specification: Coordinate System:** Through empirical testing, I verified that the Bounding Box values in the CoT (e.g., 318, 48...) do not represent absolute pixels but are scaled to a **1000x1000 unit grid**. 
- **Image Resolution:** 575x240 px (e.g.)
- **Coordinate Range:** 0-1000
- **Normalization Strategy:** All coordinates are divided by 1000.0 to obtain relative values [0, 1], ensuring compatibility regardless of the input frame resolution.

- **Parser Logic:** Uses Regular Expressions (Regex) to scan the CoT for image sections (`## Image X`) and extract coordinates from markdown tables.
- **Normalization:** Coordinates are converted from the dataset's 1000x1000 scale to a normalized [0, 1] range compatible with PyTorch/Torchvision.
- **Verification:** Successfully extracted regions for characters (e.g., 'James') and background objects to be used for the ROI-alignment.

**Status:** Phase 1 Completed. Extraction and synchronization of Grounding Information.
I have successfully closed the data loop between the raw Chain-of-Thought (CoT) text and the model's training pipeline.

#### 1. Data Parsing & Normalization
- **Regex Extraction:** Implemented a custom parser to extract entity-level bounding boxes from markdown tables within the `chain_of_thought` field.
- **Coordinate Scaling:** Verified that the dataset uses a **1000x1000 unit grid**. Developed a dynamic denormalization strategy that converts these values into absolute pixels based on the specific frame resolution (e.g., 575x240 px).
- **Persistent Mapping:** Applied the parser to the entire dataset via `.map()`, creating a structured `parsed_boxes` column for efficient access.

#### 2. ROI Integration & Pipeline Sync
- **ROI Extraction:** Integrated a `torchvision` cropping mechanism within the `__getitem__` method to provide the model with local visual features (crops of 60x125 px).
- **DataLoader Update:** Synchronized the `DataLoader` to output a 10-element batch tuple, including the new `context_rois_tensor`.
- **Loop Synchronization:** Updated both the **Main Training Loop** and the **Validation Routine** to handle the expanded data structure.
- **Verification:** Successfully verified the integrity of the visual crops through sanity check visualizations, ensuring the model "sees" the correct entities (e.g., characters or objects) before starting the training.