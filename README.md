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

**Status: Phase 2 & 3 Completed (Final Evaluation)**
The experimental phase is concluded. Both the loss ablation (Experiment 1) and the temporal awareness study (Experiment 2) have been successfully executed and documented.

- **Current Activity:** Running extended 100-epoch training sessions to evaluate long-term convergence of the MSE-based grounding signal.
- **Data Integrity:** All training logs are being systematically exported to `.csv` files to ensure reproducibility and to prevent data loss from session disconnects.

### Preliminary Results & Observations (Experiment 1)
Initial controlled runs of 20 epochs have provided the following insights:

| Metric | InfoNCE (Contrastive) | MSE (Regression) | Observation |
| :--- | :--- | :--- | :--- |
| **Convergence** | Slower / Static | **Rapid / Dynamic** | MSE shows a stronger initial gradient for ROI alignment. |
| **Grounding MSE** | Near zero (Low variance) | **Decreasing (High activity)** | The model actively learns spatial features under MSE. |
| **Text Loss** | ~2.84 | **~3.15** | Both objectives support context-aware text generation. |

**Hypothesis Refinement:** Contrary to the initial hypothesis, MSE appears more effective for the current batch size constraints. InfoNCE likely requires larger batches (more negatives) or longer training duration to restructure the latent space effectively.

### Final Results & Findings: Experiment 2 (Temporal Awareness)
I conducted a **Temporal Ablation Study** comparing frame-specific grounding versus global context matching.

| Configuration | Grounding MSE | Text Loss Stability | Observation |
| :--- | :--- | :--- | :--- |
| **Frame-Aware (Temporal)** | Higher (~0.10) | **Stable / Low** | Superior story prediction through precise timing. |
| **Global Matching (Baseline)** | **Lower (~0.02)** | High Variance | "Shortcut" learning; model fails to link actions to time. |

**Key Insights:**
- **Temporal Grounding Success:** Matching ROI $t$ to Text $t$ (Frame-Aware) forces the model to learn "who did what when," leading to more stable sequence predictions.
- **Visual Synthesis Observations:** It was observed that the Image L1-Loss leads to "regression to the mean" (grey images). This confirms that semantic grounding (text-loss) is the more reliable metric for this architecture's reasoning capabilities.
- **Data Preservation:** Training histories are preserved as `history_experiment2_global.csv` and `history_mse_experiment_recovered.csv`.

#### Evaluation & Diagnostic Tools
To evaluate the success of the grounding module, I implemented two diagnostic tools:
1. **Temporal Similarity Heatmaps:** Calculates the cosine similarity between ROI embeddings and text embeddings across the sequence to verify temporal alignment.
2. **Visual Sanity Checks:** A dedicated pipeline component that visualizes the original frame alongside the extracted ROI to ensure coordinate-to-pixel transformation integrity.