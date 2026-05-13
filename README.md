# dnnls_final_project

## Project Roadmap & Execution Plan
To extend the baseline multimodal model, I followed a structured approach focusing on visual grounding and temporal ROI-text alignment.

### Phase 1: Grounding Module Enhancement
1. **Data Parsing:** Extracting entity-level bounding boxes from the chain_of_thought markdown tables using Regex.
2. **ROI Integration:** Implementing a region-of-interest (ROI) cropping mechanism to provide the model with local visual features alongside global frame information.
3. **Architecture Update:** Injecting ROI embeddings into the sequence predictor to align text tokens with specific image regions.

### Phase 2: Implementation of Training Objectives
1. **Contrastive Alignment (Experiment 1):** Implementing an InfoNCE Loss to minimize the distance between ROI embeddings and their corresponding text descriptions in the joint embedding space.
2. **Loss Ablation:** Comparing the performance of InfoNCE vs. MSE-based alignment to determine the most effective grounding signal.

### Phase 3: Temporal & Global Evaluation
1. **Frame-Aware Matching (Experiment 2):** Testing frame-specific ROI alignment against global context matching to evaluate whether the model learns stronger temporal ROI-text correspondence.
2. **Final Analysis:** Generating similarity heatmaps and diagnostic metrics to analyse whether the grounding module produces frame-dependent alignment patterns.

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
- **Verification:** Successfully extracted local visual regions for characters and objects. The extracted ROIs were used as additional local visual context for ROI-text alignment.

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
- **Verification:** Visual sanity checks were used to inspect the extracted crops. The crops provide meaningful local visual context, although the annotations remain noisy and do not always perfectly center on the intended character or object.

**Project Status:**
- **Phase 1:** Completed. Chain-of-thought bounding boxes were parsed and converted into structured ROI annotations.
- **Phase 2:** Completed. ROI crops were integrated into the data pipeline and encoded as local visual features.
- **Phase 3:** Completed. MSE-based and InfoNCE-based ROI-text alignment objectives were compared using training curves and temporal similarity heatmaps.
- **Final Evaluation:** The results show that MSE can produce low grounding losses without clear temporal discrimination, while InfoNCE produces a more differentiated ROI-text similarity structure. The improvement is partial rather than perfect and remains limited by noisy ROI annotations and the small-scale experimental setup.

### Results & Observations: Experiment 1 — MSE vs. InfoNCE Alignment

Two 20-epoch runs were used to compare regression-based and contrastive ROI-text alignment.

| Objective | Loss Behaviour | Heatmap Pattern | Interpretation |
| :--- | :--- | :--- | :--- |
| **MSE Alignment** | Low and decreasing grounding MSE | Strong vertical patterns | Low regression loss does not necessarily imply frame-specific temporal alignment. The model appears to learn averaged or shortcut-like embedding correspondences. |
| **InfoNCE Alignment** | Higher and more variable contrastive loss | More differentiated similarity structure with partial diagonal behaviour | Contrastive learning provides a stronger discriminative signal by comparing correct ROI-text pairs against incorrect temporal pairs. |

**Key Observation:**  
The MSE objective performs well numerically, but the similarity heatmap reveals limited temporal discrimination. InfoNCE produces a less collapsed similarity structure, suggesting that contrastive alignment is more suitable for encouraging frame-aware ROI-text correspondence.

### Additional Findings: Experiment 2 — Frame-Aware vs. Global Matching
I conducted a temporal ablation study comparing frame-specific ROI-text matching with global context matching.

| Configuration | Grounding Behaviour | Observation |
| :--- | :--- | :--- |
| **Frame-Aware Matching** | More difficult optimization | Encourages direct ROI-text correspondence at the same time step. |
| **Global Matching** | Lower grounding error possible | Can lead to shortcut-like behaviour because each ROI is aligned with an averaged text context rather than a specific time step. |

**Key Insights:**
- **Frame-aware alignment:** Matching ROI $t$ to Text $t$ provides a stronger temporal training signal than aligning ROIs with a global averaged text context.
- **Shortcut behaviour:** Global matching can achieve lower numerical grounding error while still providing weaker evidence of temporal discrimination.
- **Visual synthesis limitation:** The generated images remain visually limited and tend toward averaged outputs. Therefore, similarity heatmaps and text-related losses are more informative diagnostic tools for this project than image quality alone.
- **Data Preservation:** Training histories and diagnostic figures are preserved in the repository for reproducibility.

#### Evaluation & Diagnostic Tools
To evaluate the success of the grounding module, I implemented two diagnostic tools:
1. **Temporal Similarity Heatmaps:** Calculates the cosine similarity between ROI embeddings and text embeddings across the sequence to verify temporal alignment.
2. **Visual Sanity Checks:** A dedicated pipeline component that visualizes the original frame alongside the extracted ROI to ensure coordinate-to-pixel transformation integrity.

## Final Optimization & Diagnostic Analysis
After the initial experiments, a discrepancy was observed: decreasing grounding losses did not necessarily correspond to meaningful temporal alignment in the similarity heatmaps. Therefore, additional diagnostic checks were used to inspect the ROI extraction pipeline and compare the behaviour of MSE and InfoNCE alignment.

### 1. ROI Coordinate Calibration
Systematic visual sanity checks showed that ROI quality strongly depends on the correct interpretation of the bounding box format.

- **Problem:** Early crops were often poorly positioned because the bounding box format was initially treated as `[x1, y1, x2, y2]`.
- **Solution:** The stored box representation was handled as `[x, y, width, height]`, and the pixel-space crop was computed as:
  `pixel_bbox = [x*W, y*H, (x+w)*W, (y+h)*H]`
- **Result:** The corrected transformation produced more meaningful local crops. The annotations remain noisy and do not always perfectly center on character faces, but they provide stronger frame-specific visual context than the earlier crops.

### 2. Loss Ablation: MSE vs. InfoNCE
The most significant breakthrough came from switching the grounding objective:

| Objective | Heatmap Pattern | Conclusion |
| :--- | :--- | :--- |
| **MSE (Regression)** | Vertical streaks | The model achieves low regression error, but the alignment pattern suggests limited frame-specific discrimination. |
| **InfoNCE (Contrastive)** | More differentiated structure with partial diagonal behaviour | Contrastive learning provides a stronger signal for distinguishing correct and incorrect temporal ROI-text pairs. |

### 3. Summary of Findings
The combination of corrected `[x, y, width, height]` parsing and InfoNCE-based alignment improved the temporal structure of the ROI-text similarity matrix. Compared with the MSE objective, the InfoNCE run shows a more differentiated pattern and partially stronger diagonal behaviour.

This suggests that contrastive learning is better suited for frame-aware ROI-text alignment than pure embedding regression. However, the alignment is not perfect. The ROI annotations remain noisy, and the heatmap should be interpreted as evidence of improved temporal discrimination rather than proof of complete semantic grounding or identity-level recognition.

## Limitations
- The ROI annotations are noisy and do not always perfectly capture the intended character or object.
- The experiments were conducted on a small-scale setup with limited training time and batch size.
- Component losses are used mainly as diagnostic indicators; the similarity heatmaps provide the main evidence for temporal alignment behaviour.
- The InfoNCE-based alignment improves temporal discrimination compared with MSE, but it does not fully solve grounded story understanding.
- Image generation quality remains limited, so the evaluation focuses primarily on embedding alignment and temporal ROI-text correspondence.
