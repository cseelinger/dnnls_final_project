# dnnls_final_project

## Project Roadmap & Execution Plan
To extend the baseline multimodal model, I followed a structured approach focusing on visual grounding and temporal ROI-text alignment.

### Phase 1: Grounding Module Enhancement
1. **Data Parsing:** Extracting entity-level bounding boxes from the chain_of_thought markdown tables using Regex.
2. **ROI Integration:** Implementing a region-of-interest (ROI) cropping mechanism to provide the model with local visual features alongside global frame information.
3. **Architecture Update:** Injecting ROI embeddings into the sequence predictor to align text tokens with specific image regions.

### Phase 2: Implementation of Training Objectives
1. **Contrastive Alignment (Experiment 1):** Implementing an InfoNCE loss to increase the relative similarity of matching ROI-text pairs compared with incorrect temporal pairs.
2. **Loss Ablation:** Comparing the performance of InfoNCE vs. MSE-based alignment to determine the most effective grounding signal.

### Phase 3: Temporal & Global Evaluation
1. **Frame-Aware Matching (Experiment 2):** Testing frame-specific ROI alignment against global context matching to evaluate whether the model learns stronger temporal ROI-text correspondence.
2. **Final Analysis:** Generating similarity heatmaps and diagnostic metrics to analyse whether the grounding module produces frame-dependent alignment patterns.

## Implementation Progress

### Component 1: Grounding Module

**Task:** Extraction of Grounding Information from Chain-of-Thought (CoT).
Since the `daniel3303/StoryReasoning` dataset stores grounding information (bounding boxes) within the `chain_of_thought` markdown string, I implemented a custom parser.
**Technical Specification: Coordinate System:**  
Through visual sanity checks, I verified that the CoT bounding box values match the original frame resolution and should be treated as direct pixel coordinates in `[x1, y1, x2, y2]` format.

- **Image Resolution:** approximately 575x240 px, depending on the sample.
- **Coordinate Format:** direct pixel coordinates `[x1, y1, x2, y2]`.
- **Parser Logic:** Regular Expressions are used to scan the CoT for image sections (`## Image X`) and extract coordinates from markdown tables.
- **ROI Extraction:** The extracted pixel coordinates are passed directly to the cropping function without normalization or scaling by image width/height.
- **Verification:** Visual sanity checks confirmed that the extracted boxes align with meaningful local image regions such as characters, faces, body parts, and relevant objects.

**Status:** Phase 1 Completed. Extraction and synchronization of Grounding Information.
I have successfully closed the data loop between the raw Chain-of-Thought (CoT) text and the model's training pipeline.

#### 1. Data Parsing & Coordinate Handling
- **Regex Extraction:** Implemented a custom parser to extract entity-level bounding boxes from markdown tables within the `chain_of_thought` field.
- **Coordinate Handling:** Visual inspection showed that the raw CoT values already correspond to pixel coordinates in `[x1, y1, x2, y2]` format. Therefore, the parser stores the extracted coordinates directly without dividing them by 1000 or normalizing them to `[0, 1]`.
- **Persistent Mapping:** Applied the parser to the entire dataset via `.map()`, creating a structured `parsed_boxes` column for efficient access.

#### 2. ROI Integration & Pipeline Sync
- **ROI Extraction:** Integrated a cropping mechanism within the `__getitem__` method. The final implementation uses raw CoT pixel boxes in `[x1, y1, x2, y2]` format and resizes the extracted ROIs to 60x125 px.
- **DataLoader Update:** Synchronized the `DataLoader` to output a 10-element batch tuple, including the new `context_rois_tensor`.
- **Loop Synchronization:** Updated both the **Main Training Loop** and the **Validation Routine** to handle the expanded data structure.
- **Verification:** Visual sanity checks were used to inspect the extracted crops. The crops provide meaningful local visual context, although the annotations remain noisy and do not always perfectly center on the intended character or object.

**Project Status:**
- **Phase 1:** Completed. Chain-of-thought bounding boxes were parsed and converted into structured ROI annotations.
- **Phase 2:** Completed. ROI crops were integrated into the data pipeline and encoded as local visual features.
- **Phase 3:** Completed in an initial version. MSE-based, InfoNCE-based, no-alignment, and global-matching configurations were implemented and evaluated using training curves and temporal similarity heatmaps.
- **ROI Coordinate Fix:** Completed. Visual sanity checks showed that the CoT boxes should be treated as direct pixel coordinates `[x1, y1, x2, y2]`, not as normalized 1000-grid coordinates. The ROI extraction pipeline was corrected accordingly.
- **Next Step:** The final experiments are re-run with the corrected ROI extraction to ensure that all reported results are based on verified local visual crops.

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
1. **Temporal Similarity Heatmaps:** Calculates the cosine similarity between ROI embeddings and text embeddings across the sequence to diagnose temporal alignment behaviour.
2. **Visual Sanity Checks:** A dedicated pipeline component that visualizes the original frame alongside the extracted ROI to ensure coordinate-to-pixel transformation integrity.

## Final Optimization & Diagnostic Analysis
After the initial experiments, a discrepancy was observed: decreasing grounding losses did not necessarily correspond to meaningful temporal alignment in the similarity heatmaps. Therefore, additional diagnostic checks were used to inspect the ROI extraction pipeline and compare the behaviour of MSE and InfoNCE alignment.

### 1. ROI Coordinate Calibration
Systematic visual sanity checks showed that ROI quality strongly depends on the correct interpretation of the bounding box format.

- **Problem:** The CoT coordinate format was not explicitly documented. Early implementations tested normalized coordinate interpretations, which produced unreliable or poorly positioned crops.
- **Solution:** The raw CoT values were inspected against the original frames. Since the values matched the image resolution, the final implementation treats them as direct pixel coordinates in `[x1, y1, x2, y2]` format:
  `pixel_bbox = [x1, y1, x2, y2]`
- **Result:** The corrected transformation produces meaningful local crops that align with the visual sanity checks. The annotations remain noisy and sometimes cover large image regions, but the ROIs now provide more reliable frame-specific visual context.

### 2. Loss Ablation: MSE vs. InfoNCE
The most significant breakthrough came from switching the grounding objective:

| Objective | Heatmap Pattern | Conclusion |
| :--- | :--- | :--- |
| **MSE (Regression)** | Vertical streaks | The model achieves low regression error, but the alignment pattern suggests limited frame-specific discrimination. |
| **InfoNCE (Contrastive)** | More differentiated structure with partial diagonal behaviour | Contrastive learning provides a stronger signal for distinguishing correct and incorrect temporal ROI-text pairs. |

### 3. Current Findings and Next Step
The initial experiments showed that numerical grounding losses alone are not sufficient to evaluate temporal grounding. Similarity heatmaps are necessary to inspect whether the model actually learns frame-specific ROI-text correspondence.

During final validation, the ROI coordinate interpretation was corrected to direct pixel coordinates `[x1, y1, x2, y2]`. Therefore, the final no-alignment, MSE, InfoNCE, and global-matching experiments are re-run with the verified ROI extraction pipeline before drawing the final conclusions.

The expected analysis remains the same: MSE and global matching are evaluated for shortcut-like behaviour, while InfoNCE is evaluated for whether it produces a more differentiated temporal ROI-text similarity structure.

## Limitations
- The ROI annotations are noisy, and some boxes cover large image regions rather than small, precise character or object areas.
- The experiments were conducted on a small-scale setup with limited training time and batch size.
- Component losses are used mainly as diagnostic indicators; the similarity heatmaps provide the main evidence for temporal alignment behaviour.
- The InfoNCE-based alignment improves temporal discrimination compared with MSE, but it does not fully solve grounded story understanding.
- Image generation quality remains limited, so the evaluation focuses primarily on embedding alignment and temporal ROI-text correspondence.
- The CoT coordinate format was not explicitly documented and required empirical verification through visual sanity checks.
