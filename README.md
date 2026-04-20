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