# dnnls_final_project

## Project Overview

This project extends a pre-trained multimodal model for predicting localized sequences, with a focus on **frame-aware visual localization**. The main objective is to investigate whether local visual regions extracted from “Chain-of-Thought” (CoT) bounding boxes can improve the temporal alignment of ROIs and text.

The project focuses on two defined architectural components:

1. **Grounding Module**  
   CoT bounding boxes are parsed and converted into frame-specific Region-of-Interest (ROI) patches. These ROI patches are encoded using the visual encoder and used as local visual features.

2. **Alignment / Contrastive Loss**  
   ROI embeddings are aligned with text embeddings using either MSE regression or contrastive InfoNCE learning. The resulting alignment behavior is evaluated using training curves and heatmaps of ROI-text similarity.

The final experiments compare four configurations:

- **No Alignment:** no explicit ROI-text grounding loss
- **MSE Frame-Aware Alignment:** ROI at time step `t` is aligned with the text embedding at time step `t` using MSE
- **InfoNCE frame-aware alignment:** ROI at time step `t` is aligned with the text embedding at time step `t` using contrastive learning
- **Global matching:** ROI is aligned with an averaged global text context instead of the corresponding time step