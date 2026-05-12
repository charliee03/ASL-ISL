# ASL-ISL Translation Engine (AITE)

**Team ID:** P107  
**Domain:** AI, Computer Vision, NLP, HCI  
**IEEE CS IAMPRO Internship**

---

## Team

| Name | Role |
|------|------|
| Naman Nagar | Team Leader |
| Monisha Sharma | Team Member |
| Nandita R Nadig | Team Member |

**Mentor:** Dr. Sudhamani M J

---

## Project Status

> **Current Phase:** Weeks 1–4 complete (Repository setup, Literature review, MediaPipe integration, Feature extraction pipeline)

---

## Overview

AITE is a modular pipeline for real-time cross-sign-language translation between American Sign Language (ASL) and Indian Sign Language (ISL). The system takes video input (webcam, upload, or YouTube URL), recognizes ASL signs, translates the gloss across languages, and renders the output as an avatar animation.

### Pipeline

```
Input (Video) → Preprocessing → ASL Recognition → Cross-lingual Translation → ISL Avatar Generation
```

### Stages

1. **Sign Recognition** — Transformer-based model (trained on WLASL) extracts ASL gloss/text from video frames using MediaPipe hand keypoints.
2. **Cross-Lingual Translation** — Lightweight quantized LLM transforms ASL grammar/syntax to ISL.
3. **Sign Generation** — GAN-based avatar renders ISL signs in real-time (≥15 FPS).

---

## Repository Structure

```
ASL-ISL/
├── src/
│   ├── recognition/        # ASL sign recognition (Transformer + MediaPipe)
│   ├── translation/        # ASL → ISL grammar translation (LLM)
│   ├── generation/         # GAN-based avatar animation
│   ├── web/                # Web application
│   │   ├── frontend/       # React UI
│   │   └── backend/        # Node.js API server
│   └── utils/              # Preprocessing, feature extraction, helpers
├── data/
│   ├── wlasl/              # WLASL dataset (sign videos, glosses)
│   └── isl/                # ISL-CSLGR dataset
├── notebooks/              # Jupyter notebooks for exploration & training
├── configs/                # Model & pipeline configuration files
├── models/                 # Trained model checkpoints
├── docs/                   # Documentation, literature review, IEEE paper
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| [WLASL](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | IEEE / Public | 21,000+ clips, 2,000+ ASL signs |
| [ISL-CSLGR](https://www.kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset) | Public | Indian Sign Language gestures |

---

## Tech Stack

**Languages:** Python, JavaScript  
**Frameworks:** PyTorch / TensorFlow, MediaPipe, Hugging Face Transformers, OpenCV  
**Frontend:** React + Node.js  
**Infrastructure:** GPU-enabled systems, optional cloud (AWS/GCP)

---

## Timeline

| Weeks | Milestone |
|-------|-----------|
| 1–2 | Literature review, dataset collection & preprocessing |
| 3–4 | MediaPipe integration, hand keypoint extraction |
| 5–6 | Train Transformer for ASL recognition, evaluate (WER) |
| 7–8 | Translation module (ASL → ISL), grammar rule implementation |
| 9+ | Avatar generation (GAN), web integration, testing, IEEE paper |

---

## Monthly Reports

- [Month 1 (Apr 2026)](docs/monthly_report_1.md) — Project setup, literature review, MediaPipe integration, feature extraction

---

## Evaluation Metrics

- **BLEU-4** — Translation quality
- **Word Error Rate (WER)** — Recognition accuracy
- **User Feedback** — Deaf community evaluation

---

## Literature Review

A comprehensive review of 30 papers (2018–2026) covering sign language recognition, translation, and generation. Key findings:

- **Closest to our pipeline:** Camgoz et al. (2020) — Sign Language Transformers, and Kumar et al. (2024) — Enhanced ASL↔ISL Translation
- **Recognition baselines:** Li et al. (2020), De Coster et al. (2020), Boháček & Hrúz (2022)
- **Generation reference:** Stoll et al. (2021), Saunders et al. (2020), Kissel et al. (2021)
- **Key gap identified:** No existing system performs real-time cross-sign-language translation (ASL↔ISL) handling syntactic and gestural differences

See full table in [`docs/literature_review.md`](docs/literature_review.md).

---

## Getting Started

```bash
git clone https://github.com/your-org/ASL-ISL.git
cd ASL-ISL
pip install -r requirements.txt
```

---

## References

1. Camgoz et al. — *Neural Sign Language Translation* (IEEE, 2018)
2. Duarte et al. — *How2Sign: A Large-scale Multimodal Dataset for Continuous ASL* (2020)
3. Stoll et al. — *Text2Sign: Sign Language Production Using NMT and GANs* (2021)
4. Saunders et al. — *Progressive Transformers for End-to-End Sign Language Production* (2020)
5. Saunders et al. — *Everybody Sign Now* (2020)
6. Shin & Kim — *Skeleton-Based Dynamic Hand Gesture Recognition Using GRU-RNN* (2020)
7. Li et al. — *Word-Level Deep Sign Language Recognition from Video* (IEEE WACV, 2020)
8. De Coster et al. — *Sign Language Recognition with Transformer Networks* (2020)
9. Camgoz et al. — *Sign Language Transformers: Joint End-to-End Recognition and Translation* (IEEE CVPR, 2020)
10. Zhang et al. — *MediaPipe Hands: On-device Real-Time Hand Tracking* (Google Research, 2020)
11. Chellappa — *Proceedings of the Twelfth Indian Conference on CV, Graphics and Image Processing* (2021)
12. Jiang et al. — *Skeleton Aware Multi-modal Sign Language Recognition* (2021)
13. Kissel et al. — *Pose-Guided Sign Language Video GAN with Dynamic Lambda* (2021)
14. K et al. — *Human Body Pose Estimation and Applications* (IEEE, 2021)
15. Guo et al. — *Continuous SLR Based on Spatial-Temporal Graph Attention Network* (2022)
16. Boháček & Hrúz — *Sign Pose-Based Transformer for Word-Level SLR* (2022)
17. Hu et al. — *A Speech-driven Sign Language Avatar Animation System* (2022)
18. Lee et al. — *Human Part-wise 3D Motion Context Learning for SLR* (2023)
19. Lin et al. — *Gloss-Free End-to-End Sign Language Translation* (2023)
20. Chaudhary et al. — *SignNet II: A Transformer-Based Two-Way Sign Language Translation Model* (2023)
21. Gil-Martín et al. — *Sign Language Motion Generation from Sign Characteristics* (2023)
22. Kumar et al. — *Real-time sign language detection: Empowering the disabled community* (2024)
23. Srivastava et al. — *Continuous SLR System using Deep Learning with MediaPipe Holistic* (2024)
24. Kumar et al. — *Enhanced Sign Language Translation between ASL and ISL* (arXiv, 2024)
25. Shankar et al. — *Real-Time ISL Translation Using Deep Learning and Multilingual Speech Technologies* (2025)
26. Ingoley & Bakal — *Interpretation of Indian Sign Language to Text and Speech* (2025)
27. Maia et al. — *Automatic sign language to text translation using MediaPipe and transformer architectures* (2025)
28. Altaher et al. — *Mamba vision models: Automated American sign language recognition* (2025)
29. Diksha Rade et al. — *Real-Time Speech-to-Sign Language (ISL) Converter* (2025)
30. Alkhoraif et al. — *Ensemble transformer-based word-level SLR with multi-modal fusion* (2026)
