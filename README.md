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

> **Current phase:** Research prototype with a validated MSASL-100 isolated-sign
> recognizer and limited recorded-pose playback. ASL-to-ISL translation and
> generative avatar work are not production-ready.

See [Project Status, Reproduction Guide, and Roadmap](docs/PROJECT_STATUS_AND_ROADMAP.md)
for the authoritative list of completed changes, measured model results, current
limitations, reproducible commands, and prioritized future work.

---

## Overview

AITE is a research prototype for a modular cross-sign-language pipeline between
American Sign Language (ASL) and Indian Sign Language (ISL). The intended system
takes ASL video, recognizes signs, translates the gloss across languages, and
renders ISL motion. The current tested demo supports one isolated MSASL-100-style
ASL sign from an uploaded or webcam-recorded clip when recognition is explicitly
enabled. It otherwise uses manual text and recorded 2D landmark playback;
continuous recognition and generative animation remain future work.

### Pipeline

```
Input (Video) → Preprocessing → ASL Recognition → Cross-lingual Translation → ISL Avatar Generation
```

### Stages

1. **Sign Recognition** — A calibrated 100-class isolated-sign Transformer is
   available behind an explicit feature flag. It is not a continuous ASL recognizer
   and does not cover signs outside its fixed vocabulary.
2. **Cross-Lingual Translation** — A deterministic draft rule layer is available;
   the optional Llama model is disabled and ISL expert review is still required.
3. **Sign Generation** — The demo retrieves recorded 2D landmark motion. A trained
   pose generator/GAN and realistic avatar have not been completed.

---

## Repository Structure

```
ASL-ISL/
├── src/
│   ├── recognition/        # ASL sign recognition (Transformer + MediaPipe)
│   ├── translation/        # ASL → ISL grammar translation (LLM)
│   ├── generation/         # GAN-based avatar animation
│   ├── web/                # Web application
│   └── utils/              # Preprocessing, feature extraction, helpers
├── data/
│   ├── asl/                # Validated derived MSASL features and splits
│   └── isl/                # INCLUDE-50 and ISL-CSLRT derived data
├── Dataset/MS-ASL/         # Local MSASL annotations/videos (not for deployment)
├── scripts/                # Jupyter notebooks for exploration & training
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
| [MSASL](https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset) | Microsoft | Multi-signer isolated ASL videos |
| [INCLUDE-50](https://www.kaggle.com/datasets/yuvrajjoshi1110/include-50) | Public research dataset | Isolated ISL videos used for playback |
| ISL-CSLRT | Local licensed research copy | ISL sentence frame sequences and word images |

---

## Tech Stack

**Languages:** Python, JavaScript  
**Frameworks:** PyTorch / TensorFlow, MediaPipe, Hugging Face Transformers, OpenCV  
**Frontend:** HTML, CSS, and browser JavaScript

**Infrastructure:** Local FastAPI service; GPU/cloud training is future work

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

## System Requirements

- **Main application:** project `.venv`
- **CSLRT landmark extraction:** Python 3.12 `.venv-cslrt` with MediaPipe 0.10.21
- **ffmpeg**: Must be installed and available in your system's PATH for the avatar generation to work.

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
