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
├── docs/                   # Documentation, references, IEEE paper drafts
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

## Evaluation Metrics

- **BLEU-4** — Translation quality
- **Word Error Rate (WER)** — Recognition accuracy
- **User Feedback** — Deaf community evaluation

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-org/ASL-ISL.git
cd ASL-ISL

# Install Python dependencies
pip install -r requirements.txt
```

---

## References

1. Li et al. — *WLASL: A Large-Scale Dataset for Sign Language Recognition* (IEEE, 2020)
2. Camgoz et al. — *Neural Machine Translation for Sign Languages* (CVPR, 2018)
3. *Sign Language Transformers for Continuous Sign Recognition* (IEEE, 2021)
4. Google Research — *MediaPipe Hands: On-device Real-time Hand Tracking* (2020)
5. *GAN-based Human Motion Synthesis for Sign Language Avatars* (2022)
