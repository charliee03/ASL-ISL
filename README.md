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

> **Current Phase:** All pipeline stages complete — Recognition, Translation, 2D Skeletal Generation, Web UI, Docker deployment, and QA testing.

---

## Overview

AITE is a modular pipeline for real-time cross-sign-language translation between American Sign Language (ASL) and Indian Sign Language (ISL). The system takes video input (webcam, upload, or YouTube URL), recognizes ASL signs, translates the gloss across languages, and renders the output as an avatar animation.

### Pipeline

```
Input (Video) → Preprocessing → ASL Recognition → Cross-lingual Translation → ISL Avatar Generation
```

### Stages

1. **Sign Recognition** — Transformer-based model (trained on WLASL) extracts ASL gloss/text from video frames using MediaPipe hand keypoints.
2. **Cross-Lingual Translation** — Rule-based grammar engine with optional quantized LLM (Llama-2-7b) transforms ASL grammar/syntax to ISL.
3. **Sign Generation** — 2D skeletal avatar synthesizer renders ISL signs from keypoint data in real-time (≥150 FPS).

---

## Repository Structure

```
ASL-ISL/
├── src/
│   ├── api/                # FastAPI server (REST endpoints)
│   ├── recognition/        # ASL sign recognition (Transformer + MediaPipe)
│   ├── translation/        # ASL → ISL grammar translation (rules + LLM)
│   ├── generation/         # 2D skeletal avatar animation
│   ├── web/                # Web UI (vanilla HTML/CSS/JS)
│   └── utils/              # Preprocessing, feature extraction, helpers
├── data/
│   ├── msasl/              # MSASL dataset (sign videos, glosses)
│   ├── wlasl/              # WLASL dataset
│   ├── isl/                # ISL keypoints & INCLUDE dataset
│   └── paired/             # Paired ASL-ISL data
├── scripts/                # Training, evaluation & QA scripts
├── configs/                # Model & pipeline configuration files
├── models/                 # Trained model checkpoints
├── docs/                   # Documentation, literature review, IEEE paper
│   ├── monthly_reports/    # Project progress and monthly status reports
│   └── jira_backlog_import.csv # Jira tasks export
├── tests/                  # Unit and integration tests
├── Dockerfile              # Docker containerisation
├── .dockerignore
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| [MSASL](https://microsoft.github.io/data-for-society/dataset?d=MS-ASL-American-Sign-Language-Dataset) | Microsoft | Large-scale ASL dataset |
| [INCLUDE-50](https://www.kaggle.com/datasets/yuvrajjoshi1110/include-50) | Public | Indian Sign Language dataset |

---

## Tech Stack

**Languages:** Python, JavaScript  
**Frameworks:** PyTorch, MediaPipe, Hugging Face Transformers, OpenCV, FastAPI  
**Frontend:** Vanilla HTML / CSS / JavaScript  
**Infrastructure:** CPU or GPU-enabled systems, Docker, optional cloud (Render / HF Spaces)

---

## Timeline

| Weeks | Milestone |
|-------|-----------|
| 1–2 | Literature review, dataset collection & preprocessing |
| 3–4 | MediaPipe integration, hand keypoint extraction |
| 5–6 | Train Transformer for ASL recognition, evaluate (WER) |
| 7–8 | Translation module (ASL → ISL), grammar rule implementation |
| 9+ | 2D skeletal avatar generation, Web UI, Docker, QA testing, IEEE paper |

---

## Monthly Reports

- [Month 1 Summary (Apr 2026)](docs/monthly_report_1.md) — Project setup, literature review, feature extraction
- [Month 1 Full Report (PDF)](docs/monthly_reports/P107_MonthlyProgressReport_1.docx.PDF)
- [Month 2 Full Report (PDF)](docs/monthly_reports/Monthly%20Report%20-%202.docx.PDF)
- [Month 3 Full Report (PDF)](docs/monthly_reports/Monthly_Report_3_July_2026.pdf)

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

- **Python 3.10+**
- **ffmpeg**: Must be installed and available in your system's PATH for the avatar generation to work.

---

## Getting Started

```bash
git clone https://github.com/charliee03/ASL-ISL.git
cd ASL-ISL
pip install -r requirements.txt
```

To run the application (both backend and web frontend):

```bash
python -m src.api.server
# Or, for development with hot-reload:
# uvicorn src.api.server:app --reload
```

The application will be available at `http://localhost:8000`.

### Docker

```bash
docker build -t aite:local .
docker run --rm -p 8000:8000 aite:local
```

See [`docs/deployment.md`](docs/deployment.md) for hosting and defense-recording details.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check (model status) |
| `POST` | `/extract-keypoints` | Extract hand keypoints from an image |
| `POST` | `/predict-gloss` | Predict ASL gloss from a single frame |
| `POST` | `/predict-sequence` | Predict ASL gloss from a video clip |
| `POST` | `/translate` | Translate ASL gloss to ISL gloss |
| `POST` | `/generate-avatar` | Generate ISL avatar animation video |

### QA Testing

Run the end-to-end QA harness against 20 MS-ASL clips:

```bash
python scripts/run_qa.py
```

This produces a `qa_report.csv` with per-clip recognition, translation, and avatar generation results.

---

## Troubleshooting & Models

If you encounter a `401 Client Error` indicating access to `meta-llama/Llama-2-7b-chat-hf` is restricted:

### Option 1: Authenticate with HuggingFace
1. Request access on the [Llama-2 HuggingFace page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
2. Generate a Read token in your HuggingFace settings.
3. Run `huggingface-cli login` in your terminal and provide the token.

### Option 2: Use a non-gated model
To bypass the gated model, update `configs/translation.yaml`:
Change `model_id: meta-llama/Llama-2-7b-chat-hf` to an open model, such as `model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

*(Note: Even if the translation model fails to load, the server will fall back to rule-based translation and the UI will still be accessible!)*

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
