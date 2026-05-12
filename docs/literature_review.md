# Literature Review — ASL-ISL Translation Engine (AITE)

| Title | Year | Type | Authors | Gaps Found | Methodology | Opportunities | Relevance |
|-------|------|------|---------|------------|-------------|---------------|-----------|
| Neural Sign Language Translation | 2018 | IEEE Conference | Camgoz et al. | Limited dataset scale; not cross-lingual ASL↔ISL | Neural machine translation for sign videos | Extend to multilingual translation | Baseline for translation module |
| How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language | 2020 | Dataset / arXiv | Duarte et al. | Dataset only, not translation system | Large multimodal dataset creation | Train better continuous models | Additional training/evaluation data |
| Text2Sign: Towards Sign Language Production Using Neural Machine Translation and GANs | 2021 | Research Paper | Stoll et al. | Avatar realism limitations | NMT + GAN sign generation | Improve realistic ISL avatar output | Reference for generation layer |
| Progressive Transformers for End-to-End Sign Language Production | 2020 | Research Paper | Saunders et al. | Production quality constraints | Transformer-based sign production | Better text-to-sign synthesis | Useful for avatar generation |
| Everybody Sign Now | 2020 | Research Paper | Saunders et al. | Photorealistic generation challenges | Pose/video generation | Photorealistic avatar generation | Generation inspiration |
| Skeleton-Based Dynamic Hand Gesture Recognition Using a Part-Based GRU-RNN | 2020 | Research Paper | Shin and Kim | May not scale to full sentence CSLR | Skeleton + GRU-RNN | Efficient gesture modeling | Recognition baseline |
| Word-Level Deep Sign Language Recognition from Video | 2020 | IEEE WACV | Li et al. | Word-level, not continuous translation | Deep video recognition benchmark | Upgrade to sentence-level | Core ASL recognition reference |
| Sign Language Recognition with Transformer Networks | 2020 | Conference | De Coster et al. | Needs large compute/data | Transformer recognition | Improved temporal modeling | Recognition architecture reference |
| Sign Language Transformers: Joint End-to-End Recognition and Translation | 2020 | IEEE CVPR | Camgoz et al. | Mostly benchmark datasets | Joint transformer | End-to-end translation | Closest to full pipeline |
| MediaPipe Hands: On-device Real-Time Hand Tracking | 2020 | arXiv / Google Research | Zhang et al. | Hand-only focus | Real-time landmark detection | Fast deployment | Key preprocessing module |
| Proceedings of the Twelfth Indian Conference on CV, Graphics and Image Processing | 2021 | Conference Proceedings | Chellappa | - | - | - | Possible ISL context reference |
| Skeleton Aware Multi-modal Sign Language Recognition | 2021 | Research Paper | Jiang et al. | Complex multimodal fusion | Skeleton + multimodal learning | Improve robustness | Feature fusion ideas |
| Pose-Guided Sign Language Video GAN with Dynamic Lambda | 2021 | Research Paper | Kissel et al. | GAN instability | Pose-guided GAN | Better sign avatar realism | Generation layer |
| Human Body Pose Estimation and Applications | 2021 | IEEE | K et al. | General paper, not sign-specific | Pose estimation survey/application | Broader pose techniques | Pose extraction background |
| Continuous Sign Language Recognition Based on Spatial-Temporal Graph Attention Network | 2022 | Research Paper | Guo et al. | May need heavy compute | ST-GAT | Continuous CSLR | Sentence-level recognition |
| Sign Pose-Based Transformer for Word-Level Sign Language Recognition | 2022 | Research Paper | Boháček and Hrúz | Word-level limitation | Pose transformer | Extend to continuous recognition | Recognition model ideas |
| A Speech-driven Sign Language Avatar Animation System | 2022 | Conference | Hu et al. | Speech/sign domain mismatch | Speech-to-avatar animation | Real-time avatar systems | Generation concepts |
| Human Part-wise 3D Motion Context Learning for Sign Language Recognition | 2023 | Research Paper | Lee et al. | 3D data complexity | 3D motion context learning | Higher accuracy recognition | Advanced recognition ideas |
| Gloss-Free End-to-End Sign Language Translation | 2023 | Research Paper | Lin et al. | Needs strong data | End-to-end gloss-free translation | Reduce annotation dependency | Useful if gloss scarce |
| SignNet II: A Transformer-Based Two-Way Sign Language Translation Model | 2023 | Research Paper | Chaudhary et al. | Dataset/domain limitations | Bidirectional transformer translation | Two-way translation | Relevant to ASL↔ISL vision |
| Sign Language Motion Generation from Sign Characteristics | 2023 | Research Paper | Gil-Martín et al. | Generation realism | Motion generation | Avatar improvement | Generation module |
| Real-time sign language detection: Empowering the disabled community | 2024 | Research Paper | Kumar et al. | Likely limited scope | Real-time detection | Deployment ideas | Realtime UX reference |
| Continuous Sign Language Recognition System using Deep Learning with MediaPipe Holistic | 2024 | Research Paper | Srivastava et al. | Holistic accuracy constraints | MediaPipe + deep learning | Full-body feature extraction | Realtime recognition |
| Enhanced Sign Language Translation between ASL and ISL | 2024 | arXiv | Kumar et al. | May be prototype-level | Cross-lingual translation | Direct ASL↔ISL extension | Closest problem match |
| Real-Time Indian Sign Language Translation Using Deep Learning and Multilingual Speech Technologies | 2025 | Research Paper | Shankar et al. | ISL-specific limitations | Deep learning + speech tech | Bidirectional accessibility | ISL insights |
| Interpretation of Indian Sign Language to Text and Speech | 2025 | Research Paper | Ingoley and Bakal | Limited translation scope | ISL recognition to text/speech | ISL deployment | ISL pipeline reference |
| Automatic sign language to text translation using MediaPipe and transformer architectures | 2025 | Research Paper | Maia et al. | Likely no cross-lingual mapping | MediaPipe + transformer | Efficient translation | Architecture similarity |
| Mamba vision models: Automated American sign language recognition | 2025 | Research Paper | Altaher et al. | New model maturity | Mamba vision architecture | Efficient sequence modeling | Alternative recognition model |
| Real-Time Speech-to-Sign Language (ISL) Converter | 2025 | Research Paper | Diksha Rade et al. | Speech-driven only | Speech-to-sign conversion | Text/speech to ISL output | Generation pathway ideas |
| Ensemble transformer-based word-level sign language recognition with multi-modal fusion | 2026 | Research Paper | Alkhoraif et al. | Word-level focus | Ensemble transformers + multimodal fusion | Higher recognition accuracy | Recognition benchmarking |
