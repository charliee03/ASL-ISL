# Section III: Methodology

## A. Data Preprocessing and Keypoint Extraction

The proposed recognition pipeline operates on pose-based keypoint
representations rather than raw RGB video, following the approach
established by Bohacek and Hruz [1]. This design choice yields a
compact, signer-invariant feature space that is robust to variations
in background, lighting, and appearance.

### Holistic Landmark Detection

We employ **MediaPipe Holistic** [2] to extract body, hand, and face
landmarks from each video frame. MediaPipe Holistic produces 543
landmarks per frame (33 pose + 468 face + 21 per hand × 2);
however, the majority of these landmarks carry redundant or
task-irrelevant information for isolated sign recognition. We
therefore apply a principled **27-node graph reduction** that retains
only the anatomically and linguistically salient keypoints, as
detailed in Table I.

**Table I: 27-Keypoint Graph Reduction from MediaPipe Holistic Output**

| Group | Count | MediaPipe Indices | Anatomical Description |
|---|---|---|---|
| Left Hand | 10 | 2, 4, 5, 8, 9, 12, 13, 16, 17, 20 | Thumb tip & IP joint, index fingertip & MCP, middle fingertip & MCP, ring fingertip & MCP, pinky fingertip & MCP |
| Right Hand | 10 | 2, 4, 5, 8, 9, 12, 13, 16, 17, 20 | (Same as left hand) |
| Upper-Body Pose | 7 | 11, 12, 13, 14, 15, 16, 0 | Left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, nose |

Each keypoint is represented as a 3D coordinate tuple (*x*, *y*, *z*),
yielding a per-frame feature vector of dimensionality 27 × 3 = **81**.

### Signing-Space Normalization

Raw MediaPipe coordinates are expressed in a frame-relative coordinate
system and are therefore sensitive to the signer's position, distance
from the camera, and body proportions. We apply the **signing-space
normalization** method proposed by Bohacek and Hruz [1] to project all
keypoints into a canonical, signer-invariant reference frame.

Let **p**_nose = (*x*_n, *y*_n, *z*_n) denote the nose landmark, and
let *d*_s denote the Euclidean distance between the left and right
shoulder landmarks in the (*x*, *y*) plane:

> *d*_s = sqrt((*x*_ls − *x*_rs)² + (*y*_ls − *y*_rs)²)

The **head metric** *h* is defined as:

> *h* = *d*_s / 2

Each keypoint **p** = (*x*, *y*, *z*) is then normalized as follows:

> *x̂* = (*x* − *x*_n) / (6*h*)
>
> *ŷ* = (*y* − *y*_n) / (7*h*)
>
> *ẑ* = (*z* − *z*_n) / (6*h*)

The asymmetric scaling factors (6*h* for the horizontal and depth axes
versus 7*h* for the vertical axis) account for the typical aspect
ratio of the human signing space. When pose landmarks are not
detected, we fall back to default values of *h* = 0.15 and a center
of (0.5, 0.5, 0.0).

### Temporal Sampling

Sign language videos exhibit considerable variation in duration. To
produce fixed-length input sequences, we apply **uniform temporal
sampling**: for each video with *T* total frames, we select
*N* = 32 frame indices uniformly spaced over [0, *T* − 1] via
`np.linspace(0, T − 1, 32)`. This strategy preserves the temporal
structure of the sign while standardizing the sequence length across
all samples.

### Keypoint Caching

Extracting keypoints via MediaPipe is computationally expensive. To
avoid redundant computation across training epochs, extracted keypoint
sequences are cached as `.npy` files in a per-dataset cache directory
(`data/wlasl/cached_keypoints/`). Each cache file is keyed by
`{video_name}_f{num_frames}.npy`, enabling transparent reuse across
runs with identical frame-sampling parameters.


## B. Data Augmentation

To regularize training and improve generalization, we apply a
stochastic augmentation pipeline to the normalized keypoint sequences
during training. All augmentations operate directly in keypoint space
(i.e., post-normalization), avoiding the computational cost of
video-level augmentation.

### RandomMirror (*p* = 0.5)

With probability *p* = 0.5, the sign is horizontally reflected.
This involves three operations: (1) negating the *x*-coordinate of
all keypoints, (2) swapping the left-hand keypoints (indices 0–9) with
the right-hand keypoints (indices 10–19), and (3) swapping bilaterally
symmetric pose keypoints (left shoulder ↔ right shoulder,
left elbow ↔ right elbow, left wrist ↔ right wrist). This
augmentation effectively doubles the training set by exploiting the
approximate bilateral symmetry of the signing space.

### RandomRotation (±13°)

A rotation angle θ is sampled uniformly from [−13°, +13°]. The
(*x*, *y*) coordinates of all keypoints are then rotated around the
origin:

> *x′* = *x* cos θ − *y* sin θ
>
> *y′* = *x* sin θ + *y* cos θ

The *z*-coordinate is left unchanged. This simulates slight camera
tilt or signer orientation changes.

### RandomSqueeze (±15%)

Independent scaling factors *s_x* and *s_y* are sampled uniformly
from [0.85, 1.15]. The *x* and *y* coordinates are scaled by their
respective factors:

> *x′* = *s_x* · *x*,   *y′* = *s_y* · *y*

This simulates variations in body proportions and aspect ratio, and
also approximates minor perspective distortions.

All three augmentations are composed sequentially in the order:
RandomMirror → RandomRotation → RandomSqueeze.


## C. Model Architecture

We adopt a **SPOTER-inspired Transformer** architecture [1] for
isolated sign recognition. The model processes temporal sequences of
keypoint features through a Transformer encoder, and employs a
learnable class query token decoded via cross-attention to produce the
final classification. Fig. 1 illustrates the full pipeline.

> **Fig. 1.** *(Pipeline Diagram — to be inserted)* Overview of the
> proposed sign recognition pipeline. Video frames are processed by
> MediaPipe Holistic to extract 27 keypoints per frame. After
> signing-space normalization and temporal sampling (32 frames), the
> resulting 32 × 81 feature matrix is fed to the Transformer encoder.
> A learnable class query token attends over the encoder output via
> cross-attention, and the decoded representation is projected to the
> gloss vocabulary.

### Input Projection

The 81-dimensional keypoint feature vector at each time step is
projected to the model's internal dimension via a linear layer:

> **x**_proj = Linear(81, *d*_model),   *d*_model = 128

### Positional Encoding

Sinusoidal positional encodings [3] are added to the projected
features to inject temporal ordering information. The encoding is
precomputed for a maximum sequence length of 1000 positions:

> PE(pos, 2*i*) = sin(pos / 10000^(2*i* / *d*_model))
>
> PE(pos, 2*i*+1) = cos(pos / 10000^(2*i* / *d*_model))

A dropout of 0.1 is applied to the sum of projections and positional
encodings.

### Transformer Encoder

The core feature extractor is a standard Transformer encoder stack
consisting of:

| Hyperparameter | Value |
|---|---|
| Number of layers | 3 |
| Attention heads | 4 |
| Feedforward dimension | 512 (*d*_model × 4) |
| Dropout | 0.2 |
| Batch-first mode | True |

Each encoder layer applies multi-head self-attention followed by a
position-wise feedforward network, with residual connections and layer
normalization (pre-LN variant as implemented in PyTorch).

### SPOTER-Style Class Query Decoding

Rather than applying mean-pooling or using a [CLS] token prepended to
the input, we follow the SPOTER approach [1] and introduce a
**learnable class query** parameter of shape (1, 1, *d*_model). At
inference time, this query is expanded to the batch dimension and
decoded via a single **multi-head cross-attention** layer:

> **q** = class_query.expand(*B*, 1, *d*_model)
>
> **a**, \_ = MultiheadAttention(query=**q**, key=**E**, value=**E**)

where **E** ∈ ℝ^(*B* × 32 × *d*_model) is the encoder output. A
residual connection and LayerNorm are applied:

> **o** = LayerNorm(**q** + Dropout(**a**))

### Output Projection

The decoded representation **o** ∈ ℝ^(*B* × *d*_model) (after
squeezing the sequence dimension) is projected to the gloss
vocabulary:

> logits = Linear(*d*_model, *V*)

where *V* is the vocabulary size (100 for WLASL-100).

### Model Size

The full model contains approximately **~2M trainable parameters**
with the configuration described above (*d*_model = 128, 3 encoder
layers, 4 heads, *V* = 100).


## D. Training Configuration

The model is trained on the **WLASL-100** subset of the Word-Level
American Sign Language dataset [4], comprising 100 ASL glosses with
a total of approximately 2,000 video samples split into training,
validation, and test partitions.

**Table II: Training Hyperparameters**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 5 × 10⁻⁴ |
| LR scheduler | CosineAnnealingLR |
| *T*_max | 300 |
| *η*_min | 1 × 10⁻⁶ |
| Batch size | 32 |
| Gradient clipping | max_norm = 1.0 |
| Max epochs | 300 |
| Early stopping patience | 50 epochs |
| Loss function | CrossEntropyLoss |

The learning rate follows a cosine annealing schedule from the initial
value of 5 × 10⁻⁴ to a minimum of 1 × 10⁻⁶ over 300 epochs. Early
stopping monitors validation accuracy and halts training if no
improvement is observed for 50 consecutive epochs. The best model
checkpoint (by validation accuracy) is saved to
`models/recognition/best_model.pt`.

---

### References

[1] M. Bohacek and M. Hruz, "Sign Pose-based Transformer for
Word-level Sign Language Recognition," in *Proc. IEEE/CVF Winter
Conf. on Applications of Computer Vision Workshops (WACVW)*, 2022.

[2] C. Lugaresi *et al.*, "MediaPipe: A Framework for Building
Perception Pipelines," *arXiv preprint arXiv:1906.08172*, 2019.

[3] A. Vaswani *et al.*, "Attention Is All You Need," in *Proc. 31st
Int. Conf. on Neural Information Processing Systems (NeurIPS)*, 2017.

[4] D. Li, C. Rodriguez, X. Yu, and H. Li, "Word-level Deep Sign
Language Recognition from Video: A New Large-scale Dataset and Methods
Comparison," in *Proc. IEEE/CVF Winter Conf. on Applications of
Computer Vision (WACV)*, 2020.
