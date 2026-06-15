# Section V-A: Recognition Results

## A. Training Progression

We train the SPOTER-inspired Transformer model on the WLASL-100
dataset for up to 300 epochs with early stopping (patience = 50).
Training and validation metrics are logged per epoch to
`models/recognition/training_log.csv`.

**Table III: Training Progression (Early Epochs)**

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|---|---|---|---|---|
| 1 | 4.7009 | 1.87 | 4.6222 | 1.78 |
| 2 | 4.6422 | 1.53 | 4.6040 | 1.78 |

> **Note:** The results above reflect the model at a very early stage
> of training (2 out of 300 epochs). At this point, both training and
> validation losses are close to the theoretical maximum for a
> 100-class uniform prior (−ln(1/100) ≈ 4.605), confirming that the
> model has not yet begun to discriminate between classes. Full
> convergence is expected to occur over the course of the remaining
> training schedule, guided by the cosine annealing learning rate
> policy and monitored via early stopping.

### Expected Training Dynamics

Based on comparable SPOTER-family architectures reported in the
literature [1], we anticipate the following training trajectory:

1. **Epochs 1–20 (warm-up):** Loss remains near the random-chance
   baseline as the model learns initial feature correlations.
2. **Epochs 20–100 (rapid learning):** Steep decline in loss and
   corresponding accuracy gains as the Transformer encoder learns
   temporal dependencies in the keypoint sequences.
3. **Epochs 100–200 (refinement):** Gradual improvement with
   diminishing returns; the cosine annealing schedule reduces the
   learning rate to enable fine-grained weight updates.
4. **Epochs 200–300 (plateau/convergence):** Model approaches peak
   performance; early stopping triggers if validation accuracy
   stagnates for 50 consecutive epochs.

> **Fig. 2.** *(Training curves — to be inserted upon completion)*
> Training and validation loss (left axis) and accuracy (right axis)
> plotted against epoch number, showing convergence behavior over the
> full training run.


## B. Evaluation Metrics

We evaluate the trained model on the WLASL-100 validation set using
the following metrics:

### Top-1 Accuracy

The proportion of test samples for which the model's highest-scoring
prediction matches the ground-truth gloss label:

> Acc@1 = (1 / *N*) · Σ 𝟙[argmax *f*(**x**_i) = *y*_i]

**Top-1 Accuracy: [TBD]%**

### Top-5 Accuracy

The proportion of test samples for which the ground-truth label
appears within the model's five highest-scoring predictions:

> Acc@5 = (1 / *N*) · Σ 𝟙[*y*_i ∈ top-5(*f*(**x**_i))]

**Top-5 Accuracy: [TBD]%**

### Word Error Rate (WER)

Although WER is traditionally applied to sequence-level recognition
tasks, we include it as a per-sample metric to maintain consistency
with the project's evaluation framework (see
`src/utils/metrics.py`). For isolated sign recognition, WER reduces
to:

> WER = (S + D + I) / N

where *S*, *D*, and *I* are substitutions, deletions, and insertions
respectively, computed via the `evaluate` library's WER
implementation.

**WER: [TBD]**

> **Note:** Final evaluation numbers will be populated from
> `models/recognition/eval_results.json` upon completion of the full
> training run.

**Table IV: Evaluation Summary on WLASL-100 Validation Set**

| Metric | Value |
|---|---|
| Top-1 Accuracy | [TBD]% |
| Top-5 Accuracy | [TBD]% |
| Word Error Rate | [TBD] |
| Epochs Trained | [TBD] / 300 |
| Best Epoch | [TBD] |


## C. Comparison with Baselines

We compare our approach against published results on the WLASL-100
benchmark. Table V summarizes the comparison.

**Table V: Comparison with Published WLASL-100 Results**

| Method | Input | Architecture | Top-1 Acc (%) |
|---|---|---|---|
| Pose-TGCN (Li *et al.*, 2020) [2] | Pose keypoints | Temporal GCN | 55.43 |
| SPOTER (Bohacek & Hruz, 2022) [1] | Pose keypoints (normalized) | Transformer encoder + class query | 63.18 |
| **Ours (SPOTER-inspired)** | **27-keypoint normalized pose** | **Transformer encoder + class query** | **[TBD]** |

### Architectural Comparison with SPOTER

Our model shares several design principles with the original SPOTER
architecture [1], while differing in specific implementation choices:

**Similarities:**

- **Signing-space normalization:** Both methods apply the Bohacek &
  Hruz normalization centered on the nose landmark and scaled by the
  head metric.
- **Class query decoding:** Both use a learnable class query token
  decoded via cross-attention over the encoder output, rather than
  mean-pooling or a prepended [CLS] token.
- **Keypoint-only input:** Both operate exclusively on skeletal
  keypoint features, discarding appearance information.

**Differences:**

| Aspect | SPOTER (original) | Ours |
|---|---|---|
| Keypoint graph | Full upper-body + hands (variable) | 27-node reduced graph |
| *d*_model | 256 | 128 |
| Encoder layers | 6 | 3 |
| Attention heads | 8 | 4 |
| Temporal sampling | Interpolation to fixed length | Uniform sampling (32 frames) |
| Positional encoding | Learned | Sinusoidal |
| Parameters | ~8M | ~2M |

The reduced model size (approximately 4× fewer parameters) is a
deliberate design choice motivated by the relatively small size of the
WLASL-100 training set (~2,000 videos). Smaller models are less
prone to overfitting in low-data regimes and are more suitable for
deployment in resource-constrained settings—a key consideration for
the cross-lingual ASL-to-ISL translation pipeline.


## D. Ablation Studies

To quantify the contribution of individual design decisions, we plan
the following ablation experiments. All ablations will be conducted on
the WLASL-100 validation set with identical training hyperparameters
unless otherwise noted.

### Ablation 1: Effect of Signing-Space Normalization

**Table VI: Impact of Signing-Space Normalization**

| Normalization | Top-1 Acc (%) | Δ |
|---|---|---|
| None (raw MediaPipe coordinates) | [TBD] | — |
| Bohacek & Hruz (proposed) | [TBD] | [TBD] |

**Hypothesis:** Signing-space normalization will provide a substantial
accuracy improvement by removing signer-specific positional and scale
biases, enabling the model to focus on the relative configuration of
keypoints rather than their absolute positions.

### Ablation 2: Keypoint Graph Granularity

**Table VII: Impact of Keypoint Reduction**

| Keypoint Set | Nodes | Features/Frame | Top-1 Acc (%) |
|---|---|---|---|
| Full MediaPipe landmarks | 543 | 1,629 | [TBD] |
| Upper-body + hands (no face) | ~75 | ~225 | [TBD] |
| **27-node reduced (proposed)** | **27** | **81** | **[TBD]** |
| Hands only (no pose) | 20 | 60 | [TBD] |

**Hypothesis:** The 27-node reduction will perform comparably to the
full landmark set on WLASL-100, while significantly reducing
computational cost and overfitting risk. Removing pose keypoints
entirely (hands-only) is expected to degrade performance, as many
signs incorporate upper-body movement and orientation.

### Ablation 3: Augmentation Strategy

**Table VIII: Impact of Data Augmentation**

| Augmentation Config | Top-1 Acc (%) |
|---|---|
| No augmentation | [TBD] |
| RandomMirror only | [TBD] |
| RandomMirror + RandomRotation | [TBD] |
| **Full pipeline (Mirror + Rotation + Squeeze)** | **[TBD]** |

**Hypothesis:** Each augmentation component will contribute
incrementally to generalization performance. RandomMirror is expected
to provide the largest single-augmentation gain by effectively
doubling the training set size, while RandomRotation and
RandomSqueeze will provide complementary regularization.

### Ablation 4: Model Capacity

**Table IX: Impact of Model Dimensionality and Depth**

| *d*_model | Layers | Heads | Params (M) | Top-1 Acc (%) |
|---|---|---|---|---|
| 64 | 2 | 2 | ~0.5 | [TBD] |
| **128** | **3** | **4** | **~2** | **[TBD]** |
| 256 | 6 | 8 | ~8 | [TBD] |

**Hypothesis:** The medium-capacity configuration (*d*_model = 128,
3 layers) will achieve the best trade-off between expressiveness and
overfitting on the WLASL-100 dataset. The larger configuration may
overfit given the limited training data, while the smaller
configuration may underfit complex multi-hand signs.

---

### References

[1] M. Bohacek and M. Hruz, "Sign Pose-based Transformer for
Word-level Sign Language Recognition," in *Proc. IEEE/CVF Winter
Conf. on Applications of Computer Vision Workshops (WACVW)*, 2022.

[2] D. Li, C. Rodriguez, X. Yu, and H. Li, "Word-level Deep Sign
Language Recognition from Video: A New Large-scale Dataset and Methods
Comparison," in *Proc. IEEE/CVF Winter Conf. on Applications of
Computer Vision (WACV)*, 2020.

[3] A. Vaswani *et al.*, "Attention Is All You Need," in *Proc. 31st
Int. Conf. on Neural Information Processing Systems (NeurIPS)*, 2017.
