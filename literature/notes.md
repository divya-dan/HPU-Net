# Reproducing Figure 2 from Hierarchical Probabilistic U-Net Paper - Implementation Details

## Data Section

### Dataset Source
- Uses **preprocessed LIDC-IDRI data** from Google Cloud bucket
- Data format: **180×180 pixel crops** (already preprocessed, not raw DICOM)
- Final processing: **180×180 → augment → random crop to 128×128**

### Data Structure
```
data/lidc_crops/
├── train.csv    # 8882 images
├── test.csv     # 1992 images  
└── val.csv      # 1996 images (if using validation)
```

### Data Loading Implementation
- **4 grader masks per image** `[B,4,H,W]` - some graders may have empty masks
- **Target selection strategy**: `"random"` - randomly pick one grader per training sample
- **Pad mask tracking**: Handles pixels that go out-of-bounds during augmentation

### Augmentation Pipeline (PyTorch Implementation)

**Key Difference**: Uses **PyTorch augmentation** instead of TensorFlow DeepMind's `multidim-image-augmentation` library

**Augmentation Chain**:
1. **RandomIntensity**: 
   - Brightness range: (0.95, 1.05), probability: 0.3
   - Contrast range: (0.95, 1.05), probability: 0.3
   
2. **RandomGaussianNoise**:
   - Noise std: 0.005 (for [0,1] normalized images), probability: 0.2
   
3. **RandomAffine2D**:
   - Rotation: ±15°
   - Translation: ±11px (scaled for 180×180 input)
   - Scale: (0.95, 1.05)  
   - Shear: ±10° (x-shear only)
   - Probability: 1.0 (always applied)
   
4. **RandomElastic2D**:
   - Alpha range: (11.0, 17.0) (scaled for 180×180)
   - Sigma: 8.5 (scaled for 180×180)
   - Probability: 0.5

**Parameter Scaling Logic**: All spatial parameters scaled by factor `180/128 ≈ 1.4` since augmentation applied to 180×180 before final crop to 128×128.

## Losses Section

### HPUNet Loss Function
**GECO Objective** (not standard ELBO):
```
L_GECO = KL_sum + λ(t) * (reconstruction_loss - κ * valid_pixels_mean)
```

**Components**:
- **Reconstruction Loss**: **Stochastic top-k BCE** with Gumbel-Softmax sampling
  - `k_frac = 0.02` (2% worst pixels)  
  - Uses `masked_stochastic_topk_bce_with_logits()` function
  - Stochastic sampling prevents overfitting to specific "bad" pixels
  
- **KL Divergence**: Summed over **all 8 latent scales**
  - `KL_sum = Σ_{i=1}^8 KL(q_i || p_i)`
  
- **GECO Parameters**:
  - `κ (kappa) = 0.05` - reconstruction constraint target
  - `α (alpha) = 0.99` - EMA decay for moving average
  - `step_size = 0.01` - Lagrange multiplier update rate
  - `λ_init = 1.0` - initial Lagrange multiplier

### sPUNet Loss Function
**Standard ELBO**:
```
L_ELBO = reconstruction_loss + β * KL_divergence
```

**Components**:
- **Reconstruction Loss**: **Standard BCE** (no top-k sampling)
  - Uses `masked_bce_with_logits()` function
  - Averaged over valid pixels only
  
- **KL Divergence**: Between **6 global latent variables**
  - `KL = KL(q(z|x,y) || p(z|x))`
  
- **ELBO Parameters**:
  - `β = 1.0` (standard setting)

## Models Section

### HPUNet (Hierarchical Probabilistic U-Net)

**Architecture Overview**:
- **8-scale ResUNet** backbone with **8 hierarchical latent scales**
- **Processing scales**: 128→64→32→16→8→4→2→1 (global 1×1 bottom)
- **Base channels**: 24 (paper specification for LIDC)
- **Channel progression**: (24, 48, 96, 192, 192, 192, 192, 192) - capped at 192

**Key Components**:
- **Encoder/Decoder**: Uses **3 pre-activated residual blocks** per scale
- **Latent Variables**: **Scalar latents** (`z_ch=1`) at each of 8 decoder scales  
- **Prior Network**: Uses **U-Net internal features** (no separate prior network)
- **Latent Injection**: Hierarchical - each scale conditions the next via `InjectLatent` modules

**Model Initialization**:
```python
model = HPUNet(in_ch=1, base=24, z_ch=1, n_blocks=3)
```

**Critical Implementation Details**:
- **Weight Initialization**: Orthogonal (gain=1.0) + truncated normal biases (std=0.001)
- **Hierarchical Sampling**: Each latent scale depends on previous scales
- **Forward Modes**:
  - Training: `sample_posterior=True` (condition on target mask)
  - Inference: `sample_posterior=False` (sample from prior)

### sPUNet (Standard Probabilistic U-Net)

**Architecture Overview**:
- **5-scale standard U-Net** with **6 global latent variables**
- **Processing scales**: 128→64→32→16→8 (no 1×1 global bottom)
- **Base channels**: 32 (standard sPUNet specification)  
- **Channel progression**: (32, 64, 128, 192, 192) - capped at 192

**Key Components**:
- **Encoder/Decoder**: Uses **standard 3×3 conv blocks** (NOT ResNet blocks)
- **Latent Variables**: **6 global latents** (not hierarchical)
- **Prior Network**: **Separate encoder** that mirrors main U-Net encoder
- **Posterior Network**: **Separate encoder** taking image+mask input
- **Combiner Network**: **3 final 1×1 convolutions** to merge features with latents

**Model Initialization**:
```python
model = sPUNet(in_ch=1, base=32, z_dim=6)
```

**Critical Implementation Details**:
- **Weight Initialization**: Orthogonal (gain=1.0) + truncated normal biases (std=0.001)
- **Separate Networks**: Prior p(z|x) and posterior q(z|x,y) use dedicated encoders
- **Global Latents**: Single latent vector broadcast to all spatial locations

## Training Parameters

### HPUNet Training Configuration

**Core Parameters**:
- **Total Steps**: 240,000 iterations
- **Batch Size**: 32
- **Learning Rate Schedule**: 1×10⁻⁴ → 0.5×10⁻⁵ in **4 steps**
- **LR Milestones**: [60000, 120000, 180000, 210000] 
- **LR Gamma**: 0.5 (halves LR at each milestone)
- **Weight Decay**: 1×10⁻⁵

**GECO-Specific Parameters**:
```json
"geco": {
  "kappa": 0.05,         // Reconstruction constraint target
  "alpha": 0.99,         // EMA decay  
  "lambda_init": 1.0,    // Initial Lagrange multiplier
  "step_size": 0.01      // Multiplier update rate
}
```

**Loss Configuration**:
- **Top-k Loss**: `use_topk: true`, `k_frac: 0.02`
- **Stochastic Sampling**: `stochastic_topk: true`

### sPUNet Training Configuration

**Core Parameters**:
- **Total Steps**: 240,000 iterations  
- **Batch Size**: 32
- **Learning Rate Schedule**: 0.5×10⁻⁵ → 1×10⁻⁶ in **5 steps**
- **LR Milestones**: [48000, 96000, 144000, 192000, 216000]
- **LR Gamma**: 0.5 (halves LR at each milestone)
- **Weight Decay**: 1×10⁻⁵

**ELBO-Specific Parameters**:
- **Beta**: `β = 1.0` (standard ELBO weighting)
- **No Top-k**: `use_topk: false`

### Common Training Settings
- **Optimizer**: Adam
- **Random Seed**: 42
- **Augmentation**: Enabled (`augment: true`)
- **Target Strategy**: Random grader selection (`recon_strategy: "random"`)
- **Checkpointing**: Every 10,000 steps
- **Evaluation**: Every 5,000 steps (placeholder)

## Evaluation Section

### Evaluation Methodology
**Purpose**: Generate **Figure 2-style visualizations** comparing model outputs to ground truth

**Evaluation Process**:
1. **Load trained model** from checkpoint
2. **Load test dataset** (1,992 examples)  
3. **Generate reconstructions** for each of 4 graders (posterior mode)
4. **Generate samples** from prior (unconditional sampling)
5. **Create visualization grids**

### Visualization Formats

**Style 1 - Large Grid** (`example_N_style1.png`):
- 6×8 grid showing:
  - Row 1: CT scan + 4 grader annotations + 3 empty
  - Row 2: 4 posterior reconstructions + 4 empty  
  - Rows 3-6: 24 prior samples (8 per row)

**Style 2 - Compact Grid** (`example_N_style2.png`):
- 3×6 grid showing:
  - Row 1: CT scan + 4 grader annotations  
  - Row 2: 4 posterior reconstructions
  - Row 3: 6 prior samples

### Model Inference Details

**Posterior Reconstruction** (conditioned on grader):
```python
logits, _ = model(x=image, y_target=grader_mask, sample_posterior=True)
```

**Prior Sampling** (unconditional):
```python  
logits, _ = model(x=image, y_target=None, sample_posterior=False)
```

**Post-processing**:
- Apply sigmoid to logits: `σ(logits)`
- Threshold at 0.5 for binary masks
- Convert to numpy arrays for visualization

### Expected Results

**HPUNet** (hierarchical latents):
- **Finer detail preservation**  
- **Better structural coherence**
- **More realistic segmentation boundaries**

**sPUNet** (global latents):
- **Coarser, more "blobby" outputs**
- **Less detailed structure**  
- **Baseline comparison for showing HPUNet improvements**

**Key Evaluation Commands**:

```bash
# HPUNet evaluation
python evaluate_hpu.py --checkpoint runs/hpu/hpu_last.pth \
    --config train_hpu_lidc.json --data-root data/lidc_crops \
    --output-dir hpu_evaluation_results --num-examples 5 --num-samples 24

# sPUNet evaluation  
python evaluate_spu.py --checkpoint runs/spu/spu_last.pth \
    --config train_spu_lidc.json --data-root data/lidc_crops \
    --output-dir spu_evaluation_results --num-examples 5 --num-samples 24
```

Both evaluation scripts generate identical visualization formats, enabling direct comparison of the two approaches' ability to capture multi-grader annotation ambiguity in lung lesion segmentation.