# HPU-Net Reproduction

This project reproduces Figure 2 from the [Hierarchical Probabilistic U-Net](https://arxiv.org/abs/1905.13077) paper, which compares the performance of [sPU-Net](https://arxiv.org/abs/1806.05034) and HPU-Net models on medical image segmentation with uncertainty modeling. The reproduction uses the LIDC-IDRI lung lesion dataset, available from Kohl et al. at this [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops/).


## Setup

```bash
conda create -n hpu python=3.12
conda activate hpu
pip install -r requirements.txt
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
```

## Data Preparation

Build CSV manifests:
```bash
python src/scripts/build_manifests.py \
  --data-root /path/to/lidc_crops \
  --out /path/to/lidc_crops \
  --project-root /path/to/project \
  --splits train val test
```

## Training

### HPU-Net
```bash
python src/hpunet/train/train_hpu.py \
  --config configs/train_hpu_lidc.json \
  --project-root /path/to/project \
  --data-root /path/to/lidc_crops \
  --outdir /path/to/output \
  --max-steps 240000
```

### sPU-Net
```bash
python src/hpunet/train/train_spu.py \
  --config configs/train_spu_lidc.json \
  --project-root /path/to/project \
  --data-root /path/to/lidc_crops \
  --outdir /path/to/output \
  --max-steps 240000
```

## Evaluation

```bash
jupyter notebook src/hpunet/eval/spu_hpu_eval_compare.ipynb
```

## Results

Reproduction results on LIDC-IDRI dataset comparing our implementation with original paper metrics:

**Case 1 - All samples (including empty masks):**

| Model   | GED²            | IoUrec          | Hungarian       |
|---------|-----------------|-----------------|-----------------|
| sPU-Net | 0.176±0.169     | 0.759±0.153     | 0.465±0.236     |
|         | (paper: 0.32±0.03) | (paper: 0.75±0.04) | (paper: 0.50±0.03) |
| HPU-Net | 0.022±0.024     | 0.958±0.043     | 0.489±0.229     |
|         | (paper: 0.27±0.01) | (paper: 0.97±0.00) | (paper: 0.53±0.01) |

**Case 2 - Lesions only (excluding empty masks):**

| Model   | GED²            | IoUrec          | Hungarian       |
|---------|-----------------|-----------------|-----------------|
| sPU-Net | 0.636±0.560     | 0.563±0.264     | 0.160±0.192     |
| HPU-Net | 0.109±0.211     | 0.919±0.108     | 0.176±0.186     |

<p align="center">
    <img src="./evaluation_results/comparative_row_00000.png" alt="Evaluation comparison of SPUnet and HPUnet" width="500"/>
</p>
<p align="center">
    <em>Evaluation comparison of SPUnet and HPUnet.</em>
</p>

## Acknowledgment


We thank S. A. A. Kohl et al. for their [Probabilistic U-Net](https://arxiv.org/abs/1806.05034) (Kohl et al., 2018) and [Hierarchical Probabilistic U-Net](https://arxiv.org/abs/1905.13077) (Kohl et al., 2019) papers. We acknowledge the LIDC-IDRI dataset (Armato et al., 2011; Clark et al., 2013) and thank Kohl et al. for making the preprocessed dataset publicly available through their [Google Cloud Storage bucket](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops/).