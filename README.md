# HPU-Net
I reproduce a result from the paper "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities"

python train_hpu.py \
    --config train_hpu_lidc.json \
    --project-root . \
    --data-root data/lidc_crops \
    --outdir runs/hpu \
    --max-steps 240000



  {
  "seed": 42,
  "total_steps": 240000,
  "batch_size": 32,
  "lr": 0.0001,
  "lr_milestones": [60000, 120000, 180000, 210000],
  "lr_gamma": 0.5,
  "optimizer": "adam",
  "weight_decay": 0.00001,
  "use_topk": true,
  "k_frac": 0.02,
  "recon_strategy": "random",
  "geco": { 
    "kappa": 0.05, 
    "alpha": 0.99, 
    "lambda_init": 1.0, 
    "step_size": 0.01 
  },
  "eval_every_steps": 5000,
  "ckpt_every_steps": 10000,
  "num_workers": 4,
  "augment": true
}

python evaluate_hpu.py \
    --checkpoint runs/hpu/hpu_last.pth \
    --config train_hpu_lidc.json \
    --data-root data/lidc_crops \
    --output-dir hpu_evaluation_results \
    --num-examples 5 \
    --num-samples 24







##################################################


python train_spu.py \
    --config train_spu_lidc.json \
    --project-root . \
    --data-root data/lidc_crops \
    --outdir runs/spu \
    --max-steps 240000



{
  "seed": 42,
  "total_steps": 240000,
  "batch_size": 32,
  "lr": 0.00005,
  "lr_milestones": [48000, 96000, 144000, 192000, 216000],
  "lr_gamma": 0.5,
  "optimizer": "adam",
  "weight_decay": 0.00001,
  "beta": 1.0,
  "use_topk": false,
  "recon_strategy": "random",
  "eval_every_steps": 5000,
  "ckpt_every_steps": 10000,
  "num_workers": 4,
  "augment": true,
  "pos_weight": "auto",
  "pos_weight_clip": 20.0
}    



python evaluate_spu.py \
    --checkpoint runs/spu/spu_last.pth \
    --config train_spu_lidc.json \
    --data-root data/lidc_crops \
    --output-dir spu_evaluation_results \
    --num-examples 5 \
    --num-samples 24



  Key Model Differences
ComponentHPUNetsPUNetArchitecture8-scale ResUNet, 8 hierarchical latents5-scale standard U-Net, 6 global latentsLossGECO (κ=0.05) + Stochastic top-kELBO (β=1.0) + Standard BCELearning Rate1×10⁻⁴ → 0.5×10⁻⁵ (4 steps)0.5×10⁻⁵ → 1×10⁻⁶ (5 steps)Prior NetworkUses U-Net featuresSeparate network