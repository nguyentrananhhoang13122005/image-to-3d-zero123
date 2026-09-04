# Training Metrics Report

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | Zero123++ v1.2 |
| Fine-tuning | LoRA (rank=8, alpha=1.0) |
| Trainable Params | ~1.3M (0.15% of 860M) |
| Dataset | 600 Objaverse chairs |
| Train / Val / Test | 480 / 60 / 60 |
| Epochs | 15 |
| Batch Size | 2 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| GPU | NVIDIA A10 (24GB) |
| Training Time | ~2.5 hours |

## Training Progress

| Epoch | Train Loss | Val Loss | Notes |
|-------|-----------|----------|-------|
| 1  | 0.0456 | 0.0412 |  |
| 2  | 0.0389 | 0.0356 |  |
| 3  | 0.0342 | 0.0318 |  |
| 4  | 0.0312 | 0.0289 |  |
| 5  | 0.0287 | 0.0264 | Checkpoint |
| 6  | 0.0265 | 0.0245 |  |
| 7  | 0.0248 | 0.0231 |  |
| 8  | 0.0234 | 0.0219 |  |
| 9  | 0.0221 | 0.0208 |  |
| 10 | 0.0210 | 0.0199 | Checkpoint |
| 11 | 0.0201 | **0.0194** | Best model |
| 12 | 0.0193 | 0.0198 | Overfitting |
| 13 | 0.0186 | 0.0202 |  |
| 14 | 0.0180 | 0.0205 |  |
| 15 | 0.0175 | 0.0209 | Final checkpoint |

## Test Results

| Metric | Value |
|--------|-------|
| Test Loss (MSE) | 0.0293 |
