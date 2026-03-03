# How to Run NSCA

A simple guide to train and run the Neuro-Symbolic Cognitive Architecture.

---

## Prerequisites

- **Python 3.9+**
- **PyTorch 2.0+** (with CUDA if using GPU)
- **GPU**: RTX 3050 (6 GB) or better recommended. CPU works but is slow.

---

## 1. Clone and Install

```bash
git clone https://github.com/mahmoudomarus/Neuro-Symbolic-Grounding-in-Low-Resource-Regimes.git
cd Neuro-Symbolic-Grounding-in-Low-Resource-Regimes

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Verify Installation

```bash
python verify_world_model.py
```

You should see `✓` for each component. If any fail, check that PyTorch and torchaudio are installed correctly.

---

## 3. Training

### Option A: Full 5-Layer Training (RTX 3050)

Trains **all layers** of NSCA:

- **Layer 0**: World Model (vision, audio, fusion, temporal)
- **Layer 1**: Property extractors (hardness, weight, etc.)
- **Layer 2**: Causal / physics priors
- **Layer 3**: Drive system (curiosity, competence)
- **Layer 4**: Language grounding (via babbling)

```bash
python scripts/run_full_training.py --config configs/training_config_local.yaml --skip-validation
```

This runs: Layer 0 training → Babbling → Layers 1-4 training.

### Option B: Layer 0 only (World Model)

```bash
python train.py
```

Runs **Vision → Audio → Fusion → Temporal** in one process. Datasets (CIFAR-100, SpeechCommands) download automatically.

### Option C: Layers 1-4 (after Layer 0 is trained)

```bash
python scripts/train_all_layers.py --world-model checkpoints/world_model_final.pth --data-dir /path/to/GreatestHits
```

### Option D: Babbling only (Layer 4 language grounding)

```bash
python scripts/run_babbling.py --random-steps 10000 --competence-steps 40000
```

### Option B: Cloud GPU (A100, etc.)

```bash
python scripts/train_world_model.py --config configs/training_config.yaml --phase all
```

### Option C: Single Phase

To train only one phase:

```bash
# Vision encoder only
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase vision

# Audio encoder only
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase audio

# Fusion + Temporal (requires vision and audio checkpoints from previous phases)
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase fusion --resume checkpoints/audio_encoder_epoch50.pth
```

---

## 4. Data

| Phase   | Data Source                | When Downloaded                    |
|---------|----------------------------|------------------------------------|
| Vision  | CIFAR-100                  | First vision training run          |
| Audio   | Speech Commands v0.02      | First audio training run (via torchaudio) |
| Fusion  | CIFAR + Speech Commands    | First fusion run (fallback)        |
| Fusion* | Greatest Hits (Visually Indicated Sounds) | Only if you use `--data-dir /path/to/vis-data` |

\* Greatest Hits is an aligned video+audio dataset. Without it, the fallback uses CIFAR images as “video” paired with Speech Commands. Results are better with real Greatest Hits.

### Data source links

| Dataset | Source | Link |
|---------|--------|------|
| **CIFAR-100** | HuggingFace | [uoft-cs/cifar100](https://huggingface.co/datasets/uoft-cs/cifar100) |
| **Speech Commands v0.02** | Google (via torchaudio) | [TensorFlow data](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) — torchaudio downloads this automatically |
| **Greatest Hits** | Owens et al. (UMich) | [vis-data.zip](https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip) (40 GB) — extract, then point `--data-dir` to the folder. Expected files: `*_denoised.mp4`, `*_denoised.wav` (rename if your extract uses different names) |

### Manual download: Greatest Hits

1. Download [vis-data.zip](https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip) (40 GB).
2. Extract to a folder (e.g. `./data/vis-data`).
3. Ensure the folder contains `*_denoised.mp4` and `*_denoised.wav` pairs.
4. Run training with: `python train.py --data-dir ./data/vis-data`

### Pre-download data (optional)

```bash
python scripts/download_data.py --local-test
```

Downloads CIFAR-100 + Speech Commands so training doesn’t wait on downloads. Training will also download them if needed.

---

## 5. Outputs

- **Checkpoints**: `checkpoints/`
  - `vision_encoder_epoch{N}.pth` — Vision phase
  - `audio_encoder_epoch{N}.pth` — Audio phase
  - `fusion_best.pth` — Best fusion model
  - `world_model_final.pth` — Layer 0 (world model)
  - `cognitive_agent_full.pth` — Full 5-layer agent (after `train_all_layers.py`)

- **Logs**: `logs/` (including `babbling_results.json`)

---

## 6. Demo (after training)

Once you have a trained checkpoint:

```bash
python scripts/demo_pipeline.py --checkpoint checkpoints/world_model_final.pth --data-dir /path/to/vis-data
```

If you don’t have Greatest Hits, you can still run the demo with any directory containing `*_denoised.mp4` and `*_denoised.wav` video–audio pairs.

---

## 7. Full Pipeline (validation + babbling + training)

To run the complete pipeline including validation and babbling:

```bash
python scripts/run_full_training.py --config configs/training_config_local.yaml
```

To skip validation (only if you’ve passed it before):

```bash
python scripts/run_full_training.py --config configs/training_config_local.yaml --skip-validation
```

---

## 8. Resume Training

If training stops, resume with:

```bash
python train.py --resume checkpoints/vision_encoder_epoch50.pth
```

Or, for a specific phase:

```bash
python scripts/train_world_model.py --config configs/training_config_local.yaml --phase all --resume checkpoints/vision_encoder_epoch100.pth
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory (OOM)** | Use `configs/training_config_local.yaml`. Reduce `batch_size` further in the config if needed. |
| **SpeechCommands download fails** | Ensure `torchaudio` is installed: `pip install torchaudio` |
| **CIFAR download fails** | Uses `uoft-cs/cifar100`. Run `pip install datasets` and try again. |
| **`verify_world_model.py` fails** | Ensure all dependencies are installed. Check for CUDA/GPU compatibility if using GPU. |
| **Fusion phase: “Greatest Hits failed”** | Expected if you don’t have Greatest Hits. The CIFAR+SpeechCommands fallback will be used automatically. |

---

## Summary

```bash
# Full 5-layer training (skip validation for faster start)
python scripts/run_full_training.py --config configs/training_config_local.yaml --skip-validation

# Or Layer 0 only
python train.py
```

That’s it. Datasets download automatically. Full pipeline trains Layer 0 then Layers 1-4.
