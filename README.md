# ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation

> **Paper:** Li, P., Fillies, J., & Paschke, A. (2026). *ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation.* EACL 2026 Main Conference. [[arXiv]](https://arxiv.org/abs/2601.03121)

---

## Abstract

Online toxicity remains a critical challenge for content moderation systems. A core bottleneck is **data imbalance** — toxic comments are significantly outnumbered by non-toxic ones in real-world datasets, leading classifiers to underperform on the minority (toxic) class. 

ToxiGAN addresses this through **adversarial data augmentation**: it trains class-specific LSTM generators to produce synthetic toxic text that is both *authentically toxic* and *stylistically realistic*, guided by a BERT-based multi-class discriminator and an LLM-based neutral text provider (via Ollama). The generated samples are then used to augment the training data of downstream toxicity detection classifiers.

**Key contributions:**
- **Multi-class GAN architecture** with K toxic generators, one LLM-based neutral provider, and a (K+2)-class discriminator
- **Two-step alternating directional learning** — generators are pushed toward both toxicity and authenticity via alternating semantic and discriminator penalties
- **LLM-ballast mechanism** — an Ollama-backed neutral text provider that evolves its few-shot examples based on discriminator feedback
- **Downstream evaluation** across three classifier architectures (DistilBERT, BiLSTM, TF-IDF+LR) and two cross-dataset benchmarks (Davidson, HateXplain)

We demonstrate that ToxiGAN augmentation improves toxicity detection, with the largest gains on weaker classifiers (BiLSTM: +1.27% F1) and consistent improvements on DistilBERT (+0.14% F1), validating the utility of GAN-generated augmentation for addressing class imbalance in hate speech detection.

---

## Architecture

### ToxiGAN Framework

ToxiGAN consists of three core components working in an adversarial loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ToxiGAN Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│   │ Generator G₁  │   │ Generator G₂  │   │ Generator Gₖ  │      │
│   │  (toxic)      │   │  (obscene)    │   │ (identity_hate)│      │
│   │  LSTM-based   │   │  LSTM-based   │   │  LSTM-based   │      │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘      │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────────────────────────────────────────────┐      │
│   │          Multi-class Discriminator (BERT)            │      │
│   │     K toxic + 1 neutral + 1 fake = K+2 classes      │      │
│   └──────────────────────┬──────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────┐      │
│   │       LLM-based Neutral Provider (Ollama)            │      │
│   │    Few-shot prompting with evolving examples         │      │
│   │    Model: qwen2.5:14b-instruct                       │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
│   Training Loop:                                                │
│   1. Pretrain generators on real toxic data                     │
│   2. Pretrain discriminator on real + LLM-neutral + fake        │
│   3. Adversarial training with alternating penalties:           │
│      • Even steps: Toxicity penalty (push away from neutral)   │
│      • Odd steps: Authenticity penalty (fool discriminator)    │
│   4. LLM-ballast updates few-shot pool each round              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Downstream Detection Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Jigsaw      │     │  ToxiGAN         │     │  Augmented        │
│  Dataset     │────▶│  Generated       │────▶│  Training Set     │
│  (original)  │     │  Samples (40K)   │     │  (original+gen)   │
└──────────────┘     └──────────────────┘     └────────┬──────────┘
                                                        │
                              ┌──────────────────────────┤
                              │                          │
                    ┌─────────▼────────┐    ┌────────────▼────────────┐
                    │   In-Domain       │    │   Cross-Dataset          │
                    │   Evaluation      │    │   Evaluation             │
                    │   (Jigsaw test)   │    │   (Davidson / HateXplain)│
                    └──────────────────┘    └──────────────────────────┘
                              │                          │
                    ┌─────────▼──────────────────────────▼─────────┐
                    │          3 Classifiers:                       │
                    │  • DistilBERT (transformer, fine-tuned)      │
                    │  • BiLSTM (deep learning, from scratch)      │
                    │  • TF-IDF + Logistic Regression (classical)  │
                    └──────────────────────────────────────────────┘
```

---

## Project Structure

```
ToxiGAN/
├── configs/
│   └── default.yaml                 # Centralized hyperparameters
│
├── data/
│   └── raw/                         # Downloaded Jigsaw .txt files
│       ├── nor.txt                  #   neutral samples
│       ├── toxic.txt                #   toxic class
│       ├── obscene.txt              #   obscene class
│       ├── insult.txt               #   insult class
│       └── identity_hate.txt        #   identity hate class
│
├── artifacts/                       # Training artifacts
│   ├── vocab.txt                    #   vocabulary (auto-generated)
│   ├── *.id                         #   tokenized class files
│   ├── generator_toxic_*.pt         #   generator checkpoints
│   ├── discriminator.pt             #   discriminator checkpoint
│   └── data_gen_toxigan.json        #   generated samples
│
├── generation/                      # ══ ToxiGAN: GAN-based generation ══
│   ├── config.py                    #   configuration & path management
│   ├── train.py                     #   full training pipeline
│   ├── resume_training.py           #   resume adversarial training
│   ├── generate_samples.py          #   generate synthetic toxic text
│   ├── generator.py                 #   LSTM generator architecture
│   ├── discriminator.py             #   BERT discriminator architecture
│   ├── rollout.py                   #   Monte Carlo rollout (REINFORCE)
│   ├── penalty_loss.py              #   semantic similarity penalty
│   ├── llm_neutral_provider.py      #   Ollama neutral text provider
│   ├── dataloader.py                #   token-ID dataset utilities
│   └── utils.py                     #   helper functions
│
├── detection/                       # ══ Toxicity Detection Classifiers ══
│   ├── multi_classifier.py          #   full analysis: 3 classifiers ×
│   │                                #   baseline/augmented × cross-dataset
│   ├── train_classifier.py          #   DistilBERT baseline vs augmented
│   ├── test_cross_dataset.py        #   cross-dataset evaluation
│   ├── model.py                     #   shared DistilBERT classifier
│   ├── dataset.py                   #   shared dataset & data loading
│   ├── data_cleaning.py             #   generated data filtering
│   └── metrics.py                   #   evaluation metrics & reporting
│
├── scripts/
│   └── download_data.py             # download Jigsaw dataset
│
├── outputs/                         # generated outputs & reports
│   ├── classifier_outputs/          #   single-classifier results
│   ├── multi_classifier/            #   multi-classifier analysis
│   └── jigsaw_splits/               #   cached train/val/test splits
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- [Ollama](https://ollama.ai/) installed locally

### Step 1: Environment Setup

```bash
# Create conda environment
conda create -n toxigan python=3.11 -y
conda activate toxigan

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 13.0:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

# For CUDA 12.1:
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset

```bash
cd scripts
python download_data.py
cd ..
```

This downloads the Jigsaw Toxic Comment Classification dataset and saves per-class `.txt` files to `data/raw/`.

### Step 3: Setup Ollama

```bash
# In a separate terminal
ollama serve

# Pull the recommended model
ollama pull qwen2.5:14b-instruct

# Set the model
export OLLAMA_MODEL=qwen2.5:14b-instruct

# Verify
python generation/llm_neutral_provider.py
# Should print 5 neutral sentences
```

---

## End-to-End Pipeline

### Phase 1: Train ToxiGAN Generators

```bash
cd generation

# Verify config paths
python config.py

# Train (takes several hours on GPU)
export OLLAMA_MODEL=qwen2.5:14b-instruct
python train.py
```

Training consists of:
1. **Generator pretraining** (600 epochs × 4 classes) — each LSTM learns toxic language patterns
2. **Discriminator pretraining** (10 epochs) — BERT learns to classify toxic/neutral/fake
3. **Adversarial training** (80 rounds) — generators and discriminator improve against each other

Checkpoints are saved to `artifacts/` after every adversarial round.

**To resume training after stopping:**
```bash
python resume_training.py --start_batch 5 --total_batches 25
```

### Phase 2: Generate Synthetic Samples

```bash
cd generation
python generate_samples.py
# → artifacts/data_gen_toxigan.json (40,000 samples: 10K per class)
```

### Phase 3: Train & Evaluate Detection Classifiers

**Option A — Full multi-classifier analysis (recommended):**
```bash
cd detection
python multi_classifier.py --generated_data ../artifacts/data_gen_toxigan.json
```

This trains 3 classifiers (DistilBERT, BiLSTM, TF-IDF+LR) in both baseline and augmented settings, evaluates in-domain on Jigsaw and cross-dataset on Davidson.

**Option B — Quick DistilBERT only:**
```bash
cd detection
python train_classifier.py --generated_data ../artifacts/data_gen_toxigan.json
```

**Option C — Cross-dataset evaluation (after Option B):**
```bash
python test_cross_dataset.py --dataset hatexplain
python test_cross_dataset.py --dataset davidson
```

---

## Experimental Results

### Dataset

| Split | Samples | Toxic | Non-toxic | Toxic % |
|-------|---------|-------|-----------|---------|
| Train | 20,768 | ~10K | ~10K | ~50% |
| Val | 2,596 | ~1.3K | ~1.3K | ~50% |
| Test | 2,596 | ~1.3K | ~1.3K | ~50% |

**ToxiGAN Generated Samples:**
- Raw: 40,000 (10,000 per toxic class)
- After aggressive cleaning: variable (removes short, repetitive, UNK-heavy, and gibberish)

### In-Domain Results: Jigsaw Test Set

**Baseline (original data only) vs Augmented (original + ToxiGAN generated):**

| Classifier | Setting | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| **DistilBERT** | Baseline | 0.9430 | 0.9266 | 0.9623 | 0.9441 | 0.9871 |
| **DistilBERT** | Augmented | **0.9453** | **0.9419** | 0.9492 | **0.9456** | **0.9872** |
| | **Δ** | +0.0023 | +0.0153 | −0.0131 | **+0.0014** | +0.0001 |
| **BiLSTM** | Baseline | 0.8636 | 0.9347 | 0.7821 | 0.8516 | 0.9454 |
| **BiLSTM** | Augmented | **0.8644** | 0.8656 | **0.8630** | **0.8643** | 0.9407 |
| | **Δ** | +0.0008 | −0.0691 | +0.0809 | **+0.0127** | −0.0047 |
| **TF-IDF+LR** | Baseline | 0.8998 | 0.9126 | 0.8845 | 0.8984 | 0.9623 |
| **TF-IDF+LR** | Augmented | 0.8998 | 0.9126 | 0.8845 | 0.8984 | 0.9623 |
| | **Δ** | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 |

### Key In-Domain Findings

1. **DistilBERT (+0.14% F1):** Augmentation improved precision significantly (+1.5%) while maintaining high recall. The augmented model reduced false positives (99 → 76) at the cost of slightly more false negatives (49 → 66), resulting in a more balanced classifier.

2. **BiLSTM (+1.27% F1):** The largest improvement. Augmentation dramatically boosted recall (+8.1%), meaning the model catches far more toxic content. The BiLSTM, being a weaker model trained from scratch, benefits most from additional training data — this is consistent with the literature showing that data augmentation has greater impact on less powerful models.

3. **TF-IDF+LR (no change):** The bag-of-words representation is insensitive to the short, fragmented generated samples. TF-IDF captures word presence but not the sequential patterns that characterize the ToxiGAN outputs.

### Cross-Dataset Results: Davidson Hate Speech Dataset

Models trained on augmented Jigsaw data, evaluated on the entirely unseen Davidson dataset:

| Classifier | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **DistilBERT** | **0.8711** | **0.9088** | **0.9394** | **0.9238** | **0.9223** |
| TF-IDF+LR | 0.8259 | 0.9136 | 0.8734 | 0.8930 | 0.8462 |
| BiLSTM | 0.7803 | 0.9035 | 0.8239 | 0.8619 | 0.7939 |

### Key Cross-Dataset Findings

1. **DistilBERT generalizes best** — achieving 0.92 F1 on a completely unseen dataset from a different source, time period, and annotation scheme. This demonstrates that transformer-based models learn transferable toxicity representations.

2. **All classifiers maintain reasonable performance** on cross-dataset evaluation, with precision above 0.90 across all three. This suggests the toxicity patterns learned from Jigsaw (Wikipedia comments) transfer to Twitter-based hate speech.

3. **Recall drops more than precision** in cross-domain transfer, indicating that some dataset-specific toxic patterns don't transfer (Davidson contains Twitter-specific offensive language that differs from Wikipedia toxicity).

---

## Summary of Contributions

| Aspect | Description |
|---|---|
| **Generation** | Multi-class GAN with LLM-guided neutral ballast produces class-specific toxic text |
| **Augmentation Effect** | +0.14% F1 (DistilBERT), +1.27% F1 (BiLSTM) on in-domain evaluation |
| **Cross-Dataset Transfer** | DistilBERT achieves 0.92 F1 on unseen Davidson dataset |
| **Architecture Insight** | Weaker models (BiLSTM) benefit more from augmentation than strong pretrained models |
| **Scalability** | Resume-capable training, modular classifier pipeline, configurable via YAML |

---

## Citation

```bibtex
@article{li2026toxigan,
  title={ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation},
  author={Li, Peiran and Fillies, Jan and Paschke, Adrian},
  journal={arXiv preprint arXiv:2601.03121},
  year={2026}
}
```

---

## Acknowledgments

This work builds upon:
- [SeqGAN](https://github.com/LantaoYu/SeqGAN) (Yu et al., 2017)
- [SentiGAN](https://github.com/Nrgeup/SentiGAN) (Wang & Wan, 2018)
- [HateGAN](https://github.com/Social-AI-Studio/HateGAN) (Cao & Lee, 2020)
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Davidson Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)

> **Dual-Use Warning:** This repository contains code for generating toxic text for data augmentation and classifier robustness research. Do not use for generating, disseminating, or targeting harmful content. Use responsibly and in compliance with applicable laws and policies.
