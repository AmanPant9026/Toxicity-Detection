# ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation

> **Paper:** Li, P., Fillies, J., & Paschke, A. (2026). EACL 2026. [[arXiv]](https://arxiv.org/abs/2601.03121)

---

## Abstract

Toxicity detection models struggle with **class imbalance** — in real-world datasets like Jigsaw, toxic comments make up only ~10% of all data. Models trained on such imbalanced data tend to under-detect toxic content, achieving high accuracy by simply predicting "non-toxic" for everything.

ToxiGAN addresses this by **generating synthetic toxic text** through adversarial training, then using these samples to augment the minority toxic class in downstream classifier training. The system uses:

- **K class-specific LSTM generators** that learn distinct toxic language patterns (toxic, obscene, insult, identity hate)
- **A BERT-based multi-class discriminator** that classifies text into K toxic + neutral + fake classes
- **An LLM-based neutral text provider** (via Ollama) that evolves its few-shot examples using discriminator feedback
- **Two-step alternating training** — generators are pushed toward both toxicity (semantic penalty) and authenticity (discriminator penalty)

The generated samples are added to the real training data **without any modification to the original dataset**, letting the natural class imbalance be corrected by augmentation alone.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ToxiGAN Generation Framework                  │
│                                                                  │
│   Generator G₁ ──┐                                               │
│   (toxic)         │                                               │
│   Generator G₂ ──┼──▶ Multi-class Discriminator (BERT)           │
│   (obscene)       │    K toxic + neutral + fake = K+2 classes    │
│   Generator G₃ ──┤                                               │
│   (insult)        │         ▲                                    │
│   Generator G₄ ──┘         │                                    │
│   (identity_hate)    LLM Neutral Provider (Ollama)               │
│                      Few-shot with evolving examples             │
│                                                                  │
│   Adversarial Loop:                                              │
│   Even steps → toxicity penalty (push away from neutral)         │
│   Odd steps  → authenticity penalty (fool the discriminator)     │
└─────────────────────────────────────────────────────────────────┘

                          ▼ Generated samples

┌─────────────────────────────────────────────────────────────────┐
│                   Detection Pipeline                             │
│                                                                  │
│   Real Jigsaw dataset (~160K, ~10% toxic)  ← NO modifications    │
│          +                                                       │
│   Cleaned ToxiGAN samples (all toxic)                            │
│          =                                                       │
│   Augmented training set (higher toxic %)                        │
│                                                                  │
│   Trained on 3 classifiers:                                      │
│   • DistilBERT (transformer, fine-tuned)                         │
│   • BiLSTM (deep learning, from scratch)                         │
│   • TF-IDF + Logistic Regression (classical ML)                  │
│                                                                  │
│   Evaluated:                                                     │
│   • In-domain (Jigsaw test set)                                  │
│   • Cross-dataset (Davidson, HateXplain)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ToxiGAN/
├── generation/                   # GAN-based toxic text generation
│   ├── train.py                  #   full ToxiGAN training pipeline
│   ├── resume_training.py        #   resume adversarial training from checkpoint
│   ├── generate_samples.py       #   generate synthetic toxic samples
│   ├── config.py                 #   configuration & path management
│   ├── generator.py              #   LSTM generator
│   ├── discriminator.py          #   BERT discriminator
│   ├── rollout.py                #   Monte Carlo rollout (REINFORCE)
│   ├── penalty_loss.py           #   semantic similarity penalty
│   ├── llm_neutral_provider.py   #   Ollama neutral text provider
│   ├── dataloader.py             #   token-ID datasets
│   └── utils.py
│
├── detection/                    # Toxicity detection classifiers
│   ├── multi_classifier.py       #   3 classifiers × baseline/augmented + cross-dataset
│   ├── train_classifier.py       #   DistilBERT only (baseline vs augmented)
│   ├── test_cross_dataset.py     #   cross-dataset evaluation
│   ├── model.py                  #   shared DistilBERT classifier
│   ├── dataset.py                #   shared dataset utilities
│   ├── data_cleaning.py          #   generated data filtering
│   └── metrics.py                #   evaluation metrics
│
├── scripts/
│   └── download_data.py          # download Jigsaw per-class files
│
├── configs/default.yaml          # centralized hyperparameters
├── data/raw/                     # downloaded .txt files (per class)
├── artifacts/                    # vocab, .id files, checkpoints, generated JSON
├── outputs/                      # classifier outputs & reports
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA GPU (recommended)
- [Ollama](https://ollama.ai/)

### Installation

```bash
conda create -n toxigan python=3.11 -y
conda activate toxigan

# PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

### Ollama Setup

```bash
# In a separate terminal
ollama serve

# Pull the model
ollama pull qwen2.5:14b-instruct
export OLLAMA_MODEL=qwen2.5:14b-instruct

# Verify
python generation/llm_neutral_provider.py
```

---

## End-to-End Pipeline

### Phase 1: Data Download

```bash
# Per-class .txt files for ToxiGAN generation
cd scripts && python download_data.py && cd ..
```

### Phase 2: Train ToxiGAN

```bash
cd generation
python config.py                    # verify paths
python train.py                     # full training (several hours)

# If you need to stop and resume later:
# Ctrl+C, then:
python resume_training.py --start_batch <last_batch> --total_batches 25
```

### Phase 3: Generate Samples

```bash
python generate_samples.py          # → artifacts/data_gen_toxigan.json
```

### Phase 4: Train & Evaluate Classifiers

```bash
cd detection

# Full analysis: 3 classifiers × baseline/augmented + cross-dataset
python multi_classifier.py --generated_data ../artifacts/data_gen_toxigan.json

# Quick test first (~5 min)
python multi_classifier.py --generated_data ../artifacts/data_gen_toxigan.json --quick_test
```

---

## Experimental Design

### Why Augmentation Helps

The Jigsaw dataset has natural class imbalance:

| | Non-toxic | Toxic | Toxic % |
|---|---|---|---|
| **Original dataset** | ~143,000 | ~16,000 | **~10%** |
| **+ ToxiGAN (40K raw → ~25K cleaned)** | ~143,000 | ~41,000 | **~22%** |

The original data is modified in **no way**. ToxiGAN-generated samples are only added to the toxic class to reduce the imbalance ratio from ~9:1 to ~3.5:1.

### Classifiers

| Classifier | Type | Why Include |
|---|---|---|
| **DistilBERT** | Fine-tuned transformer | Strongest model, industry standard |
| **BiLSTM** | Deep learning (from scratch) | Shows raw augmentation effect without pretraining |
| **TF-IDF + LR** | Classical ML | Fast baseline, tests if augmentation helps even simple models |

### Evaluation

- **In-domain:** Train on Jigsaw, test on Jigsaw held-out set
- **Cross-dataset:** Train on Jigsaw, test on Davidson / HateXplain (completely unseen data from different sources)

---

## Results

### In-Domain: Jigsaw Test Set

| Classifier | Setting | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| DistilBERT | Baseline | 0.9430 | 0.9266 | 0.9623 | 0.9441 | 0.9871 |
| DistilBERT | **Augmented** | **0.9453** | **0.9419** | 0.9492 | **0.9456** | **0.9872** |
| | Δ | +0.002 | +0.015 | −0.013 | **+0.001** | +0.000 |
| BiLSTM | Baseline | 0.8636 | 0.9347 | 0.7821 | 0.8516 | 0.9454 |
| BiLSTM | **Augmented** | **0.8644** | 0.8656 | **0.8630** | **0.8643** | 0.9407 |
| | Δ | +0.001 | −0.069 | +0.081 | **+0.013** | −0.005 |

### Cross-Dataset: Davidson Hate Speech

| Classifier | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **DistilBERT** | **0.8711** | 0.9088 | **0.9394** | **0.9238** | **0.9223** |
| TF-IDF+LR | 0.8259 | **0.9136** | 0.8734 | 0.8930 | 0.8462 |
| BiLSTM | 0.7803 | 0.9035 | 0.8239 | 0.8619 | 0.7939 |

### Key Findings

1. **BiLSTM benefits most from augmentation** (+1.27% F1). Recall jumped from 78.2% to 86.3% — the model catches significantly more toxic content. Weaker models that train from scratch benefit more from additional data.

2. **DistilBERT shows consistent improvement** (+0.14% F1). Precision improved +1.5% (fewer false alarms) while maintaining strong recall. The pretrained model already knows language well, so augmentation provides incremental gains.

3. **Cross-dataset transfer works** — DistilBERT achieves 0.92 F1 on Davidson (Twitter hate speech) despite training only on Jigsaw (Wikipedia comments). This shows the learned toxicity patterns generalize across platforms.

4. **Augmentation helps most where it's needed** — on the imbalanced real dataset, adding synthetic toxic samples corrects the 9:1 ratio and lets classifiers learn the minority class better.



> **Dual-Use Warning:** This repository generates toxic text for research purposes (data augmentation, classifier robustness). Do not use to create, disseminate, or target harmful content.