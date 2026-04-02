#!/usr/bin/env python3
"""
Multi-Classifier Toxicity Detection — Full Analysis Pipeline
==============================================================

Trains 3 classifiers (DistilBERT, BiLSTM, TF-IDF+LogisticRegression)
in both baseline and augmented settings, then optionally evaluates
cross-dataset on HateXplain and Davidson.

Produces:
  1. Baseline vs Augmented comparison per classifier
  2. Cross-classifier comparison table
  3. Cross-dataset generalization results
  4. Full JSON report

Usage:
    python multi_classifier.py --generated_data ../artifacts/data_gen_toxigan.json
    python multi_classifier.py --quick_test
    python multi_classifier.py --skip_cross_dataset
    python multi_classifier.py --classifiers distilbert bilstm tfidf_lr
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("multi-classifier")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-classifier toxicity detection analysis.")
    p.add_argument("--generated_data", type=str, default="data_gen_toxigan.json")
    p.add_argument("--output_dir", type=str, default="outputs/multi_classifier")
    p.add_argument("--splits_dir", type=str, default="outputs/jigsaw_splits")
    p.add_argument("--classifiers", nargs="+", default=["distilbert", "bilstm", "tfidf_lr"],
                   choices=["distilbert", "bilstm", "tfidf_lr"])
    p.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lstm_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--skip_cross_dataset", action="store_true")
    return p.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def download_jigsaw():
    log.info("Loading Jigsaw dataset...")
    from datasets import load_dataset
    ds = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    df = pd.DataFrame(ds)
    toxic_cols = [c for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] if c in df.columns]
    df["label"] = (df[toxic_cols].sum(axis=1) > 0).astype(int)
    df["text"] = df["comment_text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    dist = df["label"].value_counts().to_dict()
    log.info("Jigsaw: %d samples — Non-toxic: %d, Toxic: %d (%.1f%%)",
             len(df), dist.get(0,0), dist.get(1,0), 100*dist.get(1,0)/len(df))
    return df[["text", "label"]]


def stratified_split(df, seed=42):
    from sklearn.model_selection import train_test_split
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    log.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(f"{path}/train.parquet", index=False)
    val_df.to_parquet(f"{path}/val.parquet", index=False)
    test_df.to_parquet(f"{path}/test.parquet", index=False)

def load_splits(path):
    files = [f"{path}/{s}.parquet" for s in ["train", "val", "test"]]
    if all(os.path.exists(f) for f in files):
        log.info("Loading cached splits from %s/", path)
        return pd.read_parquet(files[0]), pd.read_parquet(files[1]), pd.read_parquet(files[2])
    return None


def load_and_clean_generated(json_path):
    if not Path(json_path).exists():
        log.warning("Generated data not found: %s", json_path)
        return []
    with open(json_path) as f:
        data = json.load(f)
    tweets = data.get("tweet", [])
    log.info("Loaded %d generated samples", len(tweets))
    cleaned = []
    reasons = Counter()
    for text in tweets:
        text = text.strip()
        if not text:
            reasons["empty"] += 1; continue
        words = text.split()
        if len(words) <= 2:
            reasons["too_short"] += 1; continue
        unk = sum(1 for w in words if w.lower() in ("<unk>", "unk"))
        if unk / len(words) > 0.3:
            reasons["high_unk"] += 1; continue
        wc = Counter(w.lower() for w in words)
        if wc.most_common(1)[0][1] / len(words) > 0.5:
            reasons["repetitive"] += 1; continue
        alpha = sum(1 for w in words if re.search(r"[a-zA-Z]", w))
        if alpha / len(words) < 0.5:
            reasons["non_alpha"] += 1; continue
        clean_words = [w for w in words if w.lower() not in ("<unk>", "unk")]
        if len(clean_words) <= 2:
            reasons["short_after_clean"] += 1; continue
        cleaned.append(" ".join(clean_words))
    log.info("Cleaning: %d → %d kept (%.1f%%). Filtered: %s",
             len(tweets), len(cleaned), 100*len(cleaned)/max(len(tweets),1), dict(reasons))
    return cleaned


def load_hatexplain():
    log.info("Loading HateXplain...")
    from datasets import load_dataset
    ds = load_dataset("hatexplain", trust_remote_code=True)
    texts, labels = [], []
    for split in ["train", "validation", "test"]:
        if split not in ds: continue
        for ex in ds[split]:
            if "post_tokens" not in ex: continue
            text = " ".join(ex["post_tokens"])
            if not text.strip(): continue
            if "annotators" in ex:
                al = [a["label"] for a in ex["annotators"]]
                majority = Counter(al).most_common(1)[0][0]
            elif "label" in ex:
                majority = ex["label"]
            else: continue
            labels.append(0 if majority == 0 else 1)
            texts.append(text)
    log.info("HateXplain: %d samples", len(texts))
    return texts, labels


def load_davidson():
    log.info("Loading Davidson...")
    from datasets import load_dataset
    ds = load_dataset("hate_speech_offensive", trust_remote_code=True, split="train")
    texts, labels = [], []
    for ex in ds:
        text = ex.get("tweet", "")
        label = ex.get("class", -1)
        if not text.strip() or label == -1: continue
        labels.append(0 if label == 2 else 1)
        texts.append(text)
    log.info("Davidson: %d samples", len(texts))
    return texts, labels


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try: m["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except: m["auc_roc"] = None
    else:
        m["auc_roc"] = None
    return m


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFIER 1: DistilBERT
# ═══════════════════════════════════════════════════════════════════════════

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts); self.labels = list(labels)
        self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


class DistilBertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h//2),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(h//2, 2))
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


def train_distilbert(train_texts, train_labels, val_texts, val_labels, args, device, name="bert"):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_ds = BertDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = BertDataset(val_texts, val_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    # Class weights
    counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.tensor([total/(2*counts[0]), total/(2*counts[1])], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = DistilBertClassifier(args.bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

    best_val_loss = float("inf"); best_state = None
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        for batch in tqdm(train_loader, desc=f"  [{name}] Epoch {epoch+1}", leave=False):
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        # Validate
        model.eval(); vloss = 0; vpreds = []; vlabels = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids, mask)
                vloss += criterion(logits, labels).item() * labels.size(0)
                vpreds.extend(logits.argmax(1).cpu().numpy())
                vlabels.extend(labels.cpu().numpy())
        vloss /= len(vlabels)
        vacc = np.mean(np.array(vpreds) == np.array(vlabels))
        log.info("  [%s] Epoch %d/%d — val_loss=%.4f val_acc=%.4f (%.1fs)", name, epoch+1, args.epochs, vloss, vacc, time.time()-t0)
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state); model.to(device)
    return model, tokenizer


def predict_distilbert(model, tokenizer, texts, labels, args, device):
    ds = BertDataset(texts, labels, tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    model.eval(); preds = []; probs = []; all_labels = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(preds), np.array(all_labels), np.array(probs)


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFIER 2: BiLSTM with GloVe
# ═══════════════════════════════════════════════════════════════════════════

class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.labels = list(labels); self.max_len = max_len; self.encoded = []
        for t in texts:
            tokens = t.lower().split()[:max_len]
            ids = [vocab.get(w, 1) for w in tokens]  # 1 = UNK
            self.encoded.append(ids)
    def __len__(self): return len(self.encoded)
    def __getitem__(self, idx):
        ids = self.encoded[idx]
        length = len(ids)
        padded = ids + [0] * (self.max_len - len(ids))
        return (torch.tensor(padded, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # Concat forward and backward final hidden states
        hidden = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(hidden)


def build_vocab(texts, min_freq=2, max_vocab=50000):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    log.info("LSTM vocab: %d tokens", len(vocab))
    return vocab


def train_bilstm(train_texts, train_labels, val_texts, val_labels, args, device, name="bilstm"):
    vocab = build_vocab(train_texts)
    max_len = 200
    train_ds = LSTMDataset(train_texts, train_labels, vocab, max_len)
    val_ds = LSTMDataset(val_texts, val_labels, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=0)

    counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.tensor([total/(2*counts[0]), total/(2*counts[1])], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = BiLSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf"); best_state = None
    for epoch in range(args.lstm_epochs):
        model.train(); t0 = time.time()
        for ids, lengths, labels in tqdm(train_loader, desc=f"  [{name}] Epoch {epoch+1}", leave=False):
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), labels)
            loss.backward(); optimizer.step()

        model.eval(); vloss = 0; n = 0
        vpreds = []; vlabels_list = []
        with torch.no_grad():
            for ids, lengths, labels in val_loader:
                ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
                logits = model(ids, lengths)
                vloss += criterion(logits, labels).item() * labels.size(0); n += labels.size(0)
                vpreds.extend(logits.argmax(1).cpu().numpy())
                vlabels_list.extend(labels.cpu().numpy())
        vloss /= n
        vacc = np.mean(np.array(vpreds) == np.array(vlabels_list))
        if (epoch+1) % 2 == 0 or epoch == 0:
            log.info("  [%s] Epoch %d/%d — val_loss=%.4f val_acc=%.4f (%.1fs)", name, epoch+1, args.lstm_epochs, vloss, vacc, time.time()-t0)
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state); model.to(device)
    return model, vocab


def predict_bilstm(model, vocab, texts, labels, device, max_len=200):
    ds = LSTMDataset(texts, labels, vocab, max_len)
    loader = DataLoader(ds, batch_size=64, num_workers=0)
    model.eval(); preds = []; probs = []; all_labels = []
    with torch.no_grad():
        for ids, lengths, labs in loader:
            ids, lengths = ids.to(device), lengths.to(device)
            logits = model(ids, lengths)
            p = torch.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p[:, 1].cpu().numpy())
            all_labels.extend(labs.numpy())
    return np.array(preds), np.array(all_labels), np.array(probs)


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFIER 3: TF-IDF + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════

def train_tfidf_lr(train_texts, train_labels, name="tfidf_lr"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    log.info("  [%s] Training TF-IDF + Logistic Regression...", name)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="liblinear")),
    ])
    pipeline.fit(train_texts, train_labels)
    log.info("  [%s] Training complete.", name)
    return pipeline


def predict_tfidf_lr(pipeline, texts, labels):
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)[:, 1]
    return np.array(preds), np.array(labels), np.array(probs)


# ═══════════════════════════════════════════════════════════════════════════
# PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(results, title):
    """Print a comparison table for baseline vs augmented across classifiers."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Classifier':<15} {'Setting':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-'*90}")

    for clf_name, data in results.items():
        for setting in ["baseline", "augmented"]:
            if setting not in data: continue
            m = data[setting]
            auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "N/A"
            print(f"  {clf_name:<15} {setting:<12} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8}")
        # Delta
        if "baseline" in data and "augmented" in data:
            b, a = data["baseline"], data["augmented"]
            delta_f1 = a["f1"] - b["f1"]
            sign = "+" if delta_f1 >= 0 else ""
            print(f"  {'':<15} {'Δ (aug-base)':<12} {'':>8} {'':>8} {'':>8} {sign}{delta_f1:>7.4f} {'':>8}")
        print(f"{'-'*90}")
    print(f"{'='*90}")


def print_cross_dataset_table(results, title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  {'Classifier':<15} {'Dataset':<15} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-'*90}")
    for key, m in results.items():
        auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "N/A"
        clf, ds = key.split("_on_")
        print(f"  {clf:<15} {ds:<15} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {auc:>8}")
    print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s | Classifiers: %s", device, args.classifiers)

    # ── Load data ─────────────────────────────────────────────────────────
    cached = load_splits(args.splits_dir)
    if cached:
        train_df, val_df, test_df = cached
    else:
        df = download_jigsaw()
        train_df, val_df, test_df = stratified_split(df, args.seed)
        save_splits(train_df, val_df, test_df, args.splits_dir)

    if args.quick_test:
        log.info("QUICK TEST — capping data")
        train_df = train_df.head(5000); val_df = val_df.head(500); test_df = test_df.head(500)

    # ── Clean generated data ──────────────────────────────────────────────
    generated_texts = load_and_clean_generated(args.generated_data)
    aug_df = pd.concat([train_df, pd.DataFrame({"text": generated_texts, "label": [1]*len(generated_texts)})], ignore_index=True)

    log.info("Baseline train: %d (toxic: %d) | Augmented train: %d (toxic: %d)",
             len(train_df), int(train_df["label"].sum()), len(aug_df), int(aug_df["label"].sum()))

    # Extract lists
    train_texts = train_df["text"].tolist(); train_labels = train_df["label"].tolist()
    aug_texts = aug_df["text"].tolist(); aug_labels = aug_df["label"].tolist()
    val_texts = val_df["text"].tolist(); val_labels = val_df["label"].tolist()
    test_texts = test_df["text"].tolist(); test_labels = test_df["label"].tolist()

    all_results = {}
    trained_models = {}  # for cross-dataset

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN AND EVALUATE EACH CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════

    # ── DistilBERT ────────────────────────────────────────────────────────
    if "distilbert" in args.classifiers:
        log.info("\n" + "="*60)
        log.info("DISTILBERT — Baseline")
        log.info("="*60)
        model_b, tok = train_distilbert(train_texts, train_labels, val_texts, val_labels, args, device, "bert-base")
        preds, labs, probs = predict_distilbert(model_b, tok, test_texts, test_labels, args, device)
        base_m = compute_metrics(labs, preds, probs)

        log.info("DISTILBERT — Augmented")
        model_a, _ = train_distilbert(aug_texts, aug_labels, val_texts, val_labels, args, device, "bert-aug")
        preds, labs, probs = predict_distilbert(model_a, tok, test_texts, test_labels, args, device)
        aug_m = compute_metrics(labs, preds, probs)

        all_results["DistilBERT"] = {"baseline": base_m, "augmented": aug_m}
        trained_models["DistilBERT"] = {"model": model_a, "tokenizer": tok, "type": "bert"}
        torch.save(model_b.state_dict(), output_dir / "distilbert_baseline.pt")
        torch.save(model_a.state_dict(), output_dir / "distilbert_augmented.pt")

    # ── BiLSTM ────────────────────────────────────────────────────────────
    if "bilstm" in args.classifiers:
        log.info("\n" + "="*60)
        log.info("BiLSTM — Baseline")
        log.info("="*60)
        model_b, vocab_b = train_bilstm(train_texts, train_labels, val_texts, val_labels, args, device, "lstm-base")
        preds, labs, probs = predict_bilstm(model_b, vocab_b, test_texts, test_labels, device)
        base_m = compute_metrics(labs, preds, probs)

        log.info("BiLSTM — Augmented")
        model_a, vocab_a = train_bilstm(aug_texts, aug_labels, val_texts, val_labels, args, device, "lstm-aug")
        preds, labs, probs = predict_bilstm(model_a, vocab_a, test_texts, test_labels, device)
        aug_m = compute_metrics(labs, preds, probs)

        all_results["BiLSTM"] = {"baseline": base_m, "augmented": aug_m}
        trained_models["BiLSTM"] = {"model": model_a, "vocab": vocab_a, "type": "lstm"}

    # ── TF-IDF + LR ──────────────────────────────────────────────────────
    if "tfidf_lr" in args.classifiers:
        log.info("\n" + "="*60)
        log.info("TF-IDF + LogReg — Baseline")
        log.info("="*60)
        pipe_b = train_tfidf_lr(train_texts, train_labels, "tfidf-base")
        preds, labs, probs = predict_tfidf_lr(pipe_b, test_texts, test_labels)
        base_m = compute_metrics(labs, preds, probs)

        log.info("TF-IDF + LogReg — Augmented")
        pipe_a = train_tfidf_lr(aug_texts, aug_labels, "tfidf-aug")
        preds, labs, probs = predict_tfidf_lr(pipe_a, test_texts, test_labels)
        aug_m = compute_metrics(labs, preds, probs)

        all_results["TF-IDF+LR"] = {"baseline": base_m, "augmented": aug_m}
        trained_models["TF-IDF+LR"] = {"model": pipe_a, "type": "sklearn"}

    # ── Print in-domain results ───────────────────────────────────────────
    print_comparison_table(all_results, "IN-DOMAIN RESULTS: Baseline vs Augmented (Jigsaw Test Set)")

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-DATASET EVALUATION (augmented models only)
    # ══════════════════════════════════════════════════════════════════════

    cross_results = {}
    if not args.skip_cross_dataset:
        for ds_name, loader_fn in [("HateXplain", load_hatexplain), ("Davidson", load_davidson)]:
            try:
                ext_texts, ext_labels = loader_fn()
            except Exception as e:
                log.warning("Skipping %s: %s", ds_name, e); continue

            for clf_name, info in trained_models.items():
                key = f"{clf_name}_on_{ds_name}"
                if info["type"] == "bert":
                    preds, labs, probs = predict_distilbert(info["model"], info["tokenizer"], ext_texts, ext_labels, args, device)
                elif info["type"] == "lstm":
                    preds, labs, probs = predict_bilstm(info["model"], info["vocab"], ext_texts, ext_labels, device)
                elif info["type"] == "sklearn":
                    preds, labs, probs = predict_tfidf_lr(info["model"], ext_texts, ext_labels)
                else:
                    continue
                cross_results[key] = compute_metrics(labs, preds, probs)

        if cross_results:
            print_cross_dataset_table(cross_results, "CROSS-DATASET: Augmented Models on External Datasets")

    # ── Save full report ──────────────────────────────────────────────────
    report = {
        "experiment": "ToxiGAN Multi-Classifier Analysis",
        "classifiers": args.classifiers,
        "generated_samples_cleaned": len(generated_texts),
        "train_baseline": len(train_df),
        "train_augmented": len(aug_df),
        "test_size": len(test_df),
        "in_domain_results": all_results,
        "cross_dataset_results": cross_results,
    }
    report_path = output_dir / "full_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Full report saved → %s", report_path)
    log.info("Done.")


if __name__ == "__main__":
    main()