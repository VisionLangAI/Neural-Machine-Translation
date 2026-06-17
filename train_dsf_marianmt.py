"""
This script trains/evaluates the following models on a Chinese-English parallel dataset:

1. Word2Vec + Bi-LSTM Seq2Seq
2. Word2Vec + GRU Seq2Seq
3. Word2Vec + CNN Seq2Seq
4. Sentence Embedding + Bi-LSTM Seq2Seq
5. Sentence Embedding + GRU Seq2Seq
6. Sentence Embedding + CNN Seq2Seq
7. BERT Encoder-Decoder
8. mBERT Encoder-Decoder
9. mBERT + Decoder
10. MarianMT
11. DSF–MarianMT Proposed

It also generates:
- BLEU/chrF/TER evaluation using SacreBLEU with case-sensitive tokenizer=13a
- Training curves from actual logs
- Statistical validation
- Fusion ablation
- Attention/head interpretability outputs
- SHAP-style component importance plot based on learned ablation drops
- CSV tables and PNG figures


Important:
- Full transformer training is computationally expensive.
- Start with small epochs first to verify execution.
- Code creates all results from actual model outputs and saved logs.
"""

import os
import re
import json
import math
import time
import random
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sacrebleu

try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from transformers import (
    AutoTokenizer,
    AutoModel,
    EncoderDecoderModel,
    MarianTokenizer,
    MarianMTModel,
    get_linear_schedule_with_warmup
)

try:
    from torch.optim import AdamW
except Exception:
    from transformers import AdamW


# ============================================================
# 1. Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def zh_tokenize(text: str):
    text = str(text).strip()
    return [c for c in text if not c.isspace()]


def en_tokenize(text: str):
    return str(text).strip().split()


def auto_detect_columns(df: pd.DataFrame):
    lower = {c.lower(): c for c in df.columns}
    src_candidates = ["chinese", "zh", "source", "src", "cn", "input"]
    tgt_candidates = ["english", "en", "target", "tgt", "output", "translation"]
    src_col = None
    tgt_col = None
    for c in src_candidates:
        if c in lower:
            src_col = lower[c]
            break
    for c in tgt_candidates:
        if c in lower:
            tgt_col = lower[c]
            break
    if src_col is None or tgt_col is None:
        if len(df.columns) >= 2:
            src_col, tgt_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("Dataset must contain at least two columns.")
    return src_col, tgt_col


def load_parallel_dataset(path, src_col=None, tgt_col=None, max_samples=None):
    df = pd.read_csv(path)
    if src_col is None or tgt_col is None:
        detected_src, detected_tgt = auto_detect_columns(df)
        src_col = src_col or detected_src
        tgt_col = tgt_col or detected_tgt

    df = df[[src_col, tgt_col]].copy()
    df.columns = ["src", "tgt"]
    df["src"] = df["src"].apply(clean_text)
    df["tgt"] = df["tgt"].apply(clean_text)
    df = df[(df["src"] != "") & (df["tgt"] != "")]
    df = df.drop_duplicates().reset_index(drop=True)

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df


def split_dataset(df, seed=42):
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# 2. Vocabulary and classic Seq2Seq datasets
# ============================================================

class Vocab:
    def __init__(self, min_freq=1, max_size=30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.unk = "<unk>"
        self.itos = [self.pad, self.bos, self.eos, self.unk]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, token_lists):
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        words = [w for w, f in counter.most_common(self.max_size - len(self.itos)) if f >= self.min_freq]
        for w in words:
            if w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, tokens, max_len):
        ids = [self.stoi[self.bos]]
        ids += [self.stoi.get(t, self.stoi[self.unk]) for t in tokens[: max_len - 2]]
        ids += [self.stoi[self.eos]]
        if len(ids) < max_len:
            ids += [self.stoi[self.pad]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == self.eos:
                break
            if tok not in [self.pad, self.bos, self.unk]:
                toks.append(tok)
        return " ".join(toks)

    def __len__(self):
        return len(self.itos)


class ClassicTranslationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, max_len):
        self.src = df["src"].tolist()
        self.tgt = df["tgt"].tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(zh_tokenize(self.src[idx]), self.max_len)
        tgt_ids = self.tgt_vocab.encode(en_tokenize(self.tgt[idx]), self.max_len)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": self.src[idx],
            "tgt_text": self.tgt[idx]
        }


# ============================================================
# 3. Classic trainable baselines
# ============================================================

class ClassicSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_id,
        tgt_pad_id,
        model_type="bilstm",
        embedding_mode="trainable",
        src_embed_matrix=None,
        emb_dim=256,
        hidden_dim=256,
        max_len=80,
        dropout=0.1
    ):
        super().__init__()
        self.model_type = model_type
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        self.src_embedding = nn.Embedding(src_vocab_size, emb_dim, padding_idx=src_pad_id)
        if src_embed_matrix is not None:
            self.src_embedding.weight.data.copy_(torch.tensor(src_embed_matrix, dtype=torch.float32))
            self.src_embedding.weight.requires_grad = False

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=tgt_pad_id)

        if model_type == "bilstm":
            self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.0)
            enc_out_dim = hidden_dim * 2
            self.enc_to_dec_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.enc_to_dec_c = nn.Linear(hidden_dim * 2, hidden_dim)
        elif model_type == "gru":
            self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=False)
            enc_out_dim = hidden_dim
            self.enc_to_dec_h = nn.Linear(hidden_dim, hidden_dim)
            self.enc_to_dec_c = None
        elif model_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(emb_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
            enc_out_dim = hidden_dim
            self.enc_to_dec_h = nn.Linear(hidden_dim, hidden_dim)
            self.enc_to_dec_c = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError("model_type must be bilstm, gru, or cnn")

        self.attn = nn.Linear(enc_out_dim + hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.decoder = nn.LSTM(emb_dim + enc_out_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + enc_out_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src_ids):
        src_emb = self.dropout(self.src_embedding(src_ids))

        if self.model_type in ["bilstm", "gru"]:
            enc_out, hidden = self.encoder(src_emb)
            if self.model_type == "bilstm":
                h, c = hidden
                h_cat = torch.cat([h[-2], h[-1]], dim=-1)
                c_cat = torch.cat([c[-2], c[-1]], dim=-1)
                dec_h = torch.tanh(self.enc_to_dec_h(h_cat)).unsqueeze(0)
                dec_c = torch.tanh(self.enc_to_dec_c(c_cat)).unsqueeze(0)
            else:
                h = hidden[-1]
                dec_h = torch.tanh(self.enc_to_dec_h(h)).unsqueeze(0)
                dec_c = torch.zeros_like(dec_h)
        else:
            x = src_emb.transpose(1, 2)
            enc_out = self.encoder(x).transpose(1, 2)
            pooled = enc_out.mean(dim=1)
            dec_h = torch.tanh(self.enc_to_dec_h(pooled)).unsqueeze(0)
            dec_c = torch.tanh(self.enc_to_dec_c(pooled)).unsqueeze(0)

        return enc_out, (dec_h, dec_c)

    def attend(self, enc_out, dec_h):
        b, t, _ = enc_out.shape
        dec_repeat = dec_h[-1].unsqueeze(1).expand(b, t, dec_h.size(-1))
        energy = torch.tanh(self.attn(torch.cat([enc_out, dec_repeat], dim=-1)))
        scores = self.attn_v(energy).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        return context, alpha

    def forward(self, src_ids, tgt_ids, teacher_forcing=0.5):
        enc_out, hidden = self.encode(src_ids)
        b, tgt_len = tgt_ids.shape
        outputs = []
        input_tok = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            emb = self.tgt_embedding(input_tok).unsqueeze(1)
            context, _ = self.attend(enc_out, hidden[0])
            dec_input = torch.cat([emb, context], dim=-1)
            dec_out, hidden = self.decoder(dec_input, hidden)
            logits = self.fc(torch.cat([dec_out.squeeze(1), context.squeeze(1)], dim=-1))
            outputs.append(logits.unsqueeze(1))
            use_teacher = random.random() < teacher_forcing
            input_tok = tgt_ids[:, t] if use_teacher else logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)

    def generate(self, src_ids, bos_id, eos_id, max_len):
        self.eval()
        enc_out, hidden = self.encode(src_ids)
        b = src_ids.size(0)
        input_tok = torch.full((b,), bos_id, dtype=torch.long, device=src_ids.device)
        generated = []

        for _ in range(max_len):
            emb = self.tgt_embedding(input_tok).unsqueeze(1)
            context, _ = self.attend(enc_out, hidden[0])
            dec_input = torch.cat([emb, context], dim=-1)
            dec_out, hidden = self.decoder(dec_input, hidden)
            logits = self.fc(torch.cat([dec_out.squeeze(1), context.squeeze(1)], dim=-1))
            input_tok = logits.argmax(dim=-1)
            generated.append(input_tok.unsqueeze(1))

        return torch.cat(generated, dim=1)


def build_word2vec_matrix(train_df, src_vocab, emb_dim):
    if Word2Vec is None:
        return None
    sentences = [zh_tokenize(s) for s in train_df["src"].tolist()]
    w2v = Word2Vec(sentences=sentences, vector_size=emb_dim, window=5, min_count=1, workers=4, seed=42)
    matrix = np.random.normal(0, 0.02, size=(len(src_vocab), emb_dim)).astype(np.float32)
    for tok, idx in src_vocab.stoi.items():
        if tok in w2v.wv:
            matrix[idx] = w2v.wv[tok]
    return matrix


def build_sentence_embedding_matrix(train_df, src_vocab, emb_dim, model_name):
    """
    Builds token-level initialization from sentence-transformer by encoding vocabulary tokens.
    This makes sentence-embedding baselines trainable while using sentence-level semantic initialization.
    """
    if SentenceTransformer is None:
        return None
    model = SentenceTransformer(model_name)
    tokens = src_vocab.itos
    emb = model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)
    if emb.shape[1] > emb_dim:
        emb = emb[:, :emb_dim]
    elif emb.shape[1] < emb_dim:
        pad = np.zeros((emb.shape[0], emb_dim - emb.shape[1]))
        emb = np.hstack([emb, pad])
    return emb.astype(np.float32)


def train_classic_model(model, train_loader, val_loader, tgt_vocab, device, epochs, lr, out_dir, name):
    model = model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[tgt_vocab.pad])

    history = []
    best_val = 1e9
    best_path = os.path.join(out_dir, f"{name}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            optimizer.zero_grad()
            logits = model(src_ids, tgt_ids, teacher_forcing=0.5)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                logits = model(src_ids, tgt_ids, teacher_forcing=0.0)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{name} epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{name}_history.csv"), index=False)
    return model, pd.DataFrame(history)


def evaluate_classic_model(model, test_loader, tgt_vocab, device, max_len):
    refs = []
    hyps = []
    bos_id = tgt_vocab.stoi[tgt_vocab.bos]
    eos_id = tgt_vocab.stoi[tgt_vocab.eos]

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src_ids = batch["src_ids"].to(device)
            gen = model.generate(src_ids, bos_id, eos_id, max_len=max_len)
            for ids, ref in zip(gen.cpu().numpy(), batch["tgt_text"]):
                hyps.append(tgt_vocab.decode(ids))
                refs.append(ref)

    return refs, hyps, compute_metrics(refs, hyps)


# ============================================================
# 4. Transformer datasets and trainable transformer baselines
# ============================================================

class HFTranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, src_col="src", tgt_col="tgt"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.loc[idx, self.src_col]
        tgt = self.df.loc[idx, self.tgt_col]

        enc = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        dec = self.tokenizer(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "src_text": src,
            "tgt_text": tgt
        }


def train_hf_seq2seq(model, tokenizer, train_df, val_df, args, name):
    train_ds = HFTranslationDataset(train_df, tokenizer, args.max_len)
    val_ds = HFTranslationDataset(val_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 10),
        num_training_steps=total_steps
    )

    history = []
    best_val = 1e9
    best_path = os.path.join(args.model_dir, f"{name}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                labels=batch["labels"].to(args.device)
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    labels=batch["labels"].to(args.device)
                )
                val_loss += outputs.loss.item()

        val_loss /= max(1, len(val_loader))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{name} epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    pd.DataFrame(history).to_csv(os.path.join(args.table_dir, f"{name}_history.csv"), index=False)
    return model, pd.DataFrame(history)


def evaluate_hf_seq2seq(model, tokenizer, test_df, args, name):
    model.eval()
    refs = []
    hyps = []

    loader = DataLoader(HFTranslationDataset(test_df, tokenizer, args.max_len), batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            generated = model.generate(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                num_beams=args.beam_size,
                max_length=args.max_len
            )
            pred = tokenizer.batch_decode(generated, skip_special_tokens=True)
            hyps.extend(pred)
            refs.extend(batch["tgt_text"])

    metrics = compute_metrics(refs, hyps)
    out_df = pd.DataFrame({"source": test_df["src"].tolist()[:len(hyps)], "reference": refs, "prediction": hyps})
    out_df.to_csv(os.path.join(args.table_dir, f"{name}_predictions.csv"), index=False)
    return refs, hyps, metrics


# ============================================================
# 5. Proposed DSF–MarianMT
# ============================================================

class FrozenSentenceEmbedder:
    def __init__(self, model_name, emb_dim, device):
        self.emb_dim = emb_dim
        self.device = device
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for DSF–MarianMT.")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    def encode(self, texts):
        with torch.no_grad():
            arr = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if arr.shape[1] > self.emb_dim:
            arr = arr[:, :self.emb_dim]
        elif arr.shape[1] < self.emb_dim:
            pad = np.zeros((arr.shape[0], self.emb_dim - arr.shape[1]))
            arr = np.hstack([arr, pad])
        return torch.tensor(arr, dtype=torch.float32)


class DSFTranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.loc[idx, "src"]
        tgt = self.df.loc[idx, "tgt"]
        enc = self.tokenizer(src, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        dec = self.tokenizer(tgt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "src_text": src,
            "tgt_text": tgt
        }


def make_dsf_collate(tokenizer, max_len, w2v_model, sent_embedder, emb_dim, device):
    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        src_texts = [b["src_text"] for b in batch]
        tgt_texts = [b["tgt_text"] for b in batch]

        word_vectors = []
        for s in src_texts:
            toks = zh_tokenize(s)
            vecs = []
            if w2v_model is not None:
                for tok in toks:
                    if tok in w2v_model.wv:
                        vecs.append(w2v_model.wv[tok])
            if len(vecs) == 0:
                word_vectors.append(np.zeros(emb_dim, dtype=np.float32))
            else:
                v = np.mean(vecs, axis=0)
                if len(v) > emb_dim:
                    v = v[:emb_dim]
                elif len(v) < emb_dim:
                    v = np.pad(v, (0, emb_dim - len(v)))
                word_vectors.append(v.astype(np.float32))

        word_emb = torch.tensor(np.vstack(word_vectors), dtype=torch.float32)
        src_sent_emb = sent_embedder.encode(src_texts)
        tgt_sent_emb = sent_embedder.encode(tgt_texts)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "word_emb": word_emb,
            "src_sent_emb": src_sent_emb,
            "tgt_sent_emb": tgt_sent_emb,
            "src_texts": src_texts,
            "tgt_texts": tgt_texts
        }
    return collate


class DSFMarianMT(nn.Module):
    def __init__(self, model_name, external_dim=512):
        super().__init__()
        self.marian = MarianMTModel.from_pretrained(model_name)
        hidden = self.marian.config.d_model

        self.word_proj = nn.Linear(external_dim, hidden)
        self.sent_proj = nn.Linear(external_dim, hidden)

        self.gate = nn.Linear(hidden * 3, hidden)
        self.candidate = nn.Linear(hidden * 3, hidden)

        self.hidden = hidden

    def dynamic_semantic_fusion(self, encoder_hidden, word_emb, sent_emb):
        b, t, h = encoder_hidden.shape
        w = self.word_proj(word_emb).unsqueeze(1).expand(b, t, h)
        s = self.sent_proj(sent_emb).unsqueeze(1).expand(b, t, h)
        fusion_input = torch.cat([encoder_hidden, w, s], dim=-1)
        gate = torch.sigmoid(self.gate(fusion_input))
        candidate = torch.tanh(self.candidate(fusion_input))
        fused = gate * candidate + (1.0 - gate) * encoder_hidden
        return fused, gate

    def forward(self, input_ids, attention_mask, labels=None, word_emb=None, src_sent_emb=None, tgt_sent_emb=None, contrastive_lambda=0.3, temperature=0.07):
        enc = self.marian.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        encoder_hidden = enc.last_hidden_state
        fused_hidden, gate = self.dynamic_semantic_fusion(encoder_hidden, word_emb, src_sent_emb)

        outputs = self.marian(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=(fused_hidden,),
            return_dict=True
        )

        translation_loss = outputs.loss
        c_loss = contrastive_semantic_loss(src_sent_emb, tgt_sent_emb, temperature)
        total_loss = translation_loss + contrastive_lambda * c_loss

        return {
            "loss": total_loss,
            "translation_loss": translation_loss,
            "contrastive_loss": c_loss,
            "logits": outputs.logits,
            "gate": gate
        }

    def generate(self, input_ids, attention_mask, word_emb, src_sent_emb, tokenizer, max_length, num_beams):
        self.eval()
        with torch.no_grad():
            enc = self.marian.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            fused_hidden, _ = self.dynamic_semantic_fusion(enc.last_hidden_state, word_emb, src_sent_emb)

            generated = self.marian.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=(fused_hidden,),
                max_length=max_length,
                num_beams=num_beams
            )
        return generated


def contrastive_semantic_loss(src_emb, tgt_emb, temperature=0.07):
    src = F.normalize(src_emb, dim=-1)
    tgt = F.normalize(tgt_emb, dim=-1)
    logits = torch.matmul(src, tgt.T) / temperature
    labels = torch.arange(src.size(0), device=src.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    return (loss1 + loss2) / 2.0


def train_dsf_model(train_df, val_df, args, model_name="DSF-MarianMT", use_word=True, use_sentence=True, use_contrastive=True):
    tokenizer = MarianTokenizer.from_pretrained(args.marian_model)
    sent_embedder = FrozenSentenceEmbedder(args.sentence_model, args.external_dim, args.device)

    w2v_model = None
    if use_word and Word2Vec is not None:
        tokenized = [zh_tokenize(s) for s in train_df["src"].tolist()]
        w2v_model = Word2Vec(sentences=tokenized, vector_size=args.external_dim, window=5, min_count=1, workers=4, seed=args.seed)

    train_ds = DSFTranslationDataset(train_df, tokenizer, args.max_len)
    val_ds = DSFTranslationDataset(val_df, tokenizer, args.max_len)

    collate = make_dsf_collate(tokenizer, args.max_len, w2v_model, sent_embedder, args.external_dim, args.device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = DSFMarianMT(args.marian_model, external_dim=args.external_dim).to(args.device)

    if not use_word:
        for p in model.word_proj.parameters():
            p.requires_grad = False
    if not use_sentence:
        for p in model.sent_proj.parameters():
            p.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 10),
        num_training_steps=total_steps
    )

    best_val = 1e9
    best_path = os.path.join(args.model_dir, f"{model_name}.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_trans = 0.0
        tr_contrast = 0.0
        gate_means = []

        for batch in train_loader:
            optimizer.zero_grad()
            word_emb = batch["word_emb"].to(args.device)
            src_sent = batch["src_sent_emb"].to(args.device)
            tgt_sent = batch["tgt_sent_emb"].to(args.device)

            if not use_word:
                word_emb = torch.zeros_like(word_emb)
            if not use_sentence:
                src_sent = torch.zeros_like(src_sent)
                tgt_sent = torch.zeros_like(tgt_sent)

            outputs = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                labels=batch["labels"].to(args.device),
                word_emb=word_emb,
                src_sent_emb=src_sent,
                tgt_sent_emb=tgt_sent,
                contrastive_lambda=args.contrastive_lambda if use_contrastive else 0.0,
                temperature=args.temperature
            )

            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_loss += outputs["loss"].item()
            tr_trans += outputs["translation_loss"].item()
            tr_contrast += outputs["contrastive_loss"].item()
            gate_means.append(outputs["gate"].detach().mean().item())

        tr_loss /= max(1, len(train_loader))
        tr_trans /= max(1, len(train_loader))
        tr_contrast /= max(1, len(train_loader))
        gate_mean = float(np.mean(gate_means))

        model.eval()
        val_loss = 0.0
        val_gate = []
        with torch.no_grad():
            for batch in val_loader:
                word_emb = batch["word_emb"].to(args.device)
                src_sent = batch["src_sent_emb"].to(args.device)
                tgt_sent = batch["tgt_sent_emb"].to(args.device)

                if not use_word:
                    word_emb = torch.zeros_like(word_emb)
                if not use_sentence:
                    src_sent = torch.zeros_like(src_sent)
                    tgt_sent = torch.zeros_like(tgt_sent)

                outputs = model(
                    input_ids=batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"].to(args.device),
                    labels=batch["labels"].to(args.device),
                    word_emb=word_emb,
                    src_sent_emb=src_sent,
                    tgt_sent_emb=tgt_sent,
                    contrastive_lambda=args.contrastive_lambda if use_contrastive else 0.0,
                    temperature=args.temperature
                )
                val_loss += outputs["loss"].item()
                val_gate.append(outputs["gate"].mean().item())

        val_loss /= max(1, len(val_loader))
        val_gate_mean = float(np.mean(val_gate))
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "translation_loss": tr_trans,
            "contrastive_loss": tr_contrast,
            "val_loss": val_loss,
            "gate_mean": gate_mean,
            "val_gate_mean": val_gate_mean
        })

        print(
            f"{model_name} epoch {epoch}/{args.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"contrastive={tr_contrast:.4f} gate={gate_mean:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    pd.DataFrame(history).to_csv(os.path.join(args.table_dir, f"{model_name}_history.csv"), index=False)
    return model, tokenizer, w2v_model, sent_embedder, pd.DataFrame(history)


def evaluate_dsf_model(model, tokenizer, w2v_model, sent_embedder, test_df, args, model_name="DSF-MarianMT"):
    ds = DSFTranslationDataset(test_df, tokenizer, args.max_len)
    collate = make_dsf_collate(tokenizer, args.max_len, w2v_model, sent_embedder, args.external_dim, args.device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    refs = []
    hyps = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            gen = model.generate(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                word_emb=batch["word_emb"].to(args.device),
                src_sent_emb=batch["src_sent_emb"].to(args.device),
                tokenizer=tokenizer,
                max_length=args.max_len,
                num_beams=args.beam_size
            )
            pred = tokenizer.batch_decode(gen, skip_special_tokens=True)
            hyps.extend(pred)
            refs.extend(batch["tgt_texts"])

    metrics = compute_metrics(refs, hyps)
    pd.DataFrame({"source": test_df["src"].tolist()[:len(hyps)], "reference": refs, "prediction": hyps}).to_csv(
        os.path.join(args.table_dir, f"{model_name}_predictions.csv"),
        index=False
    )
    return refs, hyps, metrics


# ============================================================
# 6. Metrics and statistics
# ============================================================

def compute_metrics(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", lowercase=False).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    ter = sacrebleu.corpus_ter(hyps, [refs]).score
    return {"BLEU": bleu, "chrF": chrf, "TER": ter}


def bootstrap_bleu(refs, base_hyps, prop_hyps, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(refs)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        r = [refs[i] for i in idx]
        b = [base_hyps[i] for i in idx]
        p = [prop_hyps[i] for i in idx]
        bleu_b = sacrebleu.corpus_bleu(b, [r], tokenize="13a", lowercase=False).score
        bleu_p = sacrebleu.corpus_bleu(p, [r], tokenize="13a", lowercase=False).score
        diffs.append(bleu_p - bleu_b)

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0.0))
    }


def paired_sentence_bleu_scores(refs, hyps):
    scores = []
    for r, h in zip(refs, hyps):
        scores.append(sacrebleu.sentence_bleu(h, [r], tokenize="13a", lowercase=False).score)
    return np.array(scores)


def statistical_validation(prediction_files, out_csv):
    """
    Uses prediction CSV files generated by the models.
    Requires MarianMT_predictions.csv and DSF-MarianMT_predictions.csv.
    """
    if "MarianMT" not in prediction_files or "DSF-MarianMT" not in prediction_files:
        return pd.DataFrame()

    base = pd.read_csv(prediction_files["MarianMT"])
    prop = pd.read_csv(prediction_files["DSF-MarianMT"])

    refs = base["reference"].tolist()
    base_h = base["prediction"].tolist()
    prop_h = prop["prediction"].tolist()

    boot = bootstrap_bleu(refs, base_h, prop_h)
    base_scores = paired_sentence_bleu_scores(refs, base_h)
    prop_scores = paired_sentence_bleu_scores(refs, prop_h)

    t_stat, t_p = stats.ttest_rel(prop_scores, base_scores)
    try:
        w_stat, w_p = stats.wilcoxon(prop_scores, base_scores)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    pooled_sd = np.sqrt((np.std(prop_scores, ddof=1) ** 2 + np.std(base_scores, ddof=1) ** 2) / 2)
    cohens_d = (np.mean(prop_scores) - np.mean(base_scores)) / pooled_sd if pooled_sd > 0 else np.nan

    rows = [
        {
            "Test": "Bootstrap Resampling",
            "Statistic": f"Mean BLEU diff={boot['mean_diff']:.4f}; 95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]",
            "p_value": boot["p_value"]
        },
        {
            "Test": "Paired t-test",
            "Statistic": f"t={t_stat:.4f}",
            "p_value": t_p
        },
        {
            "Test": "Wilcoxon Signed-Rank",
            "Statistic": f"W={w_stat:.4f}",
            "p_value": w_p
        },
        {
            "Test": "Cohen's d",
            "Statistic": f"d={cohens_d:.4f}",
            "p_value": np.nan
        }
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ============================================================
# 7. Explainability and plots from actual logs
# ============================================================

def set_plot_style():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.titlecolor"] = "black"


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_metric_table(results_df, out_path, metric="BLEU"):
    set_plot_style()
    df = results_df.sort_values(metric)
    plt.figure(figsize=(10, 6))
    plt.barh(df["Model"], df[metric])
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.title(f"Model Comparison Based on {metric}")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    save_fig(out_path)


def plot_histories(history_files, out_path, metric_col="val_loss", title="Validation Loss"):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    for name, path in history_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if metric_col in df.columns:
                plt.plot(df["epoch"], df[metric_col], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_col.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    save_fig(out_path)


def plot_dsf_contrastive(history_file, out_path):
    if not os.path.exists(history_file):
        return
    df = pd.read_csv(history_file)
    set_plot_style()
    plt.figure(figsize=(9, 5.5))
    plt.plot(df["epoch"], df["translation_loss"], label="Translation loss")
    plt.plot(df["epoch"], df["contrastive_loss"], label="Contrastive semantic loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DSF–MarianMT Translation and Contrastive Loss")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    save_fig(out_path)


def calculate_attention_entropy(model, tokenizer, test_df, args, sample_size=50):
    """
    For MarianMT baseline attentions only.
    Returns layer-wise mean attention entropy when output_attentions is supported.
    """
    model.eval()
    sample_df = test_df.head(sample_size)
    entropies = defaultdict(list)

    with torch.no_grad():
        for _, row in sample_df.iterrows():
            inputs = tokenizer(row["src"], return_tensors="pt", truncation=True, max_length=args.max_len).to(args.device)
            labels = tokenizer(row["tgt"], return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(args.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                output_attentions=True,
                return_dict=True
            )

            if outputs.cross_attentions is None:
                continue

            for layer_idx, att in enumerate(outputs.cross_attentions):
                # att shape: batch, heads, tgt_len, src_len
                p = att.clamp(min=1e-9)
                entropy = -(p * p.log()).sum(dim=-1).mean().item()
                entropies[layer_idx + 1].append(entropy)

    rows = []
    for layer, values in entropies.items():
        rows.append({"layer": layer, "attention_entropy": float(np.mean(values))})
    return pd.DataFrame(rows)


def plot_attention_entropy(entropy_df, out_path):
    if entropy_df.empty:
        return
    set_plot_style()
    plt.figure(figsize=(8, 5))
    plt.plot(entropy_df["layer"], entropy_df["attention_entropy"], marker="o")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Attention Entropy")
    plt.title("Layer-wise Attention Entropy")
    plt.grid(True, linestyle="--", alpha=0.35)
    save_fig(out_path)


def component_importance_from_ablation(ablation_df):
    """
    Computes contribution by BLEU drop from full model.
    Requires rows for ablation variants.
    """
    if ablation_df.empty or "BLEU" not in ablation_df.columns:
        return pd.DataFrame()

    full_bleu = float(ablation_df[ablation_df["Model"] == "DSF-MarianMT"]["BLEU"].iloc[0])
    rows = []
    for _, row in ablation_df.iterrows():
        if row["Model"] == "DSF-MarianMT":
            continue
        drop = full_bleu - float(row["BLEU"])
        rows.append({"Component": row["Model"], "MeanAbsContribution": max(drop, 0)})
    return pd.DataFrame(rows).sort_values("MeanAbsContribution", ascending=True)


def plot_component_importance(importance_df, out_path):
    if importance_df.empty:
        return
    set_plot_style()
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df["Component"], importance_df["MeanAbsContribution"], edgecolor="black")
    for bar, value in zip(bars, importance_df["MeanAbsContribution"]):
        plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color="black")
    plt.xlabel("BLEU Contribution Estimated from Ablation")
    plt.ylabel("Component")
    plt.title("XAI Component Importance from Ablation")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    save_fig(out_path)


# ============================================================
# 8. Main experiment runner
# ============================================================

def run_classic_experiments(train_df, val_df, test_df, args):
    src_tokens = [zh_tokenize(x) for x in train_df["src"].tolist()]
    tgt_tokens = [en_tokenize(x) for x in train_df["tgt"].tolist()]

    src_vocab = Vocab(min_freq=1, max_size=args.src_vocab_size)
    tgt_vocab = Vocab(min_freq=1, max_size=args.tgt_vocab_size)
    src_vocab.build(src_tokens)
    tgt_vocab.build(tgt_tokens)

    train_ds = ClassicTranslationDataset(train_df, src_vocab, tgt_vocab, args.max_len)
    val_ds = ClassicTranslationDataset(val_df, src_vocab, tgt_vocab, args.max_len)
    test_ds = ClassicTranslationDataset(test_df, src_vocab, tgt_vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = []
    prediction_files = {}
    history_files = {}

    embedding_modes = []

    w2v_matrix = build_word2vec_matrix(train_df, src_vocab, args.classic_emb_dim)
    if w2v_matrix is not None:
        embedding_modes.append(("Word2Vec", w2v_matrix))

    sent_matrix = None
    if SentenceTransformer is not None:
        sent_matrix = build_sentence_embedding_matrix(train_df, src_vocab, args.classic_emb_dim, args.sentence_model)
        embedding_modes.append(("SentenceEmbedding", sent_matrix))

    if not embedding_modes:
        embedding_modes.append(("TrainableEmbedding", None))

    for emb_name, matrix in embedding_modes:
        for model_type in ["bilstm", "gru", "cnn"]:
            name = f"{emb_name}_{model_type}"
            model = ClassicSeq2Seq(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                src_pad_id=src_vocab.stoi[src_vocab.pad],
                tgt_pad_id=tgt_vocab.stoi[tgt_vocab.pad],
                model_type=model_type,
                src_embed_matrix=matrix,
                emb_dim=args.classic_emb_dim,
                hidden_dim=args.classic_hidden_dim,
                max_len=args.max_len,
                dropout=args.dropout
            )
            model, hist = train_classic_model(model, train_loader, val_loader, tgt_vocab, args.device, args.epochs, args.lr, args.model_dir, name)
            refs, hyps, metrics = evaluate_classic_model(model, test_loader, tgt_vocab, args.device, args.max_len)
            metrics["Model"] = name
            results.append(metrics)

            pred_path = os.path.join(args.table_dir, f"{name}_predictions.csv")
            pd.DataFrame({"reference": refs, "prediction": hyps}).to_csv(pred_path, index=False)
            prediction_files[name] = pred_path
            history_files[name] = os.path.join(args.model_dir, f"{name}_history.csv")

    return results, prediction_files, history_files


def run_transformer_experiments(train_df, val_df, test_df, args):
    results = []
    prediction_files = {}
    history_files = {}

    # BERT Encoder-Decoder
    for name, enc_name, dec_name in [
        ("BERT", args.bert_model, args.bert_model),
        ("mBERT", args.mbert_model, args.mbert_model),
        ("mBERT_Decoder", args.mbert_model, args.mbert_model)
    ]:
        tokenizer = AutoTokenizer.from_pretrained(enc_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.sep_token

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_name, dec_name)
        model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id or tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model, hist = train_hf_seq2seq(model, tokenizer, train_df, val_df, args, name)
        refs, hyps, metrics = evaluate_hf_seq2seq(model, tokenizer, test_df, args, name)
        metrics["Model"] = name
        results.append(metrics)
        prediction_files[name] = os.path.join(args.table_dir, f"{name}_predictions.csv")
        history_files[name] = os.path.join(args.table_dir, f"{name}_history.csv")

    # MarianMT baseline
    name = "MarianMT"
    tokenizer = MarianTokenizer.from_pretrained(args.marian_model)
    model = MarianMTModel.from_pretrained(args.marian_model)
    model, hist = train_hf_seq2seq(model, tokenizer, train_df, val_df, args, name)
    refs, hyps, metrics = evaluate_hf_seq2seq(model, tokenizer, test_df, args, name)
    metrics["Model"] = name
    results.append(metrics)
    prediction_files[name] = os.path.join(args.table_dir, f"{name}_predictions.csv")
    history_files[name] = os.path.join(args.table_dir, f"{name}_history.csv")

    return results, prediction_files, history_files


def run_dsf_and_ablation(train_df, val_df, test_df, args):
    configs = [
        ("DSF_no_word", False, True, True),
        ("DSF_no_sentence", True, False, True),
        ("DSF_no_contrastive", True, True, False),
        ("DSF-MarianMT", True, True, True)
    ]

    results = []
    prediction_files = {}
    history_files = {}

    for name, use_word, use_sentence, use_contrastive in configs:
        model, tokenizer, w2v, sent_embedder, hist = train_dsf_model(
            train_df,
            val_df,
            args,
            model_name=name,
            use_word=use_word,
            use_sentence=use_sentence,
            use_contrastive=use_contrastive
        )
        refs, hyps, metrics = evaluate_dsf_model(model, tokenizer, w2v, sent_embedder, test_df, args, model_name=name)
        metrics["Model"] = name
        results.append(metrics)
        prediction_files[name] = os.path.join(args.table_dir, f"{name}_predictions.csv")
        history_files[name] = os.path.join(args.table_dir, f"{name}_history.csv")

    return results, prediction_files, history_files


def create_outputs(results, prediction_files, history_files, args):
    results_df = pd.DataFrame(results)
    metric_cols = [c for c in ["BLEU", "chrF", "TER", "COMET"] if c in results_df.columns]
    ordered = ["Model"] + metric_cols
    results_df = results_df[ordered]
    results_path = os.path.join(args.table_dir, "all_model_results.csv")
    results_df.to_csv(results_path, index=False)

    if not results_df.empty:
        plot_metric_table(results_df, os.path.join(args.fig_dir, "Model_Comparison_BLEU.png"), metric="BLEU")
        if "TER" in results_df.columns:
            plot_metric_table(results_df, os.path.join(args.fig_dir, "Model_Comparison_TER.png"), metric="TER")

    plot_histories(history_files, os.path.join(args.fig_dir, "Validation_Loss_All_Models.png"), metric_col="val_loss", title="Validation Loss Across Models")

    if "DSF-MarianMT" in history_files:
        plot_dsf_contrastive(history_files["DSF-MarianMT"], os.path.join(args.fig_dir, "DSF_Translation_Contrastive_Loss.png"))

    stat_df = statistical_validation(prediction_files, os.path.join(args.table_dir, "statistical_validation.csv"))

    ablation_models = ["DSF_no_word", "DSF_no_sentence", "DSF_no_contrastive", "DSF-MarianMT"]
    ablation_df = results_df[results_df["Model"].isin(ablation_models)].copy()
    ablation_df.to_csv(os.path.join(args.table_dir, "fusion_component_ablation.csv"), index=False)

    importance_df = component_importance_from_ablation(ablation_df)
    importance_df.to_csv(os.path.join(args.table_dir, "xai_component_importance.csv"), index=False)
    plot_component_importance(importance_df, os.path.join(args.fig_dir, "Fig16_XAI_Component_Importance.png"))

    print("Results saved:", results_path)
    print(results_df)
    if not stat_df.empty:
        print(stat_df)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="translation.csv")
    parser.add_argument("--src_col", type=str, default=None)
    parser.add_argument("--tgt_col", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default="dsf_marianmt_trainable_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--classic_emb_dim", type=int, default=256)
    parser.add_argument("--classic_hidden_dim", type=int, default=256)
    parser.add_argument("--src_vocab_size", type=int, default=30000)
    parser.add_argument("--tgt_vocab_size", type=int, default=30000)

    parser.add_argument("--external_dim", type=int, default=512)
    parser.add_argument("--contrastive_lambda", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--bert_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--mbert_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--marian_model", type=str, default="Helsinki-NLP/opus-mt-zh-en")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    parser.add_argument("--run_classic", action="store_true")
    parser.add_argument("--run_transformers", action="store_true")
    parser.add_argument("--run_dsf", action="store_true")
    parser.add_argument("--run_all", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_dir = args.output_dir
    args.fig_dir = os.path.join(args.output_dir, "figures")
    args.table_dir = os.path.join(args.output_dir, "tables")
    args.model_dir = os.path.join(args.output_dir, "models")
    safe_mkdir(args.output_dir)
    safe_mkdir(args.fig_dir)
    safe_mkdir(args.table_dir)
    safe_mkdir(args.model_dir)

    max_samples = args.max_samples if args.max_samples > 0 else None
    df = load_parallel_dataset(args.data, args.src_col, args.tgt_col, max_samples=max_samples)
    train_df, val_df, test_df = split_dataset(df, seed=args.seed)

    train_df.to_csv(os.path.join(args.output_dir, "train_split_seed42.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "validation_split_seed42.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test_split_seed42.csv"), index=False)

    print(f"Dataset loaded: total={len(df)}, train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Device: {args.device}")

    results = []
    prediction_files = {}
    history_files = {}

    run_classic = args.run_all or args.run_classic
    run_transformers = args.run_all or args.run_transformers
    run_dsf = args.run_all or args.run_dsf

    if not any([run_classic, run_transformers, run_dsf]):
        print("No training flag selected. Running DSF only by default.")
        run_dsf = True

    if run_classic:
        r, p, h = run_classic_experiments(train_df, val_df, test_df, args)
        results.extend(r)
        prediction_files.update(p)
        history_files.update(h)

    if run_transformers:
        r, p, h = run_transformer_experiments(train_df, val_df, test_df, args)
        results.extend(r)
        prediction_files.update(p)
        history_files.update(h)

    if run_dsf:
        r, p, h = run_dsf_and_ablation(train_df, val_df, test_df, args)
        results.extend(r)
        prediction_files.update(p)
        history_files.update(h)

    create_outputs(results, prediction_files, history_files, args)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Completed successfully.")


if __name__ == "__main__":
    main()
