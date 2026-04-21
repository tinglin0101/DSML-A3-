"""
End-to-End Sentiment Scoring Pipeline  —  v6d
=============================================
v6d change: fine-tunes cardiffnlp/twitter-roberta-base-sentiment on the
  training labels BEFORE extracting embeddings.

Pipeline per run:
  原始文字
    │
    ▼
  文字切割（Stanza）
    │
    ▼
  微調 RoBERTa（2000 筆 train + LABEL, binary classification, N epochs）
    │
    ▼
  用微調後 RoBERTa 產生 CLS 嵌入向量（768-dim）— weighted-sum aggregation
    │
    ▼
  5-fold CV（RF / GBM / ElasticNet）+ 回歸評分

Usage:
    python v6d.py --input train_2022.csv --output result_v6d.csv
    python v6d.py --model gbm --ft-epochs 5 --ft-lr 2e-5
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import stanza
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, mean_squared_error,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Constants ─────────────────────────────────────────────────────────────────

ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
ROBERTA_DIM        = 768

VADER_FEATURES = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
AGG_FEATURES   = [f"emb_{i}" for i in range(ROBERTA_DIM)]
ALL_FEATURES   = AGG_FEATURES + VADER_FEATURES

WEIGHT_SCHEMES = ("sqrt", "uniform", "linear", "log", "decay", "last", "contrast")

TRANSITION_WORDS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "nonetheless", "except", "while", "whereas",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Text splitting  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stanza_pipeline() -> stanza.Pipeline:
    stanza.download("en", verbose=False)
    return stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", verbose=False)


def split_text(text: str, nlp: stanza.Pipeline) -> list[str]:
    if not isinstance(text, str):
        return [text]

    doc = nlp(text)
    split_indices = []

    for sentence in doc.sentences:
        words = sentence.words
        for word in words:
            word_lower  = word.text.lower()
            lemma_lower = word.lemma.lower() if word.lemma else ""
            start_char  = getattr(word, "start_char", 0)

            if word_lower in TRANSITION_WORDS or lemma_lower in TRANSITION_WORDS:
                if getattr(word, "upos", "") in ("CCONJ", "SCONJ", "ADV"):
                    if start_char > 0:
                        split_indices.append(start_char)
                continue

            if word_lower == "and":
                head_id = word.head if word.head is not None else 0
                if head_id > 0:
                    head_word = words[head_id - 1]
                    if getattr(head_word, "upos", "") in ("VERB", "AUX"):
                        if start_char > 0:
                            split_indices.append(start_char)

    split_indices = sorted(set(split_indices))

    parts, last_idx = [], 0
    for idx in split_indices:
        seg = text[last_idx:idx].strip()
        if seg:
            parts.append(seg)
        last_idx = idx

    final = text[last_idx:].strip()
    if final:
        parts.append(final)

    return parts or [text]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load base RoBERTa
# ══════════════════════════════════════════════════════════════════════════════

def _load_roberta(num_labels: int = 3) -> tuple:
    """Load tokenizer + model from HuggingFace. num_labels=3 for the pretrained head."""
    print(f"  Loading RoBERTa base '{ROBERTA_MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=num_labels,
        output_hidden_states=True,
        ignore_mismatched_sizes=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2a — Fine-tune RoBERTa on training labels
# ══════════════════════════════════════════════════════════════════════════════

class _TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def finetune_roberta(
    df: pd.DataFrame,
    tokenizer,
    model,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
) -> None:
    """
    Fine-tune *model* in-place on (TEXT, LABEL) rows from df.
    LABEL is remapped to {0,1} → classifier head has 2 outputs.

    The original cardiffnlp model has a 3-class head; we replace it with a
    2-class head here so the fine-tuning objective matches binary labels.
    """
    device = next(model.parameters()).device

    # Replace classification head for binary task.
    # RobertaClassificationHead receives full sequence_output [B, T, H] and
    # extracts CLS internally — our replacement must do the same.
    hidden_size = model.config.hidden_size

    class _BinaryHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(hidden_size, 2)

        def forward(self, x, **_):
            x = x[:, 0, :]          # CLS token
            x = self.dropout(x)
            return self.out_proj(x)

    model.classifier = _BinaryHead().to(device)

    texts  = df["TEXT"].astype(str).tolist()
    labels = df["LABEL"].astype(int).tolist()

    dataset    = _TextLabelDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"  Fine-tuning RoBERTa: {len(texts)} samples, "
          f"{epochs} epochs, lr={lr}, batch={batch_size}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_batch    = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = loss_fn(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(label_batch)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == label_batch).sum().item()
            total      += len(label_batch)

        avg_loss = total_loss / total
        acc      = correct / total
        print(f"    Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    model.eval()
    print("  Fine-tuning complete.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — CLS embedding from fine-tuned model
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding(tokenizer, model, text: str) -> np.ndarray:
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy()
    norm = np.linalg.norm(cls_emb)
    if norm > 0:
        cls_emb = cls_emb / norm
    return cls_emb.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# VADER
# ══════════════════════════════════════════════════════════════════════════════

def _build_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _get_vader_scores(vader: SentimentIntensityAnalyzer, text: str) -> dict:
    scores = vader.polarity_scores(text)
    return {
        "vader_neg":      scores["neg"],
        "vader_neu":      scores["neu"],
        "vader_pos":      scores["pos"],
        "vader_compound": scores["compound"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Weight schemes
# ══════════════════════════════════════════════════════════════════════════════

def _compute_weights(n: int, scheme: str) -> list:
    if scheme == "uniform":  return [1.0] * n
    if scheme == "linear":   return [float(i + 1) for i in range(n)]
    if scheme == "sqrt":     return [np.sqrt(i + 1) for i in range(n)]
    if scheme == "log":      return [np.log(i + 2) for i in range(n)]
    if scheme == "decay":    return [np.exp(-0.5 * i) for i in range(n)]
    if scheme == "last":
        w = [1.0] * n
        if n > 0: w[-1] = 3.0
        return w
    if scheme == "contrast":
        if n == 1: return [2.0]
        w = [0.5] * n; w[0] = 2.0; w[-1] = 2.0
        return w
    raise ValueError(f"Unknown weight scheme '{scheme}'.")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE — Steps 1+2b per row, then weighted sum
# ══════════════════════════════════════════════════════════════════════════════

def process_row(text: str, label: int, row_id: int,
                nlp: stanza.Pipeline, tokenizer, model,
                vader: SentimentIntensityAnalyzer,
                weight_scheme: str = "sqrt") -> dict:
    segments = split_text(text, nlp)
    n        = len(segments)
    weights  = _compute_weights(n, weight_scheme)

    agg_emb = np.zeros(ROBERTA_DIM, dtype=np.float32)
    for i, seg in enumerate(segments):
        agg_emb += _get_embedding(tokenizer, model, seg) * weights[i]

    result = {"row_id": row_id, "LABEL": label, "NUM_SPLITS": n}
    for j, val in enumerate(agg_emb):
        result[f"emb_{j}"] = round(float(val), 6)
    result.update(_get_vader_scores(vader, text))
    return result


def run_row_by_row(df: pd.DataFrame,
                   nlp: stanza.Pipeline,
                   tokenizer, model,
                   vader: SentimentIntensityAnalyzer,
                   has_label: bool = True,
                   weight_scheme: str = "sqrt") -> pd.DataFrame:
    total = len(df)
    print(f"  Extracting embeddings for {total} rows …")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"    progress: {i}/{total}")
        label = int(row["LABEL"]) if has_label else -1
        records.append(
            process_row(str(row["TEXT"]), label, int(row["row_id"]),
                        nlp, tokenizer, model, vader,
                        weight_scheme=weight_scheme)
        )

    cols = ["row_id", "LABEL", "NUM_SPLITS"] + AGG_FEATURES + VADER_FEATURES
    result_df = pd.DataFrame(records)[cols]
    print(f"  Done — {len(result_df)} rows embedded")
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 5-Fold CV + Regression Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _make_model(model_name: str):
    model_map = {
        "rf":      RandomForestRegressor(n_estimators=100, random_state=42),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "gbm":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'.")
    return model_map[model_name]


def _print_fold_metrics(fold: int, y_true, y_pred_label, y_score) -> dict:
    cm  = confusion_matrix(y_true, y_pred_label)
    acc = accuracy_score(y_true, y_pred_label)
    pre = precision_score(y_true, y_pred_label, zero_division=0)
    rec = recall_score(y_true, y_pred_label, zero_division=0)
    f1  = f1_score(y_true, y_pred_label, zero_division=0)
    mse = mean_squared_error(y_true, y_score)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    print(f"\n  ── Fold {fold} ──────────────────────────────")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  AUC      : {auc:.4f}  MSE: {mse:.4f}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc, mse=mse, cm=cm)


def run_holdout_validation(
    train_agg_df: pd.DataFrame,
    val_agg_df: pd.DataFrame,
    model_name: str,
):
    """Train on 80% embeddings, evaluate on 20% holdout, retrain on full set."""
    print(f"  Model: {model_name}  |  Features: {ROBERTA_DIM}-dim RoBERTa + 4 VADER")
    print(f"  Train: {len(train_agg_df)} rows  |  Val: {len(val_agg_df)} rows")

    X_train = train_agg_df[ALL_FEATURES].values
    y_train = train_agg_df["LABEL"].values
    X_val   = val_agg_df[ALL_FEATURES].values
    y_val   = val_agg_df["LABEL"].values

    pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
    pipe.fit(X_train, y_train)

    val_scores = np.clip(pipe.predict(X_val), 0, 1)
    val_labels = (val_scores >= 0.5).astype(int)
    print("\n  ── Holdout Validation Results ───────────────────────")
    _print_fold_metrics(0, y_val, val_labels, val_scores)

    print("\n  Retraining on full dataset (train + val) …")
    all_agg_df = pd.concat([train_agg_df, val_agg_df], ignore_index=True)
    X_all = all_agg_df[ALL_FEATURES].values
    y_all = all_agg_df["LABEL"].values
    final_pipe = Pipeline([("scaler", StandardScaler()), ("model", _make_model(model_name))])
    final_pipe.fit(X_all, y_all)

    return final_pipe, ALL_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    path = "D:\\[課程]DSML\\DSML-A3-\\"
    parser = argparse.ArgumentParser(description="End-to-end sentiment scoring pipeline v6d")
    parser.add_argument("--input",       default=path + "train_2022_product_reviews.csv")
    parser.add_argument("--output",      default=path + "row_scores_v6b-finetune_product_reviews.csv")
    parser.add_argument("--model",       default="rf", choices=["rf", "elastic", "gbm"])
    parser.add_argument("--test",        default=path + "test_no_answer_2022.csv")
    parser.add_argument("--test-output", default=path + "result_v6b-finetune_product_reviews.csv")
    parser.add_argument("--weight",      default="sqrt", choices=list(WEIGHT_SCHEMES))
    parser.add_argument("--ft-epochs",   default=3, type=int,
                        help="Fine-tuning epochs (default: 3)")
    parser.add_argument("--ft-lr",       default=2e-5, type=float,
                        help="Fine-tuning learning rate (default: 2e-5)")
    parser.add_argument("--ft-batch",    default=16, type=int,
                        help="Fine-tuning batch size (default: 16)")
    parser.add_argument("--ft-maxlen",   default=128, type=int,
                        help="Max token length for fine-tuning (default: 128)")
    parser.add_argument("--ft-split",    default=0.8, type=float,
                        help="Fraction of train data used for fine-tuning (default: 0.8)")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  End-to-End Sentiment Scoring Pipeline  [v6d]")
    print(sep)
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.test_output}")
    print(f"  Model    : {args.model}")
    print(f"  Encoder  : {ROBERTA_MODEL_NAME}  ({ROBERTA_DIM}-dim CLS, fine-tuned)")
    print(f"  FT epochs: {args.ft_epochs}  lr={args.ft_lr}  batch={args.ft_batch}")
    print(f"  FT split : {args.ft_split:.0%} train / {1-args.ft_split:.0%} held-out")
    print(f"  Weighting: {args.weight}\n")

    # ── Load training data ────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["row_id"] = df["row_id"].astype(int)
    df["LABEL"]  = df["LABEL"].astype(int)
    print(f"  Loaded {len(df)} training rows\n")

    # ── Split 80/20: ft_df for fine-tuning, val_df as holdout ───────────────
    ft_df  = df.sample(frac=args.ft_split, random_state=42)
    val_df = df.drop(ft_df.index).reset_index(drop=True)
    ft_df  = ft_df.reset_index(drop=True)
    print(f"  Fine-tune split: {len(ft_df)} rows ({args.ft_split:.0%}) / "
          f"{len(val_df)} held-out ({1-args.ft_split:.0%})\n")

    # ── Initialise models ─────────────────────────────────────────────────────
    print("── Initialising models ─────────────────────────────────────")
    nlp   = _build_stanza_pipeline()
    vader = _build_vader()
    tokenizer, roberta_model = _load_roberta(num_labels=3)

    # ── Fine-tune RoBERTa on 80% of training data ─────────────────────────────
    print("\n── Fine-tuning RoBERTa (80% split) ─────────────────────────")
    finetune_roberta(
        ft_df, tokenizer, roberta_model,
        epochs=args.ft_epochs,
        batch_size=args.ft_batch,
        lr=args.ft_lr,
        max_length=args.ft_maxlen,
    )

    # ── Steps 1+2: embed ft (80%) and val (20%) separately ───────────────────
    print("\n── Steps 1+2: Embedding ft split (80%) ─────────────────────")
    ft_agg_df = run_row_by_row(ft_df, nlp, tokenizer, roberta_model, vader,
                               has_label=True, weight_scheme=args.weight)
    print("\n── Steps 1+2: Embedding holdout split (20%) ─────────────────")
    val_agg_df = run_row_by_row(val_df, nlp, tokenizer, roberta_model, vader,
                                has_label=True, weight_scheme=args.weight)

    # ── Step 3: Holdout Validation ────────────────────────────────────────────
    print("\n── Step 3: Holdout Validation + Regression Scoring ─────────")
    trained_pipe, feat_cols = run_holdout_validation(ft_agg_df, val_agg_df, args.model)

    # ── Process test set ──────────────────────────────────────────────────────
    # Re-fine-tune RoBERTa on full training data before predicting test
    print("\n── Re-fine-tuning RoBERTa on full training data (for test inference) ─")
    tokenizer, roberta_model = _load_roberta(num_labels=3)
    finetune_roberta(
        df, tokenizer, roberta_model,
        epochs=args.ft_epochs,
        batch_size=args.ft_batch,
        lr=args.ft_lr,
        max_length=args.ft_maxlen,
    )

    print(f"\n── Processing test set: {args.test} ────────────────────────")
    test_df = pd.read_csv(args.test)
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    print(f"  Loaded {len(test_df)} test rows\n")

    test_agg_df = run_row_by_row(test_df, nlp, tokenizer, roberta_model, vader,
                                 has_label=False, weight_scheme=args.weight)

    X_test      = test_agg_df[feat_cols].values
    test_scores = np.clip(trained_pipe.predict(X_test), 0, 1)
    test_labels = (test_scores >= 0.5).astype(int)


    test_output = test_agg_df[["row_id"]].copy()
    test_output["LABEL"] = test_labels
    test_output.to_csv(args.test_output, index=False)
    print(f"  Saved test predictions to: {args.test_output}")


if __name__ == "__main__":
    main()
