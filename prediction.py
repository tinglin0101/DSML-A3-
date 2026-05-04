# stage2 only
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import (
    SentenceTransformer, InputExample, losses,
    SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from datasets import Dataset
from transformers.trainer_callback import PrinterCallback
import logging as _pylog
_pylog.getLogger("transformers.trainer").setLevel(_pylog.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# from huggingface_hub import login
# login(huggingfacetoken="your token")

SBERT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
SBERT_DIM   = 1024
FEAT_COLS   = [f"emb_{i}" for i in range(SBERT_DIM)]


def build_sbert() -> SentenceTransformer:
    print(f"Loading '{SBERT_MODEL}' …")
    model = SentenceTransformer(SBERT_MODEL)
    model = model.float()
    return model


def _make_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 42) -> list:
    rng    = random.Random(seed)
    texts  = df["TEXT"].astype(str).tolist()
    labels = df["LABEL"].astype(int).tolist()

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]

    pairs = []
    half  = n_pairs // 2

    for _ in range(half):
        pool = pos if rng.random() < 0.5 else neg
        a, b = rng.sample(pool, 2) if len(pool) >= 2 else (pool[0], pool[0])
        pairs.append(InputExample(texts=[texts[a], texts[b]], label=1.0))

    for _ in range(n_pairs - half):
        pairs.append(InputExample(
            texts=[texts[rng.choice(pos)], texts[rng.choice(neg)]], label=0.0
        ))

    rng.shuffle(pairs)
    return pairs


def finetune(sbert: SentenceTransformer, train_df: pd.DataFrame,
             output_path=None) -> None:

    pairs = _make_pairs(train_df, 3000)
    ds = Dataset.from_dict({
        "sentence1": [p.texts[0] for p in pairs],
        "sentence2": [p.texts[1] for p in pairs],
        "label":     [float(p.label) for p in pairs],
    })

    train_args = SentenceTransformerTrainingArguments(
        output_dir=output_path or "tmp_ft",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=sbert,
        args=train_args,
        train_dataset=ds,
        loss=losses.CosineSimilarityLoss(sbert),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    if output_path:
        sbert.save(output_path)

    print(" Fine-tuning done.")


def embed(df: pd.DataFrame, sbert: SentenceTransformer,
          has_label: bool = True) -> pd.DataFrame:
    print(f"  Embedding {len(df)} rows …")
    texts = df["TEXT"].astype(str).tolist()
    embs  = sbert.encode(texts, normalize_embeddings=True,
                         batch_size=32, show_progress_bar=False)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        rec = {"row_id": int(row["row_id"]),
               "LABEL":  int(row["LABEL"]) if has_label else -1}
        rec.update({f"emb_{j}": float(embs[i, j]) for j in range(SBERT_DIM)})
        records.append(rec)

    return pd.DataFrame(records)[["row_id", "LABEL"] + FEAT_COLS]


def self_training(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  sbert_base: SentenceTransformer):
    import copy

    augmented_df = train_df.copy()
    pipe = None
    sbert_ft = None
    total_rounds = 3  # pseudo_rounds=2 + round 0

    for rnd in range(total_rounds):
        is_last = (rnd == total_rounds - 1)
        n_pseudo = len(augmented_df) - len(train_df)
        print(f"\n══ Self-training Round {rnd}/{total_rounds - 1}"
              f"  total_train={len(augmented_df)}"
              f"  (labeled={len(train_df)}, pseudo={n_pseudo}) ══")

        sbert_ft = copy.deepcopy(sbert_base)
        finetune(sbert_ft, augmented_df, output_path=None)

        X_emb = embed(augmented_df, sbert_ft, has_label=True)
        y = augmented_df["LABEL"].astype(int).values
        pipe = Pipeline([("sc", StandardScaler()), ("clf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
        pipe.fit(X_emb[FEAT_COLS].values, y)

        if not is_last:
            test_emb = embed(test_df, sbert_ft, has_label=False)
            scores = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)

            thr = 0.85
            pseudo_rows = []
            for idx, (_, row) in enumerate(test_df.iterrows()):
                if scores[idx] >= thr:
                    pseudo_rows.append({"row_id": int(row["row_id"]),
                                        "TEXT": str(row["TEXT"]), "LABEL": 1})
                elif scores[idx] <= 1.0 - thr:
                    pseudo_rows.append({"row_id": int(row["row_id"]),
                                        "TEXT": str(row["TEXT"]), "LABEL": 0})

            n_pos = sum(r["LABEL"] == 1 for r in pseudo_rows)
            n_neg = sum(r["LABEL"] == 0 for r in pseudo_rows)
            print(f"  → Pseudo-labels: {len(pseudo_rows)} selected"
                  f"  (pos={n_pos}, neg={n_neg})  threshold={thr}")

            if pseudo_rows:
                pseudo_df = pd.DataFrame(pseudo_rows)
                augmented_df = pd.concat([train_df, pseudo_df], ignore_index=True)
            else:
                print("  → No pseudo-labels selected (threshold too strict); keeping current set")

    return pipe, sbert_ft


def main():

    train_df = pd.read_csv("train_2022.csv")
    train_df["row_id"] = train_df["row_id"].astype(int)
    train_df["LABEL"]  = train_df["LABEL"].astype(int)
    print(f"\nTrain: {len(train_df)} rows  |  pos={train_df['LABEL'].sum()}  neg={(train_df['LABEL']==0).sum()}")

    test_df = pd.read_csv("test_no_answer_2022.csv")
    test_df["row_id"] = test_df["row_id"].astype(int)
    if "LABEL" not in test_df.columns:
        test_df["LABEL"] = -1
    print(f"Test : {len(test_df)} rows\n")

    sbert = build_sbert()

    pipe, sbert_final = self_training(train_df, test_df, sbert)

    print("\n── Test set prediction ──")
    test_emb   = embed(test_df, sbert_final, has_label=False)
    scores     = np.clip(pipe.predict(test_emb[FEAT_COLS].values), 0, 1)
    test_out   = test_emb[["row_id"]].copy()
    test_out["LABEL"] = (scores >= 0.5).astype(int)
    test_out.to_csv("result.csv", index=False)
    print(f"Saved result.csv")


if __name__ == "__main__":
    main()
